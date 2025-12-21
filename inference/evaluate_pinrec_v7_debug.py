import sys
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

# --- 导入模型配置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# 配置区域
# =========================================================
# [自动锁定 V7 目录]
BASE_DIR = "/workspace/data/pinrec_ckpt_v7_final" 

# [关键] 验证集
TEST_DATA = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl" 

BATCH_SIZE = 128
# [升级] 扩大检索范围，看看到底排在哪里
TOP_K_EVAL = [1, 5, 10, 50, 100] 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 评估器类
# =========================================================
class PinRecEvaluator:
    def __init__(self, checkpoint_path):
        self.device = DEVICE
        print(f"🚀 Loading Model from: {checkpoint_path}")
        
        # 1. 加载 Config
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            self.config = PinRecConfig.from_json_file(config_path)
        else:
            self.config = PinRecConfig() # Fallback default
            
        # [Fix] 自动探测正确的属性名
        if hasattr(self.config, 'item_vocab_size'):
            self.vocab_size = self.config.item_vocab_size
        elif hasattr(self.config, 'vocab_size'):
            self.vocab_size = self.config.vocab_size
        elif hasattr(self.config, 'item_size'):
            self.vocab_size = self.config.item_size
        else:
            print("⚠️ Warning: Could not find vocab size in config. Using default 150000.")
            self.vocab_size = 150000 # 兜底默认值
            
        print(f"ℹ️ Model Vocab Size: {self.vocab_size}") 
        
        # 2. Item Tower
        self.item_tower = ItemTower(self.config).to(self.device)
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if os.path.exists(item_path):
            self.item_tower.load_state_dict(torch.load(item_path, map_location=self.device))
        self.item_tower.eval()
        
        # 3. User Tower
        self.user_tower = UserTower(self.config).to(self.device)
        # 加载 LoRA
        if hasattr(self.user_tower.llm, "load_adapter"):
            try:
                self.user_tower.llm.load_adapter(checkpoint_path, adapter_name="default")
            except: pass 
        # 加载 Heads
        head_path = os.path.join(checkpoint_path, "user_tower_heads.bin")
        if os.path.exists(head_path):
            self.user_tower.load_state_dict(torch.load(head_path, map_location=self.device), strict=False)
        self.user_tower.eval()
        
    def build_item_index(self):
        """构建全库索引"""
        # [Fix] 使用刚才获取到的 self.vocab_size
        MAX_ID = self.vocab_size
        print(f"🏗️ Building Item Index for {MAX_ID} items...")
        
        batch_size = 2048
        all_embs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, MAX_ID, batch_size)):
                end = min(i + batch_size, MAX_ID)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                
                embs = self.item_tower(ids)
                # [关键] 强制归一化 (与 V7 训练一致)
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.cpu())
                
        self.item_index = torch.cat(all_embs, dim=0).to(self.device)
        print(f"✅ Index Built: {self.item_index.shape}")

    def evaluate(self, test_file, num_samples=1000):
        print(f"🧪 Evaluating on {num_samples} samples...")
        
        samples = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                if line.strip(): samples.append(json.loads(line))
        
        # Metrics buckets
        hits = {k: 0 for k in TOP_K_EVAL}
        ranks = [] # 用于计算 Average Rank
        
        all_top1_ids = []
        score_diffs = [] 
        
        # Out of bounds check
        max_index_id = self.item_index.shape[0] - 1
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(samples), BATCH_SIZE)):
                batch = samples[batch_start : batch_start + BATCH_SIZE]
                
                # --- Collate ---
                max_len = max(len(s['history_ids']) for s in batch)
                B = len(batch)
                
                h_ids = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                h_acts = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                h_deltas = torch.zeros((B, max_len), dtype=torch.float, device=self.device)
                h_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                
                t_ids = []
                for i, s in enumerate(batch):
                    l = len(s['history_ids'])
                    h_ids[i, :l] = torch.tensor(s['history_ids'], device=self.device)
                    h_acts[i, :l] = torch.tensor(s['history_acts'], device=self.device)
                    h_deltas[i, :l] = torch.tensor(s['history_deltas'], device=self.device)
                    h_mask[i, :l] = 1
                    t_ids.append(s['target_1']['id'])
                
                # --- Encode ---
                flat_h_vecs = self.item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                dummy_t_acts = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                dummy_t_deltas = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                user_out = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, dummy_t_acts, dummy_t_deltas)
                
                # [关键] 归一化 (V7 Standard)
                user_vec = F.normalize(user_out[:, 0, :], p=2, dim=-1)
                
                # --- Retrieval ---
                scores = torch.matmul(user_vec, self.item_index.T) # [B, N_Items]
                
                # --- Metrics Calculation ---
                for i in range(B):
                    truth = t_ids[i]
                    
                    # [Safety Check] ID 是否越界
                    if truth > max_index_id:
                        print(f"⚠️ Warning: Target ID {truth} is out of index range (0-{max_index_id}). Skipping...")
                        continue
                    
                    # 获取该样本的真实排名 (Rank)
                    # 1. 获取 truth score
                    truth_score = scores[i, truth].item()
                    
                    # 2. 计算有多少个 Item 分数比 Truth 高
                    # (scores[i] > truth_score).sum() 比全排序快
                    rank = (scores[i] > truth_score).sum().item() + 1
                    ranks.append(rank)
                    
                    # 3. Hit Calculation
                    for k in TOP_K_EVAL:
                        if rank <= k:
                            hits[k] += 1
                            
                    # 4. Diversity (Top 1)
                    top1_idx = torch.argmax(scores[i]).item()
                    all_top1_ids.append(top1_idx)
                    
                    # 5. Gap
                    score_diffs.append(scores[i, top1_idx].item() - truth_score)

        # --- Report ---
        n = len(ranks)
        if n == 0: n = 1 # avoid div zero
        
        print("\n" + "="*40)
        print("🔍 PINREC V7 (FINAL DEBUG) REPORT")
        print("="*40)
        
        print("--- Recall Metrics ---")
        for k in TOP_K_EVAL:
            print(f"Recall@{k:<3} : {hits[k] / n:.2%}")
            
        print("\n--- Ranking Quality ---")
        print(f"Average Rank : {np.mean(ranks):.1f} / {max_index_id+1}")
        print(f"Median Rank  : {np.median(ranks):.1f}")
        print("(Avg Rank < 1000 说明模型很强; > 50000 说明还在瞎猜)")
        
        print("\n--- Diversity & Calibration ---")
        unique_recs = len(Counter(all_top1_ids))
        print(f"Unique Top-1s: {unique_recs} / {len(samples)}")
        print(f"Avg Score Gap: {np.mean(score_diffs):.4f}")
        print("="*40)

if __name__ == "__main__":
    if os.path.exists(BASE_DIR):
        subdirs = [d for d in os.listdir(BASE_DIR) if d.startswith("checkpoint")]
        if subdirs:
            subdirs.sort(key=lambda x: int(x.split("-")[1]))
            # 默认找最新的
            latest_ckpt = os.path.join(BASE_DIR, subdirs[-1])
            
            # 运行评估
            evaluator = PinRecEvaluator(latest_ckpt)
            evaluator.build_item_index()
            evaluator.evaluate(TEST_DATA, num_samples=1000)
        else:
            print("⏳ No checkpoints yet.")
    else:
        print(f"❌ Directory not found: {BASE_DIR}")