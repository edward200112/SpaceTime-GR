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
# [自动检测] 如果你不确定具体是哪个 checkpoint，代码会自动找最新的
CHECKPOINT_DIR = "/workspace/data/pinrec_ckpt_v5_mild_logq/checkpoint-7000" 

# 测试数据
TEST_DATA = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl"
# 也可以用训练集的一部分来验证拟合情况
# TEST_DATA = "/workspace/data/processed/train_balanced_pinrec.jsonl"

MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

BATCH_SIZE = 128
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 评估器类
# =========================================================
class PinRecEvaluator:
    def __init__(self, checkpoint_path):
        self.device = DEVICE
        print(f"Loading models from {checkpoint_path}...")
        
        self.config = PinRecConfig()
        
        # 1. 加载 Item Tower
        self.item_tower = ItemTower(self.config).to(self.device)
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if os.path.exists(item_path):
            self.item_tower.load_state_dict(torch.load(item_path, map_location=self.device))
        else:
            print("⚠️ Warning: item_tower.bin not found!")
        self.item_tower.eval()
        
        # 2. 加载 User Tower (包含 LoRA 和 Projector Heads)
        self.user_tower = UserTower(self.config).to(self.device)
        
        # [Fix] 修复 load_adapter 报错，指定 adapter_name="default"
        if hasattr(self.user_tower.llm, "load_adapter"):
            try:
                self.user_tower.llm.load_adapter(checkpoint_path, adapter_name="default")
            except Exception as e:
                print(f"⚠️ Load adapter failed: {e}. Trying to ignore name...")
                # 某些旧版本可能不需要 name，或者通过 config 加载
                pass

        # 加载 Projector Heads (非 LLM 部分)
        head_path = os.path.join(checkpoint_path, "user_tower_heads.bin")
        if os.path.exists(head_path):
            self.user_tower.load_state_dict(torch.load(head_path, map_location=self.device), strict=False)
        else:
            print("⚠️ Warning: user_tower_heads.bin not found!")
            
        self.user_tower.eval()
        
    def build_item_index(self, mapping_file):
        """预计算所有 Item 的 Embedding，构建索引"""
        print("Building Item Index (Encoding all items)...")
        
        # 这里假设最大的 Item ID 是训练时已知的。
        # 为了保险起见，我们直接扫描一遍 mapping file 确定最大 ID
        max_id = 0
        with open(mapping_file, 'r') as f:
            sid_map = json.load(f)
            # 假设 sid_mapping 的结构里能推导出 ID，或者我们直接用 config 的 vocab_size
            # PinRecConfig 默认可能是 200000
            pass
            
        # 直接使用 Config 中的 Item Size 或一个足够大的数
        MAX_ITEM_ID = 150000 
        batch_size = 2048
        all_embs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, MAX_ITEM_ID, batch_size)):
                end = min(i + batch_size, MAX_ITEM_ID)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                
                # Encode & Normalize
                embs = self.item_tower(ids)
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.cpu())
                
        self.item_index = torch.cat(all_embs, dim=0).to(self.device) # [N_Items, Dim]
        print(f"Item Index Built: {self.item_index.shape}")

    def evaluate(self, test_file, num_samples=1000):
        print(f"Evaluating on {num_samples} samples from {test_file}...")
        
        samples = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                if line.strip(): samples.append(json.loads(line))
        
        hits_1 = 0
        hits_5 = 0
        hits_10 = 0
        
        # 诊断数据
        all_top1_ids = []
        score_diffs = [] # Top1 Score - Truth Score
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(samples), BATCH_SIZE)):
                batch = samples[batch_start : batch_start + BATCH_SIZE]
                
                # --- Quick Collate ---
                max_len = 0
                for s in batch: max_len = max(max_len, len(s['history_ids']))
                
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
                
                # --- Item Tower Encode History ---
                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = self.item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                # --- User Tower Encode ---
                # Placeholder for target context (Use padding or dummy)
                dummy_t_acts = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                dummy_t_deltas = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                user_out = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, dummy_t_acts, dummy_t_deltas)
                user_vec = user_out[:, 0, :] # [B, Dim]
                user_vec = F.normalize(user_vec, p=2, dim=-1)
                
                # --- Retrieval ---
                scores = torch.matmul(user_vec, self.item_index.T)
                topk_scores, topk_indices = torch.topk(scores, k=TOP_K, dim=1)
                
                # --- Metrics ---
                for i in range(B):
                    truth = t_ids[i]
                    preds = topk_indices[i].cpu().tolist()
                    
                    if truth in preds[:1]: hits_1 += 1
                    if truth in preds[:5]: hits_5 += 1
                    if truth in preds: hits_10 += 1
                    
                    all_top1_ids.append(preds[0])
                    
                    # Truth Score check
                    if truth < len(scores[i]):
                        s_truth = scores[i][truth].item()
                        s_top1 = topk_scores[i][0].item()
                        score_diffs.append(s_top1 - s_truth)

        # --- Report ---
        print("\n" + "="*40)
        print("🔍 PINREC V3 (LogQ Balanced) REPORT")
        print("="*40)
        print(f"Recall@1  : {hits_1 / num_samples:.2%}")
        print(f"Recall@5  : {hits_5 / num_samples:.2%}")
        print(f"Recall@10 : {hits_10 / num_samples:.2%}")
        
        print("-" * 20)
        print("📊 Diversity Analysis")
        counter = Counter(all_top1_ids)
        unique_recs = len(counter)
        print(f"Unique Items Recommended: {unique_recs} / {len(samples)}")
        print("Top 5 Most Popular Recommendations:")
        for iid, count in counter.most_common(5):
            print(f"  ID {iid}: {count} times ({count/len(samples):.1%})")
            
        print("-" * 20)
        print("⚖️ Score Calibration")
        if score_diffs:
            avg_diff = sum(score_diffs) / len(score_diffs)
            print(f"Avg Gap (Top1 - Truth): {avg_diff:.4f}")
        print("="*40)

if __name__ == "__main__":
    # 自动寻找最新的 checkpoint
    base_dir = "/workspace/data/pinrec_ckpt_v4_stable"
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint")]
        if subdirs:
            subdirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_ckpt = os.path.join(base_dir, subdirs[-1])
            print(f"Using latest checkpoint: {latest_ckpt}")
            evaluator = PinRecEvaluator(latest_ckpt)
            evaluator.build_item_index(MAPPING_FILE)
            evaluator.evaluate(TEST_DATA, num_samples=1000)
        else:
            print("No checkpoints found. Train more!")
    else:
        print(f"Directory not found: {base_dir}")