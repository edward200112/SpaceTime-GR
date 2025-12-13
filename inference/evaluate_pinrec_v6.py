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
# 配置区域 (V6 Normalized)
# =========================================================
# [关键] 自动锁定 V6 训练目录
BASE_DIR = "/workspace/data/pinrec_ckpt_v6_norm"

# [关键] 验证集路径 (已确认为 validation_ultimate.jsonl)
TEST_DATA = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl" 

# 映射文件
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
        print(f"🚀 Loading V6 Model from: {checkpoint_path}")
        
        self.config = PinRecConfig()
        
        # 1. Item Tower
        self.item_tower = ItemTower(self.config).to(self.device)
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if os.path.exists(item_path):
            self.item_tower.load_state_dict(torch.load(item_path, map_location=self.device))
        else:
            print("❌ Error: item_tower.bin not found!")
        self.item_tower.eval()
        
        # 2. User Tower (LoRA + Heads)
        self.user_tower = UserTower(self.config).to(self.device)
        
        # 加载 LoRA (修复 adapter_name 参数问题)
        if hasattr(self.user_tower.llm, "load_adapter"):
            try:
                self.user_tower.llm.load_adapter(checkpoint_path, adapter_name="default")
            except TypeError:
                # 旧版本兼容
                self.user_tower.llm.load_adapter(checkpoint_path)
            except Exception as e:
                print(f"⚠️ Adapter load warning: {e}")

        # 加载 Heads
        head_path = os.path.join(checkpoint_path, "user_tower_heads.bin")
        if os.path.exists(head_path):
            self.user_tower.load_state_dict(torch.load(head_path, map_location=self.device), strict=False)
        else:
            print("⚠️ Warning: user_tower_heads.bin not found!")
            
        self.user_tower.eval()
        
    def build_item_index(self):
        """构建全库索引 (强制归一化)"""
        print("🏗️ Building Full Item Index...")
        # 假设最大 Item ID (根据配置或数据)
        MAX_ITEM_ID = 150000 
        batch_size = 2048
        all_embs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, MAX_ITEM_ID, batch_size)):
                end = min(i + batch_size, MAX_ITEM_ID)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                
                # Encode
                embs = self.item_tower(ids)
                # [关键] Item 向量归一化
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.cpu())
                
        self.item_index = torch.cat(all_embs, dim=0).to(self.device)
        print(f"✅ Index Built: {self.item_index.shape}")

    def evaluate(self, test_file, num_samples=1000):
        print(f"🧪 Evaluating on {num_samples} samples...")
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return

        samples = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                if line.strip(): samples.append(json.loads(line))
        
        hits_1, hits_5, hits_10 = 0, 0, 0
        all_top1_ids = []
        score_diffs = [] 
        
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
                
                # --- Encode Item History ---
                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = self.item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                # Placeholder for target context
                dummy_t_acts = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                dummy_t_deltas = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                # --- Encode User ---
                user_out = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, dummy_t_acts, dummy_t_deltas)
                user_vec = user_out[:, 0, :] 
                
                # [关键] User 向量归一化 (与 V6 训练保持一致)
                user_vec = F.normalize(user_vec, p=2, dim=-1)
                
                # --- Retrieval (Cosine Similarity) ---
                # Index 也是归一化的，所以 Dot Product = Cosine Similarity
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
                    
                    if truth < len(scores[i]):
                        s_truth = scores[i][truth].item()
                        s_top1 = topk_scores[i][0].item()
                        score_diffs.append(s_top1 - s_truth)

        # --- Report ---
        print("\n" + "="*40)
        print("📊 PINREC V6 (NORMALIZED) REPORT")
        print("="*40)
        print(f"Recall@1  : {hits_1 / len(samples):.2%}")
        print(f"Recall@5  : {hits_5 / len(samples):.2%}")
        print(f"Recall@10 : {hits_10 / len(samples):.2%}")
        
        print("-" * 20)
        print("🌈 Diversity Check (Mode Collapse Detector)")
        counter = Counter(all_top1_ids)
        unique_recs = len(counter)
        print(f"Unique Items Recommended: {unique_recs} / {len(samples)}")
        print("Top 5 Dominant Items:")
        for iid, count in counter.most_common(5):
            print(f"  ID {iid}: {count} times ({count/len(samples):.1%})")
            
        print("-" * 20)
        print("⚖️ Score Gap (Smaller is better)")
        if score_diffs:
            avg_diff = sum(score_diffs) / len(score_diffs)
            print(f"Avg Gap (Top1 - Truth): {avg_diff:.4f}")
        print("="*40)

if __name__ == "__main__":
    # 自动寻找 V6 目录下最新的 Checkpoint
    if os.path.exists(BASE_DIR):
        subdirs = [d for d in os.listdir(BASE_DIR) if d.startswith("checkpoint")]
        if subdirs:
            subdirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_ckpt = os.path.join(BASE_DIR, subdirs[-1])
            
            # 实例化并运行
            evaluator = PinRecEvaluator(latest_ckpt)
            evaluator.build_item_index()
            evaluator.evaluate(TEST_DATA, num_samples=1000)
        else:
            print(f"⏳ No checkpoints found in {BASE_DIR}. Training hasn't started or saved yet.")
    else:
        print(f"❌ Directory not found: {BASE_DIR}")
        print("Please run 'training/train_pinrec_v6.py' first!")