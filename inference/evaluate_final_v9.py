import sys
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# 配置区域
# =========================================================
# [关键] 指向 V6 Checkpoint 2000 (多样性最好的版本)
CHECKPOINT_PATH = "/workspace/data/pinrec_ckpt_v6_norm/checkpoint-11000"
TEST_DATA = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl" 
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

BATCH_SIZE = 128
TOP_K_EVAL = [1, 5, 10, 50, 100]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PinRecEvaluator:
    def __init__(self, checkpoint_path):
        self.device = DEVICE
        print(f"🚀 Loading Model from: {checkpoint_path}")
        
        self.config = PinRecConfig()
        
        # 1. Item Tower & Size Auto-Detection
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"Item Tower weights not found at {item_path}")
            
        item_state = torch.load(item_path, map_location='cpu')
        
        # [V9 核心修复] 自动从权重中探测 Vocab Size
        # 我们寻找最大的 Embedding 权重矩阵，通常是 content_proj 或 hash_tables
        detected_size = 150000 # 默认兜底
        for k, v in item_state.items():
            # 常见的 Embedding 权重名
            if 'embedding' in k or 'proj' in k or 'table' in k:
                if v.dim() == 2 and v.shape[0] > 10000: # 可能是 Item Embedding
                    print(f"   Found weight '{k}' with shape {v.shape}")
                    detected_size = v.shape[0]
                    # 如果有多个，通常最大的那个是 Item ID Embedding (如果有的话)
                    # 假设我们用 detected_size 作为上限
                    
        self.max_id = detected_size
        print(f"ℹ️ Auto-Detected Max Item ID: {self.max_id}")

        self.item_tower = ItemTower(self.config).to(self.device)
        # 允许非严格加载，防止 config size 和 state dict size 不匹配导致的报错
        try:
            self.item_tower.load_state_dict(item_state, strict=False)
        except Exception as e:
            print(f"⚠️ Warning during load: {e}")
            
        self.item_tower.eval()
        
        # 2. User Tower
        self.user_tower = UserTower(self.config).to(self.device)
        head_path = os.path.join(checkpoint_path, "user_tower_heads.bin")
        if os.path.exists(head_path):
            self.user_tower.load_state_dict(torch.load(head_path, map_location=self.device), strict=False)
        self.user_tower.eval()
        
    def build_item_index(self):
        print(f"🏗️ Building Index for {self.max_id} items...")
        batch_size = 2048
        all_embs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, self.max_id, batch_size)):
                end = min(i + batch_size, self.max_id)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                
                # 保护性调用：如果模型内部 embedding 只有 150000，传入 >150000 会崩
                # 这里我们已经在 __init__ 里把 max_id 限制为权重大小了，所以应该是安全的
                try:
                    embs = self.item_tower(ids)
                    embs = F.normalize(embs, p=2, dim=-1)
                    all_embs.append(embs.cpu())
                except RuntimeError as e:
                    print(f"\n❌ Error at index {i}-{end}: {e}")
                    print("Stopping index build early to prevent crash.")
                    break
                
        self.item_index = torch.cat(all_embs, dim=0).to(self.device)
        # 更新 max_id 为实际构建成功的数量
        self.max_id = self.item_index.shape[0]
        print(f"✅ Index Shape: {self.item_index.shape}")

    def evaluate(self, test_file, mapping_file, num_samples=1000):
        print(f"🧪 Evaluating {num_samples} samples...")
        
        # 加载 Mapping 以便做合法性检查 (可选)
        # valid_ids = set()
        # with open(mapping_file) as f:
        #     sid_map = json.load(f)
        #     # 假设 mapping 里能解析出 ID
        
        samples = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                if line.strip(): samples.append(json.loads(line))
        
        hits = {k: 0 for k in TOP_K_EVAL}
        ranks = []
        all_top1 = []
        
        skipped = 0
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(samples), BATCH_SIZE)):
                batch = samples[batch_start : batch_start + BATCH_SIZE]
                
                # Collate
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
                
                # Forward
                flat_h_ids = h_ids.view(-1)
                
                # [Fix] History IDs 越界保护
                # 如果历史记录里有 > max_id 的物品，clip 掉或者 mask 掉
                flat_h_ids = torch.clamp(flat_h_ids, 0, self.max_id - 1)
                
                flat_h_vecs = self.item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                dummy_acts = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                dummy_deltas = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                user_out = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, dummy_acts, dummy_deltas)
                user_vec = F.normalize(user_out[:, 0, :], p=2, dim=-1)
                
                # Retrieval
                scores = torch.matmul(user_vec, self.item_index.T)
                
                # Metrics
                for i in range(B):
                    truth = t_ids[i]
                    
                    # [关键] 如果 Truth ID 超出了索引范围，无法评估，只能跳过
                    if truth >= self.max_id:
                        skipped += 1
                        continue
                    
                    truth_score = scores[i, truth].item()
                    rank = (scores[i] > truth_score).sum().item() + 1
                    ranks.append(rank)
                    
                    for k in TOP_K_EVAL:
                        if rank <= k: hits[k] += 1
                    
                    top1_idx = torch.argmax(scores[i]).item()
                    all_top1.append(top1_idx)

        print("\n" + "="*40)
        print("📊 FINAL V6 CHECKPOINT-2000 RE-EVAL (V9)")
        print("="*40)
        n = len(ranks)
        print(f"Evaluated: {n}, Skipped (Out of Bounds): {skipped}")
        
        if n > 0:
            for k in TOP_K_EVAL:
                print(f"Recall@{k:<3}: {hits[k]/n:.2%}")
            print(f"Avg Rank  : {np.mean(ranks):.1f} / {self.max_id}")
            print(f"Median Rank: {np.median(ranks):.1f}")
            print(f"Unique Top1: {len(Counter(all_top1))} / {len(samples)}")
        print("="*40)

if __name__ == "__main__":
    try:
        evaluator = PinRecEvaluator(CHECKPOINT_PATH)
        evaluator.build_item_index()
        evaluator.evaluate(TEST_DATA, MAPPING_FILE, num_samples=1000)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print("Tip: If CUDA error persists, restart the kernel/notebook.")