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
# 配置：请指向 V6 的 Checkpoint 2000 (那个多样性最好的版本)
# =========================================================
CHECKPOINT_PATH = "/workspace/data/pinrec_ckpt_v6_norm/checkpoint-11000"
TEST_DATA = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl" 

BATCH_SIZE = 128
TOP_K_EVAL = [1, 5, 10, 50, 100]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PinRecEvaluator:
    def __init__(self, checkpoint_path):
        self.device = DEVICE
        print(f"🚀 Loading Model from: {checkpoint_path}")
        
        self.config = PinRecConfig()
        
        # 1. Load State Dict first to detect Size
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"Item Tower weights not found at {item_path}")
            
        item_state = torch.load(item_path, map_location='cpu')
        
        # [关键] 自动探测 Item Embedding 大小
        # 通常 key 是 'content_proj.weight' 或者 'hash_tables' 相关
        # 我们假设用 hash_tables 或 content_proj 来推断，或者更直接：看 item_tower 结构
        # PinRec V2 ItemTower 有 hash_tables (List of Embeddings)
        # 这是一个 Trick: 我们无法直接从 hash table size 推断 vocab size (因为是哈希)
        # 但我们可以尝试加载 config.json
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            file_conf = PinRecConfig.from_json_file(config_path)
            # 尝试读取各种可能的命名字段
            if hasattr(file_conf, 'item_vocab_size'): self.max_id = file_conf.item_vocab_size
            elif hasattr(file_conf, 'vocab_size'): self.max_id = file_conf.vocab_size
            elif hasattr(file_conf, 'item_size'): self.max_id = file_conf.item_size
            else: self.max_id = 200000 # Fallback
        else:
            self.max_id = 200000 # Safe Fallback
            
        print(f"ℹ️ Detected Max Item ID: {self.max_id}")

        # 2. Item Tower
        self.item_tower = ItemTower(self.config).to(self.device)
        self.item_tower.load_state_dict(item_state)
        self.item_tower.eval()
        
        # 3. User Tower
        self.user_tower = UserTower(self.config).to(self.device)
        if hasattr(self.user_tower.llm, "load_adapter"):
            try: self.user_tower.llm.load_adapter(checkpoint_path, adapter_name="default")
            except: pass
        
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
                embs = self.item_tower(ids)
                embs = F.normalize(embs, p=2, dim=-1) # V6 必须归一化
                all_embs.append(embs.cpu())
                
        self.item_index = torch.cat(all_embs, dim=0).to(self.device)
        print(f"✅ Index Shape: {self.item_index.shape}")

    def evaluate(self, test_file, num_samples=1000):
        print(f"🧪 Evaluating {num_samples} samples...")
        samples = []
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples: break
                if line.strip(): samples.append(json.loads(line))
        
        hits = {k: 0 for k in TOP_K_EVAL}
        ranks = []
        all_top1 = []
        
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
                flat_h_vecs = self.item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                dummy_acts = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                dummy_deltas = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                user_out = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, dummy_acts, dummy_deltas)
                user_vec = F.normalize(user_out[:, 0, :], p=2, dim=-1) # V6 归一化
                
                # Retrieval
                scores = torch.matmul(user_vec, self.item_index.T)
                
                # Metrics
                for i in range(B):
                    truth = t_ids[i]
                    if truth >= self.max_id: continue # Skip invalid
                    
                    truth_score = scores[i, truth].item()
                    rank = (scores[i] > truth_score).sum().item() + 1
                    ranks.append(rank)
                    
                    for k in TOP_K_EVAL:
                        if rank <= k: hits[k] += 1
                    
                    top1_idx = torch.argmax(scores[i]).item()
                    all_top1.append(top1_idx)

        print("\n" + "="*40)
        print("📊 FINAL V6 CHECKPOINT-2000 RE-EVAL")
        print("="*40)
        n = len(ranks)
        if n > 0:
            for k in TOP_K_EVAL:
                print(f"Recall@{k:<3}: {hits[k]/n:.2%}")
            print(f"Avg Rank  : {np.mean(ranks):.1f}")
            print(f"Median Rank: {np.median(ranks):.1f}")
            print(f"Unique Top1: {len(Counter(all_top1))} / {len(samples)}")
        print("="*40)

if __name__ == "__main__":
    evaluator = PinRecEvaluator(CHECKPOINT_PATH)
    evaluator.build_item_index()
    evaluator.evaluate(TEST_DATA, num_samples=1000)