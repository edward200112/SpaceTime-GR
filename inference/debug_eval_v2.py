import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
import faiss 

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =================配置区域=================
# 建议使用最新的 checkpoint，比如 checkpoint-2500
CKPT_PATH = "/workspace/data/pinrec_ckpt_v2/checkpoint-13000" 
TEST_DATA_PATH = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl"
MAX_SAMPLES = 20 # 只看 20 条，看细节
BATCH_SIZE = 20
# =========================================

class TestDataset(Dataset):
    def __init__(self, data_path, max_samples):
        self.samples = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples: break
                if line.strip():
                    self.samples.append(json.loads(line))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_ids": torch.tensor(s['history_ids'], dtype=torch.long),
            "h_acts": torch.tensor(s['history_acts'], dtype=torch.long),
            "h_deltas": torch.tensor(s['history_deltas'], dtype=torch.float),
            "t_id": torch.tensor(s['target_1']['id'], dtype=torch.long),
            "t_act": torch.tensor(s['target_1']['act'], dtype=torch.long),
            "t_delta": torch.tensor(s['target_1']['delta'], dtype=torch.float)
        }

def collate_fn_eval(batch):
    h_ids = [b['h_ids'] for b in batch]
    h_acts = [b['h_acts'] for b in batch]
    h_deltas = [b['h_deltas'] for b in batch]
    max_len = max(len(h) for h in h_ids)
    B = len(batch)
    pad_ids = torch.zeros((B, max_len), dtype=torch.long)
    pad_acts = torch.zeros((B, max_len), dtype=torch.long)
    pad_deltas = torch.zeros((B, max_len), dtype=torch.float)
    mask = torch.zeros((B, max_len), dtype=torch.long)
    for i in range(B):
        l = len(h_ids[i])
        pad_ids[i, :l] = h_ids[i]
        pad_acts[i, :l] = h_acts[i]
        pad_deltas[i, :l] = h_deltas[i]
        mask[i, :l] = 1 
    return {
        "h_ids": pad_ids, "h_acts": pad_acts, "h_deltas": pad_deltas, "h_mask": mask,
        "t_ids": torch.stack([b['t_id'] for b in batch]),
        "t_act": torch.stack([b['t_act'] for b in batch]),
        "t_delta": torch.stack([b['t_delta'] for b in batch])
    }

def debug_evaluate():
    config = PinRecConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {CKPT_PATH}...")
    item_tower = ItemTower(config).to(device)
    item_tower.load_state_dict(torch.load(os.path.join(CKPT_PATH, "item_tower.bin"), map_location=device))
    item_tower.eval()
    
    user_tower = UserTower(config).to(device)
    user_tower.llm.load_adapter(CKPT_PATH, "default")
    user_tower.load_state_dict(torch.load(os.path.join(CKPT_PATH, "user_tower_heads.bin"), map_location=device), strict=False)
    user_tower.eval()
    
    # 1. Build Index
    print("Generating Index...")
    num_items = item_tower.content_feats.shape[0]
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, num_items, 1024):
            end = min(i + 1024, num_items)
            item_ids = torch.arange(i, end, device=device)
            embeds = item_tower(item_ids) 
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index = faiss.IndexFlatIP(1024) 
    index.add(all_embeddings.astype('float32'))
    
    # 2. Run Inference
    dataset = TestDataset(TEST_DATA_PATH, MAX_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_eval)
    
    print("\n>>> 开始诊断 (对比 Ground Truth 和 Top-1 的分数) <<<")
    with torch.no_grad():
        for batch in dataloader:
            # User Vector
            h_ids = batch['h_ids'].to(device)
            # ... 其他 inputs
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flat_h = item_tower(h_ids.view(-1)).view(h_ids.shape[0], h_ids.shape[1], -1)
                user_preds = user_tower(
                    flat_h, batch['h_acts'].to(device), batch['h_deltas'].to(device), batch['h_mask'].to(device),
                    batch['t_act'].to(device).unsqueeze(1), batch['t_delta'].to(device).unsqueeze(1)
                ).squeeze(1)
            
            # 归一化
            query_vecs = user_preds.float().cpu().numpy()
            
            # 检索 Top 5
            D, I = index.search(query_vecs, 5)
            
            # 逐个分析
            t_ids = batch['t_ids'].cpu().numpy()
            for i in range(len(t_ids)):
                target_id = t_ids[i]
                top1_id = I[i][0]
                top1_score = D[i][0]
                
                # 手动计算 Ground Truth 分数
                gt_vec = all_embeddings[target_id]
                gt_score = np.dot(query_vecs[i], gt_vec)
                
                print(f"Case {i}: Target={target_id} | Top1={top1_id}")
                print(f"   Scores -> Top1: {top1_score:.4f} | Truth: {gt_score:.4f} | Diff: {top1_score - gt_score:.4f}")
                
                if top1_id == target_id:
                    print("   🎉 HIT!")
                elif gt_score > 0.3:
                    print("   ⚠️  Truth Score 很高 (>0.3)，说明模型学到了！只是还没排进前几名。")
                else:
                    print("   ❌ Truth Score 很低，模型还没理解这个 User。")
                print("-" * 50)
            break # 只跑一个 batch

if __name__ == "__main__":
    debug_evaluate()