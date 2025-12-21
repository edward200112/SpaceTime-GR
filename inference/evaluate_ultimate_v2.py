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

# 导入模型 (使用 v2 版本)
from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =================配置区域=================
# 请修改为你想要评估的 Checkpoint 路径
CKPT_PATH = "/workspace/data/pinrec_ckpt_v2/checkpoint-1000" 
# 验证集路径
TEST_DATA_PATH = "/workspace/data/processed_pinrec_v2/validation_ultimate.jsonl"
# 评估样本数
MAX_SAMPLES_TO_EVAL = 1000 
BATCH_SIZE = 64
# 定义所有想要评估的 K 值
K_LIST = [20, 50, 100] 
# =========================================

# --- 评估指标计算 ---
def calculate_metrics_multi_k(retrieved_ids, target_ids, k_list):
    """
    同时计算多个 K 值的指标
    """
    total_queries = len(target_ids)
    if total_queries == 0:
        return {}, {}

    target_ids_expanded = np.expand_dims(target_ids, axis=1) # [N, 1]
    
    recall_results = {}
    ndcg_results = {}
    
    for k in k_list:
        # --- Recall@K ---
        # 检查 Top-K 中是否包含 Target
        hits = np.any(retrieved_ids[:, :k] == target_ids_expanded, axis=1).sum()
        recall_results[k] = hits / total_queries

        # --- NDCG@K ---
        ndcg_sum = 0.0
        for i in range(total_queries):
            # 找到 Target 在检索列表中的 rank (0-based)
            # np.where 返回的是 tuple，取 [0] 获取索引数组
            rank_arr = np.where(retrieved_ids[i, :k] == target_ids[i])[0]
            
            if rank_arr.size > 0:
                rank = rank_arr[0]
                # DCG = 1 / log2(rank + 2)
                dcg = 1.0 / np.log2(rank + 2)
                ndcg_sum += dcg
        
        ndcg_results[k] = ndcg_sum / total_queries
    
    return recall_results, ndcg_results

# --- Dataset (保持不变) ---
class TestDataset(Dataset):
    def __init__(self, data_path, max_samples):
        self.samples = []
        print(f"Loading test data from {data_path}...")
        try:
            with open(data_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples: break
                    if line.strip():
                        self.samples.append(json.loads(line))
        except FileNotFoundError:
            print(f"❌ Error: File not found {data_path}")
            sys.exit(1)
        print(f"Loaded {len(self.samples)} samples.")

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

def load_models_and_index(config, ckpt_path, device):
    print(f"Loading models from {ckpt_path}...")
    item_tower = ItemTower(config).to(device)
    item_tower.load_state_dict(torch.load(os.path.join(ckpt_path, "item_tower.bin"), map_location=device))
    item_tower.eval()
    
    user_tower = UserTower(config).to(device)
    user_tower.llm.load_adapter(ckpt_path, "default")
    heads_path = os.path.join(ckpt_path, "user_tower_heads.bin")
    user_tower.load_state_dict(torch.load(heads_path, map_location=device), strict=False)
    user_tower.eval()
    
    print("Generating Item Index...")
    num_items = item_tower.content_feats.shape[0]
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_items, 512), desc="Indexing Items"):
            end = min(i + 512, num_items)
            item_ids = torch.arange(i, end, device=item_tower.content_feats.device)
            embeds = item_tower(item_ids) 
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(all_embeddings.astype('float32'))
    print(f"Index built with {index.ntotal} items.")
    return item_tower, user_tower, index

def evaluate_full():
    config = PinRecConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found: {CKPT_PATH}")
        return

    item_tower, user_tower, index = load_models_and_index(config, CKPT_PATH, device)
    dataset = TestDataset(TEST_DATA_PATH, MAX_SAMPLES_TO_EVAL)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_eval, num_workers=4)
    
    all_targets = []
    all_retrieved = []
    
    # 最大的 K 值，用于 Faiss 检索
    MAX_K = max(K_LIST)
    
    print(f"Running Full Retrieval Evaluation (Max K={MAX_K})...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            h_ids = batch['h_ids'].to(device)
            h_acts = batch['h_acts'].to(device)
            h_deltas = batch['h_deltas'].to(device)
            h_mask = batch['h_mask'].to(device)
            
            t_ids = batch['t_ids'].cpu().numpy()
            t_act = batch['t_act'].to(device)
            t_delta = batch['t_delta'].to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                user_preds = user_tower(
                    h_vecs, h_acts, h_deltas, h_mask,
                    t_act.unsqueeze(1), t_delta.unsqueeze(1)
                ).squeeze(1)
            
            query_vecs = user_preds.float().cpu().numpy().astype('float32')
            
            # 使用最大的 K 进行检索
            D, I = index.search(query_vecs, MAX_K)
            
            all_targets.append(t_ids)
            all_retrieved.append(I)
            
    final_targets = np.concatenate(all_targets, axis=0)
    final_retrieved = np.concatenate(all_retrieved, axis=0)
    
    recall_res, ndcg_res = calculate_metrics_multi_k(final_retrieved, final_targets, K_LIST)
    
    print(f"\n================Full Retrieval Results==================")
    print(f"Evaluated Samples: {len(final_targets)}")
    print(f"Corpus Size: {index.ntotal}")
    print(f"-------------------------------------------------------")
    for k in K_LIST:
        print(f"✅ Recall@{k:<3}: {recall_res[k]:.4f} | NDCG@{k:<3}: {ndcg_res[k]:.4f}")
    print(f"========================================================")

if __name__ == "__main__":
    evaluate_full()