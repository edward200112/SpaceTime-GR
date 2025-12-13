import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
import faiss 

# --- 路径配置 (保持不变) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.pinrec_ultimate import PinRecConfig, ItemTower, UserTower

# =================配置区域=================
CKPT_PATH = "/workspace/data/pinrec_ckpt/checkpoint-88500" 
TEST_DATA_PATH = "/workspace/data/processed_pinrec/train_ultimate.jsonl" 
# 【诊断模式】只跑前 5 个 Batch 即可，目的是看日志
MAX_TEST_BATCHES = 5 
BATCH_SIZE = 64
TOP_K = 20  
# =========================================

class TestDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"Loading test data from {data_path}...")
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i > 2000: break 
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "history": torch.tensor(s['history_ids'], dtype=torch.long),
            "target_ids": torch.tensor([s['target_1']['bid_int'], s['target_2']['bid_int']], dtype=torch.long),
            "outcomes": torch.tensor([s['target_1']['outcome'], s['target_2']['outcome']], dtype=torch.long),
            "deltas": torch.tensor([s['target_1']['delta_sec'], s['target_2']['delta_sec']], dtype=torch.float)
        }

def collate_fn(batch):
    histories = [b['history'] for b in batch]
    max_len = max(len(h) for h in histories)
    padded_h = torch.zeros(len(histories), max_len, dtype=torch.long)
    for i, h in enumerate(histories):
        padded_h[i, :len(h)] = h
        
    return {
        "history": padded_h,
        "target_ids": torch.stack([b['target_ids'] for b in batch]),
        "outcomes": torch.stack([b['outcomes'] for b in batch]),
        "deltas": torch.stack([b['deltas'] for b in batch])
    }

def load_models(config, ckpt_path, device):
    print(f"Loading models from {ckpt_path}...")
    item_tower = ItemTower(config).to(device)
    item_tower.load_state_dict(torch.load(os.path.join(ckpt_path, "item_tower.bin"), map_location=device))
    item_tower.eval()
    
    user_tower = UserTower(config).to(device)
    print("Loading LoRA adapter...")
    user_tower.llm.load_adapter(ckpt_path, "default")
    print("Loading Custom Heads...")
    heads_path = os.path.join(ckpt_path, "user_tower_heads.bin")
    user_tower.load_state_dict(torch.load(heads_path, map_location=device), strict=False)
    user_tower.eval()
    
    return item_tower, user_tower

def generate_item_index(item_tower, batch_size=512):
    print("Generating Item Index...")
    num_items = item_tower.content_feats.shape[0]
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_items, batch_size), desc="Indexing Items"):
            end = min(i + batch_size, num_items)
            item_ids = torch.arange(i, end, device=item_tower.content_feats.device)
            embeds = item_tower(item_ids) 
            # 必须归一化
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(all_embeddings)
    print(f"Index built with {index.ntotal} items.")
    return index

def evaluate():
    config = PinRecConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    item_tower, user_tower = load_models(config, CKPT_PATH, device)
    index = generate_item_index(item_tower)
    
    dataset = TestDataset(TEST_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    
    hits = 0
    total_queries = 0
    debug_print_count = 0
    
    print("Running Retrieval Evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if MAX_TEST_BATCHES and i >= MAX_TEST_BATCHES: break
            
            history = batch['history'].to(device)
            targets = batch['target_ids'].to(device) 
            outcomes = batch['outcomes'].to(device)
            deltas = batch['deltas'].to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                B, Seq = history.shape
                flat_hist_vecs = item_tower(history.view(-1))
                hist_vecs = flat_hist_vecs.view(B, Seq, -1)
                user_preds = user_tower(hist_vecs, outcomes, deltas)
            
            # === 1. User 向量归一化 ===
            user_preds = F.normalize(user_preds.float(), p=2, dim=-1)
            
            # 转为 numpy 进行检索
            user_preds_np = user_preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for b in range(B):
                for q_idx in range(2):
                    query_vec = user_preds_np[b, q_idx].reshape(1, -1).astype('float32')
                    target_item_id = targets_np[b, q_idx]
                    
                    # === 2. 向量检索 ===
                    D, I = index.search(query_vec, TOP_K) 
                    
                    # === 3. [诊断核心] 强制对比分值 ===
                    if debug_print_count < 5:
                        print(f"\n[Diagnostic Case {debug_print_count}]")
                        print(f"Target ID (Label): {target_item_id}")
                        
                        top1_id = I[0][0]
                        top1_score = D[0][0]
                        print(f"Model Top-1 Prediction: ID={top1_id}, Score={top1_score:.4f}")
                        
                        # --- 手动计算 Ground Truth 的真实分数 ---
                        # 我们把 target_item_id 喂给 Item Tower，看看它生成的向量和 User 向量到底有多远
                        # 步骤 A: 获取 Target 的 Embedding
                        tgt_input = torch.tensor([target_item_id], device=device, dtype=torch.long)
                        gt_embed = item_tower(tgt_input) # [1, Dim]
                        gt_embed = F.normalize(gt_embed, p=2, dim=-1) # 必须归一化
                        
                        # 步骤 B: 获取 User 的 Embedding (转回 Tensor)
                        user_embed_tensor = torch.tensor(query_vec, device=device) # [1, Dim]
                        
                        # 步骤 C: 计算余弦相似度 (Dot Product)
                        true_score = torch.sum(user_embed_tensor * gt_embed).item()
                        
                        print(f"Ground Truth Score: {true_score:.4f}")
                        print(f"Difference (Top1 - GT): {top1_score - true_score:.4f}")
                        
                        if true_score < 0.1:
                            print(">>> [WARNING] GT Score is extremely low. ID Mismatch confirmed.")
                        elif true_score > top1_score - 0.05:
                            print(">>> [INFO] GT Score is high. Model is working, maybe just Top-K cutoff issue.")
                            
                        debug_print_count += 1
                    # =======================================

                    if target_item_id in I[0]:
                        hits += 1
                    total_queries += 1
    
    if total_queries == 0: return

    recall = hits / total_queries
    print(f"\n================Results================")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Recall@{TOP_K}: {recall:.4f}")
    print(f"=======================================")

if __name__ == "__main__":
    evaluate()