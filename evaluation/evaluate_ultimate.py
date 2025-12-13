import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
import faiss  # 用于向量检索

# 导入模型定义
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.pinrec_ultimate import PinRecConfig, ItemTower, UserTower

# =================配置区域=================
# 指向你刚刚训练好的 Checkpoint 路径
CKPT_PATH = "/workspace/data/pinrec_ckpt/checkpoint-88500" 
# 指向测试集路径 (确保你有 test_ultimate.jsonl，或者先用 train 的一小部分测试)
TEST_DATA_PATH = "/workspace/data/processed_pinrec/train_ultimate.jsonl" 
# 为了快速演示，只测试前 1000 个 Batch，正式评估时请设为 None
MAX_TEST_BATCHES = 100 
BATCH_SIZE = 64
TOP_K = 20  # 计算 Recall@20
# =========================================

class TestDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"Loading test data from {data_path}...")
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i > 10000: break # Demo: 只加载前1w条，防止内存溢出，正式跑请去掉
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "history": torch.tensor(s['history_ids'], dtype=torch.long),
            # 测试时我们需要 Ground Truth Target
            "target_ids": torch.tensor([s['target_1']['bid_int'], s['target_2']['bid_int']], dtype=torch.long),
            "outcomes": torch.tensor([s['target_1']['outcome'], s['target_2']['outcome']], dtype=torch.long),
            "deltas": torch.tensor([s['target_1']['delta_sec'], s['target_2']['delta_sec']], dtype=torch.float)
        }

def collate_fn(batch):
    # Same as training
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
    
    # 1. Load Item Tower
    item_tower = ItemTower(config).to(device)
    item_tower.load_state_dict(torch.load(os.path.join(ckpt_path, "item_tower.bin"), map_location=device))
    item_tower.eval()
    
    # 2. Load User Tower
    user_tower = UserTower(config).to(device)
    # 2a. Load LoRA Weights
    print("Loading LoRA adapter...")
    user_tower.llm.load_adapter(ckpt_path, "default")
    # 2b. Load Custom Heads (Projector, etc.)
    print("Loading Custom Heads...")
    heads_path = os.path.join(ckpt_path, "user_tower_heads.bin")
    # strict=False 因为 LLM 权重不在这个 bin 里
    user_tower.load_state_dict(torch.load(heads_path, map_location=device), strict=False)
    user_tower.eval()
    
    return item_tower, user_tower

def generate_item_index(item_tower, batch_size=512):
    """
    遍历所有 Item ID，生成向量库
    """
    print("Generating Item Index...")
    # 假设 content_feats 的行数就是 item 总数
    num_items = item_tower.content_feats.shape[0]
    all_embeddings = []
    
    # 分批次 inference
    with torch.no_grad():
        for i in tqdm(range(0, num_items, batch_size), desc="Indexing Items"):
            end = min(i + batch_size, num_items)
            item_ids = torch.arange(i, end, device=item_tower.content_feats.device)
            embeds = item_tower(item_ids) # [B, 1024]
            # 归一化 (Cosine Similarity 需要)
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeddings.append(embeds.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Build FAISS Index (IP = Inner Product，归一化后等价于 Cosine)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(all_embeddings)
    print(f"Index built with {index.ntotal} items.")
    return index

def evaluate():
    config = PinRecConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    item_tower, user_tower = load_models(config, CKPT_PATH, device)
    
    # 2. 构建商品索引 (Corpus)
    # 注意：这里我们遍历所有已知的 Item ID
    index = generate_item_index(item_tower)
    
    # 3. 准备测试数据
    dataset = TestDataset(TEST_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    
    # 4. 检索测试
    hits = 0
    total_queries = 0
    
    print("Running Retrieval Evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if MAX_TEST_BATCHES and i >= MAX_TEST_BATCHES: break
            
            history = batch['history'].to(device)
            targets = batch['target_ids'].to(device) # [B, 2]
            outcomes = batch['outcomes'].to(device)
            deltas = batch['deltas'].to(device)
            
            # --- User Tower Inference ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Encode History
                B, Seq = history.shape
                flat_hist_vecs = item_tower(history.view(-1))
                hist_vecs = flat_hist_vecs.view(B, Seq, -1)
                
                # Get User Vectors [B, 2, Dim]
                user_preds = user_tower(hist_vecs, outcomes, deltas)
            
            # 转回 float32 并在 CPU 上做检索 (FAISS GPU 显存可能不够，这里用 CPU 检索比较稳)
            user_preds = user_preds.float().cpu().numpy()
            targets = targets.cpu().numpy()
            
            # 遍历 Batch 中的每个用户
            for b in range(B):
                # 针对每个时间尺度 (Immediate vs Future)
                for q_idx in range(2):
                    query_vec = user_preds[b, q_idx].reshape(1, -1)
                    target_item_id = targets[b, q_idx]
                    
                    # FAISS Search
                    D, I = index.search(query_vec, TOP_K) # I is indices (Item IDs)
                    
                    # Check Hit
                    if target_item_id in I[0]:
                        hits += 1
                    total_queries += 1
    
    # 5. 结果
    recall = hits / total_queries
    print(f"\n================Results================")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Recall@{TOP_K}: {recall:.4f}")
    print(f"=======================================")

if __name__ == "__main__":
    evaluate()