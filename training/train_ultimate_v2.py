import sys
import os
import glob
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# --- 导入模型 (假设路径已设置) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
# =========================================================
# Dataset & Collate (保持不变)
# =========================================================
class UltimateDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")
                
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_ids": torch.tensor(s['history_ids'], dtype=torch.long),
            "h_acts": torch.tensor(s['history_acts'], dtype=torch.long),
            "h_deltas": torch.tensor(s['history_deltas'], dtype=torch.float),
            
            "t_ids": torch.tensor([s['target_1']['id'], s['target_2']['id']], dtype=torch.long),
            "t_acts": torch.tensor([s['target_1']['act'], s['target_2']['act']], dtype=torch.long),
            "t_deltas": torch.tensor([s['target_1']['delta'], s['target_2']['delta']], dtype=torch.float)
        }

def collate_fn(batch):
    # Padding History
    h_ids = [b['h_ids'] for b in batch]
    h_acts = [b['h_acts'] for b in batch]
    h_deltas = [b['h_deltas'] for b in batch]
    
    max_len = max(len(h) for h in h_ids)
    B = len(batch)
    
    pad_ids = torch.zeros((B, max_len), dtype=torch.long)
    pad_acts = torch.zeros((B, max_len), dtype=torch.long)
    pad_deltas = torch.zeros((B, max_len), dtype=torch.float)
    mask = torch.zeros((B, max_len), dtype=torch.long) # 0=pad, 1=valid
    
    for i in range(B):
        l = len(h_ids[i])
        pad_ids[i, :l] = h_ids[i]
        pad_acts[i, :l] = h_acts[i]
        pad_deltas[i, :l] = h_deltas[i]
        mask[i, :l] = 1 # Valid part
        
    return {
        "h_ids": pad_ids,
        "h_acts": pad_acts,
        "h_deltas": pad_deltas,
        "h_mask": mask,
        
        "t_ids": torch.stack([b['t_ids'] for b in batch]),
        "t_acts": torch.stack([b['t_acts'] for b in batch]),
        "t_deltas": torch.stack([b['t_deltas'] for b in batch])
    }
# =========================================================
# Checkpoint Helpers
# =========================================================
def manage_checkpoints(output_dir, limit):
    """只保留最近的 limit 个 checkpoint"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    
    if len(checkpoints) > limit:
        to_delete = checkpoints[:-limit]
        for ckpt in to_delete:
            print(f"🗑️ Deleting old checkpoint: {ckpt}")
            try:
                shutil.rmtree(ckpt)
            except Exception as e:
                print(f"Error deleting {ckpt}: {e}")

def save_checkpoint(user_tower, item_tower, optimizer, step, output_dir, max_keep):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 保存 UserTower (LoRA权重 + Custom Heads)
    if hasattr(user_tower.llm, "save_pretrained"):
        user_tower.llm.save_pretrained(save_path)
        
    # 保存非 LLM 部分 (Projector, Embeddings 等)
    custom_state = {
        k: v for k, v in user_tower.state_dict().items() 
        if "llm" not in k
    }
    torch.save(custom_state, os.path.join(save_path, "user_tower_heads.bin"))
    
    # 2. 保存 ItemTower
    torch.save(item_tower.state_dict(), os.path.join(save_path, "item_tower.bin"))
    
    # 3. 保存 Optimizer (可选，用于断点续训)
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    
    print(f"💾 Checkpoint saved to {save_path}")
    
    # 4. 删除旧的
    manage_checkpoints(output_dir, max_keep)

# =========================================================
# Evaluation Helper
# =========================================================
def in_batch_recall_at_k(user_preds, target_vecs, k=10):
    """
    计算 In-Batch Recall@K (快速评估训练进度)
    user_preds: [B, NQ, Dim] (Normalized)
    target_vecs: [B, NQ, Dim] (Normalized)
    """
    B = user_preds.shape[0]
    NQ = user_preds.shape[1]
    total_queries = 0
    hits = 0
    
    # 遍历 NQ (通常是 2, immediate and future)
    for i in range(NQ):
        q = user_preds[:, i, :] # [B, Dim]
        pos = target_vecs[:, i, :] # [B, Dim]
        
        # 相似度矩阵: [B, B]
        # B行代表B个Query，B列代表B个Target
        similarity_matrix = torch.matmul(q, pos.T)
        
        # 找到每个 Query 的 Top-K 索引
        # indices: [B, k]
        _, indices = torch.topk(similarity_matrix, k=k, dim=1)
        
        # 检查 Ground Truth (对角线上的索引) 是否在 Top-K 内
        # labels: [0, 1, 2, ..., B-1]
        labels = torch.arange(B, device=q.device).unsqueeze(1) # [B, 1]
        
        # (indices == labels) 会得到一个 [B, k] 的布尔矩阵
        is_hit = torch.any(indices == labels, dim=1) # [B]
        
        hits += is_hit.sum().item()
        total_queries += B
        
    return hits / total_queries if total_queries > 0 else 0.0

# =========================================================
# Training Loop
# =========================================================
def train():
    config = PinRecConfig()
    
    # --- 训练配置 (已完善) ---
    SAVE_STEPS = 500      # 每 500 步保存一次 Checkpoint
    LOG_STEPS = 100       # 每 100 步打印一次 Loss
    EVAL_STEPS = 1000     # 每 1000 步进行一次 In-Batch 评估
    MAX_KEEP = 3          # 最多保留 3 个 Checkpoint
    
    OUTPUT_DIR = "/workspace/data/pinrec_ckpt_v2"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 1. Init Models
    item_tower = ItemTower(config).to(device)
    user_tower = UserTower(config).to(device)
    
    # 2. Dataset
    data_path = "/workspace/data/processed_pinrec_v2/train_ultimate.jsonl"
    dataset = UltimateDataset(data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=8, 
        pin_memory=True
    )
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': 1e-5},
        {'params': item_tower.hash_tables.parameters(), 'lr': 1e-4},
        {'params': item_tower.content_proj.parameters(), 'lr': 1e-4}
    ])
    
    total_steps = len(dataloader) * 3
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)
    
    print("🚀 Starting Ultimate Training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global_step = 0
    
    for epoch in range(3):
        user_tower.train()
        item_tower.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        step_loss = 0
        
        for batch in pbar:
            global_step += 1
            
            # Prepare Data
            h_ids = batch['h_ids'].to(device)
            h_acts = batch['h_acts'].to(device)
            h_deltas = batch['h_deltas'].to(device)
            h_mask = batch['h_mask'].to(device)
            
            t_ids = batch['t_ids'].to(device)
            t_acts = batch['t_acts'].to(device)
            t_deltas = batch['t_deltas'].to(device)
            
            # --- Forward Pass ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 1. Item Tower Encode History
                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                # 2. User Tower Forward
                user_preds = user_tower(
                    h_vecs, h_acts, h_deltas, h_mask,
                    t_acts, t_deltas
                ) # [B, 2, Dim] normalized
                
                # 3. Item Tower Encode Targets
                flat_t_ids = t_ids.view(-1)
                t_vecs = item_tower(flat_t_ids)
                t_vecs = F.normalize(t_vecs, p=2, dim=-1)
                target_vecs = t_vecs.view(t_ids.shape[0], t_ids.shape[1], -1) # [B, 2, Dim] normalized
                
                # 4. Loss Calculation (In-Batch Softmax)
                loss = 0
                B = user_preds.shape[0]
                
                for i in range(2): 
                    q = user_preds[:, i, :] # Query [B, Dim]
                    pos = target_vecs[:, i, :] # Target [B, Dim]
                    
                    logits = torch.matmul(q, pos.T) / 0.07
                    labels = torch.arange(B, device=device)
                    loss += F.cross_entropy(logits, labels)
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step_loss += loss.item()
            
            # --- 监控与保存逻辑 ---
            if global_step % LOG_STEPS == 0:
                avg_loss = step_loss / LOG_STEPS
                pbar.set_postfix({'loss': avg_loss, 'step': global_step, 'lr': optimizer.param_groups[0]['lr']})
                step_loss = 0
                
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)
                user_tower.train() # 恢复训练状态
            
            if global_step % EVAL_STEPS == 0:
                user_tower.eval()
                item_tower.eval()
                
                # Quick In-Batch Eval
                with torch.no_grad():
                    recall_score = in_batch_recall_at_k(user_preds, target_vecs, k=10)
                
                print(f"\n✨ EVAL Step {global_step}: In-Batch Recall@10: {recall_score:.4f}")
                
                user_tower.train()
                item_tower.train()
            
        # End of Epoch Save
        print(f"\nEpoch {epoch} Finished.")
        save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)

if __name__ == "__main__":
    train()