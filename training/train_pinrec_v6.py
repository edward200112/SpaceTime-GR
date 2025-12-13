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
from collections import defaultdict

# --- 导入模型 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# 1. Dataset & Collate
# =========================================================
class UltimateDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"Loading data from {data_path}...")
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
        "t_ids": torch.stack([b['t_ids'] for b in batch]),
        "t_acts": torch.stack([b['t_acts'] for b in batch]),
        "t_deltas": torch.stack([b['t_deltas'] for b in batch])
    }

# =========================================================
# 2. Checkpoint Helpers
# =========================================================
def manage_checkpoints(output_dir, limit):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    if len(checkpoints) > limit:
        for ckpt in checkpoints[:-limit]:
            try: shutil.rmtree(ckpt, ignore_errors=True)
            except: pass

def save_checkpoint(user_tower, item_tower, optimizer, step, output_dir, max_keep):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    if hasattr(user_tower.llm, "save_pretrained"):
        user_tower.llm.save_pretrained(save_path)
    
    custom_state = {k: v for k, v in user_tower.state_dict().items() if "llm" not in k}
    torch.save(custom_state, os.path.join(save_path, "user_tower_heads.bin"))
    torch.save(item_tower.state_dict(), os.path.join(save_path, "item_tower.bin"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    
    print(f"💾 Checkpoint saved: {save_path}")
    manage_checkpoints(output_dir, max_keep)

# =========================================================
# 3. In-Batch Eval
# =========================================================
def in_batch_recall_at_k(user_preds, target_vecs, k=10):
    B = user_preds.shape[0]
    hits = 0
    
    # 必须对 User Vector 归一化，才能与 Target Vector (已归一化) 比较余弦相似度
    user_preds_norm = F.normalize(user_preds, p=2, dim=-1)
    
    for i in range(2):
        q = user_preds_norm[:, i, :] 
        pos = target_vecs[:, i, :] 
        similarity_matrix = torch.matmul(q, pos.T)
        _, indices = torch.topk(similarity_matrix, k=k, dim=1)
        labels = torch.arange(B, device=q.device).unsqueeze(1)
        hits += torch.any(indices == labels, dim=1).sum().item()
    return hits / (B * 2) if B > 0 else 0.0

# =========================================================
# 4. Training Loop (V6 Normalized)
# =========================================================
def train():
    # --- V6 核心配置 ---
    OUTPUT_DIR = "/workspace/data/pinrec_ckpt_v6_norm"
    DATA_PATH = "/workspace/data/processed/train_balanced_pinrec.jsonl"
    
    # [Core Fix 1] 温度系数调高到 0.2，适应余弦空间
    TEMPERATURE = 0.2
    
    # [Core Fix 2] 暂时关闭 LogQ，专注基础拟合
    LOGQ_LAMBDA = 0.0
    
    SAVE_STEPS = 1000      
    LOG_STEPS = 100       
    EVAL_STEPS = 500     
    MAX_KEEP = 3          
    MAX_STEPS = 30000 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting V6 Training (Normalized & Balanced)...")
    print(f"Config: Temp={TEMPERATURE}, LogQ={LOGQ_LAMBDA}, Output={OUTPUT_DIR}")
    
    # 1. Models
    config = PinRecConfig()
    item_tower = ItemTower(config).to(device)
    user_tower = UserTower(config).to(device)
    
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    dataset = UltimateDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # 2. Optimizer
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': 2e-5},
        {'params': item_tower.hash_tables.parameters(), 'lr': 1e-4},
        {'params': item_tower.content_proj.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, MAX_STEPS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global_step = 0
    total_epochs = 3 
    
    for epoch in range(total_epochs):
        user_tower.train()
        item_tower.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        step_loss = 0
        
        for batch in pbar:
            global_step += 1
            
            h_ids = batch['h_ids'].to(device)
            h_acts = batch['h_acts'].to(device)
            h_deltas = batch['h_deltas'].to(device)
            h_mask = batch['h_mask'].to(device)
            t_ids = batch['t_ids'].to(device)
            t_acts = batch['t_acts'].to(device)
            t_deltas = batch['t_deltas'].to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Encode History
                flat_h_vecs = item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                # User Tower Forward (Output Raw Vectors)
                user_preds_raw = user_tower(h_vecs, h_acts, h_deltas, h_mask, t_acts, t_deltas)
                
                # Encode Targets & Normalize (Item Tower always outputs raw, we normalize manually)
                t_vecs = item_tower(t_ids.view(-1))
                t_vecs = F.normalize(t_vecs, p=2, dim=-1)
                target_vecs = t_vecs.view(t_ids.shape[0], t_ids.shape[1], -1)
                
                loss = 0
                B = user_preds_raw.shape[0]
                labels = torch.arange(B, device=device)
                
                for i in range(2):
                    # [Core Fix 3] 强制对 User Vector 进行 L2 归一化
                    # 这样点积 (Dot Product) 就变成了余弦相似度 (Cosine Similarity)
                    # 彻底消除了"模长作弊"的可能性
                    q = F.normalize(user_preds_raw[:, i, :], p=2, dim=-1)
                    pos = target_vecs[:, i, :]
                    
                    # Logits = Cosine / Temp
                    logits = torch.matmul(q, pos.T) / TEMPERATURE
                    loss += F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_tower.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step_loss += loss.item()
            
            # --- Monitor ---
            if global_step % LOG_STEPS == 0:
                pbar.set_postfix({'loss': step_loss / LOG_STEPS, 'step': global_step})
                step_loss = 0
            
            if global_step % EVAL_STEPS == 0:
                user_tower.eval(); item_tower.eval()
                with torch.no_grad():
                    # 传入 raw preds，函数内部会处理归一化逻辑
                    recall = in_batch_recall_at_k(user_preds_raw, target_vecs)
                user_tower.train(); item_tower.train()
                
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)
            
            if global_step >= MAX_STEPS:
                print(f"🏁 Max Steps ({MAX_STEPS}) reached.")
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)
                return

        save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)

if __name__ == "__main__":
    train()