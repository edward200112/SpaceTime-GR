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
from collections import Counter, defaultdict
import math

# --- 导入模型 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# Dataset & Collate (保持 V3 逻辑，兼容 Ultimate 格式)
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
        
        # LogQ 即使关了，保留逻辑也不影响，系数设为0即可
        self.item_log_probs = defaultdict(lambda: -15.0) 

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        t1_id = s['target_1']['id']
        t2_id = s['target_2']['id']
        
        return {
            "h_ids": torch.tensor(s['history_ids'], dtype=torch.long),
            "h_acts": torch.tensor(s['history_acts'], dtype=torch.long),
            "h_deltas": torch.tensor(s['history_deltas'], dtype=torch.float),
            "t_ids": torch.tensor([t1_id, t2_id], dtype=torch.long),
            "t_acts": torch.tensor([s['target_1']['act'], s['target_2']['act']], dtype=torch.long),
            "t_deltas": torch.tensor([s['target_1']['delta'], s['target_2']['delta']], dtype=torch.float),
            "t_log_probs": torch.tensor([0.0, 0.0], dtype=torch.float) # Placeholder
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
        "h_ids": pad_ids,
        "h_acts": pad_acts,
        "h_deltas": pad_deltas,
        "h_mask": mask,
        "t_ids": torch.stack([b['t_ids'] for b in batch]),
        "t_acts": torch.stack([b['t_acts'] for b in batch]),
        "t_deltas": torch.stack([b['t_deltas'] for b in batch]),
        "t_log_probs": torch.zeros((B, 2))
    }

# =========================================================
# Checkpoint Helpers
# =========================================================
def manage_checkpoints(output_dir, limit):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    if len(checkpoints) > limit:
        for ckpt in checkpoints[:-limit]:
            shutil.rmtree(ckpt, ignore_errors=True)

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
# Evaluation Helper
# =========================================================
def in_batch_recall_at_k(user_preds, target_vecs, k=10):
    B = user_preds.shape[0]
    hits = 0
    # Check both Immediate and Future targets
    for i in range(2):
        q = user_preds[:, i, :] 
        pos = target_vecs[:, i, :] 
        similarity_matrix = torch.matmul(q, pos.T)
        _, indices = torch.topk(similarity_matrix, k=k, dim=1)
        labels = torch.arange(B, device=q.device).unsqueeze(1)
        hits += torch.any(indices == labels, dim=1).sum().item()
    return hits / (B * 2) if B > 0 else 0.0

# =========================================================
# Training Loop (V4 Stable)
# =========================================================
def train():
    config = PinRecConfig()
    
    # --- V4 稳健配置 ---
    OUTPUT_DIR = "/workspace/data/pinrec_ckpt_v5_mild_logq"
    DATA_PATH = "/workspace/data/processed/train_balanced_pinrec.jsonl"
    
    TEMPERATURE = 0.1  # [稳健] 回调到 0.1，防止 Logits 爆炸
    LOGQ_LAMBDA = 0.1  # [稳健] 暂时关闭 LogQ，纯靠数据平衡
    
    SAVE_STEPS = 1000      
    LOG_STEPS = 100       
    EVAL_STEPS = 500     
    MAX_KEEP = 3          
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting V4 Training (Stable Mode)...")
    print(f"Data: {DATA_PATH}")
    print(f"Config: Temp={TEMPERATURE}, LogQ={LOGQ_LAMBDA}")
    
    # 1. Models
    item_tower = ItemTower(config).to(device)
    user_tower = UserTower(config).to(device)
    
    # 2. Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    dataset = UltimateDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': 2e-5}, # 稍微加大一点点 LR
        {'params': item_tower.hash_tables.parameters(), 'lr': 1e-4},
        {'params': item_tower.content_proj.parameters(), 'lr': 1e-4}
    ])
    
    total_steps = len(dataloader) * 3
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step = 0
    
    for epoch in range(3):
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
                # Item Tower
                flat_h_vecs = item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                # User Tower
                user_preds = user_tower(h_vecs, h_acts, h_deltas, h_mask, t_acts, t_deltas)
                
                # Targets
                t_vecs = item_tower(t_ids.view(-1))
                t_vecs = F.normalize(t_vecs, p=2, dim=-1)
                target_vecs = t_vecs.view(t_ids.shape[0], t_ids.shape[1], -1)
                
                # Loss
                loss = 0
                B = user_preds.shape[0]
                labels = torch.arange(B, device=device)
                
                for i in range(2):
                    q = user_preds[:, i, :]
                    pos = target_vecs[:, i, :]
                    logits = torch.matmul(q, pos.T) / TEMPERATURE
                    loss += F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_tower.parameters(), 1.0) # [稳健] 梯度裁剪
            optimizer.step()
            scheduler.step()
            
            step_loss += loss.item()
            
            if global_step % LOG_STEPS == 0:
                pbar.set_postfix({'loss': step_loss / LOG_STEPS, 'step': global_step})
                step_loss = 0
            
            if global_step % EVAL_STEPS == 0:
                user_tower.eval(); item_tower.eval()
                with torch.no_grad():
                    recall = in_batch_recall_at_k(user_preds, target_vecs)
                print(f"\n[Eval] In-Batch Recall@10: {recall:.4f}") # 避免刷屏
                user_tower.train(); item_tower.train()
                
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)
        
        save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)

if __name__ == "__main__":
    train()