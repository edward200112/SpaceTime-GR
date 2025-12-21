import sys
import os
import glob
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import math

# --- 导入模型 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# ⚙️ 全局配置
# =========================================================
MAX_VOCAB_SIZE = 250000 

CONFIG = {
    "output_dir": "/workspace/data/pinrec_ckpt_sft_final_v3", # 继续使用同一个目录
    "data_path": "/workspace/data/processed/train_balanced_pinrec.jsonl",
    "batch_size": 64,          
    "lr_user": 2e-5,           
    "lr_item": 1e-4,           
    "temperature": 0.2,        
    "max_steps": 50000,        
    "save_steps": 2000,        
    "log_steps": 100,          
    "warmup_steps": 1000,
    "gradient_clipping": 1.0,  
    # [新增] 指定要恢复的 Checkpoint 路径
    "resume_from": "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-4000" 
}

# =========================================================
# 🛠️ 数据集 (Dataset)
# =========================================================
class PinRecSFTDataset(Dataset):
    def __init__(self, data_path, max_vocab_size):
        self.samples = []
        self.max_vocab_size = max_vocab_size
        print(f"📂 Loading data from {data_path}...")
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._safe_id(data)
                    self.samples.append(data)
        print(f"✅ Loaded {len(self.samples)} samples.")

    def _safe_id(self, data):
        """Hash Simulation"""
        # History
        new_hist = []
        for hid in data['history_ids']:
            if hid >= self.max_vocab_size:
                new_hist.append(hid % self.max_vocab_size)
            else:
                new_hist.append(hid)
        data['history_ids'] = new_hist
        
        # Targets
        if data['target_1']['id'] >= self.max_vocab_size:
            data['target_1']['id'] = data['target_1']['id'] % self.max_vocab_size
        if data['target_2']['id'] >= self.max_vocab_size:
            data['target_2']['id'] = data['target_2']['id'] % self.max_vocab_size

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_ids": torch.tensor(s['history_ids'], dtype=torch.long),
            "h_acts": torch.tensor(s['history_acts'], dtype=torch.long),
            "h_deltas": torch.tensor(s['history_deltas'], dtype=torch.float),
            "t_ids": torch.tensor([s['target_1']['id'], s['target_2']['id']], dtype=torch.long),
            "t_acts": torch.tensor([s['target_1']['act'], s['target_2']['act']], dtype=torch.long),
            "t_deltas": torch.tensor([s['target_1']['delta'], s['target_2']['delta']], dtype=torch.float),
        }

def collate_fn(batch):
    h_ids = [b['h_ids'] for b in batch]
    max_len = max(len(h) for h in h_ids)
    B = len(batch)
    
    pad_ids = torch.zeros((B, max_len), dtype=torch.long)
    pad_acts = torch.zeros((B, max_len), dtype=torch.long)
    pad_deltas = torch.zeros((B, max_len), dtype=torch.float)
    mask = torch.zeros((B, max_len), dtype=torch.long)
    
    for i in range(B):
        l = len(h_ids[i])
        pad_ids[i, :l] = batch[i]['h_ids']
        pad_acts[i, :l] = batch[i]['h_acts']
        pad_deltas[i, :l] = batch[i]['h_deltas']
        mask[i, :l] = 1 
        
    return {
        "h_ids": pad_ids, "h_acts": pad_acts, "h_deltas": pad_deltas, "h_mask": mask,
        "t_ids": torch.stack([b['t_ids'] for b in batch]),     
        "t_acts": torch.stack([b['t_acts'] for b in batch]),   
        "t_deltas": torch.stack([b['t_deltas'] for b in batch])
    }

# =========================================================
# 💾 Checkpoint 管理 (只保留 1 个以节省空间)
# =========================================================
def save_checkpoint(user_tower, item_tower, optimizer, step, output_dir, config_obj):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(user_tower.state_dict(), os.path.join(save_path, "user_tower.bin"))
    torch.save(item_tower.state_dict(), os.path.join(save_path, "item_tower.bin"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    
    try:
        config_dict = config_obj.__dict__
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=4)
    except Exception as e:
        print(f"⚠️ Warning: Could not save config.json: {e}")
    
    print(f"\n💾 Checkpoint saved: {save_path}")
    
    # [关键修改] 只保留最近 1 个 Checkpoint，防止磁盘爆满
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
    while len(checkpoints) > 1:
        oldest = checkpoints.pop(0)
        print(f"🧹 Cleaning up old checkpoint: {oldest}")
        shutil.rmtree(oldest, ignore_errors=True)

# =========================================================
# 🚀 训练主循环 (Resume Mode)
# =========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting PinRec SFT Training (Resume Mode) on {device}...")
    
    # 1. Config
    config = PinRecConfig()
    if hasattr(config, 'item_vocab_size'): config.item_vocab_size = MAX_VOCAB_SIZE
    if hasattr(config, 'vocab_size'): config.vocab_size = MAX_VOCAB_SIZE
    if hasattr(config, 'item_size'): config.item_size = MAX_VOCAB_SIZE
    
    # 2. Models
    item_tower = ItemTower(config).to(device)
    user_tower = UserTower(config).to(device)
    
    # 3. Optimizer (先定义，后加载状态)
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': CONFIG["lr_user"]},
        {'params': item_tower.parameters(), 'lr': CONFIG["lr_item"]}
    ])

    # === [关键] 断点续训加载逻辑 ===
    resume_path = CONFIG.get("resume_from", "")
    start_step = 0
    
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Found checkpoint at {resume_path}, loading...")
        
        # 加载权重
        item_tower.load_state_dict(torch.load(os.path.join(resume_path, "item_tower.bin"), map_location=device))
        user_tower.load_state_dict(torch.load(os.path.join(resume_path, "user_tower.bin"), map_location=device))
        
        # 加载优化器
        opt_path = os.path.join(resume_path, "optimizer.bin")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print("✅ Optimizer state loaded.")
        else:
            print("⚠️ Optimizer state not found, starting optimizer from scratch.")
            
        # 解析 Step
        try:
            start_step = int(resume_path.split("-")[-1])
        except:
            start_step = 0
        print(f"⏩ Resuming from global_step = {start_step}")
    else:
        print("🆕 No checkpoint found or specified, starting from scratch.")

    # 4. Scheduler (设置 last_epoch 以恢复 LR 曲线)
    # last_epoch 默认为 -1，如果要恢复，应该是 start_step - 1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        CONFIG["warmup_steps"], 
        CONFIG["max_steps"],
        last_epoch=start_step - 1 
    )
    
    # 5. Data
    dataset = PinRecSFTDataset(CONFIG["data_path"], MAX_VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # 6. Training Loop
    user_tower.train()
    item_tower.train()
    
    global_step = start_step
    running_loss = 0.0
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    while global_step < CONFIG["max_steps"]:
        pbar = tqdm(dataloader, desc=f"Training (Start: {start_step})")
        
        for batch in pbar:
            # 如果是续训，前几轮可能因为 shuffle 不同而重复数据，这在 SFT 中影响不大
            global_step += 1
            
            # --- Device ---
            h_ids = batch['h_ids'].to(device)
            h_acts = batch['h_acts'].to(device)
            h_deltas = batch['h_deltas'].to(device)
            h_mask = batch['h_mask'].to(device)
            t_ids = batch['t_ids'].to(device)
            t_acts = batch['t_acts'].to(device)
            t_deltas = batch['t_deltas'].to(device)
            
            # --- Forward ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Target
                flat_t_ids = t_ids.view(-1) 
                flat_t_vecs = item_tower(flat_t_ids)
                flat_t_vecs = F.normalize(flat_t_vecs, p=2, dim=-1)
                target_vecs = flat_t_vecs.view(t_ids.shape[0], 2, -1)
                
                # History
                flat_h_vecs = item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                # User Tower
                user_preds = user_tower(h_vecs, h_acts, h_deltas, h_mask, t_acts, t_deltas)
                user_preds = F.normalize(user_preds, p=2, dim=-1)
                
                # Loss
                loss = 0
                B = user_preds.shape[0]
                labels = torch.arange(B, device=device)
                
                for i in range(2):
                    q = user_preds[:, i, :]
                    pos = target_vecs[:, i, :]
                    logits = torch.matmul(q, pos.T) / CONFIG["temperature"]
                    loss += F.cross_entropy(logits, labels)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_tower.parameters(), CONFIG["gradient_clipping"])
            optimizer.step()
            scheduler.step()
            
            # --- Log ---
            running_loss += loss.item()
            
            if global_step % CONFIG["log_steps"] == 0:
                avg_loss = running_loss / CONFIG["log_steps"]
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.2e}", 'step': global_step})
                running_loss = 0.0
            
            # --- Save ---
            if global_step % CONFIG["save_steps"] == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, CONFIG["output_dir"], config)
            
            # --- Exit ---
            if global_step >= CONFIG["max_steps"]:
                print("🏁 Training finished!")
                save_checkpoint(user_tower, item_tower, optimizer, global_step, CONFIG["output_dir"], config)
                return

if __name__ == "__main__":
    train()