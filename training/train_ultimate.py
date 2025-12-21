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

# =========================================================
# Path Setup
# =========================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.pinrec_ultimate import PinRecConfig, ItemTower, UserTower
except ImportError as e:
    print("❌ 导入失败！请检查 models/pinrec_ultimate.py 是否存在。")
    print(f"错误详情: {e}")
    sys.exit(1)

# =========================================================
# Dataset & Collate
# =========================================================
class UltimateDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"Loading data from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到训练数据: {data_path}")
            
        with open(data_path, 'r') as f:
            for line in f:
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
        l = len(h)
        padded_h[i, :l] = h
        
    return {
        "history": padded_h,
        "target_ids": torch.stack([b['target_ids'] for b in batch]),
        "outcomes": torch.stack([b['outcomes'] for b in batch]),
        "deltas": torch.stack([b['deltas'] for b in batch])
    }

# =========================================================
# Checkpoint Helpers
# =========================================================
def manage_checkpoints(output_dir, limit=3):
    """只保留最近的 limit 个 checkpoint"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    # 按 step 数字排序
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    
    if len(checkpoints) > limit:
        to_delete = checkpoints[:-limit]
        for ckpt in to_delete:
            print(f"🗑️ Deleting old checkpoint: {ckpt}")
            try:
                shutil.rmtree(ckpt)
            except Exception as e:
                print(f"Error deleting {ckpt}: {e}")

def save_checkpoint(user_tower, item_tower, optimizer, step, output_dir):
    save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 保存 UserTower (LoRA权重 + Custom Heads)
    # 推荐使用 save_pretrained 保存 LoRA 部分
    if hasattr(user_tower.llm, "save_pretrained"):
        user_tower.llm.save_pretrained(save_path)
        
    # 保存非 LLM 部分 (Projector, Embeddings 等)
    custom_state = {
        k: v for k, v in user_tower.state_dict().items() 
        if "llm" not in k # 排除 LLM 权重，避免重复保存巨大的文件
    }
    torch.save(custom_state, os.path.join(save_path, "user_tower_heads.bin"))
    
    # 2. 保存 ItemTower
    torch.save(item_tower.state_dict(), os.path.join(save_path, "item_tower.bin"))
    
    # 3. 保存 Optimizer (可选，用于断点续训)
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    
    print(f"💾 Checkpoint saved to {save_path}")
    
    # 4. 删除旧的
    manage_checkpoints(output_dir, limit=3)

# =========================================================
# Training Loop
# =========================================================
def train():
    config = PinRecConfig()
    
    # --- 训练设置 ---
    SAVE_STEPS = 500      # 每 500 步保存一次
    LOG_STEPS = 100       # 每 100 步打印一次
    MAX_KEEP = 3          # 最多保留 3 个模型
    OUTPUT_DIR = "/workspace/data/pinrec_ckpt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 1. Init Models
    print("Initializing Item Tower...")
    item_tower = ItemTower(config).to(device)
    
    print("Initializing User Tower...")
    user_tower = UserTower(config).to(device)
    
    # 2. Dataset
    data_path = "/workspace/data/processed_pinrec/train_ultimate.jsonl"
    dataset = UltimateDataset(data_path)
    
    dataloader = DataLoader(
    dataset, 
    batch_size=128,  # <--- 改这里
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=8,   # <--- 显存大了，数据加载也要跟上，建议增加 workers
    pin_memory=True  # <--- 加上这个，加速 CPU 到 GPU 传输
)
    
    # 3. Optimizer
    # 你的设置是正确的：ItemTower 的 hash_tables 和 proj 会被训练
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': 2e-5}, 
        {'params': item_tower.hash_tables.parameters(), 'lr': 1e-4}, 
        {'params': item_tower.content_proj.parameters(), 'lr': 1e-4}
    ])
    
    total_steps = len(dataloader) * 3 # 3 epochs
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
            
            # Move to GPU
            history = batch['history'].to(device)       
            targets = batch['target_ids'].to(device)    
            outcomes = batch['outcomes'].to(device)     
            deltas = batch['deltas'].to(device)         
            B = history.shape[0]
            
            # --- 关键修改：使用 Autocast 解决 dtype 错误 ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                # 1. Encode History (Item Tower)
                flat_hist = history.view(-1)
                hist_vecs_flat = item_tower(flat_hist) 
                hist_vecs = hist_vecs_flat.view(B, history.shape[1], -1) 
                
                # 2. User Tower Forward
                # user_preds: [B, 2, Dim]
                user_preds = user_tower(hist_vecs, outcomes, deltas)
                
                # 3. Encode Targets
                flat_targets = targets.view(-1)
                target_vecs = item_tower(flat_targets)
                target_vecs = F.normalize(target_vecs, p=2, dim=-1)
                
                # 4. Loss
                loss = 0
                for i in range(2): 
                    query_vec = user_preds[:, i, :] # [B, Dim]
                    pos_target_vec = target_vecs.view(B, 2, -1)[:, i, :] 
                    
                    # Logits scaling
                    logits = torch.matmul(query_vec, pos_target_vec.T) / 0.07
                    labels = torch.arange(B, device=device)
                    loss += F.cross_entropy(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step_loss += loss.item()
            
            # --- 监控与保存逻辑 ---
            if global_step % LOG_STEPS == 0:
                avg_loss = step_loss / LOG_STEPS
                pbar.set_postfix({'loss': avg_loss, 'step': global_step})
                step_loss = 0 # reset
                
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR)
                # 保持训练状态
                user_tower.train()

        # End of Epoch Save
        print(f"Epoch {epoch} Finished.")
        save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR)

if __name__ == "__main__":
    train()