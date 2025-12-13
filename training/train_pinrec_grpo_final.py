import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import glob
import shutil
import gc # [新增] 垃圾回收

# --- 导入模型 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# ⚙️ 32GB 显存·防 OOM 优化版配置
# =========================================================
MAX_VOCAB_SIZE = 250000 
SFT_CHECKPOINT = "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-48000"

CONFIG = {
    "output_dir": "/workspace/data/pinrec_ckpt_grpo_aggressive",
    "data_path": "/workspace/data/processed/train_balanced_pinrec.jsonl",
    
    # === [⚡️ 显存优化核心] ===
    # 之前: 64 * 16 = 1024 (OOM)
    # 现在: 16 * 16 = 256 (Safe for 32GB)
    "batch_size": 16,          # 降低 Batch Size 防止 OOM
    "group_size": 16,          # 保持高 Group Size 以确保 RL 对比效果
    
    "lr": 5e-5,                # [激进] 保持高学习率
    "beta": 0.0001,            # [松绑] 保持低 KL 惩罚
    # ========================
    
    "max_steps": 10000,        # 延长时间
    "save_steps": 500,         
    "log_steps": 10,
    "warmup_steps": 200,       
    "gradient_clipping": 1.0,
    "max_keep_ckpt": 3         
}

# =========================================================
# 🛠️ 数据集
# =========================================================
class PinRecGRPODataset(Dataset):
    def __init__(self, data_path, max_vocab_size):
        self.samples = []
        self.max_vocab_size = max_vocab_size
        print(f"📂 Loading GRPO data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._safe_id(data)
                    self.samples.append(data)
        print(f"✅ Loaded {len(self.samples)} samples.")

    def _safe_id(self, data):
        new_hist = [h % self.max_vocab_size if h >= self.max_vocab_size else h for h in data['history_ids']]
        data['history_ids'] = new_hist
        if data['target_1']['id'] >= self.max_vocab_size:
            data['target_1']['id'] = data['target_1']['id'] % self.max_vocab_size

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h_ids": torch.tensor(s['history_ids'], dtype=torch.long),
            "h_acts": torch.tensor(s['history_acts'], dtype=torch.long),
            "h_deltas": torch.tensor(s['history_deltas'], dtype=torch.float),
            "t_id": torch.tensor(s['target_1']['id'], dtype=torch.long),
            "t_act": torch.tensor(s['target_1']['act'], dtype=torch.long),
            "t_delta": torch.tensor(s['target_1']['delta'], dtype=torch.float),
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
        "t_id": torch.stack([b['t_id'] for b in batch]),
        "t_act": torch.stack([b['t_act'] for b in batch]),
        "t_delta": torch.stack([b['t_delta'] for b in batch])
    }

# =========================================================
# 🚀 GRPO 训练主循环
# =========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting PinRec Aggressive GRPO (OOM Fix) on {device}...")
    print(f"🔥 Config: Batch={CONFIG['batch_size']}, Group={CONFIG['group_size']}, LR={CONFIG['lr']}")
    
    # 1. Config
    config = PinRecConfig()
    if hasattr(config, 'item_vocab_size'): config.item_vocab_size = MAX_VOCAB_SIZE
    if hasattr(config, 'vocab_size'): config.vocab_size = MAX_VOCAB_SIZE
    
    # 2. Models
    print("🔹 Loading Policy Model...")
    policy_user = UserTower(config).to(device)
    
    # [新增] 尝试从之前的 GRPO Checkpoint 恢复 (如果存在)，否则从 SFT 开始
    LATEST_GRPO = "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-3000"
    if os.path.exists(os.path.join(LATEST_GRPO, "user_tower.bin")):
        print(f"♻️ Resuming from GRPO Checkpoint: {LATEST_GRPO}")
        policy_user.load_state_dict(torch.load(os.path.join(LATEST_GRPO, "user_tower.bin"), map_location=device))
    else:
        print(f"🆕 Starting fresh from SFT: {SFT_CHECKPOINT}")
        policy_user.load_state_dict(torch.load(os.path.join(SFT_CHECKPOINT, "user_tower.bin"), map_location=device))
        
    policy_user.train()
    
    # [新增] 尝试开启 Gradient Checkpointing 节省显存
    try:
        policy_user.gradient_checkpointing_enable()
        print("✅ Gradient Checkpointing enabled.")
    except:
        print("⚠️ Gradient Checkpointing not supported or failed.")

    print("🔹 Creating Reference Model...")
    ref_user = copy.deepcopy(policy_user)
    ref_user.eval()
    for param in ref_user.parameters(): param.requires_grad = False
    
    print("🔹 Loading Item Tower...")
    item_tower = ItemTower(config).to(device)
    item_tower.load_state_dict(torch.load(os.path.join(SFT_CHECKPOINT, "item_tower.bin"), map_location=device))
    item_tower.eval()
    for param in item_tower.parameters(): param.requires_grad = False
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(policy_user.parameters(), lr=CONFIG["lr"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, CONFIG["warmup_steps"], CONFIG["max_steps"])
    
    # 4. Data
    dataset = PinRecGRPODataset(CONFIG["data_path"], MAX_VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                            collate_fn=collate_fn, drop_last=True, num_workers=4)
    
    global_step = 0
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    print("🔥 GRPO Loop Started!")
    torch.cuda.empty_cache() # 初始清理
    
    while global_step < CONFIG["max_steps"]:
        pbar = tqdm(dataloader, desc="GRPO Training")
        
        for batch in pbar:
            global_step += 1
            
            # --- 1. 数据准备 ---
            G = CONFIG["group_size"]
            B = batch['t_id'].shape[0]
            
            h_ids = batch['h_ids'].repeat_interleave(G, dim=0).to(device)
            h_acts = batch['h_acts'].repeat_interleave(G, dim=0).to(device)
            h_deltas = batch['h_deltas'].repeat_interleave(G, dim=0).to(device)
            h_mask = batch['h_mask'].repeat_interleave(G, dim=0).to(device)
            
            t_act = batch['t_act'].repeat_interleave(G, dim=0).to(device).unsqueeze(1)
            t_act_input = torch.cat([t_act, t_act], dim=1) 
            
            t_delta = batch['t_delta'].repeat_interleave(G, dim=0).to(device).unsqueeze(1)
            t_delta_input = torch.cat([t_delta, t_delta], dim=1)
            
            t_id = batch['t_id'].repeat_interleave(G, dim=0).to(device)
            
            # --- 2. 前向传播 ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Item Features
                flat_h_vecs = item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(B*G, -1, 1024) 
                
                target_vecs = item_tower(t_id)
                target_vecs = F.normalize(target_vecs, p=2, dim=-1)
                
                # Policy
                policy_out = policy_user(h_vecs, h_acts, h_deltas, h_mask, t_act_input, t_delta_input)
                policy_vecs = F.normalize(policy_out[:, 0, :], p=2, dim=-1)
                
                # Ref
                with torch.no_grad():
                    ref_out = ref_user(h_vecs, h_acts, h_deltas, h_mask, t_act_input, t_delta_input)
                    ref_vecs = F.normalize(ref_out[:, 0, :], p=2, dim=-1)
            
            # --- 3. RL Loss ---
            rewards = torch.sum(policy_vecs * target_vecs, dim=-1) 
            
            rewards_grouped = rewards.view(B, G)
            mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
            std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            
            advantages = (rewards_grouped - mean_rewards) / std_rewards
            advantages = advantages.view(-1)
            
            kl_penalty = 1.0 - torch.sum(policy_vecs * ref_vecs, dim=-1)
            
            pg_loss = - (advantages * rewards) 
            kl_loss = CONFIG["beta"] * kl_penalty
            
            loss = (pg_loss + kl_loss).mean()
            
            # --- 4. Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_user.parameters(), CONFIG["gradient_clipping"])
            optimizer.step()
            scheduler.step()
            
            # --- 5. Cleanup ---
            # [新增] 每次迭代后清理计算图缓存，防止显存碎片化
            if global_step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            # --- Log ---
            if global_step % CONFIG["log_steps"] == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'rew': f"{rewards.mean().item():.4f}",
                    'kl': f"{kl_penalty.mean().item():.4f}" 
                })
            
            # --- Save ---
            if global_step % CONFIG["save_steps"] == 0:
                save_path = os.path.join(CONFIG["output_dir"], f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(policy_user.state_dict(), os.path.join(save_path, "user_tower.bin"))
                print(f"\n💾 GRPO Checkpoint saved: {save_path}")
                
                checkpoints = sorted(glob.glob(os.path.join(CONFIG["output_dir"], "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
                while len(checkpoints) > CONFIG["max_keep_ckpt"]:
                    oldest = checkpoints.pop(0)
                    print(f"🧹 Removed old checkpoint: {oldest}")
                    shutil.rmtree(oldest, ignore_errors=True)
            
            if global_step >= CONFIG["max_steps"]:
                print("🏁 GRPO Training Finished!")
                return

if __name__ == "__main__":
    train()