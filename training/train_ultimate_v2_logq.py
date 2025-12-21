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

# --- 导入模型 (假设路径已设置) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    # 兼容性 Fallback，如果你是在当前目录运行
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# Dataset & Collate (V3: 增加频率统计)
# =========================================================
class UltimateDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"Loading data from {data_path}...")
        
        # 1. 加载数据
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples.")
        
        # 2. [关键] 统计物品频率 (用于 LogQ 修正)
        print("Calculating Item Frequencies for LogQ correction...")
        # 我们只统计作为 Target 出现的频率，因为那是 Loss 计算的地方
        all_targets = []
        for s in self.samples:
            all_targets.append(s['target_1']['id'])
            all_targets.append(s['target_2']['id'])
            
        counter = Counter(all_targets)
        total_count = len(all_targets)
        
        # 预计算 Log Probability: log(count / total)
        # 使用默认值处理没见过的 ID (设为极小概率)
        self.item_log_probs = defaultdict(lambda: -15.0) # log(1e-7) approx -16
        
        for iid, count in counter.items():
            prob = count / total_count
            self.item_log_probs[iid] = math.log(prob + 1e-9) # 防止 log(0)
            
        print(f"Frequency stats calculated. Max LogP: {max(self.item_log_probs.values()):.4f}, Min LogP: {min(self.item_log_probs.values()):.4f}")
                
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
            
            # [关键] 返回 Target 的 Log Probability
            "t_log_probs": torch.tensor([self.item_log_probs[t1_id], self.item_log_probs[t2_id]], dtype=torch.float)
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
        "t_deltas": torch.stack([b['t_deltas'] for b in batch]),
        
        # [关键] 堆叠 Log Probs
        "t_log_probs": torch.stack([b['t_log_probs'] for b in batch]) # [B, 2]
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
    
    # 1. 保存 UserTower
    if hasattr(user_tower.llm, "save_pretrained"):
        user_tower.llm.save_pretrained(save_path)
        
    # 保存非 LLM 部分
    custom_state = {k: v for k, v in user_tower.state_dict().items() if "llm" not in k}
    torch.save(custom_state, os.path.join(save_path, "user_tower_heads.bin"))
    
    # 2. 保存 ItemTower
    torch.save(item_tower.state_dict(), os.path.join(save_path, "item_tower.bin"))
    
    # 3. 保存 Optimizer
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
    
    print(f"💾 Checkpoint saved to {save_path}")
    manage_checkpoints(output_dir, max_keep)

# =========================================================
# Evaluation Helper
# =========================================================
def in_batch_recall_at_k(user_preds, target_vecs, k=10):
    B = user_preds.shape[0]
    NQ = user_preds.shape[1]
    total_queries = 0
    hits = 0
    
    for i in range(NQ):
        q = user_preds[:, i, :] 
        pos = target_vecs[:, i, :] 
        similarity_matrix = torch.matmul(q, pos.T)
        _, indices = torch.topk(similarity_matrix, k=k, dim=1)
        labels = torch.arange(B, device=q.device).unsqueeze(1)
        is_hit = torch.any(indices == labels, dim=1)
        hits += is_hit.sum().item()
        total_queries += B
        
    return hits / total_queries if total_queries > 0 else 0.0

# =========================================================
# Training Loop (Applied LogQ)
# =========================================================
def train():
    config = PinRecConfig()
    
    # --- 训练配置 ---
    SAVE_STEPS = 500      
    LOG_STEPS = 100       
    EVAL_STEPS = 1000     
    MAX_KEEP = 3          
    
    # [关键配置] 建议从头开始训练，不要加载旧的 checkpoint
    OUTPUT_DIR = "/workspace/data/pinrec_ckpt_v3_logq" 
    
    # [关键配置] 温度系数：建议调小以增加对比度
    TEMPERATURE = 0.05 
    
    # [关键配置] LogQ 修正强度 (lambda)
    # 1.0 是理论值，如果觉得惩罚太重可以设为 0.5
    LOGQ_LAMBDA = 1.0 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    print(f"Config: Temp={TEMPERATURE}, LogQ Lambda={LOGQ_LAMBDA}")
    
    # 1. Init Models
    item_tower = ItemTower(config).to(device)
    user_tower = UserTower(config).to(device)
    
    # 2. Dataset
    # [建议] 这里使用你刚刚生成的平衡数据集
    data_path = "/workspace/data/processed/train_balanced_pinrec.jsonl"
    if not os.path.exists(data_path):
        print(f"Balanced data not found at {data_path}, falling back to original...")
        data_path = "/workspace/data/processed_pinrec_v2/train_ultimate.jsonl"
        
    dataset = UltimateDataset(data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, # 建议: 如果显存允许，Batch Size 越大越好 (负样本越多)
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
    
    print(f"🚀 Starting Ultimate Training with LogQ Correction...")
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
            t_log_probs = batch['t_log_probs'].to(device) # [B, 2]
            
            # --- Forward Pass ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 1. Encode History
                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(h_ids.shape[0], h_ids.shape[1], -1)
                
                # 2. User Tower
                user_preds = user_tower(
                    h_vecs, h_acts, h_deltas, h_mask,
                    t_acts, t_deltas
                ) # [B, 2, Dim] normalized
                
                # 3. Encode Targets
                flat_t_ids = t_ids.view(-1)
                t_vecs = item_tower(flat_t_ids)
                t_vecs = F.normalize(t_vecs, p=2, dim=-1)
                target_vecs = t_vecs.view(t_ids.shape[0], t_ids.shape[1], -1) 
                
                # 4. Loss Calculation (In-Batch Softmax with LogQ)
                loss = 0
                B = user_preds.shape[0]
                
                for i in range(2): 
                    q = user_preds[:, i, :] # [B, Dim]
                    pos = target_vecs[:, i, :] # [B, Dim]
                    
                    # 获取当前 Batch 所有 Target 的 Log Prob
                    # 形状 [B]
                    batch_log_probs = t_log_probs[:, i]
                    
                    # 基础 Logits: [B, B]
                    logits = torch.matmul(q, pos.T) / TEMPERATURE
                    
                    # [关键步骤] LogQ Correction
                    # 我们要减去列向量对应的 log_prob
                    # logits[row, col] 代表 User(row) 对 Item(col) 的打分
                    # 我们要修正 Item(col) 的流行度，所以广播到每一行
                    correction = batch_log_probs.unsqueeze(0) # [1, B]
                    
                    logits = logits - (LOGQ_LAMBDA * correction)
                    
                    labels = torch.arange(B, device=device)
                    loss += F.cross_entropy(logits, labels)
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step_loss += loss.item()
            
            # --- 监控 ---
            if global_step % LOG_STEPS == 0:
                avg_loss = step_loss / LOG_STEPS
                pbar.set_postfix({'loss': avg_loss, 'step': global_step})
                step_loss = 0
                
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)
                user_tower.train() 
            
            if global_step % EVAL_STEPS == 0:
                user_tower.eval()
                item_tower.eval()
                with torch.no_grad():
                    recall_score = in_batch_recall_at_k(user_preds, target_vecs, k=10)
                print(f"\n✨ EVAL Step {global_step}: In-Batch Recall@10: {recall_score:.4f}")
                user_tower.train()
                item_tower.train()
            
        print(f"\nEpoch {epoch} Finished.")
        save_checkpoint(user_tower, item_tower, optimizer, global_step, OUTPUT_DIR, MAX_KEEP)

if __name__ == "__main__":
    train()