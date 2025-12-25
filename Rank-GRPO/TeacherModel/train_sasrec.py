import os
import argparse
import pickle
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from SASRec import SASRec

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./SASRec_Data/sasrec_dataset.pkl')
    parser.add_argument('--output_dir', type=str, default='./SASRec_Data')
    
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200, help="本次运行要追加训练多少个epoch")
    
    # 关键参数：断点续训
    parser.add_argument('--resume_path', type=str, default='/workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pth', help='指定 .pth 文件路径以继续训练')
    
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=50)
    
    return parser.parse_args()

class SASRecDataset(Dataset):
    def __init__(self, data, n_items, max_len, train=True):
        self.data = data
        self.n_items = n_items
        self.max_len = max_len
        self.train_mode = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_seq = self.data[idx]['sequence']
        
        if self.train_mode:
            input_ids = user_seq[:-1]
            target_pos = user_seq[1:]
            
            input_ids = input_ids[-(self.max_len-1):]
            target_pos = target_pos[-(self.max_len-1):]
            
            pad_len = self.max_len - len(input_ids)
            input_ids = [0] * pad_len + input_ids
            target_pos = [0] * pad_len + target_pos
            
            target_neg = []
            seq_set = set(user_seq)
            for t in target_pos:
                if t == 0:
                    target_neg.append(0)
                    continue
                neg = np.random.randint(1, self.n_items + 1)
                while neg in seq_set:
                    neg = np.random.randint(1, self.n_items + 1)
                target_neg.append(neg)
            
            return torch.LongTensor(input_ids), torch.LongTensor(target_pos), torch.LongTensor(target_neg)
        
        else:
            input_ids = user_seq[-self.max_len:]
            pad_len = self.max_len - len(input_ids)
            input_ids = [0] * pad_len + input_ids
            return torch.LongTensor(input_ids), str(self.data[idx]['user_id'])

def calculate_loss(criterion, pos_logits, neg_logits, pos_ids):
    indices = torch.where(pos_ids != 0)
    if indices[0].numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    loss = criterion(pos_logits[indices], pos_labels[indices]) + \
           criterion(neg_logits[indices], neg_labels[indices])
    return loss

def main():
    args = get_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"📥 Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'rb') as f:
        pkg = pickle.load(f)
    data = pkg['data']
    n_items = pkg['n_items']
    id2item = pkg['id2item']

    train_ds = SASRecDataset(data, n_items, args.max_len, train=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    print("🏗️ Initializing Model...")
    model = SASRec(n_items, args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # ================= [修复] 强制检查断点加载 =================
    start_epoch = 1
    if args.resume_path:
        if os.path.exists(args.resume_path):
            print(f"🔄 Resuming training from {args.resume_path}...")
            # 加载权重
            state_dict = torch.load(args.resume_path, map_location=args.device)
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully!")
            
            # 尝试解析 Epoch
            try:
                import re
                match = re.search(r'epoch_(\d+)', args.resume_path)
                if match:
                    start_epoch = int(match.group(1)) + 1
                    print(f"⏩ Detected Epoch {start_epoch-1} from filename, resuming from Epoch {start_epoch}")
                else:
                    print("⏩ Could not detect epoch from filename, starting loop from 1 (but with pretrained weights)")
            except:
                pass
        else:
            raise FileNotFoundError(f"❌ Resume path specified but not found: {args.resume_path}")
    else:
        print("⚠️ No resume path specified. Starting from scratch!")
    # ==========================================================

    print("🚀 Starting Training...")
    
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for input_ids, pos_ids, neg_ids in pbar:
            input_ids, pos_ids, neg_ids = input_ids.to(args.device), pos_ids.to(args.device), neg_ids.to(args.device)
            
            optimizer.zero_grad()
            pos_logits, neg_logits = model(input_ids, pos_ids, neg_ids)
            loss = calculate_loss(criterion, pos_logits, neg_logits, pos_ids)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'loss': f"{train_loss / train_steps:.4f}"})
        
        latest_path = os.path.join(args.output_dir, 'sasrec_full_latest.pth')
        torch.save(model.state_dict(), latest_path)
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'sasrec_full_epoch_{epoch}.pth'))

    print("✅ Training Finished!")

    # 导出逻辑保持不变
    print("\n🔮 Generating Teacher Predictions (Top-K)...")
    model.eval()
    export_ds = SASRecDataset(data, n_items, args.max_len, train=False)
    export_dl = DataLoader(export_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=16)
    
    teacher_preds = {}
    with torch.no_grad():
        for input_ids, uids in tqdm(export_dl, desc="Inference"):
            input_ids = input_ids.to(args.device)
            logits = model.predict_full(input_ids) 
            logits[:, 0] = -float('inf')
            _, indices = torch.topk(logits, args.top_k, dim=-1)
            indices = indices.cpu().numpy()
            
            for i, uid in enumerate(uids):
                if isinstance(uid, torch.Tensor):
                    uid = str(uid.item())
                recs = []
                for idx in indices[i]:
                    if idx in id2item:
                        recs.append(id2item[idx])
                teacher_preds[uid] = recs

    pred_path = os.path.join(args.output_dir, "teacher_predictions.json")
    print(f"💾 Saving predictions to {pred_path}...")
    with open(pred_path, 'w') as f:
        json.dump(teacher_preds, f)
    
    config_path = os.path.join(args.output_dir, "sasrec_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f)
        
    print(f"🎉 All Done!")

if __name__ == "__main__":
    main()