import os
import argparse
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from SASRec import SASRec # 确保 SASRec.py 在同级目录

def get_args():
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--dataset_path', type=str, default='./SASRec_Data/sasrec_dataset.pkl')
    # ⚠️ 确保这里指向你刚训练完生成的最新模型文件
    parser.add_argument('--model_path', type=str, default='./SASRec_Data/sasrec_full_latest.pth')
    parser.add_argument('--output_dir', type=str, default='./SASRec_Data')
    
    # 显存安全配置：推理 Batch Size 设小一点
    parser.add_argument('--batch_size', type=int, default=512, help="512 是绝对安全的")
    
    # 模型参数 (必须与训练时一致)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k', type=int, default=50)
    
    return parser.parse_args()

class SASRecDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_seq = self.data[idx]['sequence']
        # 推理模式：输入完整序列
        input_ids = user_seq[-self.max_len:] 
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        # 这里 user_id 需要转字符串，防止 DataLoader 报错
        return torch.LongTensor(input_ids), str(self.data[idx]['user_id'])

def main():
    args = get_args()
    print(f"🚀 Starting Export with Batch Size {args.batch_size}...")
    
    # 1. 加载数据
    print(f"📥 Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'rb') as f:
        pkg = pickle.load(f)
    
    data = pkg['data']
    n_items = pkg['n_items']
    id2item = pkg['id2item']
    
    print(f"   Total Data: {len(data)}")
    print(f"   Items: {n_items}")
    
    # 2. 准备 DataLoader
    export_ds = SASRecDataset(data, args.max_len)
    # num_workers 根据 CPU 核心数调整
    export_dl = DataLoader(export_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # 3. 加载模型
    print("🏗️ Loading Model...")
    model = SASRec(n_items, args).to(args.device)
    
    if os.path.exists(args.model_path):
        print(f"🔄 Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    model.eval()
    
    # 4. 推理循环
    teacher_preds = {}
    print("🔮 Generating Teacher Predictions...")
    
    with torch.no_grad():
        for input_ids, uids in tqdm(export_dl, desc="Inference"):
            input_ids = input_ids.to(args.device)
            
            # predict_full 会生成 [Batch, N_Items] 的大矩阵
            # Batch=512 时，显存占用约 1.9GB，非常安全
            logits = model.predict_full(input_ids) 
            logits[:, 0] = -float('inf') # 屏蔽 Padding
            
            # 获取 Top-K
            _, indices = torch.topk(logits, args.top_k, dim=-1)
            indices = indices.cpu().numpy()
            
            for i, uid in enumerate(uids):
                # 处理 DataLoader 可能返回的 tuple/tensor
                if isinstance(uid, torch.Tensor):
                    uid = str(uid.item())
                elif isinstance(uid, tuple):
                    uid = str(uid[0])
                
                recs = []
                for idx in indices[i]:
                    if idx in id2item:
                        recs.append(id2item[idx])
                teacher_preds[uid] = recs

    # 5. 保存结果
    pred_path = os.path.join(args.output_dir, "teacher_predictions.json")
    print(f"💾 Saving predictions to {pred_path}...")
    with open(pred_path, 'w') as f:
        json.dump(teacher_preds, f)
        
    # 保存配置 (RewardSystem 需要)
    config_path = os.path.join(args.output_dir, "sasrec_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f)
        
    print(f"🎉 Success! Teacher Predictions saved.")

if __name__ == "__main__":
    main()