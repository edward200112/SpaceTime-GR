import torch
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from SASRec import SASRec # 确保目录下有 SASRec.py

def main():
    # 1. 配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./SASRec_Data/sasrec_dataset.pkl')
    parser.add_argument('--model_path', default='./SASRec_Data/sasrec_full_latest.pth') # 指向你刚训练好的模型
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=2000, help="抽样数量，测2000个够了")
    parser.add_argument('--embed_dim', type=int, default=128) # 确保跟训练参数一致
    parser.add_argument('--max_len', type=int, default=50)

    # 【新增】补齐模型结构所需的参数
    # 如果你训练时改过这些值，这里必须和训练时保持一致！
    parser.add_argument('--dropout', type=float, default=0.0) # 评估脚本设为0即可，反正会被model.eval()覆盖
    parser.add_argument('--num_blocks', type=int, default=2)  # SASRec通常默认是2
    parser.add_argument('--num_heads', type=int, default=1)   # SASRec通常默认是1
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    # 2. 加载数据
    print(f"📥 Loading Data...")
    with open(args.dataset_path, 'rb') as f:
        pkg = pickle.load(f)
    
    data = pkg['data']
    n_items = pkg['n_items']
    
    # [关键步骤] 随机抽取 2000 个样本作为"考试题"
    import random
    random.seed(42)
    sample_data = random.sample(data, min(args.sample_size, len(data)))
    
    print(f"🧐 Checking {len(sample_data)} samples from Training Set...")
    
    # 3. 加载模型
    print(f"🏗️ Loading Model...")
    model = SASRec(n_items, args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 4. 评估循环
    hr_10 = 0
    hr_50 = 0
    ranks = []
    
    with torch.no_grad():
        # 简单 Batch 处理
        for i in tqdm(range(0, len(sample_data), args.batch_size)):
            batch = sample_data[i : i + args.batch_size]
            
            # 构造 Input (Seq[:-1]) 和 Target (Seq[-1])
            input_ids = []
            targets = []
            
            for item in batch:
                seq = item['sequence']
                # 输入: 取最后 max_len 长度
                inp = seq[:-1]
                target = seq[-1]
                
                # Padding (左填充)
                inp = inp[-(args.max_len-1):]
                pad_len = args.max_len - len(inp)
                inp = [0] * pad_len + inp
                
                input_ids.append(inp)
                targets.append(target)
            
            input_tensor = torch.LongTensor(input_ids).to(device)
            target_tensor = torch.LongTensor(targets).to(device)
            
            # 预测
            logits = model.predict_full(input_tensor) # [B, N+1]
            logits[:, 0] = -float('inf') # 屏蔽 Padding
            
            # 计算 Rank
            # 获取 target 的分数
            target_scores = logits.gather(1, target_tensor.unsqueeze(1))
            # 统计有多少个比 target 分数高
            rank = (logits > target_scores).sum(dim=-1) + 1
            
            ranks.extend(rank.cpu().tolist())
            
            # 计算 HR
            # Top-K indices
            _, top10 = torch.topk(logits, 10, dim=-1)
            _, top50 = torch.topk(logits, 50, dim=-1)
            
            target_cpu = target_tensor.cpu().numpy()
            top10_cpu = top10.cpu().numpy()
            top50_cpu = top50.cpu().numpy()
            
            for j in range(len(target_cpu)):
                if target_cpu[j] in top10_cpu[j]: hr_10 += 1
                if target_cpu[j] in top50_cpu[j]: hr_50 += 1

    # 5. 输出报告
    avg_rank = np.mean(ranks)
    print("\n" + "="*40)
    print("🎓 TEACHER QUALIFICATION REPORT")
    print("="*40)
    print(f"✅ Samples Checked: {len(sample_data)}")
    print(f"🎯 HR@10 (Accuracy):  {hr_10 / len(sample_data):.2%}")
    print(f"🎯 HR@50 (Recall):    {hr_50 / len(sample_data):.2%}")
    print(f"📏 Avg Rank (Error):  {avg_rank:.1f} / {n_items}")
    print("-" * 40)
    
    if avg_rank < 500:
        print("🎉 QUALIFIED: Excellent Teacher! (Memorization successful)")
    elif avg_rank < 5000:
        print("👌 ACCEPTABLE: Good enough to guide LLM.")
    else:
        print("❌ FAILED: Model is too weak/random. Check training code.")

if __name__ == "__main__":
    main()