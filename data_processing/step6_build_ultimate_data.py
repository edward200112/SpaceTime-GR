import json
import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ================= 配置 =================
RAW_REVIEW_FILE = "/workspace/data/raw/yelp_academic_dataset_review.json"
BUSINESS_FILE = "/workspace/data/raw/yelp_academic_dataset_business.json"
OUTPUT_DIR = "/workspace/data/processed_pinrec"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预训练语义模型 (用于提取 Content Features)
# 推荐用轻量高效的 all-MiniLM-L6-v2
MODEL_NAME = 'all-MiniLM-L6-v2'

def parse_time(date_str):
    # Yelp format: "2018-05-07 04:35:00"
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

def main():
    print("=== Step 1: Pre-computing Item Content Features (OmniSage) ===")
    # 1. 加载 Business 元数据
    bus_data = {}
    descriptions = []
    bids = []
    
    with open(BUSINESS_FILE, 'r') as f:
        for line in tqdm(f, desc="Reading Business"):
            item = json.loads(line)
            bid = item['business_id']
            # 构建富文本描述：Name + Categories + City + Stars
            # PinRec 强调利用 Text/Image 特征来辅助 ID
            desc = f"{item['name']} is a {item['categories']} in {item['city']}. Rated {item['stars']} stars."
            bus_data[bid] = {'desc': desc, 'stars': item['stars']}
            descriptions.append(desc)
            bids.append(bid)
            
    # 2. 编码文本特征 (离线计算，极大加速训练)
    print(f"Encoding {len(descriptions)} items with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    if torch.cuda.is_available(): model = model.to('cuda')
    
    # Batch encoding
    embeddings = model.encode(descriptions, batch_size=1024, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. 保存 ID 映射和特征矩阵
    bid_to_int = {bid: i for i, bid in enumerate(bids)}
    np.save(os.path.join(OUTPUT_DIR, "item_content_feats.npy"), embeddings)
    with open(os.path.join(OUTPUT_DIR, "bid_to_int.json"), 'w') as f:
        json.dump(bid_to_int, f)
        
    print("=== Step 2: Building Time-Aware User Sequences ===")
    # 按照 User 分组并排序 Review
    user_reviews = {} # user_id -> list of (timestamp, bid, stars)
    
    with open(RAW_REVIEW_FILE, 'r') as f:
        for line in tqdm(f, desc="Reading Reviews"):
            r = json.loads(line)
            uid = r['user_id']
            bid = r['business_id']
            if bid not in bid_to_int: continue # 过滤掉未知 item
            
            ts = parse_time(r['date'])
            stars = r['stars']
            
            if uid not in user_reviews: user_reviews[uid] = []
            user_reviews[uid].append({'ts': ts, 'bid': bid, 'stars': stars})
            
    # 构建训练样本
    train_samples = []
    
    for uid, history in tqdm(user_reviews.items(), desc="Processing Users"):
        # 按时间排序
        history.sort(key=lambda x: x['ts'])
        if len(history) < 5: continue
        
        # 滑动窗口切分
        # 我们模拟 PinRec 的多目标：同时预测 Immediate (Next) 和 Future (Gap > 1 day)
        for i in range(len(history) - 2):
            # Input: history[:i+1]
            input_seq = history[max(0, i-10) : i+1] # 保留最近10个
            current_ts = input_seq[-1]['ts']
            
            # Target 1: Immediate Next
            target_1 = history[i+1]
            delta_1 = target_1['ts'] - current_ts
            
            # Target 2: Future (尝试找一个时间间隔比较大的)
            # 简单的策略：找后面第 3 个，或者找时间差 > 1天 的第一个
            target_2 = None
            for future in history[i+2:]:
                if future['ts'] - current_ts > 86400: # > 1 day
                    target_2 = future
                    break
            if target_2 is None and i+3 < len(history):
                target_2 = history[i+3] # 如果没找到长间隔，就拿后面的凑数
                
            if target_2:
                # 构造样本
                # Outcome 逻辑: >=4 stars is SAVE(1), else CLICK(0)
                outcome_1 = 1 if target_1['stars'] >= 4.0 else 0
                outcome_2 = 1 if target_2['stars'] >= 4.0 else 0
                
                # 文本 Prompt (仅 ID 列表或 Names，这里简化为 Names 列表用于 LLM)
                prompt_ids = [bid_to_int[x['bid']] for x in input_seq]
                
                train_samples.append({
                    "history_ids": prompt_ids, # Integer IDs
                    "target_1": {
                        "bid_int": bid_to_int[target_1['bid']],
                        "delta_sec": delta_1,
                        "outcome": outcome_1
                    },
                    "target_2": {
                        "bid_int": bid_to_int[target_2['bid']],
                        "delta_sec": target_2['ts'] - current_ts,
                        "outcome": outcome_2
                    }
                })
                
    # 保存
    print(f"Saving {len(train_samples)} samples...")
    # 为了 IO 效率，存为 pt 或者 arrow 最好，这里存 jsonl
    with open(os.path.join(OUTPUT_DIR, "train_ultimate.jsonl"), 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')
            
    print("Data Ready.")

if __name__ == "__main__":
    main()