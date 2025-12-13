import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split # 引入切分工具

# ================= 配置 =================
# 请确保这些路径指向你最原始的数据集
RAW_REVIEW_FILE = "/workspace/data/raw/yelp_academic_dataset_review.json"
BUSINESS_FILE = "/workspace/data/raw/yelp_academic_dataset_business.json"
OUTPUT_DIR = "/workspace/data/processed_pinrec_v2" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'
TEST_SIZE = 0.2 # 20% 作为验证集

def parse_time(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

def main():
    print("=== Step 1: 建立严格的 ID 索引与内容特征 ===")
    
    # 1.1 读取所有 Business
    items = []
    print("Reading Business file...")
    with open(BUSINESS_FILE, 'r') as f:
        for line in tqdm(f, desc="Reading Business Meta"):
            item = json.loads(line)
            # 确保 Item ID 存在
            if 'business_id' not in item or not item['business_id']:
                continue
            desc = f"{item['name']} {item.get('categories', '')} {item.get('city', '')}"
            items.append({
                'raw_id': item['business_id'],
                'desc': desc,
                'stars': item.get('stars', 0.0)
            })
    
    df_items = pd.DataFrame(items)
    print(f"Total Raw Items: {len(df_items)}")

    # 1.2 !!! 核心修正：强制排序 !!!
    df_items = df_items.sort_values('raw_id').reset_index(drop=True)
    
    # 1.3 生成映射字典 (行号即为 int_id)
    raw_to_int = {rid: i for i, rid in enumerate(df_items['raw_id'])}
    
    # 1.4 重新生成/提取 Embedding
    # 这一步必须重新运行，以确保和 ItemTower 的 content_feats.npy 严格一致！
    print(f"Encoding {len(df_items)} items (Strict Order)...")
    model = SentenceTransformer(MODEL_NAME)
    if torch.cuda.is_available(): model = model.to('cuda')
    
    embeddings = model.encode(
        df_items['desc'].tolist(), 
        batch_size=1024, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    # 1.5 保存特征矩阵
    np.save(os.path.join(OUTPUT_DIR, "item_content_feats.npy"), embeddings)
    print(f"Feature Matrix Saved. Shape: {embeddings.shape}")

    print("\n=== Step 2: 构建 PinnerFormer 训练样本 ===")
    
    train_samples = []
    user_reviews = {}
    
    # 2.1 读取交互
    with open(RAW_REVIEW_FILE, 'r') as f:
        for line in tqdm(f, desc="Reading Reviews"):
            r = json.loads(line)
            bid = r['business_id']
            if bid not in raw_to_int: continue 
            
            uid = r['user_id']
            int_id = raw_to_int[bid]
            ts = parse_time(r['date'])
            stars = r['stars']
            
            action_type = 1 if stars >= 4.0 else 0
            
            if uid not in user_reviews: user_reviews[uid] = []
            user_reviews[uid].append({'id': int_id, 'ts': ts, 'act': action_type})

    # 2.2 滑动窗口生成序列
    for uid, history in tqdm(user_reviews.items(), desc="Processing Users"):
        history.sort(key=lambda x: x['ts'])
        if len(history) < 5: continue
        
        if len(history) > 100: history = history[-100:]
        
        for i in range(1, len(history) - 2):
            raw_seq = history[max(0, i-10) : i+1] 
            current_ts = raw_seq[-1]['ts']
            
            h_ids = [x['id'] for x in raw_seq]
            h_acts = [x['act'] for x in raw_seq]
            h_deltas = [current_ts - x['ts'] for x in raw_seq] 
            
            target_1 = history[i+1]
            
            target_2 = None
            for future in history[i+2:]:
                if future['ts'] - current_ts > 86400:
                    target_2 = future
                    break
            if target_2 is None: target_2 = history[i+2]
            
            # 构造样本
            train_samples.append({
                "history_ids": h_ids, "history_acts": h_acts, "history_deltas": h_deltas,
                "target_1": {"id": target_1['id'], "delta": target_1['ts'] - current_ts, "act": target_1['act']},
                "target_2": {"id": target_2['id'], "delta": target_2['ts'] - current_ts, "act": target_2['act']}
            })

    # -------------------------------------------------------------
    # !!! 核心修正 3: 训练集和验证集切分 !!!
    # -------------------------------------------------------------
    print(f"\nTotal generated samples: {len(train_samples)}")
    
    # 随机切分 (80% 训练, 20% 验证)
    train_set, val_set = train_test_split(
        train_samples, 
        test_size=TEST_SIZE, 
        random_state=42, # 固定随机种子以确保每次切分一致
        shuffle=True
    )
    
    print(f"Training Samples: {len(train_set)}")
    print(f"Validation Samples: {len(val_set)}")
    
    # 3.1 保存训练集
    train_output_path = os.path.join(OUTPUT_DIR, "train_ultimate.jsonl")
    print(f"Saving training samples to {train_output_path}...")
    with open(train_output_path, 'w') as f:
        for s in tqdm(train_set, desc="Saving Train"):
            f.write(json.dumps(s) + '\n')
            
    # 3.2 保存验证集
    val_output_path = os.path.join(OUTPUT_DIR, "validation_ultimate.jsonl")
    print(f"Saving validation samples to {val_output_path}...")
    with open(val_output_path, 'w') as f:
        for s in tqdm(val_set, desc="Saving Val"):
            f.write(json.dumps(s) + '\n')
            
    print(">>> Data Pipeline Finished. Training and Validation sets created.")

if __name__ == "__main__":
    main()