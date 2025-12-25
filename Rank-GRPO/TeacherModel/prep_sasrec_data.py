import os
import gzip
import json
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ================= 配置路径 =================
RAW_DATA_DIR = "/workspace/data/GoogleRAW"
ID_MAPPING_FILE = "./poi_semantic_ids.csv"
OUTPUT_DIR = "./SASRec_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义要处理的区域
REGION_FILES = [
    'review-California.json.gz',
    'review-New_York.json.gz',
    'review-New_Mexico.json.gz',
    'review-Pennsylvania.json.gz'
]

def main():
    print("1️⃣ Loading Valid POI IDs (Aligning with LLM vocabulary)...")
    if not os.path.exists(ID_MAPPING_FILE):
        raise FileNotFoundError(f"Mapping file not found: {ID_MAPPING_FILE}")
    
    # 读取 ID 映射表
    df = pd.read_csv(ID_MAPPING_FILE)
    # 确保 gmap_id 是字符串并去除空格
    df['gmap_id'] = df['gmap_id'].astype(str).str.strip()
    valid_gmap_ids = set(df['gmap_id'].tolist())
    
    print(f"   Loaded {len(valid_gmap_ids)} valid POIs.")

    # 构建内部 ID 映射
    # item2id: gmap_id -> int (1..N)
    # id2item: int -> gmap_id
    item2id = {}
    id2item = {}
    
    # 预先为所有 valid POI 分配 ID，保证 ID 空间固定
    for gid in valid_gmap_ids:
        new_id = len(item2id) + 1 # 从 1 开始
        item2id[gid] = new_id
        id2item[new_id] = gid

    print("2️⃣ Processing Interaction Data...")
    user_history = defaultdict(list)
    
    total_interactions = 0
    skipped_interactions = 0
    
    for r_file in REGION_FILES:
        path = os.path.join(RAW_DATA_DIR, r_file)
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} not found, skipping.")
            continue
            
        print(f"   Reading {r_file}...")
        with gzip.open(path, 'r') as f:
            for line in tqdm(f):
                try:
                    d = json.loads(line)
                    uid = str(d['user_id']).strip()
                    gid = str(d['gmap_id']).strip()
                    timestamp = d['time'] 
                    
                    # 转换时间戳 (如果是毫秒级，转秒)
                    if timestamp > 1000000000000:
                        timestamp = timestamp // 1000
                    
                    if gid in item2id:
                        user_history[uid].append((timestamp, item2id[gid]))
                        total_interactions += 1
                    else:
                        skipped_interactions += 1
                except Exception as e:
                    continue

    print(f"   Total interactions: {total_interactions}")
    print(f"   Skipped (not in ID map): {skipped_interactions}")

    print("3️⃣ Sorting and Filtering Users...")
    processed_dataset = []
    
    min_len = 5
    max_len = 50 
    
    for uid, hist in tqdm(user_history.items()):
        # 按时间排序
        hist.sort(key=lambda x: x[0])
        
        # 只取 Item ID
        seq = [x[1] for x in hist]
        
        # 过滤短序列
        if len(seq) < min_len:
            continue
            
        # 截断长序列 (取最近的 N 个)
        seq = seq[-max_len:]
        
        processed_dataset.append({
            "user_id": uid, # 保留原始字符串 ID，用于导出 JSON Key
            "sequence": seq
        })

    print(f"✅ Final Statistics:")
    print(f"   - Total Valid Users: {len(processed_dataset)}")
    print(f"   - Total Items: {len(item2id)}")
    
    save_path = os.path.join(OUTPUT_DIR, "sasrec_dataset.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({
            "data": processed_dataset,
            "item2id": item2id,
            "id2item": id2item,
            "n_items": len(item2id),
            "max_len": max_len
        }, f)
        
    print(f"💾 Data saved to {save_path}")

if __name__ == "__main__":
    main()