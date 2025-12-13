import json
import random
import os
from collections import defaultdict, Counter
from tqdm import tqdm

# 配置路径 (请根据你的实际情况修改)
INPUT_FILE = "/workspace/data/processed/train.jsonl"  # PinRec 用的原始序列数据
OUTPUT_FILE = "/workspace/data/processed/train_balanced_pinrec.jsonl" # 平衡后的数据
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

def main():
    print("1. Loading Mapping (ID -> Category)...")
    with open(MAPPING_FILE) as f:
        sid_map = json.load(f)
    
    # 建立 Business ID -> Category 的映射
    # 为了简化，我们使用 Layer 2 (Category) ID 来做平衡
    bid_to_cat = {}
    for bid, meta in sid_map.items():
        full = tuple(int(x) for x in meta['full_sid'])
        if len(full) >= 3:
            bid_to_cat[bid] = full[2] # Layer 2 ID

    print("2. Loading Training Sequences...")
    data_by_cat = defaultdict(list)
    skipped = 0
    
    with open(INPUT_FILE, 'r') as f:
        for line in tqdm(f):
            seq = json.loads(line)
            target_bid = seq['target']['business_id']
            
            # 找到该样本属于哪个类别
            cat_id = bid_to_cat.get(target_bid)
            
            if cat_id is not None:
                data_by_cat[cat_id].append(seq)
            else:
                skipped += 1

    print(f"Loaded. Skipped {skipped} items without category info.")
    print(f"Total Categories: {len(data_by_cat)}")

    # 3. 统计与计算目标数量
    counts = {k: len(v) for k, v in data_by_cat.items()}
    sorted_counts = sorted(counts.values())
    
    # 策略：取中位数作为基准 (比 SFT 稍微保守一点，防止过拟合)
    target_count = sorted_counts[len(sorted_counts) // 2]
    # 设定上下限
    target_count = max(target_count, 20)  # 至少 20 条
    target_count = min(target_count, 5000) # 至多 5000 条 (防止热门太热)
    
    print(f"Target Count per Category: {target_count}")
    print(f"Max: {max(counts.values())}, Min: {min(counts.values())}")

    balanced_data = []
    
    # 4. 执行重采样
    print("Balancing data...")
    for cat, items in tqdm(data_by_cat.items()):
        count = len(items)
        
        if count > target_count:
            # 热门类别：下采样 (随机丢弃)
            balanced_data.extend(random.sample(items, target_count))
        else:
            # 冷门类别：上采样 (复制)
            balanced_data.extend(items) # 先把有的都加上
            needed = target_count - count
            if needed > 0:
                balanced_data.extend(random.choices(items, k=needed))
    
    # 打乱顺序
    random.shuffle(balanced_data)
    
    print(f"Writing {len(balanced_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done! PinRec training data balanced.")

if __name__ == "__main__":
    main()