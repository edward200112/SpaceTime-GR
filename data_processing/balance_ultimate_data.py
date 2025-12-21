import json
import random
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# [配置]
# 输入必须是 Ultimate 格式的数据
INPUT_FILE = "/workspace/data/processed_pinrec_v2/train_ultimate.jsonl"
# 输出到训练代码读取的路径
OUTPUT_FILE = "/workspace/data/processed/train_balanced_pinrec.jsonl"

def main():
    print(f"1. Loading Ultimate Data from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data_by_item = defaultdict(list)
    total_lines = 0
    
    with open(INPUT_FILE, 'r') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            try:
                seq = json.loads(line)
                # [关键] 我们根据 target_1 (Next Item) 的 ID 来进行分组平衡
                # 这样可以防止某些 Item 占据过多的 Next Item 预测位
                tid = seq['target_1']['id'] 
                data_by_item[tid].append(seq)
                total_lines += 1
            except KeyError:
                continue # 跳过损坏的数据

    print(f"Loaded {total_lines} sequences.")
    print(f"Unique Items (as Target 1): {len(data_by_item)}")

    # 2. 计算统计量
    counts = [len(v) for v in data_by_item.values()]
    print(f"Stats -> Max: {max(counts)}, Min: {min(counts)}, Mean: {np.mean(counts):.2f}")
    
    # 策略：取中位数作为基准
    median_count = int(np.median(counts))
    # 设定上下限：太少的补一点，太多的砍掉
    # 热门物品限制在 200 条以内 (防止过拟合热门)
    # 冷门物品至少补到 10 条 (防止欠拟合)
    target_cap = max(median_count * 2, 50) 
    target_floor = 5
    
    # 稍微放宽上限，避免数据损失太多
    target_cap = min(target_cap, 500) 
    
    print(f"Balancing Strategy -> Floor: {target_floor}, Cap: {target_cap}")

    balanced_data = []
    
    # 3. 执行重采样
    print("Resampling...")
    for tid, items in tqdm(data_by_item.items()):
        count = len(items)
        
        if count > target_cap:
            # 热门：下采样 (随机抽取)
            balanced_data.extend(random.sample(items, target_cap))
        else:
            # 冷门：上采样 (复制)
            balanced_data.extend(items)
            if count < target_floor:
                needed = target_floor - count
                balanced_data.extend(random.choices(items, k=needed))
    
    # 打乱顺序
    random.shuffle(balanced_data)
    
    print(f"Writing {len(balanced_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done! Ultimate Data Balanced.")

if __name__ == "__main__":
    main()