import json
import random
import os
from collections import defaultdict
from tqdm import tqdm

INPUT_FILE = "/workspace/data/processed/train_prompts.jsonl"
OUTPUT_FILE = "/workspace/data/processed/train_prompts_balanced.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
WEIGHTS_FILE = "/workspace/data/processed/category_weights.json"

def main():
    print("1. Loading Mapping to get Categories...")
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")
        
    with open(MAPPING_FILE) as f:
        sid_map = json.load(f)
    
    print("2. Loading Training Data & Filtering...")
    data_by_cat = defaultdict(list)
    auxiliary_data = [] # 用于存储 Task B/C
    
    valid_count = 0
    skipped_count = 0
    
    with open(INPUT_FILE, 'r') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except:
                skipped_count += 1; continue

            # [关键修复] 严格过滤 Task 类型
            task_type = item.get('task')
            
            # 如果不是 Task A，不要去读 metadata，直接存入辅助列表
            if task_type != 'task_a_recommendation':
                auxiliary_data.append(item)
                continue
            
            # --- 只有 Task A 才会执行以下逻辑 ---
            
            # 安全获取 metadata
            meta = item.get('metadata')
            if not meta:
                skipped_count += 1; continue
                
            t_raw = meta.get('target_sid')
            if not t_raw:
                skipped_count += 1; continue

            # 解析 ID
            if isinstance(t_raw, str):
                try:
                    clean = t_raw.replace('<','').replace('>','').replace('[','').replace(']','')
                    t_sid = tuple(int(x.strip()) for x in clean.split(','))
                except:
                    skipped_count += 1; continue
            else:
                t_sid = tuple(t_raw)
            
            # 获取 Layer 2 (Category) ID
            if len(t_sid) >= 3:
                l2 = t_sid[2] 
                data_by_cat[l2].append(item)
                valid_count += 1
            else:
                skipped_count += 1

    print(f"\nProcessing Complete.")
    print(f"Task A (Recommendation): {valid_count} samples")
    print(f"Auxiliary Tasks (B/C): {len(auxiliary_data)} samples")
    print(f"Skipped/Invalid: {skipped_count}")
    
    if valid_count == 0:
        raise ValueError("没有找到任何有效的 Task A 样本！请检查数据集生成步骤。")

    # 3. 计算平衡目标
    counts = {k: len(v) for k, v in data_by_cat.items()}
    sorted_counts = sorted(counts.values())
    
    # 策略：取 75% 分位数
    target_idx = int(len(sorted_counts) * 0.75)
    target_count = sorted_counts[target_idx]
    target_count = max(target_count, 10) 
    
    print(f"Target Count per Category: {target_count}")

    balanced_data = []
    
    # 4. 执行重采样
    print("Resampling Task A data...")
    for cat, items in tqdm(data_by_cat.items()):
        count = len(items)
        if count > target_count:
            # Down-sample
            balanced_data.extend(random.sample(items, target_count))
        else:
            # Up-sample
            balanced_data.extend(items)
            needed = target_count - count
            if needed > 0:
                balanced_data.extend(random.choices(items, k=needed))
    
    # [关键] 把 Task B/C 加回来！
    # SFT 最好还是保留多任务能力，防止遗忘
    print(f"Adding {len(auxiliary_data)} auxiliary samples back...")
    balanced_data.extend(auxiliary_data)
    
    # 打乱顺序
    random.shuffle(balanced_data)
    
    print(f"Writing {len(balanced_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # 5. 生成权重文件 (给 RL 使用)
    print("Calculating Category Weights for RL...")
    weights = {}
    total_raw = sum(counts.values())
    
    for cat, count in counts.items():
        freq = count / total_raw
        weights[cat] = 1.0 / (freq ** 0.5)
    
    min_w = min(weights.values())
    final_weights = {k: min(v / min_w, 10.0) for k, v in weights.items()}
    
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(final_weights, f, indent=2)
        
    print(f"Saved category weights to {WEIGHTS_FILE}")
    print("Done! SFT V5 Data Ready.")

if __name__ == "__main__":
    main()