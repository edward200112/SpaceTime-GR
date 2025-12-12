import json
import random
import os
import math
from collections import defaultdict
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "/workspace/data/processed/train_prompts.jsonl"
OUTPUT_FILE = "/workspace/data/processed/train_prompts_balanced.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
WEIGHTS_FILE = "/workspace/data/processed/category_weights.json"

# 平衡策略参数
TARGET_PERCENTILE = 0.75  # 目标对齐到前 75% 的分位数
MAX_UPSCALE_RATIO = 10    # 限制最大复制倍数，防止过拟合太严重
# ===========================================

def main():
    print("=== Step 1: Loading & Filtering Data ===")
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")
        
    with open(MAPPING_FILE) as f:
        sid_map = json.load(f)
    
    data_by_cat = defaultdict(list)
    auxiliary_data = [] # Task B & C
    
    valid_count = 0
    skipped_count = 0
    
    with open(INPUT_FILE, 'r') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except:
                continue

            # 1. 分离辅助任务
            task_type = item.get('task')
            if task_type != 'task_a_recommendation':
                auxiliary_data.append(item)
                continue
            
            # 2. 处理推荐任务 Task A
            meta = item.get('metadata', {})
            t_raw = meta.get('target_sid')
            
            if not t_raw:
                skipped_count += 1; continue

            # 解析 Target ID
            if isinstance(t_raw, str):
                try:
                    clean = t_raw.replace('<','').replace('>','').replace('[','').replace(']','')
                    t_sid = tuple(int(x.strip()) for x in clean.split(','))
                except:
                    skipped_count += 1; continue
            else:
                t_sid = tuple(t_raw)
            
            # 按 L2 (Category) 分组
            if len(t_sid) >= 3:
                l2 = t_sid[2] 
                data_by_cat[l2].append(item)
                valid_count += 1
            else:
                skipped_count += 1

    print(f"Task A (Rec): {valid_count} | Aux (B/C): {len(auxiliary_data)} | Invalid: {skipped_count}")
    
    # --- Step 2: 计算平衡目标 ---
    counts = {k: len(v) for k, v in data_by_cat.items()}
    sorted_counts = sorted(counts.values())
    target_count = sorted_counts[int(len(sorted_counts) * TARGET_PERCENTILE)]
    target_count = max(target_count, 15) # 至少要有15条
    
    print(f"Target Count per Category: {target_count}")

    # --- Step 3: 重采样 (Resampling) ---
    balanced_data = []
    print("Resampling Task A data...")
    
    for cat, items in tqdm(data_by_cat.items()):
        count = len(items)
        if count > target_count:
            # 热门类别：下采样 (Down-sample)
            balanced_data.extend(random.sample(items, target_count))
        else:
            # 冷门类别：上采样 (Up-sample)
            balanced_data.extend(items) # 先全加进去
            needed = target_count - count
            
            # 限制最大倍率，防止某个极其罕见的只有1条的数据被复制100次
            max_add = count * MAX_UPSCALE_RATIO
            real_add = min(needed, max_add)
            
            if real_add > 0:
                balanced_data.extend(random.choices(items, k=real_add))
    
    # 把辅助任务加回来
    print(f"Adding {len(auxiliary_data)} auxiliary samples...")
    balanced_data.extend(auxiliary_data)
    random.shuffle(balanced_data)
    
    # 保存训练数据
    with open(OUTPUT_FILE, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved Balanced Data: {OUTPUT_FILE} (Total: {len(balanced_data)})")
            
    # --- Step 4: 计算平滑权重 (给 GRPO 用) ---
    print("Calculating Log-Smoothed Weights for RL...")
    weights = {}
    total_raw = sum(counts.values())
    
    max_raw_freq = max(counts.values()) / total_raw
    min_raw_freq = min(counts.values()) / total_raw
    
    for cat, count in counts.items():
        freq = count / total_raw
        # 逆频率
        inv_freq = 1.0 / (freq + 1e-6)
        # Log 平滑: log(1 + 1/freq)
        # 这种方式比 sqrt 更温和，且能拉开差距
        w = math.log(1.0 + inv_freq)
        weights[cat] = w
    
    # 归一化权重: 让最小的权重为 1.0，最大的不超过 5.0
    min_w = min(weights.values())
    max_val = 5.0
    
    final_weights = {}
    for k, v in weights.items():
        norm_w = v / min_w
        final_weights[k] = min(norm_w, max_val) # Cap at 5.0
        
    # 保存权重
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(final_weights, f, indent=2)
        
    print(f"Saved Weights to {WEIGHTS_FILE}")
    print("Done.")

if __name__ == "__main__":
    main()