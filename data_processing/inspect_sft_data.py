import json
import numpy as np
from tqdm import tqdm
from collections import Counter

# 数据路径
DATA_PATH = "/workspace/data/processed/train_balanced_pinrec.jsonl"

def inspect_data():
    print(f"🕵️‍♂️ Starting Data Inspection for: {DATA_PATH}")
    
    max_id_found = 0
    min_id_found = float('inf')
    
    act_counter = Counter()
    missing_fields = 0
    total_lines = 0
    
    # 抽样检查前 50,000 条，或者设为 None 跑全量
    LIMIT = 100000 
    
    try:
        with open(DATA_PATH, 'r') as f:
            for i, line in tqdm(enumerate(f), total=LIMIT):
                if LIMIT and i >= LIMIT: break
                total_lines += 1
                
                try:
                    row = json.loads(line)
                    
                    # 1. 检查关键字段是否存在
                    if 'target_1' not in row or 'target_2' not in row:
                        missing_fields += 1
                        continue
                        
                    t1 = row['target_1']
                    t2 = row['target_2']
                    
                    # 2. 检查 ID 范围
                    ids = row.get('history_ids', []) + [t1.get('id', 0), t2.get('id', 0)]
                    curr_max = max(ids) if ids else 0
                    if curr_max > max_id_found: max_id_found = curr_max
                    
                    # 3. 检查 Act (动作类型)
                    # 必须要有 act 字段，且值为 0 或 1
                    if 'act' in t1: act_counter[t1['act']] += 1
                    if 'act' in t2: act_counter[t2['act']] += 1
                    
                    # 4. 检查 Delta (时间)
                    if 'delta' not in t1: missing_fields += 1
                    
                except json.JSONDecodeError:
                    print(f"❌ JSON Error at line {i}")
                    
    except FileNotFoundError:
        print(f"❌ File not found: {DATA_PATH}")
        return

    print("\n" + "="*40)
    print("📊 DATA HEALTH REPORT")
    print("="*40)
    print(f"Scanned Lines: {total_lines}")
    
    print(f"\n1. ID Range Check:")
    print(f"   Max ID Found: {max_id_found}")
    if max_id_found > 250000:
        print("   ⚠️ WARNING: IDs exceed 250k. 'Modulo' logic in SFT code will be heavily active.")
        print("   (This is acceptable for Hash Embeddings, but ensure collision is expected.)")
    else:
        print("   ✅ IDs are within safe range (<250k).")

    print(f"\n2. Outcome Conditioning (Action) Check:")
    print(f"   Action Distribution: {dict(act_counter)}")
    if 0 in act_counter and 1 in act_counter:
        print("   ✅ Good! Both Clicks (0) and Repins (1) detected.")
    else:
        print("   ❌ CRITICAL: Missing one type of action! Model cannot learn Outcome Conditioning.")

    print(f"\n3. Structure Check:")
    if missing_fields == 0:
        print("   ✅ All samples have target_1, target_2, and deltas.")
    else:
        print(f"   ❌ Found {missing_fields} samples with missing fields.")

    print("="*40)

if __name__ == "__main__":
    inspect_data()