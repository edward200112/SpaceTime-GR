import json
import os

# 路径配置
file_path = "/workspace/data/processed/train_prompts.jsonl"

print(f"Inspecting: {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    # 读取前 3 行
    for i, line in enumerate(f):
        if i >= 3: break
        
        data = json.loads(line)
        meta = data.get('metadata', {})
        target_sid = meta.get('target_sid')
        
        print(f"\n[Sample {i}]")
        print(f"  Type of target_sid: {type(target_sid)}")
        print(f"  Value of target_sid: {target_sid}")
        print("-" * 30)