import json
import os
import random

# =================配置区域=================
# 请确认这些路径是否正确
TRAIN_DATA = "/workspace/data/processed/train_prompts.jsonl"
TEST_DATA = "/workspace/data/processed/test_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
# =========================================

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def inspect_jsonl(filepath, name, num_samples=3):
    print_separator(f"Inspecting {name}")
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"✅ Total samples: {len(lines)}")
        
        # 随机抽取样本
        samples = random.sample(lines, min(num_samples, len(lines)))
        
        for i, line in enumerate(samples):
            try:
                data = json.loads(line)
                print(f"\n[Sample {i+1}]")
                
                # 1. 检查 Instruction
                inst = data.get('instruction', '')
                print(f"🔍 Instruction (Raw):\n{repr(inst)}") # 使用 repr 显示换行符等隐形字符
                
                # 关键检查点：是否有格式提示？
                has_trigger = "Response: <" in inst
                has_format_hint = "semantic ID" in inst or "<c0," in inst
                
                print(f"   👉 Contains 'Response: <': {has_trigger}")
                print(f"   👉 Contains Format Hint:   {has_format_hint}")
                
                # 2. 检查 Target
                meta = data.get('metadata', {})
                target = meta.get('target_sid')
                print(f"🎯 Target SID: {target} (Type: {type(target)})")
                
            except json.JSONDecodeError:
                print(f"❌ Line {i} is not valid JSON!")
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")

def inspect_mapping(filepath):
    print_separator("Inspecting Mapping File")
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Total Items: {len(data)}")
        
        # 抽取一个看结构
        if data:
            key = next(iter(data))
            val = data[key]
            print(f"\n[Sample Item ID: {key}]")
            print(json.dumps(val, indent=2, ensure_ascii=False))
            
            # 检查 full_sid 格式
            full_sid = val.get('full_sid')
            print(f"   👉 Full SID format: {full_sid} (Type: {type(full_sid)})")
    except Exception as e:
        print(f"❌ Error reading mapping: {str(e)}")

if __name__ == "__main__":
    inspect_mapping(MAPPING_FILE)
    inspect_jsonl(TRAIN_DATA, "Train Data (RL Inputs)")
    inspect_jsonl(TEST_DATA, "Test Data (Eval Inputs)")