import json
import random

FILE_PATH = "./SFT/sft_data/sft_balanced_train.jsonl"
NUM_SAMPLES = 3

print(f"🔍 Inspecting {NUM_SAMPLES} random samples from {FILE_PATH}...\n")

with open(FILE_PATH, 'r') as f:
    lines = f.readlines()
    
    # 随机抽几条
    samples = random.sample(lines, NUM_SAMPLES)
    
    for i, line in enumerate(samples):
        data = json.loads(line)
        print(f"--- [Sample {i+1}] ---")
        
        # 1. 检查核心字段是否存在
        keys = data.keys()
        print(f"✅ Keys present: {list(keys)}")
        
        # 2. 打印 Prompt (部分)
        print(f"📝 Prompt (Truncated): {data['prompt'][:100]}...")
        
        # 3. 打印 CoT Completion
        print(f"🧠 Completion (CoT): {data['completion']}")
        
        # 4. 打印 Raw Target Code (用于课程学习)
        print(f"🎯 Raw Target Code: {data.get('raw_target_code', 'MISSING')}")
        
        # 5. 打印时空增强字段
        print(f"🌍 Region/Augment: Has 'prompt_augment'? {'prompt_augment' in data}")
        print("-" * 50 + "\n")

print("✅ Inspection Complete.")