import os
import random
from tqdm import tqdm

# ================= 配置 =================
OUTPUT_DIR = "./SFT/sft_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_balanced_train.jsonl")
TARGET_SAMPLES_PER_REGION = 300000 

# 定义文件列表 (和之前一致)
REGION_NAMES = ["California", "New_York", "New_Mexico", "Pennsylvania", "Unknown"]

def reservoir_sampling(file_path, k):
    """
    蓄水池采样：从巨大的文件中随机抽取 k 行，内存占用恒定为 O(k)
    """
    sample = []
    print(f"   Streaming {os.path.basename(file_path)}...")
    
    with open(file_path, 'r') as f:
        # 1. 先填满池子
        for i, line in enumerate(f):
            if i < k:
                sample.append(line)
            else:
                # 2. 之后的每一行，以 k/(i+1) 的概率替换池子里的元素
                j = random.randint(0, i)
                if j < k:
                    sample[j] = line
            
            # 打印进度 (每处理 100万行 打印一次，防止刷屏)
            if (i + 1) % 1000000 == 0:
                print(f"     Processed {i/1000000:.1f}M lines...", end='\r')
    
    print(f"\n     Selected {len(sample)} lines from {os.path.basename(file_path)}.")
    return sample

def main():
    print("🚀 Starting Recovery Merge (Memory Safe Mode)...")
    
    final_samples = []
    
    for r_name in REGION_NAMES:
        temp_path = os.path.join(OUTPUT_DIR, f"temp_{r_name}.jsonl")
        
        if not os.path.exists(temp_path):
            print(f"⚠️ Warning: {temp_path} not found, skipping.")
            continue
            
        print(f"Processing Region: {r_name}")
        
        # 使用蓄水池采样，而不是 readlines()
        # 这样即使 California 有 4600万行，内存也只存 30万行
        region_samples = reservoir_sampling(temp_path, TARGET_SAMPLES_PER_REGION)
        final_samples.extend(region_samples)

    print(f"🎲 Final Global Shuffle of {len(final_samples)} samples...")
    random.shuffle(final_samples)
    
    print(f"💾 Saving Balanced Data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(final_samples)
        
    print("✅ Merge Completed Successfully!")
    print("🧹 (Optional) You can now delete the temp_*.jsonl files to free space.")

if __name__ == "__main__":
    main()