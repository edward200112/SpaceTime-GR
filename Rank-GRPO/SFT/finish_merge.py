import os
import json
import random
from tqdm import tqdm

# ================= 配置 =================
DATA_DIR = "./SFT/sft_data"
# 最终输出文件
OUTPUT_FILE = os.path.join(DATA_DIR, "sft_balanced_train.jsonl")

# 均衡采样参数：每个地区最多保留 30万 条
# (如果该地区数据不够 30万，则全部保留)
TARGET_SAMPLES_PER_REGION = 300000

def merge_and_balance():
    print(f"🚀 Starting Phase 5 (Manual Resume)...")
    print(f"📂 Scanning {DATA_DIR} for temp files...")

    # 1. 自动寻找所有 temp_*.jsonl 文件
    # 你的目录下有: temp_California.jsonl, temp_New_York.jsonl 等
    temp_files = [f for f in os.listdir(DATA_DIR) if f.startswith("temp_") and f.endswith(".jsonl")]
    
    if not temp_files:
        print("❌ No temp files found! Please check the directory.")
        return

    print(f"   Found {len(temp_files)} files: {temp_files}")
    
    final_samples = []
    total_raw_count = 0

    # 2. 遍历处理每个地区文件
    for file_name in temp_files:
        file_path = os.path.join(DATA_DIR, file_name)
        # 从文件名提取地区名，例如 temp_California.jsonl -> California
        region_name = file_name.replace("temp_", "").replace(".jsonl", "")
        
        print(f"\nProcessing Region: {region_name} ...")
        
        valid_lines = []
        try:
            with open(file_path, 'r') as f:
                # 逐行读取并校验 JSON (防止上次中断导致最后一行损坏)
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        # 尝试解析，确保数据完整
                        # (虽然这步会慢一点，但为了安全是值得的)
                        json.loads(line)
                        valid_lines.append(line)
                    except json.JSONDecodeError:
                        # 忽略损坏的行
                        continue
        except Exception as e:
            print(f"   ⚠️ Error reading {file_name}: {e}")
            continue
        
        count = len(valid_lines)
        total_raw_count += count
        print(f"   - Raw Count: {count}")

        # 3. 打乱该地区数据 (Shuffle)
        random.shuffle(valid_lines)

        # 4. 均衡截断 (Subsample)
        if count > TARGET_SAMPLES_PER_REGION:
            kept_lines = valid_lines[:TARGET_SAMPLES_PER_REGION]
            print(f"   - ✂️ Subsampled to: {len(kept_lines)}")
        else:
            kept_lines = valid_lines
            print(f"   - ✅ Kept All ({count} < {TARGET_SAMPLES_PER_REGION})")
        
        final_samples.extend(kept_lines)

    # 5. 最终全局打乱 (Global Shuffle)
    print(f"\n🎲 Final Global Shuffle of {len(final_samples)} merged samples...")
    random.shuffle(final_samples)

    # 6. 写入最终结果
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for line in tqdm(final_samples):
            f.write(line + "\n")

    print("\n✅ Success! Dataset is ready.")
    print(f"   Total Raw Data: {total_raw_count}")
    print(f"   Final Balanced Data: {len(final_samples)}")
    
    # 7. (可选) 清理临时文件
    # print("🧹 Cleaning up temp files...")
    # for f in temp_files:
    #     os.remove(os.path.join(DATA_DIR, f))

if __name__ == "__main__":
    merge_and_balance()