import random
import json
import os
from tqdm import tqdm

# ================= 配置 =================
# 输入：多进程生成的原始大文件 (虽然物理上是顺序的，但内容是全的)
INPUT_FILE = "./SFT/sft_data/sft_enhanced_train.jsonl"

# 输出：最终喂给模型的训练集 (已彻底打乱，且大小适中)
OUTPUT_FILE = "./SFT/sft_data/sft_train_1M_shuffled.jsonl"

# 采样数量：推荐 100万 - 200万
# 1800万数据全跑完一个Epoch需要很久，且容易过拟合。
# 100万条高质量、打乱的数据通常能达到最佳效果。
SAMPLE_SIZE = 1_500_000 

def process_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file {INPUT_FILE} not found.")
        print("Please run sft_data_engine.py first to generate the raw data.")
        return

    print(f"📖 Reading {INPUT_FILE} into RAM...")
    print("   (With 96GB RAM, this is the fastest way. Loading 18M lines...)")
    
    # 一次性读入内存
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"✅ Loaded {total_lines} samples.")
    
    if total_lines == 0:
        print("❌ Error: File is empty.")
        return

    # 1. 彻底打乱 (这就是你想要的“均衡”)
    print("🎲 Shuffling data globally...")
    random.shuffle(lines)
    
    # 2. 降采样 (Subsampling)
    if total_lines > SAMPLE_SIZE:
        print(f"✂️ Subsampling: Keeping {SAMPLE_SIZE} samples out of {total_lines}...")
        final_lines = lines[:SAMPLE_SIZE]
    else:
        print(f"⚠️ Warning: Total lines ({total_lines}) < Sample Size ({SAMPLE_SIZE}). Keeping all.")
        final_lines = lines
        
    # 3. 写入最终文件
    print(f"💾 Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(final_lines)
        
    print(f"\n🎉 Success! The file '{OUTPUT_FILE}' is now perfectly balanced and ready for training.")
    print("   Next step: Update 'DATA_FILE' in run_sft.py to point to this file.")

if __name__ == "__main__":
    process_dataset()