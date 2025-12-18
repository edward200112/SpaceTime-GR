import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# 配置路径
DATA_FILE = "./SFT/sft_data/sft_enhanced_train.jsonl"

def analyze_distribution():
    print(f"📊 Analyzing distribution in {DATA_FILE}...")
    
    first_tokens = []
    
    try:
        with open(DATA_FILE, 'r') as f:
            for line in tqdm(f):
                try:
                    item = json.loads(line)
                    # completion 格式: "273 438 4 456"
                    # 我们取第一个数字 "273"，它代表 Region/State 层级
                    completion = item.get('completion', "")
                    if completion:
                        tokens = completion.split()
                        if len(tokens) > 0:
                            first_tokens.append(tokens[0])
                except:
                    continue
    except FileNotFoundError:
        print("❌ 文件未找到，请检查路径。")
        return

    # 统计分布
    counter = Counter(first_tokens)
    total = len(first_tokens)
    
    print(f"\n✅ Total Samples Analyzed: {total}")
    print("-" * 40)
    print("Top 10 Region Codes (Layer 1 Token):")
    print("Code\tCount\tPercentage")
    
    # 转换为 DataFrame 方便查看
    df = pd.DataFrame(counter.most_common(10), columns=['Token', 'Count'])
    df['Percentage'] = (df['Count'] / total) * 100
    
    for _, row in df.iterrows():
        print(f"{row['Token']}\t{row['Count']}\t{row['Percentage']:.2f}%")
        
    # 判断结论
    unique_tokens = len(counter)
    print("-" * 40)
    print(f"Total Unique Region Codes: {unique_tokens}")
    
    if unique_tokens < 50: # 假设总 Codebook 是 512
        print("\n🧐 Conclusion: Data is HIGHLY concentrated in a specific region.")
        print("Since you processed 'California' first, these codes likely correspond to California clusters.")
    else:
        print("\n🧐 Conclusion: Data seems mixed.")

if __name__ == "__main__":
    analyze_distribution()