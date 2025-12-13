"""
check_cluster_purity.py
检查 RQ-VAE 的 Layer 2 ID 是否具有语义一致性。
"""
import json
import os
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm

def analyze_purity(mapping_file):
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        data = json.load(f)

    # 1. Group by Layer 2 Prefix: <c0, c1, c2>
    clusters = defaultdict(list)
    for bid, meta in tqdm(data.items()):
        # full_sid: [c0, c1, c2, suffix]
        # 我们只看前三层
        prefix = tuple(meta['full_sid'][:3]) 
        # 获取类别 (categories 可能是 string "Food, Restaurants" 或 list)
        cats = meta.get('categories', '')
        if isinstance(cats, list): cats = ", ".join(cats)
        if cats:
            clusters[prefix].append(cats)

    print(f"\nTotal Layer 2 Clusters: {len(clusters)}")
    
    # 2. Analyze Consistency
    # 既然是粗略检查，我们看每个 Cluster 里出现频率最高的词覆盖了多少样本
    purity_scores = []
    
    print("\n--- Random Sample of Clusters ---")
    sample_count = 0
    
    for prefix, cat_list in clusters.items():
        if len(cat_list) < 3: continue # 忽略太小的簇
        
        # 统计高频词
        all_words = " ".join(cat_list).lower().replace(',', '').split()
        # 过滤常用无意义词
        stop_words = {'&', 'services', 'stores', 'service', 'shop'}
        filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
        
        if not filtered_words: continue
        
        counts = Counter(filtered_words)
        top_word, top_count = counts.most_common(1)[0]
        
        # 纯度 = 最主要关键词的出现次数 / 总词数 (粗略估计)
        # 或者：包含 Top Word 的 Item 比例
        hits = sum(1 for c in cat_list if top_word in c.lower())
        purity = hits / len(cat_list)
        purity_scores.append(purity)
        
        # 打印几个样本看看
        if sample_count < 10:
            print(f"Cluster {prefix} (N={len(cat_list)}):")
            print(f"  Top Concept: '{top_word}' (Covering {purity:.1%})")
            print(f"  Examples: {cat_list[:2]}")
            sample_count += 1

    avg_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0
    print("\n" + "="*40)
    print(f"AVERAGE CLUSTER PURITY: {avg_purity:.2%}")
    print("="*40)
    
    if avg_purity < 0.5:
        print("❌ 警告: 聚类纯度极低！RQ-VAE 主要是按地理位置聚类，而非语义。")
        print("   LLM 无法学习类别，因为同一个 ID 下面什么都有。")
    else:
        print("✅ 状态: 聚类纯度尚可。问题出在 LLM 训练策略(Reward)上。")

if __name__ == "__main__":
    # 指向你刚才找到的那个 71M 的文件
    MAPPING_PATH = "/workspace/data/processed/sid_mapping.json" 
    analyze_purity(MAPPING_PATH)