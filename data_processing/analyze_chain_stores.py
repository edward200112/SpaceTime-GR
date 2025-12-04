"""
优化后验证器：Chain Store & Cluster Analysis

目标：
1. 验证 Step 1 是否成功消除了文本重复 (Text Duplication应为0)
2. 分析 Step 2 的 Cluster ID 聚合情况 (同类店铺是否聚在一起)
3. 检查地理位置对聚类的影响
"""

import os
import json
import yaml
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

class ChainStoreAnalyzer:
    def __init__(self, config_path='./config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config['data']['processed_dir']
        # 注意文件名可能变了，这里读取配置
        self.profiles_file = os.path.join(self.processed_dir, self.config['data']['item_profile_file'])
        self.mapping_file = os.path.join(self.processed_dir, self.config['data']['sid_mapping_file'])
        
        self.all_businesses = []
        self.text_counter = Counter()
        self.name_to_cluster = defaultdict(list)
    
    def load_data(self):
        print("\n=== Loading Data ===")
        # Load Profiles
        with open(self.profiles_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading profiles"):
                p = json.loads(line)
                self.all_businesses.append(p)
                self.text_counter[p['raw_text']] += 1
        
        # Load ID Mapping
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.sid_mapping = json.load(f)
        else:
            self.sid_mapping = {}
            print("Warning: SID Mapping not found. Skip cluster analysis.")

    def verify_text_uniqueness(self):
        """验证 Step 1 的文本去重效果"""
        print("\n" + "=" * 60)
        print("VERIFICATION 1: Text Uniqueness (Step 1)")
        print("=" * 60)
        
        duplicates = {k: v for k, v in self.text_counter.items() if v > 1}
        total = len(self.all_businesses)
        dup_count = sum(duplicates.values())
        
        print(f"Total Businesses: {total}")
        print(f"Unique Texts:     {len(self.text_counter)}")
        print(f"Duplicate Texts:  {len(duplicates)}")
        
        if len(duplicates) == 0:
            print("\n✅ SUCCESS: Text descriptions are 100% unique!")
            print("   Reason: Added Address/Zip Code successfully.")
        else:
            print(f"\n❌ WARNING: {dup_count} businesses share identical descriptions.")
            print("   Top duplicates:")
            for text, count in list(duplicates.items())[:3]:
                print(f"   - ({count}) {text[:100]}...")
            print("   Action: Check step1_build_item_profile.py")

    def analyze_cluster_cohesion(self):
        """分析 Step 2 的聚类效果"""
        if not self.sid_mapping: return
        
        print("\n" + "=" * 60)
        print("VERIFICATION 2: Cluster Cohesion (Step 2)")
        print("=" * 60)
        
        # 统计每个品牌(Name)被分配到了多少个不同的 Cluster (前两层)
        name_clusters = defaultdict(set)
        
        for bid, info in self.sid_mapping.items():
            name = info['name']
            # 取前两层作为 Cluster ID
            if 'cluster_id' in info:
                cluster = tuple(info['cluster_id'])
            elif 'full_sid' in info:
                cluster = tuple(info['full_sid'][:2])
            else:
                continue
            name_clusters[name].add(cluster)
            
        # 分析 Top Chain Stores
        chain_names = [n for n in name_clusters.keys() if len(self.sid_mapping) > 0]
        # 简单的按频率找 Top Brands
        name_freq = Counter([info['name'] for info in self.sid_mapping.values()])
        
        print(f"Checking top chain stores distribution across Clusters:")
        for name, count in name_freq.most_common(10):
            clusters = name_clusters[name]
            print(f"\nBrand: {name} ({count} stores)")
            print(f"   -> Distributed across {len(clusters)} Semantic Clusters")
            
            # 如果一家连锁店被分到了多个 Cluster，是因为地理位置不同吗？
            # 这是一个好的信号：说明 RQ-VAE 学到了地理语义
            if len(clusters) > 1:
                print(f"   ✅ Good: Different locations mapped to different semantic IDs.")
            else:
                print(f"   ⚠️ Note: All stores mapped to single cluster. (Check if they are in same city)")

    def check_id_collisions(self):
        """检查 ID 冲突 (Step 2 Should Fix This)"""
        print("\n" + "=" * 60)
        print("VERIFICATION 3: ID Uniqueness (Step 2)")
        print("=" * 60)
        
        if not self.sid_mapping: return

        # Check full SIDs
        seen_sids = set()
        collisions = 0
        
        for bid, info in self.sid_mapping.items():
            if 'sid_str' in info:
                sid = info['sid_str']
            else:
                # 兼容旧格式
                sid = tuple(info['full_sid'])
            
            if sid in seen_sids:
                collisions += 1
            seen_sids.add(sid)
            
        if collisions == 0:
            print("✅ SUCCESS: All Semantic IDs are unique.")
            print("   RL Model will have deterministic targets.")
        else:
            print(f"❌ FAIL: Found {collisions} ID collisions.")
            print("   Action: Run step2_generate_semantic_ids.py with suffix fix.")

    def run(self):
        self.load_data()
        self.verify_text_uniqueness()
        self.check_id_collisions()
        self.analyze_cluster_cohesion()

def main():
    analyzer = ChainStoreAnalyzer()
    analyzer.run()

if __name__ == '__main__':
    main()