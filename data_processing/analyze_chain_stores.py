"""
分析 Yelp 商家数据中的连锁店（Chain Stores）问题

目的：
1. 统计重复名称的商家数量（连锁店）
2. 分析文本描述的重复度
3. 评估连锁店对碰撞率的影响
4. 为改进 text_description 提供依据
"""

import os
import json
import yaml
from collections import Counter, defaultdict
from tqdm import tqdm
import hashlib


class ChainStoreAnalyzer:
    def __init__(self, config_path='./config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config['data']['processed_dir']
        self.profiles_file = os.path.join(
            self.processed_dir,
            self.config['data']['item_profile_file']  # 修正：item_profile_file 不是 item_profiles_file
        )
        
        # 数据结构
        self.name_counter = Counter()
        self.category_counter = Counter()
        self.text_description_counter = Counter()
        self.name_to_businesses = defaultdict(list)
        self.text_to_businesses = defaultdict(list)
        self.all_businesses = []
    
    def load_profiles(self):
        """加载商家 profiles"""
        print("\n=== Loading Business Profiles ===")
        
        with open(self.profiles_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading profiles"):
                profile = json.loads(line.strip())
                self.all_businesses.append(profile)
                
                # 统计名称
                name = profile['name']
                self.name_counter[name] += 1
                self.name_to_businesses[name].append(profile)
                
                # 统计 raw_text（实际字段名）
                text_desc = profile['raw_text']  # 修正：字段名是 raw_text 不是 text_description
                self.text_description_counter[text_desc] += 1
                self.text_to_businesses[text_desc].append(profile)
                
                # 统计类别
                categories = profile.get('categories', '')
                self.category_counter[categories] += 1
        
        print(f"Loaded {len(self.all_businesses)} businesses")
    
    def analyze_chain_stores(self):
        """分析连锁店"""
        print("\n" + "=" * 70)
        print("CHAIN STORES ANALYSIS")
        print("=" * 70)
        
        # 1. 名称重复分析
        print("\n### 1. Business Name Duplication ###")
        duplicate_names = {name: count for name, count in self.name_counter.items() if count > 1}
        
        print(f"\nTotal unique names: {len(self.name_counter)}")
        print(f"Duplicate names: {len(duplicate_names)} ({len(duplicate_names)/len(self.name_counter)*100:.2f}%)")
        print(f"Total businesses in chains: {sum(duplicate_names.values())}")
        print(f"Chain store rate: {sum(duplicate_names.values())/len(self.all_businesses)*100:.2f}%")
        
        # Top 20 连锁店
        print("\n### Top 20 Chain Stores (by count) ###")
        for i, (name, count) in enumerate(self.name_counter.most_common(20), 1):
            # 获取该连锁的城市分布
            businesses = self.name_to_businesses[name]
            cities = Counter([b['city'] for b in businesses])
            top_cities = ', '.join([f"{city}({cnt})" for city, cnt in cities.most_common(3)])
            
            print(f"{i:2d}. {name:40s} | {count:4d} stores | Cities: {top_cities}")
    
    def analyze_text_duplication(self):
        """分析 text_description 重复"""
        print("\n" + "=" * 70)
        print("TEXT DESCRIPTION DUPLICATION ANALYSIS")
        print("=" * 70)
        
        # 统计完全相同的 text_description
        duplicate_texts = {text: count for text, count in self.text_description_counter.items() if count > 1}
        
        print(f"\nTotal unique text descriptions: {len(self.text_description_counter)}")
        print(f"Duplicate text descriptions: {len(duplicate_texts)} ({len(duplicate_texts)/len(self.text_description_counter)*100:.2f}%)")
        print(f"Total businesses with duplicate texts: {sum(duplicate_texts.values())}")
        print(f"Duplication rate: {sum(duplicate_texts.values())/len(self.all_businesses)*100:.2f}%")
        
        # Top 20 重复的文本
        print("\n### Top 20 Duplicate Text Descriptions ###")
        for i, (text, count) in enumerate(self.text_description_counter.most_common(20), 1):
            if count == 1:
                break
            
            # 截断文本显示
            text_preview = text[:80] + '...' if len(text) > 80 else text
            businesses = self.text_to_businesses[text]
            
            # 获取这些商家的名称分布
            names = Counter([b['name'] for b in businesses])
            top_names = ', '.join([f"{name}({cnt})" for name, cnt in names.most_common(2)])
            
            print(f"\n{i:2d}. Count: {count:4d} | Names: {top_names}")
            print(f"    Text: {text_preview}")
    
    def analyze_collision_correlation(self):
        """分析连锁店与碰撞率的关联"""
        print("\n" + "=" * 70)
        print("COLLISION CORRELATION ANALYSIS")
        print("=" * 70)
        
        # 加载 SID mapping
        sid_mapping_file = os.path.join(
            self.processed_dir,
            self.config['data']['sid_mapping_file']
        )
        
        if not os.path.exists(sid_mapping_file):
            print("⚠️  SID mapping file not found. Skipping collision analysis.")
            return
        
        with open(sid_mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        # 统计连锁店的碰撞情况
        chain_collisions = 0
        non_chain_collisions = 0
        
        # 按 full_sid 分组
        sid_to_businesses = defaultdict(list)
        for business_id, info in sid_mapping.items():
            sid_tuple = tuple(info['full_sid'])
            sid_to_businesses[sid_tuple].append(business_id)
        
        # 检查碰撞的商家
        for sid_tuple, business_ids in sid_to_businesses.items():
            if len(business_ids) > 1:  # 碰撞
                # 检查这些商家是否是连锁店
                names = []
                for bid in business_ids:
                    if bid in sid_mapping:
                        names.append(sid_mapping[bid]['name'])
                
                # 如果名称相同，是连锁店碰撞
                if len(set(names)) == 1:
                    chain_collisions += len(business_ids)
                else:
                    non_chain_collisions += len(business_ids)
        
        total_collisions = chain_collisions + non_chain_collisions
        
        print(f"\nTotal collided businesses: {total_collisions}")
        print(f"Chain store collisions: {chain_collisions} ({chain_collisions/total_collisions*100:.2f}%)")
        print(f"Non-chain collisions: {non_chain_collisions} ({non_chain_collisions/total_collisions*100:.2f}%)")
        
        print("\n### Insight ###")
        if chain_collisions > total_collisions * 0.3:
            print("🔴 连锁店碰撞占比 > 30%，这是碰撞率高的主要原因！")
            print("   建议：在 text_description 中添加地址或 ID 哈希值")
        elif chain_collisions > total_collisions * 0.1:
            print("🟡 连锁店碰撞占比 10-30%，有一定影响")
            print("   建议：考虑添加地理信息或唯一标识")
        else:
            print("🟢 连锁店碰撞占比 < 10%，不是主要问题")
            print("   建议：专注于提高 RQ-VAE 训练质量")
    
    def analyze_category_patterns(self):
        """分析类别分布"""
        print("\n" + "=" * 70)
        print("CATEGORY DISTRIBUTION")
        print("=" * 70)
        
        print("\n### Top 20 Most Common Categories ###")
        for i, (category, count) in enumerate(self.category_counter.most_common(20), 1):
            percentage = count / len(self.all_businesses) * 100
            print(f"{i:2d}. {category:50s} | {count:5d} ({percentage:5.2f}%)")
    
    def generate_improvement_suggestions(self):
        """生成改进建议"""
        print("\n" + "=" * 70)
        print("IMPROVEMENT SUGGESTIONS")
        print("=" * 70)
        
        # 统计需要改进的商家数量
        duplicate_text_count = sum(count for text, count in self.text_description_counter.items() if count > 1)
        
        print(f"\n需要优化的商家数量: {duplicate_text_count} ({duplicate_text_count/len(self.all_businesses)*100:.2f}%)")
        
        print("\n### 建议 1：修改 Step 1 的 text_description 构建 ###")
        print("当前代码位置: data_processing/step1_build_item_profile.py")
        print("\n修改方案：")
        print("```python")
        print("# 原来的代码：")
        print("text_description = f\"Name: {name}. Category: {categories}.\"")
        print()
        print("# 改进后的代码：")
        print("# 1. 添加城市和地址")
        print("text_description = f\"Name: {name}. City: {city}. Address: {address}. Category: {categories}.\"")
        print()
        print("# 2. 或者添加 business_id 的哈希后缀")
        print("import hashlib")
        print("id_hash = hashlib.md5(business_id.encode()).hexdigest()[:6]")
        print("text_description = f\"Name: {name}. ID: {id_hash}. Category: {categories}.\"")
        print("```")
        
        print("\n### 建议 2：增大 Codebook 容量 ###")
        print("修改 config/config.yaml：")
        print("```yaml")
        print("rqvae:")
        print("  num_emb_list: [128, 128, 128]  # 从 [64, 64, 64] 增加")
        print("```")
        
        print("\n### 建议 3：继续训练 ###")
        print("让当前训练完成 5000 epochs，观察 collision rate 是否下降")
    
    def export_chain_analysis(self):
        """导出详细分析结果"""
        output_file = './data/chain_store_analysis.json'
        
        # 导出 Top 50 连锁店的详细信息
        top_chains = []
        for name, count in self.name_counter.most_common(50):
            if count == 1:
                break
            
            businesses = self.name_to_businesses[name]
            cities = Counter([b['city'] for b in businesses])
            categories = Counter([b.get('categories', '') for b in businesses])
            
            top_chains.append({
                'name': name,
                'count': count,
                'cities': dict(cities.most_common(10)),
                'categories': dict(categories.most_common(3)),
                'sample_business_ids': [b['business_id'] for b in businesses[:5]]
            })
        
        # 导出重复文本的详细信息
        top_duplicate_texts = []
        for text, count in self.text_description_counter.most_common(50):
            if count == 1:
                break
            
            businesses = self.text_to_businesses[text]
            names = Counter([b['name'] for b in businesses])
            
            top_duplicate_texts.append({
                'text': text,
                'count': count,
                'top_names': dict(names.most_common(5)),
                'sample_business_ids': [b['business_id'] for b in businesses[:5]]
            })
        
        analysis_result = {
            'summary': {
                'total_businesses': len(self.all_businesses),
                'unique_names': len(self.name_counter),
                'duplicate_names': len([c for c in self.name_counter.values() if c > 1]),
                'unique_texts': len(self.text_description_counter),
                'duplicate_texts': len([c for c in self.text_description_counter.values() if c > 1]),
                'chain_store_rate': sum(c for c in self.name_counter.values() if c > 1) / len(self.all_businesses)
            },
            'top_chains': top_chains,
            'top_duplicate_texts': top_duplicate_texts
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Detailed analysis exported to: {output_file}")
    
    def run(self):
        """运行完整分析"""
        self.load_profiles()
        self.analyze_chain_stores()
        self.analyze_text_duplication()
        self.analyze_category_patterns()
        self.analyze_collision_correlation()
        self.generate_improvement_suggestions()
        self.export_chain_analysis()
        
        print("\n" + "=" * 70)
        print("✅ Analysis Completed!")
        print("=" * 70)


def main():
    analyzer = ChainStoreAnalyzer()
    analyzer.run()


if __name__ == '__main__':
    main()
