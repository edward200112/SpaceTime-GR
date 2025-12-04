"""
Step 3: 用户序列构建 (User Sequence Construction) - Optimized

优化点：
1. 完整保留 Target 的地理坐标 (lat/lon)，用于 RL 阶段计算 Geo-Reward。
2. 兼容 Step 2 生成的 Unique Semantic IDs (带后缀)。
3. 增加 Context 信息 (User Current Location)，辅助 LLM 推理。
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import yaml
import numpy as np

class UserSequenceBuilder:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        
        self.raw_dir = self.data_config['raw_dir']
        self.processed_dir = self.data_config['processed_dir']
        
        # Parameters
        self.min_user_interactions = self.preprocess_config['min_user_interactions']
        self.min_item_interactions = self.preprocess_config['min_item_interactions']
        self.max_window_size = self.preprocess_config['max_window_size']
        self.min_history_length = self.preprocess_config['min_history_length']
        self.enable_longterm_summary = self.preprocess_config['enable_longterm_summary']
        
        # Data
        self.sid_mapping = {}
        self.user_interactions = defaultdict(list)
        self.item_interaction_count = defaultdict(int)
    
    def load_sid_mapping(self):
        """加载 SID 映射 (Optimized for new format)"""
        print("\n=== Loading SID Mapping ===")
        mapping_file = os.path.join(
            self.processed_dir,
            self.data_config['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.sid_mapping = json.load(f)
        
        print(f"Loaded {len(self.sid_mapping)} item SIDs")
    
    def load_reviews(self):
        """加载评论数据"""
        print("\n=== Loading Review Data ===")
        review_file = os.path.join(self.raw_dir, self.data_config['review_file'])
        
        # 预先过滤不存在 mapping 中的 interaction，避免浪费内存
        valid_bids = set(self.sid_mapping.keys())
        
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    review = json.loads(line.strip())
                    
                    user_id = review['user_id']
                    business_id = review['business_id']
                    
                    if business_id not in valid_bids:
                        continue
                    
                    # Parse date
                    date_str = review['date']
                    timestamp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp()
                    
                    self.user_interactions[user_id].append({
                        'business_id': business_id,
                        'timestamp': timestamp,
                        'date': date_str,
                        'stars': review['stars']
                    })
                    
                    self.item_interaction_count[business_id] += 1
                except:
                    continue
        
        print(f"Loaded interactions for {len(self.user_interactions)} users")
    
    def apply_kcore_filter(self):
        """应用 K-core 过滤"""
        print("\n=== Applying K-core Filtering ===")
        
        converged = False
        iteration = 0
        
        while not converged:
            iteration += 1
            converged = True
            
            # Filter users
            users_to_remove = []
            for user_id, interactions in self.user_interactions.items():
                if len(interactions) < self.min_user_interactions:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                for interaction in self.user_interactions[user_id]:
                    self.item_interaction_count[interaction['business_id']] -= 1
                del self.user_interactions[user_id]
                converged = False
            
            # Filter items
            items_to_remove = set([
                item_id for item_id, count in self.item_interaction_count.items()
                if count < self.min_item_interactions
            ])
            
            if items_to_remove:
                for item_id in items_to_remove:
                    del self.item_interaction_count[item_id]
                
                # Update user interactions
                for user_id in list(self.user_interactions.keys()):
                    original_len = len(self.user_interactions[user_id])
                    self.user_interactions[user_id] = [
                        inter for inter in self.user_interactions[user_id]
                        if inter['business_id'] not in items_to_remove
                    ]
                    if len(self.user_interactions[user_id]) < original_len:
                        converged = False
            
            print(f"  Iteration {iteration}: {len(self.user_interactions)} users, "
                  f"{len(self.item_interaction_count)} items")
            
            if converged:
                break
        
        print(f"\nFiltered to {len(self.user_interactions)} users")
    
    def sort_user_interactions(self):
        print("\n=== Sorting User Interactions ===")
        for user_id in self.user_interactions:
            self.user_interactions[user_id] = sorted(
                self.user_interactions[user_id],
                key=lambda x: x['timestamp']
            )
    
    def generate_longterm_summary(self, long_history):
        """生成长期历史摘要"""
        if not long_history or not self.enable_longterm_summary:
            return None
        
        high_rated = [inter for inter in long_history if inter['stars'] >= 4.0]
        if not high_rated:
            return None
        
        category_counts = defaultdict(int)
        cities = set()
        
        for inter in high_rated:
            business_id = inter['business_id']
            if business_id in self.sid_mapping:
                categories = self.sid_mapping[business_id].get('categories', '')
                city = self.sid_mapping[business_id].get('city', 'Unknown')
                
                for cat in categories.split(', '):
                    category_counts[cat.strip()] += 1
                cities.add(city)
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        category_str = ', '.join([cat for cat, _ in top_categories])
        
        summary = f"User previously enjoyed {category_str}"
        # 增加 City 信息到 Summary，帮助 LLM 了解用户常去的区域
        if len(cities) == 1:
            summary += f" mainly in {list(cities)[0]}"
        elif len(cities) > 1:
            summary += f" across multiple cities like {', '.join(list(cities)[:2])}"
        summary += "."
        
        return summary
    
    def get_item_feature(self, business_id):
        """Helper to get enriched item features"""
        if business_id not in self.sid_mapping:
            return None
        
        info = self.sid_mapping[business_id]
        
        return {
            'business_id': business_id,
            'name': info['name'],
            'city': info['city'],
            'categories': info['categories'],
            # Geo Coords (Crucial for RL Reward)
            'latitude': info.get('latitude', 0.0),
            'longitude': info.get('longitude', 0.0),
            # IDs
            'cluster_id': info['cluster_id'],
            'full_sid': info['full_sid'], # <c0, c1, c2, suffix>
            'sid_str': info['sid_str']    # String format for LLM generation
        }

    def build_sequences_with_sliding_window(self):
        print("\n=== Building Sequences with Sliding Window ===")
        
        all_sequences = []
        
        for user_id, interactions in tqdm(self.user_interactions.items(), desc="Building sequences"):
            if len(interactions) < self.min_history_length + 1:
                continue
            
            stride = self.preprocess_config['sliding_stride']
            
            for t in range(self.min_history_length, len(interactions), stride):
                target_inter = interactions[t]
                target_bid = target_inter['business_id']
                
                target_feat = self.get_item_feature(target_bid)
                if not target_feat: continue
                
                # History window
                window_start = max(0, t - self.max_window_size)
                history_window = interactions[window_start:t]
                
                # Long-term summary
                longterm_summary = None
                if window_start > 0 and self.enable_longterm_summary:
                    long_history = interactions[:window_start]
                    longterm_summary = self.generate_longterm_summary(long_history)
                
                # Build history items
                history_items = []
                last_city = "Unknown" # 用于 Context
                
                for inter in history_window:
                    bid = inter['business_id']
                    feat = self.get_item_feature(bid)
                    if not feat: continue
                    
                    feat['stars'] = inter['stars']
                    feat['date'] = inter['date']
                    history_items.append(feat)
                    last_city = feat['city']
                
                if not history_items: continue
                
                # Target enrichment
                target_feat['stars'] = target_inter['stars']
                target_feat['date'] = target_inter['date']
                
                # Construct Sequence
                sequence = {
                    'user_id': user_id,
                    'current_city': last_city, # Context: 用户当前所在的城市
                    'history': history_items,
                    'target': target_feat,     # 包含 latitude/longitude
                    'longterm_summary': longterm_summary,
                    'timestamp': target_inter['timestamp']
                }
                
                all_sequences.append(sequence)
        
        print(f"Generated {len(all_sequences)} sequences")
        return all_sequences
    
    def split_and_save(self, sequences):
        """统一划分并保存逻辑"""
        print("\n=== Splitting and Saving ===")
        
        # Sort generally by time helps, but we split per user usually
        # Here we use Leave-One-Out or Ratio based on User
        user_seqs_map = defaultdict(list)
        for seq in sequences:
            user_seqs_map[seq['user_id']].append(seq)
            
        train_seqs, valid_seqs, test_seqs = [], [], []
        test_ratio = self.preprocess_config.get('test_ratio', 0.1)
        valid_ratio = self.preprocess_config.get('valid_ratio', 0.1)
        
        for uid, seqs in user_seqs_map.items():
            seqs.sort(key=lambda x: x['timestamp'])
            n = len(seqs)
            
            # 策略：保证至少有一个在训练集
            if n < 2:
                train_seqs.extend(seqs)
                continue
                
            n_test = max(1, int(n * test_ratio))
            # 保证训练集不为空
            if n - n_test < 1: n_test = 0
            
            n_valid = max(1, int(n * valid_ratio))
            if n - n_test - n_valid < 1: n_valid = 0
            
            # Slice
            if n_test > 0:
                test_seqs.extend(seqs[-n_test:])
                remainder = seqs[:-n_test]
            else:
                remainder = seqs
                
            if n_valid > 0:
                valid_seqs.extend(remainder[-n_valid:])
                train_seqs.extend(remainder[:-n_valid])
            else:
                train_seqs.extend(remainder)
                
        # Save
        def save_jsonl(data, name):
            path = os.path.join(self.processed_dir, name)
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved {len(data)} to {name}")

        save_jsonl(train_seqs, 'train.jsonl')
        save_jsonl(valid_seqs, 'valid.jsonl')
        save_jsonl(test_seqs, 'test.jsonl')
        
        # Save Metadata for Dataset Loader
        meta = {
            'train_size': len(train_seqs),
            'valid_size': len(valid_seqs),
            'test_size': len(test_seqs),
            # 记录数据集中包含的城市，方便后续分析
            'cities': list(set([s['current_city'] for s in train_seqs]))
        }
        with open(os.path.join(self.processed_dir, 'dataset_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def run(self):
        print("\n" + "="*60)
        print("Step 3: Building User Sequences (Optimized)")
        print("="*60)
        
        self.load_sid_mapping()
        self.load_reviews()
        self.apply_kcore_filter()
        self.sort_user_interactions()
        
        sequences = self.build_sequences_with_sliding_window()
        self.split_and_save(sequences)
        
        print("\n✓ Step 3 completed successfully!")

def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    builder = UserSequenceBuilder(config)
    builder.run()

if __name__ == '__main__':
    main()