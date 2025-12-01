"""
Step 3: 用户序列构建 (User Sequence Construction)

目标：清洗原始日志，生成标准的时间序列数据（带滑动窗口）
输入：review.json, sid_mapping.json
输出：user_sequences.jsonl

核心逻辑：
1. K-core 过滤（>=5 交互）
2. 按时间排序
3. 滑动窗口（最近 10-15 个）+ 长期语义摘要
4. 数据增强（训练集）
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
        """加载 SID 映射"""
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
        
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading reviews"):
                review = json.loads(line.strip())
                
                user_id = review['user_id']
                business_id = review['business_id']
                
                # Skip if business not in SID mapping
                if business_id not in self.sid_mapping:
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
        
        print(f"Loaded interactions for {len(self.user_interactions)} users")
    
    def apply_kcore_filter(self):
        """应用 K-core 过滤"""
        print("\n=== Applying K-core Filtering ===")
        
        # Iterative K-core filtering
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
                # Update item counts
                for interaction in self.user_interactions[user_id]:
                    self.item_interaction_count[interaction['business_id']] -= 1
                del self.user_interactions[user_id]
                converged = False
            
            # Filter items
            items_to_remove = [
                item_id for item_id, count in self.item_interaction_count.items()
                if count < self.min_item_interactions
            ]
            
            for item_id in items_to_remove:
                del self.item_interaction_count[item_id]
                converged = False
            
            # Remove interactions with filtered items
            for user_id in list(self.user_interactions.keys()):
                original_count = len(self.user_interactions[user_id])
                self.user_interactions[user_id] = [
                    inter for inter in self.user_interactions[user_id]
                    if inter['business_id'] in self.item_interaction_count
                ]
                if len(self.user_interactions[user_id]) < original_count:
                    converged = False
            
            print(f"  Iteration {iteration}: {len(self.user_interactions)} users, "
                  f"{len(self.item_interaction_count)} items")
            
            if converged:
                break
        
        print(f"\nFiltered to {len(self.user_interactions)} users and "
              f"{len(self.item_interaction_count)} items")
    
    def sort_user_interactions(self):
        """按时间排序每个用户的交互"""
        print("\n=== Sorting User Interactions ===")
        
        for user_id in self.user_interactions:
            self.user_interactions[user_id] = sorted(
                self.user_interactions[user_id],
                key=lambda x: x['timestamp']
            )
    
    def generate_longterm_summary(self, long_history):
        """生成长期历史的文本摘要"""
        if not long_history or not self.enable_longterm_summary:
            return None
        
        # 统计高分项的 categories
        high_rated = [inter for inter in long_history if inter['stars'] >= 4.0]
        
        if not high_rated:
            return None
        
        # 提取 categories
        category_counts = defaultdict(int)
        cities = set()
        
        for inter in high_rated:
            business_id = inter['business_id']
            if business_id in self.sid_mapping:
                categories = self.sid_mapping[business_id]['categories']
                city = self.sid_mapping[business_id]['city']
                
                # 分割多个类别
                for cat in categories.split(', '):
                    category_counts[cat.strip()] += 1
                cities.add(city)
        
        # 生成摘要
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        category_str = ', '.join([cat for cat, _ in top_categories])
        
        summary = f"User previously enjoyed {category_str}"
        if len(cities) > 1:
            summary += f" across multiple cities"
        summary += "."
        
        return summary
    
    def build_sequences_with_sliding_window(self):
        """使用滑动窗口构建序列"""
        print("\n=== Building Sequences with Sliding Window ===")
        
        all_sequences = []
        
        for user_id, interactions in tqdm(self.user_interactions.items(), desc="Building sequences"):
            if len(interactions) < self.min_history_length + 1:
                continue
            
            # 为该用户生成多个样本（滑动窗口）
            stride = self.preprocess_config['sliding_stride']
            
            for t in range(self.min_history_length, len(interactions), stride):
                # Target
                target_inter = interactions[t]
                target_business_id = target_inter['business_id']
                
                if target_business_id not in self.sid_mapping:
                    continue
                
                # History window
                window_start = max(0, t - self.max_window_size)
                history_window = interactions[window_start:t]
                
                # Long-term summary（如果有被截断的历史）
                longterm_summary = None
                if window_start > 0 and self.enable_longterm_summary:
                    long_history = interactions[:window_start]
                    longterm_summary = self.generate_longterm_summary(long_history)
                
                # Build history items
                history_items = []
                for inter in history_window:
                    bid = inter['business_id']
                    if bid not in self.sid_mapping:
                        continue
                    
                    history_items.append({
                        'business_id': bid,
                        'name': self.sid_mapping[bid]['name'],
                        'city': self.sid_mapping[bid]['city'],
                        'categories': self.sid_mapping[bid]['categories'],
                        'cluster_id': self.sid_mapping[bid]['cluster_id'],
                        'cluster_str': self.sid_mapping[bid]['cluster_str'],
                        'full_sid': self.sid_mapping[bid]['full_sid'],
                        'stars': inter['stars'],
                        'date': inter['date']
                    })
                
                if not history_items:
                    continue
                
                # Build target
                target = {
                    'business_id': target_business_id,
                    'name': self.sid_mapping[target_business_id]['name'],
                    'city': self.sid_mapping[target_business_id]['city'],
                    'categories': self.sid_mapping[target_business_id]['categories'],
                    'cluster_id': self.sid_mapping[target_business_id]['cluster_id'],
                    'cluster_str': self.sid_mapping[target_business_id]['cluster_str'],
                    'full_sid': self.sid_mapping[target_business_id]['full_sid'],
                    'stars': target_inter['stars'],
                    'date': target_inter['date']
                }
                
                sequence = {
                    'user_id': user_id,
                    'history': history_items,
                    'target': target,
                    'longterm_summary': longterm_summary,
                    'timestamp': target_inter['timestamp']
                }
                
                all_sequences.append(sequence)
        
        print(f"Generated {len(all_sequences)} sequences")
        return all_sequences
    
    def split_sequences(self, sequences):
        """划分训练/验证/测试集"""
        print("\n=== Splitting Sequences ===")
        
        # 按用户分组
        user_sequences = defaultdict(list)
        for seq in sequences:
            user_sequences[seq['user_id']].append(seq)
        
        train_seqs = []
        valid_seqs = []
        test_seqs = []
        
        test_ratio = self.preprocess_config['test_ratio']
        valid_ratio = self.preprocess_config['valid_ratio']
        
        for user_id, seqs in user_sequences.items():
            # 按时间排序
            seqs = sorted(seqs, key=lambda x: x['timestamp'])
            n = len(seqs)
            
            if n < 3:
                # 太少，全部用于训练
                train_seqs.extend(seqs)
            else:
                # 最后的用于测试/验证
                n_test = max(1, int(n * test_ratio))
                n_valid = max(1, int(n * valid_ratio))
                
                test_seqs.extend(seqs[-n_test:])
                valid_seqs.extend(seqs[-(n_test + n_valid):-n_test])
                train_seqs.extend(seqs[:-(n_test + n_valid)])
        
        print(f"Train: {len(train_seqs)} sequences")
        print(f"Valid: {len(valid_seqs)} sequences")
        print(f"Test: {len(test_seqs)} sequences")
        
        return train_seqs, valid_seqs, test_seqs
    
    def save_sequences(self, train_seqs, valid_seqs, test_seqs):
        """保存序列"""
        print("\n=== Saving Sequences ===")
        
        output_file = os.path.join(
            self.processed_dir,
            self.data_config['user_sequences_file']
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for seq in train_seqs + valid_seqs + test_seqs:
                f.write(json.dumps(seq, ensure_ascii=False) + '\n')
        
        print(f"Saved all sequences to {output_file}")
        
        # Also save split info
        split_file = output_file.replace('.jsonl', '_split.json')
        split_info = {
            'train_count': len(train_seqs),
            'valid_count': len(valid_seqs),
            'test_count': len(test_seqs),
            'total_count': len(train_seqs) + len(valid_seqs) + len(test_seqs)
        }
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Saved split info to {split_file}")
    
    def run(self):
        """执行完整流程"""
        print("\n" + "="*60)
        print("Step 3: Building User Sequences")
        print("="*60)
        
        self.load_sid_mapping()
        self.load_reviews()
        self.apply_kcore_filter()
        self.sort_user_interactions()
        
        sequences = self.build_sequences_with_sliding_window()
        train_seqs, valid_seqs, test_seqs = self.split_sequences(sequences)
        self.save_sequences(train_seqs, valid_seqs, test_seqs)
        
        print("\n✓ Step 3 completed successfully!")


def main():
    # Load config
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Build user sequences
    builder = UserSequenceBuilder(config)
    builder.run()


if __name__ == '__main__':
    main()
