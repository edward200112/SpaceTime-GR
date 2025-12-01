"""
Step 4: 多任务 Prompt 构造 (Multi-Task Prompt Construction)

目标：构造三种训练任务的 Prompt
输入：user_sequences.jsonl
输出：train_prompts.jsonl, valid_prompts.jsonl, test_prompts.jsonl

任务类型：
A. 序列推荐（主任务）- 预测下一个 Cluster ID
B. 用户偏好摘要 - 生成文本描述
C. 语义 ID 对齐 - Text ↔ ID 双向映射
"""

import json
import os
import random
from tqdm import tqdm
import yaml
from collections import defaultdict


class PromptConstructor:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.prompt_config = config['prompt']
        
        self.processed_dir = self.data_config['processed_dir']
        
        # Task weights
        self.task_weights = self.prompt_config['task_weights']
        
        # Prompt templates
        self.task_a_template = self.prompt_config['task_a_template']
        self.task_b_template = self.prompt_config['task_b_template']
        self.task_c1_template = self.prompt_config['task_c1_template']
        self.task_c2_template = self.prompt_config['task_c2_template']
        
        # History formats
        self.history_format = self.prompt_config['history_item_format']
        self.history_format_with_rating = self.prompt_config['history_item_format_with_rating']
        
        # Data
        self.sequences = []
        self.sid_mapping = {}
        self.cluster_to_businesses = defaultdict(list)
    
    def load_sequences(self):
        """加载用户序列"""
        print("\n=== Loading User Sequences ===")
        
        sequence_file = os.path.join(
            self.processed_dir,
            self.data_config['user_sequences_file']
        )
        
        with open(sequence_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading sequences"):
                self.sequences.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.sequences)} sequences")
        
        # Load split info
        split_file = sequence_file.replace('.jsonl', '_split.json')
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        train_count = split_info['train_count']
        valid_count = split_info['valid_count']
        test_count = split_info['test_count']
        
        self.train_sequences = self.sequences[:train_count]
        self.valid_sequences = self.sequences[train_count:train_count+valid_count]
        self.test_sequences = self.sequences[train_count+valid_count:]
        
        print(f"Train: {len(self.train_sequences)}")
        print(f"Valid: {len(self.valid_sequences)}")
        print(f"Test: {len(self.test_sequences)}")
    
    def load_sid_mapping(self):
        """加载 SID 映射"""
        print("\n=== Loading SID Mapping ===")
        
        mapping_file = os.path.join(
            self.processed_dir,
            self.data_config['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.sid_mapping = json.load(f)
        
        # Build cluster to businesses mapping
        for business_id, info in self.sid_mapping.items():
            cluster_tuple = tuple(info['cluster_id'])
            self.cluster_to_businesses[cluster_tuple].append({
                'business_id': business_id,
                'name': info['name'],
                'categories': info['categories'],
                'city': info['city']
            })
        
        print(f"Loaded {len(self.sid_mapping)} items")
        print(f"Found {len(self.cluster_to_businesses)} unique clusters")
    
    def format_history(self, history, include_rating=False):
        """格式化历史序列"""
        history_lines = []
        
        for idx, item in enumerate(history, 1):
            if include_rating:
                line = self.history_format_with_rating.format(
                    name=item['name'],
                    category=item['categories'].split(',')[0].strip(),
                    rating=item['stars']
                )
            else:
                line = self.history_format.format(
                    name=item['name'],
                    category=item['categories'].split(',')[0].strip(),
                    cluster_id=item['cluster_str']
                )
            history_lines.append(f"{idx}. {line}")
        
        return '\n'.join(history_lines)
    
    def create_task_a_prompt(self, sequence):
        """任务 A：序列推荐"""
        # Long-term summary
        longterm_text = ""
        if sequence.get('longterm_summary'):
            longterm_text = f"User Profile: {sequence['longterm_summary']}\n"
        
        # Format history
        history_text = self.format_history(sequence['history'], include_rating=False)
        
        # Fill template
        instruction = self.task_a_template.format(
            longterm_summary=longterm_text,
            history=history_text
        ).strip()
        
        # Output: Cluster ID
        output = sequence['target']['cluster_str']
        
        return {
            'task': 'task_a_recommendation',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {
                'user_id': sequence['user_id'],
                'target_business_id': sequence['target']['business_id'],
                'target_name': sequence['target']['name']
            }
        }
    
    def create_task_b_prompt(self, sequence):
        """任务 B：用户偏好摘要"""
        # Format history with ratings
        history_text = self.format_history(sequence['history'], include_rating=True)
        
        instruction = self.task_b_template.format(history=history_text).strip()
        
        # Generate preference summary
        output = self.generate_preference_summary(sequence['history'])
        
        return {
            'task': 'task_b_preference_summary',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {
                'user_id': sequence['user_id']
            }
        }
    
    def generate_preference_summary(self, history):
        """生成用户偏好摘要（规则）"""
        # 统计高分项的类别
        high_rated = [item for item in history if item['stars'] >= 4.0]
        
        if not high_rated:
            return "User has diverse tastes."
        
        # 提取主要类别
        category_counts = defaultdict(int)
        cities = set()
        
        for item in high_rated:
            categories = item['categories'].split(',')
            for cat in categories:
                category_counts[cat.strip()] += 1
            cities.add(item['city'])
        
        # 找到最常出现的类别
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if not top_categories:
            return "User has diverse tastes."
        
        cat_names = [cat for cat, _ in top_categories]
        
        # 构造摘要
        if len(cat_names) == 1:
            summary = f"The user enjoys {cat_names[0]}"
        elif len(cat_names) == 2:
            summary = f"The user enjoys {cat_names[0]} and {cat_names[1]}"
        else:
            summary = f"The user enjoys {', '.join(cat_names[:-1])}, and {cat_names[-1]}"
        
        if len(cities) > 2:
            summary += " across multiple cities"
        
        summary += "."
        
        return summary
    
    def create_task_c1_prompt(self):
        """任务 C1：Text -> ID"""
        # 随机选择一个商家
        business_id = random.choice(list(self.sid_mapping.keys()))
        info = self.sid_mapping[business_id]
        
        # 构造查询文本
        query = f"{info['categories']} in {info['city']}"
        
        instruction = self.task_c1_template.format(query=query).strip()
        output = info['cluster_str']
        
        return {
            'task': 'task_c_sid_alignment',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {
                'business_id': business_id,
                'name': info['name']
            }
        }
    
    def create_task_c2_prompt(self):
        """任务 C2：ID -> Text"""
        # 随机选择一个 cluster
        cluster_tuple = random.choice(list(self.cluster_to_businesses.keys()))
        businesses = self.cluster_to_businesses[cluster_tuple]
        
        cluster_str = '<' + ', '.join(map(str, cluster_tuple)) + '>'
        
        instruction = self.task_c2_template.format(cluster_id=cluster_str).strip()
        
        # 生成描述
        categories = set()
        cities = set()
        for biz in businesses:
            for cat in biz['categories'].split(','):
                categories.add(cat.strip())
            cities.add(biz['city'])
        
        output = f"Businesses in categories: {', '.join(list(categories)[:3])}"
        if len(cities) > 1:
            output += f", typically found in {', '.join(list(cities)[:3])}"
        output += "."
        
        return {
            'task': 'task_c_sid_alignment',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {
                'cluster_id': list(cluster_tuple)
            }
        }
    
    def construct_prompts_for_split(self, sequences, split_name):
        """为某个数据集构造 Prompt"""
        print(f"\n=== Constructing Prompts for {split_name} ===")
        
        prompts = []
        
        # 任务 A：序列推荐（所有序列）
        for seq in tqdm(sequences, desc=f"{split_name} Task A"):
            prompts.append(self.create_task_a_prompt(seq))
        
        # 如果是训练集，添加任务 B 和 C
        if split_name == 'train':
            # 任务 B：偏好摘要（按权重采样）
            n_task_b = int(len(sequences) * self.task_weights['task_b_preference_summary'] / 
                          self.task_weights['task_a_recommendation'])
            sampled_seqs = random.sample(sequences, min(n_task_b, len(sequences)))
            
            for seq in tqdm(sampled_seqs, desc=f"{split_name} Task B"):
                prompts.append(self.create_task_b_prompt(seq))
            
            # 任务 C：ID 对齐（按权重采样）
            n_task_c = int(len(sequences) * self.task_weights['task_c_sid_alignment'] / 
                          self.task_weights['task_a_recommendation'])
            
            for _ in tqdm(range(n_task_c), desc=f"{split_name} Task C"):
                if random.random() < 0.5:
                    prompts.append(self.create_task_c1_prompt())
                else:
                    prompts.append(self.create_task_c2_prompt())
        
        # Shuffle
        random.shuffle(prompts)
        
        print(f"Generated {len(prompts)} prompts for {split_name}")
        
        # Count by task
        task_counts = defaultdict(int)
        for p in prompts:
            task_counts[p['task']] += 1
        for task, count in task_counts.items():
            print(f"  {task}: {count}")
        
        return prompts
    
    def save_prompts(self, train_prompts, valid_prompts, test_prompts):
        """保存 Prompt 数据"""
        print("\n=== Saving Prompts ===")
        
        splits = {
            'train': train_prompts,
            'valid': valid_prompts,
            'test': test_prompts
        }
        
        for split_name, prompts in splits.items():
            if split_name == 'train':
                output_file = os.path.join(self.processed_dir, self.data_config['train_prompts_file'])
            elif split_name == 'valid':
                output_file = os.path.join(self.processed_dir, self.data_config['valid_prompts_file'])
            else:
                output_file = os.path.join(self.processed_dir, self.data_config['test_prompts_file'])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(prompts)} prompts to {output_file}")
    
    def run(self):
        """执行完整流程"""
        print("\n" + "="*60)
        print("Step 4: Constructing Multi-Task Prompts")
        print("="*60)
        
        self.load_sequences()
        self.load_sid_mapping()
        
        train_prompts = self.construct_prompts_for_split(self.train_sequences, 'train')
        valid_prompts = self.construct_prompts_for_split(self.valid_sequences, 'valid')
        test_prompts = self.construct_prompts_for_split(self.test_sequences, 'test')
        
        self.save_prompts(train_prompts, valid_prompts, test_prompts)
        
        print("\n✓ Step 4 completed successfully!")


def main():
    # Set random seed
    random.seed(42)
    
    # Load config
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Construct prompts
    constructor = PromptConstructor(config)
    constructor.run()


if __name__ == '__main__':
    main()
