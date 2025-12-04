"""
Step 4: 多任务 Prompt 构造 (Multi-Task Prompt Construction) - Optimized

优化点：
1. 融入地理 Context (User Current City) 到 Task A Prompt 中，辅助位置感知推荐。
2. 适配 Step 2/3 生成的 Unique ID (<c0, c1, c2, suffix>)。
3. 强化 Task C (Alignment) 的地理约束，确保 Text->ID 任务可解。
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
        
        # Templates (Updated for Geo-Context)
        # 建议在 config.yaml 中更新模板，或者在此处硬编码覆盖
        self.task_a_template = (
            "User is currently in {current_city}. "
            "Based on the visit history below, predict the next place to visit.\n"
            "{longterm_summary}"
            "User History:\n{history}\n"
            "Response:"
        )
        
        self.task_b_template = "Summarize the user's preferences based on the following visit history:\n{history}\nResponse:"
        
        # Text -> ID
        self.task_c1_template = "What is the Semantic ID for \"{query}\"?\nResponse:"
        
        # ID -> Text
        self.task_c2_template = "Describe the semantic meaning of Cluster ID {cluster_id}.\nResponse:"
        
        # Data
        self.train_sequences = []
        self.valid_sequences = []
        self.test_sequences = []
        self.sid_mapping = {}
        self.cluster_to_businesses = defaultdict(list)
    
    def load_sequences(self):
        """加载分好类的数据集"""
        print("\n=== Loading Sequences ===")
        
        def load_split(name):
            path = os.path.join(self.processed_dir, name)
            data = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
            return data

        self.train_sequences = load_split('train.jsonl')
        self.valid_sequences = load_split('valid.jsonl')
        self.test_sequences = load_split('test.jsonl')
        
        print(f"Train: {len(self.train_sequences)}")
        print(f"Valid: {len(self.valid_sequences)}")
        print(f"Test: {len(self.test_sequences)}")
    
    def load_sid_mapping(self):
        print("\n=== Loading SID Mapping ===")
        mapping_file = os.path.join(self.processed_dir, self.data_config['sid_mapping_file'])
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.sid_mapping = json.load(f)
        
        # Build cluster mapping (using only c0, c1 for semantic grouping)
        for business_id, info in self.sid_mapping.items():
            # 这里我们用前两层作为语义聚类 ID
            cluster_tuple = tuple(info['cluster_id']) # [c0, c1]
            self.cluster_to_businesses[cluster_tuple].append(info)
        
        print(f"Loaded {len(self.sid_mapping)} items")
    
    def format_history(self, history, include_rating=False):
        lines = []
        for idx, item in enumerate(history, 1):
            name = item['name']
            cat = item['categories'].split(',')[0].strip()
            # 可以在历史中也加入 city，增强 LLM 对移动轨迹的理解
            city = item['city']
            
            if include_rating:
                line = f"[{name}] ({cat} in {city}) - {item['stars']} stars"
            else:
                # 训练 Task A 时，历史可以用 ID 表示，也可以用 Text 表示
                # 为了让 LLM 理解语义，这里使用 Text 描述
                line = f"[{name}] ({cat} in {city})"
            
            lines.append(f"{idx}. {line}")
        return '\n'.join(lines)
    
    def create_task_a_prompt(self, sequence):
        """Task A: Recommendation (Next Item Prediction)"""
        longterm_text = ""
        if sequence.get('longterm_summary'):
            longterm_text = f"User Profile: {sequence['longterm_summary']}\n"
        
        history_text = self.format_history(sequence['history'])
        
        # 关键优化：加入 current_city
        current_city = sequence.get('current_city', 'Unknown City')
        
        instruction = self.task_a_template.format(
            current_city=current_city,
            longterm_summary=longterm_text,
            history=history_text
        ).strip()
        
        # Output: 使用 Step 2 生成的唯一 ID (sid_str)
        # e.g., "<12, 45, 88, 0>"
        output = sequence['target']['sid_str']
        
        return {
            'task': 'task_a_recommendation',
            'instruction': instruction,
            'input': '',
            'output': output,
            # 保留 Metadata 供评估和 RL 使用
            'metadata': {
                'user_id': sequence['user_id'],
                'target_id': sequence['target']['business_id'],
                'target_lat': sequence['target']['latitude'],
                'target_lon': sequence['target']['longitude'],
                'target_sid': output
            }
        }
    
    def create_task_b_prompt(self, sequence):
        """Task B: Preference Summary"""
        history_text = self.format_history(sequence['history'], include_rating=True)
        instruction = self.task_b_template.format(history=history_text).strip()
        
        # 简单的规则生成 Ground Truth Summary
        # 实际场景中可以用 GPT-4 生成高质量 Summary 作为 Teacher
        summary = self.generate_preference_summary(sequence['history'])
        
        return {
            'task': 'task_b_preference_summary',
            'instruction': instruction,
            'input': '',
            'output': summary
        }
    
    def generate_preference_summary(self, history):
        high_rated = [item for item in history if item['stars'] >= 4.0]
        if not high_rated: return "User has diverse tastes."
        
        cats = defaultdict(int)
        cities = set()
        for item in high_rated:
            for c in item['categories'].split(','):
                cats[c.strip()] += 1
            cities.add(item['city'])
            
        top_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:3]
        cat_str = ", ".join([c[0] for c in top_cats])
        
        summary = f"The user enjoys {cat_str}."
        if cities:
            summary += f" They are active in {', '.join(list(cities)[:2])}."
        return summary
    
    def create_task_c1_prompt(self):
        """Task C1: Text -> ID (Alignment)"""
        # 随机采样一个 Business
        bid = random.choice(list(self.sid_mapping.keys()))
        info = self.sid_mapping[bid]
        
        # Query 必须包含地理信息，否则无法对应到唯一的 Geo-aware ID
        # Template: "[Category] in [City]"
        cat = info['categories'].split(',')[0].strip()
        city = info['city']
        query = f"{cat} in {city}"
        
        # 还可以增加 Name 增强准确性
        # query = f"{info['name']} ({cat}) in {city}"
        
        instruction = self.task_c1_template.format(query=query)
        
        # Output: Unique ID
        output = info['sid_str']
        
        return {
            'task': 'task_c_sid_alignment',
            'instruction': instruction,
            'input': '',
            'output': output
        }

    def construct_prompts_for_split(self, sequences, split_name):
        print(f"\n=== Constructing {split_name} Prompts ===")
        prompts = []
        
        # 1. Task A (All Sequences)
        for seq in tqdm(sequences, desc="Task A"):
            prompts.append(self.create_task_a_prompt(seq))
            
        # 2. Task B & C (Only for Train)
        if split_name == 'train':
            # Task B
            n_task_b = int(len(sequences) * 0.1) # 10% ratio
            b_samples = random.sample(sequences, min(n_task_b, len(sequences)))
            for seq in tqdm(b_samples, desc="Task B"):
                prompts.append(self.create_task_b_prompt(seq))
                
            # Task C
            n_task_c = int(len(sequences) * 0.1) # 10% ratio
            for _ in tqdm(range(n_task_c), desc="Task C"):
                prompts.append(self.create_task_c1_prompt())
        
        random.shuffle(prompts)
        print(f"Total {split_name} prompts: {len(prompts)}")
        return prompts
    
    def save_prompts(self, data, filename):
        path = os.path.join(self.processed_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved to {filename}")

    def run(self):
        print("\n" + "="*60)
        print("Step 4: Constructing Prompts (Optimized)")
        print("="*60)
        
        self.load_sequences()
        self.load_sid_mapping()
        
        train_p = self.construct_prompts_for_split(self.train_sequences, 'train')
        valid_p = self.construct_prompts_for_split(self.valid_sequences, 'valid')
        test_p = self.construct_prompts_for_split(self.test_sequences, 'test')
        
        self.save_prompts(train_p, self.data_config['train_prompts_file'])
        self.save_prompts(valid_p, self.data_config['valid_prompts_file'])
        self.save_prompts(test_p, self.data_config['test_prompts_file'])

def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    constructor = PromptConstructor(config)
    constructor.run()

if __name__ == '__main__':
    main()