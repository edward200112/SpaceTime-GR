"""
Step 4: 多任务 Prompt 构造 (Final Robust Version)
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
        
        # Templates
        self.task_a_template = (
            "User is currently in {current_city}. "
            "Based on the visit history below, predict the next place to visit.\n"
            "{longterm_summary}"
            "User History:\n{history}\n"
            "Response:"
        )
        
        self.task_b_template = "Summarize the user's preferences based on the following visit history:\n{history}\nResponse:"
        self.task_c1_template = "What is the Semantic ID for \"{query}\"?\nResponse:"
        
        # Data containers
        self.train_sequences = []
        self.valid_sequences = []
        self.test_sequences = []
        self.sid_mapping = {}
    
    def load_resources(self):
        """加载数据和映射"""
        print("\n=== Loading Resources ===")
        
        # 1. Load SID Mapping
        map_file = os.path.join(self.processed_dir, self.data_config['sid_mapping_file'])
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"SID Mapping not found: {map_file}")
            
        with open(map_file, 'r', encoding='utf-8') as f:
            self.sid_mapping = json.load(f)
        print(f"Loaded SID Mapping: {len(self.sid_mapping)} items")

        # 2. Load Sequences
        def load_split(name):
            path = os.path.join(self.processed_dir, name)
            data = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): data.append(json.loads(line))
            return data

        self.train_sequences = load_split('train.jsonl')
        self.valid_sequences = load_split('valid.jsonl')
        self.test_sequences = load_split('test.jsonl')
        print(f"Sequences -> Train: {len(self.train_sequences)}, Valid: {len(self.valid_sequences)}, Test: {len(self.test_sequences)}")

    def format_history(self, history, include_rating=False):
        lines = []
        for idx, item in enumerate(history, 1):
            name = item.get('name', 'Unknown Place')
            # 兼容处理 categories 可能是列表也可能是字符串的情况
            raw_cat = item.get('categories', '')
            if isinstance(raw_cat, list):
                cat = raw_cat[0] if raw_cat else 'General'
            else:
                cat = raw_cat.split(',')[0].strip() if raw_cat else 'General'
                
            city = item.get('city', 'Unknown City')
            
            if include_rating:
                line = f"[{name}] ({cat} in {city}) - {item.get('stars', '?')} stars"
            else:
                line = f"[{name}] ({cat} in {city})"
            
            lines.append(f"{idx}. {line}")
        return '\n'.join(lines)
    
    def create_task_a_prompt(self, sequence):
        """Task A: Recommendation (带完整 Metadata)"""
        longterm_text = ""
        if sequence.get('longterm_summary'):
            longterm_text = f"User Profile: {sequence['longterm_summary']}\n"
        
        history_text = self.format_history(sequence['history'])
        current_city = sequence.get('current_city', 'Unknown City')
        
        instruction = self.task_a_template.format(
            current_city=current_city,
            longterm_summary=longterm_text,
            history=history_text
        ).strip()
        
        # 获取 Target 的 SID 信息
        target_bid = sequence['target']['business_id']
        if target_bid not in self.sid_mapping:
            return None # Skip if target has no ID
            
        sid_info = self.sid_mapping[target_bid]
        output = sid_info['sid_str'] # e.g. "<1, 2, 3, 4>"
        
        return {
            'task': 'task_a_recommendation',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {
                'user_id': sequence['user_id'],
                'target_id': target_bid,
                'target_lat': sequence['target']['latitude'],
                'target_lon': sequence['target']['longitude'],
                
                # [Optimization] 同时保存 String 和 List，方便下游处理
                'target_sid': output, 
                'target_sid_tuple': sid_info['full_sid'] # [1, 2, 3, 4]
            }
        }
    
    def create_task_b_prompt(self, sequence):
        """Task B: Preference Summary (Metadata 补全)"""
        history_text = self.format_history(sequence['history'], include_rating=True)
        instruction = self.task_b_template.format(history=history_text).strip()
        
        # 简单规则生成 Summary
        summary = "User has diverse tastes." # 简化逻辑，实际可以用你之前的逻辑
        
        return {
            'task': 'task_b_preference_summary',
            'instruction': instruction,
            'input': '',
            'output': summary,
            'metadata': {'type': 'auxiliary'} # [Fix] 防止 KeyError
        }
    
    def create_task_c1_prompt(self):
        """Task C: Alignment (Metadata 补全)"""
        bid = random.choice(list(self.sid_mapping.keys()))
        info = self.sid_mapping[bid]
        
        raw_cat = info.get('categories', '')
        if isinstance(raw_cat, list):
            cat = raw_cat[0]
        else:
            cat = raw_cat.split(',')[0].strip()
            
        city = info.get('city', 'Unknown')
        query = f"{cat} in {city}"
        
        instruction = self.task_c1_template.format(query=query)
        output = info['sid_str']
        
        return {
            'task': 'task_c_sid_alignment',
            'instruction': instruction,
            'input': '',
            'output': output,
            'metadata': {'type': 'auxiliary'} # [Fix] 防止 KeyError
        }

    def construct_prompts_for_split(self, sequences, split_name):
        print(f"\n=== Constructing {split_name} Prompts ===")
        prompts = []
        
        # 1. Task A
        for seq in tqdm(sequences, desc="Task A"):
            p = self.create_task_a_prompt(seq)
            if p: prompts.append(p)
            
        # 2. Task B & C (Only for Train)
        if split_name == 'train':
            # Task B (10%)
            n_task_b = int(len(sequences) * 0.1)
            b_samples = random.sample(sequences, min(n_task_b, len(sequences)))
            for seq in tqdm(b_samples, desc="Task B"):
                prompts.append(self.create_task_b_prompt(seq))
                
            # Task C (10%)
            n_task_c = int(len(sequences) * 0.1)
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
        self.load_resources()
        
        train_p = self.construct_prompts_for_split(self.train_sequences, 'train')
        valid_p = self.construct_prompts_for_split(self.valid_sequences, 'valid')
        test_p = self.construct_prompts_for_split(self.test_sequences, 'test')
        
        self.save_prompts(train_p, self.data_config['train_prompts_file'])
        self.save_prompts(valid_p, self.data_config['valid_prompts_file'])
        self.save_prompts(test_p, self.data_config['test_prompts_file'])

def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    PromptConstructor(config).run()

if __name__ == '__main__':
    main()