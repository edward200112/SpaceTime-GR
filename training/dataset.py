"""
Dataset for LLM Training (SFT) and GRPO
[FINAL VERSION]
Changes:
- Added Few-Shot Examples for GRPO.
- Added Pre-filling technique ("Response: <") to force ID generation.
- Compatible with both JSON and JSONL formats.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List

# 引入 HuggingFace 的 Dataset 用于 GRPO
from datasets import Dataset as HFDataset

# ==========================================
# Part 1: SFT Dataset Classes (SFT 阶段复用，保持不变)
# ==========================================

class PromptDataset(Dataset):
    """Dataset for multi-task prompts (SFT Phase)"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # 兼容 JSON List 和 JSONL
            try:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    self.data = json.load(f)
                else:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line.strip()))
            except Exception:
                # Fallback
                f.seek(0)
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: instruction + input (if any) + output
        if item.get('input'):
            text = f"{item['instruction']}\n{item['input']}\n{item['output']}"
        else:
            text = f"{item['instruction']}\n{item['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': encoding['input_ids'].copy(),  # For causal LM
            'task': item.get('task', 'unknown')
        }


class DataCollatorForPromptDataset:
    """Data collator for prompt dataset"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(len(f['input_ids']) for f in features)
        max_len = min(max_len, self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for f in features:
            input_ids = f['input_ids'][:max_len]
            attention_mask = f['attention_mask'][:max_len]
            labels = f['labels'][:max_len]
            
            # Padding
            padding_length = max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length  # -100 is ignored in loss
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long)
        }


def prepare_datasets(config, tokenizer):
    """Prepare train and validation datasets (For SFT)"""
    data_config = config['data']
    processed_dir = data_config['processed_dir']
    max_length = config['llm']['max_seq_length']
    
    train_file = f"{processed_dir}/{data_config['train_prompts_file']}"
    valid_file = f"{processed_dir}/{data_config['valid_prompts_file']}"
    
    train_dataset = PromptDataset(train_file, tokenizer, max_length)
    valid_dataset = PromptDataset(valid_file, tokenizer, max_length)
    
    return train_dataset, valid_dataset


# ==========================================
# Part 2: GRPO Dataset Function (核心修改)
# ==========================================

"""
Dataset for GRPO (Reverted Prompt Style)
去掉 Few-Shot，保持与 SFT 一致的分布，只保留 Pre-filling。
"""

import json
from datasets import Dataset as HFDataset

def get_grpo_dataset(file_path):
    print(f"Loading GRPO dataset from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    rec_data = [d for d in data if d.get('task', 'task_a_recommendation') == 'task_a_recommendation']
    print(f"Found {len(rec_data)} recommendation samples.")

    formatted_data = []
    for item in rec_data:
        raw_prompt = item.get('instruction', '')
        
        # 1. 清理：去掉原始可能存在的 Response
        if "Response:" in raw_prompt:
            base_prompt = raw_prompt.split("Response:")[0].strip()
        else:
            base_prompt = raw_prompt.strip()

        # 2. 构造 Prompt
        # [修改] 移除 Few-Shot，回归最简单的 Instruction
        # 保持与 SFT 训练时尽可能一致，只在最后引导 "<"
        
        instruction_suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
        
        # 最终 Prompt: Instruction + History + Suffix + Response: <
        final_prompt = f"{base_prompt}\n{instruction_suffix}\nResponse: <"
        
        formatted_data.append({
            "prompt": final_prompt,
            "target_sid": item.get('metadata', {}).get('target_sid'), 
            "target_lat": item.get('metadata', {}).get('target_lat'),
            "target_lon": item.get('metadata', {}).get('target_lon')
        })

    dataset = HFDataset.from_list(formatted_data)
    
    print("\n[DEBUG] Prompt Preview (Last 300 chars):")
    if len(dataset) > 0:
        print(dataset[0]['prompt'][-300:])
    print("-" * 20 + "\n")
    
    return dataset