"""
Dataset for LLM Training
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class PromptDataset(Dataset):
    """Dataset for multi-task prompts"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
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
            'task': item['task']
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
    """Prepare train and validation datasets"""
    data_config = config['data']
    processed_dir = data_config['processed_dir']
    max_length = config['llm']['max_seq_length']
    
    train_file = f"{processed_dir}/{data_config['train_prompts_file']}"
    valid_file = f"{processed_dir}/{data_config['valid_prompts_file']}"
    
    train_dataset = PromptDataset(train_file, tokenizer, max_length)
    valid_dataset = PromptDataset(valid_file, tokenizer, max_length)
    
    return train_dataset, valid_dataset
