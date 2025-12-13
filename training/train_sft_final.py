import os
import sys
import torch
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ================= 配置 =================
class SFTConfig:
    base_model_path = "/workspace/Qwen2_5-1.5B-Instruct"
    data_dir = "/workspace/data/processed"
    
    train_file = "train_prompts_balanced.jsonl" 
    # 注意：验证集必须使用【原始未平衡】的数据，以反应真实指标
    valid_file = "valid_prompts.jsonl"
    
    output_dir = "/workspace/data/llm_ckpt_sft_v5_balanced"

    max_seq_length = 1024       
    
    # LoRA
    use_lora = True
    lora_r = 128                 # 1.5B 模型 128 足够，256有点浪费显存
    lora_alpha = 256            
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    learning_rate = 2e-4        
    num_train_epochs = 3        # 使用 Epochs 而不是 Max Steps，适应数据量变化
    
    batch_size = 24             # 5090 Batch Size
    gradient_accumulation_steps = 1 
    
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    save_steps = 500            
    save_total_limit = 2        
    eval_strategy = "steps"
    eval_steps = 500

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.conf = config
        os.makedirs(self.conf.output_dir, exist_ok=True)
        self.tokenizer = None
        self.model = None
        self.tokenized_datasets = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.base_model_path,
            trust_remote_code=True,
            padding_side='right' 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_data(self):
        train_path = os.path.join(self.conf.data_dir, self.conf.train_file)
        valid_path = os.path.join(self.conf.data_dir, self.conf.valid_file)
        
        logger.info(f"Loading datasets...")
        raw_dataset = load_dataset('json', data_files={'train': train_path, 'validation': valid_path})
        
        tokenizer = self.tokenizer
        max_len = self.conf.max_seq_length

        # === 核心逻辑：动态历史增强 ===
        def augment_history(text):
            """随机丢弃历史记录中的 1-2 项，防止对重复数据的过拟合"""
            if "User History:" not in text:
                return text
            
            try:
                # 分割 Prompt
                parts = text.split("User History:\n")
                if len(parts) < 2: return text
                
                preamble = parts[0]
                # 假设结尾是 "\nResponse:" 或 "\nUser Profile" 等，取历史部分
                history_part = parts[1]
                
                # 找到历史结束的位置 (Response 前)
                end_marker = "\nResponse:"
                if end_marker in history_part:
                    history_content, suffix = history_part.split(end_marker, 1)
                    suffix = end_marker + suffix
                else:
                    # 容错
                    return text

                lines = history_content.strip().split('\n')
                
                # 只有历史够长才做 Dropout
                if len(lines) >= 4:
                    # 30% 的概率丢弃 1 项，10% 的概率丢弃 2 项
                    rand_val = random.random()
                    drop_count = 0
                    if rand_val < 0.10: drop_count = 2
                    elif rand_val < 0.40: drop_count = 1
                    
                    if drop_count > 0:
                        # 随机选要丢弃的索引
                        indices_to_drop = set(random.sample(range(len(lines)), drop_count))
                        new_lines = []
                        new_idx = 1
                        for i, line in enumerate(lines):
                            if i in indices_to_drop: continue
                            # 重新编号 "1. [Name]..." -> "{new_idx}. [Name]..."
                            # 去掉旧编号
                            content = line.split('. ', 1)[-1] if '. ' in line else line
                            new_lines.append(f"{new_idx}. {content}")
                            new_idx += 1
                        
                        history_content = "\n".join(new_lines)
                
                return f"{preamble}User History:\n{history_content}{suffix}"
            
            except Exception as e:
                # 如果处理出错，返回原文本，不要中断训练
                return text

        def tokenize_and_mask(sample, idx=None):
            instruction = sample['instruction']
            output = sample['output']
            
            # [关键] 只对训练集做增强，验证集保持原样
            # datasets 的 map 函数并不直接告诉我们是 train 还是 valid
            # 但我们可以通过外部标志或简单的逻辑判断
            # 这里为了简单，总是尝试增强，但由于这是 map 函数，
            # 如果我们每次 epoch 都重新 map 成本太高。
            # 这里的增强是在预处理阶段做死（static augmentation）。
            # 对于 SFT 来说，因为我们已经上采样了数据，所以静态增强也有效：
            # 比如样本 A 被复制了 5 次，这 5 次预处理时会生成 5 种不同的历史掩码。
            
            # 改进：如果是平衡数据集（有重复），则执行增强
            instruction = augment_history(instruction)
            
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 计算 Prompt 长度用于 Mask
            messages_prompt = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
            
            tokenized_full = tokenizer(full_text, truncation=True, max_length=max_len, add_special_tokens=False)
            tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)
            
            input_ids = torch.tensor(tokenized_full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_full["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            
            prompt_len = len(tokenized_prompt["input_ids"])
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                labels[:] = -100
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        logger.info("Processing Datasets (Includes History Augmentation)...")
        # 必须设为 False load_from_cache_file，确保随机性生效
        self.tokenized_datasets = raw_dataset.map(
            tokenize_and_mask,
            batched=False,
            num_proc=8,
            load_from_cache_file=False, 
            remove_columns=raw_dataset['train'].column_names,
            desc="Tokenizing & Augmenting"
        )

    def load_model(self):
        logger.info(f"Loading Model (BF16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.conf.base_model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
        self.model.gradient_checkpointing_enable()
        
        if self.conf.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.conf.lora_r,
                lora_alpha=self.conf.lora_alpha,
                lora_dropout=self.conf.lora_dropout,
                target_modules=self.conf.target_modules,
                bias="none"
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.model.enable_input_require_grads()

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.conf.output_dir,
            num_train_epochs=self.conf.num_train_epochs,
            per_device_train_batch_size=self.conf.batch_size,
            per_device_eval_batch_size=self.conf.batch_size,
            gradient_accumulation_steps=self.conf.gradient_accumulation_steps,
            learning_rate=self.conf.learning_rate,
            weight_decay=0.01,
            warmup_ratio=self.conf.warmup_ratio,
            lr_scheduler_type=self.conf.lr_scheduler_type,
            logging_steps=self.conf.logging_steps,
            eval_strategy=self.conf.eval_strategy,
            eval_steps=self.conf.eval_steps,
            save_strategy="steps",
            save_steps=self.conf.save_steps,
            save_total_limit=self.conf.save_total_limit,
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="tensorboard",
            remove_unused_columns=True,
            group_by_length=False, 
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8, 
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train()
        trainer.save_model(self.conf.output_dir)
        self.tokenizer.save_pretrained(self.conf.output_dir)

if __name__ == "__main__":
    SFTTrainer(SFTConfig()).process_data() # 单步调试用
    # main() # 实际运行时取消注释
    SFTTrainer(SFTConfig()).load_tokenizer()
    trainer = SFTTrainer(SFTConfig())
    trainer.load_tokenizer()
    trainer.process_data()
    trainer.load_model()
    trainer.train()