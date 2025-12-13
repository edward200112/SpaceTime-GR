import os
import sys
import yaml
import json
import torch
import logging
import shutil
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
from datasets import load_dataset, Dataset

# ==============================================================================
# 1. 配置区域
# ==============================================================================

class SFTConfig:
    # 路径配置
    base_model_path = "/workspace/Qwen2_5-1.5B-Instruct"
    data_dir = "/workspace/data/processed"
    train_file = "train_prompts_balanced.jsonl"
    valid_file = "valid_prompts.jsonl"
    
    output_dir = "/workspace/data/llm_ckpt_sft_v5_balanced"

    # 模型参数
    max_seq_length = 1024       
    
    # LoRA 参数
    use_lora = True
    lora_r = 64                 
    lora_alpha = 128            
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 训练超参
    learning_rate = 2e-4        
    num_train_epochs = 3
    batch_size = 8              
    gradient_accumulation_steps = 2 
    warmup_ratio = 0.03
    logging_steps = 10
    save_steps = 200            
    eval_steps = 200

# ==============================================================================
# 2. 日志设置
# ==============================================================================

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==============================================================================
# 3. 核心 SFT Trainer 类
# ==============================================================================

class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.conf = config
        os.makedirs(self.conf.output_dir, exist_ok=True)
        self.tokenizer = None
        self.model = None
        self.tokenized_datasets = None

    def load_tokenizer(self):
        """只加载 Tokenizer，不触碰 GPU"""
        logger.info(f"Loading Tokenizer from: {self.conf.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.base_model_path,
            trust_remote_code=True,
            padding_side='right' 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Setting pad_token to eos_token.")

    def process_data(self):
        """数据处理 (此时 CUDA 尚未初始化，可以安全使用多进程)"""
        logger.info("Processing Datasets with STRICT masking...")
        
        train_path = os.path.join(self.conf.data_dir, self.conf.train_file)
        valid_path = os.path.join(self.conf.data_dir, self.conf.valid_file)
        
        raw_dataset = load_dataset('json', data_files={'train': train_path, 'validation': valid_path})

        # 闭包函数：避免 pickle 问题，直接使用外部的 self.tokenizer
        tokenizer = self.tokenizer
        max_len = self.conf.max_seq_length

        def tokenize_and_mask(sample):
            instruction = sample['instruction']
            output = sample['output']
            
            # 1. 完整对话 (Prompt + Response)
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            # 这里的 tokenize=False 返回字符串
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 2. 仅 Prompt (用于计算 Mask 长度)
            messages_prompt = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
            
            # 3. Tokenize
            tokenized_full = tokenizer(
                full_text, 
                truncation=True, 
                max_length=max_len,
                add_special_tokens=False
            )
            
            tokenized_prompt = tokenizer(
                prompt_text, 
                truncation=True, 
                max_length=max_len,
                add_special_tokens=False
            )
            
            input_ids = torch.tensor(tokenized_full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_full["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            
            # 4. [Masking Logic]
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

        # 使用 num_proc=8 加速
        self.tokenized_datasets = raw_dataset.map(
            tokenize_and_mask,
            batched=False,
            num_proc=8,  # 现在这里是安全的
            remove_columns=raw_dataset['train'].column_names,
            desc="Tokenizing & Masking"
        )
        
        # 调试信息
        logger.info("=== Data Check ===")
        labels_example = self.tokenized_datasets['train'][0]['labels']
        masked_count = sum(1 for x in labels_example if x == -100)
        logger.info(f"Sample Total Len: {len(labels_example)}")
        logger.info(f"Masked Len (Instruction): {masked_count}")
        logger.info(f"Learned Len (Response): {len(labels_example) - masked_count}")

    def load_model(self):
        """数据处理完后，再加载模型到 GPU"""
        logger.info(f"Loading Model (BF16 Full Precision) from: {self.conf.base_model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.conf.base_model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
        
        self.model.gradient_checkpointing_enable()
        
        if self.conf.use_lora:
            logger.info(f"Applying LoRA: r={self.conf.lora_r}")
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
            
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def train(self):
        logger.info(f"Starting Training -> {self.conf.output_dir}")
        
        training_args = TrainingArguments(
            output_dir=self.conf.output_dir,
            num_train_epochs=self.conf.num_train_epochs,
            per_device_train_batch_size=self.conf.batch_size,
            per_device_eval_batch_size=self.conf.batch_size,
            gradient_accumulation_steps=self.conf.gradient_accumulation_steps,
            learning_rate=self.conf.learning_rate,
            weight_decay=0.01,
            warmup_ratio=self.conf.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=self.conf.logging_steps,
            
            eval_strategy="steps",
            eval_steps=self.conf.eval_steps,
            save_strategy="steps",
            save_steps=self.conf.save_steps,
            save_total_limit=3,
            
            bf16=True, 
            fp16=False,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="tensorboard",
            remove_unused_columns=True,
            group_by_length=True,
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
            eval_dataset=self.tokenized_datasets['validation'].select(range(min(500, len(self.tokenized_datasets['validation'])))),
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Resume Checkpoint Logic
        resume_ckpt = None
        if os.path.exists(self.conf.output_dir):
            ckpts = [d for d in os.listdir(self.conf.output_dir) if d.startswith("checkpoint-")]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split("-")[-1]))
                resume_ckpt = os.path.join(self.conf.output_dir, ckpts[-1])
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")

        trainer.train(resume_from_checkpoint=resume_ckpt)
        
        logger.info(f"Saving Final Model to {self.conf.output_dir}")
        trainer.save_model(self.conf.output_dir)
        self.tokenizer.save_pretrained(self.conf.output_dir)

def main():
    config = SFTConfig()
    trainer = SFTTrainer(config)
    
    # 1. 先只加载 Tokenizer
    trainer.load_tokenizer()
    
    # 2. 安全地处理数据 (多进程 OK)
    trainer.process_data()
    
    # 3. 最后加载模型到 GPU (初始化 CUDA)
    trainer.load_model()
    
    # 4. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()