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
# 1. 配置区域 (可以直接在这里修改，也可以读取 YAML)
# ==============================================================================

class SFTConfig:
    # 路径配置
    base_model_path = "/workspace/Qwen2_5-1.5B-Instruct"
    data_dir = "/workspace/data/processed"
    train_file = "train_prompts.jsonl"
    valid_file = "valid_prompts.jsonl"
    
    # [关键] 输出到一个新的干净目录
    output_dir = "/workspace/data/llm_ckpt_sft_v2_optimized"

    # 模型参数 (32GB 显存豪华配置)
    max_seq_length = 1024       # 根据你的Prompt长度调整，长一点没关系
    use_4bit = False            # [优化] 关闭量化，使用原生 BF16，精度更高！
    
    # LoRA 参数 (加强版)
    use_lora = True
    lora_r = 64                 # [优化] 提升 Rank，增强小模型的拟合能力
    lora_alpha = 128            # 通常是 r 的 2 倍
    lora_dropout = 0.05
    # [优化] 覆盖所有线性层，最大化 LoRA 效果
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 训练超参
    learning_rate = 2e-4        # SFT 标准学习率
    num_train_epochs = 3
    batch_size = 8              # 32G显存可以开大一点，更稳
    gradient_accumulation_steps = 2 
    warmup_ratio = 0.03
    logging_steps = 10
    save_steps = 200            # 每200步存一次
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
        # 备份当前脚本到输出目录，方便复现
        try:
            shutil.copy(__file__, os.path.join(self.conf.output_dir, "train_script_backup.py"))
        except:
            pass

    def load_model_and_tokenizer(self):
        logger.info(f"Loading Tokenizer from: {self.conf.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.base_model_path,
            trust_remote_code=True,
            padding_side='right' # 训练时通常设为 right，DataCollator 会自动处理
        )
        # Qwen 必须确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Setting pad_token to eos_token.")

        logger.info(f"Loading Model (BF16 Full Precision) from: {self.conf.base_model_path}")
        
        # [优化] 直接加载 BF16，不使用 Quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.conf.base_model_path,
            torch_dtype=torch.bfloat16,  # Ampere架构神器
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" # 既然你有32G显存，大概率支持FlashAttn2
        )
        
        # 开启 Gradient Checkpointing 节省显存 (虽然显存够，但开了可以跑更大的 Batch)
        self.model.gradient_checkpointing_enable()
        
        if self.conf.use_lora:
            logger.info(f"Applying LoRA: r={self.conf.lora_r}, targets={self.conf.target_modules}")
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
            
            # 打印可训练参数
            self.model.print_trainable_parameters()
            
            # [关键] 必须启用 input_require_grads 才能在 Checkpointing 下训练 LoRA
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def process_data(self):
        """
        [CRITICAL FIX] 修复 Label Masking 问题
        计算 Instruction 长度，将 Labels 中 Instruction 部分设为 -100
        """
        logger.info("Processing Datasets with STRICT masking...")
        
        train_path = os.path.join(self.conf.data_dir, self.conf.train_file)
        valid_path = os.path.join(self.conf.data_dir, self.conf.valid_file)
        
        raw_dataset = load_dataset('json', data_files={'train': train_path, 'validation': valid_path})

        def tokenize_and_mask(sample):
            instruction = sample['instruction']
            output = sample['output']
            
            # 1. 完整对话 (Prompt + Response)
            # Qwen 的 apply_chat_template 会自动处理特殊 Token (<|im_start|> user ...)
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 2. 仅 Prompt (用于计算 Mask 长度)
            messages_prompt = [{"role": "user", "content": instruction}]
            prompt_text = self.tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
            
            # 3. Tokenize
            tokenized_full = self.tokenizer(
                full_text, 
                truncation=True, 
                max_length=self.conf.max_seq_length,
                add_special_tokens=False # template 已经加了
            )
            
            tokenized_prompt = self.tokenizer(
                prompt_text, 
                truncation=True, 
                max_length=self.conf.max_seq_length,
                add_special_tokens=False
            )
            
            input_ids = torch.tensor(tokenized_full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_full["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
            
            # 4. [Masking Logic] 将 Prompt 部分的 loss 屏蔽
            prompt_len = len(tokenized_prompt["input_ids"])
            
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
            else:
                # 极其罕见的情况：截断导致 response 没了
                labels[:] = -100
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        # 使用 num_proc 加速处理
        self.tokenized_datasets = raw_dataset.map(
            tokenize_and_mask,
            batched=False,
            num_proc=8,
            remove_columns=raw_dataset['train'].column_names,
            desc="Tokenizing & Masking"
        )
        
        # 调试：检查第一个样本
        logger.info("=== Data Check ===")
        labels_example = self.tokenized_datasets['train'][0]['labels']
        # 统计 -100 的数量
        masked_count = sum(1 for x in labels_example if x == -100)
        logger.info(f"Sample Total Len: {len(labels_example)}")
        logger.info(f"Masked Len (Instruction): {masked_count}")
        logger.info(f"Learned Len (Response): {len(labels_example) - masked_count}")
        
        # 确保至少有东西被 Mask 了
        if masked_count == 0:
            logger.warning("⚠️ WARNING: No labels were masked! Check tokenizer logic.")

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
            save_total_limit=3, # 保留最近3个
            
            bf16=True, # 30系列显卡以上必须开启
            fp16=False,
            
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="tensorboard",
            remove_unused_columns=True,
            
            # [优化] 按长度分组，减少 Padding 浪费，训练更快
            group_by_length=True,
        )
        
        # 专门处理 padding 的 collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8, # 这里的优化对 Tensor Core 友好
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
        
        # 检查是否有 checkpoint 可以恢复
        resume_ckpt = None
        if os.path.exists(self.conf.output_dir):
            ckpts = [d for d in os.listdir(self.conf.output_dir) if d.startswith("checkpoint-")]
            if ckpts:
                # 按步数排序
                ckpts.sort(key=lambda x: int(x.split("-")[-1]))
                resume_ckpt = os.path.join(self.conf.output_dir, ckpts[-1])
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")

        trainer.train(resume_from_checkpoint=resume_ckpt)
        
        # 保存最终模型
        logger.info(f"Saving Final Model to {self.conf.output_dir}")
        trainer.save_model(self.conf.output_dir)
        self.tokenizer.save_pretrained(self.conf.output_dir)

def main():
    config = SFTConfig()
    trainer = SFTTrainer(config)
    trainer.load_model_and_tokenizer()
    trainer.process_data()
    trainer.train()

if __name__ == "__main__":
    main()