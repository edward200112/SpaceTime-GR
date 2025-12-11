import os
import sys
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
from datasets import load_dataset

# ==============================================================================
# 1. 配置区域 (V5 Ultimate Strategy)
# ==============================================================================

class SFTConfig:
    # 路径配置
    base_model_path = "/workspace/Qwen2_5-1.5B-Instruct"
    data_dir = "/workspace/data/processed"
    
    # [关键] 使用平衡采样后的数据集
    train_file = "train_prompts_balanced.jsonl" 
    valid_file = "valid_prompts.jsonl"
    
    # [关键] 输出到 V5 专用目录
    output_dir = "/workspace/data/llm_ckpt_sft_v5_balanced"

    # 模型参数 (32GB VRAM 豪华配置: 全精度 BF16)
    max_seq_length = 1024       
    
    # --- [关键优化] LoRA++ 配置 ---
    use_lora = True
    # Rank 拉到 256，提供接近全量微调的表现力，同时保持训练稳定性
    lora_r = 256                 
    lora_alpha = 512            # alpha = 2 * r
    lora_dropout = 0.05
    # 全面覆盖：不仅是 Attention，还包括 MLP (Gate/Up/Down)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 训练超参
    learning_rate = 2e-4        # High-Rank LoRA 适合这个学习率
    
    # [关键] 限制总步数，防止在 300万数据上跑太久导致过拟合
    # 20k steps * 24 batch ≈ 48万条样本，足够模型学会了
    max_steps = 20000           
    num_train_epochs = 3        # 这个参数会被 max_steps 覆盖，保留即可
    
    # 显存充裕，Batch Size 拉大到 24 (5090 应该能吃得消)
    batch_size = 24             
    gradient_accumulation_steps = 1 
    
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    
    logging_steps = 10
    save_steps = 500            # 每500步保存一次
    save_total_limit = 3        # 最多保留3个checkpoint
    eval_steps = 500
    eval_strategy = "steps"

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
        
        # 打印配置确认
        logger.info(f"Configuration Loaded.")
        logger.info(f"Target Data: {self.conf.train_file}")
        logger.info(f"Output Dir:  {self.conf.output_dir}")
        logger.info(f"LoRA Rank:   {self.conf.lora_r}")

    def load_tokenizer(self):
        """第一步：只加载 Tokenizer (CPU操作)"""
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
        """第二步：数据处理与Masking (CPU多进程安全)"""
        train_path = os.path.join(self.conf.data_dir, self.conf.train_file)
        valid_path = os.path.join(self.conf.data_dir, self.conf.valid_file)
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到平衡数据集: {train_path}。请先运行 balance_dataset.py！")

        logger.info(f"Loading datasets...")
        raw_dataset = load_dataset('json', data_files={'train': train_path, 'validation': valid_path})
        
        # 为了避免 pickling 问题，将变量本地化
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
                labels[:] = -100 # 异常保护
                
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        logger.info("Processing Datasets with STRICT masking (num_proc=8)...")
        self.tokenized_datasets = raw_dataset.map(
            tokenize_and_mask,
            batched=False,
            num_proc=8,
            remove_columns=raw_dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        # 数据检查
        logger.info("=== Data Check ===")
        labels_example = self.tokenized_datasets['train'][0]['labels']
        masked_count = sum(1 for x in labels_example if x == -100)
        logger.info(f"Sample Total Len: {len(labels_example)}")
        logger.info(f"Masked Len (Instruction): {masked_count}")
        if masked_count == 0:
            logger.warning("⚠️ WARNING: No labels masked! Check prompt template.")

    def load_model(self):
        """第三步：加载模型到 GPU (初始化 CUDA)"""
        logger.info(f"Loading Model (BF16 Full Precision)...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.conf.base_model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
        
        # 开启 Gradient Checkpointing (省显存，允许更大 Batch)
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
            self.model.print_trainable_parameters()
            
            # 必须开启 input_require_grads
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def train(self):
        """第四步：开始训练"""
        logger.info(f"Starting Training -> {self.conf.output_dir}")
        
        training_args = TrainingArguments(
            output_dir=self.conf.output_dir,
            num_train_epochs=self.conf.num_train_epochs,
            max_steps=self.conf.max_steps, # [关键] 使用 max_steps
            
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
            
            bf16=True,  # 3090/4090/A100 必须开
            fp16=False,
            
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="tensorboard",
            remove_unused_columns=True,
            
            # [关键修复] 关闭分组，防止大数据量下的启动卡死
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
            eval_dataset=self.tokenized_datasets['validation'].select(range(min(500, len(self.tokenized_datasets['validation'])))),
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # 自动断点恢复
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
    
    # 1. Tokenizer
    trainer.load_tokenizer()
    # 2. Data (CPU Multi-processing safe)
    trainer.process_data()
    # 3. Model (GPU init)
    trainer.load_model()
    # 4. Train
    trainer.train()

if __name__ == "__main__":
    main()