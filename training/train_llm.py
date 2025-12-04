"""
Step 5: SFT Training (Supervised Fine-Tuning) - Fixed for Labels

修复点：
1. [Critical] process_data 中显式生成 'labels'，解决 ValueError: model did not return a loss。
2. [Critical] 添加 enable_input_require_grads()，解决 LoRA + Gradient Checkpointing 的兼容性问题。
3. 优化 DataCollator 配置。
"""

import os
import sys
import yaml
import json
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

class SFTTrainer:
    def __init__(self, config_path='./config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logging()
        self.data_conf = self.config['data']
        self.llm_conf = self.config['llm']
        
        self.output_dir = self.data_conf['llm_ckpt_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        model_name = self.llm_conf['model_name']
        self.logger.info(f"Loading model: {model_name}")
        
        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Model
        bnb_config = None
        if self.llm_conf.get('use_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.llm_conf['bf16'] else torch.float16,
            device_map='auto',
            use_cache=False, 
            attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
        )
        
        # 3. LoRA Setup
        if self.llm_conf['use_lora']:
            self.logger.info("Setting up LoRA...")
            
            # [Fix 2] 开启 Gradient Checkpointing 时必须开启 input_require_grads
            if self.llm_conf['gradient_checkpointing']:
                self.model.gradient_checkpointing_enable()
                # 这一步对于 LoRA 训练至关重要，否则 loss 无法反传
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                
            self.model = prepare_model_for_kbit_training(self.model)
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.llm_conf['lora_r'],
                lora_alpha=self.llm_conf['lora_alpha'],
                lora_dropout=self.llm_conf['lora_dropout'],
                target_modules=self.llm_conf['target_modules']
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
    def process_data(self):
        self.logger.info("Processing Datasets...")
        
        train_file = os.path.join(self.data_conf['processed_dir'], self.data_conf['train_prompts_file'])
        valid_file = os.path.join(self.data_conf['processed_dir'], self.data_conf['valid_prompts_file'])
        
        dataset = load_dataset('json', data_files={'train': train_file, 'validation': valid_file})
        
        def format_prompt(sample):
            instruction = sample['instruction']
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": sample['output']}
            ]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": full_text}

        dataset = dataset.map(format_prompt)
        
        def tokenize_function(examples):
            # Tokenize
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.llm_conf['max_seq_length'],
                padding=False 
            )
            
            # [Fix 1] 显式创建 labels
            # 对于 Causal LM，labels 就是 input_ids。
            # 模型内部会自动将 labels 向左移动一位来计算 next-token prediction loss。
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset['train'].column_names 
        )
        
        self.train_dataset = tokenized_datasets['train']
        self.eval_dataset = tokenized_datasets['validation'].select(range(min(1000, len(tokenized_datasets['validation']))))

        
        self.logger.info(f"Train Size: {len(self.train_dataset)}")
        self.logger.info(f"Eval Size: {len(self.eval_dataset)}")
        
        # 打印一个样本检查 labels 是否存在
        self.logger.info(f"Sample keys: {self.train_dataset[0].keys()}")

    def train(self):
        self.logger.info("Starting Training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.llm_conf['epochs'],
            per_device_train_batch_size=self.llm_conf['batch_size'],
            per_device_eval_batch_size=self.llm_conf['batch_size'],
            gradient_accumulation_steps=self.llm_conf['gradient_accumulation_steps'],
            learning_rate=float(self.llm_conf['lr']),
            weight_decay=self.llm_conf['weight_decay'],
            warmup_ratio=self.llm_conf['warmup_ratio'],
            lr_scheduler_type=self.llm_conf['lr_scheduler'],
            logging_steps=10,
            
            # 使用 eval_strategy (新版 transformers)
            eval_strategy="steps", 
            eval_steps=300,
            
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            bf16=self.llm_conf['bf16'],
            fp16=self.llm_conf['fp16'],
            gradient_checkpointing=self.llm_conf['gradient_checkpointing'],
            report_to="none",
            dataloader_num_workers=4,
            remove_unused_columns=True
        )
        
        # DataCollatorForSeq2Seq 会自动处理 padding，并将 labels 中的 pad token 设为 -100
        # 这样模型计算 loss 时会忽略 padding 部分
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Resume Checkpoint Logic
        checkpoint = None
        if os.path.exists(self.output_dir):
            dirs = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint")]
            if dirs:
                latest = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[-1]
                checkpoint = os.path.join(self.output_dir, latest)
                self.logger.info(f"Resuming from checkpoint: {checkpoint}")
        
        trainer.train(resume_from_checkpoint=checkpoint)
        
        self.logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)

def main():
    trainer = SFTTrainer()
    trainer.load_model_and_tokenizer()
    trainer.process_data()
    trainer.train()

if __name__ == '__main__':
    main()