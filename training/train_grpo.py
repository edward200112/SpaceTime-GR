"""
Step 6: GRPO Training (Standard TRL Implementation)
"""

import os
import sys
import yaml
import torch
import logging
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# 引入 Rewards
from grpo_rewards import load_mapping, format_reward_func, geo_reward_func, semantic_reward_func

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    # 1. Config
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    sft_ckpt_path = "/workspace/data/llm_ckpt/checkpoint-14500" # 请确认路径
    base_model_path = config['llm']['model_name']
    output_dir = os.path.join(config['data']['llm_ckpt_dir'], "grpo_final")
    
    # 2. 初始化 Reward Mapping
    load_mapping(os.path.join(config['data']['processed_dir'], config['data']['sid_mapping_file']))
    
    # 3. Model & Tokenizer
    logger.info("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    # Load SFT & Merge
    model = PeftModel.from_pretrained(model, sft_ckpt_path)
    model = model.merge_and_unload()
    
    # New LoRA for RL
    peft_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=config['llm']['target_modules'],
        task_type="CAUSAL_LM", lora_dropout=0.1, bias="none"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt_path)
    tokenizer.padding_side = "left" # CRITICAL for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 4. Dataset
    dataset_path = os.path.join(config['data']['processed_dir'], config['data']['train_prompts_file'])
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Filter only Task A
    dataset = dataset.filter(lambda x: x['task'] == 'task_a_recommendation')
    
    # Sampling for speed (RL 10k steps is enough)
    if len(dataset) > 20000:
        dataset = dataset.shuffle(seed=42).select(range(20000))

    # Format for GRPOTrainer
    def format_data(sample):
        return {
            "prompt": [
                {"role": "user", "content": sample["instruction"]}
            ],
            "target_lat": sample['metadata']['target_lat'],
            "target_lon": sample['metadata']['target_lon'],
            "target_sid": sample['metadata']['target_sid']
        }
    dataset = dataset.map(format_data)

    # 5. Training Args
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6, # Low LR for RL
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8, # Group Size
        max_prompt_length=1024,
        max_completion_length=32, # IDs are short
        save_steps=100,
        logging_steps=10,
        bf16=True,
        report_to="none" # or "wandb"
    )

    # 6. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, geo_reward_func, semantic_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )

    logger.info("Starting GRPO...")
    trainer.train()
    
    trainer.save_model(output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    main()