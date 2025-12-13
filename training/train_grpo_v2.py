"""
Step 6 (Retry): GRPO Training V2 (Optimized for Accuracy)
"""

import os
import sys
import yaml
import torch
import logging
from datasets import load_dataset
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

# [关键] 引入新的优化版奖励函数
from grpo_rewards_optimized import load_mapping, format_reward_func, geo_reward_func, semantic_reward_func

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # [关键] 重新从 SFT (checkpoint-28000) 开始
    sft_ckpt_path = "/workspace/data/llm_ckpt/checkpoint-28000" 
    base_model_path = config['llm']['model_name']
    
    # 输出到新目录 v2
    output_dir = os.path.join(config['data']['llm_ckpt_dir'], "grpo_v2_optimized")
    
    # 加载映射
    load_mapping(os.path.join(config['data']['processed_dir'], config['data']['sid_mapping_file']))
    
    logger.info("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    logger.info(f"Loading SFT Adapter: {sft_ckpt_path}")
    model = PeftModel.from_pretrained(model, sft_ckpt_path)
    model = model.merge_and_unload()
    
    # 新的 LoRA 配置
    peft_config = LoraConfig(
        r=64, # [提升] 增加秩，提升模型学习能力
        lora_alpha=128,
        target_modules=config['llm']['target_modules'],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none"
    )
    
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 加载数据
    dataset_path = os.path.join(config['data']['processed_dir'], config['data']['train_prompts_file'])
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.filter(lambda x: x['task'] == 'task_a_recommendation')
    
    # 随机采样 30k 条数据进行强化学习（足够了）
    if len(dataset) > 30000:
        dataset = dataset.shuffle(seed=42).select(range(30000))

    def format_data(sample):
        # 确保 Instruction 包含格式要求
        base_instr = sample["instruction"]
        if "Output the semantic ID" not in base_instr:
             instr = f"{base_instr}\nOutput the semantic ID in the format <c0, c1, c2, suffix>."
        else:
             instr = base_instr
             
        return {
            "prompt": [{"role": "user", "content": instr}],
            "target_lat": sample['metadata']['target_lat'],
            "target_lon": sample['metadata']['target_lon'],
            "target_sid": sample['metadata']['target_sid']
        }
    dataset = dataset.map(format_data)

    # [关键] 调整训练参数鼓励探索
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=2e-6, # 稍微提高一点 LR
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8, 
        max_prompt_length=1024,
        max_completion_length=32,
        save_steps=300,
        logging_steps=5,
        bf16=True,
        report_to="none",
        temperature=1.2, # [关键] 提高采样温度，鼓励模型尝试不同的 ID，而不是死守热门
        beta=0.04 # [关键] 增加 KL 惩罚，防止模型完全遗忘 SFT 学到的序列知识
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, geo_reward_func, semantic_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )

    logger.info("Starting GRPO V2 (High Accuracy Mode)...")
    trainer.train()
    
    trainer.save_model(output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    main()