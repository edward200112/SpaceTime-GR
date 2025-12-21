import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from custom_trainer import CoINSFTTrainer, CurriculumCallback

# ================= 配置 =================
MODEL_ID = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_balanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output"
PROCESSED_DATA_DIR = "./SFT/sft_data/processed_cache_1.2M"

def format_instruction(sample, completion_key="completion"):
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample[completion_key]}<|im_end|>"

def format_augment_instruction(sample):
    return f"<|im_start|>user\n{sample['prompt_augment']}<|im_end|>\n<|im_start|>assistant\n{sample['completion']}<|im_end|>"

def format_negative_instruction(sample):
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['negative_completion']}<|im_end|>"

def main():
    print(f"🔄 Initializing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # ================= 数据加载 =================
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"🚀 Found cached processed dataset at: {PROCESSED_DATA_DIR}")
        try:
            full_dataset = load_from_disk(PROCESSED_DATA_DIR)
            print(f"✅ Successfully loaded {len(full_dataset)} examples.")
        except Exception as e:
            print(f"❌ Cache broken. Please re-run data processing.")
            return
    else:
        # 如果没有缓存，这里为了简化代码，建议你先用原来的脚本生成缓存
        # 或者直接报错提示
        print("❌ Error: Processed data not found. Please run the previous data processing script first.")
        return

    # [新增] 自动切分训练集和验证集
    # 120万数据，切分 0.5% (约 6000 条) 做验证足够了，太多会拖慢训练
    print("✂️ Splitting dataset into Train/Eval...")
    dataset_split = full_dataset.train_test_split(test_size=0.005, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"   Train Size: {len(train_dataset)}")
    print(f"   Eval  Size: {len(eval_dataset)}")

    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ================= [关键修改] 训练参数优化 =================
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        
        # 1. 显存压榨策略
        per_device_train_batch_size=8,   # [翻倍] 4 -> 8 (利用剩余的 14GB 显存)
        gradient_accumulation_steps=2,   # [减半] 4 -> 2 (保持总 Batch 不变，更新频率加快)
                                         # 总有效 Batch = 8 * 2 = 16，和之前一样，但吞吐量更大
        
        # 2. 速度优化开关
        gradient_checkpointing=False,    # [关键] 显存够用时，关掉它能提速 30-40%
        group_by_length=True,            # [关键] 减少 Padding 计算，提速显著
        dataloader_num_workers=8,        # [关键] 增加 CPU 预加载线程，防止 IO 瓶颈
        
        # 3. 精度与优化器
        learning_rate=2e-5,
        fp16=True,                       # 保持 fp16，如果有 3090/4090/A100 可改为 bf16=True
        optim="adamw_torch_fused",       # [可选] 使用更快的 Fused 优化器 (需要 torch 2.0+)
        
        # 4. 保存与评估策略
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=10,                # 日志打印频率调高一点，看着爽
        
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        remove_unused_columns=False,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True} 
    )

    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,   # [新增] 传入验证集
        args=training_args,
        processing_class=tokenizer, 
        callbacks=[CurriculumCallback()]
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train()
    
    print(f"Saving Final Model to {OUTPUT_DIR}...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))

if __name__ == "__main__":
    main()