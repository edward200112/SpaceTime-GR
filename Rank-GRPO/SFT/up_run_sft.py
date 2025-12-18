import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from custom_trainer import CoINSFTTrainer, CurriculumCallback

# ================= 配置 =================
MODEL_ID = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_balanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output"

# [新增] 处理后的数据集缓存路径
PROCESSED_DATA_DIR = "./SFT/sft_data/processed_cache_1.2M"

def format_instruction(sample, completion_key="completion"):
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample[completion_key]}<|im_end|>"

def format_augment_instruction(sample):
    return f"<|im_start|>user\n{sample['prompt_augment']}<|im_end|>\n<|im_start|>assistant\n{sample['completion']}<|im_end|>"

def format_negative_instruction(sample):
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['negative_completion']}<|im_end|>"

def main():
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # ================= 数据加载与缓存逻辑 =================
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"🚀 Found cached dataset at {PROCESSED_DATA_DIR}. Loading directly...")
        tokenized_dataset = load_from_disk(PROCESSED_DATA_DIR)
        print(f"✅ Loaded {len(tokenized_dataset)} examples from disk.")
    else:
        print(f"⚠️ Cache not found. Loading raw data from {DATA_FILE}...")
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")

        # 定义预处理函数 (逻辑保持不变)
        def preprocess_function(examples):
            # A. 正样本
            texts = [
                format_instruction({"prompt": p, "completion": c}, "completion") 
                for p, c in zip(examples['prompt'], examples['completion'])
            ]
            model_inputs = tokenizer(texts, max_length=1024, padding="max_length", truncation=True)
            
            # Labels
            labels = model_inputs["input_ids"].copy()
            
            # Hierarchy Mask
            hierarchy_masks = []
            for i, raw_code in enumerate(examples['raw_target_code']):
                mask = [1] * len(labels[i])
                try:
                    code_parts = raw_code.split() 
                    if len(code_parts) == 4:
                        valid_len = sum(model_inputs["attention_mask"][i])
                        mask[valid_len-2 : valid_len] = [0, 0] 
                except: pass
                hierarchy_masks.append(mask)

            # Labels Padding
            for i in range(len(labels)):
                labels[i] = [-100 if t == tokenizer.pad_token_id else t for t in labels[i]]
            
            model_inputs["labels"] = labels
            model_inputs["hierarchy_mask"] = hierarchy_masks

            # B. 增强样本 (CoIN)
            aug_texts = [
                format_augment_instruction({"prompt_augment": pa, "completion": c}) 
                for pa, c in zip(examples['prompt_augment'], examples['completion'])
            ]
            aug_inputs = tokenizer(aug_texts, max_length=1024, padding="max_length", truncation=True)
            model_inputs["augment_input_ids"] = aug_inputs["input_ids"]
            model_inputs["augment_attention_mask"] = aug_inputs["attention_mask"]

            # C. 负样本
            neg_texts = [
                format_negative_instruction({"prompt": p, "negative_completion": nc}) 
                for p, nc in zip(examples['prompt'], examples['negative_completion'])
            ]
            neg_inputs = tokenizer(neg_texts, max_length=1024, padding="max_length", truncation=True)
            model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
            model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
            
            # D. IPS
            model_inputs["ips_weight"] = examples["ips_weight"]
            
            return model_inputs

        print("⚡ Tokenizing dataset (Running with 12 processes)...")
        # [优化] 开启 num_proc 多进程并行处理，速度提升 10倍+
        tokenized_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            num_proc=12,  # 利用你的 16核 CPU
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"💾 Saving processed dataset to {PROCESSED_DATA_DIR} for future runs...")
        tokenized_dataset.save_to_disk(PROCESSED_DATA_DIR)
    
    # ================= 模型加载与训练 =================
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024,
        callbacks=[CurriculumCallback()]
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train()
    
    print("Saving Final Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))

if __name__ == "__main__":
    main()