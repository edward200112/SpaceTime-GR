import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
# [修改 1] 必须用 SFTConfig 适配 TRL v0.12+
from trl import SFTConfig
from custom_trainer import CoINSFTTrainer, CurriculumCallback 

# ================= 配置 =================
MODEL_ID = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_balanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output_coin" 

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

    print(f"Loading data from {DATA_FILE}...")
    # [修改 2] 加载完整数据集
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # [修改 3] 切分验证集 (只取 200 条作为 Eval，保证速度)
    # 使用 seed=42 保证每次跑都切出一样的数据
    print("✂️ Splitting dataset for evaluation...")
    split_dataset = full_dataset.train_test_split(test_size=200, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Eval size:  {len(eval_dataset)} (Partial sampling for speed)")

    # ================= 高级 Mask 预处理 =================
    def preprocess_function(examples):
        # A. 正样本
        texts = [
            format_instruction({"prompt": p, "completion": c}, "completion") 
            for p, c in zip(examples['prompt'], examples['completion'])
        ]
        model_inputs = tokenizer(texts, max_length=1024, padding="max_length", truncation=True)
        labels = model_inputs["input_ids"].copy()
        
        # --- Hierarchy Mask (子序列匹配逻辑) ---
        hierarchy_masks = []
        raw_codes = examples.get('raw_target_code', [None] * len(labels))
        
        for i, raw_code in enumerate(raw_codes):
            mask = [1] * len(labels[i]) # 默认全看
            
            if raw_code:
                try:
                    # 假设 raw_code: "12 34 56 78"
                    code_parts = raw_code.split() 
                    if len(code_parts) == 4:
                        # 我们要 Mask 掉 " 56 78" (Level 3 & 4)
                        fine_grained_suffix = f" {code_parts[2]} {code_parts[3]}"
                        
                        target_tokens = tokenizer.encode(fine_grained_suffix, add_special_tokens=False)
                        len_tgt = len(target_tokens)
                        
                        if len_tgt > 0:
                            valid_len = sum(model_inputs["attention_mask"][i])
                            search_start = max(0, valid_len - 30)
                            search_window = labels[i][search_start : valid_len]
                            
                            match_index = -1
                            for k in range(len(search_window) - len_tgt, -1, -1):
                                if search_window[k : k + len_tgt] == target_tokens:
                                    match_index = search_start + k
                                    break
                            
                            if match_index != -1:
                                mask[match_index : match_index + len_tgt] = [0] * len_tgt
                except Exception:
                    pass
            hierarchy_masks.append(mask)
        # -----------------------------------------------

        for i in range(len(labels)):
            labels[i] = [-100 if t == tokenizer.pad_token_id else t for t in labels[i]]
        
        model_inputs["labels"] = labels
        model_inputs["hierarchy_mask"] = hierarchy_masks

        # B. 增强样本
        if 'prompt_augment' in examples:
            aug_texts = [
                format_augment_instruction({"prompt_augment": pa, "completion": c}) 
                for pa, c in zip(examples['prompt_augment'], examples['completion'])
            ]
            aug_inputs = tokenizer(aug_texts, max_length=1024, padding="max_length", truncation=True)
            model_inputs["augment_input_ids"] = aug_inputs["input_ids"]
            model_inputs["augment_attention_mask"] = aug_inputs["attention_mask"]
        else:
             model_inputs["augment_input_ids"] = [[]] * len(texts)
             model_inputs["augment_attention_mask"] = [[]] * len(texts)

        # C. 负样本
        if 'negative_completion' in examples:
            neg_texts = [
                format_negative_instruction({"prompt": p, "negative_completion": nc}) 
                for p, nc in zip(examples['prompt'], examples['negative_completion'])
            ]
            neg_inputs = tokenizer(neg_texts, max_length=1024, padding="max_length", truncation=True)
            model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
            model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
        
        if 'ips_weight' in examples:
            model_inputs["ips_weight"] = examples["ips_weight"]
        
        return model_inputs

    print("⚡ Tokenizing Train dataset...")
    train_tokenized = train_dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=16, 
        remove_columns=train_dataset.column_names,
        desc="Tokenizing Train"
    )

    # [修改 4] 对验证集也做 Tokenize
    print("⚡ Tokenizing Eval dataset...")
    eval_tokenized = eval_dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=4, # 数据少，4核够了
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing Eval"
    )

    # 4. 加载模型
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

    # 5. 训练参数
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,   
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,        
        group_by_length=False,           
        
        learning_rate=2e-5,
        fp16=True,
        
        # [修改 5] 开启评估策略
        eval_strategy="steps",       # 按步数评估
        eval_steps=1000,             # 每 1000 步评估一次
        per_device_eval_batch_size=4, # 验证集 Batch Size
        
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        logging_steps=50,
        
        remove_unused_columns=False,
        report_to="none",
        
        dataset_kwargs={"skip_prepare_dataset": True}
    )

    # 6. 初始化 Trainer
    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=train_tokenized,
        # [修改 6] 传入验证集
        eval_dataset=eval_tokenized, 
        args=training_args,
        processing_class=tokenizer,
        callbacks=[CurriculumCallback()] 
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train()
    
    print("Saving Final Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))

if __name__ == "__main__":
    main()