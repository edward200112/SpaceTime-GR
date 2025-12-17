import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from custom_trainer import CoINSFTTrainer # 导入自定义 Trainer

# ================= 配置 =================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_enhanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output"

def format_instruction(sample, completion_key="completion"):
    # 构造 ChatML 格式
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample[completion_key]}<|im_end|>"

def main():
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 对比学习需要对齐 Hidden States

    # 2. Load Dataset
    print(f"Loading data from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 3. Preprocess (关键：构造正负样本对)
    def preprocess_function(examples):
        # A. 正样本 Tokenize
        pos_texts = [format_instruction(ex, "completion") for ex in examples['completion']] # Hacky access to row
        # HuggingFace dataset map 传入的是 batch dict
        pos_texts = [
            format_instruction({"prompt": p, "completion": c}, "completion") 
            for p, c in zip(examples['prompt'], examples['completion'])
        ]
        
        model_inputs = tokenizer(pos_texts, max_length=1024, padding="max_length", truncation=True)
        
        # 构造 Labels (Mask User Part) - 简化起见这里全量训练，SFTTrainer 会自动处理 Response Masking
        # 如果使用 DataCollatorForCompletionOnlyLM 会更好，但这里手动处理 labels 兼容性强
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        for i in range(len(model_inputs["labels"])):
            model_inputs["labels"][i] = [
                -100 if t == tokenizer.pad_token_id else t for t in model_inputs["labels"][i]
            ]

        # B. 负样本 Tokenize (用于 CoIN Loss)
        neg_texts = [
            format_instruction({"prompt": p, "negative_completion": nc}, "negative_completion")
            for p, nc in zip(examples['prompt'], examples['negative_completion'])
        ]
        neg_inputs = tokenizer(neg_texts, max_length=1024, padding="max_length", truncation=True)
        
        # 将负样本 ID 存入 model_inputs，Trainer 会接收到
        model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
        model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
        
        # C. 传递 IPS 权重
        model_inputs["ips_weight"] = examples["ips_weight"]
        
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 4. Load Model & LoRA
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

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4, # CoIN 需要双倍 Forward，显存压力大，调小 Batch
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        # 关键：不要移除自定义列，否则 Trainer 这里的 inputs 会丢掉 ips_weight
        remove_unused_columns=False, 
        report_to="none"
    )

    # 6. Initialize Custom Trainer
    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        packing=False, # 不使用 Packing，保证 Contrastive Loss 序列对齐
        max_seq_length=1024,
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    print("✅ Training Finished.")

if __name__ == "__main__":
    main()