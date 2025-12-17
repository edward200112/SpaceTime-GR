import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from custom_trainer import CoINSFTTrainer
import os

# ================= 配置 =================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "./sft_data/sft_enhanced_train.jsonl"
OUTPUT_DIR = "./sft_output"

def format_instruction(sample):
    # 格式化正样本
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['completion']}<|im_end|>"

def format_negative_instruction(sample):
    # 格式化负样本 (Prompt 一样，但 Completion 是错的)
    return f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['negative_completion']}<|im_end|>"

def main():
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 对比学习通常 padding 在右侧比较好处理 hidden state

    # 2. 加载数据集
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 3. 自定义预处理函数 (同时 Tokenize 正负样本)
    def preprocess_function(examples):
        # A. 处理正样本
        texts = [format_instruction({"prompt": p, "completion": c}) for p, c in zip(examples['prompt'], examples['completion'])]
        model_inputs = tokenizer(texts, max_length=1024, padding="max_length", truncation=True)
        
        # B. 处理 Labels (SFTTrainer 需要 labels)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        # 将 padding 部分 label 设为 -100
        for i in range(len(model_inputs["labels"])):
            model_inputs["labels"][i] = [
                -100 if token == tokenizer.pad_token_id else token 
                for token in model_inputs["labels"][i]
            ]

        # C. 处理负样本 (为 CoIN Loss)
        neg_texts = [format_negative_instruction({"prompt": p, "negative_completion": nc}) for p, nc in zip(examples['prompt'], examples['negative_completion'])]
        neg_inputs = tokenizer(neg_texts, max_length=1024, padding="max_length", truncation=True)
        
        model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
        model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
        
        # D. 传递 IPS 权重
        model_inputs["ips_weight"] = examples["ips_weight"]
        
        return model_inputs

    print("Tokenizing and formatting dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 4. 加载模型
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 开启 LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4, # 1.5B + CoIN (双倍Forward) 显存消耗较大，调小 Batch
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False, # 关键！防止 Trainer 删掉 ips_weight 和 negative_input_ids
        report_to="none"
    )

    # 6. 初始化自定义 Trainer
    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        # SFTTrainer 特定参数 (虽然我们手动 map 了，但保留这些配置)
        packing=False,
        max_seq_length=1024,
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train()
    
    print("Saving Final Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))

if __name__ == "__main__":
    main()