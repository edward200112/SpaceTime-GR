import os
import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from custom_trainer import CoINSFTTrainer, CurriculumCallback # [修改] 导入 Callback

# ================= 配置 =================
MODEL_ID = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_balanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output"

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
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 3. 增强版预处理函数
    def preprocess_function(examples):
        # ----------------------------------
        # A. 正样本 (Prompt A + Completion)
        # ----------------------------------
        texts = [
            format_instruction({"prompt": p, "completion": c}, "completion") 
            for p, c in zip(examples['prompt'], examples['completion'])
        ]
        model_inputs = tokenizer(texts, max_length=1024, padding="max_length", truncation=True)
        
        # 构造 Labels (Auto-regressive)
        labels = model_inputs["input_ids"].copy()
        
        # [优化] 构造 Hierarchy Mask (用于课程学习)
        # 逻辑：CoT部分=1, ID前2层=1, ID后2层=0 (在早期阶段)
        # 注意：examples['raw_target_code'] 是 "12 34 56 78"
        hierarchy_masks = []
        
        for i, raw_code in enumerate(examples['raw_target_code']):
            # 默认全开启
            mask = [1] * len(labels[i])
            
            # 找到 ID 在序列中的位置
            # raw_code 格式: "12 34 56 78" -> Tokenizer 后可能会变成多个 token
            # 这是一个简化的启发式查找：在 labels 中倒数寻找 ID 的 token
            # 为了准确，我们假设 ID 是序列的结尾 (CoT -> Target: ID)
            
            try:
                # 将 raw code split 成 4 段: ["12", "34", "56", "78"]
                code_parts = raw_code.split() 
                if len(code_parts) == 4:
                    # 找到最后两个部分 ("56", "78") 的 token 位置并设为 0
                    # 注意：Tokenization 可能会加空格前缀，这里做简单处理
                    # 获取最后几个非 padding token
                    valid_len = sum(model_inputs["attention_mask"][i])
                    
                    # 假设 ID 约占最后 4-8 个 token (取决于 tokenizer 分词粒度)
                    # 简单策略：将最后 2-3 个 token 视为 Fine-grained ID 并 Mask
                    # 更严谨的做法是对 code_parts[-2:] 进行 tokenize 并匹配，这里取近似值：
                    # Qwen Tokenizer 对数字处理较好。
                    # 我们Mask掉最后 2 个有效 token (通常对应 56 78)
                    mask[valid_len-2 : valid_len] = [0, 0] 
            except:
                pass # 格式不对则不 Mask
                
            hierarchy_masks.append(mask)

        # 处理 Labels 中的 Padding
        for i in range(len(labels)):
            labels[i] = [
                -100 if t == tokenizer.pad_token_id else t for t in labels[i]
            ]
        
        model_inputs["labels"] = labels
        model_inputs["hierarchy_mask"] = hierarchy_masks

        # ----------------------------------
        # B. 增强样本 (Prompt B, CoIN)
        # ----------------------------------
        aug_texts = [
            format_augment_instruction({"prompt_augment": pa, "completion": c}) 
            for pa, c in zip(examples['prompt_augment'], examples['completion'])
        ]
        aug_inputs = tokenizer(aug_texts, max_length=1024, padding="max_length", truncation=True)
        model_inputs["augment_input_ids"] = aug_inputs["input_ids"]
        model_inputs["augment_attention_mask"] = aug_inputs["attention_mask"]

        # ----------------------------------
        # C. 负样本 (Prompt A + Negative Item)
        # ----------------------------------
        neg_texts = [
            format_negative_instruction({"prompt": p, "negative_completion": nc}) 
            for p, nc in zip(examples['prompt'], examples['negative_completion'])
        ]
        neg_inputs = tokenizer(neg_texts, max_length=1024, padding="max_length", truncation=True)
        model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
        model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
        
        # D. IPS 权重
        model_inputs["ips_weight"] = examples["ips_weight"]
        
        return model_inputs

    print("Tokenizing dataset...")
    # remove_columns 很重要，防止内存溢出和格式错误
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 4. 加载模型
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

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False, # 必须保留自定义列
        report_to="none"
    )

    # 6. 初始化 Trainer
    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024,
        callbacks=[CurriculumCallback()] # [新增] 注册回调
    )

    print("🚀 Starting CoIN-SFT Training...")
    trainer.train(resume_from_checkpoint=True)
    
    print(f"Saving Final Model to {OUTPUT_DIR}...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))

if __name__ == "__main__":
    main()