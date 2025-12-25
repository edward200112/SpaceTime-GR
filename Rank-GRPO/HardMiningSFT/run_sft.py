import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig

# 确保 custom_trainer.py 在同一目录下
from custom_trainer import CoINSFTTrainer

# ================= 配置 =================
MODEL_ID = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_FILE = "./SFT/sft_data/sft_enhanced_train.jsonl"
OUTPUT_DIR = "./SFT/sft_output_sasrec_guided"


# ================= 工具函数 =================
def format_instruction(sample, completion_key="completion"):
    return (
        f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample[completion_key]}<|im_end|>"
    )


def format_augment_instruction(sample):
    return (
        f"<|im_start|>user\n{sample['prompt_augment']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['completion']}<|im_end|>"
    )


def format_negative_instruction(sample):
    return (
        f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['negative_completion']}<|im_end|>"
    )


def get_latest_checkpoint(output_dir: str):
    """
    返回 output_dir 下最新的 checkpoint 路径（checkpoint-xxxx），没有则返回 None
    """
    if not os.path.isdir(output_dir):
        return None

    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            step_str = name.split("-")[-1]
            if step_str.isdigit():
                ckpts.append((int(step_str), os.path.join(output_dir, name)))

    if not ckpts:
        return None

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


# ================= 主流程 =================
def main():
    print("🔄 Initializing SFT (Train-only, Resume-enabled)...")

    # 为了更快（Ampere+ GPU 有收益）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1) 加载 Tokenizer
    print(f"📦 Loading Tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2) 加载数据（只 train，不切 eval）
    print(f"📂 Loading data from {DATA_FILE}...")
    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"   Train samples: {len(raw_dataset)}")

    # 3) 预处理函数（包含正/增广/负样本 + ips_weight）
    def preprocess_function(examples):
        # A. 正样本 (Prompt + Completion)
        texts = [
            format_instruction({"prompt": p, "completion": c}, "completion")
            for p, c in zip(examples["prompt"], examples["completion"])
        ]
        model_inputs = tokenizer(
            texts,
            max_length=1024,
            padding="max_length",
            truncation=True,
        )

        # labels：pad token 对应位置置 -100
        labels = []
        for seq in model_inputs["input_ids"]:
            labels.append(
                [-100 if t == tokenizer.pad_token_id else t for t in seq]
            )
        model_inputs["labels"] = labels

        # B. 增强样本 (Prompt augment + Completion)
        if "prompt_augment" in examples:
            aug_texts = [
                format_augment_instruction({"prompt_augment": pa, "completion": c})
                for pa, c in zip(examples["prompt_augment"], examples["completion"])
            ]
            aug_inputs = tokenizer(
                aug_texts,
                max_length=1024,
                padding="max_length",
                truncation=True,
            )
            model_inputs["augment_input_ids"] = aug_inputs["input_ids"]
            model_inputs["augment_attention_mask"] = aug_inputs["attention_mask"]
        else:
            # 若你的数据没有 prompt_augment 字段，则不生成
            pass

        # C. 负样本 (Prompt + Negative completion)
        if "negative_completion" in examples:
            neg_texts = [
                format_negative_instruction({"prompt": p, "negative_completion": nc})
                for p, nc in zip(examples["prompt"], examples["negative_completion"])
            ]
            neg_inputs = tokenizer(
                neg_texts,
                max_length=1024,
                padding="max_length",
                truncation=True,
            )
            model_inputs["negative_input_ids"] = neg_inputs["input_ids"]
            model_inputs["negative_attention_mask"] = neg_inputs["attention_mask"]
        else:
            # 若你的数据没有 negative_completion 字段，则不生成
            pass

        # D. IPS 权重
        if "ips_weight" in examples:
            model_inputs["ips_weight"] = examples["ips_weight"]
        else:
            model_inputs["ips_weight"] = [1.0] * len(texts)

        return model_inputs

    # 4) Tokenize / Map
    print("✂️ Tokenizing train dataset...")
    num_proc = os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 2 else 1
    tokenized_train = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
    )

    # 5) 加载模型（开启 FlashAttention2）
    print(f"🧠 Loading Model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",  # 避免 auto 误 offload
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 训练时建议关

    # 6) 配置 LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 7) 训练参数：只 train，不 eval；保留 ckpt 以便 resume
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,

        learning_rate=2e-5,
        fp16=True,

        logging_steps=100,

        # ✅ 只训练，不评估
        eval_strategy="no",
        load_best_model_at_end=False,

        # ✅ 保存 ckpt 用于续训
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,

        remove_unused_columns=False,
        report_to="none",

        optim="adamw_torch_fused",
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # ✅ 你已经自己 tokenize 了，跳过 TRL 的 prepare/truncate
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        packing=False,
    )

    # 8) 初始化 Trainer（不传 eval_dataset）
    trainer = CoINSFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        args=training_args,
        processing_class=tokenizer,
    )

    # 9) 自动 resume：如果 OUTPUT_DIR 下有 checkpoint-xxxx 就接着训
    resume_ckpt = get_latest_checkpoint(OUTPUT_DIR)
    if resume_ckpt is None:
        print("🔥 Starting training from scratch (no checkpoint found).")
    else:
        print(f"♻️ Resuming training from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 10) 保存最终模型
    final_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    print(f"💾 Saving Final Model to {final_dir}...")
    trainer.save_model(final_dir)

    print("✅ Done.")


if __name__ == "__main__":
    main()
