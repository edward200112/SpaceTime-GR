import os
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model


# =========================
# Qwen chat formatting
# =========================
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"


def format_full_text(prompt: str, completion: str) -> str:
    """
    full: <im_start>user ... <im_end>\n<im_start>assistant\n + completion + <im_end>
    """
    return USER_PREFIX_TEMPLATE.format(prompt=prompt) + completion + SUFFIX


# =========================
# Collator (FAST)
# - no O(B*L) substring search
# - use assistant_start computed in preprocess
# =========================
class Stage1CollatorFast:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # assistant_start is a scalar per sample (int)
        starts = [int(f["assistant_start"]) for f in features]
        for f in features:
            f.pop("assistant_start", None)

        batch = self.pad(features)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # mask prefix tokens per sample
        # labels[:, :assistant_start] = -100
        B, L = labels.size()
        for i, s in enumerate(starts):
            if s < 0:
                s = 0
            if s > L:
                s = L
            labels[i, :s] = -100

        # mask padding
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        batch["labels"] = labels
        return batch


# =========================
# Args
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_epochs", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=50)

    ap.add_argument("--resume", action="store_true")

    # dataloader / perf
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")

    # auto batch finder
    ap.add_argument("--auto_find_batch_size", action="store_true")

    # lora
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # grad ckpt (optional)
    ap.add_argument("--gradient_checkpointing", action="store_true")

    return ap.parse_args()


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    last = ckpts[-1]
    p = os.path.join(output_dir, last)
    if os.path.isdir(p) and len(os.listdir(p)) > 0:
        return p
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # make TF32 really effective (safe + faster on Ampere+)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    ds = load_dataset("json", data_files=args.data_jsonl, split="train")

    # Preprocess: compute assistant_start cheaply by tokenizing prefix separately
    def preprocess(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]

        prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        full_texts = [pt + c + SUFFIX for pt, c in zip(prefix_texts, completions)]

        tok = tokenizer(
            full_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        prefix_tok = tokenizer(
            prefix_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

        tok["assistant_start"] = [len(ids) for ids in prefix_tok["input_ids"]]
        tok["length"] = [len(ids) for ids in tok["input_ids"]]
        return tok

    num_proc = max(1, (os.cpu_count() or 2) - 2)
    ds_tok = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        fp16=(not torch.cuda.is_available()),

        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,

        optim="adamw_torch_fused",
        tf32=True,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_persistent_workers=args.persistent_workers,

        # ✅ reduce padding waste + speed up
        # group_by_length=True,
        # length_column_name="length",

        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,

        # ✅ auto backoff if OOM
        auto_find_batch_size=args.auto_find_batch_size,
    )

    collator = Stage1CollatorFast(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"🔄 Resuming from {resume_ckpt}")

    print("🔥 Stage-1 SFT start ...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("💾 Saving Stage-1 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
