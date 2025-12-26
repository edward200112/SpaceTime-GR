# train_stage1_sft.py
import os
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
from peft import LoraConfig, get_peft_model, PeftModel


# =========================
# Qwen chat formatting
# =========================
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"


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
        starts = [int(f["assistant_start"]) for f in features]
        for f in features:
            f.pop("assistant_start", None)

        batch = self.pad(features)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        B, L = labels.size()
        for i, s in enumerate(starts):
            if s < 0:
                s = 0
            if s > L:
                s = L
            labels[i, :s] = -100

        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        batch["labels"] = labels
        return batch


# =========================
# Helpers
# =========================
def resolve_adapter_path(init_from_adapter: str) -> str:
    """
    Accept:
      - a checkpoint dir that contains adapter_config.json
      - a parent dir containing checkpoint-xxxxx subdirs (we pick latest)
    Return the resolved checkpoint dir that contains adapter_config.json
    """
    if not init_from_adapter:
        return ""

    p = init_from_adapter
    if os.path.isfile(p):
        raise ValueError(f"--init_from_adapter should be a directory, got file: {p}")

    # case1: p itself is a checkpoint dir
    if os.path.exists(os.path.join(p, "adapter_config.json")):
        return p

    # case2: p is a parent dir that has checkpoint-* subdirs
    if not os.path.isdir(p):
        raise ValueError(f"Adapter path not found: {p}")

    ckpts = [d for d in os.listdir(p) if d.startswith("checkpoint-")]
    if not ckpts:
        raise ValueError(
            f"No checkpoint-* found under {p}. "
            f"Expected {p}/checkpoint-xxxx/adapter_config.json"
        )
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(p, ckpts[-1])
    if not os.path.exists(os.path.join(latest, "adapter_config.json")):
        raise ValueError(f"Latest checkpoint has no adapter_config.json: {latest}")
    return latest


def quick_check_lora_loaded(model, n=5):
    vals = []
    for name, p in model.named_parameters():
        if "lora_B" in name and p.requires_grad:
            vals.append((name, float(p.detach().abs().mean().cpu())))
            if len(vals) >= n:
                break
    print("🔎 Sample trainable LoRA param abs(mean):")
    for k, v in vals:
        print(f"  {k} => {v:.6e}")
    if not vals:
        print("  (No trainable LoRA params found!)")


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

    # dataloader
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pin_memory", action="store_true")

    # lora (for fresh init)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # load LoRA only (no optimizer/scheduler resume)
    ap.add_argument(
        "--init_from_adapter",
        type=str,
        default="",
        help="Path to checkpoint dir (with adapter_config.json) OR parent dir containing checkpoint-*",
    )

    # speed/memory
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -------------------------
    # Dataset
    # -------------------------
    ds = load_dataset("json", data_files=args.data_jsonl, split="train")

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
        return tok

    num_proc = max(1, (os.cpu_count() or 2) - 2)
    ds_tok = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # -------------------------
    # Load base model
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # -------------------------
    # Attach LoRA
    # 1) If init_from_adapter is set: load adapter weights ONLY (no optimizer/scheduler)
    # 2) Else: fresh LoRA init
    # -------------------------
    if args.init_from_adapter:
        adapter_path = resolve_adapter_path(args.init_from_adapter)
        print(f"✅ Loading LoRA adapter ONLY from: {adapter_path}")
        # IMPORTANT: we load adapter config from adapter_path, do NOT call get_peft_model here
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print("🆕 Initializing new LoRA (fresh) ...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    quick_check_lora_loaded(model)

    # -------------------------
    # TrainingArguments
    # -------------------------
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

        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    collator = Stage1CollatorFast(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("🔥 Stage-1 SFT start (LoRA loaded ONLY; optimizer/scheduler NOT resumed) ...")
    trainer.train(resume_from_checkpoint=None)

    print("💾 Saving Stage-1 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
