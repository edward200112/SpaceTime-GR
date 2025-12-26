# HardMiningSFT/train_stage1_sft_resume_or_lora.py
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
from peft import LoraConfig, get_peft_model, PeftModel


# =========================
# Qwen chat formatting
# =========================
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"

# 你要求加的约束
PROMPT_RULE = "只输出一个地点名(类别)，不要解释"


# =========================
# Collator (FAST)
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
            s = max(0, min(int(s), L))
            labels[i, :s] = -100

        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


# =========================
# Helpers
# =========================
def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(output_dir, ckpts[-1])
    if os.path.isdir(latest):
        return latest
    return None


def ensure_dummy_safetensors_index(ckpt_dir: str):
    """
    Trainer resume 时会找：
      - pytorch_model.bin.index.json
      - model.safetensors.index.json
    但 LoRA-only checkpoint 没有这些。
    我们补一个“空索引”的 model.safetensors.index.json 让 Trainer 不报错。
    （不会覆盖你的 base model，因为 weight_map 为空）
    """
    idx1 = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    idx2 = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx1) or os.path.exists(idx2):
        return

    dummy = {
        "metadata": {"total_size": 0},
        "weight_map": {}
    }
    with open(idx2, "w", encoding="utf-8") as f:
        json.dump(dummy, f)
    print(f"🩹 Patched dummy index for Trainer resume: {idx2}")


def add_rule_to_prompt(p: str) -> str:
    """
    把规则加到 prompt 末尾（避免破坏原 prompt 格式）
    """
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    # 用换行追加，最稳
    return p + "\n" + PROMPT_RULE


def resolve_adapter_dir(path_or_parent: str) -> str:
    """
    允许传：
      - checkpoint-xxxxx（里面有 adapter_config.json）
      - parent dir（里面有 checkpoint-*，自动取最新）
    """
    if not path_or_parent:
        return ""
    p = path_or_parent
    if os.path.isfile(p):
        raise ValueError(f"Adapter path must be a directory, got file: {p}")

    if os.path.isdir(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
        return p

    if os.path.isdir(p):
        latest = find_latest_checkpoint(p)
        if latest and os.path.exists(os.path.join(latest, "adapter_config.json")):
            return latest

    raise ValueError(f"Cannot resolve adapter directory from: {path_or_parent}")


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

    # attn
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    # LoRA init (fresh)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # mode
    ap.add_argument("--resume_trainer", action="store_true",
                    help="TRUE resume: restore optimizer/scheduler/global_step from output_dir latest checkpoint")
    ap.add_argument("--resume_ckpt", type=str, default="",
                    help="Optional explicit checkpoint dir to resume from (overrides auto latest)")
    ap.add_argument("--init_from_adapter", type=str, default="",
                    help="Load LoRA adapter ONLY from this dir (no optimizer/scheduler resume). "
                         "Can be checkpoint dir or parent with checkpoint-*")

    ap.add_argument("--gradient_checkpointing", action="store_true")

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
        prompts = [add_rule_to_prompt(p) for p in examples["prompt"]]
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
    # Decide mode & load LoRA
    # -------------------------
    resume_ckpt = None
    if args.resume_trainer:
        # resume from explicit or latest under output_dir
        if args.resume_ckpt:
            resume_ckpt = resolve_adapter_dir(args.resume_ckpt)
        else:
            auto = find_latest_checkpoint(args.output_dir)
            if not auto:
                raise ValueError(f"--resume_trainer set but no checkpoint-* under {args.output_dir}")
            resume_ckpt = auto

        print(f"🔄 TRUE RESUME from: {resume_ckpt} (optimizer/scheduler/global_step restored)")

        # IMPORTANT: attach adapter weights first
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)

        # Patch dummy index so Trainer won't crash in _load_from_checkpoint
        ensure_dummy_safetensors_index(resume_ckpt)

    elif args.init_from_adapter:
        # LoRA-only init, no optimizer/scheduler resume
        adapter_dir = resolve_adapter_dir(args.init_from_adapter)
        print(f"✅ Init from adapter ONLY: {adapter_dir} (no optimizer/scheduler resume)")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)

    else:
        # fresh LoRA
        print("🆕 Initializing new LoRA (fresh) ...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

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

        # ✅ resume 时跳过已经训练过的数据（Trainer 自己会处理）
        ignore_data_skip=False,
    )

    collator = Stage1CollatorFast(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("🔥 Stage-1 SFT start ...")
    trainer.train(resume_from_checkpoint=resume_ckpt if args.resume_trainer else None)

    print("💾 Saving Stage-1 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
