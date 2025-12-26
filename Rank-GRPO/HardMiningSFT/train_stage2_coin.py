import os
import argparse
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, PeftModel

from custom_trainer import CoINSFTTrainer


# =========================
# Qwen chat formatting
# =========================
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"


def format_full_text(prompt: str, completion: str) -> str:
    return USER_PREFIX_TEMPLATE.format(prompt=prompt) + completion + SUFFIX


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--stage1_ckpt", type=str, required=True, help="Stage-1 output dir (Trainer saved)")
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_epochs", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)

    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=50)

    ap.add_argument("--resume", action="store_true")

    # perf / dataloader
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--prefetch_factor", type=int, default=4)

    # CoIN settings
    ap.add_argument("--lambda_coin", type=float, default=0.1)
    ap.add_argument("--contrastive_margin", type=float, default=0.5)

    # ✅ difficulty mix -> coin_weight
    ap.add_argument("--coin_weight_mode", type=str, default="hard_level",
                    choices=["hard_level", "gap", "none"],
                    help="how to build per-sample coin_weight")

    # hard_level weights (you can tune to “mix in easier ones”)
    ap.add_argument("--w_easy", type=float, default=1.0)
    ap.add_argument("--w_medium", type=float, default=1.0)
    ap.add_argument("--w_hard", type=float, default=1.0)
    ap.add_argument("--w_hard_plus", type=float, default=1.0)
    ap.add_argument("--w_hard_pp", type=float, default=1.0)
    ap.add_argument("--w_unknown", type=float, default=1.0)

    # gap mode params (teacher_gap usually negative: gap = score_gt - score_neg)
    # gap越接近0 => 越“难”(neg更接近gt)；gap更负 => “更硬”(neg更强)
    # 你可以按你的定义调：这里做一个可控映射到 [w_min, w_max]
    ap.add_argument("--gap_clip_min", type=float, default=-10.0)
    ap.add_argument("--gap_clip_max", type=float, default=-1.0)
    ap.add_argument("--gap_w_min", type=float, default=0.5)
    ap.add_argument("--gap_w_max", type=float, default=1.5)

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


class Stage2CollatorFast:
    """
    动态 padding：main / aug / neg 分别 pad 到 batch 内各自 max_len
    labels 只对 assistant 部分算 loss（靠 assistant_start 快速 mask）
    """
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def _pad_one(self, feats: List[Dict[str, Any]]):
        return self.pad(feats)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        starts = [int(f["assistant_start"]) for f in features]
        coin_w = [float(f.get("coin_weight", 1.0)) for f in features]
        ips_w = [float(f.get("ips_weight", 1.0)) for f in features]

        # pop helper fields
        for f in features:
            f.pop("assistant_start", None)
            # keep coin_weight / ips_weight in Python list form for trainer
            # but remove from padding features
            f.pop("coin_weight", None)
            f.pop("ips_weight", None)

        main_feats = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        aug_feats  = [{"input_ids": f["augment_input_ids"], "attention_mask": f["augment_attention_mask"]} for f in features]
        neg_feats  = [{"input_ids": f["negative_input_ids"], "attention_mask": f["negative_attention_mask"]} for f in features]

        batch_main = self._pad_one(main_feats)
        batch_aug  = self._pad_one(aug_feats)
        batch_neg  = self._pad_one(neg_feats)

        input_ids = batch_main["input_ids"]
        labels = input_ids.clone()

        B, L = labels.size()
        for i, s in enumerate(starts):
            s = max(0, min(int(s), L))
            labels[i, :s] = -100

        pad_id = self.tok.pad_token_id
        labels[labels == pad_id] = -100

        out = {
            "input_ids": batch_main["input_ids"],
            "attention_mask": batch_main["attention_mask"],
            "labels": labels,

            "augment_input_ids": batch_aug["input_ids"],
            "augment_attention_mask": batch_aug["attention_mask"],

            "negative_input_ids": batch_neg["input_ids"],
            "negative_attention_mask": batch_neg["attention_mask"],

            # pass through
            "ips_weight": ips_w,
            "coin_weight": coin_w,
        }
        return out


def build_coin_weight_from_hard_level(level: str, args) -> float:
    if level is None:
        return float(args.w_unknown)
    s = str(level).strip().lower()
    if s in ("easy",):
        return float(args.w_easy)
    if s in ("medium", "mid"):
        return float(args.w_medium)
    if s in ("hard",):
        return float(args.w_hard)
    if s in ("hard+", "hard_plus", "hardplus"):
        return float(args.w_hard_plus)
    if s in ("hard++", "hard_pp", "hardpp", "hardplusplus"):
        return float(args.w_hard_pp)
    return float(args.w_unknown)


def build_coin_weight_from_gap(gap_val, args) -> float:
    """
    将 gap 映射到 [gap_w_min, gap_w_max]
    gap一般是 score_gt - score_neg（你 report 里是负数均值 -6.x）
    你可以决定：更负 => 更hard++，权重更大 or 更小
    这里给一个默认：更负(更强负样本) => 权重更大
    """
    try:
        g = float(gap_val)
    except Exception:
        return 1.0

    g = max(float(args.gap_clip_min), min(float(args.gap_clip_max), g))  # clip
    # normalize to [0,1] where clip_min -> 1.0, clip_max -> 0.0 (更负更大)
    if args.gap_clip_max == args.gap_clip_min:
        t = 0.5
    else:
        t = (args.gap_clip_max - g) / (args.gap_clip_max - args.gap_clip_min)
    w = float(args.gap_w_min) + t * (float(args.gap_w_max) - float(args.gap_w_min))
    return float(w)


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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")

    def preprocess(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        prompts_aug = examples["prompt_augment"]
        neg_comps = examples["negative_completion"]

        # prefix for assistant_start
        prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        main_texts = [pt + c + SUFFIX for pt, c in zip(prefix_texts, completions)]

        aug_prefix = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts_aug]
        aug_texts = [pt + c + SUFFIX for pt, c in zip(aug_prefix, completions)]

        neg_prefix = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        neg_texts = [pt + c + SUFFIX for pt, c in zip(neg_prefix, neg_comps)]

        main = tokenizer(main_texts, truncation=True, max_length=args.max_length, padding=False)
        aug = tokenizer(aug_texts, truncation=True, max_length=args.max_length, padding=False)
        neg = tokenizer(neg_texts, truncation=True, max_length=args.max_length, padding=False)

        prefix_tok = tokenizer(prefix_texts, truncation=True, max_length=args.max_length, padding=False, add_special_tokens=False)
        assistant_start = [len(ids) for ids in prefix_tok["input_ids"]]

        # coin_weight
        coin_w = []
        if args.coin_weight_mode == "none":
            coin_w = [1.0] * len(prompts)
        elif args.coin_weight_mode == "hard_level":
            hard_levels = examples.get("hard_level", [None] * len(prompts))
            for hl in hard_levels:
                coin_w.append(build_coin_weight_from_hard_level(hl, args))
        elif args.coin_weight_mode == "gap":
            gaps = examples.get("gap", examples.get("teacher_gap", [None] * len(prompts)))
            for g in gaps:
                coin_w.append(build_coin_weight_from_gap(g, args))
        else:
            coin_w = [1.0] * len(prompts)

        out = {
            "input_ids": main["input_ids"],
            "attention_mask": main["attention_mask"],
            "assistant_start": assistant_start,

            "augment_input_ids": aug["input_ids"],
            "augment_attention_mask": aug["attention_mask"],

            "negative_input_ids": neg["input_ids"],
            "negative_attention_mask": neg["attention_mask"],

            "coin_weight": coin_w,
        }

        if "ips_weight" in examples:
            out["ips_weight"] = examples["ips_weight"]
        else:
            out["ips_weight"] = [1.0] * len(prompts)

        return out

    num_proc = max(1, (os.cpu_count() or 2) - 2)
    ds_tok = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing(Stage2)",
    )

    # base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # attach LoRA skeleton (must match stage1)
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # load stage1 adapter and keep trainable
    model = PeftModel.from_pretrained(model, args.stage1_ckpt, is_trainable=True)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),

        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,

        optim="adamw_torch_fused",
        tf32=True,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        dataloader_persistent_workers=args.persistent_workers,
        dataloader_prefetch_factor=args.prefetch_factor,

        remove_unused_columns=False,
        report_to="none",
    )

    collator = Stage2CollatorFast(tokenizer)

    trainer = CoINSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # set CoIN hyperparams
    trainer.lambda_coin = float(args.lambda_coin)
    trainer.contrastive_margin = float(args.contrastive_margin)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"🔄 Resuming from {resume_ckpt}")

    print("🔥 Stage-2 CoIN SFT start ...")
    print(f"   coin_weight_mode={args.coin_weight_mode}")
    print(f"   lambda_coin={trainer.lambda_coin} margin={trainer.contrastive_margin}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("💾 Saving Stage-2 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
