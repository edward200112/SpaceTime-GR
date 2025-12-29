# HardMiningSFT/train_stage2_coin_rankmargin.py
import os
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import PeftModel

from custom_trainer_rankmargin import CoINSFTTrainerRankMargin

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"


# -------------------------
# Utils
# -------------------------
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


def ensure_dummy_safetensors_index(ckpt_dir: str):
    """
    LoRA-only checkpoint 有时没有 *safetensors.index.json，
    Trainer resume 会尝试找 index 文件导致报错。补一个空索引即可。
    """
    idx1 = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    idx2 = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx1) or os.path.exists(idx2):
        return
    dummy = {"metadata": {"total_size": 0}, "weight_map": {}}
    with open(idx2, "w", encoding="utf-8") as f:
        json.dump(dummy, f)
    print(f"🩹 Patched dummy index for Trainer resume: {idx2}")


def load_batch_config_from_checkpoint(ckpt_dir: str) -> Optional[Tuple[int, int]]:
    """
    读取 checkpoint 里保存的 TrainingArguments / SFTConfig，拿到：
    - per_device_train_batch_size
    - gradient_accumulation_steps

    PyTorch 2.6+ 默认 weights_only=True，会导致读取 training_args.bin 失败；
    如果 ckpt 是你自己训练出来的（可信），这里显式 weights_only=False。
    """
    ta_bin = os.path.join(ckpt_dir, "training_args.bin")
    if not os.path.exists(ta_bin):
        return None

    # 1) PyTorch 2.6+：显式 weights_only=False（你自己 ckpt 安全）
    try:
        ta = torch.load(ta_bin, map_location="cpu", weights_only=False)
        bs = int(getattr(ta, "per_device_train_batch_size"))
        ga = int(getattr(ta, "gradient_accumulation_steps"))
        return bs, ga
    except TypeError:
        # 老 torch 没有 weights_only 参数
        pass
    except Exception as e:
        print(f"[WARN] torch.load(training_args.bin, weights_only=False) failed: {e}")

    # 2) fallback：老版本 / 兼容尝试（不保证能读）
    try:
        ta = torch.load(ta_bin, map_location="cpu")
        bs = int(getattr(ta, "per_device_train_batch_size"))
        ga = int(getattr(ta, "gradient_accumulation_steps"))
        return bs, ga
    except Exception as e:
        print(f"[WARN] Failed to read training_args.bin for batch config: {e}")
        return None

# -------------------------
# Collator
# -------------------------
class Stage2CollatorFast:
    """
    动态 padding，分别对 main/aug/neg pad。
    labels 只对 assistant 部分算 loss：利用 preprocess 算好的 assistant_start。

    额外：把 assistant_start / augment_assistant_start / negative_assistant_start
    也传给 trainer，用于“只池化 assistant token”的对比学习表征。
    """
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # starts
        starts = [int(f["assistant_start"]) for f in features]
        aug_starts = [int(f["augment_assistant_start"]) for f in features]
        neg_starts = [int(f["negative_assistant_start"]) for f in features]

        # weights (scalar per sample)
        coin_weight = [float(f.get("coin_weight", 1.0)) for f in features]
        coin_margin = [float(f.get("coin_margin", 0.2)) for f in features]
        ips_weight = [float(f.get("ips_weight", 1.0)) for f in features]

        # main
        main_feats = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch_main = self.pad(main_feats)

        labels = batch_main["input_ids"].clone()
        B, L = labels.size()
        for i, s in enumerate(starts):
            s = max(0, min(int(s), L))
            labels[i, :s] = -100
        labels[labels == self.tok.pad_token_id] = -100
        batch_main["labels"] = labels

        # aug
        aug_feats = [{"input_ids": f["augment_input_ids"], "attention_mask": f["augment_attention_mask"]} for f in features]
        batch_aug = self.pad(aug_feats)

        # neg
        neg_feats = [{"input_ids": f["negative_input_ids"], "attention_mask": f["negative_attention_mask"]} for f in features]
        batch_neg = self.pad(neg_feats)

        out = {
            # main
            "input_ids": batch_main["input_ids"],
            "attention_mask": batch_main["attention_mask"],
            "labels": batch_main["labels"],

            # aug
            "augment_input_ids": batch_aug["input_ids"],
            "augment_attention_mask": batch_aug["attention_mask"],

            # neg
            "negative_input_ids": batch_neg["input_ids"],
            "negative_attention_mask": batch_neg["attention_mask"],

            # assistant starts (for assistant-only pooling)
            "assistant_start": torch.tensor(starts, dtype=torch.long),
            "augment_assistant_start": torch.tensor(aug_starts, dtype=torch.long),
            "negative_assistant_start": torch.tensor(neg_starts, dtype=torch.long),

            # weights -> tensor
            "ips_weight": torch.tensor(ips_weight, dtype=torch.float32),
            "coin_weight": torch.tensor(coin_weight, dtype=torch.float32),
            "coin_margin": torch.tensor(coin_margin, dtype=torch.float32),
        }
        return out


# -------------------------
# Args
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--stage1_ckpt", type=str, required=True, help="Stage-1 adapter dir / checkpoint dir")
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_epochs", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-5)

    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=50)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    # ---- CoIN global config ----
    ap.add_argument("--lambda_coin", type=float, default=0.10)
    ap.add_argument("--default_margin", type=float, default=0.20)

    # ✅ 新增：neg_tau（阈值推离版本的负例阈值）
    ap.add_argument(
        "--neg_tau",
        type=float,
        default=0.60,
        help="Negative similarity threshold tau for neg push-away: ReLU(sim_neg - tau). Recommended 0.50~0.65",
    )

    # ✅ view-dropout（self-positive 两视角）
    ap.add_argument(
        "--view_dropout",
        type=float,
        default=0.10,
        help="Dropout prob applied on pos_repr twice to create two views (works even if base model has dropout=0). Recommended 0.05~0.20",
    )

    # ---- hard_level -> (coin_weight, coin_margin) mapping ----
    ap.add_argument("--w_easy", type=float, default=0.30)
    ap.add_argument("--w_medium", type=float, default=0.60)
    ap.add_argument("--w_hard", type=float, default=1.00)
    ap.add_argument("--w_hardpp", type=float, default=1.20)

    ap.add_argument("--m_easy", type=float, default=0.10)
    ap.add_argument("--m_medium", type=float, default=0.15)
    ap.add_argument("--m_hard", type=float, default=0.20)
    ap.add_argument("--m_hardpp", type=float, default=0.25)

    # perf
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--warmup_steps", type=int, default=0)

    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    return ap.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------- resume: auto align batch/accum with checkpoint --------
    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"🔄 Resuming from {resume_ckpt}")
            ensure_dummy_safetensors_index(resume_ckpt)

            ckpt_batch_cfg = load_batch_config_from_checkpoint(resume_ckpt)
            if ckpt_batch_cfg is not None:
                ckpt_bs, ckpt_ga = ckpt_batch_cfg
                if ckpt_bs != int(args.batch_size) or ckpt_ga != int(args.grad_accum):
                    print(
                        f"✅ Align batch config to checkpoint:\n"
                        f"   - CLI  per_device_train_batch_size={args.batch_size}, grad_accum={args.grad_accum}\n"
                        f"   - CKPT per_device_train_batch_size={ckpt_bs}, grad_accum={ckpt_ga}\n"
                        f"   -> Using CKPT values to avoid mismatch warning."
                    )
                    args.batch_size = ckpt_bs
                    args.grad_accum = ckpt_ga

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

    w_map = {
        "easy": float(args.w_easy),
        "medium": float(args.w_medium),
        "hard": float(args.w_hard),
        "hard++": float(args.w_hardpp),
        "hardpp": float(args.w_hardpp),
    }
    m_map = {
        "easy": float(args.m_easy),
        "medium": float(args.m_medium),
        "hard": float(args.m_hard),
        "hard++": float(args.m_hardpp),
        "hardpp": float(args.m_hardpp),
    }

    def preprocess(examples):
        prompts = examples["prompt"]
        comps = examples["completion"]
        prompt_aug = examples.get("prompt_augment", None)
        neg_comp = examples.get("negative_completion", None)

        # -------- main --------
        prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        main_texts = [pt + c + SUFFIX for pt, c in zip(prefix_texts, comps)]

        tok_main = tokenizer(
            main_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )
        tok_prefix = tokenizer(
            prefix_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

        # -------- aug --------
        aug_prompts = prompts if prompt_aug is None else prompt_aug
        aug_prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in aug_prompts]
        aug_texts = [pt + c + SUFFIX for pt, c in zip(aug_prefix_texts, comps)]

        tok_aug = tokenizer(
            aug_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )
        tok_aug_prefix = tokenizer(
            aug_prefix_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

        # -------- neg --------
        if neg_comp is None:
            neg_comp = comps
        neg_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) + nc + SUFFIX for p, nc in zip(prompts, neg_comp)]

        tok_neg = tokenizer(
            neg_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )

        out = {
            "input_ids": tok_main["input_ids"],
            "attention_mask": tok_main["attention_mask"],
            "assistant_start": [len(ids) for ids in tok_prefix["input_ids"]],

            "augment_input_ids": tok_aug["input_ids"],
            "augment_attention_mask": tok_aug["attention_mask"],
            "augment_assistant_start": [len(ids) for ids in tok_aug_prefix["input_ids"]],

            "negative_input_ids": tok_neg["input_ids"],
            "negative_attention_mask": tok_neg["attention_mask"],
            "negative_assistant_start": [len(ids) for ids in tok_prefix["input_ids"]],
        }

        # ips_weight
        out["ips_weight"] = examples["ips_weight"] if "ips_weight" in examples else [1.0] * len(prompts)

        # coin_weight / coin_margin
        if "coin_weight" in examples:
            out["coin_weight"] = examples["coin_weight"]
        else:
            hls = examples.get("hard_level", ["hard"] * len(prompts))
            out["coin_weight"] = [w_map.get(str(h), 1.0) for h in hls]

        if "coin_margin" in examples:
            out["coin_margin"] = examples["coin_margin"]
        else:
            hls = examples.get("hard_level", ["hard"] * len(prompts))
            out["coin_margin"] = [m_map.get(str(h), float(args.default_margin)) for h in hls]

        return out

    ds_tok = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=max(1, (os.cpu_count() or 2) - 2),
        desc="Tokenizing stage2",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )

    # load stage1 LoRA
    model = PeftModel.from_pretrained(model, args.stage1_ckpt, is_trainable=True)
    model.print_trainable_parameters()
    print("peft adapters:", list(getattr(model, "peft_config", {}).keys()))

    warmup_steps = int(args.warmup_steps)
    warmup_ratio = float(args.warmup_ratio) if warmup_steps <= 0 else 0.0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),

        learning_rate=float(args.lr),

        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,

        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),

        logging_steps=int(args.logging_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        save_total_limit=3,

        optim="adamw_torch_fused",
        tf32=True,

        dataloader_num_workers=int(args.num_workers),
        dataloader_pin_memory=bool(args.pin_memory),

        remove_unused_columns=False,
        report_to="none",
        seed=int(args.seed),

        max_grad_norm=float(args.max_grad_norm),
        ignore_data_skip=False,
    )

    collator = Stage2CollatorFast(tokenizer)

    trainer = CoINSFTTrainerRankMargin(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # pass config into trainer
    trainer.lambda_coin = float(args.lambda_coin)
    trainer.default_margin = float(args.default_margin)

    trainer.view_dropout = float(args.view_dropout)
    trainer.neg_tau = float(args.neg_tau)

    print(f"[INFO] trainer.view_dropout={trainer.view_dropout}")
    print(f"[INFO] trainer.neg_tau={trainer.neg_tau}")
    print(f"[INFO] batch_size={args.batch_size}, grad_accum={args.grad_accum}")

    print("🔥 Stage-2 CoIN (RankMargin + view-dropout) start ...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("💾 Saving Stage-2 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
