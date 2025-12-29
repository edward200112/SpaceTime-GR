# HardMiningSFT/train_stage1_sft_ips_resume_or_lora.py
# 尽管没用，但是加入ips证明了 能够降低top1 的指标，所以保留这份代码，但是会降低oracal指标

import os
import json
import argparse
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
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

PROMPT_RULE = "只输出一个地点名(类别)，不要解释"


# =========================
# Utils
# =========================
SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = SPECIAL_PAT.sub("", s)
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:，。；：")
    return s

def add_rule_to_prompt(p: str) -> str:
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    return p + "\n" + PROMPT_RULE

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

def resolve_adapter_dir(path_or_parent: str) -> str:
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

def ensure_dummy_safetensors_index(ckpt_dir: str):
    idx1 = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    idx2 = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx1) or os.path.exists(idx2):
        return
    dummy = {"metadata": {"total_size": 0}, "weight_map": {}}
    with open(idx2, "w", encoding="utf-8") as f:
        json.dump(dummy, f)
    print(f"🩹 Patched dummy index for Trainer resume: {idx2}")

def read_global_step_from_trainer_state(ckpt_dir: str) -> Optional[int]:
    ts = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(ts):
        return None
    try:
        with open(ts, "r", encoding="utf-8") as f:
            obj = json.load(f)
        gs = obj.get("global_step", None)
        if isinstance(gs, int):
            return gs
    except Exception:
        pass
    return None


# =========================
# IPS Weight builder
# =========================
def build_ips_from_completion_freq(
    data_jsonl: str,
    beta: float = 0.5,
    min_w: float = 0.2,
    max_w: float = 5.0,
    smoothing: float = 1.0,
    max_lines: int = 0,
) -> Dict[str, float]:
    """
    用 completion 的频次来算一个简单 IPS 权重：
      w(c) = ( mean_freq / (freq(c)+smoothing) ) ** beta
    然后 clip 到 [min_w, max_w]
    """
    cnt = Counter()
    n = 0
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            comp = norm_text(d.get("completion", ""))
            if not comp:
                continue
            cnt[comp] += 1
            n += 1
            if max_lines and n >= max_lines:
                break

    if not cnt:
        return {}

    mean_freq = sum(cnt.values()) / max(1, len(cnt))
    wmap = {}
    for comp, f0 in cnt.items():
        w = (mean_freq / (float(f0) + float(smoothing))) ** float(beta)
        w = float(max(min_w, min(max_w, w)))
        wmap[comp] = w

    print(f"✅ IPS(completion_freq) built: uniq={len(wmap)}, mean_freq={mean_freq:.3f}, beta={beta}, clip=[{min_w},{max_w}]")
    return wmap


# =========================
# Collator (FAST) + ips_weight
# =========================
class Stage1CollatorFastIPS:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        starts = [int(f["assistant_start"]) for f in features]
        ips = [float(f.get("ips_weight", 1.0)) for f in features]

        for f in features:
            f.pop("assistant_start", None)
            f.pop("ips_weight", None)

        batch = self.pad(features)
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        labels = input_ids.clone()
        B, L = labels.size()
        for i, s in enumerate(starts):
            s = max(0, min(int(s), L))
            labels[i, :s] = -100

        # ✅ padding mask 用 attention_mask 更稳（避免 pad_id==eos_id 的副作用）
        labels[attn == 0] = -100

        batch["labels"] = labels
        batch["ips_weight"] = torch.tensor(ips, dtype=torch.float32)
        return batch


# =========================
# Weighted Trainer
# =========================
class WeightedSFTTrainer(Trainer):
    """
    对每条样本计算“平均 token loss”，再用 ips_weight 做加权平均：
      loss = sum_i (w_i * loss_i) / sum_i w_i
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ips = inputs.pop("ips_weight", None)
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab = shift_logits.size(-1)
        loss_tok = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        loss_per_ex = (loss_tok * mask).sum(dim=1) / denom  # (B,)

        if ips is None:
            loss = loss_per_ex.mean()
        else:
            ips = ips.to(loss_per_ex.device, dtype=torch.float32)
            ips = ips.clamp(min=1e-6)
            loss = (loss_per_ex * ips).sum() / ips.sum()

        return (loss, outputs) if return_outputs else loss


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

    # resume/init
    ap.add_argument("--resume_trainer", action="store_true")
    ap.add_argument("--resume_ckpt", type=str, default="")
    ap.add_argument("--init_from_adapter", type=str, default="")

    ap.add_argument("--gradient_checkpointing", action="store_true")

    # ✅ IPS
    ap.add_argument("--ips_mode", type=str, default="completion_freq",
                    choices=["none", "field", "completion_freq"],
                    help="none: 不加权; field: 用数据里的 ips_weight; completion_freq: 按 completion 频次自动算 ips")
    ap.add_argument("--ips_beta", type=float, default=0.5)
    ap.add_argument("--ips_min", type=float, default=0.2)
    ap.add_argument("--ips_max", type=float, default=5.0)
    ap.add_argument("--ips_smoothing", type=float, default=1.0)
    ap.add_argument("--ips_count_max_lines", type=int, default=0,
                    help="0=全量统计completion频次；>0则只统计前N行(更快但更粗)")

    # ✅ 只再训 extra_steps（用于 resume 后再跑 2000 step）
    ap.add_argument("--extra_steps", type=int, default=0,
                    help="仅在 --resume_trainer 时生效：从 checkpoint 的 global_step 基础上 +extra_steps")
    # 也可以直接指定 max_steps（优先级更高）
    ap.add_argument("--max_steps", type=int, default=0, help=">0 则按 max_steps 停止训练（Trainer 语义）")

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
    # IPS map (optional)
    # -------------------------
    ips_map = None
    if args.ips_mode == "completion_freq":
        ips_map = build_ips_from_completion_freq(
            data_jsonl=args.data_jsonl,
            beta=args.ips_beta,
            min_w=args.ips_min,
            max_w=args.ips_max,
            smoothing=args.ips_smoothing,
            max_lines=args.ips_count_max_lines,
        )

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
            add_special_tokens=False,  # ✅ 保持一致
        )
        prefix_tok = tokenizer(
            prefix_texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )
        tok["assistant_start"] = [len(ids) for ids in prefix_tok["input_ids"]]

        # ips_weight
        if args.ips_mode == "none":
            tok["ips_weight"] = [1.0] * len(prompts)
        elif args.ips_mode == "field":
            if "ips_weight" in examples:
                tok["ips_weight"] = [float(x) for x in examples["ips_weight"]]
            else:
                tok["ips_weight"] = [1.0] * len(prompts)
        else:
            # completion_freq
            ws = []
            for c in completions:
                c0 = norm_text(c)
                w = ips_map.get(c0, 1.0) if ips_map is not None else 1.0
                ws.append(float(w))
            tok["ips_weight"] = ws

        return tok

    num_proc = max(1, (os.cpu_count() or 2) - 2)
    ds_tok = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing (stage1 + IPS)",
    )

    # -------------------------
    # Load base model
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
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
        if args.resume_ckpt:
            resume_ckpt = resolve_adapter_dir(args.resume_ckpt)
        else:
            auto = find_latest_checkpoint(args.output_dir)
            if not auto:
                raise ValueError(f"--resume_trainer set but no checkpoint-* under {args.output_dir}")
            resume_ckpt = auto

        print(f"🔄 TRUE RESUME from: {resume_ckpt}")
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
        ensure_dummy_safetensors_index(resume_ckpt)

    elif args.init_from_adapter:
        adapter_dir = resolve_adapter_dir(args.init_from_adapter)
        print(f"✅ Init from adapter ONLY: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)

    else:
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
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None

    # 如果 resume 且给了 extra_steps，自动把 max_steps 设置为 (global_step + extra_steps)
    if args.resume_trainer and args.extra_steps and args.extra_steps > 0 and max_steps is None:
        gs = read_global_step_from_trainer_state(resume_ckpt) if resume_ckpt else None
        if gs is not None:
            max_steps = int(gs + args.extra_steps)
            print(f"✅ extra_steps enabled: global_step={gs} => max_steps={max_steps}")
        else:
            # 兜底：读不到就还是按 extra_steps 直接当 max_steps（不完美但可跑）
            max_steps = int(args.extra_steps)
            print(f"⚠️ trainer_state not found, fallback max_steps={max_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs if max_steps is None else 1,

        max_steps=max_steps if max_steps is not None else -1,

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

        ignore_data_skip=False,
    )

    collator = Stage1CollatorFastIPS(tokenizer)

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("🔥 Stage-1 SFT + IPS start ...")
    trainer.train(resume_from_checkpoint=resume_ckpt if args.resume_trainer else None)

    print("💾 Saving Stage-1 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
