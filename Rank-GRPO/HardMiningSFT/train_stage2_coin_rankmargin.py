import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, PeftModel

from custom_trainer_rankmargin import CoINSFTTrainerRankMargin


USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"


def format_chat(prompt: str, completion: str) -> str:
    return USER_PREFIX_TEMPLATE.format(prompt=prompt) + completion + SUFFIX


class Stage2CollatorFast:
    """
    动态 padding，分别对 main/aug/neg pad。
    labels 只对 assistant 部分算 loss：利用 preprocess 算好的 assistant_start
    """
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features):
        # 取出 scalar fields
        starts = [int(f["assistant_start"]) for f in features]
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
            "input_ids": batch_main["input_ids"],
            "attention_mask": batch_main["attention_mask"],
            "labels": batch_main["labels"],

            "augment_input_ids": batch_aug["input_ids"],
            "augment_attention_mask": batch_aug["attention_mask"],

            "negative_input_ids": batch_neg["input_ids"],
            "negative_attention_mask": batch_neg["attention_mask"],

            "ips_weight": ips_weight,
            "coin_weight": coin_weight,
            "coin_margin": coin_margin,
        }
        return out


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--stage1_ckpt", type=str, required=True, help="Stage-1 output dir (LoRA adapter dir)")
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

    # ---- CoIN global config ----
    ap.add_argument("--lambda_coin", type=float, default=0.10)
    ap.add_argument("--default_margin", type=float, default=0.20)

    # ---- hard_level -> (coin_weight, coin_margin) mapping ----
    # 你可以用这个做“难度配比”：hard++ 给更大的 coin_weight 或更小的 margin（更严格）
    ap.add_argument("--w_easy", type=float, default=0.30)
    ap.add_argument("--w_medium", type=float, default=0.60)
    ap.add_argument("--w_hard", type=float, default=1.00)
    ap.add_argument("--w_hardpp", type=float, default=1.20)

    # 注意：这里 margin 是“ranking margin”，越大越难满足 sim_pos - sim_neg >= margin
    ap.add_argument("--m_easy", type=float, default=0.10)
    ap.add_argument("--m_medium", type=float, default=0.15)
    ap.add_argument("--m_hard", type=float, default=0.20)
    ap.add_argument("--m_hardpp", type=float, default=0.25)

    # perf
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")

    # 映射 hard_level -> coin_weight/coin_margin
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

        # main
        prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        main_texts = [pt + c + SUFFIX for pt, c in zip(prefix_texts, comps)]
        tok_main = tokenizer(main_texts, truncation=True, max_length=args.max_length, padding=False)
        tok_prefix = tokenizer(prefix_texts, truncation=True, max_length=args.max_length, padding=False, add_special_tokens=False)

        # aug
        if prompt_aug is None:
            # 兜底：没有 augment 就复用 main prompt
            aug_prompts = prompts
        else:
            aug_prompts = prompt_aug

        aug_prefix_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in aug_prompts]
        aug_texts = [pt + c + SUFFIX for pt, c in zip(aug_prefix_texts, comps)]
        tok_aug = tokenizer(aug_texts, truncation=True, max_length=args.max_length, padding=False)

        # neg
        if neg_comp is None:
            # 兜底：没有 negative 就用 completion（不推荐，但保证脚本不炸）
            neg_comp = comps
        neg_texts = [USER_PREFIX_TEMPLATE.format(prompt=p) + nc + SUFFIX for p, nc in zip(prompts, neg_comp)]
        tok_neg = tokenizer(neg_texts, truncation=True, max_length=args.max_length, padding=False)

        out = {
            "input_ids": tok_main["input_ids"],
            "attention_mask": tok_main["attention_mask"],
            "assistant_start": [len(ids) for ids in tok_prefix["input_ids"]],

            "augment_input_ids": tok_aug["input_ids"],
            "augment_attention_mask": tok_aug["attention_mask"],

            "negative_input_ids": tok_neg["input_ids"],
            "negative_attention_mask": tok_neg["attention_mask"],
        }

        # ips_weight
        if "ips_weight" in examples:
            out["ips_weight"] = examples["ips_weight"]
        else:
            out["ips_weight"] = [1.0] * len(prompts)

        # coin_weight / coin_margin：优先用数据里自带，否则根据 hard_level 映射
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

    # base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # attach LoRA (must match stage1)
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # load stage1 adapter weights
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

        remove_unused_columns=False,
        report_to="none",
    )

    collator = Stage2CollatorFast(tokenizer)

    trainer = CoINSFTTrainerRankMargin(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # set trainer global coin params
    trainer.lambda_coin = float(args.lambda_coin)
    trainer.default_margin = float(args.default_margin)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"🔄 Resuming from {resume_ckpt}")

    print("🔥 Stage-2 CoIN (RankMargin, B+C) start ...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("💾 Saving Stage-2 model ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
