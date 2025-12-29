# HardMiningGRPO/train_grpo.py
import os
import sys
import argparse
import random
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from trl import GRPOTrainer, GRPOConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec
from HardMiningGRPO.reward_sasrec import (
    load_json,
    SasrecResolver,
    ResolverConfig,
    make_reward_fn,
)

RULE = "只输出一个地点名(类别)，不要解释"
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--namecat2item_disamb", required=True)
    ap.add_argument("--name2item_disamb", required=True)

    # ✅ 新增：全量映射（用于“条件注入 target”，避免 top50 截断导致 target 不在候选）
    ap.add_argument("--namecat2item_all", default="", help="optional: namecat2item_ids_all.json")
    ap.add_argument("--name2item_all", default="", help="optional: name2item_ids_all.json")

    # 可选：如果你后面想做更强的 canonical（暂时不强依赖）
    ap.add_argument("--gmap_id2namecat", default="", help="optional: gmap_id2namecat.json")

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)

    # GRPO config
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--num_train_epochs", type=int, default=1)

    # group sampling
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)

    # reward weights
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--format_bonus", type=float, default=0.05)
    ap.add_argument("--item_match_bonus", type=float, default=0.2)
    ap.add_argument("--n_neg_sample", type=int, default=256)
    ap.add_argument("--softmax_temp", type=float, default=1.0)

    # penalties
    ap.add_argument("--extra_text_penalty", type=float, default=0.05)
    ap.add_argument("--unknown_penalty", type=float, default=0.05)
    ap.add_argument("--prefix_penalty", type=float, default=0.0)

    # disamb controls
    ap.add_argument("--max_disamb_candidates", type=int, default=64)
    ap.add_argument("--ensure_target_in_candidates", action="store_true")

    # sasrec arch
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)
    ap.add_argument("--sasrec_max_len", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    # debug
    ap.add_argument("--debug_log_every_steps", type=int, default=0)
    ap.add_argument("--debug_num_show", type=int, default=5)
    ap.add_argument("--debug_dump_jsonl", type=str, default="")
    ap.add_argument("--debug_print_full_completion", action="store_true")

    ap.add_argument("--use_chat_template", action="store_true")

    return ap.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat_prompt(tok, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return USER_PREFIX_TEMPLATE.format(prompt=user_text)


def load_sasrec_from_ckpt(
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int,
    embed_dim: int,
    num_blocks: int,
    num_heads: int,
    dropout: float,
):
    import pickle

    with open(sasrec_pkl, "rb") as f:
        obj = pickle.load(f)
    n_items = int(obj["n_items"])

    try:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu")

    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state_dict = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state_dict = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state_dict = ckpt_obj["model"]
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    # sanity
    if "item_emb.weight" in state_dict:
        ckpt_dim = int(state_dict["item_emb.weight"].shape[1])
        if ckpt_dim != int(embed_dim):
            raise ValueError(f"SASRec embed_dim mismatch: ckpt_dim={ckpt_dim} vs embed_dim={embed_dim}")
    if "pos_emb.weight" in state_dict:
        ckpt_len = int(state_dict["pos_emb.weight"].shape[0])
        if ckpt_len != int(max_len):
            raise ValueError(f"SASRec max_len mismatch: ckpt_len={ckpt_len} vs max_len={max_len}")

    class _Args:
        pass

    a = _Args()
    a.device = device
    a.max_len = int(max_len)
    a.embed_dim = int(embed_dim)
    a.num_blocks = int(num_blocks)
    a.num_heads = int(num_heads)
    a.dropout = float(dropout)

    sasrec = SASRec(item_num=n_items, args=a).to(device)
    sasrec.load_state_dict(state_dict, strict=True)
    sasrec.eval()
    for p in sasrec.parameters():
        p.requires_grad_(False)

    print(f"[OK] loaded SASRec: n_items={n_items}, max_len={a.max_len}, dim={a.embed_dim}, "
          f"blocks={a.num_blocks}, heads={a.num_heads}, dropout={a.dropout}")
    return sasrec, n_items


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # TRL constraint: generation_batch_size must be divisible by num_generations
    generation_batch_size = int(args.per_device_bs) * int(args.grad_accum)
    if generation_batch_size % int(args.num_generations) != 0:
        raise ValueError(
            f"[BAD CONFIG] per_device_bs*grad_accum={generation_batch_size} "
            f"must be divisible by num_generations={args.num_generations}"
        )

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # model + lora
    device_map = "cuda" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=device_map,
        dtype=dtype,  # 你环境里 torch_dtype 已 deprecated，这里用 dtype
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()

    # generation config tokens
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(int(tok.eos_token_id))
    for t in ["<|im_end|>", "</s>"]:
        try:
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(int(tid))
        except Exception:
            pass
    eos_ids = sorted(set(eos_ids)) if eos_ids else None
    if eos_ids is not None:
        model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = int(tok.pad_token_id)

    # dataset
    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    def format_ex(ex):
        raw = ex["prompt"]
        if RULE not in raw:
            raw = raw.rstrip() + "\n" + RULE

        prompt = build_chat_prompt(tok, raw) if args.use_chat_template else raw

        out = {
            "prompt": prompt,
            "prompt_raw": raw,  # debug用
            "history_item_ids": ex["history_item_ids"],
            "target_item_id": ex["target_item_id"],
        }
        if "target_namecat" in ex:
            out["target_namecat"] = ex["target_namecat"]
        return out

    train_ds = train_ds.map(format_ex, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_ex, remove_columns=eval_ds.column_names)

    # sasrec + resolver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sasrec, n_items = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )

    namecat2item_disamb = load_json(args.namecat2item_disamb)
    name2item_disamb = load_json(args.name2item_disamb)

    namecat2item_all = load_json(args.namecat2item_all) if args.namecat2item_all else {}
    name2item_all = load_json(args.name2item_all) if args.name2item_all else {}

    resolver = SasrecResolver(
        sasrec_model=sasrec,
        n_items=n_items,
        namecat2item_disamb=namecat2item_disamb,
        name2item_disamb=name2item_disamb,
        namecat2item_all=namecat2item_all,
        name2item_all=name2item_all,
        device=device,
    )

    r_cfg = ResolverConfig(
        n_neg_sample=int(args.n_neg_sample),
        softmax_temp=float(args.softmax_temp),
        alpha=float(args.alpha),
        format_bonus=float(args.format_bonus),
        item_match_bonus=float(args.item_match_bonus),

        extra_text_penalty=float(args.extra_text_penalty),
        unknown_penalty=float(args.unknown_penalty),
        prefix_penalty=float(args.prefix_penalty),

        max_disamb_candidates=int(args.max_disamb_candidates),
        ensure_target_in_candidates=bool(args.ensure_target_in_candidates),

        debug_log_every_steps=int(args.debug_log_every_steps),
        debug_num_show=int(args.debug_num_show),
        debug_dump_jsonl=str(args.debug_dump_jsonl),
        debug_print_full_completion=bool(args.debug_print_full_completion),
    )
    reward_fn = make_reward_fn(resolver, r_cfg)

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.per_device_bs),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.num_train_epochs),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=int(args.seed),

        max_prompt_length=int(args.max_length),
        max_completion_length=int(args.max_new_tokens),
        num_generations=int(args.num_generations),
        temperature=float(args.temperature),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        reward_funcs=reward_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("✅ GRPO done:", args.output_dir)


if __name__ == "__main__":
    main()
