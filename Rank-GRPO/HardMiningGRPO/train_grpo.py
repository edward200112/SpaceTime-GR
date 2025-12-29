# HardMiningGRPO/train_grpo.py
import os
import sys
import argparse
import random
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from trl import GRPOTrainer, GRPOConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec
from HardMiningGRPO.reward_sasrec import SasrecScorer, ResolverConfig, make_reward_fn

# fallback template（当 tokenizer 没有 chat_template 时）
USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# 如果你的 prompt 里已经包含候选+规则，这里不强加，只做兜底
RULE_FALLBACK = "只能从下面候选列表中选择一个，并且原样只输出一个地点名(类别)，不要解释。"


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)

    # GRPO
    ap.add_argument("--max_length", type=int, default=1280)
    ap.add_argument("--max_new_tokens", type=int, default=32)  # ✅ 默认拉大，避免截断导致 unknown 飙升
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--num_train_epochs", type=int, default=1)

    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)  # ✅ 默认稍低一点更稳

    # reward cfg
    ap.add_argument("--format_bonus", type=float, default=0.05)
    ap.add_argument("--in_candidates_bonus", type=float, default=0.10)  # ✅ 关键：先学会只从候选里选
    ap.add_argument("--match_reward", type=float, default=1.0)

    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--softmax_temp", type=float, default=1.0)

    ap.add_argument("--extra_text_penalty", type=float, default=0.05)
    ap.add_argument("--unknown_penalty", type=float, default=0.10)
    ap.add_argument("--prefix_penalty", type=float, default=0.05)
    ap.add_argument("--incomplete_penalty", type=float, default=0.10)  # ✅ 括号没闭合/截断的罚

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
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return USER_PREFIX_TEMPLATE.format(prompt=user_text)


def load_sasrec_from_ckpt(
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int = 50,
    embed_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
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

    # sanity check
    if "item_emb.weight" in state_dict:
        ckpt_dim = int(state_dict["item_emb.weight"].shape[1])
        if ckpt_dim != int(embed_dim):
            raise ValueError(f"SASRec embed_dim mismatch: ckpt_dim={ckpt_dim} vs args.embed_dim={embed_dim}")
    if "pos_emb.weight" in state_dict:
        ckpt_len = int(state_dict["pos_emb.weight"].shape[0])
        if ckpt_len != int(max_len):
            raise ValueError(f"SASRec max_len mismatch: ckpt_max_len={ckpt_len} vs args.max_len={max_len}")

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
    return sasrec


def _get_field(ex: Dict[str, Any], keys, required=True, default=None):
    for k in keys:
        if k in ex:
            return ex[k]
    if required:
        raise KeyError(f"missing required field, tried: {keys}")
    return default


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # TRL constraint
    generation_batch_size = int(args.per_device_bs) * int(args.grad_accum)
    if generation_batch_size % int(args.num_generations) != 0:
        raise ValueError(
            f"[BAD CONFIG] per_device_bs*grad_accum={generation_batch_size} "
            f"must be divisible by num_generations={args.num_generations}."
        )

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"  # ✅ 候选在尾部，必须保尾部

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cuda",
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()

    # stop tokens
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

    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    def format_ex(ex: Dict[str, Any]) -> Dict[str, Any]:
        raw = _get_field(ex, ["prompt"])
        # 兜底：如果有人生成数据漏了规则
        if "只输出一个地点名" not in raw:
            raw = raw.rstrip() + "\n" + RULE_FALLBACK

        cand_nc = _get_field(ex, ["candidate_namecats", "candidates_namecat", "candidates_namecats"])
        cand_it = _get_field(ex, ["candidate_item_ids", "candidates_item_ids", "candidates_ids"])

        out = {
            "prompt": build_chat_prompt(tok, raw) if args.use_chat_template else raw,
            "prompt_raw": raw,
            "history_item_ids": _get_field(ex, ["history_item_ids"]),
            "target_item_id": _get_field(ex, ["target_item_id"]),
            "target_namecat": _get_field(ex, ["target_namecat"]),
            "candidate_namecats": cand_nc,
            "candidate_item_ids": cand_it,
        }
        return out

    # map 用 num_proc 加速（不影响 determinism）
    num_proc = max(1, (os.cpu_count() or 8) // 2)
    train_ds = train_ds.map(format_ex, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_ds = eval_ds.map(format_ex, remove_columns=eval_ds.column_names, num_proc=num_proc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sasrec = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )
    scorer = SasrecScorer(sasrec_model=sasrec, device=device)

    r_cfg = ResolverConfig(
        format_bonus=float(args.format_bonus),
        in_candidates_bonus=float(args.in_candidates_bonus),
        match_reward=float(args.match_reward),
        alpha=float(args.alpha),
        softmax_temp=float(args.softmax_temp),
        extra_text_penalty=float(args.extra_text_penalty),
        unknown_penalty=float(args.unknown_penalty),
        prefix_penalty=float(args.prefix_penalty),
        incomplete_penalty=float(args.incomplete_penalty),
        debug_log_every_steps=int(args.debug_log_every_steps),
        debug_num_show=int(args.debug_num_show),
        debug_dump_jsonl=str(args.debug_dump_jsonl),
        debug_print_full_completion=bool(args.debug_print_full_completion),
    )
    reward_fn = make_reward_fn(scorer, r_cfg)

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
