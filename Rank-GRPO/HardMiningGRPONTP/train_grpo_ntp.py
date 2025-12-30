# HardMiningGRPO/train_grpo_ntp.py
import os
import sys
import argparse
import random
import re
from typing import Any, Dict, List, Tuple
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec
from HardMiningGRPO.reward_ntp_itemid import SasrecScorer, ResolverConfig as ResolverConfigP1, make_reward_fn as make_reward_fn_p1
from HardMiningGRPO.reward_ntp_phase2_recall import ResolverConfig as ResolverConfigP2, make_reward_fn as make_reward_fn_p2



USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# NTP rule: output item_id only
RULE_FALLBACK = "只输出一个 item_id（整数），不要解释，也不要输出其他文字。"

# 用于从旧 prompt 中切掉候选段（你原数据 prompt 末尾大概率有候选列表）
CAND_HEADER_CANON = "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）："
CAND_HEADERS = [
    CAND_HEADER_CANON,
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）:",
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）",
    "候选地点：",
    "候选地点:",
]

# 旧约束行（含“只能从候选”之类）在 NTP 下要去掉
OLD_CONSTRAINT_PATTERNS = [
    r".*只能从.*候选.*",
    r".*候选.*选\s*1\s*个.*",
    r".*in\s*candidates.*",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)

    # phase control
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="1: prompt no candidates, shaping uses candidate_item_ids; "
                         "2: shaping uses teacher_top_item_ids (recommended); "
                         "3: more open, weaken pool bonus.")

    # GRPO
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=8)   # 输出 item_id 足够短
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--num_train_epochs", type=int, default=1)

    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)

    # prompt control
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--strip_candidates_from_prompt", action=argparse.BooleanOptionalAction, default=True)

    # reward cfg
    ap.add_argument("--format_bonus", type=float, default=0.05)
    ap.add_argument("--exists_bonus", type=float, default=0.10)

    ap.add_argument("--correct_reward", type=float, default=2.0)
    ap.add_argument("--wrong_penalty", type=float, default=0.3)
    ap.add_argument("--unknown_penalty", type=float, default=0.6)

    # teacher shaping
    ap.add_argument("--rank_shaping_weight", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--teacher_clip", type=float, default=5.0)
    ap.add_argument("--teacher_pool_k", type=int, default=200,
                    help="Phase2/3: teacher_top_item_ids length target (offline precompute recommended).")

    # penalties
    ap.add_argument("--extra_text_penalty", type=float, default=0.05)
    ap.add_argument("--prefix_penalty", type=float, default=0.05)
    ap.add_argument("--duplicate_penalty", type=float, default=0.02)

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


def _get_field(ex: Dict[str, Any], keys, required=True, default=None):
    for k in keys:
        if k in ex:
            return ex[k]
    if required:
        raise KeyError(f"missing required field, tried: {keys}")
    return default


CAND_CUT_RE = re.compile(r"(候选|备选|candidate|candidates|候选poi|poi列表|候选列表)", re.IGNORECASE)

def strip_candidate_block(raw_prompt: str) -> str:
    p = (raw_prompt or "").rstrip()
    lines = p.splitlines()

    cut = None
    for i, ln in enumerate(lines):
        if CAND_CUT_RE.search(ln):
            cut = i
            break

    if cut is not None:
        lines = lines[:cut]

    # 再清一次老约束行
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if ("只能从" in s and "候选" in s) or ("选 1 个" in s and "候选" in s):
            continue
        out.append(ln)

    return "\n".join(out).rstrip()


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
    tok.truncation_side = "left"

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

    # generation config
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

    def format_ex(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw = _get_field(ex, ["prompt"])
        raw = (raw or "").rstrip()

        if args.strip_candidates_from_prompt:
            raw = strip_candidate_block(raw)

        # append NTP rule
        if "只输出一个 item_id" not in raw:
            raw = (raw + "\n" + RULE_FALLBACK).rstrip()

        # fields
        hist = _get_field(ex, ["history_item_ids"])
        tgt_id = int(_get_field(ex, ["target_item_id"]))

        cand_ids = _get_field(ex, ["candidate_item_ids", "candidates_item_ids", "candidates_ids"], required=False, default=[])
        if cand_ids is None:
            cand_ids = []
        cand_ids = [int(x) for x in cand_ids] if isinstance(cand_ids, list) else []

        teacher_top = _get_field(ex, ["teacher_top_item_ids"], required=False, default=[])
        if teacher_top is None:
            teacher_top = []
        teacher_top = [int(x) for x in teacher_top] if isinstance(teacher_top, list) else []

        prompt = build_chat_prompt(tok, raw) if args.use_chat_template else raw

        return {
            "prompt": prompt,
            "prompt_raw": raw,
            "history_item_ids": hist,
            "target_item_id": tgt_id,
            "candidate_item_ids": cand_ids,         # Phase1 shaping 用
            "teacher_top_item_ids": teacher_top,     # Phase2 shaping 用（可为空）
        }

    num_proc = max(1, (os.cpu_count() or 8) // 2)
    train_ds = train_ds.map(format_ex, with_indices=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_ds = eval_ds.map(format_ex, with_indices=True, remove_columns=eval_ds.column_names, num_proc=num_proc)

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
    scorer = SasrecScorer(sasrec_model=sasrec, n_items=n_items, device=device)

    # phase -> pool mode
    if args.phase == 1:
        pool_mode = "candidate"
        pool_bonus = 0.02
    elif args.phase == 2:
        pool_mode = "teacher_top"
        pool_bonus = 0.01
    else:
        pool_mode = "teacher_top"
        pool_bonus = 0.0  # Phase3 更开放

    if args.phase == 1:
        r_cfg = ResolverConfigP1(
            phase=1,
            pool_mode="candidate",
            teacher_pool_k=int(args.teacher_pool_k),

            format_bonus=float(args.format_bonus),
            exists_bonus=float(args.exists_bonus),

            correct_reward=float(args.correct_reward),
            wrong_penalty=float(args.wrong_penalty),
            unknown_penalty=float(args.unknown_penalty),

            rank_shaping_weight=float(args.rank_shaping_weight),
            alpha=float(args.alpha),
            teacher_clip=float(args.teacher_clip),
            pool_in_bonus=0.02,

            extra_text_penalty=float(args.extra_text_penalty),
            prefix_penalty=float(args.prefix_penalty),
            duplicate_penalty=float(args.duplicate_penalty),

            debug_log_every_steps=int(args.debug_log_every_steps),
            debug_num_show=int(args.debug_num_show),
            debug_dump_jsonl=str(args.debug_dump_jsonl),
            debug_print_full_completion=bool(args.debug_print_full_completion),
        )
        reward_fn = make_reward_fn_p1(scorer, r_cfg)

    else:
        # ✅ Phase2/3：更 recall 的 dense teacher shaping（不看 target 的 teacher-rank）
        r_cfg = ResolverConfigP2(
            teacher_pool_k=int(args.teacher_pool_k),

            format_bonus=float(args.format_bonus),
            exists_bonus=float(args.exists_bonus),

            correct_reward=float(args.correct_reward),
            wrong_penalty=float(args.wrong_penalty),
            unknown_penalty=float(args.unknown_penalty),

            alpha=float(args.alpha),
            teacher_rank_weight=float(args.rank_shaping_weight),  # 复用你现有参数名
            out_of_teacher_penalty=0.02 if args.phase == 2 else 0.0,  # Phase3 更开放

            extra_text_penalty=float(args.extra_text_penalty),
            prefix_penalty=float(args.prefix_penalty),
            duplicate_penalty=float(args.duplicate_penalty),

            debug_log_every_steps=int(args.debug_log_every_steps),
            debug_num_show=int(args.debug_num_show),
            debug_dump_jsonl=str(args.debug_dump_jsonl),
            debug_print_full_completion=bool(args.debug_print_full_completion),
        )
        reward_fn = make_reward_fn_p2(n_items=n_items, cfg=r_cfg)


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
    print("✅ GRPO(NTP-itemid) done:", args.output_dir)


if __name__ == "__main__":
    main()
