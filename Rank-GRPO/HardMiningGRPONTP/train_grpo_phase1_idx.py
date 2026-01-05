# train_grpo_phase1_idx.py
import os
import sys
import argparse
import random
import re
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Your SASRec code path may differ; keep flexible import.
def import_sasrec(sasrec_code_dir: str = ""):
    if sasrec_code_dir:
        sys.path.insert(0, sasrec_code_dir)
    try:
        from SASRec import SASRec
        return SASRec
    except Exception:
        # fallback if you have TeacherModel/SASRec.py
        from TeacherModel.SASRec import SASRec
        return SASRec

from reward_ntp_phase1_idx import SasrecScorer, ResolverConfig, make_reward_fn


USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Phase1 rule: output idx only
RULE_IDX = "请只输出一个候选 idx（1~50 的纯数字），不要输出地点名、不要输出括号或任何解释。"

CAND_CUT_RE = re.compile(r"(候选|备选|candidate|candidates|候选poi|poi列表|候选列表)", re.IGNORECASE)
OLD_RULE_PATTERNS = [
    r".*只能从.*候选.*",
    r".*候选.*选\s*1\s*个.*",
    r".*只输出一个地点名.*",
    r".*只输出一个地点.*",
]

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_chat_prompt(tok, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return USER_PREFIX_TEMPLATE.format(prompt=user_text)

def strip_old_candidate_block(raw_prompt: str) -> str:
    p = (raw_prompt or "").rstrip()
    lines = p.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if CAND_CUT_RE.search(ln):
            cut = i
            break
    if cut is not None:
        lines = lines[:cut]

    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        bad = False
        for pat in OLD_RULE_PATTERNS:
            if re.match(pat, s):
                bad = True
                break
        if bad:
            continue
        out.append(ln)
    return "\n".join(out).rstrip()

def format_candidate_block(candidate_namecats: List[str]) -> str:
    # Always output 1..K list.
    lines = ["候选（请从下面选择 1 个，只输出 idx 纯数字）："]
    for i, nc in enumerate(candidate_namecats, start=1):
        lines.append(f"{i}. {nc}")
    return "\n".join(lines)

def load_sasrec_from_ckpt(
    SASRecCls,
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

    class _Args: pass
    a = _Args()
    a.device = device
    a.max_len = int(max_len)
    a.embed_dim = int(embed_dim)
    a.num_blocks = int(num_blocks)
    a.num_heads = int(num_heads)
    a.dropout = float(dropout)

    sasrec = SASRecCls(item_num=n_items, args=a).to(device)
    sasrec.load_state_dict(state_dict, strict=True)
    sasrec.eval()
    for p in sasrec.parameters():
        p.requires_grad_(False)

    print(f"[OK] loaded SASRec: n_items={n_items}, max_len={a.max_len}, dim={a.embed_dim}, "
          f"blocks={a.num_blocks}, heads={a.num_heads}, dropout={a.dropout}")
    return sasrec, n_items

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)

    # SASRec
    ap.add_argument("--sasrec_code_dir", type=str, default="")
    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    # GRPO
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=3)    # idx 1~50: usually 1-2 tokens
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)   # must be >0 for sampling

    # prompt
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--strip_candidates_from_prompt", action=argparse.BooleanOptionalAction, default=True)

    # reward cfg
    ap.add_argument("--format_bonus", type=float, default=0.05)
    ap.add_argument("--exists_bonus", type=float, default=0.10)
    ap.add_argument("--correct_reward", type=float, default=2.0)
    ap.add_argument("--wrong_penalty", type=float, default=0.3)
    ap.add_argument("--unknown_penalty", type=float, default=0.6)
    ap.add_argument("--teacher_rank_weight", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--teacher_clip", type=float, default=5.0)
    ap.add_argument("--extra_text_penalty", type=float, default=0.08)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    # debug
    ap.add_argument("--debug_log_every_steps", type=int, default=0)
    ap.add_argument("--debug_dump_jsonl", type=str, default="")
    ap.add_argument("--debug_print_full_completion", action="store_true")
    return ap.parse_args()


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
        device_map="cuda" if torch.cuda.is_available() else None,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()

    # eos ids (Qwen chat compat)
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
        raw = (ex.get("prompt", "") or "").rstrip()
        if args.strip_candidates_from_prompt:
            raw = strip_old_candidate_block(raw)

        # candidate fields
        cand_namecats = ex.get("candidate_namecats", None)
        cand_item_ids = ex.get("candidate_item_ids", None)
        if cand_namecats is None or not isinstance(cand_namecats, list):
            # fallback: build placeholder text from ids
            cand_item_ids = cand_item_ids if isinstance(cand_item_ids, list) else []
            cand_namecats = [f"item_{int(x)}" for x in cand_item_ids]
        if cand_item_ids is None or not isinstance(cand_item_ids, list):
            cand_item_ids = []

        # build prompt: history + candidate list + rule
        prompt_raw = (raw + "\n" + format_candidate_block(cand_namecats) + "\n" + RULE_IDX).rstrip()
        prompt = build_chat_prompt(tok, prompt_raw) if args.use_chat_template else prompt_raw

        return {
            "prompt": prompt,
            "prompt_raw": prompt_raw,
            "history_item_ids": ex.get("history_item_ids", []),
            "target_item_id": int(ex.get("target_item_id")),
            "candidate_item_ids": [int(x) for x in cand_item_ids],
        }

    num_proc = max(1, (os.cpu_count() or 8) // 2)
    train_ds = train_ds.map(format_ex, with_indices=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_ds = eval_ds.map(format_ex, with_indices=True, remove_columns=eval_ds.column_names, num_proc=num_proc)

    # load SASRec
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SASRecCls = import_sasrec(args.sasrec_code_dir)
    sasrec, n_items = load_sasrec_from_ckpt(
        SASRecCls=SASRecCls,
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

    r_cfg = ResolverConfig(
        format_bonus=float(args.format_bonus),
        exists_bonus=float(args.exists_bonus),
        correct_reward=float(args.correct_reward),
        wrong_penalty=float(args.wrong_penalty),
        unknown_penalty=float(args.unknown_penalty),
        teacher_rank_weight=float(args.teacher_rank_weight),
        alpha=float(args.alpha),
        teacher_clip=float(args.teacher_clip),
        extra_text_penalty=float(args.extra_text_penalty),
        debug_log_every_steps=int(args.debug_log_every_steps),
        debug_dump_jsonl=str(args.debug_dump_jsonl),
        debug_print_full_completion=bool(args.debug_print_full_completion),
    )
    reward_fn = make_reward_fn(scorer, r_cfg, sasrec_max_len=int(args.sasrec_max_len))

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
        temperature=float(args.temperature),  # must be >0
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
    print("✅ GRPO Phase1(idx) done:", args.output_dir)


if __name__ == "__main__":
    main()
