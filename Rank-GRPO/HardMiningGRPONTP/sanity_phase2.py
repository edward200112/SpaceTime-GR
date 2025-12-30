#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_phase2.py

Phase2 数据 sanity：
1) 结构与基础统计：teacher_top_item_ids / target / history 合法性、重复、与 history 重叠
2) teacher_top pool 内 target 的 rank 分布（用 SASRec.predict_candidates 打分）
3) (可选) 加载 policy(base_model+adapter) 生成 item_id，统计：
   - output_in_teacher_pool_rate
   - output_hit_rate (HR@1)
   - output_rank_in_teacher_pool（输出在 teacher pool 内的 rank）
   - 输出的格式/前缀/多余文本问题

不使用 FAISS。teacher pool 只做 K=200 的打分，速度很快。
"""

import os
import sys
import re
import json
import math
import time
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


ID_FIND_RE = re.compile(r"(-?\d+)")


# ------------------------- utils -------------------------
def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("content", "text", "completion", "generated_text", "prompt"):
            if k in x:
                return str(x[k])
    return str(x)


def extract_first_item_id(text: str) -> Tuple[Optional[int], str, bool, bool]:
    """
    return: item_id, first_line, prefix_ok, has_extra
    prefix_ok: 数字前没有非空前缀（更严格）
    has_extra: 数字后还有内容 或 有多行
    """
    t = _to_text(text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)
    rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    m = ID_FIND_RE.search(first)
    if not m:
        return None, first, False, bool(rest)

    prefix = first[:m.start()].strip()
    prefix_ok = (prefix == "")

    try:
        item_id = int(m.group(1))
    except Exception:
        item_id = None

    tail = first[m.end():].strip()
    has_extra = bool(tail) or bool(rest)
    return item_id, first, prefix_ok, has_extra


def pad_left(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    seq = list(seq or [])
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def load_jsonl_sample(path: str, sample_n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Reservoir sampling: 不用读入全量文件
    """
    rng = random.Random(seed)
    out = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            seen += 1
            if len(out) < sample_n:
                out.append(ex)
            else:
                j = rng.randint(1, seen)
                if j <= sample_n:
                    out[j - 1] = ex
    return out


def _parse_score_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown --score_dtype {s}. Choose from fp16/bf16/fp32")


# ------------------------- SASRec loader -------------------------
def load_sasrec_from_ckpt(
    sasrec_code_dir: str,
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int,
    embed_dim: int,
    num_blocks: int,
    num_heads: int,
    dropout: float,
):
    if sasrec_code_dir:
        sys.path.insert(0, sasrec_code_dir)

    # 尽量兼容不同代码组织
    SASRec = None
    try:
        from SASRec import SASRec as _SASRec
        SASRec = _SASRec
    except Exception:
        try:
            from TeacherModel.SASRec import SASRec as _SASRec
            SASRec = _SASRec
        except Exception as e:
            raise ImportError(f"Cannot import SASRec. Try --sasrec_code_dir. err={repr(e)}")

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


@torch.no_grad()
def sasrec_rank_in_pool(
    sasrec,
    device: str,
    history_ids: List[int],
    pool_ids: List[int],
    query_id: int,
    max_len: int,
) -> Optional[int]:
    """
    计算 query_id 在 pool_ids 内的 rank（1=最好），pool 用 SASRec 打分降序排序。
    若 query_id 不在 pool，返回 None
    """
    if query_id not in set(pool_ids):
        return None
    hist = torch.tensor(pad_left(history_ids, max_len, pad=0), dtype=torch.long, device=device).unsqueeze(0)
    cand = torch.tensor(pool_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,K]
    scores = sasrec.predict_candidates(hist, cand).squeeze(0)  # [K]
    order = torch.argsort(scores, descending=True)
    # ranks[order[i]]=i
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=device)
    pos = pool_ids.index(query_id)
    return int(ranks[pos].item()) + 1


# ------------------------- policy loader (optional) -------------------------
def load_policy(base_model: str, adapter: str, attn_impl: str, device: str, dtype: torch.dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # eos ids 兼容 Qwen chat
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

    return tok, model


USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
RULE_FALLBACK = "只输出一个 item_id（整数），不要解释，也不要输出其他文字。"

CAND_HEADERS = [
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）：",
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）:",
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）",
    "候选地点：",
    "候选地点:",
]
OLD_CONSTRAINT_PATTERNS = [
    r".*只能从.*候选.*",
    r".*候选.*选\s*1\s*个.*",
    r".*in\s*candidates.*",
]


def strip_candidate_block(raw_prompt: str) -> str:
    p = (raw_prompt or "").rstrip()
    hit_pos = -1
    for hdr in CAND_HEADERS:
        pos = p.find(hdr)
        if pos != -1:
            hit_pos = pos
            break
    if hit_pos != -1:
        p = p[:hit_pos].rstrip()

    lines = [ln for ln in p.splitlines() if ln.strip()]
    new_lines = []
    for ln in lines:
        bad = False
        for pat in OLD_CONSTRAINT_PATTERNS:
            if re.match(pat, ln.strip()):
                bad = True
                break
        if not bad:
            new_lines.append(ln)
    return "\n".join(new_lines).rstrip()


def build_chat_prompt(tok, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return USER_PREFIX_TEMPLATE.format(prompt=user_text)


@torch.no_grad()
def policy_generate_one(
    tok,
    model,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    inputs = tok([prompt_text], return_tensors="pt", padding=True, truncation=True).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        num_return_sequences=1,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


# ------------------------- main sanity -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="phase2 jsonl with teacher_top_item_ids")

    # SASRec teacher
    ap.add_argument("--sasrec_code_dir", type=str, default="", help="优先加入 sys.path 的 SASRec 代码目录（比如 /workspace/Rank-GRPO/SASRec）")
    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    ap.add_argument("--topk", type=int, default=200)

    # sampling
    ap.add_argument("--sample_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    # device
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    # optional policy check
    ap.add_argument("--base_model", type=str, default="")
    ap.add_argument("--adapter", type=str, default="")
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--strip_candidates_from_prompt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--show_examples", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    score_dtype = _parse_score_dtype(args.score_dtype)

    print(f"[INFO] sampling up to {args.sample_n} examples from {args.jsonl}")
    samples = load_jsonl_sample(args.jsonl, sample_n=args.sample_n, seed=args.seed)
    print(f"[OK] sampled = {len(samples)}")

    # load SASRec
    sasrec, n_items = load_sasrec_from_ckpt(
        sasrec_code_dir=args.sasrec_code_dir,
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )

    # optional policy
    tok = model = None
    if args.base_model and args.adapter:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and "cuda" in device) else torch.float16
        tok, model = load_policy(args.base_model, args.adapter, args.attn_impl, device, dtype)
        print(f"[OK] loaded policy: base_model={args.base_model} adapter={args.adapter}")
    else:
        print("[INFO] policy check skipped (provide --base_model and --adapter to enable).")

    # stats accumulators
    bad_missing = 0
    bad_oob = 0
    bad_zero = 0
    bad_len = 0
    dup_pool = 0

    sum_overlap = 0.0
    cnt_overlap = 0

    # teacher rank stats
    tgt_in_pool = 0
    tgt_rank_list = []

    # policy stats
    gen_ok = 0
    gen_hit = 0
    gen_in_pool = 0
    gen_prefix_bad = 0
    gen_has_extra = 0
    gen_oob = 0
    gen_rank_list = []

    top_examples = []
    miss_examples = []

    t0 = time.time()
    pbar = tqdm(samples, desc="sanity", dynamic_ncols=True)

    for ex in pbar:
        hist = ex.get("history_item_ids", None)
        tgt = ex.get("target_item_id", None)
        pool = ex.get("teacher_top_item_ids", None)

        if hist is None or tgt is None or pool is None:
            bad_missing += 1
            continue

        hist = [int(x) for x in hist]
        tgt = int(tgt)
        pool = [int(x) for x in pool[: args.topk]]

        if len(pool) != args.topk:
            bad_len += 1

        # pool duplicate
        if len(set(pool)) != len(pool):
            dup_pool += 1

        # range / zero checks
        for x in pool + [tgt]:
            if x == 0:
                bad_zero += 1
                break
            if x < 1 or x > n_items:
                bad_oob += 1
                break

        # overlap with history
        hs = set(hist)
        ov = len(hs.intersection(set(pool))) / max(1, len(pool))
        sum_overlap += ov
        cnt_overlap += 1

        # teacher rank of target in pool
        if tgt in set(pool):
            tgt_in_pool += 1
            r_tgt = sasrec_rank_in_pool(
                sasrec=sasrec,
                device=device,
                history_ids=hist,
                pool_ids=pool,
                query_id=tgt,
                max_len=args.sasrec_max_len,
            )
            if r_tgt is not None:
                tgt_rank_list.append(r_tgt)

        # optional: policy generation + ranks
        if model is not None:
            raw_prompt = ex.get("prompt", "")
            raw_prompt = (raw_prompt or "").rstrip()
            if args.strip_candidates_from_prompt:
                raw_prompt = strip_candidate_block(raw_prompt)
            if "只输出一个 item_id" not in raw_prompt:
                raw_prompt = (raw_prompt + "\n" + RULE_FALLBACK).rstrip()

            prompt_text = build_chat_prompt(tok, raw_prompt) if args.use_chat_template else raw_prompt
            gen = policy_generate_one(
                tok=tok, model=model, prompt_text=prompt_text,
                device=device, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, do_sample=args.do_sample
            )

            item_id, first_line, prefix_ok, has_extra = extract_first_item_id(gen)
            if item_id is not None:
                gen_ok += 1
                if not prefix_ok:
                    gen_prefix_bad += 1
                if has_extra:
                    gen_has_extra += 1

                if item_id < 1 or item_id > n_items:
                    gen_oob += 1
                else:
                    if item_id == tgt:
                        gen_hit += 1
                    if item_id in set(pool):
                        gen_in_pool += 1
                        r_out = sasrec_rank_in_pool(
                            sasrec=sasrec,
                            device=device,
                            history_ids=hist,
                            pool_ids=pool,
                            query_id=item_id,
                            max_len=args.sasrec_max_len,
                        )
                        if r_out is not None:
                            gen_rank_list.append(r_out)

                # store examples
                if len(top_examples) < args.show_examples and item_id == tgt:
                    top_examples.append({"tgt": tgt, "out": item_id, "first": first_line, "gen": gen[:120]})
                if len(miss_examples) < args.show_examples and item_id != tgt:
                    miss_examples.append({"tgt": tgt, "out": item_id, "first": first_line, "gen": gen[:120]})

        # update pbar
        if cnt_overlap % 100 == 0:
            pbar.set_postfix({
                "tgt_in_pool": f"{tgt_in_pool}/{cnt_overlap}",
                "avg_ov": f"{(sum_overlap/max(1,cnt_overlap)):.4f}",
                "gen_hit": gen_hit,
                "gen_in_pool": gen_in_pool,
            })

    elapsed = time.time() - t0

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def pct(a, b):
        return float(a) / float(b) if b else 0.0

    print("\n" + "=" * 90)
    print("[SANITY RESULT]")
    print(f"  file          : {args.jsonl}")
    print(f"  sampled       : {len(samples)}")
    print(f"  n_items       : {n_items}")
    print(f"  topk          : {args.topk}")
    print(f"  device        : {device}")
    print(f"  elapsed_sec   : {elapsed:.2f}")
    print("-" * 90)
    print("[DATA CHECK]")
    print(f"  missing_fields: {bad_missing}")
    print(f"  bad_len(topk) : {bad_len}")
    print(f"  pool_dupe_rows: {dup_pool}")
    print(f"  zero_found    : {bad_zero}")
    print(f"  oob_found     : {bad_oob}")
    print(f"  avg_overlap(hist ∩ pool)/K : {sum_overlap/max(1,cnt_overlap):.6f}")
    print("-" * 90)
    print("[TEACHER SIGNAL IN POOL]")
    print(f"  target_in_pool_rate : {pct(tgt_in_pool, len(samples)):.4f}  ({tgt_in_pool}/{len(samples)})")
    if tgt_rank_list:
        tgt_rank_list_sorted = sorted(tgt_rank_list)
        print(f"  target_rank_in_pool : mean={mean(tgt_rank_list):.2f}  "
              f"median={tgt_rank_list_sorted[len(tgt_rank_list_sorted)//2]}  "
              f"p90={tgt_rank_list_sorted[int(0.9*(len(tgt_rank_list_sorted)-1))]}")
    else:
        print("  target_rank_in_pool : N/A (target never ranked?)")
    print("-" * 90)
    if model is not None:
        print("[POLICY GENERATION]")
        print(f"  parsed_rate         : {pct(gen_ok, len(samples)):.4f} ({gen_ok}/{len(samples)})")
        print(f"  output_oob_rate     : {pct(gen_oob, len(samples)):.4f}")
        print(f"  output_hit_rate(HR@1): {pct(gen_hit, len(samples)):.4f}")
        print(f"  output_in_pool_rate : {pct(gen_in_pool, len(samples)):.4f}")
        print(f"  prefix_bad_rate     : {pct(gen_prefix_bad, max(1,gen_ok)):.4f}")
        print(f"  has_extra_rate      : {pct(gen_has_extra, max(1,gen_ok)):.4f}")
        if gen_rank_list:
            srt = sorted(gen_rank_list)
            print(f"  output_rank_in_pool : mean={mean(gen_rank_list):.2f} "
                  f"median={srt[len(srt)//2]} p90={srt[int(0.9*(len(srt)-1))]}")
        else:
            print("  output_rank_in_pool : N/A (output never in pool)")
        print("-" * 90)
        if top_examples:
            print("[HIT EXAMPLES]")
            for r in top_examples[: args.show_examples]:
                print(json.dumps(r, ensure_ascii=False))
        if miss_examples:
            print("[MISS EXAMPLES]")
            for r in miss_examples[: args.show_examples]:
                print(json.dumps(r, ensure_ascii=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
