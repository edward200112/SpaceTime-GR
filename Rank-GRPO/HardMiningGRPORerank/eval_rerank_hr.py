# HardMiningGRPO/eval_rerank_hr.py
import os
import sys
import argparse
import random
from typing import Any, Dict, List, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from HardMiningGRPO.reward_sasrec import parse_namecat_keys

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
RULE_FALLBACK = "只输出一个地点名(类别)，不要解释"

CAND_HEADER_CANON = "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）："
CAND_HEADERS = [
    CAND_HEADER_CANON,
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）:",
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）",
    "候选地点：",
    "候选地点:",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_ckpt", required=True, help="e.g. .../checkpoint-2000")
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--use_chat_template", action="store_true")

    # prompt/cut
    ap.add_argument("--max_length", type=int, default=1280)
    ap.add_argument("--max_new_tokens", type=int, default=32)

    # eval speed/mem
    ap.add_argument("--eval_max_samples", type=int, default=2000, help="<=0 means full eval set")
    ap.add_argument("--prompt_bs", type=int, default=1, help="how many prompts per batch")
    ap.add_argument("--score_bs", type=int, default=2, help="how many (prompt,cand) seqs per forward (auto reduce on OOM)")
    ap.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)

    # candidate shuffle (recommended to avoid position shortcuts)
    ap.add_argument("--shuffle_candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)

    # logging
    ap.add_argument("--print_target_pos_hist", action="store_true")
    ap.add_argument("--out_json", type=str, default="")

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


def _det_shuffle_pairs(pairs: List[Tuple[Any, Any]], seed: int):
    rng = random.Random(seed)
    rng.shuffle(pairs)


def _find_target_pos(target_namecat: str, candidate_namecats: List[str]) -> int:
    _, tgt_fold, ok = parse_namecat_keys(target_namecat)
    if not ok:
        tgt_fold = str(target_namecat).strip().casefold()

    for i, s in enumerate(candidate_namecats):
        _, k_fold, ok2 = parse_namecat_keys(s)
        if ok2:
            if k_fold == tgt_fold:
                return i
        else:
            if str(s).strip().casefold() == tgt_fold:
                return i
    return -1


def _build_candidate_block(cands: List[str]) -> str:
    lines = [CAND_HEADER_CANON]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def _rewrite_prompt_candidates(raw_prompt: str, cand_namecats: List[str]) -> str:
    p = (raw_prompt or "").rstrip()
    new_block = _build_candidate_block(cand_namecats)

    hit_pos = -1
    for hdr in CAND_HEADERS:
        pos = p.find(hdr)
        if pos != -1:
            hit_pos = pos
            break

    if hit_pos != -1:
        prefix = p[:hit_pos].rstrip()
        return (prefix + "\n" + new_block).rstrip()

    return (p + "\n" + new_block).rstrip()


@torch.no_grad()
def _score_sequences_avg_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    length_norm: bool = True,
) -> torch.Tensor:
    # ✅ eval 打分：不要 KV cache
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, L, V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, Lm1 = shift_labels.shape
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(B, Lm1)

    mask = (shift_labels != -100).float()
    tok_cnt = mask.sum(dim=1).clamp_min(1.0)
    nll = (loss * mask).sum(dim=1)

    if length_norm:
        return -(nll / tok_cnt)
    return -nll


def _encode_prompt_and_candidate(tok, prompt: str, cand: str):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(cand, add_special_tokens=False)
    return p_ids, c_ids


@torch.no_grad()
def score_candidates_for_prompts(
    model,
    tok,
    prompts: List[str],
    candidates: List[List[str]],
    max_total_len: int,
    score_bs: int,
    device: str,
    length_norm: bool = True,
) -> torch.Tensor:
    """
    返回 [B, K]，分数越大越好（candidate token 的 avg logprob 或 sum logprob）
    ✅ micro-batch 动态 padding + OOM 自动降 score_bs
    """
    B = len(prompts)
    K = len(candidates[0])
    assert all(len(x) == K for x in candidates), "Eval expects fixed K per sample."

    # build unpadded sequences
    seq_ids: List[List[int]] = []
    seq_labels: List[List[int]] = []

    for b in range(B):
        prompt = prompts[b]
        p_ids = tok.encode(prompt, add_special_tokens=False)
        for k in range(K):
            cand = candidates[b][k]
            c_ids = tok.encode(cand, add_special_tokens=False)

            ids = p_ids + c_ids

            # truncate from left, keep candidate tail
            if len(ids) > max_total_len:
                if len(c_ids) <= max_total_len:
                    ids = ids[-max_total_len:]
                    ids = ids[:-len(c_ids)] + c_ids
                else:
                    ids = c_ids[-max_total_len:]

            cand_len = min(len(c_ids), len(ids))
            prompt_len = len(ids) - cand_len

            labels = [-100] * len(ids)
            for t in range(prompt_len, len(ids)):
                labels[t] = ids[t]

            seq_ids.append(ids)
            seq_labels.append(labels)

    pad_id = int(tok.pad_token_id)
    N = len(seq_ids)

    out_scores = torch.empty((N,), dtype=torch.float32)
    bs = max(1, int(score_bs))
    i = 0

    while i < N:
        cur_bs = min(bs, N - i)
        try:
            chunk_ids = seq_ids[i:i + cur_bs]
            chunk_labels = seq_labels[i:i + cur_bs]

            # ✅ 每个 chunk 单独 pad，避免全局超长 padding
            max_len = max(len(x) for x in chunk_ids)

            input_ids = []
            labels = []
            attn = []
            for ids, lab in zip(chunk_ids, chunk_labels):
                pad_n = max_len - len(ids)
                input_ids.append([pad_id] * pad_n + ids)       # left pad
                labels.append([-100] * pad_n + lab)
                attn.append([0] * pad_n + [1] * len(ids))

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            attn = torch.tensor(attn, dtype=torch.long, device=device)

            sc = _score_sequences_avg_logprob(
                model=model,
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                length_norm=length_norm,
            ).detach().float().cpu()

            out_scores[i:i + cur_bs] = sc
            i += cur_bs

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[EVAL] OOM, reduce score_bs -> {bs}", flush=True)

    return out_scores.view(B, K)


@torch.no_grad()
def compute_hr_at_k(
    model,
    tok,
    eval_ds,
    max_samples: int,
    prompt_bs: int,
    score_bs: int,
    max_total_len: int,
    device: str,
    length_norm: bool = True,
) -> Dict[str, float]:
    n = len(eval_ds)
    if max_samples and max_samples > 0:
        n = min(n, int(max_samples))

    hr1_hit = 0
    hr10_hit = 0
    total = 0

    was_train = model.training
    model.eval()

    for st in range(0, n, prompt_bs):
        ed = min(n, st + prompt_bs)
        batch = eval_ds.select(range(st, ed))

        prompts = list(batch["prompt"])
        candidates = list(batch["candidate_namecats"])
        tgt_pos = torch.tensor(batch["target_pos"], dtype=torch.long)

        sc = score_candidates_for_prompts(
            model=model,
            tok=tok,
            prompts=prompts,
            candidates=candidates,
            max_total_len=max_total_len,
            score_bs=score_bs,
            device=device,
            length_norm=length_norm,
        )

        top10 = torch.topk(sc, k=min(10, sc.size(1)), dim=1).indices
        top1 = top10[:, 0]

        hr1_hit += int((top1 == tgt_pos).sum().item())
        hr10_hit += int((top10 == tgt_pos.unsqueeze(1)).any(dim=1).sum().item())
        total += (ed - st)

        if (st // prompt_bs) % 50 == 0:
            print(f"[EVAL] done {ed}/{n}", flush=True)

    if was_train:
        model.train()

    return {
        "hr1": hr1_hit / max(1, total),
        "hr10": hr10_hit / max(1, total),
        "samples": float(total),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    # tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # ✅ 加载 LoRA checkpoint
    model = PeftModel.from_pretrained(model, args.adapter_ckpt, is_trainable=False)
    model.eval()

    # dataset
    eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    def format_ex(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw = _get_field(ex, ["prompt"])
        raw = (raw or "").rstrip()
        if "只输出一个地点名" not in raw:
            raw = (raw + "\n" + RULE_FALLBACK).rstrip()

        cand_nc = _get_field(ex, ["candidate_namecats", "candidates_namecat", "candidates_namecats"])
        cand_it = _get_field(ex, ["candidate_item_ids", "candidates_item_ids", "candidates_ids"])
        if not isinstance(cand_nc, list) or not isinstance(cand_it, list) or len(cand_nc) != len(cand_it) or len(cand_nc) == 0:
            raise ValueError(f"[BAD DATA] candidates invalid: len(namecats)={len(cand_nc)} len(item_ids)={len(cand_it)}")

        pairs = list(zip(cand_nc, cand_it))
        if args.shuffle_candidates:
            _det_shuffle_pairs(pairs, seed=args.seed + int(idx))

        cand_nc2, cand_it2 = zip(*pairs)
        cand_nc2 = list(cand_nc2)
        cand_it2 = [int(x) for x in cand_it2]

        tgt_nc = _get_field(ex, ["target_namecat"])
        tgt_pos = _find_target_pos(tgt_nc, cand_nc2)
        if tgt_pos < 0:
            raise ValueError("[BAD DATA] target_namecat not found in candidate_namecats.")

        raw2 = _rewrite_prompt_candidates(raw, cand_nc2)

        prompt_final = build_chat_prompt(tok, raw2) if args.use_chat_template else raw2

        return {
            "prompt": prompt_final,
            "candidate_namecats": cand_nc2,
            "target_namecat": tgt_nc,
            "target_pos": int(tgt_pos),
        }

    num_proc = max(1, (os.cpu_count() or 8) // 2)
    eval_ds = eval_ds.map(format_ex, with_indices=True, remove_columns=eval_ds.column_names, num_proc=num_proc)

    if args.print_target_pos_hist:
        cnt = Counter(eval_ds["target_pos"])
        total = sum(cnt.values())
        top10 = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n[TARGET_POS_HIST] eval total={total} top10(pos,count,ratio):")
        for pos, c in top10:
            print(f"  pos={pos:>3d}  count={c:>8d}  ratio={c/total:.4f}")

    device = str(next(model.parameters()).device)
    max_total_len = int(args.max_length) + int(args.max_new_tokens)

    metrics = compute_hr_at_k(
        model=model,
        tok=tok,
        eval_ds=eval_ds,
        max_samples=args.eval_max_samples,
        prompt_bs=args.prompt_bs,
        score_bs=args.score_bs,
        max_total_len=max_total_len,
        device=device,
        length_norm=args.length_norm,
    )

    print("\n" + "=" * 80)
    print(f"[EVAL DONE] adapter={args.adapter_ckpt}")
    print(f"  HR@1  = {metrics['hr1']:.6f}")
    print(f"  HR@10 = {metrics['hr10']:.6f}")
    print(f"  samples = {int(metrics['samples'])}")
    print("=" * 80, flush=True)

    if args.out_json:
        import json
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"adapter_ckpt": args.adapter_ckpt, "eval_jsonl": args.eval_jsonl, **metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[OK] wrote: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
