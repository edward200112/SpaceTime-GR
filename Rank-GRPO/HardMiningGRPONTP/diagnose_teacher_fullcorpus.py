#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_teacher_fullcorpus.py

Full-corpus Teacher(SASRec) diagnostics:
- Exact target rank among ALL items via streaming chunks (no faiss)
- Also compute rank excluding history items (unique)
- Report Recall/HR@K and rank distribution

Example:
python HardMiningGRPO/diagnose_teacher_fullcorpus.py \
  --jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --sample_n 5000 \
  --batch_size 256 \
  --chunk_size 50000 \
  --K 1,10,50,200,1000 \
  --device cuda \
  --emb_on_gpu \
  --score_dtype fp16 \
  --show_chunk_pbar false
"""

import os
import json
import math
import time
import random
import argparse
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm


# -------------------------
# Import SASRec (robust)
# -------------------------
def import_sasrec():
    try:
        # repo style
        from SASRec import SASRec
        return SASRec
    except Exception:
        try:
            # local style (same dir)
            from SASRec import SASRec
            return SASRec
        except Exception as e:
            raise ImportError(
                "Cannot import SASRec. Tried TeacherModel.SASRec and SASRec. "
                "Make sure your PYTHONPATH/repo structure is correct."
            ) from e


SASRec = import_sasrec()


def pad_left(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    seq = list(seq or [])
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def _parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}. Choose fp16/bf16/fp32")


def load_sasrec_from_ckpt(
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int = 50,
    embed_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
) -> Tuple[torch.nn.Module, int]:
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
def get_user_repr(sasrec: torch.nn.Module, input_ids: torch.LongTensor) -> torch.Tensor:
    feats = sasrec.log2feats(input_ids)      # [B,L,H]
    return feats[:, -1, :]                   # [B,H]


def reservoir_sample_jsonl(path: str, k: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Uniform reservoir sampling over jsonl lines (streaming, memory-safe).
    """
    rng = random.Random(seed)
    sample = []
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n += 1
            if len(sample) < k:
                sample.append(obj)
            else:
                j = rng.randint(1, n)
                if j <= k:
                    sample[j - 1] = obj
    return sample


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jsonl", required=True, help="jsonl with history_item_ids + target_item_id")
    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--sample_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000)

    ap.add_argument("--K", type=str, default="1,10,50,200,1000", help="comma-separated Ks")

    ap.add_argument("--show_chunk_pbar", type=str, default="false", choices=["true", "false"])

    ap.add_argument("--emb_on_gpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    # sasrec arch (must match training)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)
    ap.add_argument("--sasrec_max_len", type=int, default=50)

    return ap.parse_args()


def summarize_ranks(ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {}
    arr = np.array(ranks, dtype=np.int64)
    out = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    return out


def main():
    args = parse_args()
    device = args.device
    score_dtype = _parse_dtype(args.score_dtype)
    show_chunk_pbar = (args.show_chunk_pbar.lower() == "true")

    Ks = [int(x) for x in args.K.split(",") if x.strip()]
    Ks = sorted(set(Ks))
    Kmax = max(Ks)

    # 1) load teacher
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

    # 2) load sample
    print(f"[INFO] sampling up to {args.sample_n} examples from {args.jsonl}")
    data = reservoir_sample_jsonl(args.jsonl, k=int(args.sample_n), seed=int(args.seed))
    print(f"[OK] sampled = {len(data)}")

    # 3) prepare embedding weight
    emb = sasrec.item_emb.weight.detach()  # [N+1,H] float32
    if args.emb_on_gpu and str(device).startswith("cuda"):
        emb = emb.to(device=device, dtype=score_dtype, non_blocking=True)
        print(f"[OK] item_emb.weight on GPU: shape={tuple(emb.shape)} dtype={emb.dtype}")
    else:
        emb = emb.float().cpu()
        print(f"[OK] item_emb.weight on CPU: shape={tuple(emb.shape)} dtype={emb.dtype}")

    N = n_items
    total_chunks = math.ceil(N / int(args.chunk_size))

    # metrics accumulators
    hits_all = {k: 0 for k in Ks}
    hits_excl = {k: 0 for k in Ks}
    ranks_all: List[int] = []
    ranks_excl: List[int] = []
    bad_tgt_in_hist = 0
    bad_oob = 0

    # batching
    B = int(args.batch_size)
    L = int(args.sasrec_max_len)
    chunk_size = int(args.chunk_size)

    def iter_batches(xs, bs):
        for i in range(0, len(xs), bs):
            yield xs[i:i+bs]

    t_start = time.time()

    outer = tqdm(list(iter_batches(data, B)), desc="diagnose(full-corpus rank)", dynamic_ncols=True)
    for batch in outer:
        histories = []
        targets = []
        hist_sets = []   # unique history set per row (exclude 0)
        for ex in batch:
            hist = ex.get("history_item_ids", [])
            tgt = ex.get("target_item_id", None)
            if tgt is None:
                continue
            tgt = int(tgt)
            if tgt < 1 or tgt > n_items:
                bad_oob += 1
                continue

            hist = [int(x) for x in (hist or [])]
            if tgt in hist:
                bad_tgt_in_hist += 1

            hist_pad = pad_left(hist, L, pad=0)
            histories.append(hist_pad)
            targets.append(tgt)
            hs = set([x for x in hist_pad if x != 0])
            hist_sets.append(hs)

        if not histories:
            continue

        # tensors
        input_ids = torch.tensor(histories, dtype=torch.long, device=device)
        tgt_ids = torch.tensor(targets, dtype=torch.long, device=device)

        # user repr
        user_repr = get_user_repr(sasrec, input_ids)  # [b,H]
        u = user_repr.to(dtype=score_dtype)

        # target scores
        tgt_emb = emb.index_select(0, tgt_ids)  # [b,H]
        tgt_scores = (u * tgt_emb).sum(dim=-1)  # [b] in score_dtype

        # exact rank: 1 + count(scores > tgt_score)
        better = torch.zeros((u.size(0),), dtype=torch.int64, device=device)

        chunk_iter = range(total_chunks)
        if show_chunk_pbar:
            chunk_iter = tqdm(chunk_iter, desc="scan chunks", leave=False, dynamic_ncols=True)

        for ci in chunk_iter:
            start = 1 + ci * chunk_size
            end = min(N + 1, start + chunk_size)

            emb_chunk = emb[start:end]  # [C,H]
            # scores: [b,C]
            scores = torch.matmul(u, emb_chunk.t())
            # count strictly better than target
            better += (scores > tgt_scores.unsqueeze(1)).sum(dim=1, dtype=torch.int64)

            if show_chunk_pbar and (ci % 5 == 0):
                chunk_iter.set_postfix({"items": f"{start}-{end-1}"})

        rank_all = (better + 1).detach().cpu().tolist()

        # rank excluding history (unique):
        # better_excl = better - count(hist_item_score > tgt_score)
        better_hist = []
        for bi in range(u.size(0)):
            hs = hist_sets[bi]
            tgt = int(targets[bi])
            # exclude padding and target itself
            hs = [x for x in hs if x != 0 and x != tgt and 1 <= x <= n_items]
            if not hs:
                better_hist.append(0)
                continue
            h_ids = torch.tensor(hs, dtype=torch.long, device=device)
            h_emb = emb.index_select(0, h_ids)  # [m,H]
            h_scores = torch.matmul(u[bi:bi+1], h_emb.t()).squeeze(0)  # [m]
            cnt = int((h_scores > tgt_scores[bi]).sum().item())
            better_hist.append(cnt)

        better_hist_t = torch.tensor(better_hist, dtype=torch.int64, device=device)
        better_excl = better - better_hist_t
        rank_excl = (better_excl + 1).clamp(min=1).detach().cpu().tolist()

        # accumulate metrics
        for r in rank_all:
            ranks_all.append(int(r))
            for k in Ks:
                if r <= k:
                    hits_all[k] += 1

        for r in rank_excl:
            ranks_excl.append(int(r))
            for k in Ks:
                if r <= k:
                    hits_excl[k] += 1

        # progress text
        done = len(ranks_all)
        outer.set_postfix({
            "done": done,
            f"HR@{Ks[0]}": f"{hits_all[Ks[0]]/max(1,done):.4f}",
            f"HR@{Kmax}": f"{hits_all[Kmax]/max(1,done):.4f}",
            "meanRank": f"{(np.mean(ranks_all) if ranks_all else 0):.1f}",
        })

    elapsed = time.time() - t_start
    total = len(ranks_all)

    print("\n" + "=" * 90)
    print("[FULL-CORPUS DIAG RESULT]")
    print(f"  jsonl           : {args.jsonl}")
    print(f"  sampled         : {total} (requested={args.sample_n})")
    print(f"  n_items         : {n_items}")
    print(f"  max_len         : {L}")
    print(f"  batch_size      : {B}")
    print(f"  chunk_size      : {chunk_size} (total_chunks={total_chunks})")
    print(f"  device          : {device}")
    print(f"  score_dtype     : {args.score_dtype}")
    print(f"  emb_on_gpu      : {args.emb_on_gpu}")
    print(f"  elapsed_sec     : {elapsed:.2f}")
    print(f"  bad_tgt_in_hist : {bad_tgt_in_hist}")
    print(f"  bad_oob_target  : {bad_oob}")
    print("-" * 90)

    def fmt_hits(hits: Dict[int, int], name: str):
        print(f"[{name}]")
        for k in Ks:
            v = hits[k] / max(1, total)
            print(f"  HR@{k:<5d} : {v:.6f}")
        print()

    fmt_hits(hits_all, "Rank among ALL items")
    fmt_hits(hits_excl, "Rank excluding HISTORY items (unique)")

    s_all = summarize_ranks(ranks_all)
    s_ex = summarize_ranks(ranks_excl)

    print("[Rank distribution] (lower is better)")
    print("  ALL items:")
    for kk in ["mean", "median", "p90", "p95", "p99", "min", "max"]:
        print(f"    {kk:<6s}: {s_all.get(kk, 0):.2f}")
    print("  EXCL history:")
    for kk in ["mean", "median", "p90", "p95", "p99", "min", "max"]:
        print(f"    {kk:<6s}: {s_ex.get(kk, 0):.2f}")

    print("=" * 90)
    print("Tips:")
    print("  - 如果 HR@200(ALL/EXCL) 仍接近 0，Teacher 全库排序能力偏弱：继续训练/加强负采样通常会有收益。")
    print("  - 如果 EXCL-history 的 HR@K 明显高于 ALL，说明 Teacher 主要在“复现历史”，生成 teacher_top 时建议 filter_history。")
    print("  - bad_tgt_in_hist 不为 0：你的数据里存在 target 已在历史里，训练/评估时要么过滤，要么明确处理。")


if __name__ == "__main__":
    main()
