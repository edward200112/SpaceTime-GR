#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_sasrec_fullcorpus_plus_sampled.py

同时评测：
1) Sampled ranking (1 pos + N neg)：
   - uniform negatives (fast)
   - popularity negatives (strict-ish)
2) Full-corpus exact rank（扫描全库 item，得到真实排名分布 & HR@K）

Example:
python eval_sasrec_fullcorpus_plus_sampled.py \
  --dataset_path /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --weights_path /workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth \
  --max_len 50 --embed_dim 128 --num_blocks 2 --num_heads 2 --dropout 0.2 \
  --device cuda \
  --do_fast --fast_users 2000 --fast_neg 99 --fast_bs 256 \
  --do_strict --strict_users 2000 --strict_neg 99 --strict_bs 128 \
  --do_full --full_users 5000 --full_bs 256 --chunk_size 50000 --score_dtype fp16 --emb_on_gpu
"""

import os
import math
import time
import json
import pickle
import random
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from SASRec import SASRec


# -------------------------
# Utils
# -------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_left(seq, max_len, pad=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def fmt_sec(x):
    x = int(max(0, x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _pick_state_dict(obj):
    # 兼容各种保存格式
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (choose fp16/bf16/fp32)")


def percentile(xs, q):
    # xs: list[int]
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(round((q / 100.0) * (len(xs_sorted) - 1)))
    idx = max(0, min(idx, len(xs_sorted) - 1))
    return float(xs_sorted[idx])


# -------------------------
# Strict split: train/valid/test
# -------------------------
def build_strict_splits(raw_data_list):
    """
    raw_data_list: list of dict with keys: user_id, sequence (list[int])
    Return:
      splits: list of dict with uid, seq_train, valid_item, test_item, full_set
      item_freq: Counter computed from seq_train only
    """
    splits = []
    item_freq = Counter()

    for entry in raw_data_list:
        uid = str(entry.get("user_id", ""))
        seq = entry.get("sequence", None)
        if not isinstance(seq, (list, tuple)):
            continue
        if len(seq) < 3:
            continue

        valid_item = int(seq[-2])
        test_item = int(seq[-1])
        seq_train = [int(x) for x in seq[:-2]]
        if len(seq_train) < 2:
            continue

        full_set = set(int(x) for x in seq)

        for it in seq_train:
            if it != 0:
                item_freq[it] += 1

        splits.append({
            "uid": uid,
            "seq_train": seq_train,
            "valid_item": valid_item,
            "test_item": test_item,
            "full_set": full_set,
        })

    return splits, item_freq


# -------------------------
# Popularity Sampler
# -------------------------
class PopularitySampler:
    def __init__(self, n_items, item_freq: Counter, alpha: float = 0.75):
        self.n_items = int(n_items)

        freqs = np.zeros(self.n_items + 1, dtype=np.float64)
        for it, c in item_freq.items():
            if 1 <= it <= self.n_items:
                freqs[it] = float(c)
        freqs[0] = 0.0
        if freqs.sum() <= 0:
            freqs[1:] = 1.0

        probs = np.power(freqs, alpha)
        probs[0] = 0.0
        s = probs.sum()
        if s <= 0:
            probs[1:] = 1.0
            s = probs.sum()

        self.probs = probs / s
        self.items = np.arange(self.n_items + 1, dtype=np.int64)

    def sample(self, size: int):
        s = np.random.choice(self.items, size=size, replace=True, p=self.probs)
        if np.any(s == 0):
            s[s == 0] = 1
        return s


# -------------------------
# Sampled eval (uniform negatives)
# -------------------------
@torch.no_grad()
def eval_sampled_uniform(
    model,
    splits,
    n_items,
    max_len,
    mode="valid",
    num_eval_users=2000,
    num_neg=99,
    eval_batch_size=256,
    device="cuda",
    ks=(10, 50, 100),
    seed=123,
):
    assert mode in ("valid", "test")
    model.eval()

    rng = random.Random(seed)
    if num_eval_users is not None and 0 < num_eval_users < len(splits):
        eval_splits = rng.sample(splits, num_eval_users)
    else:
        eval_splits = splits

    total_users = len(eval_splits)
    total_batches = (total_users + eval_batch_size - 1) // eval_batch_size
    C = 1 + num_neg

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    total = 0

    def make_hist_and_pos(s):
        seq_train = s["seq_train"]
        if mode == "valid":
            hist = seq_train
            pos = int(s["valid_item"])
        else:
            hist = seq_train + [int(s["valid_item"])]
            pos = int(s["test_item"])
        x = pad_left(hist[-max_len:], max_len, pad=0)
        return x, pos, s["full_set"]

    start_t = time.time()
    processed_batches = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []
    oversample = 8

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)
            candidates = np.zeros((B, C), dtype=np.int64)
            candidates[:, 0] = np.array(batch_pos, dtype=np.int64)

            pool = np.random.randint(1, n_items + 1, size=(B, num_neg * oversample), dtype=np.int64)
            for i in range(B):
                pos_i = int(batch_pos[i])
                fs = batch_forbid[i]
                negs = []
                row = pool[i]
                for cand in row:
                    if cand == pos_i or (cand in fs):
                        continue
                    negs.append(int(cand))
                    if len(negs) >= num_neg:
                        break
                while len(negs) < num_neg:
                    cand = rng.randint(1, n_items)
                    if cand != pos_i and cand not in fs:
                        negs.append(cand)
                candidates[i, 1:] = np.array(negs, dtype=np.int64)

            input_tensor = torch.LongTensor(batch_inputs).to(device)
            cand_tensor = torch.from_numpy(candidates).long().to(device)

            scores = model.predict_candidates(input_tensor, cand_tensor)  # [B,C]

            pos_scores = scores[:, 0]
            neg_scores = scores[:, 1:]
            better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
            ranks = (better + 1).tolist()

            for r in ranks:
                total += 1
                for k in ks:
                    if r <= k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / math.log2(r + 1)

            processed_batches += 1
            batch_inputs, batch_pos, batch_forbid = [], [], []

            if processed_batches % 10 == 0 or processed_batches == total_batches:
                elapsed = time.time() - start_t
                avg = elapsed / max(1, processed_batches)
                eta = (total_batches - processed_batches) * avg
                print(f"[Uniform-{mode}] batches {processed_batches}/{total_batches} "
                      f"elapsed {fmt_sec(elapsed)} ETA {fmt_sec(eta)}")

    metrics = {"total": total}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    return metrics


# -------------------------
# Sampled eval (popularity negatives)
# -------------------------
@torch.no_grad()
def eval_sampled_popularity(
    model,
    splits,
    n_items,
    max_len,
    pop_sampler,
    mode="valid",
    num_eval_users=2000,
    num_neg=99,
    eval_batch_size=128,
    device="cuda",
    ks=(10, 50, 100),
    seed=123,
):
    assert mode in ("valid", "test")
    model.eval()

    rng = random.Random(seed)
    if num_eval_users is not None and 0 < num_eval_users < len(splits):
        eval_splits = rng.sample(splits, num_eval_users)
    else:
        eval_splits = splits

    total_users = len(eval_splits)
    total_batches = (total_users + eval_batch_size - 1) // eval_batch_size
    C = 1 + num_neg

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    total = 0

    def make_hist_and_pos(s):
        seq_train = s["seq_train"]
        if mode == "valid":
            hist = seq_train
            pos = int(s["valid_item"])
        else:
            hist = seq_train + [int(s["valid_item"])]
            pos = int(s["test_item"])
        x = pad_left(hist[-max_len:], max_len, pad=0)
        return x, pos, s["full_set"]

    start_t = time.time()
    processed_batches = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)
            candidates = torch.empty((B, C), dtype=torch.long)

            for i in range(B):
                pos_i = int(batch_pos[i])
                forbid_i = batch_forbid[i]
                negs = []
                tries = 0
                while len(negs) < num_neg and tries < num_neg * 50:
                    cand = int(pop_sampler.sample(1)[0])
                    tries += 1
                    if cand == 0 or cand == pos_i or cand in forbid_i:
                        continue
                    negs.append(cand)
                while len(negs) < num_neg:
                    cand = rng.randint(1, n_items)
                    if cand != pos_i and cand not in forbid_i:
                        negs.append(cand)

                candidates[i, 0] = pos_i
                candidates[i, 1:] = torch.tensor(negs, dtype=torch.long)

            input_tensor = torch.LongTensor(batch_inputs).to(device)
            candidates = candidates.to(device)
            scores = model.predict_candidates(input_tensor, candidates)

            pos_scores = scores[:, 0]
            neg_scores = scores[:, 1:]
            better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
            ranks = (better + 1).tolist()

            for r in ranks:
                total += 1
                for k in ks:
                    if r <= k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / math.log2(r + 1)

            processed_batches += 1
            batch_inputs, batch_pos, batch_forbid = [], [], []

            if processed_batches % 5 == 0 or processed_batches == total_batches:
                elapsed = time.time() - start_t
                avg = elapsed / max(1, processed_batches)
                eta = (total_batches - processed_batches) * avg
                print(f"[Pop-{mode}] batches {processed_batches}/{total_batches} "
                      f"elapsed {fmt_sec(elapsed)} ETA {fmt_sec(eta)}")

    metrics = {"total": total}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    return metrics


# -------------------------
# Full-corpus exact rank eval
# -------------------------
@torch.no_grad()
def eval_full_corpus_exact_rank(
    model,
    splits,
    n_items,
    max_len,
    mode="valid",
    num_eval_users=5000,
    batch_size=256,
    chunk_size=50000,
    device="cuda",
    score_dtype=torch.float16,
    emb_on_gpu=True,
    ks=(1, 10, 50, 200, 1000),
    seed=123,
):
    assert mode in ("valid", "test")
    model.eval()

    rng = random.Random(seed)
    if num_eval_users is not None and 0 < num_eval_users < len(splits):
        eval_splits = rng.sample(splits, num_eval_users)
    else:
        eval_splits = splits

    # item embeddings
    item_emb = model.item_emb.weight.detach()
    if emb_on_gpu and str(device).startswith("cuda"):
        item_emb = item_emb.to(device)
    else:
        item_emb = item_emb.cpu()

    total = len(eval_splits)
    hits_all = {k: 0 for k in ks}
    hits_excl = {k: 0 for k in ks}
    ranks_all = []
    ranks_excl = []
    bad_tgt_in_hist = 0

    def make_hist_pos_forbid(s):
        seq_train = s["seq_train"]
        if mode == "valid":
            hist = seq_train
            pos = int(s["valid_item"])
        else:
            hist = seq_train + [int(s["valid_item"])]
            pos = int(s["test_item"])
        hist = [int(x) for x in hist]
        pos = int(pos)

        forbid = set(x for x in hist if x != 0)
        if pos in forbid:
            # 数据脏：target 出现在 history
            nonlocal bad_tgt_in_hist
            bad_tgt_in_hist += 1
            forbid.discard(pos)

        x = pad_left(hist[-max_len:], max_len, pad=0)
        return x, pos, forbid

    start = time.time()
    pbar = tqdm(total=total, desc=f"full-corpus({mode})", dynamic_ncols=True)

    for bi in range(0, total, batch_size):
        batch = eval_splits[bi:bi + batch_size]
        batch_x = []
        batch_pos = []
        batch_forbid_lists = []

        for s in batch:
            x, pos, forbid = make_hist_pos_forbid(s)
            batch_x.append(x)
            batch_pos.append(pos)
            batch_forbid_lists.append(sorted(forbid))

        B = len(batch_x)
        x_t = torch.tensor(batch_x, dtype=torch.long, device=device)

        # SASRec user repr: feats[:, -1, :]
        feats = model.log2feats(x_t)  # [B,L,H]
        u = feats[:, -1, :]          # [B,H]
        u_sc = u.to(dtype=score_dtype)

        pos_ids = torch.tensor(batch_pos, dtype=torch.long, device=device)
        pos_emb = item_emb.index_select(0, pos_ids.to(item_emb.device))
        # pos_score 用 fp32 比较稳
        pos_score = (u.float() * pos_emb.float()).sum(dim=1)  # [B]

        # count how many items beat pos_score (ALL items)
        better_all = torch.zeros((B,), dtype=torch.long, device=device)
        N = n_items
        total_chunks = (N + chunk_size - 1) // chunk_size

        for ci in range(total_chunks):
            st = 1 + ci * chunk_size
            ed = min(N + 1, st + chunk_size)  # exclusive
            emb_chunk = item_emb[st:ed]
            if emb_chunk.device != u_sc.device:
                emb_chunk = emb_chunk.to(u_sc.device)
            emb_chunk = emb_chunk.to(dtype=score_dtype)

            scores = torch.matmul(u_sc, emb_chunk.t()).float()  # [B,C]
            better_all += (scores > pos_score.unsqueeze(1)).sum(dim=1).long()

        rank_all = (better_all + 1).tolist()

        # exclude-history rank：rank_excl = better_all - count(forbid_score > pos_score) + 1
        # forbid 很短（<= max_len），直接算 forbid score（不需要在 chunk 里做 isin）
        max_forbid = max((len(x) for x in batch_forbid_lists), default=0)
        if max_forbid > 0:
            forbid_mat = torch.full((B, max_forbid), -1, dtype=torch.long, device=device)
            forbid_mask = torch.zeros((B, max_forbid), dtype=torch.bool, device=device)
            for i, ids in enumerate(batch_forbid_lists):
                if not ids:
                    continue
                forbid_mat[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                forbid_mask[i, :len(ids)] = True

            forbid_ids = forbid_mat.clamp(min=0)  # -1 -> 0（后面会 mask 掉）
            forbid_emb = item_emb.index_select(0, forbid_ids.view(-1).to(item_emb.device)).view(B, max_forbid, -1)
            forbid_score = (u.float().unsqueeze(1) * forbid_emb.float()).sum(dim=2)  # [B, max_forbid]
            better_forbid = ((forbid_score > pos_score.unsqueeze(1)) & forbid_mask).sum(dim=1).long()
        else:
            better_forbid = torch.zeros((B,), dtype=torch.long, device=device)

        better_excl = better_all - better_forbid
        rank_excl = (better_excl + 1).tolist()

        # metrics
        for r in rank_all:
            ranks_all.append(int(r))
            for k in ks:
                if r <= k:
                    hits_all[k] += 1
        for r in rank_excl:
            ranks_excl.append(int(r))
            for k in ks:
                if r <= k:
                    hits_excl[k] += 1

        pbar.update(B)
        if (bi // batch_size) % 5 == 0:
            # 展示一个中间的 HR@1/HR@1000
            done = len(ranks_all)
            hr1 = hits_all.get(1, 0) / done
            hr1000 = hits_all.get(1000, 0) / done
            pbar.set_postfix({"done": done, "HR@1": f"{hr1:.4f}", "HR@1000": f"{hr1000:.4f}"})

    pbar.close()
    elapsed = time.time() - start

    def pack(prefix, hits, ranks):
        out = {
            f"{prefix}_elapsed_sec": round(elapsed, 3),
            f"{prefix}_total": len(ranks),
            f"{prefix}_mean_rank": float(np.mean(ranks)) if ranks else 0.0,
            f"{prefix}_median_rank": float(np.median(ranks)) if ranks else 0.0,
            f"{prefix}_p90_rank": percentile(ranks, 90),
            f"{prefix}_p95_rank": percentile(ranks, 95),
            f"{prefix}_p99_rank": percentile(ranks, 99),
            f"{prefix}_min_rank": float(min(ranks)) if ranks else 0.0,
            f"{prefix}_max_rank": float(max(ranks)) if ranks else 0.0,
        }
        for k in ks:
            out[f"{prefix}_HR@{k}"] = hits[k] / max(1, len(ranks))
        return out

    result = {
        "mode": mode,
        "n_items": int(n_items),
        "sampled_users": int(len(ranks_all)),
        "bad_tgt_in_hist": int(bad_tgt_in_hist),
        "all": pack("all", hits_all, ranks_all),
        "excl_history": pack("excl", hits_excl, ranks_excl),
    }
    return result


# -------------------------
# Args
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--weights_path", type=str, default="/workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth")

    # model hyperparams (MUST match training)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # sampled eval
    p.add_argument("--eval_ks", type=str, default="10,50,100")
    p.add_argument("--pop_alpha", type=float, default=0.75)

    p.add_argument("--do_fast", action="store_true")
    p.add_argument("--fast_users", type=int, default=2000)
    p.add_argument("--fast_neg", type=int, default=99)
    p.add_argument("--fast_bs", type=int, default=256)

    p.add_argument("--do_strict", action="store_true")
    p.add_argument("--strict_users", type=int, default=2000)
    p.add_argument("--strict_neg", type=int, default=99)
    p.add_argument("--strict_bs", type=int, default=128)

    # full corpus
    p.add_argument("--do_full", action="store_true")
    p.add_argument("--full_users", type=int, default=5000)
    p.add_argument("--full_bs", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=50000)
    p.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--emb_on_gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--full_ks", type=str, default="1,10,50,200,1000")

    return p.parse_args()


def main():
    args = get_args()
    set_all_seeds(args.seed)

    print(f"📥 Loading dataset: {args.dataset_path}")
    with open(args.dataset_path, "rb") as f:
        pkg = pickle.load(f)
    raw_data_list = pkg["data"]
    n_items = int(pkg["n_items"])

    print("✂️ Building strict splits ...")
    splits, item_freq = build_strict_splits(raw_data_list)
    print(f"✅ users={len(splits)}  n_items={n_items}")

    # Build a lightweight args namespace for SASRec
    class _Args: pass
    sas_args = _Args()
    sas_args.max_len = args.max_len
    sas_args.embed_dim = args.embed_dim
    sas_args.num_blocks = args.num_blocks
    sas_args.num_heads = args.num_heads
    sas_args.dropout = args.dropout
    sas_args.device = args.device

    print("🏗️ Initializing SASRec ...")
    model = SASRec(n_items, sas_args)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights_path not found: {args.weights_path}")

    print(f"🔄 Loading weights: {args.weights_path}")
    obj = torch.load(args.weights_path, map_location="cpu")
    state = _pick_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict not strict:")
        print("  missing   =", missing[:20], "..." if len(missing) > 20 else "")
        print("  unexpected=", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model = model.to(args.device)
    model.eval()
    print("✅ Loaded.")

    ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())
    results = {}

    pop_sampler = PopularitySampler(n_items=n_items, item_freq=item_freq, alpha=args.pop_alpha)

    if args.do_fast:
        print("\n⚡ Sampled FAST (uniform negatives) ...")
        t0 = time.time()
        results["fast_valid"] = eval_sampled_uniform(
            model, splits, n_items, args.max_len,
            mode="valid",
            num_eval_users=args.fast_users,
            num_neg=args.fast_neg,
            eval_batch_size=args.fast_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 111
        )
        results["fast_test"] = eval_sampled_uniform(
            model, splits, n_items, args.max_len,
            mode="test",
            num_eval_users=args.fast_users,
            num_neg=args.fast_neg,
            eval_batch_size=args.fast_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 222
        )
        print(f"[FAST] done in {time.time() - t0:.2f}s")
        print(json.dumps(results["fast_valid"], indent=2))
        print(json.dumps(results["fast_test"], indent=2))

    if args.do_strict:
        print("\n🧪 Sampled STRICT-ish (popularity negatives) ...")
        t0 = time.time()
        results["strict_valid"] = eval_sampled_popularity(
            model, splits, n_items, args.max_len, pop_sampler,
            mode="valid",
            num_eval_users=args.strict_users,
            num_neg=args.strict_neg,
            eval_batch_size=args.strict_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 333
        )
        results["strict_test"] = eval_sampled_popularity(
            model, splits, n_items, args.max_len, pop_sampler,
            mode="test",
            num_eval_users=args.strict_users,
            num_neg=args.strict_neg,
            eval_batch_size=args.strict_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 444
        )
        print(f"[STRICT] done in {time.time() - t0:.2f}s")
        print(json.dumps(results["strict_valid"], indent=2))
        print(json.dumps(results["strict_test"], indent=2))

    if args.do_full:
        print("\n🧱 Full-corpus exact rank (真实全库排名) ...")
        full_ks = tuple(int(x) for x in args.full_ks.split(",") if x.strip())
        score_dtype = _parse_dtype(args.score_dtype)

        results["full_valid"] = eval_full_corpus_exact_rank(
            model, splits, n_items, args.max_len,
            mode="valid",
            num_eval_users=args.full_users,
            batch_size=args.full_bs,
            chunk_size=args.chunk_size,
            device=args.device,
            score_dtype=score_dtype,
            emb_on_gpu=bool(args.emb_on_gpu),
            ks=full_ks,
            seed=args.seed + 555,
        )
        print(json.dumps(results["full_valid"], indent=2, ensure_ascii=False))

        results["full_test"] = eval_full_corpus_exact_rank(
            model, splits, n_items, args.max_len,
            mode="test",
            num_eval_users=args.full_users,
            batch_size=args.full_bs,
            chunk_size=args.chunk_size,
            device=args.device,
            score_dtype=score_dtype,
            emb_on_gpu=bool(args.emb_on_gpu),
            ks=full_ks,
            seed=args.seed + 666,
        )
        print(json.dumps(results["full_test"], indent=2, ensure_ascii=False))

    print("\n📌 ALL RESULTS:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
