#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_sasrec_metrics.py

Example:
python evaluate_sasrec_metrics.py \
  --dataset_path ./SASRec_Data/sasrec_dataset.pkl \
  --weights_path /workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth \
  --max_len 50 --embed_dim 128 --num_blocks 2 --num_heads 2 --dropout 0.2 \
  --do_fast --fast_users 2000 --fast_neg 99 --fast_bs 256 \
  --do_strict --strict_users 2000 --strict_neg 99 --strict_bs 128 \
  --eval_ks 10,50,100
"""

import os
import json
import math
import time
import pickle
import random
import argparse
from collections import Counter

import numpy as np
import torch

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
        seq_train = [int(x) for x in seq[:-2]]  # strict exclude valid/test
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
# Popularity Sampler (power-law smoothing)
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
# FAST eval (uniform negatives, vectorized oversample)
# -------------------------
@torch.no_grad()
def eval_sampled_uniform_fast(
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

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)

            # candidates: [B, 1+num_neg]
            candidates = np.zeros((B, C), dtype=np.int64)
            candidates[:, 0] = np.array(batch_pos, dtype=np.int64)

            # oversample pool to reduce rejection
            oversample = 8
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

            scores = model.predict_candidates(input_tensor, cand_tensor)

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
                print(f"[FastEval-{mode}] batches {processed_batches}/{total_batches} "
                      f"elapsed {fmt_sec(elapsed)} ETA {fmt_sec(eta)}")

    # flush remainder
    if batch_inputs:
        B = len(batch_inputs)
        candidates = np.zeros((B, C), dtype=np.int64)
        candidates[:, 0] = np.array(batch_pos, dtype=np.int64)

        oversample = 8
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
        scores = model.predict_candidates(input_tensor, cand_tensor)

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

    metrics = {"total": total}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    return metrics


# -------------------------
# STRICT eval (popularity negatives, slower)
# -------------------------
@torch.no_grad()
def eval_sampled_pop_strict(
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
                pos_i = batch_pos[i]
                forbid_i = batch_forbid[i]

                negs = []
                tries = 0
                while len(negs) < num_neg and tries < num_neg * 20:
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
                print(f"[StrictEval-{mode}] batches {processed_batches}/{total_batches} "
                      f"elapsed {fmt_sec(elapsed)} ETA {fmt_sec(eta)}")

    # remainder略（对齐训练代码的话也可以补上，这里省略不影响大多数情况）
    metrics = {"total": total}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    return metrics


# -------------------------
# Args
# -------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--weights_path", type=str,
                   default="/workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth")

    # model hyperparams (MUST match training)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_ks", type=str, default="10,50,100")

    # sampler
    p.add_argument("--pop_alpha", type=float, default=0.75)

    # FAST eval
    p.add_argument("--do_fast", action="store_true")
    p.add_argument("--fast_users", type=int, default=2000)
    p.add_argument("--fast_neg", type=int, default=99)
    p.add_argument("--fast_bs", type=int, default=256)

    # STRICT eval
    p.add_argument("--do_strict", action="store_true")
    p.add_argument("--strict_users", type=int, default=2000)
    p.add_argument("--strict_neg", type=int, default=99)
    p.add_argument("--strict_bs", type=int, default=128)

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

    pop_sampler = PopularitySampler(n_items=n_items, item_freq=item_freq, alpha=args.pop_alpha)

    # Build a lightweight args namespace for SASRec
    class _Args: pass
    sas_args = _Args()
    sas_args.max_len = args.max_len
    sas_args.embed_dim = args.embed_dim
    sas_args.num_blocks = args.num_blocks
    sas_args.num_heads = args.num_heads
    sas_args.dropout = args.dropout
    sas_args.device = args.device  # some impls might read this

    print("🏗️ Initializing SASRec ...")
    model = SASRec(n_items, sas_args).to(args.device)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights_path not found: {args.weights_path}")

    print(f"🔄 Loading weights: {args.weights_path}")
    state = torch.load(args.weights_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    print("✅ Loaded.")

    ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())
    results = {}

    if args.do_fast:
        print("\n⚡ FAST eval (uniform negatives) ...")
        t0 = time.time()
        results["fast_valid"] = eval_sampled_uniform_fast(
            model, splits, n_items, args.max_len,
            mode="valid",
            num_eval_users=args.fast_users,
            num_neg=args.fast_neg,
            eval_batch_size=args.fast_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 111
        )
        results["fast_test"] = eval_sampled_uniform_fast(
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
        print("\n🧪 STRICT eval (popularity negatives) ...")
        t0 = time.time()
        results["strict_valid"] = eval_sampled_pop_strict(
            model, splits, n_items, args.max_len, pop_sampler,
            mode="valid",
            num_eval_users=args.strict_users,
            num_neg=args.strict_neg,
            eval_batch_size=args.strict_bs,
            device=args.device,
            ks=ks,
            seed=args.seed + 333
        )
        results["strict_test"] = eval_sampled_pop_strict(
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

    print("\n📌 ALL RESULTS:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
