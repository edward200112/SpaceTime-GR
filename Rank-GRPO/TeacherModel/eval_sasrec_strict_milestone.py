#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_sasrec_strict_milestone.py

Strict milestone evaluation for SASRec with popularity-biased sampled negatives.

- Strict leave-one-out split: train/valid/test (train excludes valid/test)
- Negatives: popularity-biased (freq^alpha), filtered by user's full history set
- Ranking: score ONLY candidates via model.predict_candidates (fast, no full softmax)

Example:
python ./TeacherModel/eval_sasrec_strict_milestone.py \
  --dataset_path ./SASRec_Data/sasrec_dataset.pkl \
  --ckpt_path ./SASRec_Data/sasrec_full_latest.pt \
  --mode valid \
  --eval_users 2000 \
  --eval_neg 199 \
  --eval_batch_size 256 \
  --pop_alpha 0.75 \
  --max_len 50 \
  --embed_dim 128 --num_blocks 2 --num_heads 2 --dropout 0.2 \
  --device cuda \
  --ks 10,50,100

Run both valid and test:
python ./TeacherModel/eval_sasrec_strict_milestone.py ... --mode valid
python ./TeacherModel/eval_sasrec_strict_milestone.py ... --mode test
"""

import os
import re
import json
import math
import time
import pickle
import random
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from SASRec import SASRec


# =========================
# Utils
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_sec(x: float) -> str:
    x = int(max(0, x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def pad_left(seq, max_len, pad=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def try_load_model_weights(ckpt_path: str, model: torch.nn.Module, device: str):
    """
    Supports:
      - .pt full ckpt dict with key "model"
      - .pth weights-only state_dict
    PyTorch 2.6 may default weights_only=True; we try weights_only=False first.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"🔄 Loading checkpoint: {ckpt_path}", flush=True)

    obj = None
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"⚠️ Full checkpoint load failed: {repr(e)}", flush=True)
        print("⚠️ Falling back to weights-only load.", flush=True)
        obj = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        print("✅ Loaded model from full .pt checkpoint.", flush=True)
        return

    # weights-only
    model.load_state_dict(obj)
    print("✅ Loaded model from weights-only checkpoint.", flush=True)


# =========================
# Strict split (leave-one-out): train/valid/test
# =========================
def build_strict_splits(raw_data_list):
    """
    raw_data_list: list of dict {user_id, sequence: [int]}
    Return:
      splits: list of dict with
        uid, seq_train, valid_item, test_item, full_set
      item_freq: Counter counted on seq_train only
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
            "full_set": full_set
        })

    return splits, item_freq


# =========================
# Popularity Sampler
# =========================
class PopularitySampler:
    def __init__(self, n_items: int, item_freq: Counter, alpha: float = 0.75):
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

    def sample(self, size: int) -> np.ndarray:
        # Draw from 0..n_items but prob[0]=0, then fix rare 0
        x = np.random.choice(self.items, size=size, replace=True, p=self.probs)
        if np.any(x == 0):
            x[x == 0] = 1
        return x


# =========================
# Strict evaluation (pop-neg), with batched sampling
# =========================
@torch.no_grad()
def strict_pop_eval(
    model,
    splits,
    n_items: int,
    max_len: int,
    pop_sampler: PopularitySampler,
    mode: str = "valid",
    eval_users: int = 2000,
    num_neg: int = 199,
    eval_batch_size: int = 256,
    device: str = "cuda",
    ks=(10, 50, 100),
    seed: int = 42,
    oversample: int = 8,
    log_every_batches: int = 1,
):
    """
    mode: valid or test
    Candidates per user: C = 1 + num_neg
    Negatives: popularity-biased, filtered to exclude pos and user's full_set.
    Uses batched sampling pool to reduce pop_sampler.sample(1) overhead.

    oversample: pool size multiplier; bigger => fewer refill loops, more CPU work.
    """
    assert mode in ("valid", "test")
    model.eval()

    rng = random.Random(seed + (1 if mode == "valid" else 2))

    if eval_users is not None and 0 < eval_users < len(splits):
        eval_splits = rng.sample(splits, eval_users)
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
    processed_users = 0
    processed_batches = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []

    def build_candidates_batch(pos_list, forbid_list) -> np.ndarray:
        """
        Build candidates array [B, C] with:
          candidates[:,0] = pos
          candidates[:,1:] = pop-neg filtered
        Uses batched pop sampling pools + fallback uniform fill.
        """
        B = len(pos_list)
        candidates = np.zeros((B, C), dtype=np.int64)
        candidates[:, 0] = np.array(pos_list, dtype=np.int64)

        # For each row, we fill num_neg negatives.
        # We'll generate pools in chunks to reduce sampler calls.
        # Each row gets pool size = num_neg * oversample.
        pool = pop_sampler.sample(B * num_neg * oversample).reshape(B, num_neg * oversample)

        for i in range(B):
            pos_i = int(pos_list[i])
            forbid = forbid_list[i]

            negs = []
            row = pool[i]
            for cand in row:
                cand = int(cand)
                if cand == 0 or cand == pos_i or (cand in forbid):
                    continue
                negs.append(cand)
                if len(negs) >= num_neg:
                    break

            # If still not enough, refill with additional pools (few tries), then fallback uniform.
            refill_tries = 0
            while len(negs) < num_neg and refill_tries < 3:
                refill_tries += 1
                extra = pop_sampler.sample(num_neg * oversample)
                for cand in extra:
                    cand = int(cand)
                    if cand == 0 or cand == pos_i or (cand in forbid):
                        continue
                    negs.append(cand)
                    if len(negs) >= num_neg:
                        break

            while len(negs) < num_neg:
                cand = rng.randint(1, n_items)
                if cand != pos_i and cand not in forbid:
                    negs.append(cand)

            candidates[i, 1:] = np.array(negs[:num_neg], dtype=np.int64)

        return candidates

    # main loop
    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)

            t_build0 = time.time()
            cand_np = build_candidates_batch(batch_pos, batch_forbid)
            t_build1 = time.time()

            t_score0 = time.time()
            input_tensor = torch.LongTensor(batch_inputs).to(device)
            cand_tensor = torch.from_numpy(cand_np).long().to(device)
            scores = model.predict_candidates(input_tensor, cand_tensor)  # [B,C]
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t_score1 = time.time()

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

            processed_users += B
            processed_batches += 1

            if processed_batches % log_every_batches == 0:
                elapsed = time.time() - start_t
                avg = elapsed / max(1, processed_batches)
                eta = (total_batches - processed_batches) * avg
                print(
                    f"[StrictPop-{mode}] {processed_users}/{total_users} users | "
                    f"{processed_batches}/{total_batches} batches | "
                    f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
                    f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}",
                    flush=True
                )

            batch_inputs, batch_pos, batch_forbid = [], [], []

    # flush remainder
    if batch_inputs:
        B = len(batch_inputs)

        t_build0 = time.time()
        cand_np = build_candidates_batch(batch_pos, batch_forbid)
        t_build1 = time.time()

        t_score0 = time.time()
        input_tensor = torch.LongTensor(batch_inputs).to(device)
        cand_tensor = torch.from_numpy(cand_np).long().to(device)
        scores = model.predict_candidates(input_tensor, cand_tensor)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_score1 = time.time()

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

        processed_users += B
        processed_batches += 1

        elapsed = time.time() - start_t
        avg = elapsed / max(1, processed_batches)
        eta = (total_batches - processed_batches) * avg
        print(
            f"[StrictPop-{mode}] {processed_users}/{total_users} users | "
            f"{processed_batches}/{total_batches} batches | "
            f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
            f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}",
            flush=True
        )

    elapsed = time.time() - start_t
    metrics = {
        "mode": mode,
        "total": int(total),
        "elapsed_sec": float(elapsed),
        "candidates_per_user": int(C),
        "neg_sampling": "popularity_biased",
        "num_neg": int(num_neg),
        "oversample": int(oversample),
    }
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0

    print(f"[StrictPop-{mode}] DONE. evaluated={total}/{total_users}, elapsed={elapsed:.2f}s", flush=True)
    return metrics


# =========================
# Args
# =========================
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_path", type=str, required=True, help="sasrec_dataset.pkl")
    p.add_argument("--ckpt_path", type=str, required=True, help=".pt full ckpt or .pth weights")
    p.add_argument("--mode", type=str, default="valid", choices=["valid", "test"])

    # model config (must match training)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # strict eval config
    p.add_argument("--eval_users", type=int, default=2000)
    p.add_argument("--eval_neg", type=int, default=199)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--pop_alpha", type=float, default=0.75)
    p.add_argument("--ks", type=str, default="10,50,100")
    p.add_argument("--seed", type=int, default=42)

    # performance
    p.add_argument("--oversample", type=int, default=8, help="pop sampling pool multiplier per user")
    p.add_argument("--log_every_batches", type=int, default=1)

    # optional output
    p.add_argument("--save_json", type=str, default="", help="save metrics json to this path (optional)")

    return p.parse_args()


# =========================
# Main
# =========================
def main():
    args = get_args()
    set_all_seeds(args.seed)

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    print(f"📌 Ks={ks}", flush=True)

    print(f"📥 Loading dataset: {args.dataset_path}", flush=True)
    with open(args.dataset_path, "rb") as f:
        pkg = pickle.load(f)

    raw_data_list = pkg["data"]
    n_items = int(pkg["n_items"])

    print("✂️ Building strict splits (train/valid/test) ...", flush=True)
    splits, item_freq = build_strict_splits(raw_data_list)
    print(f"✅ Users after split: {len(splits)}", flush=True)
    print(f"✅ n_items={n_items}, max_len={args.max_len}", flush=True)
    print(f"✅ Train freq entries={len(item_freq)}", flush=True)

    pop_sampler = PopularitySampler(n_items=n_items, item_freq=item_freq, alpha=args.pop_alpha)

    # Build model args-like object
    class ModelArgs:
        pass

    margs = ModelArgs()
    margs.max_len = args.max_len
    margs.embed_dim = args.embed_dim
    margs.num_blocks = args.num_blocks
    margs.num_heads = args.num_heads
    margs.dropout = args.dropout
    margs.device = args.device

    print("🏗️ Initializing SASRec ...", flush=True)
    model = SASRec(n_items, margs).to(args.device)

    # Ensure predict_candidates exists
    if not hasattr(model, "predict_candidates"):
        raise AttributeError(
            "Your SASRec model must implement predict_candidates(input_ids, candidate_ids). "
            "Please add it to SASRec.py (you already did in your pasted code)."
        )

    try_load_model_weights(args.ckpt_path, model, args.device)

    print(f"\n🧪 Running STRICT milestone eval (mode={args.mode}) ...", flush=True)
    t0 = time.time()
    metrics = strict_pop_eval(
        model=model,
        splits=splits,
        n_items=n_items,
        max_len=args.max_len,
        pop_sampler=pop_sampler,
        mode=args.mode,
        eval_users=args.eval_users,
        num_neg=args.eval_neg,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        ks=ks,
        seed=args.seed,
        oversample=args.oversample,
        log_every_batches=args.log_every_batches,
    )
    t1 = time.time()

    print("\n========================================")
    print("📌 STRICT POP-NEG MILESTONE REPORT")
    print("========================================")
    print(json.dumps(metrics, indent=2))
    print(f"Total wall time: {t1 - t0:.2f}s")
    print("========================================\n")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Saved metrics to: {args.save_json}", flush=True)


if __name__ == "__main__":
    main()
