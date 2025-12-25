#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sasrec.py

- Strict leave-one-out split: train/valid/test (train excludes valid/test)
- Training negatives:
  1) In-batch negatives (shuffled positives across batch)
  2) Popularity-biased negatives (power-law smoothing)
  3) Multi-negatives per position (M total = 1 in-batch + (M-1) pop)
- Evaluation:
  A) FAST eval (default every epoch): uniform negatives, small fixed user subset (cheap, stable)
  B) STRICT eval (default every 5 epochs): popularity negatives, sampled strict ranking (costly)
- Early stopping (based on FAST eval metric): patience + min_delta
- Resume from checkpoint (.pt full ckpt or .pth weights-only)

Requires:
  - SASRec.py providing SASRec class and method predict_candidates(input_ids, candidate_ids)

Usage example:
python ./TeacherModel/train_sasrec.py \
  --dataset_path ./SASRec_Data/sasrec_dataset.pkl \
  --output_dir ./SASRec_Data_new \
  --batch_size 4096 \
  --lr 1e-4 \
  --num_epochs 50 \
  --num_negs 4 \
  --pop_alpha 0.75 \
  --do_fast_eval --fast_eval_every 10 --fast_eval_users 2000 --fast_eval_neg 99 --fast_eval_batch_size 256 \
  --do_strict_eval --strict_eval_every 10 --strict_eval_users 2000 --strict_eval_neg 99 --strict_eval_batch_size 128 \
  --early_stop --early_stop_metric NDCG@10 --early_stop_patience 5 --early_stop_min_delta 0.002 \
  --pin_memory --num_workers 14
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SASRec import SASRec


# =========================================================
# Utils
# =========================================================
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


def save_checkpoint(path, model, optimizer, epoch, args, best_metric=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "best_metric": best_metric,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(ckpt, path)


def try_load_checkpoint(path, model, optimizer, device):
    """
    Supports:
      - .pt: dict with model+optimizer+epoch (+rng)
      - .pth: weights-only state_dict
    PyTorch 2.6 default weights_only=True can break loading full ckpt containing numpy/python objects.
    We try weights_only=False first (only if you trust ckpt), else fall back.
    Return: (start_epoch, best_metric_in_ckpt_or_None)
    """
    if not path or not os.path.exists(path):
        return 1, None

    print(f"🔄 Resuming from {path} ...", flush=True)

    obj = None
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    except Exception as e:
        print(f"⚠️ Full checkpoint load failed: {repr(e)}", flush=True)
        print("⚠️ Falling back to weights-only load (model weights only).", flush=True)
        obj = torch.load(path, map_location=device, weights_only=True)

    # Full ckpt
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        if optimizer is not None and obj.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(obj["optimizer"])
                print("✅ Loaded model + optimizer state.", flush=True)
            except Exception as e:
                print(f"⚠️ Optimizer load failed: {repr(e)}. Continue with model only.", flush=True)
        else:
            print("⚠️ No optimizer state in checkpoint. Loaded model only.", flush=True)

        start_epoch = int(obj.get("epoch", 0)) + 1
        best_metric = obj.get("best_metric", None)

        # RNG restore (optional)
        try:
            if obj.get("torch_rng_state") is not None:
                torch.set_rng_state(obj["torch_rng_state"])
            if torch.cuda.is_available() and obj.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(obj["cuda_rng_state"])
            if obj.get("numpy_rng_state") is not None:
                np.random.set_state(obj["numpy_rng_state"])
            if obj.get("python_rng_state") is not None:
                random.setstate(obj["python_rng_state"])
        except Exception as e:
            print(f"⚠️ RNG restore failed (safe to ignore): {repr(e)}", flush=True)

        print(f"⏩ Resume epoch = {start_epoch}", flush=True)
        return start_epoch, best_metric

    # Weights-only
    model.load_state_dict(obj)
    print("✅ Loaded model weights only (weights-only/.pth). Optimizer not restored.", flush=True)
    m = re.search(r'epoch_(\d+)', os.path.basename(path))
    if m:
        return int(m.group(1)) + 1, None
    return 1, None


# =========================================================
# Strict split (leave-one-out): train/valid/test
# =========================================================
def build_strict_splits(raw_data_list):
    """
    raw_data_list: list of dict with keys: user_id, sequence (list[int])
    Return:
      splits: list of dict with
        uid, seq_train, valid_item, test_item, full_set
      item_freq: Counter for popularity (computed from seq_train only)
    """
    splits = []
    item_freq = Counter()

    for entry in raw_data_list:
        uid = str(entry.get("user_id", ""))
        seq = entry.get("sequence", None)
        if not isinstance(seq, (list, tuple)):
            continue

        # Need train + valid + test
        if len(seq) < 3:
            continue

        valid_item = int(seq[-2])
        test_item = int(seq[-1])
        seq_train = [int(x) for x in seq[:-2]]  # strict: exclude valid/test
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


# =========================================================
# Popularity sampler (power-law smoothing)
# =========================================================
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
        # WARNING: calling sample(1) repeatedly is slow for ~1e6 items.
        s = np.random.choice(self.items, size=size, replace=True, p=self.probs)
        if np.any(s == 0):
            s[s == 0] = 1
        return s


# =========================================================
# Dataset (train)
# =========================================================
class SASRecTrainDataset(Dataset):
    """
    Train uses seq_train only:
      input = seq_train[:-1]
      pos   = seq_train[1:]
    Return: input_ids [L], pos_ids [L], full_set
    """
    def __init__(self, splits, n_items, max_len):
        self.splits = splits
        self.n_items = int(n_items)
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        s = self.splits[idx]
        seq_train = s["seq_train"]
        full_set = s["full_set"]

        input_ids = seq_train[:-1]
        pos_ids = seq_train[1:]

        input_ids = input_ids[-(self.max_len - 1):]
        pos_ids = pos_ids[-(self.max_len - 1):]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        pos_ids = [0] * pad_len + pos_ids

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(pos_ids),
            full_set
        )


# =========================================================
# Collator: multi negatives per position
# - in-batch neg: shuffle pos_ids along batch dimension
# - pop neg: sample B*L per neg tensor
# =========================================================
class NegCollator:
    def __init__(self, n_items, pop_sampler: PopularitySampler, num_negs: int = 4, max_resample: int = 3):
        self.n_items = int(n_items)
        self.pop_sampler = pop_sampler
        self.num_negs = int(num_negs)
        self.max_resample = int(max_resample)

    def _filter_conflicts(self, neg_ids_np, forbid_sets, pos_ids_np, input_ids_np):
        """
        neg_ids_np: [B,L] numpy int64
        forbid_sets: list[set] length B
        forbid: 0, pos_ids, input_ids, or in full_set
        """
        B, L = neg_ids_np.shape
        for b in range(B):
            fs = forbid_sets[b]
            row = neg_ids_np[b]
            for j in range(L):
                v = int(row[j])
                if v == 0 or v == int(pos_ids_np[b, j]) or v == int(input_ids_np[b, j]) or (v in fs):
                    row[j] = 0
            neg_ids_np[b] = row
        return neg_ids_np

    def _resample_zeros(self, neg_ids_np, forbid_sets, pos_ids_np, input_ids_np):
        B, L = neg_ids_np.shape
        for _ in range(self.max_resample):
            zeros = (neg_ids_np == 0)
            if not np.any(zeros):
                break
            num = int(zeros.sum())
            repl = self.pop_sampler.sample(num).reshape(-1)
            neg_ids_np[zeros] = repl
            neg_ids_np = self._filter_conflicts(neg_ids_np, forbid_sets, pos_ids_np, input_ids_np)
        neg_ids_np[neg_ids_np == 0] = 1
        return neg_ids_np

    def __call__(self, batch):
        input_ids, pos_ids, forbid_sets = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)  # [B,L]
        pos_ids = torch.stack(pos_ids, dim=0)      # [B,L]
        B, L = pos_ids.shape

        input_np = input_ids.numpy()
        pos_np = pos_ids.numpy()

        negs = []

        # (1) in-batch negatives: shuffle pos_ids along batch dim
        perm = torch.randperm(B)
        inbatch = pos_ids[perm].clone().numpy()
        inbatch = self._filter_conflicts(inbatch, forbid_sets, pos_np, input_np)
        inbatch = self._resample_zeros(inbatch, forbid_sets, pos_np, input_np)
        negs.append(torch.from_numpy(inbatch).long())

        # (2) popularity negatives for remaining
        for _ in range(self.num_negs - 1):
            neg_np = self.pop_sampler.sample(B * L).reshape(B, L)
            neg_np = self._filter_conflicts(neg_np, forbid_sets, pos_np, input_np)
            neg_np = self._resample_zeros(neg_np, forbid_sets, pos_np, input_np)
            negs.append(torch.from_numpy(neg_np).long())

        return input_ids, pos_ids, negs


# =========================================================
# Loss (pairwise BCE)
# =========================================================
def bce_pairwise_loss(criterion, pos_logits, neg_logits, pos_ids):
    idx = torch.where(pos_ids != 0)
    if idx[0].numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    loss = criterion(pos_logits[idx], pos_labels[idx]) + criterion(neg_logits[idx], neg_labels[idx])
    return loss


# =========================================================
# Evaluation (sampled candidates)
# - fast: uniform negatives (cheap)
# - strict: popularity negatives (harder, slower)
# NOTE: This keeps your original "rejection sampling" logic for correctness.
#       Fast eval uses vectorized uniform sampling to be fast.
# =========================================================
@torch.no_grad()
def eval_sampled_uniform_fast(
    model,
    splits,
    n_items,
    max_len,
    mode="valid",
    eval_user_indices=None,  # list[int] indices into splits (fixed set)
    num_eval_users=2000,
    num_neg=99,
    eval_batch_size=256,
    device="cuda",
    ks=(10, 50, 100),
):
    """
    FAST eval:
      - negatives: uniform random, vectorized oversampling
      - fixed user subset recommended via eval_user_indices
    """
    assert mode in ("valid", "test")
    model.eval()

    if eval_user_indices is not None:
        eval_splits = [splits[i] for i in eval_user_indices]
    else:
        if num_eval_users is not None and 0 < num_eval_users < len(splits):
            eval_splits = random.sample(splits, num_eval_users)
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
    processed_users = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)

            # -------- build candidates (vectorized uniform oversample) --------
            t_build0 = time.time()

            candidates = np.zeros((B, C), dtype=np.int64)
            candidates[:, 0] = np.array(batch_pos, dtype=np.int64)

            # oversample to reduce rejection loops
            oversample = 8
            pool = np.random.randint(1, n_items + 1, size=(B, num_neg * oversample), dtype=np.int64)

            # fill per row
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
                    cand = random.randint(1, n_items)
                    if cand != pos_i and cand not in fs:
                        negs.append(cand)
                candidates[i, 1:] = np.array(negs, dtype=np.int64)

            t_build1 = time.time()

            # -------- score candidates --------
            t_score0 = time.time()
            input_tensor = torch.LongTensor(batch_inputs).to(device)
            cand_tensor = torch.from_numpy(candidates).long().to(device)

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
            tqdm.write(
                f"[FastEval-{mode}] {processed_users}/{total_users} users | "
                f"{processed_batches}/{total_batches} batches | "
                f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
                f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}"
            )

            batch_inputs, batch_pos, batch_forbid = [], [], []

    # flush remainder
    if batch_inputs:
        B = len(batch_inputs)
        t_build0 = time.time()

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
                cand = random.randint(1, n_items)
                if cand != pos_i and cand not in fs:
                    negs.append(cand)
            candidates[i, 1:] = np.array(negs, dtype=np.int64)

        t_build1 = time.time()

        t_score0 = time.time()
        input_tensor = torch.LongTensor(batch_inputs).to(device)
        cand_tensor = torch.from_numpy(candidates).long().to(device)
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
        tqdm.write(
            f"[FastEval-{mode}] {processed_users}/{total_users} users | "
            f"{processed_batches}/{total_batches} batches | "
            f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
            f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}"
        )

    elapsed = time.time() - start_t
    metrics = {"total": total, "elapsed_sec": elapsed}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    tqdm.write(f"[FastEval-{mode}] DONE. evaluated={total}/{total_users} users, elapsed={elapsed:.2f}s")
    return metrics


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
):
    """
    STRICT eval (harder):
      - negatives: popularity biased + rejection vs full_set
      - slower, so run less frequently
    """
    assert mode in ("valid", "test")
    model.eval()

    if num_eval_users is not None and 0 < num_eval_users < len(splits):
        eval_splits = random.sample(splits, num_eval_users)
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
    processed_users = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)
            t_build0 = time.time()

            candidates = torch.empty((B, C), dtype=torch.long)
            for i in range(B):
                if i % 64 == 0:
                    tqdm.write(f"[StrictEval-{mode}] building candidates: {i}/{B} ...")

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
                    cand = random.randint(1, n_items)
                    if cand != pos_i and cand not in forbid_i:
                        negs.append(cand)

                candidates[i, 0] = pos_i
                candidates[i, 1:] = torch.tensor(negs, dtype=torch.long)

            t_build1 = time.time()

            t_score0 = time.time()
            input_tensor = torch.LongTensor(batch_inputs).to(device)
            candidates = candidates.to(device)
            scores = model.predict_candidates(input_tensor, candidates)
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
            tqdm.write(
                f"[StrictEval-{mode}] {processed_users}/{total_users} users | "
                f"{processed_batches}/{total_batches} batches | "
                f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
                f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}"
            )

            batch_inputs, batch_pos, batch_forbid = [], [], []

    # flush remainder (optional, keeps code correct)
    if batch_inputs:
        B = len(batch_inputs)
        t_build0 = time.time()

        candidates = torch.empty((B, C), dtype=torch.long)
        for i in range(B):
            if i % 64 == 0:
                tqdm.write(f"[StrictEval-{mode}] building candidates: {i}/{B} ...")

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
                cand = random.randint(1, n_items)
                if cand != pos_i and cand not in forbid_i:
                    negs.append(cand)

            candidates[i, 0] = pos_i
            candidates[i, 1:] = torch.tensor(negs, dtype=torch.long)

        t_build1 = time.time()

        t_score0 = time.time()
        input_tensor = torch.LongTensor(batch_inputs).to(device)
        candidates = candidates.to(device)
        scores = model.predict_candidates(input_tensor, candidates)
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
        tqdm.write(
            f"[StrictEval-{mode}] {processed_users}/{total_users} users | "
            f"{processed_batches}/{total_batches} batches | "
            f"build {t_build1 - t_build0:.2f}s | score {t_score1 - t_score0:.2f}s | "
            f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.2f}s | ETA {fmt_sec(eta)}"
        )

    elapsed = time.time() - start_t
    metrics = {"total": total, "elapsed_sec": elapsed}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total else 0.0
    tqdm.write(f"[StrictEval-{mode}] DONE. evaluated={total}/{total_users} users, elapsed={elapsed:.2f}s")
    return metrics


# =========================================================
# Early Stopper
# =========================================================
class EarlyStopper:
    def __init__(self, metric_name="NDCG@10", patience=5, min_delta=0.002, maximize=True):
        self.metric_name = metric_name
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.maximize = bool(maximize)

        self.best = None
        self.bad_count = 0

    def update(self, metrics: dict):
        if self.metric_name not in metrics:
            raise KeyError(f"EarlyStop metric '{self.metric_name}' not found in metrics keys={list(metrics.keys())}")

        cur = float(metrics[self.metric_name])

        if self.best is None:
            self.best = cur
            self.bad_count = 0
            return True, True, cur, self.best, self.bad_count  # improved, is_best

        improved = (cur >= self.best + self.min_delta) if self.maximize else (cur <= self.best - self.min_delta)

        if improved:
            self.best = cur
            self.bad_count = 0
            return True, True, cur, self.best, self.bad_count
        else:
            self.bad_count += 1
            is_best = False
            should_continue = self.bad_count < self.patience
            return should_continue, is_best, cur, self.best, self.bad_count


# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # training
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--resume_path", type=str, default="./SASRec_Data_new/sasrec_full_latest.pt")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # negatives (training)
    p.add_argument("--num_negs", type=int, default=4, help="M total negatives per position: 1 in-batch + (M-1) pop")
    p.add_argument("--pop_alpha", type=float, default=0.75)

    # dataloader
    p.add_argument("--num_workers", type=int, default=14)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--torch_num_threads", type=int, default=1)

    # perf toggles
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--fused_optim", action="store_true", help="use fused AdamW if available")

    # FAST eval (cheap)
    p.add_argument("--do_fast_eval", action="store_true")
    p.add_argument("--fast_eval_every", type=int, default=1)
    p.add_argument("--fast_eval_users", type=int, default=2000)
    p.add_argument("--fast_eval_neg", type=int, default=99)
    p.add_argument("--fast_eval_batch_size", type=int, default=256)
    p.add_argument("--fast_eval_fixed_users", action="store_true", help="fix eval user subset (recommended)")

    # STRICT eval (costly)
    p.add_argument("--do_strict_eval", action="store_true")
    p.add_argument("--strict_eval_every", type=int, default=5)
    p.add_argument("--strict_eval_users", type=int, default=2000)
    p.add_argument("--strict_eval_neg", type=int, default=99)
    p.add_argument("--strict_eval_batch_size", type=int, default=128)

    # metrics
    p.add_argument("--eval_ks", type=str, default="10,50,100")

    # early stop
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--early_stop_metric", type=str, default="NDCG@10")
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--early_stop_min_delta", type=float, default=0.002)
    p.add_argument("--early_stop_maximize", action="store_true", help="maximize metric (default True)")
    p.add_argument("--early_stop_minimize", action="store_true", help="minimize metric (override maximize)")

    # save
    p.add_argument("--save_every", type=int, default=1, help="save latest every N epochs")
    p.add_argument("--save_best", action="store_true", help="save best checkpoint on FAST eval improvement")

    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(args.seed)

    # perf knobs
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    print(f"📥 Loading dataset from {args.dataset_path} ...", flush=True)
    with open(args.dataset_path, "rb") as f:
        pkg = pickle.load(f)

    raw_data_list = pkg["data"]
    n_items = int(pkg["n_items"])

    print("✂️ Building strict splits (train/valid/test) ...", flush=True)
    splits, item_freq = build_strict_splits(raw_data_list)
    print(f"✅ Users after split: {len(splits)} (dropped short sequences)", flush=True)
    print(f"✅ n_items = {n_items}, max_len = {args.max_len}", flush=True)
    print(f"✅ Train interaction freq entries: {len(item_freq)}", flush=True)

    pop_sampler = PopularitySampler(n_items=n_items, item_freq=item_freq, alpha=args.pop_alpha)

    # Dataset / Loader
    train_ds = SASRecTrainDataset(splits, n_items=n_items, max_len=args.max_len)
    collator = NegCollator(n_items=n_items, pop_sampler=pop_sampler, num_negs=args.num_negs)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collator,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    print("🏗️ Initializing SASRec ...", flush=True)
    model = SASRec(n_items, args).to(args.device)

    # Optim
    try:
        if args.fused_optim:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()

    # Resume
    start_epoch, best_metric = (1, None)
    if args.resume_path:
        start_epoch, best_metric = try_load_checkpoint(args.resume_path, model, optimizer, args.device)

    ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())
    print(f"📌 Eval Ks: {ks}", flush=True)

    # Fixed FAST eval user subset (recommended)
    fast_eval_indices = None
    if args.do_fast_eval and args.fast_eval_fixed_users:
        rng = random.Random(args.seed + 999)
        # fixed indices into splits
        if args.fast_eval_users > 0 and args.fast_eval_users < len(splits):
            fast_eval_indices = rng.sample(range(len(splits)), args.fast_eval_users)
        else:
            fast_eval_indices = list(range(len(splits)))
        print(f"✅ FAST eval fixed users: {len(fast_eval_indices)}", flush=True)

    # Early stopper
    early_stopper = None
    if args.early_stop:
        maximize = True
        if args.early_stop_minimize:
            maximize = False
        elif args.early_stop_maximize:
            maximize = True
        early_stopper = EarlyStopper(
            metric_name=args.early_stop_metric,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            maximize=maximize
        )
        # If resuming, we don't know prior best unless it was stored in ckpt; handled via best_metric
        if best_metric is not None and args.early_stop_metric in best_metric:
            # (optional) if you stored dict; in our save_checkpoint we store scalar best_metric
            pass

    # Track best (scalar)
    best_fast_metric_value = None
    if isinstance(best_metric, (float, int)):
        best_fast_metric_value = float(best_metric)

    print("🚀 Starting Training ...", flush=True)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.train()
        train_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for input_ids, pos_ids, neg_list in pbar:
            input_ids = input_ids.to(args.device, non_blocking=True)
            pos_ids = pos_ids.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # First neg for a forward pass
            neg0 = neg_list[0].to(args.device, non_blocking=True)
            pos_logits, neg_logits = model(input_ids, pos_ids, neg0)
            loss = bce_pairwise_loss(criterion, pos_logits, neg_logits, pos_ids)

            # Multi negatives: compute additional neg logits
            if len(neg_list) > 1:
                for neg_ids in neg_list[1:]:
                    neg_ids = neg_ids.to(args.device, non_blocking=True)
                    _, neg_logits_m = model(input_ids, pos_ids, neg_ids)
                    loss = loss + bce_pairwise_loss(criterion, pos_logits, neg_logits_m, pos_ids)
                loss = loss / float(len(neg_list))

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += float(loss.item())
            steps += 1
            pbar.set_postfix({"loss": f"{train_loss / steps:.4f}"})

        # ---------------- save latest ----------------
        if args.save_every > 0 and (epoch % args.save_every == 0):
            latest_pt = os.path.join(args.output_dir, "sasrec_full_latest.pt")
            latest_pth = os.path.join(args.output_dir, "sasrec_full_latest.pth")
            save_checkpoint(latest_pt, model, optimizer, epoch, args, best_metric=best_fast_metric_value)
            torch.save(model.state_dict(), latest_pth)

            if epoch % 10 == 0:
                save_checkpoint(os.path.join(args.output_dir, f"sasrec_full_epoch_{epoch}.pt"), model, optimizer, epoch, args, best_metric=best_fast_metric_value)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"sasrec_full_epoch_{epoch}.pth"))

        # ---------------- FAST eval ----------------
        fast_metrics_valid = None
        if args.do_fast_eval and (epoch % args.fast_eval_every == 0):
            print("\n⚡ FAST Evaluation (uniform negatives, cheap) ...", flush=True)
            t0 = time.time()
            fast_metrics_valid = eval_sampled_uniform_fast(
                model=model,
                splits=splits,
                n_items=n_items,
                max_len=args.max_len,
                mode="valid",
                eval_user_indices=fast_eval_indices,
                num_eval_users=args.fast_eval_users,
                num_neg=args.fast_eval_neg,
                eval_batch_size=args.fast_eval_batch_size,
                device=args.device,
                ks=ks,
            )
            t1 = time.time()
            print(f"[FastEval-valid] total time: {t1 - t0:.2f}s", flush=True)
            print("FastEval VALID:", json.dumps(fast_metrics_valid, indent=2), flush=True)

            # Optional: quick test as well (costs about the same)
            fast_metrics_test = eval_sampled_uniform_fast(
                model=model,
                splits=splits,
                n_items=n_items,
                max_len=args.max_len,
                mode="test",
                eval_user_indices=fast_eval_indices,
                num_eval_users=args.fast_eval_users,
                num_neg=args.fast_eval_neg,
                eval_batch_size=args.fast_eval_batch_size,
                device=args.device,
                ks=ks,
            )
            t2 = time.time()
            print(f"[FastEval-test ] total time: {t2 - t1:.2f}s", flush=True)
            print("FastEval TEST :", json.dumps(fast_metrics_test, indent=2), flush=True)

            # Save best
            if args.save_best and fast_metrics_valid is not None:
                cur_val = float(fast_metrics_valid.get(args.early_stop_metric, fast_metrics_valid.get("NDCG@10", 0.0)))
                if best_fast_metric_value is None or cur_val > best_fast_metric_value:
                    best_fast_metric_value = cur_val
                    best_path = os.path.join(args.output_dir, "sasrec_best.pt")
                    save_checkpoint(best_path, model, optimizer, epoch, args, best_metric=best_fast_metric_value)
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "sasrec_best.pth"))
                    print(f"🏆 New BEST FAST metric={best_fast_metric_value:.6f}. Saved to {best_path}", flush=True)

            # Early stopping decision
            if args.early_stop and early_stopper is not None and fast_metrics_valid is not None:
                cont, is_best, cur, bestv, bad = early_stopper.update(fast_metrics_valid)
                print(
                    f"🧭 EarlyStop monitor={early_stopper.metric_name} cur={cur:.6f} best={bestv:.6f} "
                    f"bad_count={bad}/{early_stopper.patience} min_delta={early_stopper.min_delta}",
                    flush=True
                )
                if not cont:
                    print("🛑 Early stopping triggered (FAST eval no improvement).", flush=True)
                    break

        # ---------------- STRICT eval (slow) ----------------
        if args.do_strict_eval and (epoch % args.strict_eval_every == 0):
            print("\n🧪 STRICT Evaluation (popularity negatives, costly) ...", flush=True)
            t0 = time.time()
            m_valid = eval_sampled_pop_strict(
                model=model,
                splits=splits,
                n_items=n_items,
                max_len=args.max_len,
                pop_sampler=pop_sampler,
                mode="valid",
                num_eval_users=args.strict_eval_users,
                num_neg=args.strict_eval_neg,
                eval_batch_size=args.strict_eval_batch_size,
                device=args.device,
                ks=ks,
            )
            t1 = time.time()
            print(f"[StrictEval-valid] total time: {t1 - t0:.2f}s", flush=True)
            print("StrictEval VALID:", json.dumps(m_valid, indent=2), flush=True)

            m_test = eval_sampled_pop_strict(
                model=model,
                splits=splits,
                n_items=n_items,
                max_len=args.max_len,
                pop_sampler=pop_sampler,
                mode="test",
                num_eval_users=args.strict_eval_users,
                num_neg=args.strict_eval_neg,
                eval_batch_size=args.strict_eval_batch_size,
                device=args.device,
                ks=ks,
            )
            t2 = time.time()
            print(f"[StrictEval-test ] total time: {t2 - t1:.2f}s", flush=True)
            print("StrictEval TEST :", json.dumps(m_test, indent=2), flush=True)

    print("✅ Training Finished!", flush=True)


if __name__ == "__main__":
    main()
