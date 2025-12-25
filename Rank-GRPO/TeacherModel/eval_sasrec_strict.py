import os
import math
import json
import pickle
import random
import argparse
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from SASRec import SASRec


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


def build_strict_splits(raw_data_list):
    splits = []
    item_freq = Counter()

    for entry in raw_data_list:
        uid = str(entry["user_id"])
        seq = entry["sequence"]
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


@torch.no_grad()
def eval_mode(model, splits, n_items, max_len, pop_sampler: PopularitySampler,
              mode="valid", num_users=20000, num_neg=999, batch_size=8,
              device="cuda", ks=(10, 50, 100)):
    assert mode in ("valid", "test")
    model.eval()

    if num_users is not None and num_users > 0 and num_users < len(splits):
        eval_splits = random.sample(splits, num_users)
    else:
        eval_splits = splits

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    total = 0

    def make_input(s):
        if mode == "valid":
            hist = s["seq_train"]
            pos = s["valid_item"]
        else:
            hist = s["seq_train"] + [s["valid_item"]]
            pos = s["test_item"]
        x = pad_left(hist[-max_len:], max_len, pad=0)
        return x, int(pos), s["full_set"]

    batch_inputs, batch_pos, batch_forbid = [], [], []

    for s in tqdm(eval_splits, desc=f"Eval-{mode}"):
        x, pos, forbid = make_input(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == batch_size:
            X = torch.LongTensor(batch_inputs).to(device)
            logits = model.predict_full(X)
            logits[:, 0] = -float("inf")

            for i in range(len(batch_inputs)):
                pos_i = batch_pos[i]
                forbid_i = batch_forbid[i]

                negs = []
                tries = 0
                while len(negs) < num_neg and tries < num_neg * 10:
                    cand = int(pop_sampler.sample(1)[0])
                    tries += 1
                    if cand == 0 or cand == pos_i or cand in forbid_i:
                        continue
                    negs.append(cand)
                if len(negs) < num_neg:
                    while len(negs) < num_neg:
                        cand = random.randint(1, n_items)
                        if cand != pos_i and cand not in forbid_i:
                            negs.append(cand)

                candidates = [pos_i] + negs
                cand_tensor = torch.LongTensor(candidates).to(device)
                scores = logits[i].gather(0, cand_tensor)

                _, order = torch.sort(scores, descending=True)
                rank = (order == 0).nonzero(as_tuple=False)
                if rank.numel() == 0:
                    continue
                rank = int(rank.item()) + 1

                total += 1
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / math.log2(rank + 1)

            batch_inputs, batch_pos, batch_forbid = [], [], []

    # flush
    if batch_inputs:
        X = torch.LongTensor(batch_inputs).to(device)
        logits = model.predict_full(X)
        logits[:, 0] = -float("inf")

        for i in range(len(batch_inputs)):
            pos_i = batch_pos[i]
            forbid_i = batch_forbid[i]

            negs = []
            tries = 0
            while len(negs) < num_neg and tries < num_neg * 10:
                cand = int(pop_sampler.sample(1)[0])
                tries += 1
                if cand == 0 or cand == pos_i or cand in forbid_i:
                    continue
                negs.append(cand)
            if len(negs) < num_neg:
                while len(negs) < num_neg:
                    cand = random.randint(1, n_items)
                    if cand != pos_i and cand not in forbid_i:
                        negs.append(cand)

            candidates = [pos_i] + negs
            cand_tensor = torch.LongTensor(candidates).to(device)
            scores = logits[i].gather(0, cand_tensor)

            _, order = torch.sort(scores, descending=True)
            rank = (order == 0).nonzero(as_tuple=False)
            if rank.numel() == 0:
                continue
            rank = int(rank.item()) + 1

            total += 1
            for k in ks:
                if rank <= k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / math.log2(rank + 1)

    if total == 0:
        return {"total": 0}

    out = {"total": total}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / total
        out[f"NDCG@{k}"] = ndcgs[k] / total
    return out


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default="./SASRec_Data/sasrec_dataset.pkl")
    p.add_argument("--ckpt", type=str, required=True, help="sasrec_full_latest.pt or .pth")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model cfg (must match training)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    # eval cfg
    p.add_argument("--users", type=int, default=20000)
    p.add_argument("--num_neg", type=int, default=999)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--pop_alpha", type=float, default=0.75)
    p.add_argument("--ks", type=str, default="10,50,100")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = get_args()
    set_all_seeds(args.seed)

    with open(args.dataset_path, "rb") as f:
        pkg = pickle.load(f)
    raw_data_list = pkg["data"]
    n_items = int(pkg["n_items"])

    splits, item_freq = build_strict_splits(raw_data_list)
    pop_sampler = PopularitySampler(n_items, item_freq, alpha=args.pop_alpha)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    # build a dummy args object for SASRec init
    class Dummy:
        pass
    margs = Dummy()
    margs.max_len = args.max_len
    margs.embed_dim = args.embed_dim
    margs.num_blocks = args.num_blocks
    margs.num_heads = args.num_heads
    margs.dropout = args.dropout
    margs.device = args.device

    model = SASRec(n_items, margs).to(args.device)

    obj = torch.load(args.ckpt, map_location=args.device)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
    else:
        model.load_state_dict(obj)

    print("✅ Model loaded.")
    print(f"Users for eval: {args.users}, num_neg={args.num_neg}, batch={args.batch_size}, ks={ks}")

    valid = eval_mode(model, splits, n_items, args.max_len, pop_sampler,
                      mode="valid", num_users=args.users, num_neg=args.num_neg,
                      batch_size=args.batch_size, device=args.device, ks=ks)
    test = eval_mode(model, splits, n_items, args.max_len, pop_sampler,
                     mode="test", num_users=args.users, num_neg=args.num_neg,
                     batch_size=args.batch_size, device=args.device, ks=ks)

    print("========================================")
    print("STRICT SAMPLE-BASED EVAL REPORT")
    print("========================================")
    print("VALID:", valid)
    print("TEST: ", test)
    print("========================================")


if __name__ == "__main__":
    main()
