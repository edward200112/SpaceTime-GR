# HardMiningGRPO/check_teacher_alignment.py
import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# 适配你的工程结构
from SASRec import SASRec


def pad_left(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    seq = list(seq or [])
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


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


def apply_shift(ids: List[int], shift: int, n_items: int) -> Tuple[List[int], bool]:
    """对 >0 的 id 做 shift；如果越界返回 ok=False"""
    if shift == 0:
        ok = all(0 <= x <= n_items for x in ids)
        return ids, ok
    out = []
    for x in ids:
        if x == 0:
            out.append(0)
            continue
        y = x + shift
        if y < 1 or y > n_items:
            return [], False
        out.append(y)
    return out, True


def sample_negatives(n_items: int, forbid: set, need: int, rng: np.random.Generator) -> List[int]:
    """高效采负样本，避免 forbid；不使用 faiss，全内存随机足够快"""
    negs = []
    # oversample 减少 while 次数
    while len(negs) < need:
        pool = rng.integers(1, n_items + 1, size=need * 4, dtype=np.int64).tolist()
        for x in pool:
            if x not in forbid:
                negs.append(int(x))
                if len(negs) >= need:
                    break
    return negs[:need]


@dataclass
class Metrics:
    total: int = 0
    hr1: int = 0
    hr10: int = 0
    hr50: int = 0
    mrr_sum: float = 0.0
    rank_sum: float = 0.0

    def add(self, rank: int):
        self.total += 1
        if rank <= 1:
            self.hr1 += 1
        if rank <= 10:
            self.hr10 += 1
        if rank <= 50:
            self.hr50 += 1
        self.mrr_sum += 1.0 / float(rank)
        self.rank_sum += float(rank)

    def report(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        return {
            "total": self.total,
            "HR@1": self.hr1 / self.total,
            "HR@10": self.hr10 / self.total,
            "HR@50": self.hr50 / self.total,
            "MRR": self.mrr_sum / self.total,
            "MeanRank": self.rank_sum / self.total,
        }


@torch.no_grad()
def eval_variant(
    sasrec: SASRec,
    n_items: int,
    samples: List[Dict[str, Any]],
    max_len: int,
    num_neg: int,
    batch_size: int,
    device: str,
    reverse_history: bool,
    shift: int,
    seed: int,
    desc: str,
    show_examples: int = 3,
):
    rng = np.random.default_rng(seed + 12345)

    metrics = Metrics()
    bad_leak_tgt_in_hist = 0
    bad_oob = 0
    bad_short = 0

    # 记录几个最好/最差例子帮助肉眼判断
    best = []  # (rank, ex)
    worst = [] # (rank, ex)

    def push_best(rank, ex):
        best.append((rank, ex))
        best.sort(key=lambda x: x[0])
        del best[show_examples:]

    def push_worst(rank, ex):
        worst.append((rank, ex))
        worst.sort(key=lambda x: -x[0])
        del worst[show_examples:]

    # mini-batch
    buf_hist = []
    buf_cand = []
    buf_meta = []

    for ex in tqdm(samples, desc=desc, dynamic_ncols=True):
        hist = ex.get("history_item_ids", None)
        tgt = ex.get("target_item_id", None)
        if hist is None or tgt is None:
            continue

        try:
            hist = [int(x) for x in hist]
            tgt = int(tgt)
        except Exception:
            continue

        if reverse_history:
            hist = list(reversed(hist))

        # 基本 sanity：target 不应出现在 history
        if tgt in set(hist):
            bad_leak_tgt_in_hist += 1

        # shift 检测（很关键：定位 off-by-one）
        hist2, ok1 = apply_shift(hist, shift, n_items)
        tgt2_list, ok2 = apply_shift([tgt], shift, n_items)
        if not (ok1 and ok2):
            bad_oob += 1
            continue
        tgt2 = tgt2_list[0]

        if len(hist2) < 1:
            bad_short += 1
            continue

        # pad 到 max_len（SASRec 假设最后位是最新）
        hist_pad = pad_left(hist2, max_len, pad=0)

        forbid = set(hist2)
        forbid.add(0)
        forbid.add(tgt2)

        negs = sample_negatives(n_items, forbid=forbid, need=num_neg, rng=rng)
        cand = [tgt2] + negs  # 位置0是 target，方便算 rank

        buf_hist.append(hist_pad)
        buf_cand.append(cand)
        buf_meta.append({"tgt": tgt, "tgt_used": tgt2, "hist_tail": hist2[-10:]})

        if len(buf_hist) >= batch_size:
            input_ids = torch.tensor(buf_hist, dtype=torch.long, device=device)
            cand_ids = torch.tensor(buf_cand, dtype=torch.long, device=device)
            scores = sasrec.predict_candidates(input_ids, cand_ids)  # [B, 1+neg]
            # rank: 1 + count(scores_neg > score_tgt)
            tgt_scores = scores[:, 0]
            neg_scores = scores[:, 1:]
            better = (neg_scores > tgt_scores.unsqueeze(1)).sum(dim=1)
            ranks = (better + 1).detach().cpu().tolist()

            for r, meta in zip(ranks, buf_meta):
                metrics.add(int(r))
                # 存几个例子
                if r <= 10:
                    push_best(int(r), meta)
                if r >= num_neg:
                    push_worst(int(r), meta)

            buf_hist, buf_cand, buf_meta = [], [], []

    # flush
    if buf_hist:
        input_ids = torch.tensor(buf_hist, dtype=torch.long, device=device)
        cand_ids = torch.tensor(buf_cand, dtype=torch.long, device=device)
        scores = sasrec.predict_candidates(input_ids, cand_ids)
        tgt_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        better = (neg_scores > tgt_scores.unsqueeze(1)).sum(dim=1)
        ranks = (better + 1).detach().cpu().tolist()
        for r, meta in zip(ranks, buf_meta):
            metrics.add(int(r))
            if r <= 10:
                push_best(int(r), meta)
            if r >= num_neg:
                push_worst(int(r), meta)

    rep = metrics.report()
    rep.update({
        "reverse_history": int(reverse_history),
        "shift": shift,
        "bad_tgt_in_hist": bad_leak_tgt_in_hist,
        "bad_oob_after_shift": bad_oob,
        "bad_short_hist": bad_short,
    })

    print("\n" + "=" * 90)
    print(f"[RESULT] {desc}")
    for k, v in rep.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.6f}")
        else:
            print(f"  {k:18s}: {v}")
    print("  [BEST examples] (rank small is good)")
    for r, meta in best:
        print(f"    rank={r:4d} tgt={meta['tgt']} tgt_used={meta['tgt_used']} hist_tail={meta['hist_tail']}")
    print("  [WORST examples] (rank large is bad)")
    for r, meta in worst:
        print(f"    rank={r:4d} tgt={meta['tgt']} tgt_used={meta['tgt_used']} hist_tail={meta['hist_tail']}")
    print("=" * 90 + "\n")

    return rep


def read_samples(jsonl_path: str, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            # reservoir-like：简单随机抽样
            if len(samples) < max_samples:
                samples.append(ex)
            else:
                j = rng.randint(0, len(samples))
                if j < max_samples:
                    samples[j] = ex
    return samples


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--num_neg", type=int, default=199)  # 200候选规模足够看趋势
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device

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

    print(f"[INFO] sampling up to {args.max_samples} examples from {args.jsonl}")
    samples = read_samples(args.jsonl, max_samples=args.max_samples, seed=args.seed)
    print(f"[OK] sampled = {len(samples)}")

    variants = [
        (False, 0,  "forward + shift0"),
        (True,  0,  "reversed + shift0"),
        (False, -1, "forward + shift-1"),
        (True,  -1, "reversed + shift-1"),
        (False, +1, "forward + shift+1"),
        (True,  +1, "reversed + shift+1"),
    ]

    results = []
    for rev, sh, name in variants:
        rep = eval_variant(
            sasrec=sasrec,
            n_items=n_items,
            samples=samples,
            max_len=args.sasrec_max_len,
            num_neg=args.num_neg,
            batch_size=args.batch_size,
            device=device,
            reverse_history=rev,
            shift=sh,
            seed=args.seed,
            desc=name,
        )
        results.append(rep)

    # 汇总：按 HR@10 / HR@1 排序
    def key_fn(x):
        return (x.get("HR@10", 0.0), x.get("HR@1", 0.0), -x.get("MeanRank", 1e9))

    results_sorted = sorted(results, key=key_fn, reverse=True)

    print("\n" + "#" * 90)
    print("[SUMMARY] sort by HR@10, then HR@1, then MeanRank (lower is better)")
    for r in results_sorted:
        print(f"  name=reverse{r['reverse_history']}_shift{r['shift']:<+d}  "
              f"HR@1={r.get('HR@1',0):.6f} HR@10={r.get('HR@10',0):.6f} "
              f"MRR={r.get('MRR',0):.6f} MeanRank={r.get('MeanRank',0):.2f} "
              f"bad_tgt_in_hist={r['bad_tgt_in_hist']}")
    print("#" * 90 + "\n")


if __name__ == "__main__":
    main()
