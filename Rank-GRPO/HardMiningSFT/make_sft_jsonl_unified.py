# HardMiningSFT/make_sft_jsonl_unified.py
import os
import json
import gzip
import time
import math
import argparse
import pickle
import random
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from SASRec import SASRec


# =========================
# MetaDataManager
# =========================
class MetaDataManager:
    def __init__(self, raw_dir, valid_gmap_ids):
        self.raw_dir = raw_dir
        self.valid_gmap_ids = set(valid_gmap_ids)
        self.meta_dict = {}

    def load(self):
        print("📚 Loading Metadata (gmap_id -> text)...")
        if not self.raw_dir or (not os.path.exists(self.raw_dir)):
            print("⚠️ raw_meta_dir not found. Fallback to POI_<id>.")
            return

        files = [f for f in os.listdir(self.raw_dir) if f.startswith("meta-") and f.endswith(".json.gz")]
        loaded = 0
        for fname in files:
            fp = os.path.join(self.raw_dir, fname)
            try:
                with gzip.open(fp, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        gid = d.get("gmap_id")
                        if gid is None or gid not in self.valid_gmap_ids:
                            continue

                        name = (d.get("name") or "Unknown Place").strip()
                        cats = d.get("category")
                        if isinstance(cats, list) and len(cats) > 0:
                            cat = str(cats[0]).strip()
                        elif isinstance(cats, str) and cats.strip():
                            cat = cats.strip()
                        else:
                            cat = "Place"

                        self.meta_dict[gid] = f"{name} ({cat})"
                        loaded += 1
            except Exception:
                continue

        print(f"✅ Metadata loaded: {loaded} items.")

    def get_text(self, gmap_id: str):
        return self.meta_dict.get(gmap_id, f"POI_{gmap_id}")


# =========================
# Utils
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pad_left_np(seq, max_len, pad=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return np.array(seq[-max_len:], dtype=np.int32)
    out = np.full((max_len,), pad, dtype=np.int32)
    out[-len(seq):] = np.array(seq, dtype=np.int32)
    return out

def build_prompt(hist_texts, kind="prompt"):
    hist_str = " -> ".join(hist_texts)
    if kind == "prompt":
        return f"User History: {hist_str}\nPredict the next location:"
    return f"Trajectory: {hist_str}. Suggest the next likely stop:"

def load_teacher_ckpt(path, model, device):
    obj = None
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        print("✅ Loaded teacher from full .pt checkpoint.")
    elif isinstance(obj, dict):
        model.load_state_dict(obj)
        print("✅ Loaded teacher from dict state_dict.")
    else:
        model.load_state_dict(obj)
        print("✅ Loaded teacher weights.")
    return model

def sample_popular_items(pop_items: np.ndarray, n: int, rng: np.random.Generator):
    idx = rng.integers(0, len(pop_items), size=n, endpoint=False)
    return pop_items[idx]

def make_candidate_row(gt_idx, forbid_set, n_items, pop_items, C, oversample, rng):
    need = C - 1
    m = need * oversample
    m_pop = m // 2
    m_uni = m - m_pop

    cand_pop = sample_popular_items(pop_items, m_pop, rng).tolist()
    cand_uni = rng.integers(1, n_items + 1, size=m_uni).tolist()

    pool = cand_pop + cand_uni
    rng.shuffle(pool)

    negs = []
    for x in pool:
        x = int(x)
        if x == 0 or x == gt_idx:
            continue
        if x in forbid_set:
            continue
        negs.append(x)
        if len(negs) >= need:
            break

    while len(negs) < need:
        x = int(rng.integers(1, n_items + 1))
        if x != gt_idx and x not in forbid_set:
            negs.append(x)

    return [int(gt_idx)] + negs

def pick_mixed_neg_from_scores(
    gt_score, cand_ids, cand_scores, forbid_set,
    neg_cap_counter, neg_cap,
    rng, p_hard=0.7, p_semi=0.2, p_easy=0.1, semi_margin=1.0
):
    """
    diff = score_neg - score_gt
      hard: diff >= 0
      semi: -semi_margin <= diff < 0
      easy: diff < -semi_margin
    """
    gt = float(gt_score)

    valid = []
    for cid, sc in zip(cand_ids[1:], cand_scores[1:]):
        cid = int(cid)
        sc = float(sc)
        if cid == 0 or cid in forbid_set:
            continue
        if neg_cap > 0 and neg_cap_counter[cid] >= neg_cap:
            continue
        diff = sc - gt
        valid.append((cid, sc, diff))

    if not valid:
        return None

    hard = [x for x in valid if x[2] >= 0]
    semi = [x for x in valid if (-semi_margin <= x[2] < 0)]
    easy = [x for x in valid if (x[2] < -semi_margin)]

    r = rng.random()
    if r < p_hard:
        bucket = hard
        tag = "hard"
    elif r < p_hard + p_semi:
        bucket = semi
        tag = "semi"
    else:
        bucket = easy
        tag = "easy"

    fallbacks = [bucket, semi, hard, easy, valid]
    chosen = None
    chosen_tag = tag

    for b in fallbacks:
        if not b:
            continue
        if b is hard:
            # boundary hard: minimal positive diff
            b_sorted = sorted(b, key=lambda x: (x[2], -x[1]))
            chosen = b_sorted[0]
            chosen_tag = "hard"
            break
        # semi/easy/any: random
        idx = int(rng.integers(0, len(b)))
        chosen = b[idx]
        if b is semi:
            chosen_tag = "semi"
        elif b is easy:
            chosen_tag = "easy"
        else:
            chosen_tag = "any"
        break

    if chosen is None:
        return None

    cid, sc, diff = chosen
    return cid, sc, chosen_tag, diff


# =========================
# Args
# =========================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)

    ap.add_argument("--sasrec_data_path", type=str, required=True)
    ap.add_argument("--raw_meta_dir", type=str, default="")
    ap.add_argument("--output_jsonl", type=str, required=True)

    # stage2 only
    ap.add_argument("--sasrec_model_path", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--write_buffer", type=int, default=20000)

    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--max_hist_text", type=int, default=5)
    ap.add_argument("--min_seq_len", type=int, default=2, help="min length of raw seq")

    # stage2 candidates
    ap.add_argument("--num_neg", type=int, default=199)
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--pop_top", type=int, default=200000)

    # stage2 difficulty mix
    ap.add_argument("--p_hard", type=float, default=0.7)
    ap.add_argument("--p_semi", type=float, default=0.2)
    ap.add_argument("--p_easy", type=float, default=0.1)
    ap.add_argument("--semi_margin", type=float, default=1.0)

    # caps / limits
    ap.add_argument("--neg_cap", type=int, default=5000)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def main():
    args = parse_args()

    if args.stage == "stage2":
        psum = args.p_hard + args.p_semi + args.p_easy
        if abs(psum - 1.0) > 1e-6:
            raise ValueError(f"p_hard+p_semi+p_easy must sum to 1.0, got {psum}")
        if not args.sasrec_model_path:
            raise ValueError("--sasrec_model_path is required for stage2")

    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"📥 Loading SASRec dataset: {args.sasrec_data_path}")
    with open(args.sasrec_data_path, "rb") as f:
        pkg = pickle.load(f)

    raw_data_list = pkg["data"]
    item2id = pkg["item2id"]
    id2item = pkg["id2item"]
    n_items = int(pkg["n_items"])
    print(f"✅ users={len(raw_data_list)} items(n_items)={n_items}")

    # metadata
    meta_mgr = MetaDataManager(args.raw_meta_dir, valid_gmap_ids=list(item2id.keys()))
    meta_mgr.load()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    # -------------------------
    # Stage1: positive-only
    # -------------------------
    if args.stage == "stage1":
        written = 0
        skipped = 0
        start = time.time()
        with open(args.output_jsonl, "w", encoding="utf-8", buffering=1024 * 1024) as fw:
            pbar = tqdm(total=len(raw_data_list), desc="Stage1Pos")
            for entry in raw_data_list:
                pbar.update(1)
                if args.max_samples and written >= args.max_samples:
                    break

                uid = str(entry.get("user_id"))
                seq = entry.get("sequence", [])
                if not isinstance(seq, (list, tuple)) or len(seq) < max(2, args.min_seq_len):
                    skipped += 1
                    continue

                gt_idx = int(seq[-1])
                hist = [int(x) for x in seq[:-1] if int(x) != 0]
                if not hist:
                    skipped += 1
                    continue

                # history text (last K)
                recent = hist[-args.max_hist_text:]
                hist_texts = []
                for idx in recent:
                    gid = id2item.get(int(idx))
                    if gid is not None:
                        hist_texts.append(meta_mgr.get_text(gid))
                if not hist_texts:
                    skipped += 1
                    continue

                gt_gmap = id2item.get(gt_idx)
                if gt_gmap is None:
                    skipped += 1
                    continue

                sample = {
                    "prompt": build_prompt(hist_texts, kind="prompt"),
                    "completion": meta_mgr.get_text(gt_gmap),
                    "meta": {
                        "user_id": uid,
                        "gt_idx": int(gt_idx),
                        "target_id": str(gt_gmap),
                    }
                }
                fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

            pbar.close()

        elapsed = time.time() - start
        print("========================================")
        print("✅ STAGE1 JSONL GENERATED (positive-only)")
        print("========================================")
        print(f"Output: {args.output_jsonl}")
        print(f"Written: {written}")
        print(f"Skipped: {skipped}")
        print(f"Total wall time: {elapsed:.2f}s | throughput: {written/max(1e-9, elapsed):.2f} lines/s")
        print("========================================")
        return

    # -------------------------
    # Stage2: CoIN triplets
    # -------------------------
    print("🔥 Building popularity from train interactions ...")
    freq = Counter()
    t0 = time.time()
    for entry in tqdm(raw_data_list, desc="PopCount"):
        seq = entry.get("sequence", [])
        if not isinstance(seq, (list, tuple)) or len(seq) < 3:
            continue
        for it in seq[:-2]:
            it = int(it)
            if it > 0:
                freq[it] += 1
    pop_sorted = [it for it, _ in freq.most_common(args.pop_top)]
    if len(pop_sorted) == 0:
        raise RuntimeError("Popularity pool is empty.")
    pop_items = np.array(pop_sorted, dtype=np.int32)
    print(f"✅ pop pool size={len(pop_items)} (top={args.pop_top}), build_time={time.time()-t0:.2f}s")

    # teacher
    class MArgs:
        def __init__(self):
            self.embed_dim = args.embed_dim
            self.max_len = args.max_len
            self.num_blocks = args.num_blocks
            self.num_heads = args.num_heads
            self.dropout = args.dropout
            self.device = args.device

    teacher = SASRec(n_items, MArgs()).to(args.device)
    teacher.eval()
    teacher = load_teacher_ckpt(args.sasrec_model_path, teacher, args.device)

    C = 1 + args.num_neg
    neg_cap_counter = Counter()
    bucket_counter = Counter()

    written = 0
    skipped = Counter()
    buffer_lines = []

    def flush(fw):
        nonlocal buffer_lines
        if buffer_lines:
            fw.write("".join(buffer_lines))
            buffer_lines = []

    infer_bs = int(args.infer_bs)
    X = np.zeros((infer_bs, args.max_len), dtype=np.int32)
    GT = np.zeros((infer_bs,), dtype=np.int32)
    UID = [None] * infer_bs
    FULLSETS = [None] * infer_bs
    HISTS = [None] * infer_bs
    bsz = 0

    print("🧱 Generating STAGE2 jsonl (BATCH candidates + predict_candidates) ...")
    start_time = time.time()

    with open(args.output_jsonl, "w", encoding="utf-8", buffering=1024 * 1024) as fw:
        with torch.no_grad():
            pbar = tqdm(total=len(raw_data_list), desc="Stage2Triplets")
            for entry in raw_data_list:
                pbar.update(1)
                if args.max_samples and written >= args.max_samples:
                    break

                uid = str(entry.get("user_id"))
                seq = entry.get("sequence", [])
                if not isinstance(seq, (list, tuple)) or len(seq) < max(2, args.min_seq_len):
                    skipped["short"] += 1
                    continue

                gt_idx = int(seq[-1])
                hist = [int(x) for x in seq[:-1] if int(x) != 0]
                if len(hist) == 0:
                    skipped["no_hist"] += 1
                    continue

                full_set = set(int(x) for x in seq if int(x) != 0)

                X[bsz] = pad_left_np(hist[-args.max_len:], args.max_len, pad=0)
                GT[bsz] = gt_idx
                UID[bsz] = uid
                FULLSETS[bsz] = full_set
                HISTS[bsz] = hist
                bsz += 1

                if bsz < infer_bs:
                    continue

                # build candidates
                cand = np.zeros((infer_bs, C), dtype=np.int32)
                for i in range(infer_bs):
                    cand[i] = np.array(
                        make_candidate_row(
                            gt_idx=int(GT[i]),
                            forbid_set=FULLSETS[i],
                            n_items=n_items,
                            pop_items=pop_items,
                            C=C,
                            oversample=args.oversample,
                            rng=rng
                        ),
                        dtype=np.int32
                    )

                # score
                x_t = torch.from_numpy(X).long().to(args.device, non_blocking=True)
                cand_t = torch.from_numpy(cand).long().to(args.device, non_blocking=True)

                use_amp = args.device.startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    scores = teacher.predict_candidates(x_t, cand_t)
                scores = scores.float().cpu().numpy()
                cand_cpu = cand

                # pick neg & write
                for i in range(infer_bs):
                    if args.max_samples and written >= args.max_samples:
                        break

                    uid_i = UID[i]
                    gt_idx_i = int(GT[i])
                    full_set_i = FULLSETS[i]
                    hist_i = HISTS[i]

                    row_ids = cand_cpu[i].tolist()
                    row_sc = scores[i].tolist()
                    gt_score = row_sc[0]

                    picked = pick_mixed_neg_from_scores(
                        gt_score=gt_score,
                        cand_ids=row_ids,
                        cand_scores=row_sc,
                        forbid_set=full_set_i,
                        neg_cap_counter=neg_cap_counter,
                        neg_cap=args.neg_cap,
                        rng=rng,
                        p_hard=args.p_hard,
                        p_semi=args.p_semi,
                        p_easy=args.p_easy,
                        semi_margin=args.semi_margin,
                    )
                    if picked is None:
                        skipped["no_neg"] += 1
                        continue

                    neg_idx_i, neg_score_i, mix_bucket, diff = picked
                    neg_cap_counter[int(neg_idx_i)] += 1
                    bucket_counter[mix_bucket] += 1

                    gt_gmap = id2item.get(gt_idx_i)
                    neg_gmap = id2item.get(int(neg_idx_i))
                    if gt_gmap is None or neg_gmap is None:
                        skipped["id_map"] += 1
                        continue

                    # history text
                    recent = hist_i[-args.max_hist_text:]
                    hist_texts = []
                    for idx in recent:
                        gid = id2item.get(int(idx))
                        if gid is not None:
                            hist_texts.append(meta_mgr.get_text(gid))
                    if not hist_texts:
                        skipped["no_hist_text"] += 1
                        continue

                    gt_text = meta_mgr.get_text(gt_gmap)
                    neg_text = meta_mgr.get_text(neg_gmap)

                    prompt = build_prompt(hist_texts, kind="prompt")
                    prompt_aug = build_prompt(hist_texts, kind="augment")

                    gap = float(gt_score - float(neg_score_i))
                    if mix_bucket == "hard":
                        hard_level = "hard++"
                    elif mix_bucket == "semi":
                        hard_level = "hard"
                    else:
                        hard_level = "medium"

                    sample = {
                        "prompt": prompt,
                        "prompt_augment": prompt_aug,
                        "completion": gt_text,
                        "negative_completion": neg_text,
                        "ips_weight": 1.0,

                        "teacher_score_gt": float(gt_score),
                        "teacher_score_neg": float(neg_score_i),
                        "teacher_gap": float(gap),          # gt - neg
                        "teacher_diff": float(diff),        # neg - gt
                        "hard_level": hard_level,
                        "mix_bucket": mix_bucket,

                        "meta": {
                            "user_id": uid_i,
                            "target_id": str(gt_gmap),
                            "hard_neg_id": str(neg_gmap),
                            "gt_idx": int(gt_idx_i),
                            "neg_idx": int(neg_idx_i),
                            "candidates": int(C),
                            "neg_sampling": "pop+uniform",
                            "oversample": int(args.oversample),
                            "pop_top": int(args.pop_top),
                            "neg_cap": int(args.neg_cap),
                            "p_hard": float(args.p_hard),
                            "p_semi": float(args.p_semi),
                            "p_easy": float(args.p_easy),
                            "semi_margin": float(args.semi_margin),
                        }
                    }

                    buffer_lines.append(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    if len(buffer_lines) >= args.write_buffer:
                        flush(fw)

                bsz = 0  # reset batch

            flush(fw)
            pbar.close()

    elapsed = time.time() - start_time
    total_b = sum(bucket_counter.values())

    print("========================================")
    print("✅ STAGE2 JSONL GENERATED (CoIN triplets)")
    print("========================================")
    print(f"Output: {args.output_jsonl}")
    print(f"Written: {written}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Unique neg used: {len(neg_cap_counter)} (neg_cap={args.neg_cap})")
    if total_b > 0:
        print("----------------------------------------")
        print("[mix_bucket distribution]")
        for k in ["hard", "semi", "easy", "any"]:
            if bucket_counter.get(k, 0) > 0:
                print(f"  {k:>4s}: {bucket_counter[k]} ({bucket_counter[k]/total_b*100:.2f}%)")
        print("----------------------------------------")
    print(f"Total wall time: {elapsed:.2f}s | throughput: {written/max(1e-9, elapsed):.2f} samples/s")
    print("========================================")


if __name__ == "__main__":
    main()
