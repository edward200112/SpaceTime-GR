import os
import json
import gzip
import argparse
import pickle
import random
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch

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
            print("⚠️ raw_meta_dir not found, fallback to POI_<id> text.")
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


def hard_level_from_gap(gap: float):
    if gap < 0:
        return "hard++"
    if gap <= 1.0:
        return "hard"
    if gap <= 3.0:
        return "medium"
    return "easy-hard"


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
        print("✅ Loaded teacher from state_dict dict.")
    else:
        model.load_state_dict(obj)
        print("✅ Loaded teacher weights.")
    return model


def sample_popular_items(pop_items, n, rng):
    # pop_items: 1D numpy array of item ids (int)
    idx = rng.integers(0, len(pop_items), size=n, endpoint=False)
    return pop_items[idx]


def make_candidate_row(gt_idx, forbid_set, n_items, pop_items, C, oversample, rng):
    """
    Build one candidate row: [gt] + (C-1) neg candidates
    - sources: popular + uniform
    - oversample to make filtering succeed
    """
    need = C - 1
    # oversample pool
    m = need * oversample

    # half popular, half uniform
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

    # if still not enough, fallback random
    while len(negs) < need:
        x = int(rng.integers(1, n_items + 1))
        if x != gt_idx and x not in forbid_set:
            negs.append(x)

    return [int(gt_idx)] + negs


def pick_boundary_neg_from_scores(gt_score, cand_ids, cand_scores, forbid_set, neg_cap_counter, neg_cap):
    """
    cand_ids/scores include gt at position 0.
    Return (neg_idx, neg_score)
    boundary rule:
      - prefer score_neg >= score_gt, but minimal (score_neg - score_gt)
      - else pick closest to score_gt (abs diff minimal)
    Also enforce neg_cap.
    """
    gt = float(gt_score)
    best = None
    best_key = None

    # First pass: beat GT minimally
    for cid, sc in zip(cand_ids[1:], cand_scores[1:]):
        cid = int(cid)
        sc = float(sc)
        if cid == 0 or cid in forbid_set:
            continue
        if neg_cap > 0 and neg_cap_counter[cid] >= neg_cap:
            continue
        diff = sc - gt
        if diff >= 0:
            key = (diff, -sc)
            if best_key is None or key < best_key:
                best_key = key
                best = (cid, sc)

    if best is not None:
        return best

    # Second pass: closest to GT
    best = None
    best_abs = None
    for cid, sc in zip(cand_ids[1:], cand_scores[1:]):
        cid = int(cid)
        sc = float(sc)
        if cid == 0 or cid in forbid_set:
            continue
        if neg_cap > 0 and neg_cap_counter[cid] >= neg_cap:
            continue
        a = abs(gt - sc)
        if best_abs is None or a < best_abs:
            best_abs = a
            best = (cid, sc)

    return best


# =========================
# Args
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec_data_path", type=str, required=True)
    ap.add_argument("--sasrec_model_path", type=str, required=True)
    ap.add_argument("--raw_meta_dir", type=str, default="")
    ap.add_argument("--output_jsonl", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--max_hist_text", type=int, default=5)

    # batching
    ap.add_argument("--infer_bs", type=int, default=1024)
    ap.add_argument("--write_buffer", type=int, default=20000)

    # candidates
    ap.add_argument("--num_neg", type=int, default=199, help="C-1 negatives per sample")
    ap.add_argument("--oversample", type=int, default=8, help="oversample factor for candidate pool")
    ap.add_argument("--pop_top", type=int, default=200000, help="top popular items used as pop pool")

    # control
    ap.add_argument("--neg_cap", type=int, default=5000, help="max uses per neg_idx; 0 disables")
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"📥 Loading SASRec dataset: {args.sasrec_data_path}")
    with open(args.sasrec_data_path, "rb") as f:
        pkg = pickle.load(f)

    raw_data_list = pkg["data"]
    item2id = pkg["item2id"]
    id2item = pkg["id2item"]
    n_items = int(pkg["n_items"])

    print(f"✅ users={len(raw_data_list)} items(n_items)={n_items}")

    # build popularity from TRAIN PART (seq[:-2]) to avoid leakage
    print("🔥 Building popularity from train interactions ...")
    freq = Counter()
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
    print(f"✅ pop pool size={len(pop_items)} (top={args.pop_top})")

    # metadata
    meta_mgr = MetaDataManager(args.raw_meta_dir, valid_gmap_ids=list(item2id.keys()))
    meta_mgr.load()

    # teacher (must match your training config)
    class MArgs:
        def __init__(self):
            self.embed_dim = 128
            self.max_len = args.max_len
            self.num_blocks = 2
            self.num_heads = 2
            self.dropout = 0.2
            self.device = args.device

    teacher = SASRec(n_items, MArgs()).to(args.device)
    teacher.eval()
    teacher = load_teacher_ckpt(args.sasrec_model_path, teacher, args.device)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    C = 1 + args.num_neg
    neg_cap_counter = Counter()

    written = 0
    skipped_short = 0
    skipped_no_hist = 0
    skipped_no_neg = 0

    buffer_lines = []

    def flush(fw):
        nonlocal buffer_lines
        if buffer_lines:
            fw.write("".join(buffer_lines))
            buffer_lines = []

    # batch buffers
    infer_bs = args.infer_bs
    X = np.zeros((infer_bs, args.max_len), dtype=np.int32)
    GT = np.zeros((infer_bs,), dtype=np.int32)
    UID = [None] * infer_bs
    FULLSETS = [None] * infer_bs
    HISTS = [None] * infer_bs
    bsz = 0

    print("🧱 Generating SFT jsonl (BATCH candidates + predict_candidates) ...")
    with open(args.output_jsonl, "w", encoding="utf-8", buffering=1024 * 1024) as fw:
        with torch.no_grad():
            pbar = tqdm(total=len(raw_data_list), desc="GenerateBatchCand")

            for entry in raw_data_list:
                pbar.update(1)
                if args.max_samples and written >= args.max_samples:
                    break

                uid = str(entry.get("user_id"))
                seq = entry.get("sequence", [])
                if not isinstance(seq, (list, tuple)) or len(seq) < 3:
                    skipped_short += 1
                    continue

                gt_idx = int(seq[-1])
                hist = [int(x) for x in seq[:-1] if int(x) != 0]
                if len(hist) == 0:
                    skipped_no_hist += 1
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

                # ===== build candidates batch on CPU =====
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

                # ===== teacher score candidates on GPU =====
                x_t = torch.from_numpy(X).long().to(args.device, non_blocking=True)
                cand_t = torch.from_numpy(cand).long().to(args.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                    scores = teacher.predict_candidates(x_t, cand_t)  # [B, C]
                scores = scores.float().cpu().numpy()
                cand_cpu = cand  # already cpu

                # ===== pick boundary neg per row =====
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

                    picked = pick_boundary_neg_from_scores(
                        gt_score=gt_score,
                        cand_ids=row_ids,
                        cand_scores=row_sc,
                        forbid_set=full_set_i,
                        neg_cap_counter=neg_cap_counter,
                        neg_cap=args.neg_cap
                    )
                    if picked is None:
                        skipped_no_neg += 1
                        continue
                    neg_idx_i, neg_score_i = picked
                    neg_cap_counter[int(neg_idx_i)] += 1

                    gt_gmap = id2item.get(gt_idx_i)
                    neg_gmap = id2item.get(int(neg_idx_i))
                    if gt_gmap is None or neg_gmap is None:
                        skipped_no_neg += 1
                        continue

                    # hist text
                    recent = hist_i[-args.max_hist_text:]
                    hist_texts = []
                    for idx in recent:
                        gid = id2item.get(int(idx))
                        if gid is not None:
                            hist_texts.append(meta_mgr.get_text(gid))
                    if not hist_texts:
                        skipped_no_hist += 1
                        continue

                    gt_text = meta_mgr.get_text(gt_gmap)
                    neg_text = meta_mgr.get_text(neg_gmap)

                    prompt = build_prompt(hist_texts, kind="prompt")
                    prompt_aug = build_prompt(hist_texts, kind="augment")

                    gap = float(gt_score - float(neg_score_i))
                    sample = {
                        "prompt": prompt,
                        "prompt_augment": prompt_aug,
                        "completion": gt_text,
                        "negative_completion": neg_text,
                        "ips_weight": 1.0,
                        "teacher_score_gt": float(gt_score),
                        "teacher_score_neg": float(neg_score_i),
                        "teacher_gap": gap,
                        "hard_level": hard_level_from_gap(gap),
                        "meta": {
                            "user_id": uid_i,
                            "target_id": str(gt_gmap),
                            "hard_neg_id": str(neg_gmap),
                            "gt_idx": int(gt_idx_i),
                            "neg_idx": int(neg_idx_i),
                            "candidates": int(C),
                            "neg_sampling": "pop+uniform",
                            "oversample": int(args.oversample),
                        }
                    }

                    buffer_lines.append(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    if len(buffer_lines) >= args.write_buffer:
                        flush(fw)

                bsz = 0

            # flush last
            flush(fw)
            pbar.close()

    print("========================================")
    print("✅ SFT JSONL GENERATED (batch candidates)")
    print("========================================")
    print(f"Output: {args.output_jsonl}")
    print(f"Written: {written}")
    print(f"Skipped short: {skipped_short}")
    print(f"Skipped no hist: {skipped_no_hist}")
    print(f"Skipped no neg: {skipped_no_neg}")
    if args.neg_cap > 0:
        print(f"Neg cap: {args.neg_cap}")
        print(f"Unique neg used: {len(neg_cap_counter)}")
    print("========================================")


if __name__ == "__main__":
    main()
