import os
import json
import math
import time
import argparse
import random
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch

# 只有在做 spot verify 时才需要 SASRec
try:
    from SASRec import SASRec
except Exception:
    SASRec = None


# -------------------------
# helpers
# -------------------------
def safe_json_loads(line: str):
    try:
        return json.loads(line)
    except Exception as e:
        return None


def pct(x, total):
    return 100.0 * x / max(1, total)


def quantiles(arr, qs=(0.5, 0.9, 0.95, 0.99)):
    if len(arr) == 0:
        return {f"p{int(q*100)}": None for q in qs}
    a = np.array(arr, dtype=np.float64)
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(np.quantile(a, q))
    return out


def pad_left_np(seq, max_len, pad=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return np.array(seq[-max_len:], dtype=np.int32)
    out = np.full((max_len,), pad, dtype=np.int32)
    out[-len(seq):] = np.array(seq, dtype=np.int32)
    return out


def load_sasrec_checkpoint(path, model, device):
    obj = None
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        kind = "full .pt checkpoint (model field)"
    elif isinstance(obj, dict):
        model.load_state_dict(obj)
        kind = "state_dict dict"
    else:
        model.load_state_dict(obj)
        kind = "raw state_dict"
    return kind


def hard_level_expected(gap: float):
    if gap < 0:
        return "hard++"
    if gap <= 1.0:
        return "hard"
    if gap <= 3.0:
        return "medium"
    return "easy-hard"


def reservoir_sample(reservoir, item, k, n_seen):
    # standard reservoir sampling
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = random.randint(1, n_seen)
    if j <= k:
        reservoir[j - 1] = item


# -------------------------
# main checks
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)

    # fast scan controls
    ap.add_argument("--max_lines", type=int, default=0, help="0 = scan all")
    ap.add_argument("--print_samples", type=int, default=3, help="print N random samples")
    ap.add_argument("--reservoir_k", type=int, default=200, help="keep K samples for stats/spot verify")
    ap.add_argument("--dup_check_first", type=int, default=200000, help="check duplicate prompts for first N lines (memory heavy if too large)")
    ap.add_argument("--seed", type=int, default=42)

    # spot verify (optional)
    ap.add_argument("--spot_verify", action="store_true")
    ap.add_argument("--sasrec_pkl", type=str, default="")
    ap.add_argument("--teacher_ckpt", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--verify_n", type=int, default=64, help="spot verify N samples")
    ap.add_argument("--score_tol", type=float, default=1e-2, help="teacher score compare tolerance")

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.jsonl), f"jsonl not found: {args.jsonl}"

    required_keys = [
        "prompt", "prompt_augment", "completion", "negative_completion",
        "ips_weight", "teacher_score_gt", "teacher_score_neg",
        "teacher_gap", "hard_level", "meta"
    ]
    required_meta = ["user_id", "gt_idx", "neg_idx"]

    # counters
    total = 0
    bad_json = 0
    missing_key = Counter()
    missing_meta = Counter()

    empty_prompt = 0
    empty_completion = 0
    empty_neg = 0
    same_comp_neg = 0

    hard_level_cnt = Counter()
    gap_list = []
    gt_score_list = []
    neg_score_list = []
    gap_mismatch = 0
    hardlevel_mismatch = 0

    # duplicates (prompt hash)
    dup_total = 0
    seen_prompts = set()
    dup_cap = max(0, int(args.dup_check_first))

    # reservoir for printing and spot verify
    reservoir = []
    n_seen_valid = 0

    t0 = time.time()
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines and total >= args.max_lines:
                break
            total += 1

            obj = safe_json_loads(line)
            if obj is None:
                bad_json += 1
                continue

            # required keys
            ok = True
            for k in required_keys:
                if k not in obj:
                    missing_key[k] += 1
                    ok = False
            if not ok:
                continue

            # meta check
            meta = obj.get("meta", {})
            for mk in required_meta:
                if mk not in meta:
                    missing_meta[mk] += 1

            prompt = obj.get("prompt", "")
            comp = obj.get("completion", "")
            neg = obj.get("negative_completion", "")

            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                empty_prompt += 1
            if not isinstance(comp, str) or len(comp.strip()) == 0:
                empty_completion += 1
            if not isinstance(neg, str) or len(neg.strip()) == 0:
                empty_neg += 1
            if isinstance(comp, str) and isinstance(neg, str) and comp.strip() == neg.strip():
                same_comp_neg += 1

            # duplicates (only for first N lines)
            if dup_cap and total <= dup_cap and isinstance(prompt, str):
                h = hash(prompt)
                if h in seen_prompts:
                    dup_total += 1
                else:
                    seen_prompts.add(h)

            # scores / gap
            try:
                g = float(obj["teacher_gap"])
                sgt = float(obj["teacher_score_gt"])
                sneg = float(obj["teacher_score_neg"])
                gap_list.append(g)
                gt_score_list.append(sgt)
                neg_score_list.append(sneg)

                # gap consistency: gap == gt - neg
                if abs((sgt - sneg) - g) > 1e-4:
                    gap_mismatch += 1
            except Exception:
                pass

            hl = obj.get("hard_level", None)
            if hl is not None:
                hard_level_cnt[str(hl)] += 1
                try:
                    exp = hard_level_expected(float(obj["teacher_gap"]))
                    if str(hl) != exp:
                        hardlevel_mismatch += 1
                except Exception:
                    pass

            # reservoir sample
            n_seen_valid += 1
            reservoir_sample(reservoir, obj, args.reservoir_k, n_seen_valid)

    t1 = time.time()

    print("========================================")
    print("📌 SFT JSONL FAST HEALTH REPORT")
    print("========================================")
    print(f"File:            {args.jsonl}")
    print(f"Lines scanned:   {total}")
    print(f"Bad JSON lines:  {bad_json} ({pct(bad_json,total):.4f}%)")
    print(f"Scan time:       {t1 - t0:.2f}s  (throughput ~ {total / max(1e-9,(t1-t0)):.1f} lines/s)")
    print("----------------------------------------")

    if missing_key:
        print("[Missing top-level keys] (count)")
        for k, c in missing_key.most_common():
            print(f"  {k:24s}: {c}")
        print("----------------------------------------")
    else:
        print("✅ No missing top-level keys detected (for valid JSON lines).")
        print("----------------------------------------")

    if missing_meta:
        print("[Missing meta keys] (count)")
        for k, c in missing_meta.most_common():
            print(f"  meta.{k:18s}: {c}")
        print("----------------------------------------")
    else:
        print("✅ No missing meta keys detected (for samples that have meta).")
        print("----------------------------------------")

    print(f"Empty prompt:        {empty_prompt} ({pct(empty_prompt,total):.4f}%)")
    print(f"Empty completion:    {empty_completion} ({pct(empty_completion,total):.4f}%)")
    print(f"Empty neg_completion:{empty_neg} ({pct(empty_neg,total):.4f}%)")
    print(f"completion==neg:     {same_comp_neg} ({pct(same_comp_neg,total):.4f}%)")
    print("----------------------------------------")

    if dup_cap:
        print(f"Duplicate prompt (first {dup_cap} lines): {dup_total} ({pct(dup_total, min(total,dup_cap)):.4f}%)")
        print("----------------------------------------")

    if hard_level_cnt:
        print("[hard_level distribution]")
        tot_hl = sum(hard_level_cnt.values())
        for k, c in hard_level_cnt.most_common():
            print(f"  {k:10s}: {c} ({pct(c, tot_hl):.2f}%)")
        print(f"hardlevel mismatch vs gap-rule: {hardlevel_mismatch}")
        print("----------------------------------------")

    if len(gap_list) > 0:
        print("[teacher_gap stats]")
        gq = quantiles(gap_list, qs=(0.5, 0.9, 0.95, 0.99))
        print(f"  mean={np.mean(gap_list):.4f}, std={np.std(gap_list):.4f}, min={np.min(gap_list):.4f}, max={np.max(gap_list):.4f}")
        print(f"  p50={gq['p50']:.4f}, p90={gq['p90']:.4f}, p95={gq['p95']:.4f}, p99={gq['p99']:.4f}")
        print(f"gap mismatch (gt-neg != gap): {gap_mismatch}")
        print("----------------------------------------")

    # print random samples
    if args.print_samples > 0 and len(reservoir) > 0:
        print("========================================")
        print(f"🧾 RANDOM SAMPLES (n={args.print_samples})")
        print("========================================")
        for i in range(min(args.print_samples, len(reservoir))):
            ex = random.choice(reservoir)
            meta = ex.get("meta", {})
            print(f"\n--- SAMPLE {i+1} ---")
            print("uid:", meta.get("user_id"))
            print("gt_idx:", meta.get("gt_idx"), "neg_idx:", meta.get("neg_idx"))
            print("hard_level:", ex.get("hard_level"), "gap:", ex.get("teacher_gap"))
            print("prompt:", ex.get("prompt", "")[:200].replace("\n", "\\n"), "...")
            print("completion:", ex.get("completion", ""))
            print("neg_completion:", ex.get("negative_completion", ""))

    # -------------------------
    # spot verify
    # -------------------------
    if args.spot_verify:
        print("\n========================================")
        print("🔎 SPOT VERIFY (teacher + sasrec pkl)")
        print("========================================")

        if SASRec is None:
            raise RuntimeError("Cannot import SASRec. Put this script where SASRec.py is importable.")
        if not args.sasrec_pkl or not os.path.exists(args.sasrec_pkl):
            raise RuntimeError("spot_verify requires --sasrec_pkl path")
        if not args.teacher_ckpt or not os.path.exists(args.teacher_ckpt):
            raise RuntimeError("spot_verify requires --teacher_ckpt path")

        # pick verify samples
        verify_n = min(args.verify_n, len(reservoir))
        verify_samples = random.sample(reservoir, verify_n)

        need_uids = set()
        for ex in verify_samples:
            uid = str(ex.get("meta", {}).get("user_id"))
            need_uids.add(uid)

        print(f"Will verify N={verify_n} samples, unique uids={len(need_uids)}")
        print(f"Loading pkl: {args.sasrec_pkl}")
        with open(args.sasrec_pkl, "rb") as f:
            pkg = pickle.load(f)
        data_list = pkg["data"]
        n_items = int(pkg["n_items"])

        # fetch sequences only for needed uids (1 pass over data_list)
        uid2seq = {}
        for entry in data_list:
            uid = str(entry.get("user_id"))
            if uid in need_uids:
                seq = entry.get("sequence", [])
                uid2seq[uid] = [int(x) for x in seq if int(x) != 0]
                if len(uid2seq) >= len(need_uids):
                    break

        missing_u = [u for u in need_uids if u not in uid2seq]
        if missing_u:
            print(f"⚠️ Missing sequences in pkl for {len(missing_u)} uids (will skip those).")

        # init teacher (must match)
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
        kind = load_sasrec_checkpoint(args.teacher_ckpt, teacher, args.device)
        print(f"✅ Loaded teacher ckpt ({kind})")

        # run verification
        ok_cnt = 0
        bad_hist_leak = 0
        bad_gt_last = 0
        bad_score = 0
        skipped = 0

        with torch.no_grad():
            for ex in verify_samples:
                meta = ex.get("meta", {})
                uid = str(meta.get("user_id"))
                gt_idx = int(meta.get("gt_idx", -1))
                neg_idx = int(meta.get("neg_idx", -1))
                if uid not in uid2seq:
                    skipped += 1
                    continue

                seq = uid2seq[uid]
                if len(seq) < 3:
                    skipped += 1
                    continue

                # our generator uses gt = last item of seq
                if gt_idx != int(seq[-1]):
                    bad_gt_last += 1

                full_set = set(seq)
                if neg_idx in full_set:
                    bad_hist_leak += 1

                # history = seq[:-1]
                hist = seq[:-1]
                x = pad_left_np(hist[-args.max_len:], args.max_len, pad=0)
                x_t = torch.from_numpy(x).long().unsqueeze(0).to(args.device)

                cand = torch.tensor([[gt_idx, neg_idx]], dtype=torch.long, device=args.device)
                scores = teacher.predict_candidates(x_t, cand).float().cpu().numpy()[0]
                sgt = float(scores[0])
                sneg = float(scores[1])

                # compare with saved score
                try:
                    saved_gt = float(ex["teacher_score_gt"])
                    saved_neg = float(ex["teacher_score_neg"])
                except Exception:
                    skipped += 1
                    continue

                if (abs(saved_gt - sgt) > args.score_tol) or (abs(saved_neg - sneg) > args.score_tol):
                    bad_score += 1

                ok_cnt += 1

        print("----------------------------------------")
        print(f"Verified samples (effective): {ok_cnt}")
        print(f"Skipped (missing uid/parse):  {skipped}")
        print(f"GT != last item in pkl:       {bad_gt_last}")
        print(f"NEG in user history set:      {bad_hist_leak}")
        print(f"Teacher score mismatch:       {bad_score} (tol={args.score_tol})")
        print("----------------------------------------")
        if ok_cnt > 0:
            print(f"Leak rate (NEG in history):   {bad_hist_leak / ok_cnt:.4%}")
            print(f"Score mismatch rate:          {bad_score / ok_cnt:.4%}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
