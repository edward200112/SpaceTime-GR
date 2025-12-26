import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, Optional, Tuple


def safe_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
    return None


def extract_uid_gt(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    """
    uid: prefer meta.user_id
    gt : prefer meta.gt_idx / meta.target_id / meta.target_gmap_id, fallback completion
    """
    meta = obj.get("meta") or {}
    # uid
    uid = (
        meta.get("user_id")
        or meta.get("uid")
        or obj.get("uid")
        or obj.get("user_id")
    )
    if uid is not None:
        uid = str(uid)

    # gt key candidates (prefer idx -> stable)
    gt = (
        meta.get("gt_idx")
        or meta.get("target_idx")
        or meta.get("target_id")
        or meta.get("target_gmap_id")
        or obj.get("gt_idx")
        or obj.get("target_id")
        or obj.get("target_gmap_id")
    )

    # try cast to int if possible
    gt_int = safe_int(gt)
    if gt_int is not None:
        gt = gt_int

    # fallback: completion text (not ideal for item-id stats, but keeps script robust)
    if gt is None:
        gt = obj.get("completion")

    return uid, gt


def bucket_count(c: int) -> str:
    # 频次桶：你可以按需改
    if c <= 1:
        return "1"
    if c == 2:
        return "2"
    if 3 <= c <= 5:
        return "3-5"
    if 6 <= c <= 10:
        return "6-10"
    if 11 <= c <= 20:
        return "11-20"
    if 21 <= c <= 50:
        return "21-50"
    if 51 <= c <= 100:
        return "51-100"
    if 101 <= c <= 200:
        return "101-200"
    if 201 <= c <= 500:
        return "201-500"
    if 501 <= c <= 1000:
        return "501-1000"
    return ">1000"


def percentile_from_sorted(arr, p: float) -> float:
    """
    arr: sorted list of numbers
    p: 0..1
    """
    if not arr:
        return 0.0
    if p <= 0:
        return float(arr[0])
    if p >= 1:
        return float(arr[-1])
    k = (len(arr) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(arr[int(k)])
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return float(d0 + d1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--max_lines", type=int, default=0, help="0 means all")
    ap.add_argument("--topn_gt", type=int, default=50, help="print topN GT by frequency")
    ap.add_argument("--hash_uid", action="store_true",
                    help="store uid as 64-bit hash to reduce memory (approx, tiny collision risk)")
    args = ap.parse_args()

    total = 0
    bad = 0
    missing_uid = 0
    missing_gt = 0

    # uid uniques
    uid_set = set()

    # gt frequency
    gt_counter = Counter()

    def uid_key(u: str):
        if not args.hash_uid:
            return u
        # 64-bit hash (collision extremely unlikely in practice)
        return hash(u) & ((1 << 64) - 1)

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines and total >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            uid, gt = extract_uid_gt(obj)
            if uid is None:
                missing_uid += 1
            else:
                uid_set.add(uid_key(uid))

            if gt is None or gt == "":
                missing_gt += 1
            else:
                gt_counter[gt] += 1

    # ---- report ----
    print("========================================")
    print("UID / GT STATS REPORT")
    print("========================================")
    print(f"File:                 {args.jsonl}")
    print(f"Lines scanned:         {total}")
    print(f"Bad JSON lines:        {bad}")
    print("----------------------------------------")
    print(f"Unique UIDs:           {len(uid_set)}")
    print(f"Missing UID:           {missing_uid} ({(missing_uid/max(1,total))*100:.4f}%)")
    print("----------------------------------------")
    print(f"Unique GT:             {len(gt_counter)}")
    print(f"Missing GT:            {missing_gt} ({(missing_gt/max(1,total))*100:.4f}%)")

    if len(gt_counter) > 0:
        freqs = sorted(gt_counter.values())
        mean_f = sum(freqs) / len(freqs)
        p50 = percentile_from_sorted(freqs, 0.50)
        p90 = percentile_from_sorted(freqs, 0.90)
        p95 = percentile_from_sorted(freqs, 0.95)
        p99 = percentile_from_sorted(freqs, 0.99)
        mx = freqs[-1]

        print("----------------------------------------")
        print("[GT frequency summary]")
        print(f"mean: {mean_f:.4f} | p50: {p50:.0f} | p90: {p90:.0f} | p95: {p95:.0f} | p99: {p99:.0f} | max: {mx}")

        # histogram buckets
        bucket_counter = Counter()
        for c in freqs:
            bucket_counter[bucket_count(int(c))] += 1

        bucket_order = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "501-1000", ">1000"]
        print("----------------------------------------")
        print("[GT frequency histogram]  (#GT items in each freq bucket)")
        for b in bucket_order:
            if b in bucket_counter:
                print(f"{b:>8}: {bucket_counter[b]}  ({bucket_counter[b]/len(freqs)*100:.2f}%)")

        print("----------------------------------------")
        print(f"[Top {args.topn_gt} most frequent GT]")
        for i, (gt, c) in enumerate(gt_counter.most_common(args.topn_gt), 1):
            print(f"#{i:02d}  gt={gt}  count={c}  ratio={c/max(1,total)*100:.4f}%")

    print("========================================")


if __name__ == "__main__":
    main()
