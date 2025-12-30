# HardMiningGRPO/verify_teacher_topk.py
import os
import sys
import json
import argparse
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import pickle
from tqdm import tqdm


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def load_n_items_from_pkl(pkl_path: str) -> int:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return int(obj["n_items"])


def to_int_safe(x):
    try:
        return int(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="phase2 jsonl with teacher_top_item_ids")
    ap.add_argument("--sasrec_pkl", required=True, help="to load n_items for range check")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max_lines", type=int, default=0, help="0 means all")
    ap.add_argument("--count_total", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--show_examples", type=int, default=3)
    args = ap.parse_args()

    n_items = load_n_items_from_pkl(args.sasrec_pkl)
    total = None
    if args.count_total:
        total = count_lines(args.jsonl)

    # stats
    n = 0
    missing_top = 0
    missing_tgt = 0
    bad_len = 0
    non_int = 0
    out_of_range = 0
    has_zero = 0
    dup_cnt = 0
    tgt_not_in_top = 0
    forced_rate_cnt = 0

    overlap_sum = 0.0
    overlap_nonzero = 0
    overlap_hist_counts = Counter()
    forced_examples = []
    bad_examples = []

    pbar = tqdm(total=total, desc="verify teacher_top_item_ids", dynamic_ncols=True)
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines > 0 and n >= args.max_lines:
                break
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            ex = json.loads(line)

            tgt = ex.get("target_item_id", None)
            if tgt is None:
                missing_tgt += 1
                if len(bad_examples) < args.show_examples:
                    bad_examples.append({"reason": "missing target_item_id", "ex": ex})
                pbar.update(1)
                continue
            tgt = int(tgt)

            top = ex.get("teacher_top_item_ids", None)
            if not isinstance(top, list):
                missing_top += 1
                if len(bad_examples) < args.show_examples:
                    bad_examples.append({"reason": "missing/invalid teacher_top_item_ids", "tgt": tgt, "ex": ex})
                pbar.update(1)
                continue

            # length check
            if len(top) != int(args.topk):
                bad_len += 1
                if len(bad_examples) < args.show_examples:
                    bad_examples.append({"reason": f"bad_len {len(top)} != {args.topk}", "tgt": tgt, "top_head": top[:10]})

            # int / range / zero / dup
            top_int = []
            bad_int_local = False
            for x in top:
                v = to_int_safe(x)
                if v is None:
                    non_int += 1
                    bad_int_local = True
                    continue
                top_int.append(v)
                if v == 0:
                    has_zero += 1
                if v < 1 or v > n_items:
                    out_of_range += 1

            if bad_int_local and len(bad_examples) < args.show_examples:
                bad_examples.append({"reason": "non_int in teacher_top_item_ids", "tgt": tgt, "top_head": top[:10]})

            if len(set(top_int)) != len(top_int):
                dup_cnt += 1

            # target included?
            if tgt not in set(top_int):
                tgt_not_in_top += 1
                if len(bad_examples) < args.show_examples:
                    bad_examples.append({"reason": "target not in teacher_top_item_ids", "tgt": tgt, "top_tail": top_int[-10:]})
            else:
                # forced? 近似判断：target 只出现在最后一位
                if (len(top_int) == len(top)) and (len(top_int) > 0):
                    if top_int[-1] == tgt and tgt not in set(top_int[:-1]):
                        forced_rate_cnt += 1
                        if len(forced_examples) < args.show_examples:
                            forced_examples.append({"tgt": tgt, "tail": top_int[-10:]})

            # overlap with history
            hist = ex.get("history_item_ids", [])
            if isinstance(hist, list):
                hist_set = set(int(x) for x in hist if to_int_safe(x) is not None and int(x) != 0)
                inter = len(hist_set.intersection(set(top_int)))
                overlap_hist_counts[inter] += 1
                overlap_sum += inter / max(1, int(args.topk))
                overlap_nonzero += 1

            n += 1
            pbar.update(1)

    pbar.close()

    def ratio(x, denom): return (x / denom) if denom else 0.0

    print("\n" + "=" * 90)
    print(f"[VERIFY] file={args.jsonl}")
    print(f"  checked_lines={n} (max_lines={args.max_lines if args.max_lines>0 else 'ALL'})")
    print(f"  n_items(from pkl)={n_items}  expected_topk={args.topk}")
    print("-" * 90)
    print(f"  missing target_item_id : {missing_tgt}  rate={ratio(missing_tgt, n):.4f}")
    print(f"  missing teacher_top... : {missing_top}  rate={ratio(missing_top, n):.4f}")
    print(f"  bad_len                : {bad_len}      rate={ratio(bad_len, n):.4f}")
    print(f"  non_int values         : {non_int}      (count of bad entries)")
    print(f"  out_of_range ids       : {out_of_range} (id<1 or id>n_items)")
    print(f"  contains zero          : {has_zero}     (padding leaked)")
    print(f"  duplicate lists        : {dup_cnt}      rate={ratio(dup_cnt, n):.4f}")
    print(f"  target_not_in_top      : {tgt_not_in_top} rate={ratio(tgt_not_in_top, n):.4f}")
    print(f"  forced_target_at_last  : {forced_rate_cnt} rate≈{ratio(forced_rate_cnt, n):.4f}")
    if overlap_nonzero:
        print(f"  avg_overlap_with_hist  : {overlap_sum/overlap_nonzero:.4f} (intersection/topk)")
        # show some overlap distribution head
        top_overlap = overlap_hist_counts.most_common(10)
        print(f"  overlap_hist_count top10 (intersections -> count): {top_overlap}")
    print("=" * 90)

    if bad_examples:
        print("\n[Bad examples]")
        for e in bad_examples:
            print(json.dumps(e, ensure_ascii=False)[:2000])

    if forced_examples:
        print("\n[Forced-target examples] (approx: tgt only at last pos)")
        for e in forced_examples:
            print(json.dumps(e, ensure_ascii=False))

    # hard fail conditions
    hard_fail = (missing_tgt > 0) or (missing_top > 0) or (out_of_range > 0)
    # 你可以视情况把 bad_len/dup/zero 也当 hard fail
    if hard_fail:
        print("\n❌ HARD FAIL: missing fields or out-of-range ids detected.", file=sys.stderr)
        sys.exit(2)

    print("\n✅ OK: no missing fields, no out-of-range ids.")
    sys.exit(0)


if __name__ == "__main__":
    main()
