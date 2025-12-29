# HardMiningSFT/stats_top_completions.py
import re
import argparse
from collections import Counter

from datasets import load_dataset

SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = SPECIAL_PAT.sub("", s)
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:，。；：")
    return s

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--n", type=int, default=0, help="0 表示全量；否则只统计前 n 条")
    return ap.parse_args()

def main():
    args = parse_args()
    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    if args.n and args.n > 0:
        ds = ds.select(range(min(args.n, len(ds))))

    c = Counter()
    for ex in ds:
        gt = norm_text(ex.get("completion", ""))
        if gt:
            c[gt] += 1

    total = sum(c.values())
    print(f"Total counted: {total}")
    print("=" * 80)
    for i, (k, v) in enumerate(c.most_common(args.topk), 1):
        ratio = v / total if total else 0.0
        show = k[:120] + ("..." if len(k) > 120 else "")
        print(f"{i:02d}. {v:8d}  ({ratio:.4%})  {show}")

if __name__ == "__main__":
    main()
