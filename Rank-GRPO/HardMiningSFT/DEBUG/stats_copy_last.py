# HardMiningSFT/stats_copy_last.py
import re
import argparse
from datasets import load_dataset

# 从 prompt 里抓最后一个 "Name (Category)"，你 prompt 里就是这种结构
PAT = re.compile(r"([^\n>]+?\([^)]+\))")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--n", type=int, default=200000, help="统计前 n 条，0 表示全量(2M会慢)")
    return ap.parse_args()

def main():
    args = parse_args()
    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    if args.n and args.n > 0:
        ds = ds.select(range(min(args.n, len(ds))))

    total = 0
    gt_eq_last = 0
    gt_in_prompt = 0
    no_match = 0

    for ex in ds:
        prompt = ex["prompt"]
        gt = ex["completion"].strip()

        ms = PAT.findall(prompt)
        if not ms:
            no_match += 1
            continue

        last = ms[-1].strip()
        total += 1

        if gt == last:
            gt_eq_last += 1
        if gt in prompt:
            gt_in_prompt += 1

    print(f"Counted: {total} (no_match={no_match})")
    if total > 0:
        print(f"GT == last_item_in_prompt: {gt_eq_last}/{total} ({gt_eq_last/total:.4%})")
        print(f"GT appears somewhere in prompt: {gt_in_prompt}/{total} ({gt_in_prompt/total:.4%})")

if __name__ == "__main__":
    main()
