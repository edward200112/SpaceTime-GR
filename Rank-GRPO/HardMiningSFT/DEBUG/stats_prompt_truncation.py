# HardMiningSFT/stats_prompt_truncation.py
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n", type=int, default=200000, help="抽样统计，0=全量(慢)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    if args.n and args.n > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    lens = []
    hit_max = 0

    for ex in ds:
        full = USER_PREFIX_TEMPLATE.format(prompt=ex["prompt"]) + ex["completion"] + SUFFIX
        ids = tok(full, add_special_tokens=False)["input_ids"]
        L = len(ids)
        lens.append(L)
        if L >= args.max_length:
            hit_max += 1

    arr = np.array(lens, dtype=np.int64)
    print(f"Counted: {len(arr)}")
    print(f"max_length={args.max_length}, hit_max(>=max): {hit_max}/{len(arr)} ({hit_max/len(arr):.4%})")
    for p in [50, 90, 95, 99]:
        print(f"P{p}: {int(np.percentile(arr, p))}")
    print(f"Mean: {arr.mean():.2f}, Max: {arr.max()}")

if __name__ == "__main__":
    main()
