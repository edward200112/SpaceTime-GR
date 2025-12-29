# HardMiningSFT/debug_assistant_start.py
import os
import argparse
import random

from datasets import load_dataset
from transformers import AutoTokenizer

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"

PROMPT_RULE = "只输出一个地点名(类别)，不要解释"

def add_rule_to_prompt(p: str) -> str:
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    return p + "\n" + PROMPT_RULE

def decode_span(tok, ids, l, r):
    l = max(0, l); r = min(len(ids), r)
    return tok.decode(ids[l:r], skip_special_tokens=False)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    n = min(args.n, len(ds))
    idxs = random.sample(range(len(ds)), n)

    print(f"[INFO] sample idxs={idxs}")
    print("[INFO] 你要看的重点：assistant_start 是否正好落在 assistant 内容开始处。\n")

    for k, i in enumerate(idxs, 1):
        ex = ds[i]
        prompt = add_rule_to_prompt(ex["prompt"])
        completion = ex["completion"]

        prefix_text = USER_PREFIX_TEMPLATE.format(prompt=prompt)
        full_text = prefix_text + completion + SUFFIX

        # 复现你 stage1 的做法：
        full_tok_default = tok(full_text, truncation=True, max_length=args.max_length, add_special_tokens=True)
        prefix_tok_no_special = tok(prefix_text, truncation=True, max_length=args.max_length, add_special_tokens=False)

        assistant_start_stage1 = len(prefix_tok_no_special["input_ids"])
        full_ids = full_tok_default["input_ids"]

        # 对照：两边都 add_special_tokens=False（更一致的写法）
        full_tok_nospecial = tok(full_text, truncation=True, max_length=args.max_length, add_special_tokens=False)
        prefix_tok_nospecial = tok(prefix_text, truncation=True, max_length=args.max_length, add_special_tokens=False)
        assistant_start_consistent = len(prefix_tok_nospecial["input_ids"])
        full_ids_nospecial = full_tok_nospecial["input_ids"]

        print("=" * 80)
        print(f"[{k}] idx={i}")
        print(f"assistant_start_stage1_like = {assistant_start_stage1}")
        print(f"assistant_start_consistent  = {assistant_start_consistent}")
        print(f"delta(stage1-consistent)    = {assistant_start_stage1 - assistant_start_consistent}")
        print("-" * 80)

        # 打印断点附近 40 tokens 的 decode
        s = assistant_start_stage1
        print("[Stage1-like] prefix tail (s-40:s):")
        print(decode_span(tok, full_ids, s - 40, s))
        print("[Stage1-like] assistant head (s:s+80):")
        print(decode_span(tok, full_ids, s, s + 80))

        sc = assistant_start_consistent
        print("\n[Consistent no-special] prefix tail (sc-40:sc):")
        print(decode_span(tok, full_ids_nospecial, sc - 40, sc))
        print("[Consistent no-special] assistant head (sc:sc+80):")
        print(decode_span(tok, full_ids_nospecial, sc, sc + 80))

        print("\n[GT completion head (raw text)]:")
        print(completion[:200].replace("\n", "\\n"))

if __name__ == "__main__":
    main()
