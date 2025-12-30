import argparse
import json
import re
from collections import Counter
from typing import Tuple, Optional

NAMECAT_RE = re.compile(r"^\s*(.*?)\s*\(\s*(.*?)\s*\)\s*$")

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # unicode 标点归一化（你现在 true_miss 很多就是这类）
    s = (
        s.replace("’", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
    )
    # 合并多空格
    s = " ".join(s.split())
    return s

def parse_namecat(text: str) -> Tuple[str, str]:
    """
    解析 "Name (Category)".
    若失败：name=整串, cat=""
    """
    t = norm_text(text)
    m = NAMECAT_RE.match(t)
    if not m:
        return t, ""
    name = norm_text(m.group(1))
    cat = norm_text(m.group(2))
    return name, cat

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str, n: int):
    import json
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n > 0 and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--namecat2item_unique", required=True)
    ap.add_argument("--namecat2item_disamb", required=True)
    ap.add_argument("--name2item_disamb", required=False, default=None, help="optional name-only fallback map")
    ap.add_argument("--field", default="completion")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--show_topk", type=int, default=20)
    args = ap.parse_args()

    namecat_unique = load_json(args.namecat2item_unique)       # key -> [id] (一般 len=1)
    namecat_disamb = load_json(args.namecat2item_disamb)       # key -> [ids] (len>=1)
    name2item_disamb = load_json(args.name2item_disamb) if args.name2item_disamb else None  # name -> [ids]

    total = 0

    hit_unique = 0
    hit_ambig_namecat = 0

    hit_name_only_unique = 0
    hit_name_only_ambig = 0

    true_miss = 0

    ambig_namecat_counter = Counter()
    ambig_nameonly_counter = Counter()
    true_miss_counter = Counter()

    for ex in iter_jsonl(args.sft_jsonl, args.n):
        total += 1
        text = ex.get(args.field, "")
        # 有的 completion 可能多行；只取第一行更稳
        if isinstance(text, str):
            text = text.splitlines()[0].strip()
        else:
            text = str(text)

        name, cat = parse_namecat(text)
        key = f"{name} ({cat})" if cat else ""

        # 1) namecat unique
        if key and key in namecat_unique:
            hit_unique += 1
            continue

        # 2) namecat ambiguous
        if key and key in namecat_disamb:
            hit_ambig_namecat += 1
            ambig_namecat_counter[key] += 1
            continue

        # 3) name-only fallback
        if name2item_disamb is not None and name in name2item_disamb:
            ids = name2item_disamb[name]
            if isinstance(ids, list) and len(ids) == 1:
                hit_name_only_unique += 1
            else:
                hit_name_only_ambig += 1
                ambig_nameonly_counter[name] += 1
            continue

        # 4) true miss
        true_miss += 1
        true_miss_counter[text] += 1

    print("========== HIT REPORT (v3: name-only fallback) ==========")
    print(f"total={total}")

    def ratio(x): 
        return 0.0 if total == 0 else x / total

    print(f"hit_unique(namecat)       = {hit_unique} ({ratio(hit_unique):.4f})")
    print(f"hit_ambiguous(namecat)    = {hit_ambig_namecat} ({ratio(hit_ambig_namecat):.4f})")

    if name2item_disamb is not None:
        print(f"hit_name_only_unique      = {hit_name_only_unique} ({ratio(hit_name_only_unique):.4f})")
        print(f"hit_name_only_ambiguous   = {hit_name_only_ambig} ({ratio(hit_name_only_ambig):.4f})")

    print(f"true_miss                 = {true_miss} ({ratio(true_miss):.4f})")
    print("--------------------------------")

    topk = int(args.show_topk)

    if hit_ambig_namecat > 0:
        print(f"Top ambiguous namecat keys (top {topk}):")
        for i, (k, c) in enumerate(ambig_namecat_counter.most_common(topk), 1):
            print(f"{i:02d}. {c}/{total}  {k}")
        print("--------------------------------")

    if name2item_disamb is not None and hit_name_only_ambig > 0:
        print(f"Top ambiguous name-only keys (top {topk}):")
        for i, (k, c) in enumerate(ambig_nameonly_counter.most_common(topk), 1):
            print(f"{i:02d}. {c}/{total}  {k}")
        print("--------------------------------")

    if true_miss > 0:
        print(f"Top true miss (top {topk}):")
        for i, (k, c) in enumerate(true_miss_counter.most_common(topk), 1):
            print(f"{i:02d}. {c}/{total}  {k}")

if __name__ == "__main__":
    main()
