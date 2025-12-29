import json, argparse
from collections import Counter

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--namecat2item_unique", required=True)
    ap.add_argument("--field", default="completion")
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args()

    m = load_json(args.namecat2item_unique)

    total = 0
    hit = 0
    miss = 0
    top_miss = Counter()

    with open(args.sft_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if total >= args.n:
                break
            obj = json.loads(line)
            s = norm(obj.get(args.field, ""))
            total += 1
            if s in m:
                hit += 1
            else:
                miss += 1
                if s:
                    top_miss[s] += 1

    print(f"[OK] total={total} hit={hit} ({hit/total:.4f}) miss={miss} ({miss/total:.4f})")
    print("Top miss:")
    for i, (k,v) in enumerate(top_miss.most_common(20), 1):
        print(f"{i:02d}. {v}/{total} {k}")

if __name__ == "__main__":
    main()
