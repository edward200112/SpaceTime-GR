import argparse, json
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--max_lines", type=int, default=0)
    args = ap.parse_args()

    cnt = Counter()
    total = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines and total >= args.max_lines:
                break
            obj = json.loads(line)
            neg_idx = obj["meta"]["neg_idx"]
            cnt[int(neg_idx)] += 1
            total += 1

    print("========================================")
    print("NEG REPETITION REPORT")
    print("========================================")
    print(f"lines: {total}")
    for i, (k, c) in enumerate(cnt.most_common(args.topn), 1):
        print(f"#{i:02d} neg_idx={k}  count={c}  ratio={c/total:.4%}")
    top1 = cnt.most_common(1)[0][1] if cnt else 0
    top10 = sum(c for _, c in cnt.most_common(10))
    top100 = sum(c for _, c in cnt.most_common(100))
    print("----------------------------------------")
    print(f"top1  coverage:  {top1/total:.4%}")
    print(f"top10 coverage:  {top10/total:.4%}")
    print(f"top100 coverage: {top100/total:.4%}")
    print("========================================")

if __name__ == "__main__":
    main()
