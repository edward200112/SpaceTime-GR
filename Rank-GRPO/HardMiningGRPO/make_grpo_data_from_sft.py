import re, json, argparse
from tqdm import tqdm

NAMECAT_RE = re.compile(r"([^\n\r\(\)]{1,200})\s*\(\s*([^\n\r\(\)]{1,120})\s*\)")

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def extract_namecats(text: str):
    """
    从 prompt 里抽出形如 'Name (Category)' 的片段
    """
    out = []
    for m in NAMECAT_RE.finditer(text or ""):
        name = norm(m.group(1))
        cat  = norm(m.group(2))
        if name and cat:
            out.append(f"{name} ({cat})")
    return out

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--namecat2item_unique", required=True)
    ap.add_argument("--max_hist", type=int, default=50)  # 跟你 prep 的 max_len 对齐
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    namecat2item = load_json(args.namecat2item_unique)

    kept = 0
    dropped_gt_unmapped = 0
    dropped_no_hist = 0

    fin = open(args.in_jsonl, "r", encoding="utf-8")
    fout = open(args.out_jsonl, "w", encoding="utf-8")

    for i, line in enumerate(tqdm(fin, desc="build grpo data")):
        if args.limit > 0 and i >= args.limit:
            break
        obj = json.loads(line)
        prompt = obj.get("prompt", "")
        gt_text = norm(obj.get("completion", ""))

        gt_id = namecat2item.get(gt_text)
        if gt_id is None:
            dropped_gt_unmapped += 1
            continue

        # history
        hist_namecats = extract_namecats(prompt)
        hist_ids = []
        for nc in hist_namecats:
            iid = namecat2item.get(nc)
            if iid is not None:
                hist_ids.append(int(iid))

        if len(hist_ids) == 0:
            dropped_no_hist += 1
            continue

        hist_ids = hist_ids[-args.max_hist:]

        out = {
            "prompt": prompt,
            "target_namecat": gt_text,
            "target_item_id": int(gt_id),
            "history_item_ids": hist_ids,
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        kept += 1

    fin.close()
    fout.close()

    print("========== DONE ==========")
    print(f"kept={kept}")
    print(f"dropped_gt_unmapped={dropped_gt_unmapped}")
    print(f"dropped_no_hist={dropped_no_hist}")

if __name__ == "__main__":
    main()
