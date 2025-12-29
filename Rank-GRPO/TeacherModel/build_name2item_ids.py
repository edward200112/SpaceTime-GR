import argparse, json, pickle
from collections import defaultdict

def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # 常见 unicode 标点归一化
    s = (s.replace("’", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-"))
    # 合并多空格
    s = " ".join(s.split())
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="sasrec_dataset.pkl")
    ap.add_argument("--gmap_id2namecat", required=True, help="gmap_id2namecat.json (from meta)")
    ap.add_argument("--out", required=True, help="output name2item_ids_disambiguation.json")
    ap.add_argument("--unique_only", action="store_true", help="only keep names that map to exactly 1 item_id")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        obj = pickle.load(f)
    item2id = obj["item2id"]  # gmap_id -> item_id

    with open(args.gmap_id2namecat, "r", encoding="utf-8") as f:
        gmap2namecat = json.load(f)  # gmap_id -> {"name":..., "cat":...} or "Name (Cat)"

    name2item = defaultdict(set)

    for gmap_id, v in gmap2namecat.items():
        if gmap_id not in item2id:
            continue
        item_id = int(item2id[gmap_id])

        # 兼容两种格式
        if isinstance(v, dict):
            name = v.get("name", "")
        else:
            # "Name (Cat)" -> Name
            name = str(v).rsplit("(", 1)[0].strip()

        key = norm(name)
        if key:
            name2item[key].add(item_id)

    # 输出
    out = {}
    amb = 0
    for name, ids in name2item.items():
        ids = sorted(list(ids))
        if args.unique_only and len(ids) != 1:
            amb += 1
            continue
        out[name] = ids

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print("========== DONE ==========")
    print(f"[INFO] total names: {len(name2item)}")
    print(f"[INFO] saved names: {len(out)}")
    if args.unique_only:
        print(f"[INFO] skipped ambiguous: {amb}")
    print(f"[OK] saved: {args.out}")

if __name__ == "__main__":
    main()
