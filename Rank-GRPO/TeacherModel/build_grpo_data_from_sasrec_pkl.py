import argparse, json, pickle, random
from tqdm import tqdm

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def default_prompt(history_namecats):
    lines = ["User visited places (older->newer):"]
    for i, nc in enumerate(history_namecats, 1):
        lines.append(f"{i}. {nc}")
    lines.append("")
    lines.append("Predict the next POI in EXACT format: Name (Category)")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="sasrec_dataset.pkl")
    ap.add_argument("--gmap_id2namecat", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_hist", type=int, default=50)
    ap.add_argument("--min_hist", type=int, default=5)
    ap.add_argument("--n_samples", type=int, default=800000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # load pkl
    with open(args.pkl, "rb") as f:
        pk = pickle.load(f)
    data = pk["data"]
    id2item = pk["id2item"]  # item_id -> gmap_id

    gmap2namecat = load_json(args.gmap_id2namecat)

    # build item_id -> namecat
    item2namecat = {}
    missing = 0
    for iid_str, gid in id2item.items():
        iid = int(iid_str) if isinstance(iid_str, str) else int(iid_str)
        nc = gmap2namecat.get(str(gid))
        if nc:
            item2namecat[iid] = norm(nc)
        else:
            missing += 1
    print(f"[INFO] item2namecat built: {len(item2namecat)}; missing_namecat_for_item={missing}")

    out = open(args.out_jsonl, "w", encoding="utf-8")

    kept = 0
    tried = 0

    # 随机打散用户，避免只集中某些用户
    random.shuffle(data)

    pbar = tqdm(total=args.n_samples, desc="build grpo samples")
    while kept < args.n_samples:
        for u in data:
            if kept >= args.n_samples:
                break
            seq = u["sequence"]  # list[item_id]
            if len(seq) < args.min_hist + 1:
                continue

            # 取最后一个作为 target，前面作为 history（你也可以改成随机切分）
            target_id = int(seq[-1])
            hist_ids = [int(x) for x in seq[:-1]]
            hist_ids = hist_ids[-args.max_hist:]

            if len(hist_ids) < args.min_hist:
                continue

            # namecat 文本
            if target_id not in item2namecat:
                continue
            hist_namecats = []
            ok = True
            for iid in hist_ids:
                nc = item2namecat.get(iid)
                if not nc:
                    ok = False
                    break
                hist_namecats.append(nc)
            if not ok:
                continue

            prompt = default_prompt(hist_namecats)
            target_namecat = item2namecat[target_id]

            rec = {
                "prompt": prompt,
                "history_item_ids": hist_ids,
                "target_item_id": target_id,
                "target_namecat": target_namecat,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            tried += 1
            pbar.update(1)

        if tried == 0:
            raise RuntimeError("No samples were generated. Check your mappings.")
    pbar.close()
    out.close()
    print(f"[OK] saved: {args.out_jsonl}  samples={kept}")

if __name__ == "__main__":
    main()
