import os
import json
import pickle
import argparse
import pandas as pd

def norm(s: str) -> str:
    return " ".join(str(s).strip().split())

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="poi_semantic_ids.csv path")
    ap.add_argument("--pkl", type=str, required=True, help="sasrec_dataset.pkl path (contains item2id)")
    ap.add_argument("--out", type=str, required=True, help="output poi_text2id.json")
    ap.add_argument("--format", type=str, default="name_cat", choices=["name_cat", "gmap_only"],
                    help="name_cat: 'Name (Category)' if possible; else fallback. gmap_only: key=gmap_id")
    args = ap.parse_args()

    # 1) load sasrec_dataset.pkl -> item2id
    with open(args.pkl, "rb") as f:
        obj = pickle.load(f)
    if "item2id" not in obj:
        raise RuntimeError(f"Cannot find item2id in {args.pkl}. keys={list(obj.keys())}")
    item2id = obj["item2id"]  # gmap_id -> int

    # 2) load csv
    df = pd.read_csv(args.csv)
    if "gmap_id" not in df.columns:
        raise RuntimeError(f"csv must contain gmap_id col, got cols={list(df.columns)}")
    df["gmap_id"] = df["gmap_id"].astype(str).str.strip()

    # try auto-detect name/category columns
    name_col = pick_col(df, ["name", "poi_name", "title", "place_name"])
    cat_col  = pick_col(df, ["category", "cat", "poi_cat", "categories", "main_category"])

    poi_text2id = {}
    missing_in_item2id = 0
    used_fallback = 0

    for _, row in df.iterrows():
        gid = str(row["gmap_id"]).strip()
        if gid not in item2id:
            missing_in_item2id += 1
            continue
        iid = int(item2id[gid])

        if args.format == "gmap_only":
            key = norm(gid)
            poi_text2id[key] = iid
            continue

        # name_cat format preferred
        if name_col and cat_col:
            name = norm(row[name_col])
            cat = norm(row[cat_col])
            if name and cat:
                key = norm(f"{name} ({cat})")
                poi_text2id[key] = iid
            else:
                key = norm(gid)
                poi_text2id[key] = iid
                used_fallback += 1
        else:
            # csv 没有 name/cat，就只能先用 gid（后面你要么改 LLM 输出格式，要么补齐 csv）
            key = norm(gid)
            poi_text2id[key] = iid
            used_fallback += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(poi_text2id, f, ensure_ascii=False)

    print(f"[OK] saved: {args.out}")
    print(f"poi_text2id size: {len(poi_text2id)}")
    print(f"missing gmap_id not in item2id: {missing_in_item2id}")
    print(f"fallback_to_gmap_id keys: {used_fallback}")
    print(f"csv cols: {list(df.columns)}")
    print(f"detected name_col={name_col} cat_col={cat_col}")

if __name__ == "__main__":
    main()
