# TeacherModel/build_namecat_maps_from_meta.py
import os
import re
import json
import gzip
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

def norm_gid(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.strip('"').strip("'").replace("\ufeff", "")
    s = re.sub(r"\s+", "", s)
    return s  # 注意：你验证 raw 就能命中，所以这里不强制 lower

def pick_primary_category(cat_field):
    """
    meta 里的 category 通常是 list[str]，也可能是 str / None
    我们统一取“主类目”：list 取第一个非空字符串；str 直接用；否则返回空
    """
    if cat_field is None:
        return ""
    if isinstance(cat_field, list):
        for x in cat_field:
            if x is None:
                continue
            x = str(x).strip()
            if x:
                return x
        return ""
    # str / other
    x = str(cat_field).strip()
    return x

def make_namecat(name: str, cat: str) -> str:
    name = (name or "").strip()
    cat = (cat or "").strip()
    if not name or not cat:
        return ""
    return f"{name} ({cat})"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="poi_semantic_ids.csv")
    ap.add_argument("--meta_files", required=True, nargs="+", help="meta-*.json.gz list")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_ids_per_key", type=int, default=50,
                    help="每个 namecat 最多保存多少个 gmap_id（防止极端 key 爆内存）")
    ap.add_argument("--encoding", type=str, default="utf-8")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] loading csv: {args.csv}")
    df = pd.read_csv(args.csv)
    if "gmap_id" not in df.columns:
        raise ValueError(f"csv missing gmap_id column. got cols={list(df.columns)}")

    df["gmap_id"] = df["gmap_id"].astype(str)
    valid = set(norm_gid(x) for x in df["gmap_id"].tolist())
    print(f"[INFO] valid gmap_id from csv: {len(valid)}")

    gmap_id2namecat = {}
    namecat2gmap_ids = defaultdict(list)

    scanned = 0
    matched = 0
    kept = 0
    skipped_no_namecat = 0
    skipped_not_valid = 0

    for mf in args.meta_files:
        print(f"[INFO] scanning meta: {mf}")
        with gzip.open(mf, "rt", encoding=args.encoding, errors="ignore") as f:
            for line in tqdm(f):
                scanned += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                gid = norm_gid(obj.get("gmap_id"))
                if not gid:
                    continue
                if gid not in valid:
                    skipped_not_valid += 1
                    continue

                matched += 1
                name = obj.get("name", "")
                cat = pick_primary_category(obj.get("category", ""))  # ✅ meta 的真实字段就是 category
                key = make_namecat(name, cat)
                if not key:
                    skipped_no_namecat += 1
                    continue

                gmap_id2namecat[gid] = key

                # 反向表：一个 namecat 可能对应多个 gmap_id
                if len(namecat2gmap_ids[key]) < args.max_ids_per_key:
                    namecat2gmap_ids[key].append(gid)

                kept += 1

    print("========== DONE ==========")
    print(f"[INFO] scanned meta rows: {scanned}")
    print(f"[INFO] matched gmap_ids (in valid): {matched}")
    print(f"[INFO] kept namecat pairs: {kept}")
    print(f"[INFO] skipped_not_valid: {skipped_not_valid}")
    print(f"[INFO] skipped_no_namecat: {skipped_no_namecat}")
    print(f"[INFO] unique namecat keys: {len(namecat2gmap_ids)}")

    out1 = os.path.join(args.out_dir, "gmap_id2namecat.json")
    out2 = os.path.join(args.out_dir, "namecat2gmap_ids.json")

    with open(out1, "w", encoding="utf-8") as f:
        json.dump(gmap_id2namecat, f, ensure_ascii=False)
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(namecat2gmap_ids, f, ensure_ascii=False)

    print(f"[OK] saved: {out1}")
    print(f"[OK] saved: {out2}")

if __name__ == "__main__":
    main()
