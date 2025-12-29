# TeacherModel/build_namecat2item_ids.py
import os
import json
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="SASRec_Data/sasrec_dataset.pkl")
    ap.add_argument("--namecat2gmap", required=True, help="SASRec_Data/namecat2gmap_ids.json")
    ap.add_argument("--out", required=True, help="SASRec_Data/namecat2item_ids.json")
    ap.add_argument("--unique_only", action="store_true",
                    help="只输出能映射到唯一 item_id 的 namecat（推荐用于 GRPO）")
    return ap.parse_args()

def main():
    args = parse_args()

    with open(args.pkl, "rb") as f:
        obj = pickle.load(f)
    item2id = obj["item2id"]  # gmap_id -> int

    with open(args.namecat2gmap, "r", encoding="utf-8") as f:
        namecat2gmap = json.load(f)

    out_map = {}
    ambiguous = 0
    missing_gid = 0
    kept = 0

    for key, gids in tqdm(namecat2gmap.items()):
        item_ids = []
        for gid in gids:
            iid = item2id.get(gid)
            if iid is None:
                missing_gid += 1
                continue
            item_ids.append(int(iid))

        if not item_ids:
            continue

        # 去重
        uniq = sorted(set(item_ids))

        if args.unique_only:
            if len(uniq) == 1:
                out_map[key] = uniq[0]
                kept += 1
            else:
                ambiguous += 1
        else:
            out_map[key] = uniq
            if len(uniq) > 1:
                ambiguous += 1
            kept += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_map, f, ensure_ascii=False)

    print("========== DONE ==========")
    print(f"[INFO] total keys in: {len(namecat2gmap)}")
    print(f"[INFO] kept keys out: {kept}")
    print(f"[INFO] ambiguous keys: {ambiguous}")
    print(f"[INFO] missing gmap_id (not in item2id): {missing_gid}")
    print(f"[OK] saved: {args.out}")

if __name__ == "__main__":
    main()
