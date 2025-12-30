import argparse, json, re
from collections import Counter, defaultdict

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def split_namecat(s: str):
    s = norm(s)
    m = re.match(r"^(.*)\s*\((.*)\)\s*$", s)
    if not m:
        return s, None
    return norm(m.group(1)), norm(m.group(2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss_list_txt", required=True, help="把 check_namecat_hit_v2 的 true miss 87 条保存成 txt，一行一个")
    ap.add_argument("--namecat2gmap", required=True, help="namecat2gmap_ids.json")
    ap.add_argument("--topk_name_only", type=int, default=20)
    args = ap.parse_args()

    namecat2gmap = load_json(args.namecat2gmap)

    # 建反向：name -> keys
    name2keys = defaultdict(list)
    for k in namecat2gmap.keys():
        name, cat = split_namecat(k)
        name2keys[name].append(k)

    misses = []
    with open(args.miss_list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = norm(line)
            if s:
                misses.append(s)

    print(f"[INFO] loaded misses: {len(misses)}")

    not_found = 0
    has_same_name = 0

    for s in misses:
        if s in namecat2gmap:
            print(f"[WEIRD] miss key actually exists: {s}")
            continue

        name, cat = split_namecat(s)
        cand_keys = name2keys.get(name, [])
        if cand_keys:
            has_same_name += 1
            print("="*60)
            print(f"MISS: {s}")
            print(f"FOUND same-name keys: {len(cand_keys)}")
            # 打印前 topk
            for kk in cand_keys[:args.topk_name_only]:
                print("  ", kk)
        else:
            not_found += 1
            print("="*60)
            print(f"MISS: {s}")
            print("NO same-name keys in meta maps.")

    print("========== SUMMARY ==========")
    print(f"miss_total={len(misses)}")
    print(f"miss_has_same_name_diff_cat={has_same_name}")
    print(f"miss_no_same_name_in_maps={not_found}")

if __name__ == "__main__":
    main()
