import os
import json
import pickle
import random
import argparse
from typing import Dict, Any, List, Tuple

RULE = "只输出一个地点名(类别)，不要解释"

PROMPT_TPL = """你将看到用户最近访问的地点列表（按时间从旧到新），请预测用户下一次最可能去的一个地点。
历史：
{history_lines}
{rule}
"""

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def item_id_to_namecat(item_id: int, id2item: Dict[int, str], gmap_id2namecat: Dict[str, str]) -> str:
    gid = id2item.get(int(item_id))
    if gid is None:
        return f"<UNK_ITEM:{item_id}>"
    return gmap_id2namecat.get(gid, f"<UNK_GMAP:{gid}>")

def build_one_example(
    seq: List[int],
    id2item: Dict[int, str],
    gmap_id2namecat: Dict[str, str],
    history_len: int
) -> Dict[str, Any]:
    # seq: [i1, i2, ..., it]  (len>=2)
    target_item = int(seq[-1])
    hist = [int(x) for x in seq[:-1]]

    hist = hist[-history_len:]
    hist_namecats = [item_id_to_namecat(x, id2item, gmap_id2namecat) for x in hist]
    target_namecat = item_id_to_namecat(target_item, id2item, gmap_id2namecat)

    history_lines = "\n".join([f"{i+1}. {t}" for i, t in enumerate(hist_namecats)])

    prompt = PROMPT_TPL.format(history_lines=history_lines, rule=RULE)

    return {
        "prompt": prompt,

        # 用于 reward 的“干净真值”
        "history_item_ids": hist,
        "target_item_id": target_item,

        # 方便 debug/人工看
        "history_namecats": hist_namecats,
        "target_namecat": target_namecat,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="sasrec_dataset.pkl")
    ap.add_argument("--gmap_id2namecat", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--history_len", type=int, default=20)
    ap.add_argument("--min_seq_len", type=int, default=5)
    ap.add_argument("--max_users", type=int, default=0, help="0=use all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--test_ratio", type=float, default=0.01)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pkl, "rb") as f:
        obj = pickle.load(f)

    data = obj["data"]  # list of {"user_id":..., "sequence":[...]}
    id2item = obj["id2item"]  # int -> gmap_id (可能是 dict[int,str] 或 dict[str,str])
    # 统一 key 类型
    id2item = {int(k): v for k, v in id2item.items()}

    gmap_id2namecat = load_json(args.gmap_id2namecat)

    # 过滤 + 可选抽样用户
    users = []
    for row in data:
        seq = row.get("sequence", [])
        if not isinstance(seq, list) or len(seq) < max(2, args.min_seq_len):
            continue
        users.append(row)

    if args.max_users and len(users) > args.max_users:
        users = random.sample(users, args.max_users)

    random.shuffle(users)

    n = len(users)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    test_users = users[:n_test]
    val_users = users[n_test:n_test+n_val]
    train_users = users[n_test+n_val:]

    def build_split(split_users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for u in split_users:
            seq = u["sequence"]
            ex = build_one_example(seq, id2item, gmap_id2namecat, args.history_len)
            ex["user_id"] = u.get("user_id", "")
            rows.append(ex)
        return rows

    train_rows = build_split(train_users)
    val_rows = build_split(val_users)
    test_rows = build_split(test_users)

    save_jsonl(os.path.join(args.out_dir, "grpo_train.jsonl"), train_rows)
    save_jsonl(os.path.join(args.out_dir, "grpo_val.jsonl"), val_rows)
    save_jsonl(os.path.join(args.out_dir, "grpo_test.jsonl"), test_rows)

    print("========== DONE ==========")
    print(f"users_total={n}")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
