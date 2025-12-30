# HardMiningGRPO/build_candidate_item_ids_precise_v2.py
import os
import sys
import json
import argparse
import random
import pickle
from typing import Dict, List, Any, Tuple, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec


def keep_raw(s: str) -> str:
    """保持原样（只做 strip + 多空格压缩），不做 centre->center 改写。"""
    s = "" if s is None else str(s).strip()
    return " ".join(s.split())


def canon_lookup(s: str) -> str:
    """只用于查表/对齐，不写回字段。"""
    s = keep_raw(s)
    # canonical：centre->center（仅 lookup）
    s = s.replace("centre", "center")
    return s


def alt_spellings(s: str) -> List[str]:
    """生成一些 lookup 备选 key（不写回）。"""
    s0 = keep_raw(s)
    s1 = canon_lookup(s0)
    # 反向：如果原来是 center，也试 centre
    s2 = keep_raw(s0).replace("center", "centre")
    s3 = keep_raw(s1).replace("center", "centre")
    keys = []
    for x in [s0, s1, s2, s3]:
        x = keep_raw(x)
        if x and x not in keys:
            keys.append(x)
    return keys


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sasrec_from_ckpt(
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int = 50,
    embed_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
):
    with open(sasrec_pkl, "rb") as f:
        obj = pickle.load(f)
    n_items = int(obj["n_items"])

    try:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu")

    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state_dict = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state_dict = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state_dict = ckpt_obj["model"]
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    class _Args:
        pass

    a = _Args()
    a.device = device
    a.max_len = int(max_len)
    a.embed_dim = int(embed_dim)
    a.num_blocks = int(num_blocks)
    a.num_heads = int(num_heads)
    a.dropout = float(dropout)

    sasrec = SASRec(item_num=n_items, args=a).to(device)
    sasrec.load_state_dict(state_dict, strict=True)
    sasrec.eval()
    for p in sasrec.parameters():
        p.requires_grad_(False)

    print(f"[OK] loaded SASRec: n_items={n_items}, max_len={a.max_len}, dim={a.embed_dim}, "
          f"blocks={a.num_blocks}, heads={a.num_heads}, dropout={a.dropout}")
    return sasrec, n_items, a.max_len


@torch.no_grad()
def predict_candidates_batch(sasrec, hist_batch: torch.Tensor, cand_batch: torch.Tensor) -> torch.Tensor:
    out = sasrec.predict_candidates(hist_batch, cand_batch)
    if out.dim() == 3:
        out = out.squeeze(1)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--namecat2item_all", required=True)
    ap.add_argument("--namecat2item_disamb", default="")

    ap.add_argument("--max_ids_per_namecat", type=int, default=256)
    ap.add_argument("--use_disamb_for_large", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=2000)

    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    return ap.parse_args()


def pick_random_valid(n_items: int, forbid: set) -> int:
    while True:
        x = random.randint(1, n_items)
        if x not in forbid:
            return x


def process_batch(
    batch: List[Dict[str, Any]],
    sasrec,
    n_items: int,
    max_len: int,
    mp_all: Dict[str, Any],
    mp_disamb: Optional[Dict[str, Any]],
    max_ids_per_namecat: int,
    use_disamb_for_large: bool,
    device: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:

    stats = {
        "missing_map_pos": 0,          # 按候选位置计
        "target_not_found_by_norm": 0, # 目标在候选里找不到（按 norm 比较）
        "used_disamb_for_large": 0,
        "had_ambiguous": 0,
        "total_pool_size": 0,
        "pool_overflow_trunc": 0,
    }

    B = len(batch)
    hist_tensor = torch.zeros((B, max_len), dtype=torch.long, device=device)

    # 这些用于 SASRec 打分
    pools: List[List[int]] = []
    amb_groups: List[List[Tuple[int, int, int]]] = []  # (cand_index, start, end)

    # 输出
    out_cands_raw: List[List[str]] = []
    out_ids: List[List[int]] = []

    for b, o in enumerate(batch):
        c = o.get("candidate_namecats")
        if c is None:
            c = o.get("candidates_namecat")
        if c is None:
            raise ValueError("missing candidates namecats field (candidate_namecats or candidates_namecat)")
        if isinstance(c, str):
            c = [c]
        c_raw = [keep_raw(x) for x in c if keep_raw(x)]
        c_norm = [canon_lookup(x) for x in c_raw]

        tgt_raw = keep_raw(o.get("target_namecat", ""))
        tgt_norm = canon_lookup(tgt_raw)
        tgt_id = int(o["target_item_id"])

        # history: 右对齐，左 pad 0
        hist = [int(x) for x in o.get("history_item_ids", []) if int(x) >= 0]
        hist = hist[-max_len:]
        if len(hist) < max_len:
            hist = [0] * (max_len - len(hist)) + hist
        hist_tensor[b] = torch.tensor(hist, dtype=torch.long, device=device)

        chosen = [-999] * len(c_raw)  # 临时占位，最后保证全是合法 [1..n_items]
        pool: List[int] = []
        groups: List[Tuple[int, int, int]] = []

        # 目标对齐：按 norm 找到候选的索引（不依赖 centre/center 拼写）
        target_idx = None
        for j, cn in enumerate(c_norm):
            if cn == tgt_norm and tgt_raw:
                target_idx = j
                break
        if target_idx is None and tgt_raw:
            stats["target_not_found_by_norm"] += 1
        else:
            chosen[target_idx] = tgt_id

        forbid = {tgt_id}

        for j, (raw_nc, norm_nc) in enumerate(zip(c_raw, c_norm)):
            if chosen[j] == tgt_id:
                continue

            ids = None
            # 依次尝试多种 spelling
            for key_try in alt_spellings(raw_nc):
                ids = mp_all.get(key_try)
                if ids is None and mp_disamb is not None:
                    ids = mp_disamb.get(key_try)
                if ids is not None:
                    break

            if ids is None:
                stats["missing_map_pos"] += 1
                # 缺失也填一个合法 id，避免训练严格校验炸
                rid = pick_random_valid(n_items, forbid)
                chosen[j] = rid
                forbid.add(rid)
                continue

            if isinstance(ids, int):
                chosen[j] = int(ids)
                forbid.add(int(ids))
                continue

            ids = [int(x) for x in ids if isinstance(x, (int, float, str)) and str(x).isdigit()]
            ids = [x for x in ids if 1 <= x <= n_items]
            if not ids:
                stats["missing_map_pos"] += 1
                rid = pick_random_valid(n_items, forbid)
                chosen[j] = rid
                forbid.add(rid)
                continue

            if len(ids) == 1:
                chosen[j] = int(ids[0])
                forbid.add(int(ids[0]))
                continue

            # 多对多：需要 SASRec disamb
            if len(ids) > max_ids_per_namecat:
                if use_disamb_for_large and mp_disamb is not None:
                    dis = None
                    for key_try in alt_spellings(raw_nc):
                        dis = mp_disamb.get(key_try)
                        if isinstance(dis, list) and len(dis) > 0:
                            dis2 = [int(x) for x in dis if 1 <= int(x) <= n_items]
                            if dis2:
                                ids = dis2
                                stats["used_disamb_for_large"] += 1
                                break
                if len(ids) > max_ids_per_namecat:
                    ids = ids[:max_ids_per_namecat]
                    stats["pool_overflow_trunc"] += 1

            start = len(pool)
            pool.extend(ids)
            end = len(pool)
            groups.append((j, start, end))

        if groups:
            stats["had_ambiguous"] += 1
        stats["total_pool_size"] += len(pool)

        out_cands_raw.append(c_raw)
        out_ids.append(chosen)
        pools.append(pool)
        amb_groups.append(groups)

    # SASRec disamb
    Cmax = max((len(p) for p in pools), default=0)
    if Cmax > 0:
        cand_tensor = torch.zeros((B, Cmax), dtype=torch.long, device=device)
        mask = torch.zeros((B, Cmax), dtype=torch.bool, device=device)

        for b in range(B):
            p = pools[b]
            if not p:
                continue
            cand_tensor[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            mask[b, :len(p)] = True

        scores = predict_candidates_batch(sasrec, hist_tensor, cand_tensor)  # [B,Cmax]
        scores = scores.masked_fill(~mask, -1e9)

        for b in range(B):
            chosen = out_ids[b]
            for (j, s, e) in amb_groups[b]:
                if e <= s:
                    continue
                sub = scores[b, s:e]
                best_pos = int(torch.argmax(sub).item()) + s
                best_id = int(cand_tensor[b, best_pos].item())
                chosen[j] = best_id
            out_ids[b] = chosen

    # 写回：candidate_namecats 必须保持原样；candidate_item_ids 必须合法
    out_batch = []
    for o, c_raw, ids in zip(batch, out_cands_raw, out_ids):
        # 清理旧字段
        if "candidates_namecat" in o:
            del o["candidates_namecat"]

        o["candidate_namecats"] = c_raw
        o["candidate_item_ids"] = [int(x) for x in ids]

        out_batch.append(o)

    return out_batch, stats


def main():
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sasrec, n_items, _ = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )

    mp_all = load_json(args.namecat2item_all)
    mp_disamb = load_json(args.namecat2item_disamb) if args.namecat2item_disamb else None

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    total = 0
    agg = {
        "missing_map_pos": 0,
        "target_not_found_by_norm": 0,
        "used_disamb_for_large": 0,
        "had_ambiguous": 0,
        "total_pool_size": 0,
        "pool_overflow_trunc": 0,
    }

    batch: List[Dict[str, Any]] = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f, open(args.out_jsonl, "w", encoding="utf-8") as g:
        for line in f:
            o = json.loads(line)
            batch.append(o)
            if len(batch) >= args.batch_size:
                out_batch, st = process_batch(
                    batch=batch,
                    sasrec=sasrec,
                    n_items=n_items,
                    max_len=args.sasrec_max_len,
                    mp_all=mp_all,
                    mp_disamb=mp_disamb,
                    max_ids_per_namecat=args.max_ids_per_namecat,
                    use_disamb_for_large=args.use_disamb_for_large,
                    device=device,
                )
                for oo in out_batch:
                    g.write(json.dumps(oo, ensure_ascii=False) + "\n")
                total += len(batch)
                for k in agg:
                    agg[k] += st.get(k, 0)
                if args.log_every > 0 and total % args.log_every == 0:
                    avg_pool = agg["total_pool_size"] / max(1, total)
                    print(f"[{total}] missing_pos={agg['missing_map_pos']} "
                          f"target_not_found_norm={agg['target_not_found_by_norm']} "
                          f"had_ambiguous={agg['had_ambiguous']} avg_pool={avg_pool:.2f} "
                          f"trunc={agg['pool_overflow_trunc']} disamb_large={agg['used_disamb_for_large']}")
                batch = []

        if batch:
            out_batch, st = process_batch(
                batch=batch,
                sasrec=sasrec,
                n_items=n_items,
                max_len=args.sasrec_max_len,
                mp_all=mp_all,
                mp_disamb=mp_disamb,
                max_ids_per_namecat=args.max_ids_per_namecat,
                use_disamb_for_large=args.use_disamb_for_large,
                device=device,
            )
            for oo in out_batch:
                g.write(json.dumps(oo, ensure_ascii=False) + "\n")
            total += len(batch)
            for k in agg:
                agg[k] += st.get(k, 0)

    avg_pool = agg["total_pool_size"] / max(1, total)
    print("saved:", args.out_jsonl)
    print("total:", total)
    print("missing_map_pos:", agg["missing_map_pos"])
    print("target_not_found_by_norm:", agg["target_not_found_by_norm"])
    print("had_ambiguous:", agg["had_ambiguous"], f"rate={agg['had_ambiguous']/max(1,total):.4f}")
    print("avg_pool_size:", f"{avg_pool:.2f}")
    print("pool_overflow_trunc:", agg["pool_overflow_trunc"])
    print("used_disamb_for_large:", agg["used_disamb_for_large"])


if __name__ == "__main__":
    main()
