# HardMiningGRPO/build_candidate_item_ids_precise.py
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


def norm_text(s: str) -> str:
    s = "" if s is None else str(s).strip()
    s = " ".join(s.split())
    # 跟你 reward 的 canon_key 保持一致（centre->center）
    s = s.replace("centre", "center")
    return s


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
    # 期望输出 [B, C]
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

    # 全量 & 截断映射（推荐两者都给：大歧义 key 用 disamb，其他用 all）
    ap.add_argument("--namecat2item_all", required=True)
    ap.add_argument("--namecat2item_disamb", default="")

    # 控制极端歧义 key 的候选上限
    ap.add_argument("--max_ids_per_namecat", type=int, default=256)
    ap.add_argument("--use_disamb_for_large", action="store_true")

    # 处理效率
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=2000)

    # SASRec arch（必须匹配）
    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    return ap.parse_args()


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
        "missing_map": 0,
        "used_disamb_large": 0,
        "had_ambiguous": 0,
        "total_amb_groups": 0,
        "total_pool_size": 0,
        "pool_overflow_trunc": 0,
    }

    B = len(batch)

    # 每条样本：最终输出 candidate_item_ids（len=50）
    final_ids: List[List[int]] = []
    final_namecats: List[List[str]] = []

    # 需要 SASRec 打分的 pool（每条样本一个 pool）
    pools: List[List[int]] = []
    # 记录每条样本里，哪些候选索引是 ambiguous，需要从 pool 里挑 best
    # amb_groups[b] = list of (cand_index, start, end)
    amb_groups: List[List[Tuple[int, int, int]]] = []
    # history tensor
    hist_tensor = torch.zeros((B, max_len), dtype=torch.long, device=device)

    for b, o in enumerate(batch):
        c = o.get("candidate_namecats")
        if c is None:
            c = o.get("candidates_namecat")
        if c is None:
            raise ValueError("missing candidates namecats field (candidate_namecats or candidates_namecat)")
        if isinstance(c, str):
            c = [c]
        c = [norm_text(x) for x in c if norm_text(x)]

        tgt_nc = norm_text(o.get("target_namecat", ""))
        tgt_id = int(o["target_item_id"])

        # history: 右对齐，左 pad 0
        hist = [int(x) for x in o.get("history_item_ids", []) if int(x) >= 0]
        hist = hist[-max_len:]
        if len(hist) < max_len:
            hist = [0] * (max_len - len(hist)) + hist
        hist_tensor[b] = torch.tensor(hist, dtype=torch.long, device=device)

        chosen = [-1] * len(c)
        pool: List[int] = []
        groups: List[Tuple[int, int, int]] = []

        for j, namecat in enumerate(c):
            # target 保证严格对齐
            if tgt_nc and namecat == tgt_nc:
                chosen[j] = tgt_id
                continue

            ids = mp_all.get(namecat)
            if ids is None and mp_disamb is not None:
                ids = mp_disamb.get(namecat)

            if ids is None:
                stats["missing_map"] += 1
                chosen[j] = -1
                continue

            if isinstance(ids, int):
                chosen[j] = int(ids)
                continue

            ids = [int(x) for x in ids if isinstance(x, (int, float, str)) and str(x).isdigit()]
            ids = [x for x in ids if 1 <= x <= n_items]
            if not ids:
                stats["missing_map"] += 1
                chosen[j] = -1
                continue

            if len(ids) == 1:
                chosen[j] = int(ids[0])
                continue

            # 多对多：要用 SASRec disambiguate
            # 极端长的 ids：优先用 disamb（<=50），否则截断
            if len(ids) > max_ids_per_namecat:
                if use_disamb_for_large and mp_disamb is not None:
                    dis = mp_disamb.get(namecat)
                    if isinstance(dis, list) and len(dis) > 0:
                        ids2 = [int(x) for x in dis if 1 <= int(x) <= n_items]
                        if ids2:
                            ids = ids2
                            stats["used_disamb_large"] += 1
                if len(ids) > max_ids_per_namecat:
                    ids = ids[:max_ids_per_namecat]
                    stats["pool_overflow_trunc"] += 1

            start = len(pool)
            pool.extend(ids)
            end = len(pool)
            groups.append((j, start, end))

        if groups:
            stats["had_ambiguous"] += 1
            stats["total_amb_groups"] += len(groups)

        stats["total_pool_size"] += len(pool)

        final_ids.append(chosen)
        final_namecats.append(c)
        pools.append(pool)
        amb_groups.append(groups)

    Cmax = max((len(p) for p in pools), default=0)
    if Cmax == 0:
        # 没有任何 ambiguous，不需要打分
        out = []
        for o, c, chosen in zip(batch, final_namecats, final_ids):
            o["candidate_namecats"] = c
            o["candidate_item_ids"] = chosen
            # 清理旧字段避免混乱
            if "candidates_namecat" in o:
                del o["candidates_namecat"]
            out.append(o)
        return out, stats

    cand_tensor = torch.zeros((B, Cmax), dtype=torch.long, device=device)
    mask = torch.zeros((B, Cmax), dtype=torch.bool, device=device)

    for b in range(B):
        p = pools[b]
        if not p:
            continue
        cand_tensor[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        mask[b, :len(p)] = True

    scores = predict_candidates_batch(sasrec, hist_tensor, cand_tensor)  # [B,Cmax]
    # mask pad positions
    scores = scores.masked_fill(~mask, -1e9)

    # 为每个 ambiguous namecat 选最大 score 的 id
    for b in range(B):
        chosen = final_ids[b]
        p = pools[b]
        if not p:
            continue
        for (j, s, e) in amb_groups[b]:
            if e <= s:
                continue
            sub = scores[b, s:e]
            best_pos = int(torch.argmax(sub).item()) + s
            best_id = int(cand_tensor[b, best_pos].item())
            chosen[j] = best_id
        final_ids[b] = chosen

    out = []
    for o, c, chosen in zip(batch, final_namecats, final_ids):
        o["candidate_namecats"] = c
        o["candidate_item_ids"] = chosen
        if "candidates_namecat" in o:
            del o["candidates_namecat"]
        out.append(o)

    return out, stats


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
        "missing_map": 0,
        "used_disamb_large": 0,
        "had_ambiguous": 0,
        "total_amb_groups": 0,
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
                    print(f"[{total}] missing_map={agg['missing_map']} had_ambiguous={agg['had_ambiguous']} "
                          f"avg_pool={avg_pool:.2f} trunc={agg['pool_overflow_trunc']} disamb_large={agg['used_disamb_large']}")
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
    print("missing_map:", agg["missing_map"])
    print("had_ambiguous:", agg["had_ambiguous"], f"rate={agg['had_ambiguous']/max(1,total):.4f}")
    print("avg_pool_size:", f"{avg_pool:.2f}")
    print("pool_overflow_trunc:", agg["pool_overflow_trunc"])
    print("used_disamb_for_large:", agg["used_disamb_large"])


if __name__ == "__main__":
    main()
