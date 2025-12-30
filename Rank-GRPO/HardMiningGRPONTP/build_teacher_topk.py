# HardMiningGRPO/build_teacher_topk.py
import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Set, Optional

import torch
from tqdm import tqdm

from SASRec import SASRec


# -----------------------------
# Utils: dedup + finalize pool
# -----------------------------
def dedup_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def finalize_pool(
    pool: List[int],
    tgt: int,
    K: int,
    fill_from: List[int],
    banned: Optional[Set[int]] = None,   # 比如 history（如果你要 filter_history）
    n_items: Optional[int] = None,
) -> List[int]:
    banned = banned or set()

    cleaned = []
    for x in pool:
        if x is None:
            continue
        x = int(x)
        if x == 0:
            continue
        if n_items is not None and not (1 <= x <= n_items):
            continue
        if x in banned:
            continue
        cleaned.append(x)

    pool = dedup_keep_order(cleaned)

    # 确保 tgt 在里面（先加进去，后面如果超长再截断）
    if tgt not in pool and (tgt not in banned):
        pool.append(int(tgt))

    seen = set(pool)

    # 用 fill_from 补齐（必须是“真实候选来源”，绝不要用常数 1）
    for x in fill_from:
        x = int(x)
        if x == 0:
            continue
        if n_items is not None and not (1 <= x <= n_items):
            continue
        if x in banned or x in seen:
            continue
        pool.append(x)
        seen.add(x)
        if len(pool) >= K:
            break

    # 还不够（极少发生）：兜底补齐（可选）
    if len(pool) < K and n_items is not None:
        for x in range(1, n_items + 1):
            if x in banned or x in seen:
                continue
            pool.append(x)
            if len(pool) >= K:
                break

    pool = pool[:K]

    # 最后强制确保 tgt 在 pool
    if tgt not in pool and (tgt not in banned) and len(pool) > 0:
        pool[-1] = int(tgt)

    return pool


def pad_left(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    seq = list(seq or [])
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


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
    import pickle

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
    return sasrec, n_items


@torch.no_grad()
def get_user_repr(sasrec: SASRec, input_ids: torch.LongTensor) -> torch.Tensor:
    feats = sasrec.log2feats(input_ids)  # [B,L,H]
    return feats[:, -1, :]               # [B,H]


def _parse_score_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown --score_dtype {s}. Choose from: fp16/bf16/fp32")


@torch.no_grad()
def mine_teacher_components_streaming(
    user_repr: torch.Tensor,          # [B,H] on device
    target_ids: torch.LongTensor,     # [B] on device
    item_emb_weight: torch.Tensor,    # [N+1,H] on same device
    scan_topk: int,                  # ✅ 方案A：扫描保留更长 top 列表
    chunk_size: int,
    score_dtype: torch.dtype,
    pool_mode: str,
    topk_out: int,                   # 最终输出 K（例如 200），用于 hit_rate 统计
    head_k: int,
    near_above_k: int,
    near_below_k: int,
    show_chunk_pbar: bool,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    返回：
      head_full: [B, scan_topk] 全库 top 列表（真实来源，用于补齐）
      above:     [B, near_above_k] 贴近 target 且 score>=target 的最接近项（排除target）
      below:     [B, near_below_k] 贴近 target 且 score<=target 的最接近项（排除target）
      hit_topk:  [B] target 是否自然出现在 head_full 的前 topk_out（诊断 teacher 强度）
    """
    device = user_repr.device
    B, H = user_repr.shape
    N = item_emb_weight.size(0) - 1
    if N <= 0:
        empty_head = torch.zeros((B, scan_topk), device=device, dtype=torch.long)
        empty_a = torch.zeros((B, near_above_k), device=device, dtype=torch.long)
        empty_b = torch.zeros((B, near_below_k), device=device, dtype=torch.long)
        hit = torch.zeros((B,), device=device, dtype=torch.bool)
        return empty_head, empty_a, empty_b, hit

    scan_topk = int(scan_topk)
    scan_topk = max(1, scan_topk)

    # user repr cast
    u = user_repr.to(dtype=score_dtype)

    # target score
    tgt_emb = item_emb_weight[target_ids].to(dtype=score_dtype)  # [B,H]
    tgt_score = (u * tgt_emb).sum(dim=-1, keepdim=True).float()  # [B,1]

    # head_full topk
    best_scores = torch.full((B, scan_topk), -1e9, device=device, dtype=torch.float32)
    best_ids = torch.zeros((B, scan_topk), device=device, dtype=torch.long)

    # near lists
    if pool_mode == "head_near":
        best_above = torch.full((B, near_above_k), -1e9, device=device, dtype=torch.float32)
        best_above_ids = torch.zeros((B, near_above_k), device=device, dtype=torch.long)

        best_below = torch.full((B, near_below_k), -1e9, device=device, dtype=torch.float32)
        best_below_ids = torch.zeros((B, near_below_k), device=device, dtype=torch.long)
    else:
        best_above_ids = torch.empty((B, 0), device=device, dtype=torch.long)
        best_below_ids = torch.empty((B, 0), device=device, dtype=torch.long)

    total_chunks = math.ceil(N / chunk_size)
    it = range(total_chunks)
    pbar = tqdm(it, desc="scan item chunks", leave=False, dynamic_ncols=True) if show_chunk_pbar else it

    # 用于排除 target 自身（避免 near_above/near_below 被 target 抢占）
    row_all = torch.arange(B, device=device)

    for ci in pbar:
        start = 1 + ci * chunk_size
        end = min(N + 1, start + chunk_size)  # exclusive
        emb = item_emb_weight[start:end].to(dtype=score_dtype)  # [C,H]

        scores = torch.matmul(u, emb.t())      # [B,C]
        scores_f = scores.float()

        # ---- head_full merge ----
        kk = min(scan_topk, scores_f.size(1))
        sc, idx = torch.topk(scores_f, k=kk, dim=1)
        ids = idx + start

        merged_scores = torch.cat([best_scores, sc], dim=1)
        merged_ids = torch.cat([best_ids, ids], dim=1)

        new_scores, new_idx = torch.topk(merged_scores, k=scan_topk, dim=1)
        best_scores = new_scores
        best_ids = merged_ids.gather(1, new_idx)

        # ---- near mining ----
        if pool_mode == "head_near":
            diff = scores_f - tgt_score  # [B,C]

            # 屏蔽当前 chunk 内的 target 本身
            in_range = (target_ids >= start) & (target_ids < end)
            if in_range.any():
                rr = row_all[in_range]
                cc = (target_ids[in_range] - start).long()
                # 下面会对 score_above/score_below 用 -1e9 作为不可选
                # 这里先准备好索引，后面分别置 -1e9
            else:
                rr = cc = None

            if near_above_k > 0:
                score_above = torch.where(diff >= 0, -diff, torch.full_like(diff, -1e9))
                if rr is not None:
                    score_above[rr, cc] = -1e9

                ka = min(near_above_k, score_above.size(1))
                sca, idxa = torch.topk(score_above, k=ka, dim=1)
                ida = idxa + start

                merged_a = torch.cat([best_above, sca], dim=1)
                merged_ai = torch.cat([best_above_ids, ida], dim=1)
                new_a, new_ai = torch.topk(merged_a, k=near_above_k, dim=1)
                best_above = new_a
                best_above_ids = merged_ai.gather(1, new_ai)

            if near_below_k > 0:
                score_below = torch.where(diff <= 0, diff, torch.full_like(diff, -1e9))
                if rr is not None:
                    score_below[rr, cc] = -1e9

                kb = min(near_below_k, score_below.size(1))
                scb, idxb = torch.topk(score_below, k=kb, dim=1)
                idb = idxb + start

                merged_b = torch.cat([best_below, scb], dim=1)
                merged_bi = torch.cat([best_below_ids, idb], dim=1)
                new_b, new_bi = torch.topk(merged_b, k=near_below_k, dim=1)
                best_below = new_b
                best_below_ids = merged_bi.gather(1, new_bi)

        if show_chunk_pbar and ci % 10 == 0:
            pbar.set_postfix({"items": f"{start}-{end-1}"})

    hit_topk = (best_ids[:, :int(topk_out)] == target_ids.unsqueeze(1)).any(dim=1)
    return best_ids, best_above_ids, best_below_ids, hit_topk


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--topk", type=int, default=200)              # 最终输出 K
    ap.add_argument("--scan_topk", type=int, default=1000)        # ✅ 方案A：扫描保留更长 top 列表
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--item_emb_on_gpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)
    ap.add_argument("--sasrec_max_len", type=int, default=50)

    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--count_total", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--pool_mode", type=str, default="head_near", choices=["topk", "head_near"])
    ap.add_argument("--head_k", type=int, default=80)
    ap.add_argument("--near_above_k", type=int, default=60)
    ap.add_argument("--near_below_k", type=int, default=59)

    ap.add_argument("--filter_history", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--show_chunk_pbar", action=argparse.BooleanOptionalAction, default=False)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    if os.path.exists(args.output_jsonl) and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} exists. Use --overwrite to replace it.")

    if int(args.scan_topk) < int(args.topk):
        raise ValueError(f"[BAD CONFIG] scan_topk({args.scan_topk}) must be >= topk({args.topk}).")

    if args.pool_mode == "head_near":
        # 这条不是必须，但通常你就是想这么配（不等也没问题，因为 finalize_pool 会补齐/截断）
        if args.head_k + args.near_above_k + args.near_below_k + 1 != int(args.topk):
            print(
                f"[WARN] head_k({args.head_k})+near_above_k({args.near_above_k})+near_below_k({args.near_below_k})+1 "
                f"!= topk({args.topk}). finalize_pool will handle length anyway."
            )

    device = args.device
    score_dtype = _parse_score_dtype(args.score_dtype)

    sasrec, n_items = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )

    item_emb_weight = sasrec.item_emb.weight.detach()
    if args.item_emb_on_gpu and str(device).startswith("cuda"):
        item_emb_weight = item_emb_weight.to(device)
        print(f"[OK] item_emb.weight on GPU: shape={tuple(item_emb_weight.shape)} dtype={item_emb_weight.dtype}")
    else:
        item_emb_weight = item_emb_weight.float().cpu()
        print(f"[OK] item_emb.weight on CPU: shape={tuple(item_emb_weight.shape)} dtype={item_emb_weight.dtype}")

    total = None
    if args.count_total:
        print(f"[INFO] counting lines for progress bar: {args.input_jsonl} ...")
        total = count_lines(args.input_jsonl)
        print(f"[OK] total lines = {total}")

    batch_hist: List[List[int]] = []
    batch_raw: List[Dict[str, Any]] = []
    batch_tgt: List[int] = []
    batch_banned: List[Set[int]] = []

    processed = 0
    hit_cnt = 0
    hit_total = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        outer_pbar = tqdm(total=total, desc="build teacher_top_item_ids", dynamic_ncols=True)

        for line in fin:
            line = line.strip()
            if not line:
                outer_pbar.update(1)
                continue

            ex = json.loads(line)

            hist = ex.get("history_item_ids", [])
            tgt = ex.get("target_item_id", None)
            if tgt is None:
                raise KeyError("missing target_item_id in input jsonl")

            hist_int = [int(x) for x in (hist or []) if int(x) != 0]
            hist_pad = pad_left(hist_int, args.sasrec_max_len, pad=0)

            batch_hist.append(hist_pad)
            batch_raw.append(ex)
            batch_tgt.append(int(tgt))

            if args.filter_history:
                # ✅ banned 里不要包含 target（永远保留 target）
                b = set(hist_int)
                if int(tgt) in b:
                    b.remove(int(tgt))
                batch_banned.append(b)
            else:
                batch_banned.append(set())

            if len(batch_hist) >= args.batch_size:
                input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
                user_repr = get_user_repr(sasrec, input_ids)

                if item_emb_weight.device != user_repr.device:
                    item_emb_weight = item_emb_weight.to(user_repr.device)
                    print("[WARN] moved item_emb_weight to device for speed.")

                tgt_ids = torch.tensor(batch_tgt, dtype=torch.long, device=user_repr.device)

                head_full, above_ids, below_ids, hit = mine_teacher_components_streaming(
                    user_repr=user_repr,
                    target_ids=tgt_ids,
                    item_emb_weight=item_emb_weight,
                    scan_topk=int(args.scan_topk),
                    chunk_size=int(args.chunk_size),
                    score_dtype=score_dtype,
                    pool_mode=str(args.pool_mode),
                    topk_out=int(args.topk),
                    head_k=int(args.head_k),
                    near_above_k=int(args.near_above_k),
                    near_below_k=int(args.near_below_k),
                    show_chunk_pbar=bool(args.show_chunk_pbar),
                )

                head_full = head_full.detach().cpu().tolist()
                above_ids = above_ids.detach().cpu().tolist()
                below_ids = below_ids.detach().cpu().tolist()
                hit = hit.detach().cpu().tolist()

                for ex0, tgt0, head0, ab0, bl0, banned0, hit0 in zip(
                    batch_raw, batch_tgt, head_full, above_ids, below_ids, batch_banned, hit
                ):
                    tgt0 = int(tgt0)
                    head0 = [int(x) for x in head0]
                    ab0 = [int(x) for x in ab0]
                    bl0 = [int(x) for x in bl0]

                    if args.pool_mode == "topk":
                        base_pool = head0[: int(args.topk)]
                    else:
                        base_pool = head0[: int(args.head_k)] + ab0 + [tgt0] + bl0

                    ids0 = finalize_pool(
                        pool=base_pool,
                        tgt=tgt0,
                        K=int(args.topk),
                        fill_from=head0,          # ✅ 用 scan_topk 的真实 head_full 补齐
                        banned=banned0,
                        n_items=n_items,
                    )

                    ex0["teacher_top_item_ids"] = ids0
                    fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

                    hit_cnt += (1 if bool(hit0) else 0)
                    hit_total += 1

                processed += len(batch_hist)
                batch_hist, batch_raw, batch_tgt, batch_banned = [], [], [], []

                if args.log_every > 0 and processed % args.log_every == 0:
                    hit_rate = hit_cnt / max(1, hit_total)
                    outer_pbar.set_postfix({"processed": processed, "teacher_hit@K": f"{hit_rate:.6f}"})

            outer_pbar.update(1)

        # flush last
        if batch_hist:
            input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
            user_repr = get_user_repr(sasrec, input_ids)

            if item_emb_weight.device != user_repr.device:
                item_emb_weight = item_emb_weight.to(user_repr.device)
                print("[WARN] moved item_emb_weight to device for speed.")

            tgt_ids = torch.tensor(batch_tgt, dtype=torch.long, device=user_repr.device)

            head_full, above_ids, below_ids, hit = mine_teacher_components_streaming(
                user_repr=user_repr,
                target_ids=tgt_ids,
                item_emb_weight=item_emb_weight,
                scan_topk=int(args.scan_topk),
                chunk_size=int(args.chunk_size),
                score_dtype=score_dtype,
                pool_mode=str(args.pool_mode),
                topk_out=int(args.topk),
                head_k=int(args.head_k),
                near_above_k=int(args.near_above_k),
                near_below_k=int(args.near_below_k),
                show_chunk_pbar=bool(args.show_chunk_pbar),
            )

            head_full = head_full.detach().cpu().tolist()
            above_ids = above_ids.detach().cpu().tolist()
            below_ids = below_ids.detach().cpu().tolist()
            hit = hit.detach().cpu().tolist()

            for ex0, tgt0, head0, ab0, bl0, banned0, hit0 in zip(
                batch_raw, batch_tgt, head_full, above_ids, below_ids, batch_banned, hit
            ):
                tgt0 = int(tgt0)
                head0 = [int(x) for x in head0]
                ab0 = [int(x) for x in ab0]
                bl0 = [int(x) for x in bl0]

                if args.pool_mode == "topk":
                    base_pool = head0[: int(args.topk)]
                else:
                    base_pool = head0[: int(args.head_k)] + ab0 + [tgt0] + bl0

                ids0 = finalize_pool(
                    pool=base_pool,
                    tgt=tgt0,
                    K=int(args.topk),
                    fill_from=head0,
                    banned=banned0,
                    n_items=n_items,
                )

                ex0["teacher_top_item_ids"] = ids0
                fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

                hit_cnt += (1 if bool(hit0) else 0)
                hit_total += 1

            processed += len(batch_hist)

        outer_pbar.close()

    hit_rate = hit_cnt / max(1, hit_total)
    print(f"✅ DONE. wrote teacher_top_item_ids to: {args.output_jsonl} (processed={processed})")
    print(f"📌 teacher_hit@{int(args.topk)} (target naturally in head_full[:K]) = {hit_rate:.6f}")


if __name__ == "__main__":
    main()
