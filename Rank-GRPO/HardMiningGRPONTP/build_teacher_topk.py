# HardMiningGRPO/build_teacher_topk.py
import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm

from SASRec import SASRec


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

    # 兼容多种保存格式
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


def _dedup_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _filter_history_keep_target(ids: List[int], hist_set: set, target_id: int) -> List[int]:
    # 关键：history 过滤时，绝对不删 target
    if not hist_set:
        return ids
    out = []
    for x in ids:
        if x == target_id:
            out.append(x)
            continue
        if x in hist_set:
            continue
        out.append(x)
    return out


@torch.no_grad()
def mine_teacher_pool_streaming(
    user_repr: torch.Tensor,          # [B,H] on device
    target_ids: torch.LongTensor,     # [B] on device
    item_emb_weight: torch.Tensor,    # [N+1,H] on same device
    topk: int,
    chunk_size: int,
    score_dtype: torch.dtype,
    pool_mode: str,
    head_k: int,
    near_above_k: int,
    near_below_k: int,
    show_chunk_pbar: bool,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """
    pool_mode:
      - topk: 纯全库 topk（头部）
      - head_near: head(top) + near_above(贴近且高于target) + target + near_below(贴近且低于target)
    返回:
      ids: [B, topk]（未做 history 过滤/未强行补齐 target）
      hit: [B] target 是否自然出现在“纯 topk 头部”里（用于诊断 teacher 强度）
    """
    device = user_repr.device
    B, H = user_repr.shape
    N = item_emb_weight.size(0) - 1
    if N <= 0:
        return torch.zeros((B, topk), device=device, dtype=torch.long), torch.zeros((B,), device=device, dtype=torch.bool)

    # user repr cast
    u = user_repr.to(dtype=score_dtype)

    # target score（用于 near mining）
    tgt_emb = item_emb_weight[target_ids].to(dtype=score_dtype)  # [B,H]
    tgt_score = (u * tgt_emb).sum(dim=-1, keepdim=True).float()  # [B,1] float32

    # ------- head topk -------
    best_scores = torch.full((B, topk), -1e9, device=device, dtype=torch.float32)
    best_ids = torch.zeros((B, topk), device=device, dtype=torch.long)

    # ------- near-above / near-below (仅 head_near 用) -------
    if pool_mode == "head_near":
        # above: maximize (-diff) where diff>=0  (diff = score - tgt_score)
        best_above = torch.full((B, near_above_k), -1e9, device=device, dtype=torch.float32)
        best_above_ids = torch.zeros((B, near_above_k), device=device, dtype=torch.long)
        # below: maximize (diff) where diff<=0 (diff negative close to 0 is larger)
        best_below = torch.full((B, near_below_k), -1e9, device=device, dtype=torch.float32)
        best_below_ids = torch.zeros((B, near_below_k), device=device, dtype=torch.long)
    else:
        best_above = best_above_ids = best_below = best_below_ids = None

    total_chunks = math.ceil(N / chunk_size)
    it = range(total_chunks)
    pbar = tqdm(it, desc="scan item chunks", leave=False, dynamic_ncols=True) if show_chunk_pbar else it

    for ci in pbar:
        start = 1 + ci * chunk_size
        end = min(N + 1, start + chunk_size)  # exclusive
        emb = item_emb_weight[start:end].to(dtype=score_dtype)  # [C,H]

        # scores [B,C]
        scores = torch.matmul(u, emb.t())      # fp16/bf16/fp32
        scores_f = scores.float()              # [B,C] float32

        # ---- head topk merge ----
        kk = min(topk, scores_f.size(1))
        sc, idx = torch.topk(scores_f, k=kk, dim=1)
        ids = idx + start

        merged_scores = torch.cat([best_scores, sc], dim=1)
        merged_ids = torch.cat([best_ids, ids], dim=1)

        new_scores, new_idx = torch.topk(merged_scores, k=topk, dim=1)
        best_scores = new_scores
        best_ids = merged_ids.gather(1, new_idx)

        # ---- near mining ----
        if pool_mode == "head_near":
            diff = scores_f - tgt_score  # [B,C]

            if near_above_k > 0:
                # above: diff>=0, want minimal diff => maximize -diff
                score_above = torch.where(diff >= 0, -diff, torch.full_like(diff, -1e9))
                ka = min(near_above_k, score_above.size(1))
                sca, idxa = torch.topk(score_above, k=ka, dim=1)
                ida = idxa + start

                merged_a = torch.cat([best_above, sca], dim=1)
                merged_ai = torch.cat([best_above_ids, ida], dim=1)
                new_a, new_ai = torch.topk(merged_a, k=near_above_k, dim=1)
                best_above = new_a
                best_above_ids = merged_ai.gather(1, new_ai)

            if near_below_k > 0:
                # below: diff<=0, want closest to 0 from below => maximize diff (negative but large)
                score_below = torch.where(diff <= 0, diff, torch.full_like(diff, -1e9))
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

    # 诊断：target 是否“自然出现在 head topk 里”
    hit = (best_ids == target_ids.unsqueeze(1)).any(dim=1)

    if pool_mode == "topk":
        return best_ids, hit

    # head_near: 只取 head 的前 head_k（避免 head 全塞满导致 target 永远在底部）
    head_k = max(0, min(head_k, topk))
    head = best_ids[:, :head_k] if head_k > 0 else torch.empty((B, 0), device=device, dtype=torch.long)

    # near 上下
    above = best_above_ids if near_above_k > 0 else torch.empty((B, 0), device=device, dtype=torch.long)
    below = best_below_ids if near_below_k > 0 else torch.empty((B, 0), device=device, dtype=torch.long)

    # 拼成 pool（先 above，再放 target，再 below；target 不会总在最后）
    # 最后再补齐到 topk（如果去重后不足）
    out = []
    for b in range(B):
        ids_b = []
        ids_b += head[b].tolist()
        ids_b += above[b].tolist()
        ids_b.append(int(target_ids[b].item()))
        ids_b += below[b].tolist()

        ids_b = [int(x) for x in ids_b if int(x) != 0]
        ids_b = _dedup_keep_order(ids_b)

        # 裁剪/补齐
        if len(ids_b) >= topk:
            ids_b = ids_b[:topk]
        else:
            # 不够就用 head 的剩余补齐（再不够就继续用 best_ids 补）
            fill = best_ids[b].tolist()
            for x in fill:
                if len(ids_b) >= topk:
                    break
                x = int(x)
                if x == 0:
                    continue
                if x in ids_b:
                    continue
                ids_b.append(x)

            # 极端情况还不够：用 1 补
            while len(ids_b) < topk:
                ids_b.append(1)

        out.append(ids_b)

    out_ids = torch.tensor(out, device=device, dtype=torch.long)
    return out_ids, hit


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--topk", type=int, default=200)
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

    # ✅ 新增：pool 形态（更利于 HR@1）
    ap.add_argument("--pool_mode", type=str, default="head_near", choices=["topk", "head_near"])
    ap.add_argument("--head_k", type=int, default=80)
    ap.add_argument("--near_above_k", type=int, default=60)
    ap.add_argument("--near_below_k", type=int, default=59)

    # ✅ 新增：过滤 history 但不删 target
    ap.add_argument("--filter_history", action=argparse.BooleanOptionalAction, default=False)

    # 进度条
    ap.add_argument("--show_chunk_pbar", action=argparse.BooleanOptionalAction, default=False)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    if os.path.exists(args.output_jsonl) and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} exists. Use --overwrite to replace it.")

    if args.pool_mode == "head_near":
        if args.head_k + args.near_above_k + args.near_below_k + 1 != int(args.topk):
            raise ValueError(
                f"[BAD CONFIG] head_k({args.head_k}) + near_above_k({args.near_above_k}) + "
                f"near_below_k({args.near_below_k}) + 1(target) must equal topk({args.topk})."
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
    batch_hist_sets: List[set] = []

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

            hist_int = [int(x) for x in (hist or [])]
            hist_pad = pad_left(hist_int, args.sasrec_max_len, pad=0)

            batch_hist.append(hist_pad)
            batch_raw.append(ex)
            batch_tgt.append(int(tgt))

            if args.filter_history:
                batch_hist_sets.append(set([x for x in hist_int if x != 0]))
            else:
                batch_hist_sets.append(set())

            if len(batch_hist) >= args.batch_size:
                input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
                user_repr = get_user_repr(sasrec, input_ids)  # [B,H]

                if item_emb_weight.device != user_repr.device:
                    item_emb_weight = item_emb_weight.to(user_repr.device)
                    print("[WARN] moved item_emb_weight to device for speed.")

                tgt_ids = torch.tensor(batch_tgt, dtype=torch.long, device=user_repr.device)

                pool_ids, hit = mine_teacher_pool_streaming(
                    user_repr=user_repr,
                    target_ids=tgt_ids,
                    item_emb_weight=item_emb_weight,
                    topk=int(args.topk),
                    chunk_size=int(args.chunk_size),
                    score_dtype=score_dtype,
                    pool_mode=str(args.pool_mode),
                    head_k=int(args.head_k),
                    near_above_k=int(args.near_above_k),
                    near_below_k=int(args.near_below_k),
                    show_chunk_pbar=bool(args.show_chunk_pbar),
                )

                pool_ids = pool_ids.detach().cpu().tolist()
                hit = hit.detach().cpu().tolist()

                for ex0, tgt0, ids0, hset0, hit0 in zip(batch_raw, batch_tgt, pool_ids, batch_hist_sets, hit):
                    ids0 = [int(x) for x in ids0]

                    # ✅ history 过滤：但永远保留 target
                    if args.filter_history and hset0:
                        ids0 = _filter_history_keep_target(ids0, hset0, int(tgt0))

                    # ✅ 最终兜底：确保 target 在列表里（一般 head_near 会自然包含）
                    if int(tgt0) not in ids0:
                        if len(ids0) >= int(args.topk):
                            ids0[-1] = int(tgt0)
                        else:
                            ids0.append(int(tgt0))

                    # 补齐长度
                    ids0 = _dedup_keep_order(ids0)
                    if len(ids0) >= int(args.topk):
                        ids0 = ids0[: int(args.topk)]
                    else:
                        while len(ids0) < int(args.topk):
                            ids0.append(1)

                    ex0["teacher_top_item_ids"] = ids0
                    fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

                    hit_cnt += (1 if bool(hit0) else 0)
                    hit_total += 1

                processed += len(batch_hist)
                batch_hist, batch_raw, batch_tgt, batch_hist_sets = [], [], [], []

                if args.log_every > 0 and processed % args.log_every == 0:
                    hit_rate = hit_cnt / max(1, hit_total)
                    outer_pbar.set_postfix({"processed": processed, "teacher_topk_hit": f"{hit_rate:.4f}"})

            outer_pbar.update(1)

        # flush last
        if batch_hist:
            input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
            user_repr = get_user_repr(sasrec, input_ids)

            if item_emb_weight.device != user_repr.device:
                item_emb_weight = item_emb_weight.to(user_repr.device)
                print("[WARN] moved item_emb_weight to device for speed.")

            tgt_ids = torch.tensor(batch_tgt, dtype=torch.long, device=user_repr.device)

            pool_ids, hit = mine_teacher_pool_streaming(
                user_repr=user_repr,
                target_ids=tgt_ids,
                item_emb_weight=item_emb_weight,
                topk=int(args.topk),
                chunk_size=int(args.chunk_size),
                score_dtype=score_dtype,
                pool_mode=str(args.pool_mode),
                head_k=int(args.head_k),
                near_above_k=int(args.near_above_k),
                near_below_k=int(args.near_below_k),
                show_chunk_pbar=bool(args.show_chunk_pbar),
            )

            pool_ids = pool_ids.detach().cpu().tolist()
            hit = hit.detach().cpu().tolist()

            for ex0, tgt0, ids0, hset0, hit0 in zip(batch_raw, batch_tgt, pool_ids, batch_hist_sets, hit):
                ids0 = [int(x) for x in ids0]

                if args.filter_history and hset0:
                    ids0 = _filter_history_keep_target(ids0, hset0, int(tgt0))

                if int(tgt0) not in ids0:
                    if len(ids0) >= int(args.topk):
                        ids0[-1] = int(tgt0)
                    else:
                        ids0.append(int(tgt0))

                ids0 = _dedup_keep_order(ids0)
                if len(ids0) >= int(args.topk):
                    ids0 = ids0[: int(args.topk)]
                else:
                    while len(ids0) < int(args.topk):
                        ids0.append(1)

                ex0["teacher_top_item_ids"] = ids0
                fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

                hit_cnt += (1 if bool(hit0) else 0)
                hit_total += 1

            processed += len(batch_hist)

        outer_pbar.close()

    hit_rate = hit_cnt / max(1, hit_total)
    print(f"✅ DONE. wrote teacher_top_item_ids to: {args.output_jsonl} (processed={processed})")
    print(f"📌 teacher_topk_hit_rate (target naturally in head topk) = {hit_rate:.6f}")
    print("   如果这个值长期接近 0：要么 teacher 太弱，要么 item_id 映射/序列方向不一致（强烈建议先排查）。")


if __name__ == "__main__":
    main()
