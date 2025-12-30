# HardMiningGRPO/build_teacher_topk.py
import os
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional

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
    state_dict = None
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state_dict = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state_dict = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state_dict = ckpt_obj["model"]
        else:
            # 有的 ckpt 直接就是 state_dict
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
    feats = sasrec.log2feats(input_ids)     # [B,L,H]
    return feats[:, -1, :]                  # [B,H]


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
def topk_streaming_all_items(
    user_repr: torch.Tensor,          # [B,H] on device
    item_emb_weight: torch.Tensor,    # [N+1,H] on same device (or CPU if you want, but then会慢/需要搬运)
    k: int,
    chunk_size: int,
    score_dtype: torch.dtype,
    show_chunk_pbar: bool = True,
) -> torch.LongTensor:
    """
    全库 streaming topk（inner product）
    - 跳过 item_id=0 (padding)
    - 返回 [B,k] item_id
    """
    device = user_repr.device
    B, H = user_repr.shape
    N = item_emb_weight.size(0) - 1
    if N <= 0:
        return torch.zeros((B, k), device=device, dtype=torch.long)

    # 用 float32 存 best_scores，稳定一些
    best_scores = torch.full((B, k), -1e9, device=device, dtype=torch.float32)
    best_ids = torch.zeros((B, k), device=device, dtype=torch.long)

    # user_repr cast 一次即可
    u = user_repr.to(dtype=score_dtype)

    total_chunks = math.ceil(N / chunk_size)
    chunk_iter = range(total_chunks)

    pbar = None
    if show_chunk_pbar:
        pbar = tqdm(chunk_iter, desc="scan item chunks", leave=False, dynamic_ncols=True)
    else:
        pbar = chunk_iter

    for ci in pbar:
        start = 1 + ci * chunk_size
        end = min(N + 1, start + chunk_size)  # exclusive
        emb = item_emb_weight[start:end].to(dtype=score_dtype)  # [C,H]

        # scores: [B,C]
        # 注意：matmul 输出 dtype 可能是 fp16/bf16，topk 用 fp32 更稳
        scores = torch.matmul(u, emb.t())
        scores_f = scores.float()

        kk = min(k, scores_f.size(1))
        sc, idx = torch.topk(scores_f, k=kk, dim=1)
        ids = idx + start  # map to item_id

        # merge keep topk
        merged_scores = torch.cat([best_scores, sc], dim=1)
        merged_ids = torch.cat([best_ids, ids], dim=1)

        new_scores, new_idx = torch.topk(merged_scores, k=k, dim=1)
        best_scores = new_scores
        best_ids = merged_ids.gather(1, new_idx)

        # 让 chunk pbar 更直观一些
        if show_chunk_pbar and ci % 5 == 0:
            # 只显示一下当前 chunk 范围（避免太频繁刷新）
            pbar.set_postfix({"items": f"{start}-{end-1}"})

    return best_ids


def ensure_target_in_topk(ids: List[int], target_id: int) -> List[int]:
    if target_id in ids:
        return ids
    if not ids:
        return [int(target_id)]
    ids[-1] = int(target_id)
    return ids


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

    # 让 item embedding 放哪里（你内存很大：CPU也行，但会慢很多；推荐 GPU）
    ap.add_argument("--item_emb_on_gpu", action=argparse.BooleanOptionalAction, default=True)

    # 计算分数时的 dtype（默认 fp16 更快/省显存）
    ap.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)
    ap.add_argument("--sasrec_max_len", type=int, default=50)

    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--count_total", action=argparse.BooleanOptionalAction, default=True,
                    help="先扫一遍文件统计行数，让总进度条更准确（会多一次顺序读）")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    if os.path.exists(args.output_jsonl) and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} exists. Use --overwrite to replace it.")

    device = args.device
    score_dtype = _parse_score_dtype(args.score_dtype)

    # 加载 SASRec
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

    # item embedding：你内存很大，所以可以常驻；推荐常驻 GPU（更快）
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

    processed = 0

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

            hist_pad = pad_left([int(x) for x in hist], args.sasrec_max_len, pad=0)

            batch_hist.append(hist_pad)
            batch_raw.append(ex)
            batch_tgt.append(int(tgt))

            if len(batch_hist) >= args.batch_size:
                # run batch
                input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
                user_repr = get_user_repr(sasrec, input_ids)  # [B,H]

                # 如果 item_emb 在 CPU，但 device 是 cuda，需要每 chunk 搬运，会慢；你说内存够但追求速度建议 item_emb_on_gpu
                # 这里统一保证在同一 device 上算
                if item_emb_weight.device != user_repr.device:
                    # CPU -> GPU 全量搬运一次（需要显存能容纳）
                    item_emb_weight = item_emb_weight.to(user_repr.device)
                    print("[WARN] moved item_emb_weight to device for speed.")

                top_ids = topk_streaming_all_items(
                    user_repr=user_repr,
                    item_emb_weight=item_emb_weight,
                    k=int(args.topk),
                    chunk_size=int(args.chunk_size),
                    score_dtype=score_dtype,
                    show_chunk_pbar=True,
                ).detach().cpu().tolist()

                for ex0, tgt0, ids0 in zip(batch_raw, batch_tgt, top_ids):
                    ids0 = [int(x) for x in ids0]
                    ids0 = ensure_target_in_topk(ids0, int(tgt0))
                    ex0["teacher_top_item_ids"] = ids0
                    fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

                processed += len(batch_hist)
                batch_hist, batch_raw, batch_tgt = [], [], []

                if args.log_every > 0 and processed % args.log_every == 0:
                    outer_pbar.set_postfix({"processed": processed})

            outer_pbar.update(1)

        # flush last batch
        if batch_hist:
            input_ids = torch.tensor(batch_hist, dtype=torch.long, device=device)
            user_repr = get_user_repr(sasrec, input_ids)

            if item_emb_weight.device != user_repr.device:
                item_emb_weight = item_emb_weight.to(user_repr.device)
                print("[WARN] moved item_emb_weight to device for speed.")

            top_ids = topk_streaming_all_items(
                user_repr=user_repr,
                item_emb_weight=item_emb_weight,
                k=int(args.topk),
                chunk_size=int(args.chunk_size),
                score_dtype=score_dtype,
                show_chunk_pbar=True,
            ).detach().cpu().tolist()

            for ex0, tgt0, ids0 in zip(batch_raw, batch_tgt, top_ids):
                ids0 = [int(x) for x in ids0]
                ids0 = ensure_target_in_topk(ids0, int(tgt0))
                ex0["teacher_top_item_ids"] = ids0
                fout.write(json.dumps(ex0, ensure_ascii=False) + "\n")

            processed += len(batch_hist)

        outer_pbar.close()

    print(f"✅ DONE. wrote teacher_top_item_ids to: {args.output_jsonl} (processed={processed})")


if __name__ == "__main__":
    main()
