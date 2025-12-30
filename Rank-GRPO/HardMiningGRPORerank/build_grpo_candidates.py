# HardMiningGRPO/build_grpo_candidates.py
import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec

RULE = "只能从下面候选列表中选择一个，并且原样只输出一个地点名(类别)，不要解释。"


# -------------------------
# Fast JSON (optional)
# -------------------------
try:
    import orjson  # type: ignore

    def json_loads(b: bytes) -> Any:
        return orjson.loads(b)

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")

    ORJSON_OK = True
except Exception:
    ORJSON_OK = False

    def json_loads(b: bytes) -> Any:
        return json.loads(b.decode("utf-8"))

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


# -------------------------
# Utils
# -------------------------
def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s


def strip_rule(prompt: str) -> str:
    """Remove trailing RULE if exists (only strip once, at end-ish)."""
    if not prompt:
        return prompt
    p = prompt.rstrip()
    # try exact tail
    if p.endswith(RULE):
        p = p[: -len(RULE)].rstrip()
    return p


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str) -> Any:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# SASRec loader
# -------------------------
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
    obj = load_pickle(sasrec_pkl)
    n_items = int(obj["n_items"])

    # torch 2.6+: weights_only default True, force False
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

    # basic sanity
    if "item_emb.weight" in state_dict:
        ckpt_dim = int(state_dict["item_emb.weight"].shape[1])
        if ckpt_dim != int(embed_dim):
            raise ValueError(f"SASRec embed_dim mismatch: ckpt_dim={ckpt_dim} vs args.embed_dim={embed_dim}")
    if "pos_emb.weight" in state_dict:
        ckpt_len = int(state_dict["pos_emb.weight"].shape[0])
        if ckpt_len != int(max_len):
            raise ValueError(f"SASRec max_len mismatch: ckpt_max_len={ckpt_len} vs args.max_len={max_len}")

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

    print(
        f"[OK] loaded SASRec: n_items={n_items}, max_len={a.max_len}, dim={a.embed_dim}, "
        f"blocks={a.num_blocks}, heads={a.num_heads}, dropout={a.dropout}"
    )
    return sasrec, n_items, obj


# -------------------------
# Mapping: item_id -> namecat
# -------------------------
class ItemNamecatMapper:
    def __init__(self, sasrec_pkl_obj: Dict[str, Any], gmap_id2namecat: Dict[str, str]):
        self.gmap_id2namecat = gmap_id2namecat

        id2item = sasrec_pkl_obj.get("id2item", None)
        if id2item is None:
            raise KeyError("sasrec_dataset.pkl missing key: id2item")

        self.id2item = id2item  # could be dict or list

    def item_id_to_gmap(self, item_id: int) -> Optional[str]:
        if item_id is None:
            return None
        iid = int(item_id)
        try:
            if isinstance(self.id2item, dict):
                # dict keys could be int or str
                if iid in self.id2item:
                    return self.id2item[iid]
                if str(iid) in self.id2item:
                    return self.id2item[str(iid)]
                return None
            elif isinstance(self.id2item, list):
                if 0 <= iid < len(self.id2item):
                    return self.id2item[iid]
                return None
            else:
                return None
        except Exception:
            return None

    def item_id_to_namecat(self, item_id: int) -> Optional[str]:
        g = self.item_id_to_gmap(item_id)
        if not g:
            return None
        nc = self.gmap_id2namecat.get(g)
        if not nc:
            return None
        nc = norm_text(nc)
        if not nc or nc.startswith("<UNK_GMAP:"):
            return None
        return nc


# -------------------------
# Batch scoring
# -------------------------
@torch.no_grad()
def score_pool_batch(
    sasrec: Any,
    histories: List[List[int]],
    pools: List[List[int]],
    max_len: int,
    device: str,
) -> torch.Tensor:
    """
    histories: len=B, each <= max_len
    pools:     len=B, each length=P (must be same P)
    return scores: [B,P] on device
    """
    B = len(histories)
    P = len(pools[0])

    hist = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, h in enumerate(histories):
        h = [int(x) for x in (h or [])][-max_len:]
        if len(h) > 0:
            hist[i, -len(h) :] = torch.tensor(h, dtype=torch.long, device=device)

    cand = torch.tensor(pools, dtype=torch.long, device=device)  # [B,P]
    scores = sasrec.predict_candidates(hist, cand)  # [B,P]
    return scores


# -------------------------
# Candidate generation config
# -------------------------
@dataclass
class CandConfig:
    pool_size: int = 4096
    max_candidates: int = 50
    min_candidates: int = 10
    topk_fetch_factor: int = 4  # fetch = max_candidates * factor, then filter by mapping/uniq
    exclude_history: bool = True


def make_candidate_pool(
    rng: np.random.Generator,
    n_items: int,
    target_item_id: int,
    history_item_ids: List[int],
    pool_size: int,
    exclude_history: bool = True,
) -> List[int]:
    """
    Build a dedup pool of item_ids (length=pool_size), always include target.
    Sampling uses numpy chunks for speed.
    """
    target_item_id = int(target_item_id)
    hist_set = set(int(x) for x in (history_item_ids or [])) if exclude_history else set()

    pool = [target_item_id]
    used = set(pool)
    used.update(hist_set)
    # 0 reserved padding in SASRec, avoid 0
    while len(pool) < pool_size:
        need = pool_size - len(pool)
        # oversample to reduce python loop count
        chunk = int(need * 2 + 256)
        xs = rng.integers(1, n_items + 1, size=chunk, dtype=np.int64)
        for x in xs.tolist():
            if x not in used:
                used.add(x)
                pool.append(int(x))
                if len(pool) >= pool_size:
                    break
    return pool


def build_prompt_with_candidates(
    raw_prompt: str,
    candidates: List[str],
    rule: str = RULE,
) -> str:
    """
    Append candidate block near the end, keep RULE at the very end.
    """
    base = strip_rule(raw_prompt).rstrip()
    cand_lines = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    block = (
        "\n候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）：\n"
        f"{cand_lines}\n"
    )
    out = base + block + rule
    return out


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)

    ap.add_argument("--gmap_id2namecat", required=True)

    # SASRec arch (must match training)
    ap.add_argument("--sasrec_max_len", type=int, default=50)
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)

    # candidate config
    ap.add_argument("--pool_size", type=int, default=4096)
    ap.add_argument("--max_candidates", type=int, default=50)
    ap.add_argument("--min_candidates", type=int, default=10)
    ap.add_argument("--topk_fetch_factor", type=int, default=4)
    ap.add_argument("--exclude_history", action="store_true", help="exclude history items from candidate pool")

    # speed
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--write_buffer", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    # misc
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep_original_prompt", action="store_true")
    ap.add_argument("--keep_candidate_item_ids", action="store_true")
    ap.add_argument("--verbose_stats", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    seed_everything(int(args.seed))

    # perf toggles
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # load sasrec + pkl obj
    sasrec, n_items, pkl_obj = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )

    gmap_id2namecat = load_json(args.gmap_id2namecat)
    mapper = ItemNamecatMapper(pkl_obj, gmap_id2namecat)

    cfg = CandConfig(
        pool_size=int(args.pool_size),
        max_candidates=int(args.max_candidates),
        min_candidates=int(args.min_candidates),
        topk_fetch_factor=int(args.topk_fetch_factor),
        exclude_history=bool(args.exclude_history),
    )

    rng = np.random.default_rng(int(args.seed))

    # counters
    total = 0
    kept = 0
    dropped_bad_fields = 0
    dropped_no_target_map = 0
    dropped_too_few_cands = 0
    missing_cand_namecat = 0

    buf: List[str] = []

    def flush(fout):
        nonlocal buf
        if not buf:
            return
        fout.write("\n".join(buf) + "\n")
        buf = []

    topk_fetch = min(cfg.pool_size, max(cfg.max_candidates * cfg.topk_fetch_factor, cfg.max_candidates))
    write_buffer = max(200, int(args.write_buffer))
    bs = max(1, int(args.batch_size))

    # read lines in binary for speed (works with orjson)
    with open(args.in_jsonl, "rb") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        batch_recs = []
        for line in tqdm(fin, desc="build grpo candidates"):
            if args.limit > 0 and total >= int(args.limit):
                break
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json_loads(line)
            except Exception:
                dropped_bad_fields += 1
                continue

            # required fields
            if "prompt" not in obj or "history_item_ids" not in obj or "target_item_id" not in obj:
                dropped_bad_fields += 1
                continue

            batch_recs.append(obj)

            if len(batch_recs) < bs:
                continue

            # process one batch
            out_lines = process_batch(batch_recs, rng, sasrec, mapper, cfg, device, n_items, topk_fetch, args)
            for out_obj, status in out_lines:
                if status == "ok":
                    buf.append(json_dumps(out_obj))
                    kept += 1
                elif status == "no_target_map":
                    dropped_no_target_map += 1
                elif status == "too_few_cands":
                    dropped_too_few_cands += 1
                elif status == "bad_fields":
                    dropped_bad_fields += 1
                elif status == "missing_cand_map":
                    missing_cand_namecat += 1
                else:
                    dropped_bad_fields += 1

                if len(buf) >= write_buffer:
                    flush(fout)

            batch_recs = []

        # last batch
        if batch_recs:
            out_lines = process_batch(batch_recs, rng, sasrec, mapper, cfg, device, n_items, topk_fetch, args)
            for out_obj, status in out_lines:
                if status == "ok":
                    buf.append(json_dumps(out_obj))
                    kept += 1
                elif status == "no_target_map":
                    dropped_no_target_map += 1
                elif status == "too_few_cands":
                    dropped_too_few_cands += 1
                elif status == "bad_fields":
                    dropped_bad_fields += 1
                elif status == "missing_cand_map":
                    missing_cand_namecat += 1
                else:
                    dropped_bad_fields += 1

            flush(fout)
        else:
            flush(fout)

    print("========== DONE ==========")
    print("orjson:", ORJSON_OK)
    print("in :", args.in_jsonl)
    print("out:", args.out_jsonl)
    print(f"total={total}")
    print(f"kept={kept}")
    print(f"dropped_bad_fields={dropped_bad_fields}")
    print(f"dropped_no_target_map={dropped_no_target_map}")
    print(f"dropped_too_few_cands={dropped_too_few_cands}")
    print(f"missing_cand_namecat={missing_cand_namecat}")
    if args.verbose_stats:
        if total > 0:
            print(f"keep_rate={kept/total:.6f}")
            print(f"drop_rate={1.0-kept/total:.6f}")


def process_batch(
    batch_recs: List[Dict[str, Any]],
    rng: np.random.Generator,
    sasrec: Any,
    mapper: ItemNamecatMapper,
    cfg: CandConfig,
    device: str,
    n_items: int,
    topk_fetch: int,
    args: argparse.Namespace,
) -> List[Tuple[Dict[str, Any], str]]:
    """
    Return: list of (out_obj, status)
      status in {"ok","bad_fields","no_target_map","too_few_cands","missing_cand_map"}
    """
    B = len(batch_recs)

    prompts_raw: List[str] = []
    histories: List[List[int]] = []
    targets: List[int] = []
    target_namecats: List[str] = []

    # prepare
    for obj in batch_recs:
        try:
            p = str(obj["prompt"])
            hist = obj["history_item_ids"]
            tgt = int(obj["target_item_id"])
        except Exception:
            # placeholder; handled later
            prompts_raw.append("")
            histories.append([])
            targets.append(0)
            target_namecats.append("")
            continue

        hist = [int(x) for x in (hist or [])]
        tn = norm_text(obj.get("target_namecat", ""))
        if not tn:
            tn = mapper.item_id_to_namecat(tgt) or ""
        prompts_raw.append(p)
        histories.append(hist)
        targets.append(tgt)
        target_namecats.append(tn)

    # build pools
    pools: List[List[int]] = []
    valid_mask = [True] * B
    for i in range(B):
        if not prompts_raw[i] or not histories[i] or targets[i] <= 0:
            valid_mask[i] = False
            pools.append([1] * cfg.pool_size)
            continue
        if not target_namecats[i] or target_namecats[i].startswith("<UNK_GMAP:"):
            valid_mask[i] = False
            pools.append([1] * cfg.pool_size)
            continue

        pool = make_candidate_pool(
            rng=rng,
            n_items=n_items,
            target_item_id=targets[i],
            history_item_ids=histories[i],
            pool_size=cfg.pool_size,
            exclude_history=cfg.exclude_history,
        )
        pools.append(pool)

    # score batch (only if any valid)
    any_valid = any(valid_mask)
    scores = None
    if any_valid:
        scores = score_pool_batch(
            sasrec=sasrec,
            histories=histories,
            pools=pools,
            max_len=args.sasrec_max_len,
            device=device,
        )

    outs: List[Tuple[Dict[str, Any], str]] = []
    for i, obj in enumerate(batch_recs):
        if not valid_mask[i]:
            # distinguish bad fields vs no target map
            if not prompts_raw[i] or not histories[i] or targets[i] <= 0:
                outs.append(({}, "bad_fields"))
            else:
                outs.append(({}, "no_target_map"))
            continue

        tgt = int(targets[i])
        tgt_nc = target_namecats[i]
        raw_prompt = prompts_raw[i]
        hist = histories[i]
        pool = pools[i]

        # rank in pool
        row_scores = scores[i]  # type: ignore
        kk = min(topk_fetch, row_scores.numel())
        top_scores, top_idx = torch.topk(row_scores, k=kk, dim=0)
        top_item_ids = [pool[int(j)] for j in top_idx.tolist()]

        # map to namecat, keep unique
        cand_texts: List[str] = []
        cand_item_ids: List[int] = []
        used_text = set()

        # ensure target first
        if tgt_nc and tgt_nc not in used_text:
            used_text.add(tgt_nc)
            cand_texts.append(tgt_nc)
            cand_item_ids.append(tgt)

        for iid in top_item_ids:
            if len(cand_texts) >= cfg.max_candidates:
                break
            nc = mapper.item_id_to_namecat(int(iid))
            if not nc:
                continue
            if nc in used_text:
                continue
            used_text.add(nc)
            cand_texts.append(nc)
            cand_item_ids.append(int(iid))

        if len(cand_texts) < cfg.min_candidates:
            outs.append(({}, "too_few_cands"))
            continue

        new_prompt = build_prompt_with_candidates(raw_prompt, cand_texts, RULE)

        out_obj = {
            "prompt": new_prompt,
            "history_item_ids": hist,
            "target_item_id": tgt,
            "target_namecat": tgt_nc,
            "candidates_namecat": cand_texts,
        }
        if args.keep_original_prompt:
            out_obj["prompt_raw"] = raw_prompt
        if args.keep_candidate_item_ids:
            out_obj["candidates_item_ids"] = cand_item_ids

        outs.append((out_obj, "ok"))

    return outs


if __name__ == "__main__":
    main()
