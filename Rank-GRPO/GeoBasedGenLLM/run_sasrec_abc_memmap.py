import os
import re
import math
import gzip
import json
import pickle
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def iter_gz_jsonlines(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def safe_first_category(cat_field):
    if cat_field is None:
        return None
    if isinstance(cat_field, list):
        return cat_field[0] if len(cat_field) > 0 else None
    if isinstance(cat_field, str):
        return cat_field
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bucketize(distance_km: float, edges_km: List[float]) -> int:
    # return bucket id in [1..len(edges)+1], 0 reserved for padding.
    for i, e in enumerate(edges_km):
        if distance_km <= e:
            return i + 1
    return len(edges_km) + 1

def find_files(raw_dir: str, prefix: str, states_regex: Optional[str]):
    files = []
    for fn in os.listdir(raw_dir):
        if fn.startswith(prefix) and fn.endswith(".json.gz"):
            if states_regex is None or re.search(states_regex, fn):
                files.append(os.path.join(raw_dir, fn))
    files.sort()
    return files

# -------------------------
# Preprocess -> memmap arrays
# -------------------------
@dataclass
class PreprocessConfig:
    raw_dir: str
    out_dir: str
    geohash_precision: int = 6
    min_user_len: int = 5
    states_regex: Optional[str] = None
    max_users: Optional[int] = None
    verbose: bool = True
    dist_edges_km: Tuple[float, ...] = (0.2, 1.0, 5.0, 20.0)

def preprocess_to_memmap(cfg: PreprocessConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_pack_dir = os.path.join(cfg.out_dir, f"pack_p{cfg.geohash_precision}_dist")
    os.makedirs(out_pack_dir, exist_ok=True)

    meta_files = find_files(cfg.raw_dir, "meta-", cfg.states_regex)
    review_files = find_files(cfg.raw_dir, "review-", cfg.states_regex)

    if cfg.verbose:
        print("Meta files:", meta_files)
        print("Review files:", review_files)

    try:
        import pygeohash as pgh
    except Exception as e:
        raise RuntimeError("Please install pygeohash: pip install pygeohash") from e

    item2idx: Dict[str, int] = {"<pad>": 0}
    geo2idx: Dict[str, int]  = {"<pad>": 0}
    cat2idx: Dict[str, int]  = {"<pad>": 0}

    gmap2item: Dict[str, int] = {}
    gmap2geo: Dict[str, int]  = {}
    gmap2cat: Dict[str, int]  = {}
    gmap2latlon: Dict[str, Tuple[float, float]] = {}

    # 1) meta -> mappings
    for mp in meta_files:
        for obj in tqdm(iter_gz_jsonlines(mp), desc=f"Reading {os.path.basename(mp)}"):
            gid = obj.get("gmap_id")
            lat = obj.get("latitude")
            lon = obj.get("longitude")
            if not gid or lat is None or lon is None:
                continue

            if gid not in item2idx:
                item2idx[gid] = len(item2idx)
            gmap2item[gid] = item2idx[gid]

            code = pgh.encode(float(lat), float(lon), precision=cfg.geohash_precision)
            if code not in geo2idx:
                geo2idx[code] = len(geo2idx)
            gmap2geo[gid] = geo2idx[code]

            cat = safe_first_category(obj.get("category"))
            if cat is None:
                cat_idx = 0
            else:
                if cat not in cat2idx:
                    cat2idx[cat] = len(cat2idx)
                cat_idx = cat2idx[cat]
            gmap2cat[gid] = cat_idx

            gmap2latlon[gid] = (float(lat), float(lon))

    item_num = len(item2idx) - 1
    geo_num  = len(geo2idx) - 1
    cat_num  = len(cat2idx) - 1

    if cfg.verbose:
        print(f"Items={item_num}, Geos={geo_num}, Cats={cat_num}")

    # item_idx -> lat/lon arrays (for distance)
    item_lat = np.zeros((item_num + 1,), dtype=np.float32)
    item_lon = np.zeros((item_num + 1,), dtype=np.float32)
    for gid, idx in gmap2item.items():
        lat, lon = gmap2latlon.get(gid, (0.0, 0.0))
        item_lat[idx] = lat
        item_lon[idx] = lon

    # 2) read reviews -> user events (store compact)
    user_events: Dict[str, List[Tuple[int, int, int, int]]] = {}
    # (time, item, geo, cat)
    for rp in review_files:
        for obj in tqdm(iter_gz_jsonlines(rp), desc=f"Reading {os.path.basename(rp)}"):
            uid = obj.get("user_id")
            gid = obj.get("gmap_id")
            t = obj.get("time")
            if not uid or not gid or t is None:
                continue
            if gid not in gmap2item:
                continue
            if uid not in user_events:
                user_events[uid] = []
            user_events[uid].append((int(t), gmap2item[gid], gmap2geo[gid], gmap2cat[gid]))

    # 3) sort & filter & pack (avoid giant python nested list in saved artifacts)
    edges = list(cfg.dist_edges_km)
    dist_bucket_max = len(edges) + 1

    # First pass: build per-user lengths to allocate big arrays
    users = []
    lengths = []
    for uid, evs in user_events.items():
        if len(evs) < cfg.min_user_len:
            continue
        evs.sort(key=lambda x: x[0])
        users.append(uid)
        lengths.append(len(evs))

    if cfg.max_users is not None and len(users) > cfg.max_users:
        users = users[:cfg.max_users]
        lengths = lengths[:cfg.max_users]

    U = len(users)
    if cfg.verbose:
        print(f"Kept users: {U} (min_len={cfg.min_user_len})")

    offsets = np.zeros((U + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(np.array(lengths, dtype=np.int64))
    total = int(offsets[-1])

    # allocate big arrays
    items_all = np.zeros((total,), dtype=np.int32)
    geos_all  = np.zeros((total,), dtype=np.int32)
    cats_all  = np.zeros((total,), dtype=np.int32)
    dists_all = np.zeros((total,), dtype=np.int16)  # dist buckets small, int16 enough

    # Second pass: fill arrays
    for ui, uid in enumerate(tqdm(users, desc="Packing user sequences")):
        evs = user_events[uid]
        evs.sort(key=lambda x: x[0])
        start, end = offsets[ui], offsets[ui + 1]
        n = end - start

        items = [e[1] for e in evs]
        geos  = [e[2] for e in evs]
        cats  = [e[3] for e in evs]

        # distance buckets aligned to items (prev->cur)
        dists = [0]
        for t in range(1, len(items)):
            i1, i2 = items[t-1], items[t]
            lat1, lon1 = float(item_lat[i1]), float(item_lon[i1])
            lat2, lon2 = float(item_lat[i2]), float(item_lon[i2])
            if (lat1 == 0.0 and lon1 == 0.0) or (lat2 == 0.0 and lon2 == 0.0):
                dists.append(0)
            else:
                dk = haversine_km(lat1, lon1, lat2, lon2)
                dists.append(bucketize(dk, edges))

        items_all[start:end] = np.asarray(items, dtype=np.int32)
        geos_all[start:end]  = np.asarray(geos, dtype=np.int32)
        cats_all[start:end]  = np.asarray(cats, dtype=np.int32)
        dists_all[start:end] = np.asarray(dists, dtype=np.int16)

    # Save memmap-friendly .npy
    np.save(os.path.join(out_pack_dir, "offsets.npy"), offsets)
    np.save(os.path.join(out_pack_dir, "items.npy"), items_all)
    np.save(os.path.join(out_pack_dir, "geos.npy"), geos_all)
    np.save(os.path.join(out_pack_dir, "cats.npy"), cats_all)
    np.save(os.path.join(out_pack_dir, "dists.npy"), dists_all)

    meta = {
        "item2idx": item2idx,
        "geo2idx": geo2idx,
        "cat2idx": cat2idx,
        "geohash_precision": cfg.geohash_precision,
        "dist_edges_km": edges,
        "dist_bucket_max": dist_bucket_max,
        "num_users": U,
        "total_events": total,
        "pack_dir": out_pack_dir,
    }
    meta_path = os.path.join(out_pack_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    if cfg.verbose:
        print("Saved packed dataset to:", out_pack_dir)
        print("Meta:", meta_path)

    return out_pack_dir

# -------------------------
# Model: A+B+C
# -------------------------
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRecABC(nn.Module):
    def __init__(
        self,
        item_num: int,
        geo_num: int,
        cat_num: int,
        dist_bucket_max: int,
        max_len: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        item_drop_p: float = 0.0,    # B
        geo_aux_weight: float = 0.0, # C
    ):
        super().__init__()
        self.item_num = item_num
        self.geo_num = geo_num
        self.cat_num = cat_num
        self.dist_bucket_max = dist_bucket_max

        self.embed_dim = embed_dim
        self.item_drop_p = float(item_drop_p)
        self.geo_aux_weight = float(geo_aux_weight)

        self.item_emb = nn.Embedding(item_num + 1, embed_dim, padding_idx=0)
        self.geo_emb  = nn.Embedding(geo_num + 1, embed_dim, padding_idx=0)
        self.cat_emb  = nn.Embedding(cat_num + 1, embed_dim, padding_idx=0)
        self.dist_emb = nn.Embedding(dist_bucket_max + 1, embed_dim, padding_idx=0)

        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.emb_dropout = nn.Dropout(p=dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(embed_dim, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(embed_dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout))
            self.forward_layernorms.append(nn.LayerNorm(embed_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(embed_dim, dropout))

        self.register_buffer(
            "attn_mask_full",
            ~torch.tril(torch.ones((max_len, max_len), dtype=torch.bool)),
            persistent=False
        )

        self.geo_head = nn.Linear(embed_dim, geo_num + 1) if geo_aux_weight > 0 else None

    def _apply_item_dropout(self, item_emb: torch.Tensor, log_items: torch.Tensor):
        if (not self.training) or self.item_drop_p <= 0:
            return item_emb
        keep = (log_items != 0)
        drop = (torch.rand_like(log_items.float()) < self.item_drop_p) & keep
        return item_emb.masked_fill(drop.unsqueeze(-1), 0.0)

    def log2feats(self, log_items, log_geos, log_cats, log_dists):
        seqs = self.item_emb(log_items)
        seqs = self._apply_item_dropout(seqs, log_items)  # B
        seqs = seqs + self.geo_emb(log_geos) + self.cat_emb(log_cats) + self.dist_emb(log_dists)  # A (+geo/cat)

        seqs *= (self.embed_dim ** 0.5)

        B, L = log_items.size()
        positions = torch.arange(L, device=log_items.device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_items == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        attention_mask = self.attn_mask_full[:L, :L]

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # [L,B,H]
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)  # [B,L,H]
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        return self.last_layernorm(seqs)  # [B,L,H]

    def forward(self, log_items, pos_items, neg_items, log_geos, log_cats, log_dists):
        feats = self.log2feats(log_items, log_geos, log_cats, log_dists)  # [B,L,H]
        pos_embs = self.item_emb(pos_items)
        neg_embs = self.item_emb(neg_items)
        pos_logits = (feats * pos_embs).sum(dim=-1)
        neg_logits = (feats * neg_embs).sum(dim=-1)

        geo_logits = None
        if self.geo_head is not None:
            geo_logits = self.geo_head(feats)  # C

        return pos_logits, neg_logits, geo_logits

    @torch.no_grad()
    def predict_candidates(self, log_items, candidate_ids, log_geos, log_cats, log_dists):
        feats = self.log2feats(log_items, log_geos, log_cats, log_dists)
        user_repr = feats[:, -1, :]
        item_emb = self.item_emb(candidate_ids)
        return (user_repr.unsqueeze(1) * item_emb).sum(-1)

# -------------------------
# Dataset from packed arrays (memmap)
# -------------------------
class PackedSeq:
    def __init__(self, pack_dir: str):
        self.pack_dir = pack_dir
        self.offsets = np.load(os.path.join(pack_dir, "offsets.npy"), mmap_mode="r")
        self.items   = np.load(os.path.join(pack_dir, "items.npy"),   mmap_mode="r")
        self.geos    = np.load(os.path.join(pack_dir, "geos.npy"),    mmap_mode="r")
        self.cats    = np.load(os.path.join(pack_dir, "cats.npy"),    mmap_mode="r")
        self.dists   = np.load(os.path.join(pack_dir, "dists.npy"),   mmap_mode="r")
        self.num_users = len(self.offsets) - 1

    def get(self, u: int):
        s, e = int(self.offsets[u]), int(self.offsets[u+1])
        return (self.items[s:e], self.geos[s:e], self.cats[s:e], self.dists[s:e])

class TrainDatasetABC(Dataset):
    def __init__(self, packed: PackedSeq, max_len: int):
        self.packed = packed
        self.max_len = max_len

    def __len__(self):
        return self.packed.num_users

    def __getitem__(self, idx):
        items, geos, cats, dists = self.packed.get(idx)
        n = len(items)
        if n < 3:
            return self._pad_all()

        # reserve last2 for val/test
        items_tr = items[:-2]
        geos_tr  = geos[:-2]
        cats_tr  = cats[:-2]
        dists_tr = dists[:-2]

        if len(items_tr) <= 1:
            return self._pad_all()

        # log = [: -1], pos = [1:]
        log_items = items_tr[:-1]
        pos_items = items_tr[1:]

        log_geos  = geos_tr[:-1]
        log_cats  = cats_tr[:-1]
        log_dists = dists_tr[:-1]
        pos_geos  = geos_tr[1:]  # C target

        # take last max_len
        log_items = log_items[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        log_geos  = log_geos[-self.max_len:]
        log_cats  = log_cats[-self.max_len:]
        log_dists = log_dists[-self.max_len:]
        pos_geos  = pos_geos[-self.max_len:]

        pad = self.max_len - len(log_items)
        def lp(x, padv=0):
            if pad <= 0:
                return x
            return np.concatenate([np.full((pad,), padv, dtype=x.dtype), x], axis=0)

        sample = {
            "log_items": torch.from_numpy(lp(log_items.astype(np.int32))),
            "pos_items": torch.from_numpy(lp(pos_items.astype(np.int32))),
            "log_geos":  torch.from_numpy(lp(log_geos.astype(np.int32))),
            "log_cats":  torch.from_numpy(lp(log_cats.astype(np.int32))),
            "log_dists": torch.from_numpy(lp(log_dists.astype(np.int16))),
            "pos_geos":  torch.from_numpy(lp(pos_geos.astype(np.int32))),
        }
        return sample

    def _pad_all(self):
        L = self.max_len
        return {
            "log_items": torch.zeros((L,), dtype=torch.int32),
            "pos_items": torch.zeros((L,), dtype=torch.int32),
            "log_geos":  torch.zeros((L,), dtype=torch.int32),
            "log_cats":  torch.zeros((L,), dtype=torch.int32),
            "log_dists": torch.zeros((L,), dtype=torch.int16),
            "pos_geos":  torch.zeros((L,), dtype=torch.int32),
        }

class EvalDatasetABC(Dataset):
    def __init__(self, packed: PackedSeq, item_num: int, max_len: int, mode: str, num_neg: int):
        assert mode in ["val", "test"]
        self.packed = packed
        self.item_num = item_num
        self.max_len = max_len
        self.mode = mode
        self.num_neg = num_neg

        # For eval only, build seen-set index cheaply: store start/end and check by linear scan is too slow.
        # We'll do a small optimization: sample negatives by rejection on a hash set per user, created on the fly (only in __getitem__).
        # It is OK because eval runs once per epoch and batch large.

    def __len__(self):
        return self.packed.num_users

    def __getitem__(self, idx):
        items, geos, cats, dists = self.packed.get(idx)
        n = len(items)
        if n < 3:
            return self._pad_eval(0)

        if self.mode == "val":
            log_items = items[:-2]
            log_geos  = geos[:-2]
            log_cats  = cats[:-2]
            log_dists = dists[:-2]
            target = int(items[-2])
        else:
            log_items = items[:-1]
            log_geos  = geos[:-1]
            log_cats  = cats[:-1]
            log_dists = dists[:-1]
            target = int(items[-1])

        log_items = log_items[-self.max_len:]
        log_geos  = log_geos[-self.max_len:]
        log_cats  = log_cats[-self.max_len:]
        log_dists = log_dists[-self.max_len:]
        pad = self.max_len - len(log_items)

        def lp(x, padv=0):
            if pad <= 0:
                return x
            return np.concatenate([np.full((pad,), padv, dtype=x.dtype), x], axis=0)

        # candidates: [true] + negs
        seen = set(items.tolist())
        negs = []
        while len(negs) < self.num_neg:
            r = random.randint(1, self.item_num)
            if r not in seen:
                negs.append(r)
        candidates = [target] + negs

        return {
            "log_items": torch.from_numpy(lp(log_items.astype(np.int32))),
            "log_geos":  torch.from_numpy(lp(log_geos.astype(np.int32))),
            "log_cats":  torch.from_numpy(lp(log_cats.astype(np.int32))),
            "log_dists": torch.from_numpy(lp(log_dists.astype(np.int16))),
            "candidates": torch.tensor(candidates, dtype=torch.int32),
        }

    def _pad_eval(self, target: int):
        L = self.max_len
        return {
            "log_items": torch.zeros((L,), dtype=torch.int32),
            "log_geos":  torch.zeros((L,), dtype=torch.int32),
            "log_cats":  torch.zeros((L,), dtype=torch.int32),
            "log_dists": torch.zeros((L,), dtype=torch.int16),
            "candidates": torch.zeros((1 + self.num_neg,), dtype=torch.int32),
        }

def collate_train_fast(batch, item_num: int):
    log_items = torch.stack([b["log_items"] for b in batch], dim=0)  # int32
    pos_items = torch.stack([b["pos_items"] for b in batch], dim=0)
    log_geos  = torch.stack([b["log_geos"]  for b in batch], dim=0)
    log_cats  = torch.stack([b["log_cats"]  for b in batch], dim=0)
    log_dists = torch.stack([b["log_dists"] for b in batch], dim=0)
    pos_geos  = torch.stack([b["pos_geos"]  for b in batch], dim=0)

    B, L = pos_items.shape
    neg_items = torch.randint(1, item_num + 1, (B, L), dtype=torch.int32)

    return {
        "log_items": log_items, "pos_items": pos_items, "neg_items": neg_items,
        "log_geos": log_geos, "log_cats": log_cats, "log_dists": log_dists,
        "pos_geos": pos_geos
    }

def collate_eval(batch):
    return {
        "log_items": torch.stack([b["log_items"] for b in batch], dim=0),
        "log_geos":  torch.stack([b["log_geos"]  for b in batch], dim=0),
        "log_cats":  torch.stack([b["log_cats"]  for b in batch], dim=0),
        "log_dists": torch.stack([b["log_dists"] for b in batch], dim=0),
        "candidates": torch.stack([b["candidates"] for b in batch], dim=0),
    }

# -------------------------
# Metrics
# -------------------------
def recall_ndcg_at_k(scores: torch.Tensor, k: int):
    topk = torch.topk(scores, k=k, dim=1).indices
    hit = (topk == 0).any(dim=1).float()
    recall = hit.mean().item()
    ranks = torch.argsort(scores, dim=1, descending=True)
    pos = (ranks == 0).nonzero(as_tuple=False)
    rank = torch.full((scores.size(0),), fill_value=10**9, device=scores.device, dtype=torch.long)
    rank[pos[:, 0]] = pos[:, 1]
    ndcg = torch.where(rank < k, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(rank, dtype=torch.float)).mean().item()
    return recall, ndcg

# -------------------------
# Train
# -------------------------
def train_abc(pack_dir: str, save_dir: str, device: str,
              embed_dim: int, num_blocks: int, num_heads: int, dropout: float,
              max_len: int, lr: float, weight_decay: float,
              batch_size: int, epochs: int,
              num_workers: int, pin_memory: int, prefetch_factor: int, persistent_workers: int,
              amp: int, eval_k: int, eval_num_neg: int, eval_every: int,
              drop_p: float, geo_aux_w: float, seed: int):

    # speed knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    meta_path = os.path.join(pack_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    item_num = len(meta["item2idx"]) - 1
    geo_num  = len(meta["geo2idx"]) - 1
    cat_num  = len(meta["cat2idx"]) - 1
    dist_bucket_max = int(meta["dist_bucket_max"])

    packed = PackedSeq(pack_dir)

    print(f"Loaded pack_dir={pack_dir} users={packed.num_users} item_num={item_num} geo_num={geo_num} cat_num={cat_num} dist_max={dist_bucket_max}")

    train_ds = TrainDatasetABC(packed, max_len=max_len)
    val_ds   = EvalDatasetABC(packed, item_num=item_num, max_len=max_len, mode="val",  num_neg=eval_num_neg)
    test_ds  = EvalDatasetABC(packed, item_num=item_num, max_len=max_len, mode="test", num_neg=eval_num_neg)

    use_persistent = bool(persistent_workers) and (num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent,
        collate_fn=lambda b: collate_train_fast(b, item_num),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # eval 用 0 worker，避免再吃一波 RAM
        pin_memory=bool(pin_memory),
        collate_fn=collate_eval,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=collate_eval,
    )

    def run_one(exp_name: str, item_drop: float, aux_w: float):
        print(f"\n===== {exp_name} ===== A(dist)=ON B(drop)={item_drop} C(aux_w)={aux_w}")
        model = SASRecABC(
            item_num=item_num, geo_num=geo_num, cat_num=cat_num, dist_bucket_max=dist_bucket_max,
            max_len=max_len, embed_dim=embed_dim,
            num_blocks=num_blocks, num_heads=num_heads, dropout=dropout,
            item_drop_p=item_drop, geo_aux_weight=aux_w
        ).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        ce  = nn.CrossEntropyLoss(ignore_index=0)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))

        best_val = -1.0
        best_path = os.path.join(save_dir, f"{exp_name}_best.pt")

        def to_dev(x, dtype=torch.long):
            # indices on CUDA 更稳用 long
            return x.to(device=device, non_blocking=True).to(dtype=dtype)

        @torch.no_grad()
        def eval_loop(loader):
            model.eval()
            all_rec, all_ndcg, n = 0.0, 0.0, 0
            for batch in loader:
                log_items = to_dev(batch["log_items"])
                log_geos  = to_dev(batch["log_geos"])
                log_cats  = to_dev(batch["log_cats"])
                log_dists = to_dev(batch["log_dists"])
                cands     = to_dev(batch["candidates"])
                scores = model.predict_candidates(log_items, cands, log_geos, log_cats, log_dists)
                rec, ndcg = recall_ndcg_at_k(scores, eval_k)
                bs = scores.size(0)
                all_rec += rec * bs
                all_ndcg += ndcg * bs
                n += bs
            return all_rec / max(n, 1), all_ndcg / max(n, 1)

        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                log_items = to_dev(batch["log_items"])
                pos_items = to_dev(batch["pos_items"])
                neg_items = to_dev(batch["neg_items"])
                log_geos  = to_dev(batch["log_geos"])
                log_cats  = to_dev(batch["log_cats"])
                log_dists = to_dev(batch["log_dists"])
                pos_geos  = to_dev(batch["pos_geos"])

                mask = (pos_items != 0).float()

                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(amp)):
                    pos_logits, neg_logits, geo_logits = model(log_items, pos_items, neg_items, log_geos, log_cats, log_dists)
                    pos_loss = bce(pos_logits, torch.ones_like(pos_logits)) * mask
                    neg_loss = bce(neg_logits, torch.zeros_like(neg_logits)) * mask
                    rec_loss = (pos_loss.sum() + neg_loss.sum()) / (mask.sum() + 1e-8)
                    loss = rec_loss
                    if aux_w > 0 and geo_logits is not None:
                        geo_loss = ce(geo_logits.reshape(-1, geo_logits.size(-1)), pos_geos.reshape(-1))
                        loss = loss + aux_w * geo_loss

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                global_step += 1
                if global_step % 200 == 0:
                    print(f"[{exp_name}] epoch={epoch} step={global_step} loss={loss.item():.4f}")

            if epoch % eval_every != 0:
                continue

            val_rec, val_ndcg = eval_loop(val_loader)
            print(f"[{exp_name}] Epoch {epoch} VAL  Recall@{eval_k}={val_rec:.4f} NDCG@{eval_k}={val_ndcg:.4f}")

            if val_ndcg > best_val:
                best_val = val_ndcg
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_ndcg": best_val}, best_path)
                print(f"[{exp_name}] Saved best: {best_path}")

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_rec, test_ndcg = eval_loop(test_loader)
        print(f"[{exp_name}] TEST Recall@{eval_k}={test_rec:.4f} NDCG@{eval_k}={test_ndcg:.4f} (best_val_ndcg={best_val:.4f})")
        return (best_val, test_rec, test_ndcg, best_path)

    results = []
    results.append(("ABC_A_DIST",) + run_one("ABC_A_DIST", item_drop=0.0,    aux_w=0.0))
    results.append(("ABC_AB",)     + run_one("ABC_AB",     item_drop=drop_p, aux_w=0.0))
    results.append(("ABC_ABC",)    + run_one("ABC_ABC",    item_drop=drop_p, aux_w=geo_aux_w))

    print("\n===== Summary =====")
    for r in results:
        name, best_val, test_rec, test_ndcg, path = r
        print(f"{name}: best_val_ndcg={best_val:.4f}, test_ndcg={test_ndcg:.4f}, test_rec={test_rec:.4f}, ckpt={path}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preprocess_memmap")
    p1.add_argument("--raw_dir", type=str, default="/workspace/data/GoogleRAW")
    p1.add_argument("--out_dir", type=str, default="/workspace/data/GooglePROC")
    p1.add_argument("--geohash_precision", type=int, default=6)
    p1.add_argument("--min_user_len", type=int, default=5)
    p1.add_argument("--states_regex", type=str, default=None)
    p1.add_argument("--max_users", type=int, default=None)
    p1.add_argument("--dist_edges_km", type=str, default="0.2,1,5,20")
    p1.add_argument("--quiet", action="store_true")

    p2 = sub.add_parser("train_abc_memmap")
    p2.add_argument("--pack_dir", type=str, required=True)
    p2.add_argument("--save_dir", type=str, default="/workspace/checkpoints_sasrec_abc_memmap")
    p2.add_argument("--device", type=str, default="cuda")

    p2.add_argument("--embed_dim", type=int, default=256)
    p2.add_argument("--num_blocks", type=int, default=3)
    p2.add_argument("--num_heads", type=int, default=8)
    p2.add_argument("--dropout", type=float, default=0.2)
    p2.add_argument("--max_len", type=int, default=100)

    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--weight_decay", type=float, default=0.0)

    p2.add_argument("--batch_size", type=int, default=2048)
    p2.add_argument("--epochs", type=int, default=5)

    # dataloader safe defaults
    p2.add_argument("--num_workers", type=int, default=2)
    p2.add_argument("--pin_memory", type=int, default=0)
    p2.add_argument("--prefetch_factor", type=int, default=2)
    p2.add_argument("--persistent_workers", type=int, default=0)

    p2.add_argument("--amp", type=int, default=1)
    p2.add_argument("--eval_k", type=int, default=10)
    p2.add_argument("--eval_num_neg", type=int, default=50)   # 默认更省
    p2.add_argument("--eval_every", type=int, default=2)      # 默认降频更快

    p2.add_argument("--drop_p", type=float, default=0.1)
    p2.add_argument("--geo_aux_w", type=float, default=0.2)
    p2.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    if args.cmd == "preprocess_memmap":
        edges = tuple(float(x) for x in args.dist_edges_km.split(",") if x.strip())
        cfg = PreprocessConfig(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            geohash_precision=args.geohash_precision,
            min_user_len=args.min_user_len,
            states_regex=args.states_regex,
            max_users=args.max_users,
            verbose=(not args.quiet),
            dist_edges_km=edges
        )
        preprocess_to_memmap(cfg)

    elif args.cmd == "train_abc_memmap":
        train_abc(
            pack_dir=args.pack_dir,
            save_dir=args.save_dir,
            device=args.device,
            embed_dim=args.embed_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_len=args.max_len,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            amp=args.amp,
            eval_k=args.eval_k,
            eval_num_neg=args.eval_num_neg,
            eval_every=args.eval_every,
            drop_p=args.drop_p,
            geo_aux_w=args.geo_aux_w,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
