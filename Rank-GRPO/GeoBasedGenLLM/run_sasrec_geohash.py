import os
import re
import math
import gzip
import json
import pickle
import random
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def iter_gz_jsonlines(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def safe_first_category(cat_field):
    # meta["category"] may be list or str or None
    if cat_field is None:
        return None
    if isinstance(cat_field, list):
        return cat_field[0] if len(cat_field) > 0 else None
    if isinstance(cat_field, str):
        return cat_field
    return None

# Haversine distance (optional for later analysis)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# -------------------------
# Preprocess
# -------------------------
@dataclass
class PreprocessConfig:
    raw_dir: str
    out_dir: str
    geohash_precision: int = 6
    min_user_len: int = 5
    states_regex: Optional[str] = None  # e.g. "New_York|California"
    max_users: Optional[int] = None     # for quick debug
    verbose: bool = True

def find_files(raw_dir: str, prefix: str, states_regex: Optional[str]):
    files = []
    for fn in os.listdir(raw_dir):
        if fn.startswith(prefix) and fn.endswith(".json.gz"):
            if states_regex is None or re.search(states_regex, fn):
                files.append(os.path.join(raw_dir, fn))
    files.sort()
    return files

def preprocess_google_maps(cfg: PreprocessConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    meta_files = find_files(cfg.raw_dir, "meta-", cfg.states_regex)
    review_files = find_files(cfg.raw_dir, "review-", cfg.states_regex)

    if cfg.verbose:
        print("Meta files:", meta_files)
        print("Review files:", review_files)

    # GeoHash encoder
    try:
        import pygeohash as pgh
    except Exception as e:
        raise RuntimeError("Please install pygeohash: pip install pygeohash") from e

    # Build mappings from meta
    item2idx: Dict[str, int] = {"<pad>": 0}
    geo2idx: Dict[str, int]  = {"<pad>": 0}
    cat2idx: Dict[str, int]  = {"<pad>": 0}

    gmap2item: Dict[str, int] = {}
    gmap2geo: Dict[str, int]  = {}
    gmap2cat: Dict[str, int]  = {}

    # If you want later: gmap2latlon
    gmap2latlon: Dict[str, Tuple[float, float]] = {}

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

            code = pgh.encode(lat, lon, precision=cfg.geohash_precision)
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

            gmap2latlon[gid] = (lat, lon)

    if cfg.verbose:
        print(f"Items: {len(item2idx)-1}, Geos: {len(geo2idx)-1}, Cats: {len(cat2idx)-1}")

    # Read reviews and build user events
    user_events: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    # tuple: (time, item_idx, geo_idx, cat_idx)
    for rp in review_files:
        for obj in tqdm(iter_gz_jsonlines(rp), desc=f"Reading {os.path.basename(rp)}"):
            uid = obj.get("user_id")
            gid = obj.get("gmap_id")
            t = obj.get("time")
            if not uid or not gid or t is None:
                continue
            if gid not in gmap2item:
                continue
            user_events[uid].append((int(t), gmap2item[gid], gmap2geo[gid], gmap2cat[gid]))

    # Sort & filter by length
    user_seqs = []
    for uid, evs in user_events.items():
        if len(evs) < cfg.min_user_len:
            continue
        evs.sort(key=lambda x: x[0])
        items = [e[1] for e in evs]
        geos  = [e[2] for e in evs]
        cats  = [e[3] for e in evs]
        user_seqs.append((uid, items, geos, cats))

    # Optionally limit users (debug)
    if cfg.max_users is not None and len(user_seqs) > cfg.max_users:
        user_seqs = user_seqs[:cfg.max_users]

    if cfg.verbose:
        print(f"Kept users: {len(user_seqs)} (min_len={cfg.min_user_len})")

    # Save artifacts
    artifacts = {
        "item2idx": item2idx,
        "geo2idx": geo2idx,
        "cat2idx": cat2idx,
        "user_seqs": user_seqs,  # list of (uid, items, geos, cats)
        "geohash_precision": cfg.geohash_precision,
    }

    out_path = os.path.join(cfg.out_dir, f"processed_p{cfg.geohash_precision}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    if cfg.verbose:
        print("Saved:", out_path)
    return out_path

# -------------------------
# SASRec Model (Base / +Geo / +Geo+Cat)
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

class SASRec(nn.Module):
    """
    Supports:
      - base: use only item_emb
      - geo:  item_emb + geo_emb
      - cat:  item_emb + geo_emb + cat_emb (depending on flags)
    """
    def __init__(self, item_num: int, args, geo_num: int = 0, cat_num: int = 0, use_geo: bool = False, use_cat: bool = False):
        super().__init__()
        self.item_num = item_num
        self.dev = args.device
        self.embed_dim = args.embed_dim
        self.use_geo = use_geo
        self.use_cat = use_cat

        self.item_emb = nn.Embedding(self.item_num + 1, args.embed_dim, padding_idx=0)
        self.geo_emb  = nn.Embedding(geo_num + 1, args.embed_dim, padding_idx=0) if use_geo else None
        self.cat_emb  = nn.Embedding(cat_num + 1, args.embed_dim, padding_idx=0) if use_cat else None

        self.pos_emb = nn.Embedding(args.max_len, args.embed_dim)
        self.emb_dropout = nn.Dropout(p=args.dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.embed_dim, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.embed_dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(args.embed_dim, args.num_heads, args.dropout))
            self.forward_layernorms.append(nn.LayerNorm(args.embed_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.embed_dim, args.dropout))

    def log2feats(self, log_items, log_geos=None, log_cats=None):
        seqs = self.item_emb(log_items)

        if self.use_geo:
            assert log_geos is not None
            seqs = seqs + self.geo_emb(log_geos)

        if self.use_cat:
            assert log_cats is not None
            seqs = seqs + self.cat_emb(log_cats)

        seqs *= (self.embed_dim ** 0.5)

        B, L = log_items.size()
        positions = torch.arange(L, device=log_items.device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_items == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        attention_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=log_items.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # [L,B,H]
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)  # [B,L,H]

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        return self.last_layernorm(seqs)

    def forward(self, log_items, pos_items, neg_items, log_geos=None, log_cats=None):
        log_feats = self.log2feats(log_items, log_geos, log_cats)  # [B,L,H]

        pos_embs = self.item_emb(pos_items)
        neg_embs = self.item_emb(neg_items)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [B,L]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [B,L]
        return pos_logits, neg_logits

    @torch.no_grad()
    def predict_candidates(self, log_items, candidate_ids, log_geos=None, log_cats=None):
        feats = self.log2feats(log_items, log_geos, log_cats)   # [B,L,H]
        user_repr = feats[:, -1, :]                             # [B,H]
        item_emb = self.item_emb(candidate_ids)                 # [B,C,H]
        scores = (user_repr.unsqueeze(1) * item_emb).sum(-1)    # [B,C]
        return scores

# -------------------------
# Dataset (Train)
# -------------------------
class SASRecTrainDataset(Dataset):
    """
    For each user sequence, create one training instance by taking last max_len part.
    Then within that window, we produce:
      log_items: [L] items up to t-1
      pos_items: [L] next items at each position
      neg_items: [L] negative sampled items
    """
    def __init__(self, user_seqs, item_num, max_len, use_geo=False, use_cat=False):
        self.user_seqs = user_seqs
        self.item_num = item_num
        self.max_len = max_len
        self.use_geo = use_geo
        self.use_cat = use_cat

        # Build user history set for negative sampling
        self.user_hist = []
        for (_, items, _, _) in user_seqs:
            self.user_hist.append(set(items))

    def __len__(self):
        return len(self.user_seqs)

    def _sample_neg(self, hist_set):
        while True:
            neg = random.randint(1, self.item_num)
            if neg not in hist_set:
                return neg

    def __getitem__(self, idx):
        uid, items, geos, cats = self.user_seqs[idx]
        hist = self.user_hist[idx]

        # Use the training split: items[:-2] (reserve last2 for val/test)
        if len(items) < 3:
            # should have been filtered earlier
            items_tr = items
            geos_tr = geos
            cats_tr = cats
        else:
            items_tr = items[:-2]
            geos_tr  = geos[:-2]
            cats_tr  = cats[:-2]

        # We need pairs (log -> pos). For length n, there are n-1 training transitions.
        # Construct within last max_len:
        # log_items = [0..] + items_tr[:-1]
        # pos_items = [0..] + items_tr[1:]
        n = len(items_tr)
        if n <= 1:
            # no transitions
            log_items = [0] * self.max_len
            pos_items = [0] * self.max_len
            neg_items = [0] * self.max_len
            log_geos = [0] * self.max_len
            log_cats = [0] * self.max_len
            return self._pack(log_items, pos_items, neg_items, log_geos, log_cats)

        log_seq = items_tr[:-1]
        pos_seq = items_tr[1:]
        log_geo_seq = geos_tr[:-1]
        log_cat_seq = cats_tr[:-1]

        # take last max_len
        log_seq = log_seq[-self.max_len:]
        pos_seq = pos_seq[-self.max_len:]
        log_geo_seq = log_geo_seq[-self.max_len:]
        log_cat_seq = log_cat_seq[-self.max_len:]

        # left pad
        pad_len = self.max_len - len(log_seq)
        log_items = [0]*pad_len + log_seq
        pos_items = [0]*pad_len + pos_seq

        # neg sampling per position (only where pos != 0)
        neg_items = [0]*pad_len + [self._sample_neg(hist) for _ in range(len(pos_seq))]

        log_geos = [0]*pad_len + log_geo_seq
        log_cats = [0]*pad_len + log_cat_seq

        return self._pack(log_items, pos_items, neg_items, log_geos, log_cats)

    def _pack(self, log_items, pos_items, neg_items, log_geos, log_cats):
        sample = {
            "log_items": torch.LongTensor(log_items),
            "pos_items": torch.LongTensor(pos_items),
            "neg_items": torch.LongTensor(neg_items),
        }
        if self.use_geo:
            sample["log_geos"] = torch.LongTensor(log_geos)
        if self.use_cat:
            sample["log_cats"] = torch.LongTensor(log_cats)
        return sample

# -------------------------
# Dataset (Eval)
# -------------------------
class SASRecEvalDataset(Dataset):
    """
    For each user:
      train: items[:-2]
      val_target: items[-2]
      test_target: items[-1]

    Build log sequence for val: items[:-2]
    Build log sequence for test: items[:-1]
    We'll evaluate both val and test via two datasets, or one dataset with mode.
    """
    def __init__(self, user_seqs, item_num, max_len, mode="val", num_neg=100, use_geo=False, use_cat=False):
        assert mode in ["val", "test"]
        self.user_seqs = user_seqs
        self.item_num = item_num
        self.max_len = max_len
        self.mode = mode
        self.num_neg = num_neg
        self.use_geo = use_geo
        self.use_cat = use_cat

        self.user_hist = []
        for (_, items, _, _) in user_seqs:
            self.user_hist.append(set(items))

    def __len__(self):
        return len(self.user_seqs)

    def _sample_negs(self, hist_set, k):
        negs = []
        while len(negs) < k:
            neg = random.randint(1, self.item_num)
            if neg not in hist_set:
                negs.append(neg)
        return negs

    def __getitem__(self, idx):
        uid, items, geos, cats = self.user_seqs[idx]
        hist = self.user_hist[idx]

        if len(items) < 3:
            # skip-ish, but keep shape valid
            log_items = [0]*self.max_len
            log_geos  = [0]*self.max_len
            log_cats  = [0]*self.max_len
            target = 0
        else:
            if self.mode == "val":
                # log: items[:-2], target: items[-2]
                log_seq_items = items[:-2]
                log_seq_geos  = geos[:-2]
                log_seq_cats  = cats[:-2]
                target = items[-2]
            else:
                # test
                log_seq_items = items[:-1]
                log_seq_geos  = geos[:-1]
                log_seq_cats  = cats[:-1]
                target = items[-1]

            log_seq_items = log_seq_items[-self.max_len:]
            log_seq_geos  = log_seq_geos[-self.max_len:]
            log_seq_cats  = log_seq_cats[-self.max_len:]

            pad_len = self.max_len - len(log_seq_items)
            log_items = [0]*pad_len + log_seq_items
            log_geos  = [0]*pad_len + log_seq_geos
            log_cats  = [0]*pad_len + log_seq_cats

        # candidates: [true] + negatives
        if target == 0:
            candidates = [0] + [0]*self.num_neg
        else:
            negs = self._sample_negs(hist, self.num_neg)
            candidates = [target] + negs

        sample = {
            "log_items": torch.LongTensor(log_items),
            "candidates": torch.LongTensor(candidates),
            "target_index": torch.LongTensor([0]),  # true item is at index 0
        }
        if self.use_geo:
            sample["log_geos"] = torch.LongTensor(log_geos)
        if self.use_cat:
            sample["log_cats"] = torch.LongTensor(log_cats)
        return sample

# -------------------------
# Metrics
# -------------------------
def recall_ndcg_at_k(scores: torch.Tensor, k: int):
    """
    scores: [B, C] where true item is at index 0 in each row
    """
    # rank descending
    topk = torch.topk(scores, k=k, dim=1).indices  # [B,k]
    # hit if 0 is in topk
    hit = (topk == 0).any(dim=1).float()           # [B]
    recall = hit.mean().item()

    # ndcg: if true item rank = r (0-based), ndcg=1/log2(r+2)
    # find rank of index 0
    # argsort descending
    ranks = torch.argsort(scores, dim=1, descending=True)
    # position of 0 in each row
    pos = (ranks == 0).nonzero(as_tuple=False)  # [B,2], columns: row_idx, rank
    # gather rank per row
    rank = torch.full((scores.size(0),), fill_value=10**9, device=scores.device, dtype=torch.long)
    rank[pos[:, 0]] = pos[:, 1]
    # Only count if rank < k? Usually NDCG@k is 0 if outside topk
    ndcg = torch.where(rank < k, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(rank, dtype=torch.float)).mean().item()
    return recall, ndcg

# -------------------------
# Train / Eval
# -------------------------
@dataclass
class TrainArgs:
    device: str = "cuda"
    embed_dim: int = 128
    num_blocks: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    max_len: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    epochs: int = 50
    num_workers: int = 8
    amp: bool = True
    log_every: int = 200
    eval_k: int = 10
    eval_num_neg: int = 100
    seed: int = 2026
    save_dir: str = "./checkpoints"

def train_one_experiment(
    exp_name: str,
    artifacts_path: str,
    args: TrainArgs,
    use_geo: bool,
    use_cat: bool
):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    user_seqs = artifacts["user_seqs"]
    item2idx = artifacts["item2idx"]
    geo2idx = artifacts["geo2idx"]
    cat2idx = artifacts["cat2idx"]

    item_num = len(item2idx) - 1
    geo_num  = len(geo2idx) - 1
    cat_num  = len(cat2idx) - 1

    print(f"\n===== Experiment: {exp_name} =====")
    print(f"Users={len(user_seqs)}, Items={item_num}, Geos={geo_num}, Cats={cat_num}")
    print(f"use_geo={use_geo}, use_cat={use_cat}")

    # Datasets
    train_ds = SASRecTrainDataset(user_seqs, item_num, args.max_len, use_geo=use_geo, use_cat=use_cat)
    val_ds   = SASRecEvalDataset(user_seqs, item_num, args.max_len, mode="val", num_neg=args.eval_num_neg, use_geo=use_geo, use_cat=use_cat)
    test_ds  = SASRecEvalDataset(user_seqs, item_num, args.max_len, mode="test", num_neg=args.eval_num_neg, use_geo=use_geo, use_cat=use_cat)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = SASRec(
        item_num=item_num, args=args,
        geo_num=geo_num, cat_num=cat_num,
        use_geo=use_geo, use_cat=use_cat
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1.0
    best_path = os.path.join(args.save_dir, f"{exp_name}_best.pt")

    def eval_loop(loader, k):
        model.eval()
        all_rec, all_ndcg, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                log_items = batch["log_items"].to(args.device, non_blocking=True)
                candidates = batch["candidates"].to(args.device, non_blocking=True)

                log_geos = batch.get("log_geos")
                log_cats = batch.get("log_cats")
                if log_geos is not None:
                    log_geos = log_geos.to(args.device, non_blocking=True)
                if log_cats is not None:
                    log_cats = log_cats.to(args.device, non_blocking=True)

                scores = model.predict_candidates(log_items, candidates, log_geos=log_geos, log_cats=log_cats)  # [B,C]
                rec, ndcg = recall_ndcg_at_k(scores, k)
                bs = scores.size(0)
                all_rec += rec * bs
                all_ndcg += ndcg * bs
                n += bs
        return all_rec / max(n, 1), all_ndcg / max(n, 1)

    # Train
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for it, batch in enumerate(train_loader, start=1):
            log_items = batch["log_items"].to(args.device, non_blocking=True)  # [B,L]
            pos_items = batch["pos_items"].to(args.device, non_blocking=True)
            neg_items = batch["neg_items"].to(args.device, non_blocking=True)

            log_geos = batch.get("log_geos")
            log_cats = batch.get("log_cats")
            if log_geos is not None:
                log_geos = log_geos.to(args.device, non_blocking=True)
            if log_cats is not None:
                log_cats = log_cats.to(args.device, non_blocking=True)

            # mask out padded positions
            mask = (pos_items != 0).float()  # [B,L]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pos_logits, neg_logits = model(
                    log_items, pos_items, neg_items,
                    log_geos=log_geos, log_cats=log_cats
                )

                # BCE loss
                pos_loss = bce(pos_logits, torch.ones_like(pos_logits)) * mask
                neg_loss = bce(neg_logits, torch.zeros_like(neg_logits)) * mask
                loss = (pos_loss.sum() + neg_loss.sum()) / (mask.sum() + 1e-8)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"[{exp_name}] epoch={epoch} step={global_step} loss={running_loss/args.log_every:.4f}")
                running_loss = 0.0

        # Eval each epoch
        val_rec, val_ndcg = eval_loop(val_loader, args.eval_k)
        print(f"[{exp_name}] Epoch {epoch} VAL  Recall@{args.eval_k}={val_rec:.4f} NDCG@{args.eval_k}={val_ndcg:.4f}")

        if val_ndcg > best_val:
            best_val = val_ndcg
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ndcg": best_val}, best_path)
            print(f"[{exp_name}] Saved best checkpoint to {best_path}")

    # Test with best
    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_rec, test_ndcg = eval_loop(test_loader, args.eval_k)
    print(f"[{exp_name}] TEST Recall@{args.eval_k}={test_rec:.4f} NDCG@{args.eval_k}={test_ndcg:.4f} (best_val_ndcg={best_val:.4f})")
    return {"exp": exp_name, "best_val_ndcg": best_val, "test_rec": test_rec, "test_ndcg": test_ndcg, "ckpt": best_path}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # preprocess
    p1 = sub.add_parser("preprocess")
    p1.add_argument("--raw_dir", type=str, default="/workspace/data/GoogleRAW")
    p1.add_argument("--out_dir", type=str, default="/workspace/data/GooglePROC")
    p1.add_argument("--geohash_precision", type=int, default=6)
    p1.add_argument("--min_user_len", type=int, default=5)
    p1.add_argument("--states_regex", type=str, default=None, help="Regex filter for filenames, e.g. 'New_York|California'")
    p1.add_argument("--max_users", type=int, default=None)
    p1.add_argument("--quiet", action="store_true")

    # train
    p2 = sub.add_parser("train")
    p2.add_argument("--artifacts", type=str, required=True)

    p2.add_argument("--device", type=str, default="cuda")
    p2.add_argument("--embed_dim", type=int, default=128)
    p2.add_argument("--num_blocks", type=int, default=2)
    p2.add_argument("--num_heads", type=int, default=4)
    p2.add_argument("--dropout", type=float, default=0.2)
    p2.add_argument("--max_len", type=int, default=100)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--weight_decay", type=float, default=0.0)
    p2.add_argument("--batch_size", type=int, default=2048)
    p2.add_argument("--epochs", type=int, default=50)
    p2.add_argument("--num_workers", type=int, default=8)
    p2.add_argument("--no_amp", action="store_true")
    p2.add_argument("--log_every", type=int, default=200)
    p2.add_argument("--eval_k", type=int, default=10)
    p2.add_argument("--eval_num_neg", type=int, default=100)
    p2.add_argument("--seed", type=int, default=2026)
    p2.add_argument("--save_dir", type=str, default="./checkpoints")

    # one flag to run all 3 experiments
    p2.add_argument("--run_all", action="store_true", help="Run base / +geo / +geo+cat sequentially")
    p2.add_argument("--use_geo", action="store_true")
    p2.add_argument("--use_cat", action="store_true")
    p2.add_argument("--exp_name", type=str, default="sasrec")

    args0 = parser.parse_args()

    if args0.cmd == "preprocess":
        cfg = PreprocessConfig(
            raw_dir=args0.raw_dir,
            out_dir=args0.out_dir,
            geohash_precision=args0.geohash_precision,
            min_user_len=args0.min_user_len,
            states_regex=args0.states_regex,
            max_users=args0.max_users,
            verbose=(not args0.quiet),
        )
        preprocess_google_maps(cfg)

    elif args0.cmd == "train":
        targs = TrainArgs(
            device=args0.device,
            embed_dim=args0.embed_dim,
            num_blocks=args0.num_blocks,
            num_heads=args0.num_heads,
            dropout=args0.dropout,
            max_len=args0.max_len,
            lr=args0.lr,
            weight_decay=args0.weight_decay,
            batch_size=args0.batch_size,
            epochs=args0.epochs,
            num_workers=args0.num_workers,
            amp=(not args0.no_amp),
            log_every=args0.log_every,
            eval_k=args0.eval_k,
            eval_num_neg=args0.eval_num_neg,
            seed=args0.seed,
            save_dir=args0.save_dir,
        )

        if args0.run_all:
            results = []
            results.append(train_one_experiment("SASRec_BASE", args0.artifacts, targs, use_geo=False, use_cat=False))
            results.append(train_one_experiment("SASRec_GEO",  args0.artifacts, targs, use_geo=True,  use_cat=False))
            results.append(train_one_experiment("SASRec_GEO_CAT", args0.artifacts, targs, use_geo=True, use_cat=True))

            print("\n===== Summary =====")
            for r in results:
                print(f'{r["exp"]}: best_val_ndcg={r["best_val_ndcg"]:.4f}, test_ndcg={r["test_ndcg"]:.4f}, test_rec={r["test_rec"]:.4f}, ckpt={r["ckpt"]}')
        else:
            exp = args0.exp_name
            train_one_experiment(exp, args0.artifacts, targs, use_geo=args0.use_geo, use_cat=args0.use_cat)

if __name__ == "__main__":
    main()
