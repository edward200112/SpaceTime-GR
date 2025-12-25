import os
import re
import json
import math
import time
import pickle
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SASRec import SASRec


# =========================================================
# Utils
# =========================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_left(seq, max_len, pad=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


def save_checkpoint(path, model, optimizer, epoch, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(ckpt, path)


def try_load_checkpoint(path, model, optimizer, device):
    """
    支持两种：
    - .pt：包含 model+optimizer+epoch(+RNG)
    - .pth：仅 model 权重
    PyTorch 2.6 默认 weights_only=True，会导致含 numpy/python RNG 的 ckpt 反序列化失败。
    """
    if not path or not os.path.exists(path):
        return 1

    print(f"🔄 Resuming from {path} ...")

    obj = None
    # 1) 优先用 weights_only=False 读取完整 ckpt（仅在你信任 ckpt 来源时使用）
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版本 torch.load 没有 weights_only 参数
        obj = torch.load(path, map_location=device)
    except Exception as e:
        print(f"⚠️ Full checkpoint load failed: {repr(e)}")
        print("⚠️ Falling back to weights-only load (model weights only).")
        obj = torch.load(path, map_location=device, weights_only=True)

    # 2) 完整 checkpoint
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        if optimizer is not None and obj.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(obj["optimizer"])
                print("✅ Loaded model + optimizer state.")
            except Exception as e:
                print(f"⚠️ Optimizer state load failed: {repr(e)}. Continue with model only.")
        else:
            print("⚠️ No optimizer state in checkpoint. Loaded model only.")

        start_epoch = int(obj.get("epoch", 0)) + 1

        # RNG（可选）
        try:
            if obj.get("torch_rng_state") is not None:
                torch.set_rng_state(obj["torch_rng_state"])
            if torch.cuda.is_available() and obj.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(obj["cuda_rng_state"])
            if obj.get("numpy_rng_state") is not None:
                np.random.set_state(obj["numpy_rng_state"])
            if obj.get("python_rng_state") is not None:
                random.setstate(obj["python_rng_state"])
        except Exception as e:
            print(f"⚠️ RNG restore failed (safe to ignore): {repr(e)}")

        print(f"⏩ Resume epoch = {start_epoch}")
        return start_epoch

    # 3) 旧式：只有权重
    model.load_state_dict(obj)
    print("✅ Loaded model weights only (old .pth or weights-only). Optimizer not restored.")
    m = re.search(r'epoch_(\d+)', os.path.basename(path))
    if m:
        return int(m.group(1)) + 1
    return 1

# =========================================================
# Strict split (leave-one-out): train/valid/test
# =========================================================
def build_strict_splits(raw_data_list):
    """
    raw_data_list: list of dict with keys: user_id, sequence (list[int])
    Return:
      splits: list of dict with
        uid, seq_train, valid_item, test_item, full_set
      item_freq: Counter for popularity (computed from seq_train only)
    """
    splits = []
    item_freq = Counter()

    for entry in raw_data_list:
        uid = str(entry["user_id"])
        seq = entry["sequence"]
        if not isinstance(seq, (list, tuple)):
            continue
        # 至少需要 3 个点：train至少1个转移 + valid + test
        if len(seq) < 3:
            continue

        valid_item = int(seq[-2])
        test_item = int(seq[-1])
        seq_train = [int(x) for x in seq[:-2]]  # 严格：train 不包含 valid/test
        if len(seq_train) < 2:
            continue

        full_set = set(int(x) for x in seq)

        # 统计训练部分的 item 频次
        for it in seq_train:
            if it != 0:
                item_freq[it] += 1

        splits.append({
            "uid": uid,
            "seq_train": seq_train,
            "valid_item": valid_item,
            "test_item": test_item,
            "full_set": full_set
        })

    return splits, item_freq


# =========================================================
# Popularity sampler (power-law smoothing)
# =========================================================
class PopularitySampler:
    def __init__(self, n_items, item_freq: Counter, alpha: float = 0.75):
        """
        n_items: item id range is [1..n_items]
        item_freq: frequency on train interactions
        alpha: smoothing exponent
        """
        self.n_items = int(n_items)

        # build prob for 1..n_items
        freqs = np.zeros(self.n_items + 1, dtype=np.float64)  # index 0 unused
        for it, c in item_freq.items():
            if 1 <= it <= self.n_items:
                freqs[it] = float(c)

        freqs[0] = 0.0
        # avoid all-zero
        if freqs.sum() <= 0:
            freqs[1:] = 1.0

        probs = np.power(freqs, alpha)
        probs[0] = 0.0
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs[1:] = 1.0
            probs_sum = probs.sum()

        self.probs = probs / probs_sum
        self.items = np.arange(self.n_items + 1, dtype=np.int64)  # 0..n_items

    def sample(self, size: int):
        # sample from 1..n_items according to popularity
        # exclude 0 by sampling from full then fix zeros (rare since prob[0]=0)
        s = np.random.choice(self.items, size=size, replace=True, p=self.probs)
        # ensure not 0
        if np.any(s == 0):
            s[s == 0] = 1
        return s


# =========================================================
# Dataset (train)
# - returns input_ids [L], pos_ids [L], full_set (for filtering neg)
# =========================================================
class SASRecTrainDataset(Dataset):
    def __init__(self, splits, n_items, max_len):
        self.splits = splits
        self.n_items = int(n_items)
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        s = self.splits[idx]
        seq_train = s["seq_train"]
        full_set = s["full_set"]

        # teacher-style next-item training: input = seq_train[:-1], pos = seq_train[1:]
        input_ids = seq_train[:-1]
        pos_ids = seq_train[1:]

        # keep last max_len-1 then pad to max_len
        input_ids = input_ids[-(self.max_len - 1):]
        pos_ids = pos_ids[-(self.max_len - 1):]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        pos_ids = [0] * pad_len + pos_ids

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(pos_ids),
            full_set  # python set
        )


# =========================================================
# Collator: build negatives with:
# - in-batch negatives: shuffled pos_ids across batch (per position)
# - popularity negatives: sample (M-1) negatives
# Output:
#   input_ids [B,L], pos_ids [B,L], neg_ids_list: list of [B,L] tensors length M
# =========================================================
class NegCollator:
    def __init__(self, n_items, pop_sampler: PopularitySampler, num_negs: int = 4, max_resample: int = 3):
        self.n_items = int(n_items)
        self.pop_sampler = pop_sampler
        self.num_negs = int(num_negs)
        self.max_resample = int(max_resample)

    def _filter_conflicts(self, neg_ids_np, forbid_sets, pos_ids_np, input_ids_np):
        """
        neg_ids_np: [B,L] numpy int64
        forbid_sets: list[set] length B (user full set)
        also forbid: pos_ids and input_ids (fast path)
        """
        B, L = neg_ids_np.shape

        # forbid if equals pos at same position OR equals input at same position is not enough;
        # we need forbid if neg in full history set. We'll do per-row loop (B=4096 ok).
        for b in range(B):
            fs = forbid_sets[b]
            # also avoid 0
            row = neg_ids_np[b]
            # quick mask: 0 or equals pos_ids or equals input_ids
            mask = (row == 0) | (row == pos_ids_np[b]) | (row == input_ids_np[b])
            # and membership in full set
            # (loop for correctness; could be optimized but ok)
            if np.any(mask):
                pass
            for j in range(L):
                if row[j] == 0 or row[j] == pos_ids_np[b, j] or row[j] == input_ids_np[b, j] or (row[j] in fs):
                    # mark as conflict by setting 0
                    row[j] = 0
            neg_ids_np[b] = row

        return neg_ids_np

    def _resample_zeros(self, neg_ids_np, forbid_sets, pos_ids_np, input_ids_np):
        """
        Replace zeros with popularity samples, with limited tries.
        """
        B, L = neg_ids_np.shape
        for _ in range(self.max_resample):
            zeros = (neg_ids_np == 0)
            if not np.any(zeros):
                break
            num = int(zeros.sum())
            repl = self.pop_sampler.sample(num).reshape(-1)
            neg_ids_np[zeros] = repl
            neg_ids_np = self._filter_conflicts(neg_ids_np, forbid_sets, pos_ids_np, input_ids_np)
        # if still zeros, force to 1
        neg_ids_np[neg_ids_np == 0] = 1
        return neg_ids_np

    def __call__(self, batch):
        input_ids, pos_ids, forbid_sets = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)  # [B,L]
        pos_ids = torch.stack(pos_ids, dim=0)      # [B,L]
        B, L = pos_ids.shape

        input_np = input_ids.numpy()
        pos_np = pos_ids.numpy()

        negs = []

        # (1) in-batch negatives: shuffle pos_ids along batch dim
        perm = torch.randperm(B)
        inbatch = pos_ids[perm].clone().numpy()  # [B,L]
        inbatch = self._filter_conflicts(inbatch, forbid_sets, pos_np, input_np)
        inbatch = self._resample_zeros(inbatch, forbid_sets, pos_np, input_np)
        negs.append(torch.from_numpy(inbatch).long())

        # (2) popularity negatives for remaining
        for _ in range(self.num_negs - 1):
            neg_np = self.pop_sampler.sample(B * L).reshape(B, L)
            neg_np = self._filter_conflicts(neg_np, forbid_sets, pos_np, input_np)
            neg_np = self._resample_zeros(neg_np, forbid_sets, pos_np, input_np)
            negs.append(torch.from_numpy(neg_np).long())

        return input_ids, pos_ids, negs


# =========================================================
# Loss
# - pos_logits: [B,L]
# - neg_logits: [B,L]
# - only compute where pos_ids != 0
# =========================================================
def bce_pairwise_loss(criterion, pos_logits, neg_logits, pos_ids):
    idx = torch.where(pos_ids != 0)
    if idx[0].numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    loss = criterion(pos_logits[idx], pos_labels[idx]) + criterion(neg_logits[idx], neg_labels[idx])
    return loss


# =========================================================
# Strict Evaluation (sampled negatives)
# - For each user, evaluate predicting valid/test given history
# - Candidate set: 1 positive + N negatives (neg not in user's full_set)
# - Use model.predict_full on small batch to get logits over all items, then gather candidates
#   (batch must be small due to huge item space)
# =========================================================
import time
import math
import random
import torch

@torch.no_grad()
def evaluate_strict_sampled_fast(
    model, splits, n_items, max_len, pop_sampler,
    mode="valid", num_eval_users=20000, num_neg=199,
    eval_batch_size=512, device="cuda",
    ks=(10, 50, 100),
):
    assert mode in ("valid", "test")
    model.eval()

    if num_eval_users is not None and 0 < num_eval_users < len(splits):
        eval_splits = random.sample(splits, num_eval_users)
    else:
        eval_splits = splits

    total_users = len(eval_splits)
    total_batches = (total_users + eval_batch_size - 1) // eval_batch_size
    C = 1 + num_neg

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    total = 0

    def pad_left_local(seq, L, pad=0):
        if len(seq) >= L:
            return seq[-L:]
        return [pad] * (L - len(seq)) + seq

    def make_hist_and_pos(s):
        seq_train = s["seq_train"]
        if mode == "valid":
            hist = seq_train
            pos = int(s["valid_item"])
        else:
            hist = seq_train + [int(s["valid_item"])]
            pos = int(s["test_item"])
        x = pad_left_local(hist[-max_len:], max_len, pad=0)
        return x, pos, s["full_set"]

    def fmt_sec(x):
        x = int(max(0, x))
        h = x // 3600
        m = (x % 3600) // 60
        s = x % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    start_t = time.time()
    processed_batches = 0
    processed_users = 0

    batch_inputs, batch_pos, batch_forbid = [], [], []

    def log_progress():
        now = time.time()
        elapsed = now - start_t
        avg = elapsed / max(1, processed_batches)
        eta = (total_batches - processed_batches) * avg
        print(
            f"[Eval-{mode}] {processed_users}/{total_users} users | "
            f"{processed_batches}/{total_batches} batches | "
            f"elapsed {fmt_sec(elapsed)} | avg/batch {avg:.3f}s | ETA {fmt_sec(eta)}",
            flush=True
        )

    for s in eval_splits:
        x, pos, forbid = make_hist_and_pos(s)
        batch_inputs.append(x)
        batch_pos.append(pos)
        batch_forbid.append(forbid)

        if len(batch_inputs) == eval_batch_size:
            B = len(batch_inputs)
            candidates = torch.empty((B, C), dtype=torch.long)

            # 负采样：这里可能是最慢的部分，所以加一个小提示
            for i in range(B):
                if i % 128 == 0:
                    print(f"[Eval-{mode}] building candidates: {i}/{B} ...", flush=True)

                pos_i = batch_pos[i]
                forbid_i = batch_forbid[i]
                negs = []
                tries = 0
                while len(negs) < num_neg and tries < num_neg * 20:
                    cand = int(pop_sampler.sample(1)[0])
                    tries += 1
                    if cand == 0 or cand == pos_i or cand in forbid_i:
                        continue
                    negs.append(cand)
                while len(negs) < num_neg:
                    cand = random.randint(1, n_items)
                    if cand != pos_i and cand not in forbid_i:
                        negs.append(cand)

                candidates[i, 0] = pos_i
                candidates[i, 1:] = torch.tensor(negs, dtype=torch.long)

            input_tensor = torch.LongTensor(batch_inputs).to(device)
            candidates = candidates.to(device)

            # ✅ only score candidates
            scores = model.predict_candidates(input_tensor, candidates)  # [B, C]

            pos_scores = scores[:, 0]         # [B]
            neg_scores = scores[:, 1:]        # [B, num_neg]
            better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # [B]
            ranks = (better + 1).tolist()

            for r in ranks:
                total += 1
                for k in ks:
                    if r <= k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / math.log2(r + 1)

            processed_users += B
            processed_batches += 1
            log_progress()

            batch_inputs, batch_pos, batch_forbid = [], [], []

    # flush remainder
    if batch_inputs:
        B = len(batch_inputs)
        candidates = torch.empty((B, C), dtype=torch.long)

        for i in range(B):
            if i % 128 == 0:
                print(f"[Eval-{mode}] building candidates: {i}/{B} ...", flush=True)

            pos_i = batch_pos[i]
            forbid_i = batch_forbid[i]
            negs = []
            tries = 0
            while len(negs) < num_neg and tries < num_neg * 20:
                cand = int(pop_sampler.sample(1)[0])
                tries += 1
                if cand == 0 or cand == pos_i or cand in forbid_i:
                    continue
                negs.append(cand)
            while len(negs) < num_neg:
                cand = random.randint(1, n_items)
                if cand != pos_i and cand not in forbid_i:
                    negs.append(cand)

            candidates[i, 0] = pos_i
            candidates[i, 1:] = torch.tensor(negs, dtype=torch.long)

        input_tensor = torch.LongTensor(batch_inputs).to(device)
        candidates = candidates.to(device)
        scores = model.predict_candidates(input_tensor, candidates)

        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        better = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
        ranks = (better + 1).tolist()

        for r in ranks:
            total += 1
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / math.log2(r + 1)

        processed_users += B
        processed_batches += 1
        log_progress()

    elapsed = time.time() - start_t
    print(f"[Eval-{mode}] DONE. evaluated={total}/{total_users} users, elapsed={elapsed:.2f}s", flush=True)

    metrics = {"total": total, "elapsed_sec": elapsed}
    for k in ks:
        metrics[f"HR@{k}"] = hits[k] / total if total > 0 else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total > 0 else 0.0
    return metrics


# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_path", type=str, default="./SASRec_Data/sasrec_dataset.pkl")
    p.add_argument("--output_dir", type=str, default="./SASRec_Data")

    # training
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--resume_path", type=str, default="./SASRec_Data/sasrec_full_latest.pt")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # model config
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # negatives
    p.add_argument("--num_negs", type=int, default=4, help="M: 每个位置的负样本数（包含 1 个 in-batch + M-1 个 popularity）")
    p.add_argument("--pop_alpha", type=float, default=0.75, help="popularity smoothing exponent")

    # dataloader
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--pin_memory", action="store_true")

    # eval
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_users", type=int, default=20000)
    p.add_argument("--eval_neg", type=int, default=999)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_ks", type=str, default="10,50,100")

    # export preds (optional)
    p.add_argument("--export_top_k", type=int, default=50)

    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(args.seed)

    print(f"📥 Loading dataset from {args.dataset_path} ...")
    with open(args.dataset_path, "rb") as f:
        pkg = pickle.load(f)

    raw_data_list = pkg["data"]
    n_items = int(pkg["n_items"])
    id2item = pkg.get("id2item", None)

    print("✂️ Building strict splits (train/valid/test) ...")
    splits, item_freq = build_strict_splits(raw_data_list)
    print(f"✅ Users after split: {len(splits)} (dropped short sequences)")
    print(f"✅ n_items = {n_items}, max_len = {args.max_len}")
    print(f"✅ Train interaction freq entries: {len(item_freq)}")

    pop_sampler = PopularitySampler(n_items=n_items, item_freq=item_freq, alpha=args.pop_alpha)

    # dataset / loader
    train_ds = SASRecTrainDataset(splits, n_items=n_items, max_len=args.max_len)
    collator = NegCollator(n_items=n_items, pop_sampler=pop_sampler, num_negs=args.num_negs)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collator,
        persistent_workers=True,      # ✅
        prefetch_factor=4,            # ✅（默认2）
    )
    torch.set_num_threads(1)
    print("🏗️ Initializing SASRec ...")
    model = SASRec(n_items, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
    criterion = nn.BCEWithLogitsLoss()

    # resume
    start_epoch = try_load_checkpoint(args.resume_path, model, optimizer, args.device)

    print("🚀 Starting Training ...")
    ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        model.train()
        train_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for input_ids, pos_ids, neg_list in pbar:
            input_ids = input_ids.to(args.device, non_blocking=True)
            pos_ids = pos_ids.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 先用第一个 neg 计算 pos_logits（pos_logits 不依赖 neg，通常同一次 forward）
            neg0 = neg_list[0].to(args.device, non_blocking=True)
            pos_logits, neg_logits = model(input_ids, pos_ids, neg0)
            loss = bce_pairwise_loss(criterion, pos_logits, neg_logits, pos_ids)

            # 多负样本：对其余 neg 逐个算 neg_logits（不改模型本体）
            if len(neg_list) > 1:
                for neg_ids in neg_list[1:]:
                    neg_ids = neg_ids.to(args.device, non_blocking=True)
                    # 只取 neg_logits，pos_logits 丢弃
                    _, neg_logits_m = model(input_ids, pos_ids, neg_ids)
                    loss = loss + bce_pairwise_loss(criterion, pos_logits, neg_logits_m, pos_ids)

                loss = loss / float(len(neg_list))  # 平均

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += float(loss.item())
            steps += 1
            pbar.set_postfix({"loss": f"{train_loss / steps:.4f}"})

        # save
        latest_pt = os.path.join(args.output_dir, "sasrec_full_latest.pt")
        latest_pth = os.path.join(args.output_dir, "sasrec_full_latest.pth")
        save_checkpoint(latest_pt, model, optimizer, epoch, args)
        torch.save(model.state_dict(), latest_pth)

        if epoch % 10 == 0:
            save_checkpoint(os.path.join(args.output_dir, f"sasrec_full_epoch_{epoch}.pt"), model, optimizer, epoch, args)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"sasrec_full_epoch_{epoch}.pth"))

        # strict eval
        if args.do_eval and (epoch % args.eval_every == 0):
            print("\n🧪 Strict Evaluation (sampled negatives) ...")
            t0 = time.time()
            m_valid = evaluate_strict_sampled_fast(
                model=model, splits=splits, n_items=n_items, max_len=args.max_len,
                pop_sampler=pop_sampler, mode="valid",
                num_eval_users=args.eval_users, num_neg=args.eval_neg,
                eval_batch_size=args.eval_batch_size, device=args.device, ks=ks,
            )
            t1 = time.time()
            print(f"[Eval-valid] total time: {t1 - t0:.2f}s")

            m_test = evaluate_strict_sampled_fast(
                model=model, splits=splits, n_items=n_items, max_len=args.max_len,
                pop_sampler=pop_sampler, mode="test",
                num_eval_users=args.eval_users, num_neg=args.eval_neg,
                eval_batch_size=args.eval_batch_size, device=args.device, ks=ks,
            )
            t2 = time.time()
            print(f"[Eval-test ] total time: {t2 - t1:.2f}s")

            print("VALID:", m_valid)
            print("TEST: ", m_test)
            print("")
            print("")

    print("✅ Training Finished!")

    # 可选：导出 topK 推荐（注意：这里是 sampled 形式更实际；你原来用 predict_full 导出全量 topK 会很重）
    # 这里保留你原导出逻辑的风格，但强烈建议别用超大 batch
    if id2item is not None and args.export_top_k and args.export_top_k > 0:
        print("\n🔮 Exporting Teacher Predictions (Top-K, using predict_full) ...")
        model.eval()

        # 用较小 batch 避免 logits 过大
        export_bs = min(256, args.batch_size)
        export_inputs = []
        export_uids = []

        teacher_preds = {}
        with torch.no_grad():
            for s in tqdm(splits, desc="Export"):
                # 用 test history 作为导出 history（train + valid）
                hist = s["seq_train"] + [s["valid_item"]]
                inp = pad_left(hist[-args.max_len:], args.max_len, pad=0)
                export_inputs.append(inp)
                export_uids.append(s["uid"])

                if len(export_inputs) == export_bs:
                    x = torch.LongTensor(export_inputs).to(args.device)
                    logits = model.predict_full(x)
                    logits[:, 0] = -float("inf")
                    _, indices = torch.topk(logits, args.export_top_k, dim=-1)
                    indices = indices.cpu().numpy()

                    for i, uid in enumerate(export_uids):
                        recs = []
                        for idx in indices[i]:
                            if idx in id2item:
                                recs.append(id2item[idx])
                        teacher_preds[uid] = recs

                    export_inputs, export_uids = [], []

            # flush
            if export_inputs:
                x = torch.LongTensor(export_inputs).to(args.device)
                logits = model.predict_full(x)
                logits[:, 0] = -float("inf")
                _, indices = torch.topk(logits, args.export_top_k, dim=-1)
                indices = indices.cpu().numpy()
                for i, uid in enumerate(export_uids):
                    recs = []
                    for idx in indices[i]:
                        if idx in id2item:
                            recs.append(id2item[idx])
                    teacher_preds[uid] = recs

        pred_path = os.path.join(args.output_dir, "teacher_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(teacher_preds, f)
        print(f"💾 Saved teacher_predictions.json to {pred_path}")

    print("🎉 All Done!")


if __name__ == "__main__":
    main()
