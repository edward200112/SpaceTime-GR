import os
import json
import gzip
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# --------- 导入 SASRec（按你的工程结构兼容）---------
try:
    from SASRec import SASRec
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    try:
        from TeacherModel.model import SASRec
    except ImportError:
        raise ImportError("❌ 无法导入 SASRec！请确保 SASRec.py 或 TeacherModel/model.py 存在。")


def parse_args():
    p = argparse.ArgumentParser("Check how 'hard' the mined negatives are.")
    p.add_argument("--sasrec_model_path", type=str, default="./SASRec_Data/sasrec_full_latest.pth")
    p.add_argument("--sasrec_data_path", type=str, default="./SASRec_Data/sasrec_dataset.pkl")
    p.add_argument("--sft_jsonl", type=str, default="./SFT/sft_data/sft_enhanced_train.jsonl")

    p.add_argument("--raw_meta_dir", type=str, default="/workspace/data/GoogleRAW",
                   help="包含 meta-*.json.gz 的目录，用于取 category/lat/lon")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 采样与推理参数
    p.add_argument("--sample_n", type=int, default=50000, help="从 SFT jsonl 采样多少条来评估")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16, help="评估 batch，不要太大（logits 很大）")

    # 评估 topK
    p.add_argument("--k_gt", type=int, default=200, help="判定 GT 是否在 topK")
    p.add_argument("--k_neg", type=int, default=50, help="判定 NEG 是否在 topK")
    p.add_argument("--k_top", type=int, default=500, help="取 topK 用于估计排名/覆盖率")

    # TrueHard 阈值
    p.add_argument("--gap_th", type=float, default=0.05, help="score_gt - score_neg <= gap_th 认为很接近")
    p.add_argument("--dist_km", type=float, default=10.0, help="地理距离阈值（km）")

    return p.parse_args()


# --------- Haversine 计算地理距离（km）---------
def haversine_km(lat1, lon1, lat2, lon2):
    # 允许缺失
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(R * c)


# --------- 从 meta-*.json.gz 读 POI 信息（category/lat/lon）---------
class POIMeta:
    def __init__(self, raw_meta_dir):
        self.raw_meta_dir = raw_meta_dir
        self.data = {}  # gmap_id -> dict(category0, lat, lon)

    def load_for_ids(self, needed_ids_set):
        if not os.path.isdir(self.raw_meta_dir):
            print("⚠️ raw_meta_dir not found, skip meta loading.")
            return

        files = [f for f in os.listdir(self.raw_meta_dir) if f.startswith("meta-") and f.endswith(".json.gz")]
        loaded = 0
        for fname in files:
            path = os.path.join(self.raw_meta_dir, fname)
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                        except:
                            continue
                        gid = d.get("gmap_id")
                        if gid in needed_ids_set and gid not in self.data:
                            cats = d.get("category")
                            cat0 = None
                            if isinstance(cats, list) and len(cats) > 0:
                                cat0 = str(cats[0])
                            lat = d.get("latitude", None)
                            lon = d.get("longitude", None)
                            # 有些数据里是字符串
                            try:
                                lat = float(lat) if lat is not None else None
                                lon = float(lon) if lon is not None else None
                            except:
                                lat, lon = None, None

                            self.data[gid] = {"cat0": cat0, "lat": lat, "lon": lon}
                            loaded += 1
            except:
                continue
        print(f"✅ Loaded POI meta for {loaded} ids.")

    def get(self, gid):
        return self.data.get(gid, {"cat0": None, "lat": None, "lon": None})


# --------- 采样 SFT lines（只用 meta.user_id/target_id/hard_neg_id）---------
def sample_sft_lines(path, sample_n, seed):
    random.seed(seed)
    buf = []
    # reservoir sampling：适合超大文件
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except:
                continue

            meta = d.get("meta", {})
            uid = meta.get("user_id", None)
            tgt = meta.get("target_id", None)
            neg = meta.get("hard_neg_id", None)
            if uid is None or tgt is None or neg is None:
                continue

            item = {"uid": str(uid), "target_gmap": str(tgt), "neg_gmap": str(neg)}
            if len(buf) < sample_n:
                buf.append(item)
            else:
                j = random.randint(0, i)
                if j < sample_n:
                    buf[j] = item
    return buf


# --------- 从 sasrec_dataset.pkl 找 uid -> sequence（只取采样 uid）---------
def build_uid2seq(sasrec_data_path, needed_uids_set):
    with open(sasrec_data_path, "rb") as f:
        dataset = pickle.load(f)
    raw_data_list = dataset["data"]
    max_len = dataset["max_len"]
    item2id = dataset["item2id"]
    id2item = dataset["id2item"]
    n_items = dataset["n_items"]
    item_size = n_items  # 按你原脚本

    uid2seq = {}
    for entry in tqdm(raw_data_list, desc="Scanning sasrec_dataset for sampled uids"):
        uid = str(entry["user_id"])
        if uid in needed_uids_set and uid not in uid2seq:
            uid2seq[uid] = entry["sequence"]

    return dataset, uid2seq, max_len, item2id, id2item, item_size


def seq_to_input(seq, max_len):
    # seq 是 idx 列表（包含 target），输入用 history=seq[:-1]
    if len(seq) < 2:
        return None
    history = seq[:-1]
    input_seq = np.zeros([max_len], dtype=np.int32)
    length = min(len(history), max_len)
    if length > 0:
        input_seq[-length:] = history[-length:]
    return torch.LongTensor(input_seq)


def load_teacher(args, item_size, max_len):
    class ModelArgs:
        def __init__(self):
            self.embed_dim = 128
            self.max_len = max_len
            self.num_blocks = 2
            self.num_heads = 2
            self.dropout = 0.2
            self.device = args.device

    margs = ModelArgs()
    model = SASRec(item_size, margs)
    state_dict = torch.load(args.sasrec_model_path, map_location=args.device)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def main():
    args = parse_args()
    print("🚀 Checking hard-negative quality ...")
    print("Config:", vars(args))

    # 1) 采样 SFT
    samples = sample_sft_lines(args.sft_jsonl, args.sample_n, args.seed)
    print(f"✅ Sampled {len(samples)} lines from SFT jsonl")

    needed_uids = {s["uid"] for s in samples}
    needed_gids = {s["target_gmap"] for s in samples} | {s["neg_gmap"] for s in samples}

    # 2) 读 sasrec_dataset，只取采样 uid 的序列
    dataset, uid2seq, max_len, item2id, id2item, item_size = build_uid2seq(args.sasrec_data_path, needed_uids)
    print(f"✅ Found sequences for {len(uid2seq)}/{len(needed_uids)} sampled uids")

    # 3) 加载 POI meta（只加载目标+负样本需要的 gid）
    poi_meta = POIMeta(args.raw_meta_dir)
    poi_meta.load_for_ids(needed_gids)

    # 4) 加载 teacher
    teacher = load_teacher(args, item_size=item_size, max_len=max_len)

    # 5) 组装评估 batch
    eval_items = []
    missing_uid = 0
    missing_idmap = 0
    for s in samples:
        uid = s["uid"]
        seq = uid2seq.get(uid, None)
        if seq is None:
            missing_uid += 1
            continue

        # gmap_id -> idx
        gt_gid = s["target_gmap"]
        neg_gid = s["neg_gmap"]
        gt_idx = item2id.get(gt_gid, None)
        neg_idx = item2id.get(neg_gid, None)
        if gt_idx is None or neg_idx is None:
            missing_idmap += 1
            continue

        inp = seq_to_input(seq, max_len)
        if inp is None:
            continue
        eval_items.append((uid, inp, int(gt_idx), int(neg_idx), gt_gid, neg_gid))

    print(f"Eval usable: {len(eval_items)} | missing_uid={missing_uid} missing_idmap={missing_idmap}")
    if len(eval_items) == 0:
        print("❌ No usable eval items. Check uid mapping / item2id mapping.")
        return

    # 6) 统计量
    total = 0
    gt_in_top100 = 0
    gt_in_top200 = 0
    gt_in_top500 = 0

    neg_in_top20 = 0
    neg_in_top50 = 0
    neg_in_top100 = 0

    neg_beats_gt = 0
    gap_list = []

    same_cat = 0
    dist_ok = 0
    dist_list = []

    # TrueHard（你可以按需改）
    true_hard = 0

    B = args.batch_size
    k_top = args.k_top

    for b_start in tqdm(range(0, len(eval_items), B), desc="Teacher inference"):
        batch = eval_items[b_start:b_start+B]
        input_seqs = torch.stack([x[1] for x in batch], dim=0).to(args.device)  # [B, max_len]
        gt_idxs = torch.tensor([x[2] for x in batch], device=args.device)        # [B]
        neg_idxs = torch.tensor([x[3] for x in batch], device=args.device)       # [B]

        logits = teacher.predict_full(input_seqs)  # [B, item_size]
        # 取 topK（k_top）
        top_scores, top_indices = torch.topk(logits, k=k_top, dim=1)  # [B, k_top]

        # 分数
        gt_scores = logits.gather(1, gt_idxs.view(-1, 1)).squeeze(1)
        neg_scores = logits.gather(1, neg_idxs.view(-1, 1)).squeeze(1)

        # topK 命中
        # 用广播判断是否在 topK
        gt_in_top = (top_indices == gt_idxs.view(-1, 1)).any(dim=1)
        neg_in_top = (top_indices == neg_idxs.view(-1, 1)).any(dim=1)

        # 更细粒度的 topK（20/50/100/200/500），从 top_indices 的前 K 切片判断
        def in_topK(indices, target, K):
            return (indices[:, :K] == target.view(-1, 1)).any(dim=1)

        gt_top100_mask = in_topK(top_indices, gt_idxs, min(100, k_top))
        gt_top200_mask = in_topK(top_indices, gt_idxs, min(200, k_top))
        gt_top500_mask = in_topK(top_indices, gt_idxs, min(500, k_top))

        neg_top20_mask = in_topK(top_indices, neg_idxs, min(20, k_top))
        neg_top50_mask = in_topK(top_indices, neg_idxs, min(50, k_top))
        neg_top100_mask = in_topK(top_indices, neg_idxs, min(100, k_top))

        # score gap
        gap = (gt_scores - neg_scores).detach().cpu().numpy().tolist()
        gap_list.extend(gap)

        beats = (neg_scores > gt_scores).detach().cpu().numpy().tolist()

        # meta 相似度统计
        for i, item in enumerate(batch):
            uid, _, gt_idx, neg_idx, gt_gid, neg_gid = item
            m_gt = poi_meta.get(gt_gid)
            m_ng = poi_meta.get(neg_gid)

            # 类别一致（cat0）
            if m_gt["cat0"] is not None and m_ng["cat0"] is not None and m_gt["cat0"] == m_ng["cat0"]:
                same_cat += 1

            dkm = haversine_km(m_gt["lat"], m_gt["lon"], m_ng["lat"], m_ng["lon"])
            if dkm is not None:
                dist_list.append(dkm)
                if dkm <= args.dist_km:
                    dist_ok += 1

        # TrueHard 判定（向量化 + 条件融合）
        # 条件：
        #   neg in top_k_neg
        #   gt in top_k_gt
        #   (neg beats gt OR gap <= gap_th)
        #   (same_cat OR dist<=dist_km)  —— 这部分我们用逐样本计数，所以这里简化：先用 rank/score 判定计数，再用 meta 额外统计
        cond_rank = neg_top50_mask & gt_top200_mask  # 这里用默认 50/200，可按 args.k_neg/k_gt 修改
        cond_gap = (neg_scores > gt_scores) | ((gt_scores - neg_scores) <= args.gap_th)
        hard_mask = (cond_rank & cond_gap).detach().cpu().numpy().tolist()

        total += len(batch)
        gt_in_top100 += int(gt_top100_mask.sum().item())
        gt_in_top200 += int(gt_top200_mask.sum().item())
        gt_in_top500 += int(gt_top500_mask.sum().item())

        neg_in_top20 += int(neg_top20_mask.sum().item())
        neg_in_top50 += int(neg_top50_mask.sum().item())
        neg_in_top100 += int(neg_top100_mask.sum().item())

        neg_beats_gt += int(sum(beats))
        true_hard += int(sum(hard_mask))

    # 7) 输出结果
    gap_arr = np.array(gap_list, dtype=np.float32)
    dist_arr = np.array(dist_list, dtype=np.float32) if len(dist_list) > 0 else None

    print("\n========================================")
    print("📌 HARD NEGATIVE QUALITY REPORT")
    print("========================================")
    print(f"Samples evaluated: {total}")

    print("\n[Teacher Coverage]")
    print(f"GT in top100: {gt_in_top100/total*100:.2f}%")
    print(f"GT in top200: {gt_in_top200/total*100:.2f}%")
    print(f"GT in top500: {gt_in_top500/total*100:.2f}%")
    print(f"NEG in top20:  {neg_in_top20/total*100:.2f}%")
    print(f"NEG in top50:  {neg_in_top50/total*100:.2f}%")
    print(f"NEG in top100: {neg_in_top100/total*100:.2f}%")

    print("\n[Hardness]")
    print(f"NEG beats GT (score_neg > score_gt): {neg_beats_gt/total*100:.2f}%")
    print(f"score_gap = score_gt - score_neg: mean={gap_arr.mean():.4f}, "
          f"p50={np.percentile(gap_arr,50):.4f}, p90={np.percentile(gap_arr,90):.4f}, p95={np.percentile(gap_arr,95):.4f}")

    print("\n[Meta Similarity]")
    print(f"Same top-1 category ratio: {same_cat/total*100:.2f}%")
    if dist_arr is not None:
        print(f"Geo distance (km): mean={dist_arr.mean():.2f}, p50={np.percentile(dist_arr,50):.2f}, p90={np.percentile(dist_arr,90):.2f}")
        print(f"Distance <= {args.dist_km}km ratio: {dist_ok/total*100:.2f}%")
    else:
        print("Geo distance: N/A (missing lat/lon)")

    print("\n[Proxy TrueHard (rank/score only)]")
    print("Definition: (NEG in top50) & (GT in top200) & (neg_beats_gt OR gap<=gap_th)")
    print(f"gap_th={args.gap_th}")
    print(f"TrueHard ratio (without meta constraint): {true_hard/total*100:.2f}%")
    print("========================================\n")


if __name__ == "__main__":
    main()
