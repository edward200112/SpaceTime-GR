import os
import json
import gzip
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 尝试导入 SASRec 模型
try:
    from SASRec import SASRec
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    try:
        from TeacherModel.model import SASRec
    except ImportError:
        raise ImportError("❌ 无法导入 SASRec！请确保 SASRec.py 或 TeacherModel/model.py 存在。")


# -------------------------
# Geo utils
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


# ==========================================
# 1. 参数配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data using SASRec Teacher (Two Negatives: teacher + plausible)")

    # 路径
    parser.add_argument("--sasrec_model_path", type=str, default="./SASRec_Data/sasrec_full_latest.pth")
    parser.add_argument("--sasrec_data_path", type=str, default="./SASRec_Data/sasrec_dataset.pkl")
    parser.add_argument("--raw_meta_dir", type=str, default="/workspace/data/GoogleRAW", help="原始元数据目录")
    parser.add_argument("--output_file", type=str, default="./SFT/sft_data/sft_enhanced_train_two_negs.jsonl")

    # 教师模型参数（需与你训练 SASRec 的配置一致）
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 推理/生成参数
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k_candidates", type=int, default=500, help="teacher topK 候选池大小（建议>=500，便于在约束下找 plausible neg）")

    # plausible 约束（可替代性）
    parser.add_argument("--dist_km", type=float, default=10.0, help="plausible neg 的地理距离阈值（km）")
    parser.add_argument("--require_same_category", action="store_true", help="plausible neg 是否强制同 top-1 类别")
    parser.add_argument("--allow_either_cat_or_dist", action="store_true",
                        help="若开启：plausible 只需满足(同类 OR 距离<=dist_km)。若关闭：需同时满足(同类 AND 距离<=dist_km)。")

    # 历史窗口用于 prompt（LLM 端）
    parser.add_argument("--recent_hist_n", type=int, default=5)

    return parser.parse_args()


# ==========================================
# 2. 元数据管理器（同时提供 desc + cat + lat/lon）
# ==========================================
class MetaDataManager:
    """
    meta_dict[gmap_id] = {
        "desc": "name (cat0)",
        "cat0": cat0,
        "lat": lat,
        "lon": lon
    }
    """
    def __init__(self, raw_dir, valid_gmap_ids):
        self.raw_dir = raw_dir
        self.valid_gmap_ids = set(valid_gmap_ids)
        self.meta_dict = {}

    def load(self):
        print("📚 Loading Metadata (desc + category + lat/lon)...")
        if not os.path.exists(self.raw_dir):
            print("⚠️ Raw meta dir not found. Fallback to IDs.")
            return

        files = [f for f in os.listdir(self.raw_dir) if f.startswith("meta-") and f.endswith(".json.gz")]
        loaded_count = 0

        for fname in files:
            path = os.path.join(self.raw_dir, fname)
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                        except:
                            continue
                        gid = d.get("gmap_id")
                        if gid is None or gid not in self.valid_gmap_ids:
                            continue

                        if gid in self.meta_dict:
                            continue

                        name = (d.get("name") or "Unknown Place").strip()
                        cats = d.get("category")
                        cat0 = None
                        if isinstance(cats, list) and len(cats) > 0:
                            cat0 = str(cats[0])
                        elif isinstance(cats, str) and cats:
                            cat0 = cats

                        lat = d.get("latitude", None)
                        lon = d.get("longitude", None)
                        try:
                            lat = float(lat) if lat is not None else None
                            lon = float(lon) if lon is not None else None
                        except:
                            lat, lon = None, None

                        cat_str = cat0 if cat0 else "Place"
                        desc = f"{name} ({cat_str})"

                        self.meta_dict[gid] = {
                            "desc": desc,
                            "cat0": cat0,
                            "lat": lat,
                            "lon": lon,
                        }
                        loaded_count += 1
            except:
                continue

        print(f"✅ Metadata loaded for {loaded_count} POIs.")

    def get_text(self, gmap_id):
        m = self.meta_dict.get(gmap_id)
        if m is None:
            return f"POI_{gmap_id}"
        return m["desc"]

    def get_meta(self, gmap_id):
        return self.meta_dict.get(gmap_id, {"cat0": None, "lat": None, "lon": None})


# ==========================================
# 3. 数据集类：准备 teacher 输入序列 + 历史集合
# ==========================================
class SequenceListDataset(Dataset):
    def __init__(self, data_list, maxlen):
        self.samples = []
        self.maxlen = maxlen
        self.user_history_lookup = {}

        print("🔄 Processing sequences for inference...")
        for entry in tqdm(data_list):
            seq = entry["sequence"]
            uid = str(entry["user_id"])
            if len(seq) < 2:
                continue

            history = seq[:-1]
            target = seq[-1]

            # 历史集合：包含 target（严格过滤）
            self.user_history_lookup[uid] = set(seq)

            # padding to maxlen
            input_seq = np.zeros([self.maxlen], dtype=np.int32)
            length = min(len(history), self.maxlen)
            if length > 0:
                input_seq[-length:] = history[-length:]

            self.samples.append({
                "uid": uid,
                "input_seq": torch.LongTensor(input_seq),
                "target_idx": int(target),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 4. 主程序
# ==========================================
def main():
    args = parse_args()
    print("🚀 Starting Teacher-Guided Data Generation (Two Negatives)...")
    print("Config:", vars(args))

    # --- 1) 加载 sasrec_dataset.pkl ---
    print(f"📥 Loading dataset: {args.sasrec_data_path}")
    with open(args.sasrec_data_path, "rb") as f:
        dataset = pickle.load(f)

    raw_data_list = dataset["data"]
    item2id = dataset["item2id"]   # gmap_id -> idx
    id2item = dataset["id2item"]   # idx -> gmap_id
    n_items = dataset["n_items"]
    max_len = dataset["max_len"]
    item_size = n_items  # 与你原脚本保持一致

    print(f"   Data Count: {len(raw_data_list)}")
    print(f"   Item Size: {item_size}, Max Len: {max_len}")

    # --- 2) 加载元数据 ---
    valid_gmap_ids = list(item2id.keys())
    meta_manager = MetaDataManager(args.raw_meta_dir, valid_gmap_ids)
    if os.path.exists(args.raw_meta_dir):
        meta_manager.load()

    # --- 3) 加载 SASRec teacher ---
    print(f"🏗️ Loading SASRec Model: {args.sasrec_model_path}")

    class ModelArgs:
        def __init__(self):
            self.embed_dim = args.hidden_units
            self.max_len = max_len
            self.num_blocks = args.num_blocks
            self.num_heads = args.num_heads
            self.dropout = args.dropout_rate
            self.device = args.device

    model_args = ModelArgs()
    model = SASRec(item_size, model_args)

    try:
        state_dict = torch.load(args.sasrec_model_path, map_location=args.device)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    model.to(args.device)
    model.eval()

    # --- 4) 准备 inference dataset / loader ---
    dataset_obj = SequenceListDataset(raw_data_list, max_len)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=4)
    user_history_lookup = dataset_obj.user_history_lookup

    # --- 5) 挖掘双负样本并写 JSONL ---
    print("⛏️ Mining Two Negatives & Writing JSONL...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    generated_count = 0
    skipped_count = 0
    skipped_no_teacher_neg = 0
    skipped_no_plausible_neg = 0

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in tqdm(loader):
                input_seqs = batch["input_seq"].to(args.device)  # [B, max_len]
                target_idxs = batch["target_idx"].numpy()        # [B]
                uids = batch["uid"]                              # list[str]

                # teacher full logits
                logits = model.predict_full(input_seqs)          # [B, item_size]

                # topK candidates
                top_scores, top_indices = torch.topk(logits, k=args.top_k_candidates, dim=1)
                top_scores = top_scores.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

                # 遍历 batch
                for i in range(len(uids)):
                    uid = str(uids[i])
                    gt_idx = int(target_idxs[i])
                    candidates = top_indices[i]
                    cand_scores = top_scores[i]

                    full_history_set = user_history_lookup.get(uid, set())

                    # 1) teacher-preferred neg（不加 meta 约束）
                    teacher_neg_idx = None
                    teacher_neg_rank = None
                    teacher_neg_score = None

                    for r, cand in enumerate(candidates):
                        cand = int(cand)
                        if cand == 0:
                            continue
                        if cand == gt_idx:
                            continue
                        if cand in full_history_set:
                            continue
                        teacher_neg_idx = cand
                        teacher_neg_rank = r + 1
                        teacher_neg_score = float(cand_scores[r])
                        break

                    if teacher_neg_idx is None:
                        skipped_no_teacher_neg += 1
                        skipped_count += 1
                        continue

                    # 2) plausible neg（加 meta 约束：同类/近距离）
                    plausible_neg_idx = None
                    plausible_neg_rank = None
                    plausible_neg_score = None
                    plausible_dist_km = None
                    plausible_same_cat = None

                    # 获取 GT meta
                    gt_gmap_id = id2item.get(gt_idx) or id2item.get(int(gt_idx))
                    if not gt_gmap_id:
                        skipped_count += 1
                        continue
                    gt_meta = meta_manager.get_meta(gt_gmap_id)
                    gt_cat0, gt_lat, gt_lon = gt_meta.get("cat0"), gt_meta.get("lat"), gt_meta.get("lon")

                    # 在 teacher topK 中，找第一个满足约束的
                    for r, cand in enumerate(candidates):
                        cand = int(cand)
                        if cand == 0:
                            continue
                        if cand == gt_idx:
                            continue
                        if cand in full_history_set:
                            continue

                        cand_gmap_id = id2item.get(cand) or id2item.get(int(cand))
                        if not cand_gmap_id:
                            continue
                        cand_meta = meta_manager.get_meta(cand_gmap_id)
                        cand_cat0, cand_lat, cand_lon = cand_meta.get("cat0"), cand_meta.get("lat"), cand_meta.get("lon")

                        same_cat = (gt_cat0 is not None and cand_cat0 is not None and gt_cat0 == cand_cat0)
                        dist = haversine_km(gt_lat, gt_lon, cand_lat, cand_lon)

                        # 判断约束
                        if args.require_same_category and not same_cat:
                            continue

                        if args.allow_either_cat_or_dist:
                            ok = same_cat or (dist is not None and dist <= args.dist_km)
                        else:
                            ok = same_cat and (dist is not None and dist <= args.dist_km)

                        if not ok:
                            continue

                        plausible_neg_idx = cand
                        plausible_neg_rank = r + 1
                        plausible_neg_score = float(cand_scores[r])
                        plausible_dist_km = dist
                        plausible_same_cat = bool(same_cat)
                        break

                    if plausible_neg_idx is None:
                        skipped_no_plausible_neg += 1
                        # 注意：方案 C 不要求每条都有 plausible neg
                        # 你可以选择：仍然写出 teacher_neg，plausible 置空
                        # 或者 continue 丢弃该条。这里选择写出 teacher_neg，plausible 置空。
                        plausible_neg_rank = None
                        plausible_neg_score = None
                        plausible_dist_km = None
                        plausible_same_cat = None

                    # 3) 转换 idx -> gmap_id
                    teacher_neg_gmap_id = id2item.get(teacher_neg_idx) or id2item.get(int(teacher_neg_idx))
                    plausible_neg_gmap_id = None
                    if plausible_neg_idx is not None:
                        plausible_neg_gmap_id = id2item.get(plausible_neg_idx) or id2item.get(int(plausible_neg_idx))

                    if not teacher_neg_gmap_id:
                        skipped_count += 1
                        continue

                    # 4) 文本
                    gt_text = meta_manager.get_text(gt_gmap_id)
                    teacher_neg_text = meta_manager.get_text(teacher_neg_gmap_id)
                    plausible_neg_text = meta_manager.get_text(plausible_neg_gmap_id) if plausible_neg_gmap_id else None

                    # 5) prompt history（最近 N 个）
                    seq_np = input_seqs[i].detach().cpu().numpy()
                    valid_hist_idxs = seq_np[seq_np > 0]
                    recent_idxs = valid_hist_idxs[-args.recent_hist_n:]

                    hist_texts = []
                    for idx in recent_idxs:
                        gmap_id = id2item.get(int(idx)) or id2item.get(int(idx))
                        if gmap_id:
                            hist_texts.append(meta_manager.get_text(gmap_id))

                    if not hist_texts:
                        skipped_count += 1
                        continue
                    hist_str = " -> ".join(hist_texts)

                    # 6) 同时记录 gt/neg score
                    gt_score = float(logits[i, gt_idx].detach().cpu().item())

                    # 7) 写 JSONL（双负样本）
                    sample = {
                        "prompt": f"User History: {hist_str}\nPredict the next location:",
                        "prompt_augment": f"Trajectory: {hist_str}. Suggest the next likely stop:",
                        "completion": gt_text,

                        # 负样本 A：teacher-preferred（更适合作为 shaping/偏好方向）
                        "negative_completion_teacher": teacher_neg_text,

                        # 负样本 B：plausible（同类/近距离，更适合“可替代但不正确”）
                        # 可能为空（None），训练时要处理
                        "negative_completion_plausible": plausible_neg_text,

                        "ips_weight": 1.0,
                        "meta": {
                            "user_id": uid,
                            "target_id": gt_gmap_id,

                            "teacher_neg_id": teacher_neg_gmap_id,
                            "teacher_neg_rank": teacher_neg_rank,
                            "teacher_neg_score": teacher_neg_score,

                            "plausible_neg_id": plausible_neg_gmap_id,
                            "plausible_neg_rank": plausible_neg_rank,
                            "plausible_neg_score": plausible_neg_score,
                            "plausible_dist_km": plausible_dist_km,
                            "plausible_same_cat": plausible_same_cat,

                            "gt_score": gt_score
                        }
                    }

                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    generated_count += 1

    print("========================================")
    print(f"✅ Generated {generated_count} samples.")
    print(f"⚠️ Skipped total: {skipped_count}")
    print(f"   - no teacher neg:   {skipped_no_teacher_neg}")
    print(f"   - no plausible neg: {skipped_no_plausible_neg} (still written with plausible=None)")
    print(f"💾 Saved to: {args.output_file}")
    print("========================================")


if __name__ == "__main__":
    main()
