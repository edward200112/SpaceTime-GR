import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置路径 (请确保与数据生成脚本一致) =================
OUTPUT_DIR = "/workspace/data/processed_pinrec_v2"
TRAIN_DATA_FILE = os.path.join(OUTPUT_DIR, "train_ultimate.jsonl")
FEATS_FILE = os.path.join(OUTPUT_DIR, "item_content_feats.npy")

# 只检查前 N 个样本
NUM_SAMPLES_TO_CHECK = 5 
# =====================================================================

def check_file_integrity():
    """ 检查核心文件是否存在及其维度。"""
    print("--- 1. 文件完整性与维度检查 ---")
    
    # 检查训练数据文件
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ 错误：找不到训练数据文件: {TRAIN_DATA_FILE}")
        return None, None
    print(f"✅ 找到训练数据文件: {TRAIN_DATA_FILE}")

    # 检查特征矩阵文件
    if not os.path.exists(FEATS_FILE):
        print(f"❌ 错误：找不到特征矩阵文件: {FEATS_FILE}")
        return None, None
    
    try:
        item_feats = np.load(FEATS_FILE)
        num_items, feat_dim = item_feats.shape
        print(f"✅ 找到特征矩阵。维度: {item_feats.shape}")
        print(f"   总 Item 数量 (索引范围): {num_items}")
        print(f"   特征维度 (content_dim): {feat_dim}")
        return item_feats, num_items
    except Exception as e:
        print(f"❌ 错误：加载特征矩阵失败: {e}")
        return None, None

def check_data_alignment_and_structure(num_items):
    """ 检查 ID 对齐和样本结构。"""
    print("\n--- 2. ID 对齐与样本结构检查 ---")
    
    samples = []
    with open(TRAIN_DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES_TO_CHECK: break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ 错误：JSON 解析失败 (行 {i+1}): {e}")
                return

    if not samples:
        print("❌ 错误：训练文件为空或无法读取样本。")
        return

    max_id_found = 0
    for i, s in enumerate(samples):
        print(f"\n--- 样本 {i+1} ---")
        
        # --- A. History 结构检查 ---
        h_ids = s.get('history_ids', [])
        h_acts = s.get('history_acts', [])
        h_deltas = s.get('history_deltas', [])
        
        print(f"  History 长度: {len(h_ids)} (IDs, Acts, Deltas 长度需一致)")
        if not (len(h_ids) == len(h_acts) == len(h_deltas)):
            print("❌ 结构错误：History IDs/Acts/Deltas 长度不一致！")
        
        # 提取 History 中的最大 ID 和时间差
        if h_ids:
            max_id_found = max(max_id_found, max(h_ids))
            
            # 检查时间差是否合理 (非负)
            if any(d < 0 for d in h_deltas):
                print("❌ 数据错误：History Delta 中存在负值！")
            
            print(f"  History 尾部: ID={h_ids[-1]}, Act={h_acts[-1]}, Delta={h_deltas[-1]:.2f}s")


        # --- B. Target 结构和 ID 检查 ---
        for target_key in ['target_1', 'target_2']:
            t = s.get(target_key, {})
            t_id = t.get('id', -1)
            t_delta = t.get('delta', -1.0)
            t_act = t.get('act', -1)
            
            print(f"  Target [{target_key.split('_')[-1]}]: ID={t_id}, Act={t_act}, Delta={t_delta:.2f}s")

            # 检查 Target ID
            if t_id >= num_items or t_id < 0:
                print(f"❌ ID 错位：{target_key} 的 ID ({t_id}) 超出特征矩阵范围 (0-{num_items-1})！")
            else:
                print(f"✅ ID 范围检查通过。")
                max_id_found = max(max_id_found, t_id)
            
            # 检查 Target Delta
            if t_delta <= 0:
                 print(f"❌ Delta 错误：{target_key} 的 Delta ({t_delta:.2f}s) 必须大于 0 (预测未来)！")


    # --- C. 总结 ---
    if num_items:
        print("\n--- 总结 ---")
        if max_id_found < num_items:
            print(f"✅ ID 范围：最大使用的 ID ({max_id_found}) < 总 Item 数 ({num_items})。对齐成功。")
        else:
             print(f"❌ ID 错位：发现 ID ({max_id_found}) >= 总 Item 数 ({num_items})。")


def main_check():
    item_feats, num_items = check_file_integrity()
    if num_items:
        check_data_alignment_and_structure(num_items)

if __name__ == "__main__":
    main_check()