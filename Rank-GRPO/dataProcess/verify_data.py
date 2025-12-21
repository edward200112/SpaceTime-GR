import torch
import numpy as np
import pandas as pd
import os

def verify_dataset(data_dir="./processed_data"):
    print("🕵️ Starting Data Verification Checks...\n")
    
    # 1. 检查文件是否存在
    data_path = os.path.join(data_dir, "train_data.pt")
    state_class_path = os.path.join(data_dir, "state_classes.npy")
    city_class_path = os.path.join(data_dir, "city_classes.npy")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Main data file not found: {data_path}")
    
    # 2. 加载数据
    print(f"📥 Loading {data_path} ...")
    data_dict = torch.load(data_path, weights_only=False)
    
    features = data_dict['features']
    state_ids = data_dict['state_ids']
    city_ids = data_dict['city_ids']
    gmap_ids = data_dict['gmap_ids']
    # 检查是否有我们保存的 state_names 用于核对
    state_names_saved = data_dict.get('state_names', None) 
    
    state_classes = np.load(state_class_path, allow_pickle=True)
    city_classes = np.load(city_class_path, allow_pickle=True)
    
    # 3. 维度检查 (Shape Check)
    N = features.shape[0]
    dim = features.shape[1]
    print("-" * 40)
    print("✅ [Basic Shapes]")
    print(f"  Total Samples (N): {N}")
    print(f"  Feature Dimension: {dim}")
    print(f"  Num States:        {len(state_classes)}")
    print(f"  Num Cities:        {len(city_classes)}")
    
    assert len(state_ids) == N, "State IDs length mismatch!"
    assert len(city_ids) == N, "City IDs length mismatch!"
    
    # 4. 类别分布检查 (Distribution Check)
    print("\n✅ [State Distribution]")
    # 统计每个ID出现的次数
    unique_ids, counts = torch.unique(state_ids, return_counts=True)
    
    found_states = []
    for uid, count in zip(unique_ids, counts):
        state_name = state_classes[uid.item()]
        found_states.append(state_name)
        percentage = (count.item() / N) * 100
        print(f"  State ID {uid.item()} ({state_name}): {count.item()} samples ({percentage:.2f}%)")
    
    # 验证是否只有目标的4个州
    expected_states = {'California', 'New York', 'New Mexico', 'Pennsylvania'}
    loaded_states_set = set(found_states)
    
    if loaded_states_set == expected_states:
        print("  🎉 PERFECT! Exactly the 4 target states found.")
    else:
        print(f"  ⚠️ WARNING: Found states {loaded_states_set}. Expected {expected_states}")
        missing = expected_states - loaded_states_set
        extra = loaded_states_set - expected_states
        if missing: print(f"    Missing: {missing}")
        if extra: print(f"    Extra (Noise): {extra}")

    # 5. 特征质量检查 (Feature Quality)
    print("\n✅ [Feature Health]")
    # 检查 NaN / Inf
    if torch.isnan(features).any() or torch.isinf(features).any():
        print("  ❌ FATAL: Features contain NaNs or Infs!")
    else:
        print("  Clean (No NaNs/Infs).")
        
    # 检查数值范围 (SBERT通常是归一化的或者数值较小，Lat/Lon编码在[-1, 1])
    print(f"  Min Value: {features.min():.4f}")
    print(f"  Max Value: {features.max():.4f}")
    print(f"  Mean Value: {features.mean():.4f}")
    
    # 6. 随机样本抽样 (Sanity Check)
    print("\n✅ [Random Sample Inspection]")
    indices = torch.randint(0, N, (3,))
    for idx in indices:
        i = idx.item()
        s_id = state_ids[i].item()
        c_id = city_ids[i].item()
        s_name = state_classes[s_id]
        c_name = city_classes[c_id]
        
        # 如果保存了原始名字，进行对比
        original_state_str = state_names_saved[i] if state_names_saved is not None else "N/A"
        
        print(f"  Sample {i}:")
        print(f"    Gmap ID: {gmap_ids[i]}")
        print(f"    Mapped:  {c_name}, {s_name} (IDs: {c_id}, {s_id})")
        print(f"    Raw:     State='{original_state_str}'")
        # 简单核对
        if state_names_saved is not None:
             if s_name != original_state_str:
                 print(f"    ❌ MAPPING ERROR: Class '{s_name}' != Raw '{original_state_str}'")
             else:
                 print(f"    ✨ Mapping Correct")
        print("-" * 20)

if __name__ == "__main__":
    try:
        verify_dataset()
        print("\n🚀 Data verification passed. You are ready to train!")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")