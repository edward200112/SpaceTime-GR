import json
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. Yelp18 的文件
YELP18_ITEM_LIST = "/workspace/yelp18/item_list.txt"

# 2. SFT 训练时生成的 Mapping 文件 (必须有这个，否则模型不知道 <c0> 是啥)
#    请确认这个路径！这是你之前训练 GRPO 时用的那个 mapping
TRAIN_MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# 3. Yelp 原始数据集 (用于补充名字)
#    通常叫 yelp_academic_dataset_business.json
RAW_YELP_FILE = "/workspace/data/raw/yelp_academic_dataset_business.json"

# 4. 输出文件
OUTPUT_FILE = "./yelp18/yelp18_to_semantic_full.json"
# ===========================================

def load_yelp18_map():
    print(f"Loading Yelp18 ID Map from {YELP18_ITEM_LIST}...")
    remap2org = {} # 0 -> "na4Th..."
    with open(YELP18_ITEM_LIST, 'r') as f:
        # 跳过第一行表头 (org_id remap_id)
        header = next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                org_id, remap_id = parts[0], int(parts[1])
                remap2org[remap_id] = org_id
    print(f"Loaded {len(remap2org)} items from Yelp18.")
    return remap2org

def load_semantic_map():
    print(f"Loading SFT Semantic Map from {TRAIN_MAPPING_FILE}...")
    with open(TRAIN_MAPPING_FILE, 'r') as f:
        # 结构: "na4Th...": {"full_sid": [12, 5, 3, 10], ...}
        sid_map = json.load(f)
    print(f"Loaded {len(sid_map)} items from SFT Mapping.")
    return sid_map

def load_raw_names(target_org_ids):
    print(f"Loading Raw Names from {RAW_YELP_FILE}...")
    org2name = {}
    target_set = set(target_org_ids)
    
    with open(RAW_YELP_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            bid = item['business_id']
            if bid in target_set:
                org2name[bid] = item['name']
    return org2name

def main():
    # 1. 加载 Yelp18 映射
    remap2org = load_yelp18_map()
    
    # 2. 加载 SFT 语义映射
    org2semantic = load_semantic_map()
    
    # 3. 加载原始名字
    org_ids = list(remap2org.values())
    org2name = load_raw_names(org_ids)
    
    # 4. 融合
    final_map = {} # remap_id (int) -> {semantic info}
    missing_semantic = 0
    missing_name = 0
    
    for remap_id, org_id in tqdm(remap2org.items(), desc="Merging"):
        # 必须同时存在于 SFT Mapping 中，模型才能预测它
        if org_id in org2semantic:
            sem_info = org2semantic[org_id]
            name = org2name.get(org_id, "Unknown Item")
            if name == "Unknown Item": missing_name += 1
            
            # 构建语义字符串 <c0, c1, c2, item>
            # 注意：remap_id 对应的 item_id 可能和 SFT 的不一样
            # 这里我们用 SFT 的 item_id 还是 Yelp18 的？
            # 关键：模型输出的是 SFT 里的 item_id (suffix)。
            # 我们需要记录这个对应关系。
            
            full_sid = sem_info['full_sid'] # [12, 5, 3, 999]
            
            final_map[str(remap_id)] = {
                "org_id": org_id,
                "name": name,
                "sft_sid_tuple": full_sid, # [c0, c1, c2, sft_suffix]
                "sft_sid_str": f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>" if 'full_code' in locals() else "", 
                # 方便后续 Prompt
                "prompt_name": name 
            }
        else:
            missing_semantic += 1
    
    print("="*40)
    print(f"Total Yelp18 Items: {len(remap2org)}")
    print(f"Mapped Successfully: {len(final_map)}")
    print(f"Missing in SFT Map: {missing_semantic} (These items were not in your training set!)")
    print(f"Missing Names: {missing_name}")
    print("="*40)
    
    # 保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved merged mapping to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()