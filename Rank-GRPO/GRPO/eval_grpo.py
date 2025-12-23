import os
import json
import torch
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

# ================= 配置区域 =================
# 注意：因为你的 checkpoint-800 是 merge 后的全量模型，
# 我们直接加载这个目录，不需要加载 Base Model 路径。
ADAPTER_PATH = "./GRPO/output_sarank/checkpoint-800" 
DATA_PATH = "./SFT/sft_data/sft_balanced_train.jsonl"
ID_MAP_FILE = "./poi_semantic_ids.csv"
META_FILES = [
    "meta-California.json.gz", "meta-New_York.json.gz", 
    "meta-New_Mexico.json.gz", "meta-Pennsylvania.json.gz"
]
RAW_DATA_DIR = "/workspace/data/GoogleRAW"

NUM_TEST_SAMPLES = 50 # 先设为50个，测试跑通

# ================= 数据工具 =================
class MiniDataManager:
    def __init__(self):
        self.poi_db = {}
        print("🚀 Loading ID Map & Metadata (Scanning)...")
        if not os.path.exists(ID_MAP_FILE):
            print("⚠️ ID Map file missing!")
            return
            
        df = pd.read_csv(ID_MAP_FILE)
        valid_gmaps = set(df['gmap_id'].astype(str).tolist())
        gmap2code = {str(row['gmap_id']): f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}" for _, row in df.iterrows()}

        import gzip
        for m_file in META_FILES:
            path = os.path.join(RAW_DATA_DIR, m_file)
            if not os.path.exists(path): continue
            with gzip.open(path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        if gid in valid_gmaps:
                            code = gmap2code[gid]
                            self.poi_db[code] = {
                                'loc': (float(d['latitude']), float(d['longitude'])),
                                'name': d.get('name', 'Unknown')
                            }
                    except: continue

    def get_info(self, code): return self.poi_db.get(code)

def haversine_km(loc1, loc2):
    if not loc1 or not loc2: return 9999.0
    R = 6371.0
    dlat, dlon = math.radians(loc2[0]-loc1[0]), math.radians(loc2[1]-loc1[1])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(loc1[0]))*math.cos(math.radians(loc2[0]))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def extract_target(text):
    # 增加鲁棒性，匹配多种可能的 Target 格式
    m = re.search(r"Target:\s*(\d+\s+\d+\s+\d+\s+\d+)", text)
    if m: return m.group(1).strip()
    return None

# ================= 主流程 =================
def main():
    dm = MiniDataManager()
    
    print(f"🔄 Loading Merged Full Model from {ADAPTER_PATH}...")
    # 修复 Tokenizer 警告并加载
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    # 直接加载全量模型
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("📊 Preparing Test Samples...")
    dataset = []
    with open(DATA_PATH, 'r') as f:
        all_lines = f.readlines()
        # 抽取最后 NUM_TEST_SAMPLES 条作为评估
        test_lines = all_lines[-NUM_TEST_SAMPLES:]
        for line in test_lines:
            item = json.loads(line)
            dataset.append({"prompt": item['prompt'], "gt": item['raw_target_code']})

    results = []
    print(f"🚀 Running Evaluation...")
    for item in tqdm(dataset):
        prompt_input = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # 提取回答部分
        response = full_text.split("assistant")[-1].strip()
        pred_code = extract_target(response)
        
        # 计算距离
        dist = 9999.0
        gt_info = dm.get_info(item['gt'])
        pred_info = dm.get_info(pred_code) if pred_code else None
        
        if gt_info and pred_info:
            dist = haversine_km(gt_info['loc'], pred_info['loc'])

        results.append({
            "exact": 1 if pred_code == item['gt'] else 0,
            "dist": dist if pred_info else None,
            "valid": 1 if pred_code else 0
        })

    # 汇总显示
    valid_count = sum(r['valid'] for r in results)
    exact_count = sum(r['exact'] for r in results)
    dists = [r['dist'] for r in results if r['dist'] is not None]
    
    print("\n" + "="*20 + " GRPO CKPT-800 REPORT " + "="*20)
    print(f"Format Validity: {valid_count/len(results)*100:.1f}%")
    print(f"Exact Match:     {exact_count/len(results)*100:.1f}%")
    if dists:
        print(f"Avg Distance:    {np.mean(dists):.2f} km")
        print(f"Median Distance: {np.median(dists):.2f} km")
    else:
        print("Avg Distance:    N/A (No valid POI predicted)")
    print("="*60)

if __name__ == "__main__":
    main()