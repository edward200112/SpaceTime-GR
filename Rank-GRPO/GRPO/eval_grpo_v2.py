import os
import json
import torch
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= ⚙️ 配置区域 (请修改这里) =================
# 你想测试的 Checkpoint 路径 (Stage 1 或 Stage 2 的输出目录)
CKPT_PATH = "./GRPO/output_sarank_stage3/checkpoint-100" 

# 数据文件路径
DATA_PATH = "./SFT/sft_data/sft_balanced_train.jsonl"
ID_MAP_FILE = "./poi_semantic_ids.csv"
RAW_DATA_DIR = "/workspace/data/GoogleRAW"
META_FILES = [
    "meta-California.json.gz", "meta-New_York.json.gz", 
    "meta-New_Mexico.json.gz", "meta-Pennsylvania.json.gz"
]

# 测试样本数量 (建议 50-100)
NUM_TEST_SAMPLES = 50 
# ==========================================================

class MiniDataManager:
    """快速加载坐标数据的工具类"""
    def __init__(self):
        self.poi_db = {}
        print("🚀 Loading ID Map & Metadata (Scanning)...")
        if not os.path.exists(ID_MAP_FILE):
            print(f"⚠️ Error: ID Map file not found at {ID_MAP_FILE}")
            return
            
        # 1. 加载 ID 映射
        df = pd.read_csv(ID_MAP_FILE)
        # 创建 gmap_id -> hierarchical code 的映射
        gmap2code = {str(row['gmap_id']): f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}" for _, row in df.iterrows()}
        valid_gmaps = set(gmap2code.keys())

        # 2. 加载元数据 (只加载有坐标的)
        import gzip
        count = 0
        for m_file in META_FILES:
            path = os.path.join(RAW_DATA_DIR, m_file)
            if not os.path.exists(path): continue
            print(f"   - Scanning {m_file}...")
            with gzip.open(path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        if gid in valid_gmaps:
                            code = gmap2code[gid]
                            self.poi_db[code] = {
                                'loc': (float(d['latitude']), float(d['longitude'])),
                                'name': d.get('name', 'Unknown'),
                                'address': d.get('address', '')
                            }
                            count += 1
                    except: continue
        print(f"✅ Loaded {count} POIs into database.")

    def get_info(self, code): 
        # 尝试完全匹配
        return self.poi_db.get(code)

def haversine_km(loc1, loc2):
    """计算两点经纬度的公里距离"""
    if not loc1 or not loc2: return 9999.0
    R = 6371.0
    dlat = math.radians(loc2[0] - loc1[0])
    dlon = math.radians(loc2[1] - loc1[1])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(loc1[0])) * math.cos(math.radians(loc2[0])) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def extract_target(text):
    """从模型输出中提取 Target ID"""
    # 匹配 "Target: 123 456 789 012" 格式
    m = re.search(r"Target:\s*([\d\s]+)", text)
    if m: 
        code = m.group(1).strip()
        # 简单的清洗，确保只保留数字和空格
        return code
    return None

def main():
    # 1. 准备数据
    dm = MiniDataManager()
    
    # 2. 加载模型 (全量加载模式)
    print(f"🔄 Loading Model from: {CKPT_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            CKPT_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        print("Tip: Ensure the folder contains 'model.safetensors' or 'pytorch_model.bin'")
        return

    # 3. 准备测试集
    print("📊 Sampling Data...")
    dataset = []
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        # 取最后 NUM_TEST_SAMPLES 条，避免和训练集头部的重叠（如果是 shuffled 则无所谓）
        test_lines = lines[-NUM_TEST_SAMPLES:] 
        for line in test_lines:
            item = json.loads(line)
            dataset.append({"prompt": item['prompt'], "gt": item['raw_target_code']})

    # 4. 开始推理
    results = []
    print(f"🚀 Running Inference on {len(dataset)} samples...")
    
    # 表格头
    print(f"\n{'IDX':<4} | {'Valid':<5} | {'Dist(km)':<10} | {'GT Name':<30} | {'Pred Name'}")
    print("-" * 100)

    for i, item in enumerate(dataset):
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # 使用 Greedy Search 以获得最稳定的能力评估
                repetition_penalty=1.2, # 核心：对重复生成的词进行 20% 的降权
                no_repeat_ngram_size=3, # 核心：严禁出现重复的 3-gram 短语
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("assistant\n")[-1].strip()
        
        pred_code = extract_target(response)
        
        # 计算指标
        dist = 9999.0
        gt_info = dm.get_info(item['gt'])
        pred_info = dm.get_info(pred_code) if pred_code else None
        
        if gt_info and pred_info:
            dist = haversine_km(gt_info['loc'], pred_info['loc'])
        
        # 记录结果
        res = {
            "valid_format": bool(pred_code),
            "exact_match": (pred_code == item['gt']),
            "distance": dist,
            "response": response
        }
        results.append(res)
        
        # 实时打印简报
        gt_name = gt_info['name'][:28] if gt_info else "Unknown ID"
        pred_name = pred_info['name'][:28] if pred_info else ("Bad ID" if pred_code else "Format Err")
        dist_str = f"{dist:.2f}" if dist < 9000 else "N/A"
        print(f"{i:<4} | {str(res['valid_format']):<5} | {dist_str:<10} | {gt_name:<30} | {pred_name}")

    # 5. 最终统计报告
    print("\n" + "="*30 + " FINAL REPORT " + "="*30)
    
    valid_count = sum(r['valid_format'] for r in results)
    exact_count = sum(r['exact_match'] for r in results)
    distances = [r['distance'] for r in results if r['distance'] < 9000]
    
    print(f"✅ Format Validity Rate: {valid_count / len(results) * 100:.1f}%")
    print(f"🎯 Exact Match Rate:     {exact_count / len(results) * 100:.1f}%")
    
    if distances:
        print(f"📏 Median Distance Error: {np.median(distances):.2f} km  <-- 核心指标")
        print(f"📏 Mean Distance Error:   {np.mean(distances):.2f} km")
        print(f"🏆 Best Prediction:       {min(distances):.2f} km")
        print(f"📉 Worst Prediction:      {max(distances):.2f} km")
    else:
        print("⚠️ No valid distances computed (Check ID Map or Model Output)")
    
    # 打印一个最好的推理样例
    if distances:
        best_idx = np.argmin([r['distance'] if r['distance'] < 9000 else 99999 for r in results])
        print("\n🌟 Best Reasoning Sample:")
        print(results[best_idx]['response'])
        
    print("="*80)

if __name__ == "__main__":
    main()