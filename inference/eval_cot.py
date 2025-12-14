import os
import json
import re
import torch
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================================================================
# 1. Configuration
# ==============================================================================

# [配置] 这里的路径要和你训练时的设置对齐
BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
ADAPTER_PATH = "/workspace/data/grpo_v4_4_cot/checkpoint-2000"
TEST_DATA_PATH = "/workspace/data/processed/test_prompts.jsonl" # 或者是 eval_prompts.jsonl
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
OUTPUT_FILE = "/workspace/data/grpo_v4_4_cot/eval_results.json"

# 生成参数 (CoT 需要长一点的 token 数来容纳思考过程)
GEN_PARAMS = {
    "max_new_tokens": 128,  # 给足空间让它说话
    "temperature": 0.0,     # Eval 时通常设为 0 以保证稳定复现
    "top_p": 1.0,
    "do_sample": False      # 贪婪解码
}

# ✅ 修改后的代码（强制它输出数字）
COT_SUFFIX = (
    "Step 1: Predict the category name of the next item.\n"
    "Step 2: Output the semantic ID. \n"
    "Example 1: User likes Fast Food -> ID: <12, 5, 88, 10>\n"
    "Example 2: User likes Gyms -> ID: <4, 22, 11, 7>\n"
    "Important: The ID must be a sequence of 4 integers.\n"
    "Response: The user is interested in" 
)

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def load_global_mapping(mapping_file):
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        sid_map = json.load(f)
    
    # 构建 ID -> Metadata 的快速查找表
    tree_map = {}
    for bid, meta in sid_map.items():
        # 假设 full_sid 是 list [12, 5, 88, 10]
        full_code = tuple(int(x) for x in meta['full_sid'])
        tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta.get('city', 'Unknown'),
            'categories': meta.get('categories', '')
        }
    return tree_map

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_output_cot(text):
    """
    解析 CoT 输出。
    输出可能包含: "User likes Spicy Food. The ID is <12, 5, 88, 10>."
    我们需要提取最后的 <...> 部分。
    """
    # 1. 提取 ID (正则寻找最后一次出现的 ID 模式)
    # 模式匹配: <数字, 数字, 数字, 数字>，允许空格
    match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    
    pred_id = None
    if match:
        pred_id = tuple(int(g) for g in match.groups())
    
    # 2. (可选) 提取思考过程，即 < 之前的所有文本
    reasoning_text = ""
    if match:
        reasoning_text = text[:match.start()].strip()
    else:
        reasoning_text = text.strip() # 如果没找到 ID，整个都是废话
        
    return pred_id, reasoning_text

# ==============================================================================
# 3. Evaluation Logic
# ==============================================================================

def main():
    # 1. Load Resources
    tree_map = load_global_mapping(MAPPING_FILE)
    
    print(f"Loading Model: {BASE_MODEL_PATH} + Adapter: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    # 加载 LoRA
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # 2. Load Data
    print(f"Loading Test Data: {TEST_DATA_PATH}")
    test_samples = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    # 缩小规模用于快速测试 (可选)
    test_samples = test_samples[:100] 

    metrics = {
        "total": 0,
        "format_error": 0,  # 没生成 <ID> 的
        "hallucination": 0, # 生成了 <ID> 但不在库里的
        "hit_rate_l1": 0,   # 第一层对 (Region)
        "hit_rate_l2": 0,   # 第二层对 (City/District)
        "hit_rate_l3": 0,   # 第三层对 (Category)
        "hit_rate_exact": 0, # 完全对
        "geo_distance_sum": 0.0 # 距离误差总和
    }
    
    results_log = []

    print("🚀 Starting CoT Evaluation...")
    
    # Batch 处理
    BATCH_SIZE = 16
    for i in tqdm(range(0, len(test_samples), BATCH_SIZE)):
        batch_items = test_samples[i : i + BATCH_SIZE]
        
        # 构造 Prompt：必须包含 CoT Suffix
        batch_prompts = []
        batch_targets = []
        batch_target_coords = []
        
        for item in batch_items:
            raw_inst = item.get('instruction', '')
            if "Response:" in raw_inst:
                base_prompt = raw_inst.split("Response:")[0].strip()
            else:
                base_prompt = raw_inst
            
            # 【关键】拼接 CoT 引导词
            full_prompt = f"{base_prompt}\n{COT_SUFFIX}"
            batch_prompts.append(full_prompt)
            
            meta = item.get('metadata', {})
            batch_targets.append(meta.get('target_sid'))
            batch_target_coords.append((meta.get('target_lat'), meta.get('target_lon')))

        # Tokenize & Generate
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                **GEN_PARAMS
            )
        
        # Decode
        # 只保留新生成的部分
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Compute Metrics
        for idx, text in enumerate(decoded_outputs):
            metrics["total"] += 1
            t_sid = tuple(batch_targets[idx]) if batch_targets[idx] else None
            t_lat, t_lon = batch_target_coords[idx]
            
            # 解析 CoT
            pred_id, reasoning = parse_output_cot(text)
            
            log_item = {
                "reasoning": reasoning, # 保存思考过程
                "pred_id": str(pred_id),
                "target_id": str(t_sid),
                "correct": False
            }

            if not pred_id:
                metrics["format_error"] += 1
                results_log.append(log_item)
                continue
            
            if pred_id not in tree_map:
                metrics["hallucination"] += 1
                # 依然尝试计算前缀匹配 (如果 ID 结构合法)
            else:
                # 计算地理距离
                pred_meta = tree_map[pred_id]
                if t_lat is not None:
                    dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                    metrics["geo_distance_sum"] += dist
            
            # 层级匹配指标
            if t_sid:
                if len(pred_id) >= 1 and pred_id[0] == t_sid[0]:
                    metrics["hit_rate_l1"] += 1
                    if len(pred_id) >= 2 and pred_id[1] == t_sid[1]:
                        metrics["hit_rate_l2"] += 1
                        if len(pred_id) >= 3 and pred_id[2] == t_sid[2]:
                            metrics["hit_rate_l3"] += 1 # 关键指标：类别是否对
                            if len(pred_id) >= 4 and pred_id[3] == t_sid[3]:
                                metrics["hit_rate_exact"] += 1
                                log_item["correct"] = True

            results_log.append(log_item)

    # 4. Summary & Save
    final_score = {
        "Total Samples": metrics["total"],
        "Format Error Rate": f"{metrics['format_error'] / metrics['total']:.2%}",
        "Hallucination Rate": f"{metrics['hallucination'] / metrics['total']:.2%}",
        "HR@L1 (Region)": f"{metrics['hit_rate_l1'] / metrics['total']:.2%}",
        "HR@L2 (City/Dist)": f"{metrics['hit_rate_l2'] / metrics['total']:.2%}",
        "HR@L3 (Category)": f"{metrics['hit_rate_l3'] / metrics['total']:.2%}",
        "HR@Exact (Item)": f"{metrics['hit_rate_exact'] / metrics['total']:.2%}",
        "Avg Geo Distance (km)": f"{metrics['geo_distance_sum'] / (metrics['total'] - metrics['format_error'] + 1e-6):.2f}"
    }

    print("\n" + "="*40)
    print("📊 Evaluation Results (CoT Scheme)")
    print("="*40)
    for k, v in final_score.items():
        print(f"{k:<20}: {v}")
    print("="*40)
    
    # 打印几个真实的例子来看看推理质量
    print("\n🔍 Qualitative Examples:")
    for i in range(min(5, len(results_log))):
        ex = results_log[i]
        print(f"Ref: {ex['target_id']}")
        print(f"CoT: {ex['reasoning']}") # 看看模型说了什么
        print(f"Out: {ex['pred_id']}")
        print("-" * 20)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": final_score, "details": results_log}, f, indent=2, ensure_ascii=False)
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()