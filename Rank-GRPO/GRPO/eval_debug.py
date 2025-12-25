import os
import json
import torch
import re
import math
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= ⚙️ Debug 配置 =================
CKPT_PATH = "./GRPO/output_sarank_stage2/checkpoint-300" # 换成你崩掉的那个 checkpoint
DATA_PATH = "./SFT/sft_data/sft_balanced_train.jsonl"
ID_MAP_FILE = "./poi_semantic_ids.csv"
NUM_TEST_SAMPLES = 20 # 只需要看前20个就足够分析了
# =================================================

def extract_target_debug(text):
    """提取 Target 并返回原始匹配结果"""
    # 尝试匹配标准格式
    m = re.search(r"Target:\s*([\d\s]+)", text)
    if m: 
        return m.group(1).strip(), "Valid Format"
    
    # 如果没找到，看看是不是根本没写 Target
    if "Target:" in text:
        return None, "Has Header but No ID"
    
    return None, "No Header Found"

def main():
    # 1. 简单的 ID 校验器 (只加载 ID Map，不加载大元数据，为了快)
    print("🚀 Loading ID Map for validation...")
    valid_ids = set()
    if os.path.exists(ID_MAP_FILE):
        df = pd.read_csv(ID_MAP_FILE)
        for _, row in df.iterrows():
            code = f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}"
            valid_ids.add(code)
    else:
        print("⚠️ ID Map not found, cannot validate 'Bad ID'")

    # 2. 加载模型
    print(f"🔄 Loading Model: {CKPT_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(CKPT_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.eval()
    except Exception as e:
        print(f"Load Error: {e}")
        return

    # 3. 准备数据
    dataset = []
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()[-NUM_TEST_SAMPLES:]
        for line in lines:
            item = json.loads(line)
            dataset.append(item)

    print(f"\n{'='*40} DEBUG START {'='*40}")
    
    for i, item in enumerate(dataset):
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # 必须用 Greedy 来复现问题
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("assistant\n")[-1].strip()
        
        pred_code, status = extract_target_debug(response)
        gt_code = item['raw_target_code']
        
        # 判断 ID 是否真实存在
        id_status = "Unknown"
        if pred_code:
            if pred_code in valid_ids:
                id_status = "✅ Valid DB ID"
            else:
                id_status = "❌ Hallucinated ID" # 格式对，但 ID 不在库里

        # === 核心分析打印 ===
        print(f"\n🔸 Sample {i}")
        print(f"   [GT ID]:   {gt_code}")
        print(f"   [Pred ID]: {pred_code if pred_code else 'None'} ({status})")
        if pred_code:
             print(f"   [DB Check]: {id_status}")
        
        print(f"   [Model Raw Output (First 150 chars)]: ")
        print(f"   👉 {response[:150].replace(chr(10), ' ')} ...") # 把换行符替换为空格，方便看
        print("-" * 80)

if __name__ == "__main__":
    main()