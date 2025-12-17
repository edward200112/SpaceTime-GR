import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from tqdm import tqdm
import pandas as pd

# ================= 配置 =================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_DIR = "./SFT/sft_output/checkpoint-epoch-1" # 假设跑完第一轮
TRAIN_DATA = "./SFT/sft_data/sft_enhanced_train.jsonl"
OUTPUT_DATA = "./SFT/sft_data/sft_dynamic_epoch2.jsonl"
ID_MAPPING = "./poi_semantic_ids.csv"

def mine_false_positives():
    print("🚀 Starting Dynamic Model-Aware Mining (Phase 3.2.2)...")
    
    # 1. 加载上一轮的模型
    print("Loading Model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, CHECKPOINT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model.eval()
    
    # 2. 加载 ID 列表用于校验
    id_df = pd.read_csv(ID_MAPPING)
    valid_codes = set()
    for _, r in id_df.iterrows():
        valid_codes.add(f"{r['code_0']} {r['code_1']} {r['code_2']} {r['code_3']}")

    # 3. 遍历训练数据，找 False Positives
    new_samples = []
    print("Scanning Training Data for Hard Negatives...")
    
    with open(TRAIN_DATA, 'r') as f:
        # 为了演示，只处理前 5000 条，全量处理极慢
        lines = f.readlines()[:5000] 
    
    for line in tqdm(lines):
        item = json.loads(line)
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # 生成 Top-3 预测
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                num_return_sequences=3, 
                num_beams=3, 
                do_sample=False
            )
        
        gt_code = item['completion']
        hard_negative = None
        
        # 检查 Top-3 里的错误答案
        for out in outputs:
            pred = tokenizer.decode(out, skip_special_tokens=True).split("assistant\n")[-1].strip()
            
            # 如果预测不仅错了，而且是一个有效的 POI ID (不是乱码)，且不是 GT
            if pred != gt_code and pred in valid_codes:
                hard_negative = pred
                break # 找到了一个 False Positive
        
        # 更新样本
        if hard_negative:
            # 用动态挖掘出的 False Positive 替换原来的 Negative
            item['negative_completion'] = hard_negative
            # 可以在这里打个标记
            item['is_dynamic_hard'] = True
            
        new_samples.append(item)
    
    # 4. 保存新一轮训练数据
    print(f"💾 Saving Dynamic Hard Negatives to {OUTPUT_DATA}")
    with open(OUTPUT_DATA, 'w') as f:
        for item in new_samples:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    mine_false_positives()