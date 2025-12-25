import torch
import os
import json
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 配置 =================
# 基础模型 (和训练时一致)
BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
# SFT 训练后的 LoRA 路径
LORA_PATH = "/workspace/Rank-GRPO/SFT/sft_output_coin/checkpoint-44000"
# 测试数据 (从 balanced_train 里切一部分，或者如果有单独的 test set)
# 这里为了演示，我们直接从 train 里面采样，或者你可以指定单独的文件
TEST_DATA_FILE = "./SFT/sft_data/sft_balanced_train.jsonl"
# 评估样本数 (设为 None 则跑全量，建议先跑 1000 条看效果)
NUM_SAMPLES = 1000
# 输出结果
OUTPUT_RESULT_FILE = "./SFT/eval_results.csv"

# 生成参数
GEN_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,       # 评估时建议贪婪搜索 (Greedy)，结果更稳定
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.05
}

def load_model_and_tokenizer():
    print(f"🔄 Loading Tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # [关键] 生成任务必须左侧 Padding，否则 Batch 生成会错位
    tokenizer.padding_side = "left" 
    
    print(f"🔄 Loading Base Model from {BASE_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    print(f"🔄 Loading LoRA Adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    return model, tokenizer

def extract_target_code(text):
    """从模型输出中提取 Target Code"""
    # 匹配 "Target: 12 34 56 78"
    match = re.search(r"Target:\s*([\d\s]+)", text)
    if match:
        return match.group(1).strip()
    return None

def calculate_metrics(predictions, references):
    """计算分层准确率"""
    level_1_hits = 0
    level_2_hits = 0
    level_3_hits = 0
    level_4_hits = 0 # Exact Match
    valid_format_count = 0
    
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        if pred is None: continue # 格式错误
        valid_format_count += 1
        
        pred_parts = pred.split()
        ref_parts = ref.split()
        
        # 确保长度足够进行比较
        p_len = len(pred_parts)
        r_len = len(ref_parts)
        
        if p_len >= 1 and r_len >= 1 and pred_parts[0] == ref_parts[0]:
            level_1_hits += 1
        if p_len >= 2 and r_len >= 2 and pred_parts[:2] == ref_parts[:2]:
            level_2_hits += 1
        if p_len >= 3 and r_len >= 3 and pred_parts[:3] == ref_parts[:3]:
            level_3_hits += 1
        if p_len >= 4 and r_len >= 4 and pred_parts[:4] == ref_parts[:4]:
            level_4_hits += 1

    return {
        "Format_Rate": valid_format_count / total,
        "Acc_L1 (Region)": level_1_hits / total,
        "Acc_L2 (City/Area)": level_2_hits / total,
        "Acc_L3 (Neighborhood)": level_3_hits / total,
        "Acc_L4 (Exact POI)": level_4_hits / total,
        "Total_Samples": total
    }

def main():
    model, tokenizer = load_model_and_tokenizer()
    
    print(f"📂 Loading Dataset from {TEST_DATA_FILE}...")
    dataset = load_dataset("json", data_files=TEST_DATA_FILE, split="train")
    
    # 如果指定了采样数，随机采样
    if NUM_SAMPLES and NUM_SAMPLES < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    
    print(f"📊 Evaluating on {len(dataset)} samples...")
    
    results = []
    batch_size = 32 # 根据显存调整，3090/4090 可以开到 32 或 64
    
    prompts = []
    ground_truths = []
    original_records = []

    # 1. 准备数据
    for item in dataset:
        # 构造 ChatML 格式输入
        prompt_text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt_text)
        ground_truths.append(item['raw_target_code'])
        original_records.append(item)

    # 2. 批量推理
    generated_outputs = []
    
    # 分 Batch 处理
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **GEN_CONFIG,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码 (只保留生成部分)
        batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for j, full_text in enumerate(batch_outputs):
            # 去掉 Prompt 部分，只留回答
            # Qwen output 包含 input，需要截断
            # 注意：batch_decode 可能会自动去掉 special tokens，导致 split 比较麻烦
            # 这里简单处理：取 assistant 后的内容
            if "assistant\n" in full_text:
                response = full_text.split("assistant\n")[-1].strip()
            else:
                response = full_text # Fallback
            
            generated_outputs.append(response)

    # 3. 解析与评估
    parsed_preds = [extract_target_code(text) for text in generated_outputs]
    
    metrics = calculate_metrics(parsed_preds, ground_truths)
    
    print("\n" + "="*30)
    print("📈 Evaluation Metrics")
    print("="*30)
    for k, v in metrics.items():
        if "Acc" in k or "Rate" in k:
            print(f"{k:<25}: {v:.2%}")
        else:
            print(f"{k:<25}: {v}")
    print("="*30)
    
    # 4. 保存详细结果 (方便分析 Bad Case)
    df = pd.DataFrame({
        "User_Loc": [r.get('prompt', '').split('Location:')[-1].split(',')[0] if 'Location:' in r.get('prompt', '') else 'N/A' for r in original_records],
        "GT_Code": ground_truths,
        "Pred_Code": parsed_preds,
        "Generated_Text": generated_outputs,
        "Is_Exact_Match": [p == g for p, g in zip(parsed_preds, ground_truths)]
    })
    
    df.to_csv(OUTPUT_RESULT_FILE, index=False)
    print(f"💾 Detailed results saved to {OUTPUT_RESULT_FILE}")
    
    # 打印几个样例
    print("\n🔍 Sample Predictions:")
    for i in range(min(3, len(df))):
        print(f"\n[Case {i}]")
        print(f"GT  : {df.iloc[i]['GT_Code']}")
        print(f"Pred: {df.iloc[i]['Pred_Code']}")
        print(f"Text: {df.iloc[i]['Generated_Text'][:100]}...")

if __name__ == "__main__":
    main()