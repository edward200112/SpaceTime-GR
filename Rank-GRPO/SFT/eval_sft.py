import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm

# ================= 配置 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./sft_output/final_checkpoint"
TEST_DATA = "./sft_data/sft_val.jsonl" # 假设你有验证集

def load_model_for_eval():
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loading LoRA Adapters from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model.eval()
    return model, tokenizer

def eval_sft():
    model, tokenizer = load_model_for_eval()
    
    print("Loading Test Data...")
    test_samples = []
    with open(TEST_DATA, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
            if len(test_samples) >= 200: break # 仅测试 200 条做演示
            
    valid_format_count = 0
    hit_count_at_1 = 0
    total = 0
    
    print("🚀 Starting Evaluation...")
    for item in tqdm(test_samples):
        # 构造 Prompt
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, # ID 不会长
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 Assistant 回复部分
        response = generated_text.split("assistant\n")[-1].strip()
        
        ground_truth = item['completion'].strip()
        
        # 1. Format Compliance Check
        # 我们的目标是 Semantic ID 格式 (数字 空格 数字...)
        # 简单检查：是否由数字和空格组成
        is_valid = all(c.isdigit() or c.isspace() for c in response)
        if is_valid and len(response) > 0:
            valid_format_count += 1
            
        # 2. Next-Item Recall@1 (Exact Match)
        # 真实场景应该比较 Semantic ID 的相似度，这里做 Exact Match
        # 因为 ID 是离散的，Exact Match 是最严格的
        if response == ground_truth:
            hit_count_at_1 += 1
            
        total += 1
        
    print("-" * 30)
    print(f"Total Samples: {total}")
    print(f"Format Compliance: {valid_format_count/total:.2%}")
    print(f"Next-Item Recall@1: {hit_count_at_1/total:.2%}")

if __name__ == "__main__":
    eval_sft()