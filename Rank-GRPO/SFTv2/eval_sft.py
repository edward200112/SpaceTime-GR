import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./SFT/sft_output/final_checkpoint"
# 假设我们用一部分训练数据做简单测试，实际应用应划分验证集
TEST_DATA = "./SFT/sft_data/sft_enhanced_train.jsonl" 

def eval_sft():
    # 1. Load Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model.eval()

    # 2. Load Samples
    samples = []
    with open(TEST_DATA, 'r') as f:
        for i, line in enumerate(f):
            samples.append(json.loads(line))
            if i >= 200: break # Test 200 samples

    valid_cnt, hit_cnt, total = 0, 0, 0
    print("🚀 Starting Evaluation...")

    for item in tqdm(samples):
        # Construct Prompt
        prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False # Greedy decoding for recall
            )
        
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1].strip()
        gt = item['completion'].strip()

        # Metric 1: Format Compliance (Digits & Spaces)
        if all(c.isdigit() or c.isspace() for c in resp) and len(resp) > 0:
            valid_cnt += 1
            
        # Metric 2: Next-Item Recall@1 (Exact Match)
        if resp == gt:
            hit_cnt += 1
        
        total += 1

    print(f"Format Compliance: {valid_cnt/total:.2%}")
    print(f"Recall@1: {hit_cnt/total:.2%}")

if __name__ == "__main__":
    eval_sft()