"""
Check SFT Quality
快速检查模型是否学会了新的 ID 格式
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

# 配置
BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
LORA_PATH = "/workspace/data/llm_ckpt/checkpoint-14500" # 等你有第一个checkpoint时修改这里
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
TEST_DATA = "/workspace/data/processed/test_prompts.jsonl"

def check_model():
    print(f"Loading mapping from {MAPPING_FILE}...")
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    # 创建反向索引: sid_str -> business info
    sid_to_info = {}
    for bid, info in mapping.items():
        if 'sid_str' in info:
            sid_to_info[info['sid_str']] = info

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 如果有 Checkpoint，加载 LoRA
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA adapter from {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print("⚠️ No checkpoint found yet, using base model (Expect garbage output)")

    model.eval()    

    # 加载一条测试数据
    with open(TEST_DATA) as f:
        sample = json.loads(f.readline())
    
    instruction = sample['instruction']
    ground_truth = sample['output']
    
    print("\n" + "="*50)
    print(f"INPUT INSTRUCTION:\n{instruction[:200]}...")
    print(f"GROUND TRUTH: {ground_truth}")
    print("="*50)

    # 构造 Prompt
    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"MODEL OUTPUT: {response}")
    
    # 验证
    if response.strip() == ground_truth:
        print("✅ Perfect Match!")
    elif response.strip() in sid_to_info:
        info = sid_to_info[response.strip()]
        print(f"✅ Valid ID! Mapped to: {info['name']} in {info['city']}")
        print("   (Prediction is valid, even if not Ground Truth)")
    else:
        if "<" in response and ">" in response and response.count(",") == 3:
             print("⚠️ Format looks correct (Has suffix), but ID not found in mapping.")
        else:
             print("❌ Format Wrong (Model hasn't learned the new format yet)")

if __name__ == "__main__":
    check_model()