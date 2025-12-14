import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置
MODEL_PATH = "/workspace/data/grpo_v4_3_logit_masking/checkpoint-2000"
BASE_MODEL = "/workspace/Qwen2_5-1.5B-Instruct"
TEST_DATA = "/workspace/data/processed/test_prompts.jsonl"

def debug():
    print(f"🔍 Loading Adapter: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()
    
    print("\n>>> Sampling 5 examples for inspection:\n")
    
    with open(TEST_DATA, 'r') as f:
        lines = f.readlines()[:5] # 只看前5条
        
    for i, line in enumerate(lines):
        item = json.loads(line)
        raw_inst = item.get('instruction', '').strip()
        if "Response:" in raw_inst:
            prompt_text = raw_inst.split("Response:")[0].strip()
        else:
            prompt_text = raw_inst
            
        # 还原方案一的 Prompt
        suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
        final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"
        
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=32, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码
        # 只解码新生成的部分
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Sample {i+1}:")
        print(f"Prompt Tail: ...{final_prompt[-20:]}") # 打印Prompt末尾确认格式
        print(f"Model Output (Raw): '{gen_text}'") # 单引号包裹，看清是否有空格
        print("-" * 40)

if __name__ == "__main__":
    debug()