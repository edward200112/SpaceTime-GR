import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 这里的路径换成你的 SFT 路径
BASE_MODEL = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_PATH = "/workspace/data/llm_ckpt/checkpoint-28000"

print("Loading SFT Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, SFT_PATH)
model.eval()

# 2. 测试两个截然不同的 Prompt
prompts = [
    # Case 1: 想要 Pizza (New York)
    "User History: [Joe's Pizza] (Pizza in New York)\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <",
    
    # Case 2: 想要 Coffee (San Francisco)
    "User History: [Blue Bottle] (Coffee in San Francisco)\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
]

print("\n=== SFT Inference Check ===")
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"Input: ...{p[-30:]}")
    print(f"Output: {tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)}")
    print("-" * 20)