import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 配置
BASE_MODEL = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_CKPT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"
GRPO_CKPT = "/workspace/data/grpo_v4_1_breadcrumbs/checkpoint-4800"
OUTPUT_DIR = "/workspace/data/final_model_v4_1"

print("1. Loading Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16, # 合并时建议用 float16 或 bfloat16
    device_map="cpu", # 在 CPU 上合并以节省显存
    trust_remote_code=True
)

print("2. Merging SFT Adapter...")
model = PeftModel.from_pretrained(model, SFT_CKPT)
model = model.merge_and_unload()

print("3. Merging GRPO Adapter...")
model = PeftModel.from_pretrained(model, GRPO_CKPT)
model = model.merge_and_unload()

print("4. Saving Final Model...")
model.save_pretrained(OUTPUT_DIR)

print("5. Saving Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Model saved to {OUTPUT_DIR}")