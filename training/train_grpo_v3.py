import os
import json
import re
import torch
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType  # <--- [新增] 引入 LoraConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

# ==============================================================================
# 1. 配置与全局变量
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_CHECKPOINT = "/workspace/data/llm_ckpt/checkpoint-28000"
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
OUTPUT_DIR = "/workspace/data/grpo_v3_optimized"

_sid_map = {}
_tree_map = {}

# ==============================================================================
# 2. 工具函数 & 奖励函数逻辑 (保持不变)
# ==============================================================================

def load_global_mapping(mapping_file):
    global _sid_map, _tree_map
    print(f"[Init] Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    _tree_map = {}
    for bid, meta in _sid_map.items():
        full_code = tuple(int(x) for x in meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta.get('city', 'Unknown'),
        }
    print(f"[Init] Loaded {len(_tree_map)} items.")

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_output(text):
    text = text.replace("Response:", "").replace("<", "").replace(">", "")
    match = re.search(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text.strip())
    if match:
        return tuple(int(g) for g in match.groups())
    return None

def parse_target(target_raw):
    if isinstance(target_raw, (list, tuple)): return tuple(int(x) for x in target_raw)
    if isinstance(target_raw, str):
        clean = target_raw.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        try: return tuple(int(x.strip()) for x in clean.split(','))
        except: pass
    return None

# --- Reward Functions ---

def hierarchical_accuracy_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        if not pred_id:
            rewards.append(-2.0)
            continue
            
        if pred_id in _tree_map:
            score = 0.5 
            pred_meta = _tree_map[pred_id]
            
            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                score += 0.2
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    score += 0.3
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        score += 1.0 
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            score += 2.0
            
            if t_lat is not None:
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                if dist <= 1.0: score += 0.5
                elif dist <= 5.0: score += 0.2
                elif dist <= 20.0: score += 0.0
                else: score -= 0.1
            rewards.append(score)
        else:
            rewards.append(-1.0)
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
    for c in completions:
        clean_c = c.replace("Response:", "").replace("<", "").strip()
        if re.match(pattern, clean_c): rewards.append(0.1)
        else: rewards.append(-1.0)
    return rewards

# ==============================================================================
# 3. 数据处理 (保持不变)
# ==============================================================================

def prepare_dataset(data_path):
    print(f"Loading dataset from {data_path}...")
    data_list = []
    with open(data_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            if item.get('task') != 'task_a_recommendation': continue
            
            meta = item.get('metadata', {})
            raw_inst = item.get('instruction', '').strip()
            prompt_text = raw_inst.split("Response:")[0].strip() if "Response:" in raw_inst else raw_inst
            
            # 统一添加后缀
            suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
            final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"
            
            data_list.append({
                "prompt": final_prompt,
                "target_sid": meta.get('target_sid'),
                "target_lat": meta.get('target_lat'),
                "target_lon": meta.get('target_lon')
            })
            
    dataset = Dataset.from_list(data_list)
    print(f"Prepared {len(dataset)} samples for GRPO.")
    return dataset

# ==============================================================================
# 4. 主训练逻辑 (修正版)
# ==============================================================================

def main():
    load_global_mapping(MAPPING_FILE)
    
    print("Loading Base Model + SFT Adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 1. 加载并融合 SFT 权重
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload() 
    # 此时 model 变成了一个普通的 dense model，没有 trainable parameters
    
    # 2. 启用 Gradient Checkpointing 以节省显存 (可选，但推荐)
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # 3. [关键修复] 定义新的 LoRA 配置用于 GRPO 训练
    # 这会告诉 Trainer 在 dense model 上挂载一个新的 adapter 进行训练
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,            # 秩
        lora_alpha=32,   # 缩放系数
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 全量 LoRA 效果更好
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v3_optimized",
        
        learning_rate=1e-6,           # RL 阶段 LR 要非常小！建议 1e-6 到 5e-6
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=200,
        max_steps=1000,
        
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        
        num_generations=8,
        max_completion_length=24,
        beta=0.04,
        
        use_vllm=False,
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        gradient_checkpointing=True, # 显式开启
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            hierarchical_accuracy_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config, # <--- [关键修复] 必须传入这个！
    )

    print("Starting GRPO Training...")
    trainer.train()
    
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()