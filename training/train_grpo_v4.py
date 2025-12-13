import os
import json
import re
import torch
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

# ==============================================================================
# 1. Configuration & Paths
# ==============================================================================

# 路径配置
BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_CHECKPOINT = "/workspace/data/llm_ckpt/checkpoint-28000"
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
OUTPUT_DIR = "/workspace/data/grpo_v4_strict"  # V4 输出目录

# 全局变量
_sid_map = {}
_tree_map = {}

# ==============================================================================
# 2. Helper Functions
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
    print(f"[Init] Loaded {len(_tree_map)} items into Semantic Tree.")

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
    # 移除可能的前缀
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

# ==============================================================================
# 3. V4 Strict Reward Function (The "No Mercy" Logic)
# ==============================================================================

def strict_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    """
    V4 Strategy:
    - Format Error: -2.0
    - Invalid ID: -2.0
    - Valid ID but Wrong Layer 1 (District): -0.5 (Punish 'Safety' behavior)
    - Valid ID + District Match: +0.5 (Break-even point)
    - Valid ID + Category Match: +2.0 (Goal)
    """
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        # 1. 格式/解析失败：极刑
        if not pred_id:
            rewards.append(-2.0)
            continue
            
        # 2. 合法性检查
        if pred_id in _tree_map:
            pred_meta = _tree_map[pred_id]
            
            # 默认起步分是 0.0，不再给 0.5 的 Valid 奖励
            # 只有证明了自己有能力区分 District，才给分
            
            score = 0.0
            
            # --- 层级判定 ---
            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                # Layer 0 (City) 对了
                # 关键改动：如果止步于此（Layer 1 错了），给负分！
                # 迫使模型不要停留在 City 层面躺平
                current_status = -0.5 
                
                # Layer 1: District Match
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    # 只有对上街区，才开始给正分
                    current_status = 0.5
                    
                    # Layer 2: Category Match (核心目标)
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        current_status = 2.0 # 重奖
                        
                        # Layer 3: Exact Item
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            current_status = 5.0 # 大奖
                
                score = current_status
            else:
                # 连城市都错了，给重一点的惩罚，但比 Invalid 轻一点
                score = -1.0

            # Geo Bonus (辅助，仅当距离非常近时生效)
            if t_lat is not None:
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                if dist <= 1.0: 
                    score += 0.2 # 稍微奖励一下精准定位
            
            rewards.append(score)
            
        else:
            # 非法 ID：极刑
            # 必须比 "猜错街区" (-0.5) 惩罚更重，防止模型编造 ID
            rewards.append(-2.0)
            
    return rewards

def format_reward_func(completions, **kwargs):
    """格式检查"""
    rewards = []
    pattern = r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
    for c in completions:
        clean_c = c.replace("Response:", "").replace("<", "").strip()
        if re.match(pattern, clean_c):
            rewards.append(0.1)
        else:
            rewards.append(-2.0)
    return rewards

# ==============================================================================
# 4. Data Preparation
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
            
            # Handle Pre-filling
            if "Response:" in raw_inst:
                prompt_text = raw_inst.split("Response:")[0].strip()
            else:
                prompt_text = raw_inst
            
            suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
            final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"
            
            data_list.append({
                "prompt": final_prompt,
                "target_sid": meta.get('target_sid'),
                "target_lat": meta.get('target_lat'),
                "target_lon": meta.get('target_lon')
            })
            
    # 全量数据
    dataset = Dataset.from_list(data_list)
    print(f"Prepared {len(dataset)} samples for GRPO.")
    return dataset

# ==============================================================================
# 5. Main Training Logic (V4)
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
    
    # Merge SFT (Starting Point)
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable() 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # [V4 Config Change 1] Higher Rank LoRA
    # 给模型更大的参数空间去调整 SFT 带来的思维定势
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,             # V3 was 16. Increased to 64 for V4.
        lora_alpha=128,   # Alpha = 2 * r
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # [V4 Config Change 2] High Exploration Config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v4_strict",
        
        learning_rate=2e-6,           # Conservative LR
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=200,
        
        # Steps Calculation:
        # 1500 steps * 2 (batch) * 16 (G) = 48,000 experiences
        max_steps=3000,               # Increased from 1000
        
        per_device_train_batch_size=2,# Reduced to 2 to handle G=16
        gradient_accumulation_steps=8,# Increased to maintain effective batch size
        
        num_generations=16,           # Increased from 8. Vital for sparse rewards!
        max_completion_length=24,
        
        beta=0.01,                    # Reduced KL penalty (0.04 -> 0.01) to allow larger policy shift
        
        use_vllm=False,
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            strict_hierarchical_reward_func # The strict V4 logic
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting GRPO V4 Strict Training...")
    print(f"Config: Steps={training_args.max_steps}, G={training_args.num_generations}, Beta={training_args.beta}")
    trainer.train()
    
    print(f"Saving V4 model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()