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
# 1. Configuration (RESUME MODE)
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"

# [关键修改] 这里不再是 SFT，而是你刚刚训练了一半的 GRPO 模型
# 这一步是为了加载底座和 adapter 的配置，实际权重会通过 resume_from_checkpoint 加载
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"

# [关键修改] 输出目录保持一致，这样 Trainer 才能找到之前的 checkpoint
OUTPUT_DIR = "/workspace/data/grpo_v4_1_breadcrumbs"

DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# 全局变量
_sid_map = {}
_tree_map = {}
_valid_prefixes_l1 = set() 
_valid_prefixes_l2 = set() 

# ==============================================================================
# 2. Helper Functions (保持不变)
# ==============================================================================

def load_global_mapping(mapping_file):
    global _sid_map, _tree_map, _valid_prefixes_l1, _valid_prefixes_l2
    print(f"[Init] Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    
    _tree_map = {}
    _valid_prefixes_l1 = set()
    _valid_prefixes_l2 = set()
    
    for bid, meta in _sid_map.items():
        full_code = tuple(int(x) for x in meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta.get('city', 'Unknown'),
        }
        if len(full_code) >= 1: _valid_prefixes_l1.add(full_code[:1])
        if len(full_code) >= 2: _valid_prefixes_l2.add(full_code[:2])
        
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

# ==============================================================================
# 3. Reward Function (Boosting Category Reward)
# ==============================================================================

def strict_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        if not pred_id:
            rewards.append(-2.0)
            continue
            
        if pred_id in _tree_map:
            score = -0.5 
            pred_meta = _tree_map[pred_id]

            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                score = -0.5 
                
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    # Layer 1 (District) Correct -> 基准分
                    score = 0.5 
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        # [关键修改] Layer 2 (Category) Correct
                        # 加大奖励力度：从 +2.0 提升到 +4.0
                        # 告诉模型：猜对类别比单纯在对的街区里混日子要赚得多！
                        score = 4.0 
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            score = 8.0 # 完美大奖翻倍
            else:
                score = -1.0

            if t_lat is not None:
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                if dist <= 1.0: score += 0.2
            
            rewards.append(score)
            
        else:
            score = -2.0 
            p_tuple = tuple(pred_id)
            prefix_l2_valid = (len(p_tuple) >= 2 and p_tuple[:2] in _valid_prefixes_l2)
            
            if prefix_l2_valid:
                score = -1.2 
            elif len(p_tuple) >= 1 and p_tuple[:1] in _valid_prefixes_l1:
                score = -1.5
            rewards.append(score)
            
    return rewards

def format_reward_func(completions, **kwargs):
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
# 4. Main (Resume Logic)
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
            
    dataset = Dataset.from_list(data_list)
    return dataset

def main():
    load_global_mapping(MAPPING_FILE)
    
    print(f"Loading Base: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading SFT Config (Weights will be overridden by resume): {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable() 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # 保持配置一致
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,             
        lora_alpha=128,   
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v4_1_extended",
        
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=500,               # 每500步存一次
        
        # [关键修改] 延长到 5000 步
        max_steps=5000,               
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        num_generations=16,
        max_completion_length=24,
        beta=0.01,
        
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
            strict_hierarchical_reward_func 
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # [关键修改] 寻找最新的 checkpoint 并恢复
    resume_ckpt = None
    if os.path.exists(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[-1]))
            # 找到 checkpoint-1500
            resume_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])
            print(f"Resuming training from: {resume_ckpt}")
    
    print("Starting Extended GRPO Training...")
    # 传入 resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    print(f"Saving Extended Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()