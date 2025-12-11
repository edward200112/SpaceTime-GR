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
# 1. 配置区域 (V5 Ultimate Config)
# ==============================================================================

# [关键] 指向你刚刚 SFT 结束的那个 Checkpoint
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v5_balanced/checkpoint-2000"

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_PATH = "/workspace/data/processed/train_prompts_balanced.jsonl" # 继续使用平衡数据
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
WEIGHTS_FILE = "/workspace/data/processed/category_weights.json" # 必须存在！
OUTPUT_DIR = "/workspace/data/grpo_v5_weighted"

# 全局变量
_sid_map = {}
_tree_map = {}
_valid_prefixes_l1 = set() 
_valid_prefixes_l2 = set()
_cat_weights = {}

# ==============================================================================
# 2. 资源加载
# ==============================================================================

def load_resources():
    global _sid_map, _tree_map, _valid_prefixes_l1, _valid_prefixes_l2, _cat_weights
    
    print(f"[Init] Loading mapping: {MAPPING_FILE}")
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
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
        
    print(f"[Init] Loaded {_valid_prefixes_l2} L2 Prefixes.")

    if os.path.exists(WEIGHTS_FILE):
        print(f"[Init] Loading Category Weights: {WEIGHTS_FILE}")
        with open(WEIGHTS_FILE, 'r') as f:
            raw_weights = json.load(f)
            _cat_weights = {int(k): v for k, v in raw_weights.items()}
        print(f"Loaded weights for {len(_cat_weights)} categories.")
        # 打印一下权重的分布情况
        vals = list(_cat_weights.values())
        print(f"Weights -> Max: {max(vals):.2f}, Min: {min(vals):.2f}, Avg: {sum(vals)/len(vals):.2f}")
    else:
        raise FileNotFoundError(f"找不到权重文件 {WEIGHTS_FILE}！请先运行 balance_dataset.py 生成。")

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

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==============================================================================
# 3. Reward Function V5 (Weighted + Bias Penalty)
# ==============================================================================

def weighted_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        # 1. Format Check
        if not pred_id:
            rewards.append(-2.0); continue
            
        # 2. Validity Check
        if pred_id in _tree_map:
            score = -0.5 # Valid but Wrong
            
            # Get Target Weight
            target_l2 = t_sid[2] if len(t_sid) >= 3 else -1
            w = _cat_weights.get(target_l2, 1.0)
            
            # Limit max weight to prevent gradient explosion
            w = min(w, 8.0)

            # --- Hierarchy Matching ---
            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]: # City
                score = -0.5
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]: # District
                    score = 0.5 # Base Reward
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]: # Category
                        # [V5] Weighted Reward
                        # Hot Category: 2.0 * 1.0 = 2.0
                        # Cold Category: 2.0 * 8.0 = 16.0 !
                        score = 2.0 * w
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]: # Item
                            score = 5.0 * w
            
            # --- [V5] Bias Penalty (偏见惩罚) ---
            # 如果 Target 是冷门 (权重高)，但预测结果是热门 (权重低)
            pred_l2 = pred_id[2]
            pred_w = _cat_weights.get(pred_l2, 1.0)
            
            # 如果应该猜难的(w>3)，却猜了简单的(pred_w<1.5)，重罚
            if target_l2 != pred_l2:
                if w > 3.0 and pred_w < 1.5:
                    score -= 2.0 

            # Geo Bonus
            if t_lat:
                dist = haversine((t_lat, t_lon), (_tree_map[pred_id]['lat'], _tree_map[pred_id]['lon']))
                if dist <= 1.0: score += 0.2

            rewards.append(score)
            
        else:
            # Breadcrumbs (Legacy V4.1)
            score = -2.0
            p_tuple = tuple(pred_id)
            if len(p_tuple) >= 2 and p_tuple[:2] in _valid_prefixes_l2: score = -1.2
            elif len(p_tuple) >= 1 and p_tuple[:1] in _valid_prefixes_l1: score = -1.5
            rewards.append(score)
            
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
    for c in completions:
        clean_c = c.replace("Response:", "").replace("<", "").strip()
        if re.match(pattern, clean_c): rewards.append(0.1)
        else: rewards.append(-2.0)
    return rewards

# ==============================================================================
# 4. Main
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

def main():
    load_resources()
    
    print(f"Loading Base: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Merging SFT V5: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable() 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # V5 配置: High Rank, High Steps
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
        run_name="grpo_v5_weighted",
        
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=500,
        
        # [V5] 跑满 5000 步，给它足够的时间去探索长尾
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
        reward_funcs=[format_reward_func, weighted_hierarchical_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting GRPO V5 Weighted Training...")
    trainer.train()
    
    print(f"Saving V5 Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()