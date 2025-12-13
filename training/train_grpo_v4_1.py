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
# [关键] 这里换成你刚刚验证过的 SFT Checkpoint
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
OUTPUT_DIR = "/workspace/data/grpo_v4_1_breadcrumbs"  # V4.1 输出目录

# 全局变量
_sid_map = {}
_tree_map = {}
# 面包屑集合：用于引导模型即便生成了非法ID，只要前缀对也给分
_valid_prefixes_l1 = set() 
_valid_prefixes_l2 = set() 

# ==============================================================================
# 2. Helper Functions
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
        # 构建前缀集合
        if len(full_code) >= 1: _valid_prefixes_l1.add(full_code[:1])
        if len(full_code) >= 2: _valid_prefixes_l2.add(full_code[:2])
        
    print(f"[Init] Loaded {len(_tree_map)} items.")
    print(f"[Init] Breadcrumbs: L1 Prefixes={len(_valid_prefixes_l1)}, L2 Prefixes={len(_valid_prefixes_l2)}")

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
# 3. V4.1 Reward: Strict Hierarchy + Breadcrumbs (核心逻辑)
# ==============================================================================

def strict_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        # 1. 格式/解析失败：死刑
        if not pred_id:
            rewards.append(-2.0)
            continue
            
        # 2. 是否在 _tree_map 中? (Valid vs Invalid)
        if pred_id in _tree_map:
            # === [Track A] 合法 ID ===
            # 因为是合法ID，至少起步比 Invalid 的最高分 (-1.2) 要高一点
            # 但如果层级全错，也要惩罚
            
            score = -0.5 # 默认：合法但全错
            pred_meta = _tree_map[pred_id]

            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                # Layer 0 (City) Correct
                score = -0.5 # 维持原判，除非下一层也对
                
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    # Layer 1 (District) Correct -> [翻身点]
                    # SFT 已经有 78% 的概率能走到这里，所以这是训练的基石
                    score = 0.5 
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        # Layer 2 (Category) Correct -> [主要目标]
                        score = 2.0 
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            # Layer 3 (Item) Correct -> [大奖]
                            score = 5.0
            else:
                # 合法 ID，但连城市都错了
                score = -1.0

            # Geo Bonus (辅助)
            if t_lat is not None:
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                if dist <= 1.0: score += 0.2
            
            rewards.append(score)
            
        else:
            # === [Track B] 非法 ID (但可能有正确的前缀) ===
            # 这就是 "Breadcrumbs" 面包屑机制
            
            score = -2.0 # 默认极刑
            p_tuple = tuple(pred_id)
            
            # 检查前缀合法性 (注意：这里检查的是 Prefix 是否在数据库中存在，而不是是否匹配 Target)
            # 这里的逻辑是：引导模型先生成"真实存在的区域代码"，哪怕它还没学会匹配 User Target
            
            prefix_l2_valid = (len(p_tuple) >= 2 and p_tuple[:2] in _valid_prefixes_l2)
            
            if prefix_l2_valid:
                # ID 是编的，但街区代码是真实的
                # 鼓励一下，防止梯度消失
                score = -1.2 
            elif len(p_tuple) >= 1 and p_tuple[:1] in _valid_prefixes_l1:
                # ID 是编的，但城市代码是真实的
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
# 4. Data & Main
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
    print(f"Prepared {len(dataset)} samples for GRPO.")
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
    
    print(f"Merging SFT Checkpoint: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable() 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # V4.1 Config: High Rank, High Generations
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
        run_name="grpo_v4_1_breadcrumbs",
        
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=200,
        max_steps=1500,               
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        
        num_generations=16,           # 关键：16次采样，增加命中 Category 的概率
        max_completion_length=24,
        
        beta=0.01,                    # 允许模型偏离 SFT，去探索更高分的策略
        
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

    print("Starting GRPO V4.1 Breadcrumbs Training...")
    trainer.train()
    
    print(f"Saving Final Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()