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

# [配置] 保持和上一次运行一致
BASE_MODEL_PATH = "/workspace/Qwen2.5-7B-Instruct"

# [注意] 虽然是 Resume，但初始化 Trainer 时仍需加载原始 SFT 权重
# 实际训练时，Trainer 会读取 checkpoint 中的权重覆盖这里加载的权重
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_qwen2.5_7b_balanced/checkpoint-9400"

DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# [关键] 必须指向你中断前输出的同一个文件夹
OUTPUT_DIR = "/workspace/data/grpo_qwen2.5_7b_breadcrumbs"

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
    text = text.replace("Response:", "").replace("<", "").replace(">", "").replace("[", "").replace("]", "")
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
# 3. Reward Function (保持不变)
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
                    score = 0.5 
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        score = 2.0 
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            score = 5.0
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
        clean_c = c.replace("Response:", "").replace("<", "").replace("[", "").strip()
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
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

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
    
    print(f"Loading Initial SFT Weights: {SFT_CHECKPOINT}")
    try:
        model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Warning: Could not load SFT checkpoint directly ({e}). Assuming Resume will handle weights.")

    model.gradient_checkpointing_enable() 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(DATA_PATH)

    # LoRA Config
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
        run_name="grpo_qwen2.5_7b_resume",
        
        learning_rate=1e-6,           # 保持低 LR
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=5,
        save_steps=500,               # 每100步存一次，方便随时停
        max_steps=10000,               # [修改] 延长训练步数
        
        # === 显存优化参数 (必须与上次一致，否则报错) ===
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        num_generations=8,            
        max_completion_length=24,     
        
        # [修改] 增加采样温度，帮助 7B 模型探索
        temperature=1.2, 
        
        beta=0.001,                    
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

    # === 自动断点续传逻辑 ===
    resume_ckpt = None
    if os.path.exists(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            # 按数字排序找到最大的 checkpoint
            ckpts.sort(key=lambda x: int(x.split("-")[-1]))
            resume_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])
            print(f"✅ Found checkpoint! Resuming from: {resume_ckpt}")
        else:
            print("⚠️ No checkpoint found in output dir. Starting fresh.")
    
    print("Starting/Resuming GRPO Training...")
    
    # 如果找到了 checkpoint，传入 resume_from_checkpoint=True 或路径
    # 如果没找到，传入 None (从头开始)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    print(f"Saving Final Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()