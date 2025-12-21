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
# 1. Configuration
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"
OUTPUT_DIR = "/workspace/data/grpo_v4_1_breadcrumbs"
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# 全局变量
_sid_map = {}
_tree_map = {}
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
    # 增强解析鲁棒性
    text = text.replace("Response:", "").replace("<", "").replace(">", "").strip()
    # 匹配开头是数字的序列
    match = re.search(r"^(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
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
# 3. Re-Optimized Reward Function (Focus on Accuracy, Relax Format)
# ==============================================================================

def optimized_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        # [优化1: 格式宽容]
        # 既然SFT已经教会了格式，这里如果偶尔解析失败，给一个温和的惩罚即可
        # 不让格式错误占据梯度的主要方向
        if not pred_id:
            rewards.append(-1.0) 
            continue
            
        # 检查是否是“幻觉ID”
        if pred_id in _tree_map:
            score = 0.0 
            pred_meta = _tree_map[pred_id]

            # --- 层级奖励 (Aggressive Semantic Reward) ---
            # 目标：全力提升 Hit@1 和 Hit@5
            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                score += 0.5 # Layer 0 (Region): 基础分
                
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    score += 1.0 # Layer 1 (District): 进阶分
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        # [重点优化] Layer 2 (Category)
                        # 猜对类别是推荐系统“懂”用户的标志，权重加大
                        score += 4.0 
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            # [极致优化] Layer 3 (Exact Item)
                            # 如果完全猜中，给予“大奖”，强力引导模型记住这个Pattern
                            score += 10.0 
            
            # --- 地理奖励 (Geo-Aware) ---
            # 保持指数衰减，这是训练地理感知最有效的方法
            if t_lat is not None:
                try:
                    dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                    # 0km -> +0.5, 10km -> +0.18
                    geo_score = 0.5 * math.exp(-dist / 10.0)
                    score += geo_score
                except:
                    pass
            
            rewards.append(score)
            
        else:
            # 幻觉 ID (格式对，但ID不存在)
            # 既然SFT很好，这可能是模型在尝试探索新ID
            # 给一个温和的惩罚，不要打断它的探索欲
            score = -1.0 
            
            # 仍然给部分分鼓励
            p_tuple = tuple(pred_id)
            if (len(p_tuple) >= 2 and p_tuple[:2] in _valid_prefixes_l2):
                score += 0.5 # 就算ID不存在，如果类别是对的，也值得鼓励！
            
            rewards.append(score)
            
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
    for c in completions:
        clean_c = c.replace("Response:", "").replace("<", "").strip()
        if re.match(pattern, clean_c):
            # [优化2: 格式维持]
            # 只要格式对，给一个微小的正向反馈，作为一个"心跳包"信号
            rewards.append(0.1) 
        else:
            # 格式错，轻微惩罚
            rewards.append(-0.5)
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
            
            # 提取纯 Prompt
            if "Response:" in raw_inst:
                prompt_text = raw_inst.split("Response:")[0].strip()
            else:
                prompt_text = raw_inst
            
            # 强制加上格式后缀，作为 Prompt 的一部分
            suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
            final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"
            
            data_list.append({
                "prompt": final_prompt,
                "target_sid": meta.get('target_sid'),
                "target_lat": meta.get('target_lat'),
                "target_lon": meta.get('target_lon')
            })
            
    dataset = Dataset.from_list(data_list)
    print(f"Loaded {len(dataset)} samples.")
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
    
    print(f"Loading SFT Config: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
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
        run_name="grpo_v4_2_optimized", # 更新 Run Name
        
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=500,
        max_steps=10000,               
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, 
        
        num_generations=16,
        max_completion_length=32, #稍微放宽一点长度
        
        # [优化4] 生成参数调优
        temperature=0.8, # 降低随机性，提高生成的有效率
        
        beta=0.04, # KL 惩罚系数 (GRPO默认0.04，保持稳定)
        
        use_vllm=False,
        bf16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            optimized_hierarchical_reward_func # 使用新的奖励函数
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # 智能断点恢复
    resume_ckpt = None
    if os.path.exists(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if ckpts:
            # 按数字排序，找到最大的
            ckpts.sort(key=lambda x: int(x.split("-")[-1]))
            latest_ckpt = ckpts[-1]
            # 检查文件夹是否为空（有时候训练崩了会留个空文件夹）
            if os.listdir(os.path.join(OUTPUT_DIR, latest_ckpt)):
                resume_ckpt = os.path.join(OUTPUT_DIR, latest_ckpt)
                print(f"🔄 Resuming training from: {resume_ckpt}")
            else:
                print(f"⚠️ Found empty checkpoint {latest_ckpt}, ignoring.")
    
    print("🚀 Starting Optimized GRPO Training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    print(f"💾 Saving Final Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()