import os
import json
import re
import torch
import math
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

# ================= 配置 =================
# [Check] 确保这里的路径指向刚刚 SFT 跑出来的目录
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v5_balanced" 
BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
DATA_PATH = "/workspace/data/processed/train_prompts_balanced.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
WEIGHTS_FILE = "/workspace/data/processed/category_weights.json"
OUTPUT_DIR = "/workspace/data/grpo_v5_weighted"

# 全局资源
_sid_map = {}
_tree_map = {}
_cat_weights = {}

def load_resources():
    global _sid_map, _tree_map, _cat_weights
    
    print(f"Loading mapping: {MAPPING_FILE}")
    with open(MAPPING_FILE, 'r') as f:
        _sid_map = json.load(f)
    
    _tree_map = {}
    for bid, meta in _sid_map.items():
        # meta['full_sid'] 是 [c0, c1, c2, c3]
        full_code = tuple(int(x) for x in meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'l2': full_code[2] if len(full_code) >= 3 else -1
        }
        
    print(f"Loading Weights: {WEIGHTS_FILE}")
    with open(WEIGHTS_FILE, 'r') as f:
        _cat_weights = {int(k): v for k, v in json.load(f).items()}
    print(f"Weights Loaded. Max: {max(_cat_weights.values()):.2f}")

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- 解析工具 ---
def parse_output(text):
    text = text.replace("Response:", "").replace("<", "").replace(">", "")
    # 匹配 "1, 2, 3, 4"
    match = re.search(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text.strip())
    if match:
        return tuple(int(g) for g in match.groups())
    return None

def parse_target(target_raw):
    # 处理 datasets 可能返回的格式差异
    if isinstance(target_raw, (list, tuple)): return tuple(int(x) for x in target_raw)
    if isinstance(target_raw, str):
        clean = target_raw.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        try: return tuple(int(x.strip()) for x in clean.split(','))
        except: pass
    return None

# --- 核心 V5.1 Reward Function ---
def weighted_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        if not pred_id:
            rewards.append(-2.0); continue
            
        if pred_id in _tree_map:
            # 基础分：命中了有效ID
            score = -0.5 
            
            # 获取目标权重
            target_l2 = t_sid[2] if len(t_sid) >= 3 else -1
            w = _cat_weights.get(target_l2, 1.0)
            
            # [Optimization] 使用 Log 缩放，防止数值过大
            # w 已经在 1.0 - 5.0 之间。
            # hot(1.0) -> scale=1.0
            # cold(5.0) -> scale=1.0 + log(5) ≈ 2.6
            w_scale = 1.0 + math.log(w)

            # --- 1. 层级匹配奖励 ---
            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]: # City
                score = 0.1
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]: # District
                    score = 0.5 
                    
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]: # Category
                        # 核心加权：Category 命中
                        score = 2.0 * w_scale
                        
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]: # Item
                            # 最终命中，给予额外加成
                            score += 3.0 * w_scale
            
            # --- 2. 偏见惩罚 (Bias Penalty) ---
            # 逻辑：如果 Target 是冷门，但模型预测了一个热门，且预测错误 -> 重罚
            pred_l2 = _tree_map[pred_id]['l2']
            pred_w = _cat_weights.get(pred_l2, 1.0)
            
            if target_l2 != pred_l2: # 类别错了
                # 定义：Target冷门(w>2.5), 预测热门(pred_w < 1.5)
                if w > 2.5 and pred_w < 1.5:
                    score -= 1.5 # 狠狠地罚，逼它去猜冷门
            
            # --- 3. 地理距离 (Relaxed) ---
            if t_lat:
                dist = haversine((t_lat, t_lon), (_tree_map[pred_id]['lat'], _tree_map[pred_id]['lon']))
                if dist <= 3.0: score += 0.5   # 3km 内
                elif dist <= 10.0: score += 0.1 # 10km 内鼓励

            rewards.append(score)
        else:
            # 幻觉 (Invalid ID)
            rewards.append(-3.0)
            
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+"
    for c in completions:
        if re.match(pattern, c.replace("Response:", "").replace("<", "").strip()):
            rewards.append(0.1)
        else:
            rewards.append(-2.0)
    return rewards

# --- Main Training Loop ---
def main():
    load_resources()
    
    # 1. 加载 SFT 模型 (Merge LoRA)
    print(f"Loading Base + Adapter: {SFT_CHECKPOINT}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    # 合并 SFT 权重，作为 RL 的起点
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left" # RL 生成必须左对齐
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备数据
    def prepare_dataset():
        data_list = []
        with open(DATA_PATH, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if item.get('task') != 'task_a_recommendation': continue
                
                # 提取 Prompt
                raw_inst = item.get('instruction', '')
                prompt = raw_inst.split("Response:")[0].strip() + "\nResponse: <"
                
                data_list.append({
                    "prompt": prompt,
                    "target_sid": item['metadata']['target_sid'],
                    "target_lat": item['metadata']['target_lat'],
                    "target_lon": item['metadata']['target_lon']
                })
        return Dataset.from_list(data_list)

    dataset = prepare_dataset()
    print(f"GRPO Dataset Size: {len(dataset)}")

    # 3. 配置 LoRA (RL 阶段仅训练少量参数)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        lora_dropout=0.05
    )

    # 4. GRPO 参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v5_final",
        learning_rate=2e-6,           # RL 学习率要低
        max_steps=5000,               # 足够让它探索
        per_device_train_batch_size=4, # 5090 可以大一点
        gradient_accumulation_steps=4,
        num_generations=16,           # 16个采样，增加探索空间
        max_completion_length=24,
        beta=0.02,                    # KL 惩罚系数，防止偏离 SFT 太多
        bf16=True,
        logging_steps=5,
        save_steps=500,
        report_to="tensorboard"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, weighted_hierarchical_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting GRPO Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()