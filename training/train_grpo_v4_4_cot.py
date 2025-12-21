import os
import json
import re
import torch
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

# ==============================================================================
# 1. Configuration
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
# 使用 SFT 后的模型作为起点
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"

# [修改 1] 方案二的独立输出目录
OUTPUT_DIR = "/workspace/data/grpo_v4_4_cot"

DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# 全局变量
_sid_map = {}
_tree_map = {}
_l2_to_keywords = {} # { (c0, c1, c2): ["food", "mexican"] }

# ==============================================================================
# 2. Semantic Reverse Engineering (方案二核心)
# ==============================================================================

def build_semantic_dictionary(mapping_file):
    """
    逆向工程：分析每个 Layer 2 ID (Cluster) 到底代表什么现实含义。
    方法：统计该 Cluster 下所有 Item 的 categories 词频。
    """
    global _sid_map, _tree_map, _l2_to_keywords
    print(f"[CoT Init] Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    
    _tree_map = {}
    cluster_counters = defaultdict(Counter) # Key: L2 Tuple, Value: Counter of words
    
    print("[CoT Init] Reverse engineering category semantics...")
    for bid, meta in _sid_map.items():
        # 1. 构建基础 Geo Map
        full_code = tuple(int(x) for x in meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta.get('city', 'Unknown'),
        }
        
        # 2. 统计语义
        # 假设 Layer 2 是前三位 (c0, c1, c2) -> 对应 Cluster
        if len(full_code) >= 3:
            l2_key = tuple(full_code[:3])
            
            # 解析 categories 字符串
            # 格式: "Doctors, Traditional Chinese Medicine, ..."
            raw_cats = meta.get('categories', '')
            if raw_cats:
                # 简单分词：按逗号分割，再转小写
                cats = [c.strip().lower() for c in raw_cats.split(',')]
                # 统计
                for c in cats:
                    cluster_counters[l2_key][c] += 1
    
    # 3. 提炼关键词
    # 每个 Cluster 选出 Top 3 最能代表它的词
    _l2_to_keywords = {}
    for l2_key, counter in cluster_counters.items():
        # 选 Top 3 高频词
        top_words = [word for word, count in counter.most_common(3)]
        _l2_to_keywords[l2_key] = top_words
        
    print(f"[CoT Init] Analyzed semantics for {len(_l2_to_keywords)} categories.")
    # 示例打印
    sample_key = next(iter(_l2_to_keywords))
    print(f"   Sample L2 {sample_key} -> Keywords: {_l2_to_keywords[sample_key]}")

# ==============================================================================
# 3. Helper Functions & Reward
# ==============================================================================

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
    # 兼容 CoT 输出：文本可能很长，我们只找最后的 ID 部分
    # 假设 ID 格式为 <1, 2, 3, 4>
    match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    if match: return tuple(int(g) for g in match.groups())
    return None

def parse_target(target_raw):
    if isinstance(target_raw, (list, tuple)): return tuple(int(x) for x in target_raw)
    if isinstance(target_raw, str):
        clean = target_raw.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        try: return tuple(int(x.strip()) for x in clean.split(','))
        except: pass
    return None

# --- [核心] CoT 语义奖励 ---
def cot_semantic_reward_func(prompts, completions, target_sid, **kwargs):
    rewards = []
    for completion, t_sid_raw in zip(completions, target_sid):
        t_sid = parse_target(t_sid_raw)
        
        # 1. 获取 Target 的真实语义关键词
        target_keywords = []
        if t_sid and len(t_sid) >= 3:
            l2_key = tuple(t_sid[:3])
            target_keywords = _l2_to_keywords.get(l2_key, [])
            
        # 2. 检查模型生成的文本中是否包含这些关键词
        # 即使 ID 生成错了，只要它"嘴里念叨"的类别是对的，就给分！
        hit = False
        completion_lower = completion.lower()
        
        # 排除 prompt 部分（TRL通常只传 completion，但为了保险）
        # 这里假设 completion 是纯生成的
        
        if target_keywords:
            for kw in target_keywords:
                if kw in completion_lower:
                    hit = True
                    break
        
        if hit:
            # [思维链奖励] 
            # 如果模型提到了正确的类别词 (比如 "pizza")，给予极大奖励
            # 这鼓励模型显式输出思考过程
            rewards.append(2.0) 
        else:
            rewards.append(0.0)
            
    return rewards

# --- 传统的层级奖励 (保持不变，作为最终兜底) ---
def strict_hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion) # 解析 ID
        
        if not pred_id:
            rewards.append(-1.0)
            continue
            
        if pred_id in _tree_map:
            score = 0.0
            pred_meta = _tree_map[pred_id]

            if t_sid and len(t_sid) >= 1 and pred_id[0] == t_sid[0]:
                score += 0.5 
                if len(t_sid) >= 2 and pred_id[1] == t_sid[1]:
                    score += 1.0 
                    if len(t_sid) >= 3 and pred_id[2] == t_sid[2]:
                        score += 4.0 # Category ID 正确
                        if len(t_sid) >= 4 and pred_id[3] == t_sid[3]:
                            score += 8.0 
            
            if t_lat is not None:
                try:
                    dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                    geo_score = 0.5 * math.exp(-dist / 10.0)
                    score += geo_score
                except: pass
            rewards.append(score)
        else:
            rewards.append(-1.0) # 幻觉惩罚
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    # 只要包含了 <d, d, d, d> 就算格式对，不管前面有多少废话
    pattern = r".*<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
    for c in completions:
        # replace newline to allow regex match across lines
        clean_c = c.replace("\n", " ") 
        if re.search(pattern, clean_c):
            rewards.append(0.1)
        else:
            rewards.append(-1.0)
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
            
            if "Response:" in raw_inst:
                prompt_text = raw_inst.split("Response:")[0].strip()
            else:
                prompt_text = raw_inst
            
            # [CoT 核心修改]
            # 修改 Prompt，引导模型先输出类别名称，再输出 ID
            # "Reasoning about category first, then output ID..."
            suffix = (
                "Step 1: Predict the category name of the next item.\n"
                "Step 2: Output the semantic ID <c0, c1, c2, suffix>.\n"
                "Response: The user is interested in" # [Priming] 强制模型开始说话
            )
            final_prompt = f"{prompt_text}\n{suffix}"
            
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
    # 1. 构建语义字典
    build_semantic_dictionary(MAPPING_FILE)
    
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
        run_name="grpo_v4_4_cot",
        
        learning_rate=2e-6,           
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        logging_steps=10,
        save_steps=500,
        save_total_limit=3, # Max save 3
        max_steps=5000,               
        
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, 
        
        num_generations=16,
        max_completion_length=64, # [修改] 增加长度，给 CoT 留出说话的空间
        
        temperature=0.9, # CoT 需要一点创造力，稍微调高
        
        beta=0.04,
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
            cot_semantic_reward_func, # [新] CoT 语义奖励
            strict_hierarchical_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ Warning: Output dir {OUTPUT_DIR} exists. Starting fresh for CoT Scheme.")
    
    print("🚀 Starting GRPO Scheme 2 (CoT/Reasoning)...")
    trainer.train()
    
    print(f"💾 Saving Final Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()