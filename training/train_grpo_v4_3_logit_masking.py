import os
import json
import re
import torch
import math
import types  # [FIX] 用于 Monkey Patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LogitsProcessor, 
    LogitsProcessorList
)
from peft import PeftModel, LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset

# ==============================================================================
# 1. Configuration & Global Setup
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"
SFT_CHECKPOINT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"
OUTPUT_DIR = "/workspace/data/grpo_v4_3_logit_masking_full"

DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# 全局变量
_sid_map = {}
_tree_map = {}
_sid_strings = [] 

# ==============================================================================
# 2. Strict Trie & Logits Processor (NO SIMPLIFICATION)
# ==============================================================================

class TokenTrie:
    def __init__(self):
        # 节点结构: {token_id: next_node, ...}
        # 特殊键 "__end__" 标记序列结束
        self.root = {}

    def insert(self, token_ids: List[int]):
        node = self.root
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        node["__end__"] = True

    def get_valid_next_tokens(self, current_sequence: List[int]) -> List[int]:
        """
        根据当前生成的序列，在 Trie 中查找合法的下一个 Token 集合。
        """
        node = self.root
        for tid in current_sequence:
            if tid not in node:
                return [] # 路径已经跑偏，Trie 无法提供建议
            node = node[tid]
        
        # 返回当前节点下的所有 Key（除了 __end__ 标记）
        return [k for k in node.keys() if k != "__end__"]

class StrictHierarchicalLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, trie: TokenTrie, start_token_id: int, end_token_id: int):
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token_id = start_token_id # 对应 '<'
        self.end_token_id = end_token_id     # 对应 '>'
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: [batch_size, seq_len]
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            seq = input_ids[i]
            
            # 1. 寻找锚点：找到最后一个 start_token_id ('<') 的位置
            start_indices = (seq == self.start_token_id).nonzero(as_tuple=True)[0]
            
            if len(start_indices) == 0:
                continue
                
            start_idx = start_indices[-1].item()
            
            # 2. 获取 '<' 之后生成的所有 Token (当前正在生成的上下文)
            generated_ctx = seq[start_idx+1:].tolist()
            
            # 3. 查询 Trie 获取合法列表
            valid_next_tokens = self.trie.get_valid_next_tokens(generated_ctx)
            
            # 4. 执行 Masking
            if valid_next_tokens:
                # 创建全 -inf 的 mask
                mask = torch.ones_like(scores[i]) * float('-inf')
                # 仅激活合法的 token
                mask[valid_next_tokens] = scores[i][valid_next_tokens] # 保留原有分数，或者直接设为 0
                scores[i] = mask
            else:
                # 路径不在 Trie 里 (跑偏了)，或者已经结束
                # 这里不做处理，依靠 EOS 或长度惩罚
                pass

        return scores

def build_strict_token_trie(tokenizer, sid_strings):
    print("[Trie] Building STRICT Token-level Trie...")
    token_trie = TokenTrie()
    
    # 这一步至关重要：我们需要模拟模型生成时的 Token 序列。
    # 格式应该是 "12, 34, 56, 78>"
    encoded_batch = tokenizer.batch_encode_plus(
        sid_strings, 
        add_special_tokens=False
    )['input_ids']
    
    count = 0
    for ids in encoded_batch:
        token_trie.insert(ids)
        count += 1
        
    print(f"[Trie] Built with {count} paths.")
    return token_trie

# ==============================================================================
# 3. Reward Functions
# ==============================================================================

def load_global_mapping(mapping_file):
    global _sid_map, _tree_map, _sid_strings
    print(f"[Init] Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    
    _tree_map = {}
    _sid_strings = []
    
    for bid, meta in _sid_map.items():
        full_code = tuple(int(x) for x in meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude']
        }
        
        # 构建用于 Trie 的字符串 (必须以 > 结尾)
        sid_str = f"{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
        _sid_strings.append(sid_str)
        
    print(f"[Init] Loaded {len(_tree_map)} items for Trie.")

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_output_strict(text):
    if "Response:" in text:
        text = text.split("Response:")[-1]
    # 移除 < 和 >
    text = text.replace("<", "").replace(">", "").strip()
    try:
        parts = [int(x.strip()) for x in text.split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except:
        pass
    return None

def parse_target(target_raw):
    if isinstance(target_raw, (list, tuple)): return tuple(int(x) for x in target_raw)
    try:
        clean = str(target_raw).replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        return tuple(int(x.strip()) for x in clean.split(','))
    except:
        return None

def hierarchical_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        pred_id = parse_output_strict(completion)
        t_sid = parse_target(t_sid_raw)

        if not pred_id:
            rewards.append(-1.0) # 格式错误
            continue
        
        if pred_id not in _tree_map:
            rewards.append(-0.5) # 幻觉ID
            continue
            
        # 命中合法 ID
        score = 0.1 # 格式分
        pred_meta = _tree_map[pred_id]
        
        # 层级匹配
        if t_sid:
            if pred_id[0] == t_sid[0]: score += 0.5
            if pred_id[:2] == t_sid[:2]: score += 1.0
            if pred_id[:3] == t_sid[:3]: score += 2.0
            if pred_id == t_sid: score += 4.0
        
        # 地理距离
        if t_lat is not None:
            dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
            score += 2.0 * math.exp(-dist / 50.0)
            
        rewards.append(score)
    return rewards

# ==============================================================================
# 4. Main Execution
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
            
            # 【重要】Prompt 必须以 < 结尾，触发 LogitsProcessor
            suffix = "Output the semantic ID in format <c0, c1, c2, c3>."
            final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"
            
            data_list.append({
                "prompt": final_prompt,
                "target_sid": meta.get('target_sid'),
                "target_lat": meta.get('target_lat'),
                "target_lon": meta.get('target_lon')
            })
    return Dataset.from_list(data_list)

def main():
    # 1. Load Data & Map
    load_global_mapping(MAPPING_FILE)
    dataset = prepare_dataset(DATA_PATH)

    # 2. Model & Tokenizer
    print(f"Loading Model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, SFT_CHECKPOINT)
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable()

    # 3. Build Trie & Processor
    # 找到 '<' 和 '>' 的 Token ID
    start_token_id = tokenizer.encode("<", add_special_tokens=False)[0]
    end_token_id = tokenizer.encode(">", add_special_tokens=False)[0]
    
    print(f"Detected Anchors: '<' ID={start_token_id}, '>' ID={end_token_id}")

    token_trie = build_strict_token_trie(tokenizer, _sid_strings)
    
    strict_processor = StrictHierarchicalLogitsProcessor(
        tokenizer=tokenizer,
        trie=token_trie,
        start_token_id=start_token_id,
        end_token_id=end_token_id
    )
    logits_processor_list = LogitsProcessorList([strict_processor])

    # 4. Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=128, 
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="grpo_v4_scheme1_full",
        learning_rate=2e-6,
        logging_steps=5,
        save_steps=200,
        max_steps=2000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=8,
        max_completion_length=40,
        temperature=1.0, 
        use_vllm=False, # [FIX] Monkey Patch 必须为 False
        bf16=True,
        report_to="tensorboard"
    )

    # ==============================================================================
    # 5. [FIXED] Inject Processor via Monkey Patching
    # ==============================================================================
    
    # 1. 在 Config 中移除 logits_processor，只保留基础参数
    training_args.generation_kwargs = {
        "max_new_tokens": 40,
        "do_sample": True,
    }

    # 2. 定义 Monkey Patch 函数：在调用原 generate 前强行注入 logits_processor
    original_generate = model.generate 

    def generate_with_constraints(self, *args, **kwargs):
        # 强制将我们的 LogitsProcessorList 塞入 kwargs
        kwargs['logits_processor'] = logits_processor_list
        return original_generate(*args, **kwargs)

    # 3. 替换模型实例的方法
    model.generate = types.MethodType(generate_with_constraints, model)
    
    print("✅ Successfully patched model.generate with Strict Logits Processor.")

    # 6. Train
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[hierarchical_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("🚀 Starting GRPO Scheme 1 (FULL IMPLEMENTATION - NO SIMPLIFICATION)...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()