"""
grpo_rewards_optimized.py
[FINAL FIXED] 完美适配 Step2/Step4 的数据格式
"""

import json
import math
from haversine import haversine
import re
import os

# 全局变量
_sid_map = {}
_tree_map = {}

def load_mapping(mapping_file):
    global _sid_map, _tree_map
    print(f"Loading Reward Mapping from {mapping_file}...")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    
    # Step 2 生成的 full_sid 是 List [c0, c1, c2, s]，转为 tuple 做 key
    for bid, meta in _sid_map.items():
        full_code = tuple(meta['full_sid'])
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta['city']
        }
    print(f"Reward Mapping Loaded: {len(_tree_map)} items.")

def _extract_content(data):
    """从 GRPOTrainer 的输出中提取文本"""
    if isinstance(data, dict):
        return data.get('content', '')
    return str(data)

def parse_output(text_or_dict):
    """解析预测结果：输入可能是 dict，输出是 tuple(int)"""
    text = _extract_content(text_or_dict)
    
    # 匹配 <12, 34, 56, 78> 允许空格
    match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
    if match: return tuple(int(g) for g in match.groups())
    
    # 兼容方括号 [12, 34...]
    match = re.search(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", text)
    if match: return tuple(int(g) for g in match.groups())
    
    return None

def parse_target(target_raw):
    """解析 Ground Truth：适配 Step 4 生成的 String 格式"""
    # 情况 A: Step 4 生成的 String "<12, 34, 56, 0>"
    if isinstance(target_raw, str):
        clean = target_raw.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        try:
            return tuple(int(x.strip()) for x in clean.split(','))
        except:
            return None
    # 情况 B: 已经是 List/Tuple
    return tuple(target_raw)

# ======================================================================
# Rewards (返回扁平列表)
# ======================================================================

def format_reward_func(completions, **kwargs):
    flat_rewards = []
    for group in completions:
        for item in group:
            text = _extract_content(item)
            if parse_output(text):
                flat_rewards.append(0.1)
            else:
                # 软惩罚：有内容给 -0.9，空的给 -1.0
                flat_rewards.append(-0.9 if len(text.strip()) > 0 else -1.0)
    return flat_rewards

def geo_reward_func(prompts, completions, target_lat, target_lon, **kwargs):
    flat_rewards = []
    for group, t_lat, t_lon in zip(completions, target_lat, target_lon):
        for item in group:
            pred_id = parse_output(item)
            if not pred_id or pred_id not in _tree_map:
                flat_rewards.append(0.0)
                continue
                
            pred_meta = _tree_map[pred_id]
            dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
            
            if dist <= 5.0: flat_rewards.append(0.3)
            elif dist <= 20.0: flat_rewards.append(0.1)
            elif dist <= 50.0: flat_rewards.append(0.0)
            else: flat_rewards.append(-0.1)
    return flat_rewards

def semantic_reward_func(prompts, completions, target_sid, **kwargs):
    flat_rewards = []
    for group, t_sid_raw in zip(completions, target_sid):
        # [FIX] 解析 Step 4 的 String Target
        t_sid = parse_target(t_sid_raw)
        
        for item in group:
            pred_id = parse_output(item)
            if not pred_id or not t_sid:
                flat_rewards.append(0.0)
                continue
                
            score = 0.0
            # Layer 0
            if pred_id[0] == t_sid[0]:
                score += 0.1
                # Layer 1
                if pred_id[1] == t_sid[1]:
                    score += 0.2
                    # Layer 2 (Category) - 重点奖励
                    if pred_id[2] == t_sid[2]:
                        score += 1.0 
                        # Exact (Item) - 暴击奖励
                        if pred_id[3] == t_sid[3]:
                            score += 3.0
            flat_rewards.append(score)
    return flat_rewards