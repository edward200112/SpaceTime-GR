"""
grpo_rewards_v3.py
[FINAL OPTIMIZED VERSION]
Feature: Progressive Rewards (渐进式奖励)
- If ID is invalid but prefix is valid -> Partial Penalty (Soft Guide)
- If ID is valid -> Gated Semantic + Geo Reward
"""

import json
from haversine import haversine
import re
import os

# 全局变量
_sid_map = {}
_tree_map = {}
_valid_prefixes = set()

def load_mapping(mapping_file):
    global _sid_map, _tree_map, _valid_prefixes
    
    print(f"[V3 Rewards] Loading Mapping from {mapping_file}...")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        _sid_map = json.load(f)
    
    _tree_map = {}
    _valid_prefixes = set()
    
    for bid, meta in _sid_map.items():
        # 转为 tuple
        full_code = tuple(int(x) for x in meta['full_sid'])
        
        _tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta.get('city', ''), 
        }
        
        # 构建前缀集合，用于渐进式奖励
        # 假设 full_code 是 (c0, c1, c2, s)
        if len(full_code) >= 1: _valid_prefixes.add(full_code[:1])
        if len(full_code) >= 2: _valid_prefixes.add(full_code[:2])
        if len(full_code) >= 3: _valid_prefixes.add(full_code[:3])

    print(f"[V3 Rewards] Loaded {len(_tree_map)} items. Prefix set built.")

def _extract_content(data):
    if isinstance(data, dict): return data.get('content', '')
    return str(data)

def parse_output(text_or_dict):
    """
    宽容解析器：匹配字符串开头的数字序列
    """
    text = _extract_content(text_or_dict)
    # 匹配开头是 "数字, 数字..."
    match = re.search(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text.strip())
    if match: 
        return tuple(int(g) for g in match.groups())
    return None

def parse_target(target_raw):
    """解析 Ground Truth"""
    if isinstance(target_raw, str):
        clean = target_raw.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        try:
            return tuple(int(x.strip()) for x in clean.split(','))
        except:
            return None
    return tuple(target_raw)

# ======================================================================
# The "Gatekeeper" Reward Function with Progressive Logic
# ======================================================================

def gated_recommendation_reward_func(prompts, completions, target_sid, target_lat, target_lon, **kwargs):
    rewards = []
    
    for completion, t_sid_raw, t_lat, t_lon in zip(completions, target_sid, target_lat, target_lon):
        t_sid = parse_target(t_sid_raw)
        pred_id = parse_output(completion)
        
        # 1. 格式解析失败 -> 重罚 (-1.0)
        if not pred_id:
            rewards.append(-1.0)
            continue
        
        # 2. ID 是否存在于 Map 中?
        if pred_id in _tree_map:
            # === 情况 A: ID 合法 (在字典里) ===
            pred_meta = _tree_map[pred_id]
            score = 0.0
            
            # 核心门控: 前3层 (Category) 是否匹配?
            layer2_match = (t_sid and len(t_sid) >= 3 and 
                            pred_id[0] == t_sid[0] and 
                            pred_id[1] == t_sid[1] and 
                            pred_id[2] == t_sid[2])
            
            if layer2_match:
                # ✅ 语义正确 (+1.0)
                score += 1.0 
                
                # 地理距离奖励
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                if dist <= 2.0: score += 1.5
                elif dist <= 5.0: score += 1.0
                elif dist <= 20.0: score += 0.5
                elif dist <= 50.0: score += 0.1
                else: score -= 0.1
                
                # 完全命中奖励 (+2.0)
                if len(t_sid) == 4 and pred_id[3] == t_sid[3]:
                    score += 2.0
            else:
                # ❌ 语义错误
                dist = haversine((t_lat, t_lon), (pred_meta['lat'], pred_meta['lon']))
                # 给同城一点安慰分，防止梯度彻底消失
                if dist <= 20.0: score = -0.1
                else: score = -0.5
            
            rewards.append(score)
            
        else:
            # === 情况 B: ID 非法 (不在字典里) ===
            # [核心改进] 渐进式奖励：如果前缀合法，少扣分！
            
            score = -1.0 # 默认重罚
            
            if len(pred_id) >= 3 and pred_id[:3] in _valid_prefixes:
                score = -0.2 # 前3层有效 (类目存在)，只是后缀错了 -> 轻微惩罚
            elif len(pred_id) >= 2 and pred_id[:2] in _valid_prefixes:
                score = -0.4 # 前2层有效 -> 中等惩罚
            elif len(pred_id) >= 1 and pred_id[:1] in _valid_prefixes:
                score = -0.6 # 第1层有效 -> 较重惩罚
            
            # 如果连第1层都不在字典里，那就是 -1.0
            rewards.append(score)
            
    return rewards

def format_reward_func(completions, **kwargs):
    """格式奖励"""
    return [0.1 if parse_output(c) else -1.0 for c in completions]