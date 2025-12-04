"""
training/grpo_rewards.py
包含：格式奖励 + 地理奖励 (Geo Reward) + 语义奖励
(已修复 AttributeError: 'list' object has no attribute 'strip')
"""
import re
import json
import math
from geopy.distance import geodesic

# 全局变量缓存
SID_MAPPING = {}
SID_TO_COORDS = {}

def load_mapping(mapping_path):
    global SID_MAPPING, SID_TO_COORDS
    if SID_MAPPING: return
    print(f"Loading Mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        SID_MAPPING = json.load(f)
    for info in SID_MAPPING.values():
        if 'sid_str' in info:
            SID_TO_COORDS[info['sid_str']] = (info.get('latitude'), info.get('longitude'))

def get_coords(sid_str):
    return SID_TO_COORDS.get(sid_str, (None, None))

# --- 辅助函数：安全提取文本 ---
def extract_content(c):
    """
    兼容处理 trl 传递的 completion 可能是字符串，也可能是 list[dict] 的情况
    """
    # 如果是列表（常见于 Chat 格式的 trl 输出）
    if isinstance(c, list):
        if len(c) > 0:
            # 或者是 [{"role": "assistant", "content": "..."}]
            if isinstance(c[0], dict) and "content" in c[0]:
                return c[0]["content"]
            # 或者是 ["text"]
            return str(c[0])
        return ""
    # 如果已经是字符串
    return str(c)

# --- 奖励函数 ---

def format_reward_func(completions, **kwargs):
    """奖励1: 格式必须是 <num, num, num, num>"""
    # 允许空白，要求4个数字
    pattern = r"^<(\d+,\s*){3}\d+>$"
    rewards = []
    for c in completions:
        text = extract_content(c).strip() # 使用提取函数
        if re.match(pattern, text):
            rewards.append(0.5) # 格式对给0.5
        else:
            rewards.append(0.0)
    return rewards

def geo_reward_func(prompts, completions, target_lat, target_lon, **kwargs):
    """奖励2: 地理距离 (核心)"""
    rewards = []
    for i, completion in enumerate(completions):
        text = extract_content(completion).strip() # 使用提取函数
        pred_lat, pred_lon = get_coords(text)
        
        if pred_lat is None:
            rewards.append(0.0)
            continue
            
        try:
            # 真实坐标
            t_lat = target_lat[i]
            t_lon = target_lon[i]
            
            # 计算距离
            dist = geodesic((pred_lat, pred_lon), (t_lat, t_lon)).km
            
            # 评分逻辑: 20km内线性给分，0km得1.0分
            score = max(0.0, 1.0 - (dist / 20.0))
            rewards.append(score)
        except:
            rewards.append(0.0)
    return rewards

def semantic_reward_func(prompts, completions, target_sid, **kwargs):
    """奖励3: 语义准确性 (Cluster ID 匹配)"""
    rewards = []
    for i, completion in enumerate(completions):
        pred = extract_content(completion).strip() # 使用提取函数
        gt = target_sid[i]
        
        if pred == gt:
            rewards.append(1.0)
            continue
        
        # 即使没完全猜对，如果 Cluster ID (前两位) 对了，也给分
        try:
            pred_nums = [int(x) for x in pred.replace('<','').replace('>','').split(',')]
            gt_nums = [int(x) for x in gt.replace('<','').replace('>','').split(',')]
            
            if len(pred_nums)>=2 and len(gt_nums)>=2 and pred_nums[:2] == gt_nums[:2]:
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards