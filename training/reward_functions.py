"""
Reward Functions for GRPO Training

实现多种reward函数用于强化学习训练
"""

import math
from typing import List


def rule_reward(prompts: List[str], completions: List[str], targets: List[str], **kwargs) -> List[float]:
    """
    层次化奖励：支持部分匹配
    
    Args:
        prompts: 输入prompts（未使用）
        completions: 模型生成的completions
        targets: 目标答案
    
    Returns:
        rewards: 奖励列表
    """
    rewards = []
    for completion, target in zip(completions, targets):
        # 清理和比较
        completion_clean = completion.strip().strip('"').strip()
        target_clean = target.strip().strip('"').strip()
        
        if completion_clean == target_clean:
            # 完全匹配：最高奖励
            rewards.append(1.0)
        else:
            # 尝试提取cluster数字进行层次化比较
            # 格式: <X, Y>
            try:
                # 提取数字
                comp_nums = [int(x.strip()) for x in completion_clean.strip('<>').split(',')]
                targ_nums = [int(x.strip()) for x in target_clean.strip('<>').split(',')]
                
                # 两个数字都匹配
                if len(comp_nums) >= 2 and len(targ_nums) >= 2:
                    if comp_nums[0] == targ_nums[0] and comp_nums[1] == targ_nums[1]:
                        rewards.append(1.0)  # 完全匹配（应该和上面相同）
                    elif comp_nums[0] == targ_nums[0]:
                        rewards.append(0.5)  # 第一层（粗粒度）匹配
                    else:
                        rewards.append(0.0)  # 完全不匹配
                else:
                    rewards.append(0.0)
            except:
                # 解析失败，给小奖励（至少生成了内容）
                if len(completion_clean) > 0:
                    rewards.append(0.1)
                else:
                    rewards.append(0.0)
    
    return rewards


def ndcg_rule_reward(
    prompts: List[str], 
    completions: List[str], 
    targets: List[str],
    num_generations: int = 16,
    **kwargs
) -> List[float]:
    """
    基于NDCG的排序奖励
    
    如果目标出现在生成的结果中，根据排名给予奖励
    如果目标没有出现，所有生成都得0分
    
    Args:
        prompts: 输入prompts
        completions: 模型生成的completions  
        targets: 目标答案
        num_generations: 每个prompt生成的数量
    
    Returns:
        rewards: 奖励列表
    """
    # 预计算NDCG权重
    ndcg_weights = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_weights = [-w / sum(ndcg_weights) for w in ndcg_weights]
    
    rewards = []
    
    for i in range(0, len(completions), num_generations):
        # 获取这组completions和目标
        group_completions = completions[i:i+num_generations]
        group_target = targets[i].strip().strip('"').strip()
        
        # 检查目标是否在这组中
        found = False
        group_rewards = []
        
        for j, completion in enumerate(group_completions):
            completion_clean = completion.strip().strip('"').strip()
            
            if completion_clean == group_target:
                # 找到目标，给予0.0奖励（其他的会得到负奖励）
                group_rewards.append(0.0)
                found = True
            else:
                # 没找到，根据位置给予负奖励
                group_rewards.append(ndcg_weights[j])
        
        # 如果目标没有在任何生成中出现，所有奖励都是0
        if not found:
            group_rewards = [0.0] * num_generations
        
        rewards.extend(group_rewards)
    
    return rewards


def combined_reward(
    prompts: List[str], 
    completions: List[str], 
    targets: List[str],
    num_generations: int = 16,
    rule_weight: float = 0.5,
    ndcg_weight: float = 0.5,
    **kwargs
) -> List[float]:
    """
    组合奖励：rule + NDCG
    
    Args:
        prompts: 输入prompts
        completions: 模型生成的completions
        targets: 目标答案
        num_generations: 每个prompt生成的数量
        rule_weight: rule reward的权重
        ndcg_weight: NDCG reward的权重
    
    Returns:
        rewards: 奖励列表
    """
    rule_rewards = rule_reward(prompts, completions, targets, **kwargs)
    ndcg_rewards = ndcg_rule_reward(prompts, completions, targets, num_generations=num_generations, **kwargs)
    
    combined_rewards = [
        rule_weight * r + ndcg_weight * n
        for r, n in zip(rule_rewards, ndcg_rewards)
    ]
    
    return combined_rewards


def create_reward_function(reward_type: str = "rule", **config):
    """
    创建reward函数
    
    Args:
        reward_type: reward类型 ("rule", "ndcg", "combined")
        **config: reward函数的配置参数
    
    Returns:
        reward_func: reward函数
    """
    if reward_type == "rule":
        return rule_reward
    elif reward_type == "ndcg":
        def ndcg_reward_wrapper(prompts, completions, targets, **kwargs):
            return ndcg_rule_reward(
                prompts, 
                completions, 
                targets,
                num_generations=config.get("num_generations", 16),
                **kwargs
            )
        return ndcg_reward_wrapper
    elif reward_type == "combined":
        def combined_reward_wrapper(prompts, completions, targets, **kwargs):
            return combined_reward(
                prompts,
                completions,
                targets,
                num_generations=config.get("num_generations", 16),
                rule_weight=config.get("rule_weight", 0.5),
                ndcg_weight=config.get("ndcg_weight", 0.5),
                **kwargs
            )
        return combined_reward_wrapper
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
