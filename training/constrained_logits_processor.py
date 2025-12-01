"""
Constrained Logits Processor for HierGR-SeqRec

约束生成，确保模型只输出有效的 Cluster IDs
"""

import torch
from transformers.generation import LogitsProcessor
from transformers.utils import add_start_docstrings
from typing import Callable, List


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head.
    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
"""


class ConstrainedClusterLogitsProcessor(LogitsProcessor):
    """
    约束生成处理器，确保模型只生成有效的 Cluster ID 序列
    
    例如：生成 "<3, 12>" 时，确保每一步都是有效的 token
    """
    
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        model_type: str = "qwen"
    ):
        """
        Args:
            prefix_allowed_tokens_fn: 根据当前前缀返回允许的下一个tokens
            num_beams: Beam search 的数量
            model_type: 模型类型（保留参数以兼容，但不再使用）
        """
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.prompt_length = None  # 记录prompt的原始长度
    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        在生成的每一步，约束只能选择有效的tokens
        """
        # Log-softmax 转换
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        
        # 创建mask，默认所有tokens都被mask（-inf）
        mask = torch.full_like(scores, -1e10)
        
        # 第一次调用时记录prompt长度
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[-1]
        
        # 对每个beam进行处理
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                # 获取当前的prefix（用于查找允许的tokens）
                # 只从生成的新token中提取，不包括原始prompt
                generated_tokens = sent[self.prompt_length:].tolist()
                
                if len(generated_tokens) == 0:
                    # 第一步：使用空前缀，允许所有起始tokens
                    hash_key = []
                else:
                    # 后续步骤：使用已生成的tokens作为前缀
                    hash_key = generated_tokens
                
                # 获取允许的tokens
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                
                if len(prefix_allowed_tokens) == 0:
                    # 如果没有允许的tokens，跳过（保持-inf）
                    continue
                
                # 只对允许的tokens解除mask
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0
        
        # 增加计数（跟踪生成的步数）
        self.count += 1
        
        # 应用mask
        scores = scores + mask
        return scores


def build_cluster_hash_dict(sid_mapping: dict, tokenizer, model_type: str = "qwen"):
    """
    构建 Cluster ID 的哈希字典，用于约束生成
    
    Args:
        sid_mapping: SID映射字典，格式: {business_id: {cluster_str: "<3, 12>", ...}}
        tokenizer: Tokenizer对象
        model_type: 模型类型
    
    Returns:
        hash_dict: 哈希字典，key为token序列的hash，value为允许的下一个tokens
        prefix_index: 前缀索引位置
    """
    # 提取所有唯一的cluster_str
    unique_clusters = set()
    for item_info in sid_mapping.values():
        unique_clusters.add(item_info['cluster_str'])
    
    unique_clusters = sorted(list(unique_clusters))
    print(f"Found {len(unique_clusters)} unique clusters")
    print(f"Example clusters: {list(unique_clusters)[:5]}")
    
    # 直接使用cluster_str，不添加格式前缀
    # 训练数据的prompt已经包含了instruction，生成部分直接是cluster
    
    # Tokenize
    if "llama" in model_type.lower():
        prefix_ids = [tokenizer(cluster).input_ids[1:] for cluster in unique_clusters]  # Skip BOS
        prefix_index = 0  # 从第一个token开始
    elif "gpt2" in model_type.lower():
        prefix_ids = [tokenizer(cluster).input_ids for cluster in unique_clusters]
        prefix_index = 0
    else:  # Qwen and others
        prefix_ids = [tokenizer(cluster).input_ids for cluster in unique_clusters]
        prefix_index = 0  # 从第一个token开始，不需要跳过prompt部分
    
    # Debug: 检查tokenization（已关闭）
    # print(f"DEBUG: 检查cluster的tokenization:")
    # for i in range(min(3, len(unique_clusters))):
    #     cluster = unique_clusters[i]
    #     tokens = prefix_ids[i]
    #     print(f"  '{cluster}' -> {tokens} (长度: {len(tokens)})")
    #     decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    #     print(f"    解码回: '{decoded}'")
    
    # 构建哈希字典
    hash_dict = {}
    
    def get_hash(x):
        """将token序列转换为字符串hash"""
        return '-'.join([str(token) for token in x])
    
    for token_ids in prefix_ids:
        # 添加EOS token
        token_ids.append(tokenizer.eos_token_id)
        
        # 为每个位置创建hash映射
        for i in range(prefix_index, len(token_ids)):
            if i == prefix_index:
                hash_key = get_hash(token_ids[:i])
            else:
                hash_key = get_hash(token_ids[prefix_index:i])
            
            if hash_key not in hash_dict:
                hash_dict[hash_key] = set()
            
            hash_dict[hash_key].add(token_ids[i])
    
    # 转换set为list
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    
    print(f"Built hash dict with {len(hash_dict)} entries")
    
    # Debug: 显示前几个entries（已关闭）
    # print(f"DEBUG hash_dict示例 (前5个):")
    # for i, (k, v) in enumerate(list(hash_dict.items())[:5]):
    #     print(f"  '{k}' -> {v[:3] if len(v) > 3 else v}... (共{len(v)}个tokens)")
    
    return hash_dict, prefix_index, get_hash


def create_prefix_allowed_tokens_fn(hash_dict: dict, get_hash_fn):
    """
    创建prefix_allowed_tokens_fn函数
    
    Args:
        hash_dict: 哈希字典
        get_hash_fn: 哈希函数
    
    Returns:
        prefix_allowed_tokens_fn: 用于ConstrainedLogitsProcessor的函数
    """
    # 预计算所有可能的起始tokens（用于空前缀）
    # 现在hash_dict中空字符串key对应第一步生成的起始tokens
    all_start_tokens = set()
    
    # 查找空字符串key（对应空prefix）
    empty_key = get_hash_fn([])  # 应该返回空字符串 ''
    
    # print(f"DEBUG start_tokens查找:")
    # print(f"  - 空key表示: '{empty_key}'")
    
    if empty_key in hash_dict:
        all_start_tokens.update(hash_dict[empty_key])
        # print(f"  - ✅ 找到空key对应的起始tokens: {len(all_start_tokens)} 个")
    else:
        # print(f"  - ⚠️ 警告：空key '{empty_key}' 不在hash_dict中！")
        # print(f"  - hash_dict的前5个key: {list(hash_dict.keys())[:5]}")
        pass
        # Fallback: 使用最短key
        min_len = min(len(k) for k in hash_dict.keys())
        for key, tokens in hash_dict.items():
            if len(key) == min_len:
                all_start_tokens.update(tokens)
                break
    
    all_start_tokens = list(all_start_tokens)
    
    # print(f"DEBUG: Found {len(all_start_tokens)} start tokens")
    # if len(all_start_tokens) > 0:
    #     print(f"DEBUG: Example start tokens: {all_start_tokens[:5]}")
    
    _call_count = [0]  # 使用列表来在闭包中修改
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # 特殊处理：空前缀（第一步生成）
        if len(input_ids) == 0:
            # if _call_count[0] == 0:
            #     print(f"DEBUG prefix_allowed_tokens_fn: 空前缀，返回 {len(all_start_tokens)} 个start tokens")
            _call_count[0] += 1
            return all_start_tokens
        
        hash_key = get_hash_fn(input_ids)
        if hash_key in hash_dict:
            result = hash_dict[hash_key]
            # if _call_count[0] < 3:  # 只打印前3次
            #     print(f"DEBUG prefix_allowed_tokens_fn: hash_key='{hash_key}' -> {len(result)} tokens")
            _call_count[0] += 1
            return result
        
        # if _call_count[0] < 3:
        #     print(f"DEBUG prefix_allowed_tokens_fn: hash_key='{hash_key}' 不在hash_dict中，返回空列表")
        _call_count[0] += 1
        return []
    
    return prefix_allowed_tokens_fn
