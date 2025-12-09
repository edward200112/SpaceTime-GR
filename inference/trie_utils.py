import torch
from transformers import LogitsProcessor

class Trie:
    def __init__(self, sequences=None):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for seq in sequences:
                self.insert(seq)

    def insert(self, sequence):
        """将一个 token id 序列插入 Trie"""
        node = self.trie_dict
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        # 标记结束符 (这里用 -1)
        node[-1] = True
        self.len += 1

    def get_next_tokens(self, prefix_sequence):
        """给定前缀，返回下一个允许的 token list"""
        node = self.trie_dict
        for token in prefix_sequence:
            if token not in node:
                return [] # 前缀不在树中，无路可走
            node = node[token]
        
        # 返回所有合法的 key (除了结束标记 -1)
        return [k for k in node.keys() if k != -1]

    def has_next(self, prefix_sequence):
        return len(self.get_next_tokens(prefix_sequence)) > 0

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, num_beams):
        self.trie = trie
        self.num_beams = num_beams
        self._prefix_cache = {} # 简单的缓存优化

    def __call__(self, input_ids, scores):
        # input_ids shape: [batch_size * num_beams, cur_len]
        # 我们只关心新生成的 token，所以需要知道 prompt 的长度
        # 但这里的 input_ids 通常包含了 prompt。
        # 这里我们假设 processor 是在 generate 内部调用的，我们需要动态追踪生成的部分。
        
        # 为了简化，我们假设 input_ids 的最后一部分是我们生成的。
        # 实际上 HuggingFace 的 LogitsProcessor 接收的是完整的 input_ids。
        # 我们需要一种方法知道 "生成从哪里开始"。
        # 通常做法：在外部记录 prompt_length，或者由 Evaluator 传入 start_len
        pass 
        # (由于这个类需要状态，我们在 evaluate_metrics.py 里直接定义一个带闭包的类更方便)
        return scores