"""
Best-of-N Evaluation for GRPO 7B
Strategy: Sampling -> Validity Filter -> Majority Voting
"""

import os
import sys
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from haversine import haversine
from collections import defaultdict, Counter

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

# ==========================================
# 1. Trie Constraints (保持不变，用于保证生成的全是合法ID)
# ==========================================
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, sequence):
        node = self.root
        for token in sequence:
            if token not in node: node[token] = {}
            node = node[token]
        node[-1] = True 
    def get_next_tokens(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node: return None
            node = node[token]
        return [k for k in node.keys() if k != -1]

class TrieConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length, trie):
        self.prompt_length = prompt_length
        self.trie = trie
    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            generated_tokens = input_ids[i, self.prompt_length:].tolist()
            allowed_next = self.trie.get_next_tokens(generated_tokens)
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if allowed_next is not None and len(allowed_next) > 0:
                mask[allowed_next] = False
                scores[i] = scores[i].masked_fill(mask, -float('inf'))
        return scores

# ==========================================
# 2. Evaluator Class (Modified for Best-of-N)
# ==========================================

class BestOfNEvaluator:
    def __init__(self, base_model, sft_ckpt, grpo_ckpt, mapping_file, device="cuda"):
        self.device = device
        self.sft_path = sft_ckpt
        self.grpo_path = grpo_ckpt
        self.load_mapping(mapping_file)
        self.load_model(base_model, sft_ckpt, grpo_ckpt)
        self.build_city_tries()
        
    def load_mapping(self, mapping_file):
        print(f"Loading mapping from {mapping_file}...")
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.sid_map = json.load(f)
        self.tree_map = {}
        self.city_sid_strings = defaultdict(list)
        for bid, meta in self.sid_map.items():
            full_code = tuple(int(x) for x in meta['full_sid'])
            self.tree_map[full_code] = {
                'lat': meta['latitude'], 'lon': meta['longitude'],
                'city': meta.get('city', 'Unknown')
            }
            sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
            self.city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)
            
    def load_model(self, base, sft, grpo):
        print(f"Loading Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 显存优化加载
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, sft)
        model = model.merge_and_unload()
        try:
            model = PeftModel.from_pretrained(model, grpo)
        except:
            print("Warning: GRPO adapter load failed.")
        model.eval()
        self.model = model

    def build_city_tries(self):
        print("Building Tries...")
        self.city_tries = {}
        for city, strings in tqdm(self.city_sid_strings.items(), desc="Cities"):
            trie = Trie()
            tokens_list = self.tokenizer(strings, add_special_tokens=False).input_ids
            for tokens in tokens_list: trie.insert(tokens)
            self.city_tries[city] = trie

    def parse_output(self, text):
        text = text.replace("[", "<").replace("]", ">")
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: return tuple(int(g) for g in match.groups())
        return None

    def evaluate(self, test_file, num_samples=500, n_samples=16):
        """
        n_samples: 对于每个问题，生成多少个候选答案 (Best-of-N 中的 N)
        """
        print(f"Evaluating {num_samples} samples with Best-of-{n_samples} Strategy...")
        data = []
        if not os.path.exists(test_file): return
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        test_data = [d for d in data if d.get('task') == 'task_a_recommendation']
        if len(test_data) > num_samples: test_data = test_data[-num_samples:]
        
        metrics = {'hit1': 0, 'hit5': 0, 'hit10': 0}
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0}
        distances = []
        
        for item in tqdm(test_data):
            meta = item.get('metadata', {})
            t_raw = meta.get('target_sid')
            if isinstance(t_raw, str): t_sid = tuple(int(x.strip()) for x in t_raw.replace('<','').replace('>','').split(','))
            else: t_sid = tuple(t_raw)
            t_city = self.tree_map[t_sid]['city'] if t_sid in self.tree_map else None
            
            raw_inst = item.get('instruction', '')
            base_prompt = raw_inst.split("Response:")[0].strip() if "Response:" in raw_inst else raw_inst.strip()
            final_prompt = f"{base_prompt}\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
            
            trie = self.city_tries.get(t_city)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            logits_processor = [TrieConstraintLogitsProcessor(prompt_len, trie)] if trie else []
            
            with torch.no_grad():
                # === [关键修改] 开启采样模式 ===
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=True,        # 开启随机采样
                    top_p=0.85,            # 核采样，去掉尾部离谱的答案
                    temperature=0.7,       # 适中的温度
                    num_return_sequences=n_samples, # 一次生成 N 个
                    logits_processor=logits_processor,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # === [关键修改] Best-of-N 选择逻辑 ===
            valid_candidates = []
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                text = "<" + self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pid = self.parse_output(text)
                # 过滤：必须是合法ID (在mapping里)
                if pid and pid in self.tree_map:
                    valid_candidates.append(pid)
            
            final_prediction = None
            if not valid_candidates:
                # 如果没有一个合法的，就随便瞎猜一个或者跳过
                continue
            else:
                # 投票策略：选出现次数最多的 ID
                # 如果有并列第一，选第一个遇到的
                counts = Counter(valid_candidates)
                final_prediction = counts.most_common(1)[0][0]
            
            # === 下面计算 Metrics 只看 final_prediction 这一次预测 ===
            # 这才是公平的 Hit@1，因为我们最终只输出了一个结果
            
            top1 = final_prediction
            if top1 == t_sid: metrics['hit1'] += 1
            # Hit@5/10 在 Best-of-N 模式下通常看选出的前几个高频项，或者只看 Top1
            # 为了保持一致，我们这里只计算 Strict Hit@1
            
            p_info = self.tree_map[top1]
            if meta.get('target_lat') is not None:
                dist = haversine((meta['target_lat'], meta['target_lon']), (p_info['lat'], p_info['lon']))
                distances.append(dist)
            
            if top1[0] == t_sid[0]:
                layer_hits[0] += 1
                if top1[1] == t_sid[1]:
                    layer_hits[1] += 1
                    if top1[2] == t_sid[2]:
                        layer_hits[2] += 1
                        if top1[3] == t_sid[3]:
                            layer_hits[3] += 1

        n = len(distances)
        if n == 0: return

        print("\n" + "="*40)
        print(f"BEST-OF-{n_samples} RESULTS (Consistency Vote)")
        print("="*40)
        print(f"Mean Distance: {np.mean(distances):.4f} km")
        print("-" * 20)
        # 注意：BoN 模式下，Hit@1 就是最终准确率
        print(f"Strict Hit@1 (Final Pred): {metrics['hit1']/n:.2%}")
        print("-" * 20)
        print("Hierarchical Accuracy (Final Pred):")
        print(f"Layer 0 (City)     : {layer_hits[0]/n:.2%}")
        print(f"Layer 1 (District) : {layer_hits[1]/n:.2%}")
        print(f"Layer 2 (Category) : {layer_hits[2]/n:.2%} <--- Watch This")
        print(f"Layer 3 (Item)     : {layer_hits[3]/n:.2%}")
        print("="*40)

if __name__ == "__main__":
    BASE = "/workspace/Qwen2.5-7B-Instruct"
    SFT  = "/workspace/data/llm_ckpt_sft_qwen2.5_7b_balanced/checkpoint-9400"
    # 这里填你最新的 ckpt 8400
    GRPO = "/workspace/data/grpo_qwen2.5_7b_breadcrumbs/checkpoint-10000"
    MAP  = "/workspace/data/processed/sid_mapping.json"
    TEST = "/workspace/data/processed/valid_prompts.jsonl"
    
    # N=16 意味着生成 16 个，然后投票选 1 个
    evaluator = BestOfNEvaluator(BASE, SFT, GRPO, MAP)
    evaluator.evaluate(TEST, num_samples=500, n_samples=16)