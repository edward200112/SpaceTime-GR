"""
Final Evaluation for Qwen2.5-7B GRPO with Full Metrics (@20)
"""

import os
import sys
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from haversine import haversine
from collections import defaultdict
from math import log2

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

# ==========================================
# 1. Trie Constraint Logic (核心约束)
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
# 2. Evaluator Class (Updated Metrics)
# ==========================================

class GRPOEvaluator:
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
                'lat': meta['latitude'], 
                'lon': meta['longitude'],
                'city': meta.get('city', 'Unknown')
            }
            # 预先生成 Token String 供 Trie 使用
            sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
            self.city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)
            
        print(f"Loaded {len(self.tree_map)} POIs.")

    def load_model(self, base, sft, grpo):
        print(f"Loading Tokenizer: {base}")
        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading Base Model: {base}")
        # 显存优化：使用 bfloat16 加载
        model = AutoModelForCausalLM.from_pretrained(
            base, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device, 
            trust_remote_code=True
        )
        
        print(f"Merging SFT Adapter: {sft}")
        model = PeftModel.from_pretrained(model, sft)
        model = model.merge_and_unload()
        
        print(f"Loading GRPO Adapter: {grpo}")
        try:
            model = PeftModel.from_pretrained(model, grpo)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load GRPO adapter ({e}). Predicting with SFT only.")
        
        model.eval()
        self.model = model

    def build_city_tries(self):
        print("Building Tries (Constrained Decoding)...")
        self.city_tries = {}
        for city, strings in tqdm(self.city_sid_strings.items(), desc="Cities"):
            trie = Trie()
            tokens_list = self.tokenizer(strings, add_special_tokens=False).input_ids
            for tokens in tokens_list:
                trie.insert(tokens)
            self.city_tries[city] = trie

    def parse_output(self, text):
        text = text.replace("[", "<").replace("]", ">")
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: return tuple(int(g) for g in match.groups())
        return None

    def evaluate(self, test_file, num_samples=500, beams=20):
        """
        Modified to calculate Recall@20, NDCG@20, HitRate@20
        """
        print(f"Evaluating {num_samples} samples from {test_file} (Beams={beams})...")
        data = []
        
        if not os.path.exists(test_file):
            print(f"Error: Test file {test_file} not found.")
            return

        with open(test_file, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        
        test_data = [d for d in data if d.get('task') == 'task_a_recommendation']
        
        if len(test_data) > num_samples:
            test_data = test_data[-num_samples:]
        
        # 初始化指标计数器
        metrics = {
            'hit1': 0, 'hit5': 0, 'hit10': 0, 'hit20': 0, 
            'ndcg20': 0.0
        }
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0}
        distances = []
        
        print(f"Starting Inference...")
        
        for item in tqdm(test_data):
            meta = item.get('metadata', {})
            t_raw = meta.get('target_sid')
            
            if isinstance(t_raw, str):
                t_sid = tuple(int(x.strip()) for x in t_raw.replace('<','').replace('>','').split(','))
            else:
                t_sid = tuple(t_raw)
            
            t_city = self.tree_map[t_sid]['city'] if t_sid in self.tree_map else None
            
            raw_inst = item.get('instruction', '')
            if "Response:" in raw_inst:
                base_prompt = raw_inst.split("Response:")[0].strip()
            else:
                base_prompt = raw_inst.strip()
                
            final_prompt = f"{base_prompt}\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
            
            trie = self.city_tries.get(t_city)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            
            logits_processor = [TrieConstraintLogitsProcessor(prompt_len, trie)] if trie else []
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=24,
                    num_beams=beams,             # 建议设为 20 以计算 @20 指标
                    num_return_sequences=beams,
                    logits_processor=logits_processor,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            candidates = []
            seen = set()
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                text = "<" + self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pid = self.parse_output(text)
                
                if pid and pid in self.tree_map and pid not in seen:
                    candidates.append(pid)
                    seen.add(pid)
            
            if not candidates: continue

            # === 计算指标 ===
            # Hit@K (Recall@K / HitRate@K)
            if t_sid in candidates[:1]: metrics['hit1'] += 1
            if t_sid in candidates[:5]: metrics['hit5'] += 1
            if t_sid in candidates[:10]: metrics['hit10'] += 1
            if t_sid in candidates[:20]: metrics['hit20'] += 1 # 也就是 Recall@20 / HitRate@20
            
            # NDCG@20
            # 只有一个正确答案 t_sid，所以 IDCG = 1.0
            # 如果 t_sid 在前 20 个候选里，DCG = 1 / log2(rank + 2)
            if t_sid in candidates[:20]:
                rank = candidates.index(t_sid) # 0-indexed
                metrics['ndcg20'] += 1.0 / log2(rank + 2)
            
            # 距离与层级 (只看 Top-1)
            top1 = candidates[0]
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
        if n == 0: 
            print("No valid predictions generated.")
            return

        print("\n" + "="*40)
        print(f"FINAL GRPO RESULTS (N={n}, Beams={beams})")
        print("="*40)
        print(f"Mean Distance: {np.mean(distances):.4f} km")
        print("-" * 20)
        print(f"Hit@1      : {metrics['hit1']/n:.2%}")
        print(f"Hit@5      : {metrics['hit5']/n:.2%}")
        print(f"Hit@10     : {metrics['hit10']/n:.2%}")
        print(f"HitRate@20 : {metrics['hit20']/n:.2%}")
        print(f"Recall@20  : {metrics['hit20']/n:.2%}")
        print(f"NDCG@20    : {metrics['ndcg20']/n:.4f}")
        print("-" * 20)
        print("Hierarchical Accuracy (Top-1):")
        print(f"Layer 0 (City)     : {layer_hits[0]/n:.2%}")
        print(f"Layer 1 (District) : {layer_hits[1]/n:.2%}")
        print(f"Layer 2 (Category) : {layer_hits[2]/n:.2%}")
        print(f"Layer 3 (Item)     : {layer_hits[3]/n:.2%}")
        print("="*40)

if __name__ == "__main__":
    BASE = "/workspace/Qwen2.5-7B-Instruct"
    SFT  = "/workspace/data/llm_ckpt_sft_qwen2.5_7b_balanced/checkpoint-9400"
    GRPO = "/workspace/data/grpo_qwen2.5_7b_breadcrumbs/checkpoint-10000"
    
    MAP  = "/workspace/data/processed/sid_mapping.json"
    
    # 建议使用验证集
    TEST = "/workspace/data/processed/valid_prompts.jsonl"
    if not os.path.exists(TEST):
        TEST = "/workspace/data/processed/train_prompts.jsonl"
    
    evaluator = GRPOEvaluator(BASE, SFT, GRPO, MAP)
    
    # [关键] 为了计算 NDCG@20，Beams 至少要设为 20
    # 如果显存 OOM，请调小 Beams，但那样 @20 指标就不准确了
    evaluator.evaluate(TEST, num_samples=500, beams=20)