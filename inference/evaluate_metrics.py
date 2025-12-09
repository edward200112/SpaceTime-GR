"""
Quantitative Evaluation for HierGR-SeqRec (Final Version)
Metrics: Hit@K, NDCG@K, Distance, AND Layer-wise Accuracy
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from haversine import haversine 
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

def setup_environment():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

setup_environment()

# ... (Trie 和 LogitsProcessor 类保持不变，为了节省篇幅省略，请保留之前的 Trie 类代码) ...
# 请确保 Trie 和 TrieConstraintLogitsProcessor 还在代码里！
# ---------------------------------------------------------------------------
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
        generated_tokens = input_ids[:, self.prompt_length:]
        for i in range(generated_tokens.shape[0]):
            beam_tokens_list = generated_tokens[i].tolist()
            allowed_next = self.trie.get_next_tokens(beam_tokens_list)
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if allowed_next is not None and len(allowed_next) > 0:
                mask[allowed_next] = False
                scores[i] = scores[i].masked_fill(mask, -float('inf'))
        return scores
# ---------------------------------------------------------------------------

class RecEvaluator:
    def __init__(self, config_path, sft_path, grpo_path):
        self.config = self.load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self.load_model(sft_path, grpo_path)
        self.sid_map, self.tree_map, self.city_tries = self.load_mapping_and_build_tries()
        
    def load_config(self, path):
        with open(path, 'r') as f: return yaml.safe_load(f)

    def load_model(self, sft_path, grpo_path):
        base_path = self.config['llm']['model_name']
        print(f"Loading Base: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True
        )
        print(f"Merging SFT: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
        print(f"Loading GRPO: {grpo_path}")
        model = PeftModel.from_pretrained(model, grpo_path)
        model.eval()
        return tokenizer, model

    def load_mapping_and_build_tries(self):
        map_file = os.path.join(self.config['data']['processed_dir'], self.config['data']['sid_mapping_file'])
        print(f"Loading mapping... {map_file}")
        with open(map_file, 'r') as f: raw_map = json.load(f)
            
        tree_map = {} 
        city_sid_strings = defaultdict(list) 

        for bid, meta in raw_map.items():
            full_code = tuple(int(x) for x in meta['full_sid'])
            city = meta['city']
            tree_map[full_code] = {
                'lat': meta['latitude'], 'lon': meta['longitude'],
                'city': city, 'business_id': bid, 'name': meta.get('name', 'Unknown')
            }
            sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
            city_sid_strings[city].append(sid_str)

        print(f"Building Tries for {len(city_sid_strings)} cities...")
        city_tries = {}
        for city, sid_strs in tqdm(city_sid_strings.items(), desc="Building Tries"):
            trie = Trie()
            tokenized_ids = self.tokenizer(sid_strs, add_special_tokens=False).input_ids
            for seq in tokenized_ids: trie.insert(seq)
            city_tries[city] = trie
        return raw_map, tree_map, city_tries

    def parse_output(self, text):
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: return tuple(int(g) for g in match.groups())
        return None

    def load_test_data(self, limit=None):
        data_path = os.path.join(self.config['data']['processed_dir'], self.config['data']['train_prompts_file'])
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            try:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
            except json.JSONDecodeError:
                f.seek(0); data = json.load(f)
        task_a_data = [d for d in data if d['task'] == 'task_a_recommendation']
        # 取最后 1000 条
        test_set = task_a_data[-1000:] 
        if limit: test_set = test_set[:limit]
        return test_set

    def evaluate(self, k_list=[1, 5, 10], num_beams=10):
        # [建议] 这里增加到 500 或 1000 条，以获得稳定的统计结果
        test_data = self.load_test_data(limit=500) 
        print(f"Starting evaluation on {len(test_data)} samples...")
        
        metrics = {k: 0 for k in k_list}
        ndcg_metrics = {k: 0 for k in k_list}
        distance_errors = [] 
        
        # [NEW] 层级指标
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0} # Layer 0, 0-1, 0-2, Full
        
        valid_count = 0
        skipped_count = 0
        
        for i, sample in tqdm(enumerate(test_data), total=len(test_data)):
            try:
                raw_sid = sample['metadata']['target_sid']
                if isinstance(raw_sid, str):
                    clean_str = raw_sid.replace('<', '').replace('>', '')
                    target_sid = tuple(int(x.strip()) for x in clean_str.split(','))
                else:
                    target_sid = tuple(int(x) for x in raw_sid)
                
                if target_sid in self.tree_map:
                    target_city = self.tree_map[target_sid]['city']
                else:
                    skipped_count += 1; continue
            except Exception: skipped_count += 1; continue
            
            target_coords = (sample['metadata']['target_lat'], sample['metadata']['target_lon'])
            base_instruction = sample['instruction']
            if "Output the semantic ID" not in base_instruction:
                instruction = f"{base_instruction}\nOutput the semantic ID in the format <c0, c1, c2, suffix>."
            else: instruction = base_instruction
            
            messages = [{"role": "user", "content": instruction}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            
            current_trie = self.city_tries.get(target_city)
            
            # 这里为了速度，我们只用 Beam=1 或 Beam=3 也可以，或者保持 10
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    logits_processor=[TrieConstraintLogitsProcessor(prompt_len, current_trie)] if current_trie else [],
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            candidates = []
            for output_seq in outputs:
                new_tokens = output_seq[inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed_id = self.parse_output(text)
                if parsed_id and parsed_id in self.tree_map: candidates.append(parsed_id)
            
            unique_candidates = []
            [unique_candidates.append(x) for x in candidates if x not in unique_candidates]
            
            if not unique_candidates: continue 
            valid_count += 1
            
            # --- Metrics Calculation ---
            
            # 1. Standard Metrics
            for k in k_list:
                top_k = unique_candidates[:k]
                if target_sid in top_k:
                    metrics[k] += 1
                    rank = top_k.index(target_sid) + 1
                    ndcg_metrics[k] += 1.0 / np.log2(rank + 1)
            
            # 2. Geo Distance (Top 1)
            top_1_id = unique_candidates[0]
            if top_1_id in self.tree_map:
                pred_meta = self.tree_map[top_1_id]
                dist = haversine(target_coords, (pred_meta['lat'], pred_meta['lon']))
                distance_errors.append(dist)

            # 3. [NEW] Layer-wise Accuracy (Top 1 Only)
            # 只要 Top 1 的前 N 层对上了就算对
            # target_sid: (c0, c1, c2, suffix)
            # top_1_id: (c0', c1', c2', suffix')
            
            if top_1_id[0] == target_sid[0]:
                layer_hits[0] += 1 # 城市/大区对上了
                if top_1_id[1] == target_sid[1]:
                    layer_hits[1] += 1 # 街区对上了
                    if top_1_id[2] == target_sid[2]:
                        layer_hits[2] += 1 # 类别对上了
                        if top_1_id[3] == target_sid[3]:
                            layer_hits[3] += 1 # 完全命中 (Hit@1)

        n_samples = len(test_data) - skipped_count
        if n_samples == 0: return

        print("\n" + "="*40)
        print(f"FINAL RESULTS (N={n_samples})")
        print("="*40)
        print(f"Mean Distance Error: {np.mean(distance_errors):.4f} km")
        print("-" * 20)
        print("Standard Metrics:")
        for k in k_list:
            print(f"Hit@{k:<2}: {metrics[k]/n_samples:.4f} | NDCG@{k:<2}: {ndcg_metrics[k]/n_samples:.4f}")
        print("-" * 20)
        print("Hierarchical Accuracy (Top-1):")
        print(f"Layer 0 Match (City/Region): {layer_hits[0]/n_samples:.2%}")
        print(f"Layer 1 Match (District)   : {layer_hits[1]/n_samples:.2%}")
        print(f"Layer 2 Match (Category)   : {layer_hits[2]/n_samples:.2%}")
        print(f"Exact Match (Item)         : {layer_hits[3]/n_samples:.2%}")
        print("="*40)

def main():
    config_path = './config/config.yaml'
    sft_ckpt = "/workspace/data/llm_ckpt/checkpoint-28000"
    grpo_ckpt = "/workspace/data/grpo_checkpoints/checkpoint-8400"
    
    evaluator = RecEvaluator(config_path, sft_ckpt, grpo_ckpt)
    evaluator.evaluate(k_list=[1, 5, 10], num_beams=10)

if __name__ == "__main__":
    main()