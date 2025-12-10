"""
Quantitative Evaluation for HierGR-SeqRec (Final Fixed Version)
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
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

def setup_environment():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
setup_environment()

# --- Trie Components ---
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
            beam_tokens = generated_tokens[i].tolist()
            allowed = self.trie.get_next_tokens(beam_tokens)
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if allowed is not None and len(allowed) > 0:
                mask[allowed] = False
                scores[i] = scores[i].masked_fill(mask, -float('inf'))
        return scores

# --- Evaluator ---
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
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        
        print(f"Merging SFT: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
        
        if grpo_path:
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
            # Step 2 生成的 sid_str 格式: "<12, 34, 56, 0>"
            city_sid_strings[city].append(meta['sid_str'])

        print(f"Building Tries for {len(city_sid_strings)} cities...")
        city_tries = {}
        for city, sid_strs in tqdm(city_sid_strings.items(), desc="Building"):
            trie = Trie()
            tokenized = self.tokenizer(sid_strs, add_special_tokens=False).input_ids
            for seq in tokenized: trie.insert(seq)
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
        
        # 只保留 Task A
        task_a = [d for d in data if d['task'] == 'task_a_recommendation']
        # 取最后 500 条
        test_set = task_a[-500:] 
        if limit: test_set = test_set[:limit]
        return test_set

    def evaluate(self, k_list=[1, 5, 10], num_beams=10):
        test_data = self.load_test_data(limit=100) 
        print(f"Starting evaluation on {len(test_data)} samples...")
        
        metrics = {k: 0 for k in k_list}
        ndcg_metrics = {k: 0 for k in k_list}
        distance_errors = []
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0}
        
        valid_count = 0
        skipped = 0
        
        for i, sample in tqdm(enumerate(test_data), total=len(test_data)):
            try:
                # [FIX] 解析 Step 4 的 String Target
                raw_sid = sample['metadata']['target_sid']
                if isinstance(raw_sid, str):
                    clean = raw_sid.replace('<','').replace('>','')
                    target_sid = tuple(int(x.strip()) for x in clean.split(','))
                else:
                    target_sid = tuple(int(x) for x in raw_sid)
                
                if target_sid in self.tree_map:
                    target_city = self.tree_map[target_sid]['city']
                else:
                    skipped += 1; continue
            except Exception: skipped += 1; continue
            
            target_coords = (sample['metadata']['target_lat'], sample['metadata']['target_lon'])
            
            # 使用包含 History 的原始 Instruction
            base_instr = sample['instruction']
            if "Output the semantic ID" not in base_instr:
                instr = f"{base_instr}\nOutput the semantic ID in the format <c0, c1, c2, suffix>."
            else: instr = base_instr
            
            messages = [{"role": "user", "content": instr}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            
            trie = self.city_tries.get(target_city)
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=32, num_beams=num_beams,
                        num_return_sequences=num_beams,
                        logits_processor=[TrieConstraintLogitsProcessor(prompt_len, trie)] if trie else [],
                        early_stopping=True, pad_token_id=self.tokenizer.eos_token_id
                    )
            except Exception: continue

            candidates = []
            for seq in outputs:
                new_tokens = seq[inputs.input_ids.shape[1]:]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pid = self.parse_output(decoded)
                if pid and pid in self.tree_map: candidates.append(pid)
            
            unique_cands = []
            [unique_cands.append(x) for x in candidates if x not in unique_cands]
            
            if not unique_cands: continue
            valid_count += 1
            
            # Metrics
            for k in k_list:
                top_k = unique_cands[:k]
                if target_sid in top_k:
                    metrics[k] += 1
                    rank = top_k.index(target_sid) + 1
                    ndcg_metrics[k] += 1.0 / np.log2(rank + 1)
            
            top1 = unique_cands[0]
            if top1 in self.tree_map:
                pmeta = self.tree_map[top1]
                dist = haversine(target_coords, (pmeta['lat'], pmeta['lon']))
                distance_errors.append(dist)
            
            # Layer Accuracy
            if top1[0] == target_sid[0]:
                layer_hits[0] += 1
                if top1[1] == target_sid[1]:
                    layer_hits[1] += 1
                    if top1[2] == target_sid[2]:
                        layer_hits[2] += 1
                        if top1[3] == target_sid[3]:
                            layer_hits[3] += 1

        n = len(test_data) - skipped
        if n == 0: return

        print("\n" + "="*40)
        print(f"FINAL RESULTS (N={n})")
        print(f"Mean Distance: {np.mean(distance_errors):.4f} km")
        print("-" * 20)
        print(f"Hit@1 : {metrics[1]/n:.4f}")
        print(f"Hit@5 : {metrics[5]/n:.4f}")
        print("-" * 20)
        print(f"Layer 0 Match: {layer_hits[0]/n:.2%}")
        print(f"Layer 1 Match: {layer_hits[1]/n:.2%}")
        print(f"Layer 2 Match: {layer_hits[2]/n:.2%}")
        print(f"Exact Match  : {layer_hits[3]/n:.2%}")
        print("="*40)

def main():
    config_path = './config/config.yaml'
    sft_ckpt = "/workspace/data/llm_ckpt/checkpoint-28000"
    # 记得改成新的 GRPO 路径
    grpo_ckpt = "/workspace/data/llm_ckpt/grpo_optimized" 
    if not os.path.exists(grpo_ckpt): grpo_ckpt = None
    
    evaluator = RecEvaluator(config_path, sft_ckpt, grpo_ckpt)
    evaluator.evaluate(k_list=[1, 5, 10], num_beams=10)

if __name__ == "__main__":
    main()