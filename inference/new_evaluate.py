"""
Quantitative Evaluation for HierGR-SeqRec (Final Optimized Version)
Metrics: Hit@K, NDCG@K, Distance, AND Layer-wise Accuracy
Author: HierGR Team
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import re
import argparse
from tqdm import tqdm
from haversine import haversine 
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

# =============================================================================
# 1. Trie Constraint Logic (Full Implementation)
# =============================================================================

class Trie:
    def __init__(self):
        self.root = {}
        
    def insert(self, sequence):
        """Sequence is a list of token IDs"""
        node = self.root
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        node[-1] = True # Mark end of sequence

    def get_next_tokens(self, prefix):
        """Returns allowed next tokens given a prefix list of token IDs"""
        node = self.root
        for token in prefix:
            if token not in node:
                return None # Prefix not in Trie
            node = node[token]
        return [k for k in node.keys() if k != -1]

class TrieConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length, trie):
        self.prompt_length = prompt_length
        self.trie = trie

    def __call__(self, input_ids, scores):
        # input_ids: [batch_size, current_seq_len]
        # scores: [batch_size, vocab_size]
        
        # We process each beam/sequence in the batch
        for i in range(input_ids.shape[0]):
            # Extract only the generated part
            full_seq = input_ids[i].tolist()
            generated_part = full_seq[self.prompt_length:]
            
            allowed_next = self.trie.get_next_tokens(generated_part)
            
            # Create a mask of -inf
            mask = torch.full_like(scores[i], float('-inf'))
            
            if allowed_next is not None and len(allowed_next) > 0:
                # Allow only valid tokens
                mask[allowed_next] = 0
                scores[i] = scores[i] + mask
            else:
                # If path is dead end or finished, usually we let EOS happen or standard sampling
                # For strict constraint, if it returns None, it means we went off track. 
                # Ideally this shouldn't happen if we constrained from step 1.
                pass 
                
        return scores

# =============================================================================
# 2. Evaluator Class
# =============================================================================

class RecEvaluator:
    def __init__(self, config_path, sft_path, grpo_path, device="cuda"):
        self.config = self.load_config(config_path)
        self.device = device
        self.tokenizer, self.model = self.load_model(sft_path, grpo_path)
        self.sid_map, self.tree_map, self.city_tries = self.load_mapping_and_build_tries()
        
    def load_config(self, path):
        with open(path, 'r') as f: return yaml.safe_load(f)

    def load_model(self, sft_path, grpo_path):
        base_path = self.config['llm']['model_name']
        print(f"Loading Base Model: {base_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        tokenizer.padding_side = 'left' # Important for batch generation if extended
        
        model = AutoModelForCausalLM.from_pretrained(
            base_path, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device, 
            trust_remote_code=True
        )
        
        if sft_path:
            print(f"Merging SFT Adapter: {sft_path}")
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()
        
        if grpo_path:
            print(f"Loading GRPO Adapter: {grpo_path}")
            model = PeftModel.from_pretrained(model, grpo_path)
        
        model.eval()
        return tokenizer, model

    def load_mapping_and_build_tries(self):
        map_file = os.path.join(self.config['data']['processed_dir'], self.config['data']['sid_mapping_file'])
        print(f"Loading Semantic ID Mapping from {map_file}...")
        
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Mapping file not found: {map_file}")
            
        with open(map_file, 'r') as f: 
            raw_map = json.load(f)
            
        tree_map = {} 
        city_sid_strings = defaultdict(list) 

        # Build Lookups
        for bid, meta in raw_map.items():
            full_code = tuple(int(x) for x in meta['full_sid'])
            city = meta['city']
            tree_map[full_code] = {
                'lat': meta['latitude'], 
                'lon': meta['longitude'],
                'city': city, 
                'categories': meta.get('categories', ''),
                'name': meta.get('name', 'Unknown')
            }
            # String format for Tokenizer: <c0, c1, c2, c3>
            sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
            city_sid_strings[city].append(sid_str)

        print(f"Building Constraints (Tries) for {len(city_sid_strings)} cities...")
        city_tries = {}
        # Pre-build Tries for cities to speed up inference
        for city, sid_strs in tqdm(city_sid_strings.items(), desc="Building Tries"):
            trie = Trie()
            # Note: add_special_tokens=False is crucial here
            tokenized_ids = self.tokenizer(sid_strs, add_special_tokens=False).input_ids
            for seq in tokenized_ids: 
                trie.insert(seq)
            city_tries[city] = trie
            
        return raw_map, tree_map, city_tries

    def parse_output(self, text):
        # Regex to find <12, 34, 56, 78>
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: 
            return tuple(int(g) for g in match.groups())
        return None

    def load_test_data(self, limit=None):
        data_path = os.path.join(self.config['data']['processed_dir'], self.config['data']['train_prompts_file'])
        print(f"Loading Test Data from {data_path}")
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # Handle both line-json and full-json list
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
        
        # Filter for Task A and take the LAST portion (Simulating Test Split)
        task_a_data = [d for d in data if d['task'] == 'task_a_recommendation']
        
        # Heuristic: Take last 1000 items as test set
        test_set = task_a_data[-1000:] 
        
        if limit: 
            test_set = test_set[:limit]
            
        print(f"Evaluation Set Size: {len(test_set)}")
        return test_set

    def evaluate(self, k_list=[1, 5, 10], num_beams=10, limit=500):
        test_data = self.load_test_data(limit=limit)
        
        metrics = {k: 0 for k in k_list}
        ndcg_metrics = {k: 0 for k in k_list}
        distance_errors = [] 
        
        # Hierarchical Counters
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0} 
        
        skipped_count = 0
        valid_samples = 0
        
        # For Case Study
        case_studies = []
        
        print(f"Starting Inference (Beams={num_beams})...")
        
        for i, sample in tqdm(enumerate(test_data), total=len(test_data)):
            # 1. Parse Ground Truth
            try:
                raw_sid = sample['metadata']['target_sid']
                if isinstance(raw_sid, str):
                    clean_str = raw_sid.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
                    target_sid = tuple(int(x.strip()) for x in clean_str.split(','))
                else:
                    target_sid = tuple(int(x) for x in raw_sid)
                
                if target_sid in self.tree_map:
                    target_city = self.tree_map[target_sid]['city']
                    target_info = self.tree_map[target_sid]
                else:
                    # Skip if target not in mapping (should be rare)
                    skipped_count += 1
                    continue
            except Exception as e: 
                skipped_count += 1
                continue
            
            # 2. Prepare Input
            target_coords = (sample['metadata']['target_lat'], sample['metadata']['target_lon'])
            base_instruction = sample['instruction']
            
            # Ensure prompt explicitly asks for format
            if "Output the semantic ID" not in base_instruction:
                instruction = f"{base_instruction}\nOutput the semantic ID in the format <c0, c1, c2, suffix>."
            else: 
                instruction = base_instruction
            
            messages = [{"role": "user", "content": instruction}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            
            # 3. Constrained Generation
            current_trie = self.city_tries.get(target_city)
            logits_processor = LogitsProcessorList()
            if current_trie:
                logits_processor.append(TrieConstraintLogitsProcessor(prompt_len, current_trie))
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=24, # ID shouldn't be very long
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    logits_processor=logits_processor,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 4. Process Outputs
            candidates = []
            for output_seq in outputs:
                new_tokens = output_seq[inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed_id = self.parse_output(text)
                
                # Filter valid IDs
                if parsed_id and parsed_id in self.tree_map:
                    candidates.append(parsed_id)
            
            # Deduplicate preserving order
            unique_candidates = []
            seen = set()
            for c in candidates:
                if c not in seen:
                    unique_candidates.append(c)
                    seen.add(c)
            
            if not unique_candidates: 
                # Model failed to generate valid ID even with constraints (very rare)
                continue 
                
            valid_samples += 1
            
            # 5. Calculate Metrics
            # --- Standard Rec Metrics ---
            for k in k_list:
                top_k = unique_candidates[:k]
                if target_sid in top_k:
                    metrics[k] += 1
                    rank = top_k.index(target_sid) + 1
                    ndcg_metrics[k] += 1.0 / np.log2(rank + 1)
            
            # --- Geo Distance (Top 1) ---
            top_1_id = unique_candidates[0]
            pred_meta = self.tree_map[top_1_id]
            dist = haversine(target_coords, (pred_meta['lat'], pred_meta['lon']))
            distance_errors.append(dist)

            # --- Layer-wise Accuracy (Top 1) ---
            # target_sid: (c0, c1, c2, suffix)
            layer0_match = (top_1_id[0] == target_sid[0])
            layer1_match = layer0_match and (top_1_id[1] == target_sid[1])
            layer2_match = layer1_match and (top_1_id[2] == target_sid[2])
            layer3_match = layer2_match and (top_1_id[3] == target_sid[3])
            
            if layer0_match: layer_hits[0] += 1
            if layer1_match: layer_hits[1] += 1
            if layer2_match: layer_hits[2] += 1
            if layer3_match: layer_hits[3] += 1
            
            # Capture Case Study (First 3 samples)
            if valid_samples <= 3:
                case_studies.append({
                    "input_city": target_city,
                    "target_name": target_info['name'],
                    "target_id": target_sid,
                    "pred_name": pred_meta['name'],
                    "pred_id": top_1_id,
                    "distance": dist,
                    "hit_layer_2": layer2_match
                })

        # =====================================================================
        # Final Report
        # =====================================================================
        if valid_samples == 0:
            print("No valid samples evaluated.")
            return

        print("\n" + "="*50)
        print(f"EVALUATION REPORT (N={valid_samples})")
        print("="*50)
        
        print("\n[1] Geographic Performance")
        print(f"Mean Distance Error: {np.mean(distance_errors):.4f} km")
        
        print("\n[2] Standard Recommendation Metrics")
        print(f"{'K':<5} | {'Hit@K':<10} | {'NDCG@K':<10}")
        print("-" * 30)
        for k in k_list:
            print(f"{k:<5} | {metrics[k]/valid_samples:.4f}     | {ndcg_metrics[k]/valid_samples:.4f}")
            
        print("\n[3] Hierarchical Accuracy (Diagnostic)")
        print(f"Layer 0 (City/Region) Match : {layer_hits[0]/valid_samples:.2%}  <-- Should be > 90%")
        print(f"Layer 1 (District) Match    : {layer_hits[1]/valid_samples:.2%}  <-- Should be > 60%")
        print(f"Layer 2 (Category) Match    : {layer_hits[2]/valid_samples:.2%}  <-- V2 Goal: > 20%")
        print(f"Exact Match (Item Level)    : {layer_hits[3]/valid_samples:.2%}")

        print("\n[4] Case Studies (First 3)")
        for cs in case_studies:
            status = "✅" if cs['hit_layer_2'] else "❌"
            print(f"- Target: {cs['target_name']} ({cs['target_id']})")
            print(f"  Pred  : {cs['pred_name']} ({cs['pred_id']})")
            print(f"  Dist  : {cs['distance']:.2f}km | L2 Match: {status}")
            print("-" * 20)
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate HierGR-SeqRec Model")
    parser.add_argument("--config", default="./config/config.yaml", help="Path to config file")
    # 默认路径：请确保这里指向你最新的 checkpoint
    parser.add_argument("--sft_path", default="/workspace/data/llm_ckpt/checkpoint-28000", help="Path to SFT adapter")
    parser.add_argument("--grpo_path", default="/workspace/data/grpo_checkpoints/checkpoint-7000", help="Path to GRPO adapter")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--beams", type=int, default=10, help="Beam size for generation")
    
    args = parser.parse_args()
    
    print(f"Evaluation Configuration:")
    print(f"- SFT: {args.sft_path}")
    print(f"- GRPO: {args.grpo_path}")
    print(f"- Samples: {args.num_samples}")
    
    evaluator = RecEvaluator(args.config, args.sft_path, args.grpo_path)
    evaluator.evaluate(k_list=[1, 5, 10], num_beams=args.beams, limit=args.num_samples)

if __name__ == "__main__":
    main()