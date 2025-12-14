import os
import json
import re
import torch
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

# ==============================================================================
# 1. Configuration
# ==============================================================================

BASE_MODEL_PATH = "/workspace/Qwen2_5-1.5B-Instruct"

# 指向你的模型路径 (v4.1 breadcrumbs 或 v4.3 masking)
ADAPTER_PATH = "/workspace/data/grpo_v4_1_breadcrumbs/checkpoint-4800" 

TEST_DATA_PATH = "/workspace/data/processed/test_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# Trigger必须匹配
SUFFIX = "Output the semantic ID in the format <c0, c1, c2, suffix>."
TRIGGER = f"\n{SUFFIX}\nResponse: <"

# ==============================================================================
# 2. Trie Constraint Logic
# ==============================================================================

class Trie:
    def __init__(self):
        self.root = {}
    
    def insert(self, sequence):
        node = self.root
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        node[-1] = True 

    def get_valid_next_tokens(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node:
                return None 
            node = node[token]
        return [k for k in node.keys() if k != -1]

class TrieConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length, trie):
        self.prompt_length = prompt_length
        self.trie = trie
        
    def __call__(self, input_ids, scores):
        current_seq = input_ids[0, self.prompt_length:].tolist()
        valid_next = self.trie.get_valid_next_tokens(current_seq)
        
        mask = torch.ones_like(scores[0]) * float('-inf')
        if valid_next:
            mask[valid_next] = 0
            scores[0] = scores[0] + mask
        return scores

# ==============================================================================
# 3. Helpers
# ==============================================================================

def load_mapping_and_build_tries(mapping_file, tokenizer):
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        sid_map = json.load(f)
    
    tree_map = {}
    city_sid_strings = defaultdict(list)
    
    print("Building Metadata & Trie Strings...")
    for bid, meta in sid_map.items():
        if 'full_sid' not in meta: continue
        full_code = tuple(int(x) for x in meta['full_sid'])
        
        tree_map[full_code] = {
            'lat': meta['latitude'], 
            'lon': meta['longitude'],
            'city': meta.get('city', 'Unknown')
        }
        
        # 构建 Trie 字符串: "12, 34, 56, 78>"
        # 注意：这里我们假设 Prompt 结尾是 Response: < 
        # 所以 Trie 内容不需要包含 <
        sid_str = f"{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
        city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)

    print(f"Building Tries for {len(city_sid_strings)} cities...")
    city_tries = {}
    for city, strings in tqdm(city_sid_strings.items()):
        trie = Trie()
        encoded_batch = tokenizer(strings, add_special_tokens=False).input_ids
        for tokens in encoded_batch:
            trie.insert(tokens)
        city_tries[city] = trie
        
    return tree_map, city_tries

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_output(text):
    text = text.replace("Response:", "").replace("<", "").replace(">", "").strip()
    match = re.search(r"(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?", text)
    if match:
        return tuple(int(g) for g in match.groups() if g is not None)
    return None

def parse_target_sid(raw):
    """
    【修复核心】正确解析 target_sid 字符串
    """
    if isinstance(raw, list) or isinstance(raw, tuple):
        return tuple(int(x) for x in raw)
    if isinstance(raw, str):
        # 去掉 < > 并按逗号分割
        clean = raw.replace('<', '').replace('>', '').strip()
        if not clean: return None
        return tuple(int(x.strip()) for x in clean.split(','))
    return None

# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    # 1. Init
    print(f"Loading Tokenizer: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Build Tries
    tree_map, city_tries = load_mapping_and_build_tries(MAPPING_FILE, tokenizer)
    
    # 3. Model
    print(f"Loading Model: {ADAPTER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    except:
        print("⚠️ Failed to load adapter. Using Base Model only.")
    model.eval()

    # 4. Data
    print(f"Loading Test Data: {TEST_DATA_PATH}")
    test_samples = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            if line.strip(): test_samples.append(json.loads(line))
            
    print(f"Loaded {len(test_samples)} samples. Eval first 200...")
    test_samples = test_samples[:200]

    metrics = defaultdict(int)
    geo_dists = []

    print(">>> Starting Evaluation with Trie...")
    
    # Batch Size = 1 for Trie Constraint
    for item in tqdm(test_samples):
        metrics["total"] += 1
        
        # Prepare Prompt
        raw_inst = item.get('instruction', '')
        base_prompt = raw_inst.split("Response:")[0].strip() if "Response:" in raw_inst else raw_inst.strip()
        full_prompt = f"{base_prompt}{TRIGGER}"
        
        # 【修复】正确解析 Target SID
        meta = item.get('metadata', {})
        t_raw = meta.get('target_sid')
        t_sid = parse_target_sid(t_raw) # 使用修复后的解析函数
        
        # Oracle City Lookup
        t_city = None
        if t_sid and t_sid in tree_map:
            t_city = tree_map[t_sid]['city']
        
        # Prepare Inputs
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        # Setup Constraint
        processor_list = []
        if t_city and t_city in city_tries:
            trie = city_tries[t_city]
            processor_list = [TrieConstraintLogitsProcessor(input_len, trie)]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                do_sample=False,
                logits_processor=processor_list
            )
            
        # Parse
        generated_ids = outputs[:, input_len:]
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_id = parse_output(text)
        
        if pred_id:
            metrics["format_ok"] += 1
            if t_sid:
                if len(pred_id) >= 1 and pred_id[0] == t_sid[0]:
                    metrics["l0_acc"] += 1
                    if len(pred_id) >= 2 and pred_id[1] == t_sid[1]:
                        metrics["l1_acc"] += 1
                        if len(pred_id) >= 3 and pred_id[2] == t_sid[2]:
                            metrics["l2_acc"] += 1
                            if len(pred_id) >= 4 and pred_id[3] == t_sid[3]:
                                metrics["l3_acc"] += 1
            
            if pred_id in tree_map and meta.get('target_lat'):
                p_coord = (tree_map[pred_id]['lat'], tree_map[pred_id]['lon'])
                t_coord = (meta['target_lat'], meta['target_lon'])
                geo_dists.append(haversine(t_coord, p_coord))
            else:
                geo_dists.append(100.0)
        else:
            geo_dists.append(100.0)

    # Report
    total = metrics["total"]
    print("\n" + "="*40)
    print(f"📊 Results: {os.path.basename(ADAPTER_PATH)}")
    print("="*40)
    print(f"Format Validity: {metrics['format_ok'] / total:.2%}")
    print(f"L0 (Region)    : {metrics['l0_acc'] / total:.2%}")
    print(f"L1 (District)  : {metrics['l1_acc'] / total:.2%}")
    print(f"L2 (Category)  : {metrics['l2_acc'] / total:.2%}")
    print(f"L3 (Exact)     : {metrics['l3_acc'] / total:.2%}")
    print("-" * 20)
    if geo_dists:
        print(f"Avg Geo Dist   : {sum(geo_dists)/len(geo_dists):.4f} km")
    print("="*40)

if __name__ == "__main__":
    main()