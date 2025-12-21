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

# 指向你的模型路径
ADAPTER_PATH = "/workspace/data/grpo_v4_3_logit_masking_full/checkpoint-1000"

TEST_DATA_PATH = "/workspace/data/processed/test_prompts.jsonl"
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"

# Trigger (Prompt 结尾)
SUFFIX = "Output the semantic ID in format <c0, c1, c2, c3>."
TRIGGER = f"\n{SUFFIX}\nResponse: <"

# ==============================================================================
# 2. Trie Logic (Strict)
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
        self.debug_printed = False # 只打印一次 Debug 信息
        
    def __call__(self, input_ids, scores):
        current_seq = input_ids[0, self.prompt_length:].tolist()
        valid_next = self.trie.get_valid_next_tokens(current_seq)
        
        # Debug: 看看第一步是不是空的
        if not self.debug_printed and len(current_seq) == 0:
            # print(f"   [Trie Debug] Start of Gen. Allowed Tokens Count: {len(valid_next) if valid_next else 0}")
            self.debug_printed = True

        mask = torch.ones_like(scores[0]) * float('-inf')
        
        if valid_next:
            mask[valid_next] = 0
            scores[0] = scores[0] + mask
        else:
            # 路径跑偏，锁死（防止输出乱码）
            # 强制输出 EOS 或 pad 防止报错，或者不做操作让它乱说然后被 eval 捕获
            pass
            
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
    
    print("Building Trie Strings (Including '<')...")
    for bid, meta in sid_map.items():
        if 'full_sid' not in meta: continue
        full_code = tuple(int(x) for x in meta['full_sid'])
        tree_map[full_code] = {'lat': meta['latitude'], 'lon': meta['longitude'], 'city': meta.get('city', 'Unknown')}
        
        # 【关键修改】带上 <，确保 Tokenizer 边界一致
        # 这会导致模型生成 double <<，但这很安全
        sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
        city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)

    print(f"Building Tries for {len(city_sid_strings)} cities...")
    city_tries = {}
    for city, strings in tqdm(city_sid_strings.items()):
        trie = Trie()
        # 这里使用 batch encode 提高速度
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
    # 增强解析，处理 <<12, 34... 的情况
    text = text.replace("Response:", "").replace("<", "").replace(">", "").strip()
    match = re.search(r"(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?", text)
    if match:
        return tuple(int(g) for g in match.groups() if g is not None)
    return None

def parse_target_sid(raw):
    if isinstance(raw, str):
        clean = raw.replace('<', '').replace('>', '').strip()
        try: return tuple(int(x.strip()) for x in clean.split(','))
        except: return None
    return tuple(raw) if raw else None

# ==============================================================================
# 4. Main
# ==============================================================================

def main():
    print(f"Loading Tokenizer: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 构建 Trie
    tree_map, city_tries = load_mapping_and_build_tries(MAPPING_FILE, tokenizer)
    
    print(f"Loading Model: {ADAPTER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    except:
        print("⚠️ Failed to load adapter.")
        return
    model.eval()

    print(f"Loading Data: {TEST_DATA_PATH}")
    test_samples = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            if line.strip(): test_samples.append(json.loads(line))
            
    print(f"Loaded {len(test_samples)} samples. Eval first 200...")
    test_samples = test_samples[:200]

    metrics = defaultdict(int)
    geo_dists = []

    print(">>> Starting Evaluation (Scheme 1 Final)...")
    
    for item in tqdm(test_samples):
        metrics["total"] += 1
        
        # Prompt
        raw_inst = item.get('instruction', '')
        base_prompt = raw_inst.split("Response:")[0].strip() if "Response:" in raw_inst else raw_inst.strip()
        full_prompt = f"{base_prompt}{TRIGGER}"
        
        # Target
        meta = item.get('metadata', {})
        t_sid = parse_target_sid(meta.get('target_sid'))
        
        # Oracle City Trie
        t_city = tree_map[t_sid]['city'] if (t_sid and t_sid in tree_map) else None
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        processor_list = []
        if t_city and t_city in city_tries:
            processor_list = [TrieConstraintLogitsProcessor(input_len, city_tries[t_city])]
        
        with torch.no_grad():
            # 贪婪解码更稳定
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False, 
                logits_processor=processor_list
            )
            
        generated_ids = outputs[:, input_len:]
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_id = parse_output(text)
        
        if pred_id:
            metrics["format_ok"] += 1
            if t_sid:
                if len(pred_id) >= 1 and pred_id[0] == t_sid[0]: metrics["l0_acc"] += 1
                if len(pred_id) >= 2 and pred_id[1] == t_sid[1]: metrics["l1_acc"] += 1
                if len(pred_id) >= 3 and pred_id[2] == t_sid[2]: metrics["l2_acc"] += 1
                if len(pred_id) >= 4 and pred_id[3] == t_sid[3]: metrics["l3_acc"] += 1
            
            if pred_id in tree_map and meta.get('target_lat'):
                p = (tree_map[pred_id]['lat'], tree_map[pred_id]['lon'])
                t = (meta['target_lat'], meta['target_lon'])
                geo_dists.append(haversine(t, p))
            else:
                geo_dists.append(100.0)
        else:
            geo_dists.append(100.0)

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
    if geo_dists: print(f"Avg Geo Dist   : {sum(geo_dists)/len(geo_dists):.4f} km")
    print("="*40)

if __name__ == "__main__":
    main()