import os
import json
import re
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置 =================
MAPPING_FILE = "/workspace/data/processed/sid_mapping.json"
TEST_DATA_PATH = "/workspace/data/processed/test_prompts.jsonl"
# =======================================

def parse_target_sid(raw):
    # 你的解析逻辑
    if isinstance(raw, str):
        clean = raw.replace('<', '').replace('>', '').strip()
        try:
            return tuple(int(x.strip()) for x in clean.split(','))
        except:
            return None
    return None

def main():
    print("🕵️‍♂️ Starting Debug Investigation...")
    
    # 1. 加载 Mapping
    print(f"1. Loading Mapping from {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'r') as f:
        sid_map = json.load(f)
    
    tree_map = {}
    sample_key = None
    for bid, meta in sid_map.items():
        if 'full_sid' in meta:
            # 确保转换成 tuple(int)
            full_code = tuple(int(x) for x in meta['full_sid'])
            tree_map[full_code] = meta.get('city', 'Unknown')
            if sample_key is None: sample_key = full_code
            
    print(f"   ✅ Loaded {len(tree_map)} unique IDs in Mapping.")
    print(f"   🔑 Sample Key in Mapping (Python Tuple): {sample_key}")
    print(f"   🔑 Type of Key: {type(sample_key[0])} inside {type(sample_key)}")

    # 2. 检查测试数据
    print(f"\n2. Inspecting Test Data from {TEST_DATA_PATH}...")
    hits = 0
    misses = 0
    
    with open(TEST_DATA_PATH, 'r') as f:
        # 只检查前 10 条
        for i, line in enumerate(f):
            if i >= 10: break
            
            data = json.loads(line)
            raw_target = data['metadata'].get('target_sid')
            parsed_target = parse_target_sid(raw_target)
            
            print(f"\n[Sample {i}]")
            print(f"   📄 Raw Target in JSON: {repr(raw_target)}")
            print(f"   ⚙️ Parsed Target:      {parsed_target}")
            
            if parsed_target in tree_map:
                city = tree_map[parsed_target]
                print(f"   ✅ FOUND in Mapping! City: {city}")
                hits += 1
            else:
                print(f"   ❌ NOT FOUND in Mapping!")
                # 尝试做一下模糊匹配诊断
                print(f"   🔎 Diagnostics:")
                print(f"      - Is it a tuple? {isinstance(parsed_target, tuple)}")
                if parsed_target:
                    print(f"      - Element types: {[type(x) for x in parsed_target]}")
                misses += 1

    print("\n" + "="*30)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*30)
    if hits == 0:
        print("🚨 CRITICAL FAILURE: No Target IDs matched the Mapping!")
        print("   This explains why Trie Constraint was never activated.")
        print("   -> Possible fix: Check if Mapping file is outdated or Test Data uses different IDs.")
    else:
        print(f"✅ Success Rate: {hits}/{hits+misses}")
        print("   If this is high, then the bug is in the Trie Construction or Tokenizer.")

if __name__ == "__main__":
    main()