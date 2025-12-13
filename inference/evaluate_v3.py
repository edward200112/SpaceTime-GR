"""
Evaluate GRPO V3 Model
[SPECIALIZED FOR PRE-FILLING STRATEGY]
"""

import os
import sys
import json
import torch
import numpy as np
import re
import argparse
from tqdm import tqdm
from haversine import haversine
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. Helper Functions
# ==========================================

def parse_model_output(output_text):
    """
    解析 V3 模型的输出。
    因为输入已经是 "Response: <"，模型只会输出 "12, 34, 56, 7>" 或 "12, 34, 56, 7>..."
    """
    # 匹配开头是 "数字, 数字, 数字, 数字"
    # 允许中间有空格
    match = re.search(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", output_text.strip())
    if match:
        return tuple(int(g) for g in match.groups())
    return None

def load_mapping(mapping_file):
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tree_map = {}
    for bid, meta in data.items():
        full_code = tuple(int(x) for x in meta['full_sid'])
        tree_map[full_code] = {
            'lat': meta['latitude'],
            'lon': meta['longitude'],
            'city': meta['city'],
            'name': meta['name'],
            'categories': meta.get('categories', [])
        }
    print(f"Loaded {len(tree_map)} valid items.")
    return tree_map

# ==========================================
# 2. Evaluator Class
# ==========================================

class GRPOEvaluator:
    def __init__(self, base_model_path, sft_path, grpo_path, mapping_path, device="cuda"):
        self.device = device
        self.tree_map = load_mapping(mapping_path)
        
        # --- Load Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        
        # --- Load Model (Merge SFT + Load GRPO) ---
        print(f"Loading Base: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"Merging SFT: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
        
        print(f"Loading GRPO: {grpo_path}")
        model = PeftModel.from_pretrained(model, grpo_path)
        model.eval()
        self.model = model

    def evaluate(self, test_file, num_samples=500, beams=10):
        print(f"Loading test data from {test_file}")
        data = []
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        
        # 只取 Task A
        data = [d for d in data if d.get('task', 'task_a_recommendation') == 'task_a_recommendation']
        
        # 取最后 num_samples 个样本 (模拟测试集)
        test_data = data[-num_samples:]
        print(f"Evaluating {len(test_data)} samples...")

        # Metrics
        metrics = {'hit1': 0, 'hit5': 0, 'hit10': 0}
        layer_hits = {0: 0, 1: 0, 2: 0, 3: 0}
        distances = []
        
        # Case Study buffer
        cases = []

        for item in tqdm(test_data):
            # 1. Get Ground Truth
            meta = item.get('metadata', {})
            target_raw = meta.get('target_sid')
            target_lat = meta.get('target_lat')
            target_lon = meta.get('target_lon')
            
            # 解析 Target ID
            if isinstance(target_raw, str):
                clean = target_raw.replace('<', '').replace('>', '')
                target_sid = tuple(int(x.strip()) for x in clean.split(','))
            else:
                target_sid = tuple(target_raw)
            
            # 2. Construct Prompt (Pre-filling Strategy)
            raw_prompt = item.get('instruction', '')
            if "Response:" in raw_prompt:
                base_prompt = raw_prompt.split("Response:")[0].strip()
            else:
                base_prompt = raw_prompt.strip()
            
            # 必须和训练时的一样！
            instruction_suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
            final_prompt = f"{base_prompt}\n{instruction_suffix}\nResponse: <"
            
            # 3. Generate
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=24,      # 只需要生成数字，很短
                    num_beams=beams,        # Beam Search 提高准确率
                    num_return_sequences=beams,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # 4. Parse Candidates
            candidates = []
            seen = set()
            
            for output_seq in outputs:
                # 只解码新生成的部分
                new_tokens = output_seq[inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # 解析
                pid = parse_model_output(text)
                
                # 过滤：必须解析成功，且必须在 Mapping 里 (Valid ID)
                if pid and pid in self.tree_map and pid not in seen:
                    candidates.append(pid)
                    seen.add(pid)
            
            # 如果没有生成任何有效 ID，跳过
            if not candidates:
                continue
            
            # 5. Calculate Metrics
            # Hit@K
            if target_sid in candidates[:1]: metrics['hit1'] += 1
            if target_sid in candidates[:5]: metrics['hit5'] += 1
            if target_sid in candidates[:10]: metrics['hit10'] += 1
            
            # Top-1 Analysis
            top1 = candidates[0]
            
            # Distance
            pred_info = self.tree_map[top1]
            dist = haversine((target_lat, target_lon), (pred_info['lat'], pred_info['lon']))
            distances.append(dist)
            
            # Layer Accuracy
            if top1[0] == target_sid[0]:
                layer_hits[0] += 1
                if top1[1] == target_sid[1]:
                    layer_hits[1] += 1
                    if top1[2] == target_sid[2]:
                        layer_hits[2] += 1
                        if top1[3] == target_sid[3]:
                            layer_hits[3] += 1

            # Save Case
            if len(cases) < 5:
                cases.append({
                    "target": target_sid,
                    "target_city": self.tree_map[target_sid]['city'],
                    "pred": top1,
                    "pred_city": pred_info['city'],
                    "distance": dist,
                    "layer2_match": (top1[2] == target_sid[2])
                })

        # 6. Report
        n = len(distances)
        if n == 0:
            print("No valid predictions found.")
            return

        print("\n" + "="*40)
        print(f"RESULTS (N={n})")
        print("="*40)
        print(f"Mean Distance Error: {np.mean(distances):.4f} km")
        print("-" * 20)
        print(f"Hit@1 : {metrics['hit1']/n:.2%}")
        print(f"Hit@5 : {metrics['hit5']/n:.2%}")
        print(f"Hit@10: {metrics['hit10']/n:.2%}")
        print("-" * 20)
        print("Hierarchical Accuracy (Top-1):")
        print(f"Layer 0 (City)    : {layer_hits[0]/n:.2%}")
        print(f"Layer 1 (District): {layer_hits[1]/n:.2%}")
        print(f"Layer 2 (Category): {layer_hits[2]/n:.2%} <--- V3 KEY METRIC")
        print(f"Layer 3 (Item)    : {layer_hits[3]/n:.2%}")
        print("="*40)
        
        print("\n[Case Studies]")
        for c in cases:
            status = "✅" if c['layer2_match'] else "❌"
            print(f"T: {c['target']} ({c['target_city']})")
            print(f"P: {c['pred']} ({c['pred_city']})")
            print(f"Dist: {c['distance']:.2f}km | L2: {status}")
            print("-" * 20)

if __name__ == "__main__":
    # 配置你的路径
    BASE = "/workspace/Qwen2_5-1.5B-Instruct"
    # 注意：这里 SFT 要用你刚才重训过的那个
    SFT = "/workspace/data/llm_ckpt/checkpoint-28000" 
    # GRPO 用正在跑的 V3 checkpoint
    GRPO = "/workspace/data/grpo_v3_gated/checkpoint-500" # <-- 改为你实际保存的 step
    MAP = "/workspace/data/processed/sid_mapping.json"
    TEST = "/workspace/data/processed/train_prompts.jsonl" # 暂时用 train 测试，或者换 test_prompts
    
    # 自动寻找最新的 GRPO checkpoint
    if not os.path.exists(GRPO):
        checkpoints = [d for d in os.listdir("/workspace/data/grpo_v3_gated") if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            GRPO = os.path.join("/workspace/data/grpo_v3_gated", checkpoints[-1])
            print(f"Auto-detected latest GRPO: {GRPO}")
    
    evaluator = GRPOEvaluator(BASE, SFT, GRPO, MAP)
    evaluator.evaluate(TEST, num_samples=500)