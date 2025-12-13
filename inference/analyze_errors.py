"""
Deep Error Analysis for GRPO Model
"""
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

# 复用 Trie 逻辑 (省略部分代码，保持与之前一致)
class Trie:
    def __init__(self): self.root = {}
    def insert(self, s):
        n = self.root
        for t in s: n = n.setdefault(t, {})
        n[-1] = True
    def get_next(self, p):
        n = self.root
        for t in p:
            if t not in n: return None
            n = n[t]
        return [k for k in n.keys() if k != -1]

class TrieConstraint(LogitsProcessor):
    def __init__(self, start_len, trie):
        self.start_len = start_len
        self.trie = trie
    def __call__(self, input_ids, scores):
        for i in range(len(input_ids)):
            p = input_ids[i, self.start_len:].tolist()
            allow = self.trie.get_next(p)
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if allow:
                mask[allow] = False
                scores[i] = scores[i].masked_fill(mask, -float('inf'))
        return scores

def load_resources(base_path, sft_path, grpo_path, map_file):
    print("Loading resources...")
    with open(map_file) as f: sid_map = json.load(f)
    
    # 建立反向索引：ID -> Name/Category
    id_to_meta = {}
    city_tries = {}
    
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    
    for meta in tqdm(sid_map.values(), desc="Building Index"):
        sid = tuple(int(x) for x in meta['full_sid'])
        id_to_meta[sid] = {
            'name': meta.get('name', 'Unknown'),
            'city': meta.get('city', 'Unknown'),
            'categories': meta.get('categories', [])
        }
        
        # Build Trie
        city = meta.get('city', 'Unknown')
        if city not in city_tries: city_tries[city] = Trie()
        # Tokenize sid string
        sid_str = f"<{sid[0]}, {sid[1]}, {sid[2]}, {sid[3]}>"
        tokens = tokenizer.encode(sid_str, add_special_tokens=False)
        city_tries[city].insert(tokens)

    # Load Model
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()
    try:
        model = PeftModel.from_pretrained(model, grpo_path)
    except:
        # Fallback to checkpoint path
        model = PeftModel.from_pretrained(model, os.path.join(grpo_path, "adapter_config.json"))
        
    model.eval()
    return model, tokenizer, id_to_meta, city_tries

def analyze(model, tokenizer, id_to_meta, city_tries, test_file, num_samples=50):
    data = []
    with open(test_file) as f:
        for l in f: data.append(json.loads(l))
    
    # 只取 Task A
    data = [d for d in data if d.get('task') == 'task_a_recommendation'][-num_samples:]
    
    print("\n" + "="*50)
    print(" CASE STUDY ANALYSIS ")
    print("="*50)
    
    for i, item in enumerate(data):
        target_raw = item['metadata']['target_sid']
        if isinstance(target_raw, str):
            target_sid = tuple(int(x.strip()) for x in target_raw.replace('<','').replace('>','').split(','))
        else: target_sid = tuple(target_raw)
        
        target_meta = id_to_meta.get(target_sid, {})
        city = target_meta.get('city', 'Unknown')
        
        # Prompt
        raw_inst = item.get('instruction', '').split("Response:")[0].strip()
        prompt = f"{raw_inst}\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        trie = city_tries.get(city)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=32, num_beams=5, 
                logits_processor=[TrieConstraint(inputs.input_ids.shape[1], trie)] if trie else [],
                pad_token_id=tokenizer.eos_token_id
            )
            
        pred_text = "<" + tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Parse
        import re
        m = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", pred_text)
        if not m:
            print(f"[{i}] Parse Error: {pred_text}")
            continue
            
        pred_sid = tuple(int(g) for g in m.groups())
        pred_info = id_to_meta.get(pred_sid, {})
        
        # Analyze Match
        l1_match = (pred_sid[:2] == target_sid[:2])
        l2_match = (pred_sid[:3] == target_sid[:3])
        
        print(f"[{i+1}] Target: {target_meta.get('name')} | Cat: {target_meta.get('categories')}")
        print(f"    Pred  : {pred_info.get('name')} | Cat: {pred_info.get('categories')}")
        print(f"    ID Trg: {target_sid}")
        print(f"    ID Prd: {pred_sid}")
        
        status = "❌ Fail"
        if l2_match: status = "✅ Category Match!"
        elif l1_match: status = "⚠️ District Match Only"
        
        print(f"    Result: {status}")
        print("-" * 30)

if __name__ == "__main__":
    BASE = "/workspace/Qwen2_5-1.5B-Instruct"
    SFT = "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000"
    GRPO = "/workspace/data/grpo_v4_1_breadcrumbs/checkpoint-4800"
    MAP = "/workspace/data/processed/sid_mapping.json"
    TEST = "/workspace/data/processed/train_prompts.jsonl"
    
    m, t, im, ct = load_resources(BASE, SFT, GRPO, MAP)
    analyze(m, t, im, ct, TEST, num_samples=10) # 看10个例子