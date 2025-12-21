import sys
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from peft import PeftModel

# --- 路径 Hack ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    print("⚠️ 警告: 未找到 models/pinrec_ultimate_v2.py")

# =========================================================
# ⚙️ 配置
# =========================================================
CONFIG = {
    "test_data": "/workspace/data/processed/train_prompts.jsonl", 
    "sid_mapping": "/workspace/data/processed/sid_mapping.json",
    "item_profiles": "/workspace/data/processed/item_profiles.jsonl", 
    "num_samples": 400, # 保持和之前一致以便对比
    "top_k_list": [1, 5, 10, 20],
    
    "hier": {
        "enabled": True,
        "base_model": "/workspace/Qwen2_5-1.5B-Instruct",
        "sft_ckpt": "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000",
        "grpo_ckpt": "/workspace/data/grpo_v4_1_breadcrumbs/checkpoint-5000",
        "device": "cuda",
        "beams": 5 # 稍微调小一点加快速度
    },
    
    "pinrec": {
        "enabled": True,
        "grpo_ckpt": "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-10000",
        "sft_ckpt": "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-48000",
        "device": "cuda"
    }
}

# =========================================================
# 🔧 ID 映射管理器 (核心修复)
# =========================================================
class IDManager:
    """全局 ID 管理，确保所有模型使用同一套 String->Int 映射"""
    def __init__(self, profile_path, mapping_path):
        print(f"🔧 [IDManager] Building Bridge from {profile_path}...")
        self.str_to_int = {}
        self.int_to_str = {}
        
        with open(profile_path, 'r') as f:
            for idx, line in enumerate(f):
                if not line.strip(): continue
                data = json.loads(line)
                bid = data.get('business_id') or data.get('id')
                if bid:
                    self.str_to_int[str(bid)] = idx
                    self.int_to_str[idx] = str(bid)
        print(f"✅ Loaded {len(self.str_to_int)} items.")
        
        # 加载 Semantic Mapping
        self.sid_to_int = {}
        self.tree_map = {}
        self.city_sid_strings = defaultdict(list)
        
        with open(mapping_path, 'r') as f:
            raw_map = json.load(f)
            
        for item_id_str, meta in raw_map.items():
            if item_id_str in self.str_to_int:
                int_id = self.str_to_int[item_id_str]
                full_code = tuple(int(x) for x in meta['full_sid'])
                
                self.sid_to_int[full_code] = int_id
                self.tree_map[full_code] = {'city': meta.get('city', 'Unknown')}
                
                sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
                self.city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)

# =========================================================
# 🥊 HierGR Wrapper
# =========================================================
# ... (Tool Classes Trie, LogitsProcessor omitted for brevity, logic same as before) ...
class Trie:
    def __init__(self): self.root = {}
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

class HierGRWrapper:
    def __init__(self, config, id_manager):
        self.device = config['device']
        self.beams = config['beams']
        self.id_mgr = id_manager
        
        # Load Model
        print(f"🥊 [HierGR] Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(config['base_model'], torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, config['sft_ckpt']).merge_and_unload()
        try:
            model = PeftModel.from_pretrained(model, config['grpo_ckpt'])
        except:
            model = PeftModel.from_pretrained(model, config['grpo_ckpt'], config_file="adapter_config.json")
        model.eval()
        self.model = model
        
        # Build Tries
        self.city_tries = {}
        print("Building Tries...")
        for city, strings in tqdm(id_manager.city_sid_strings.items()):
            trie = Trie()
            tokens_list = self.tokenizer.encode_batch(strings, add_special_tokens=False) if hasattr(self.tokenizer, 'encode_batch') else [self.tokenizer.encode(s, add_special_tokens=False) for s in strings]
            for tokens in tokens_list: trie.insert(tokens)
            self.city_tries[city] = trie

    def parse_output(self, text):
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: return tuple(int(g) for g in match.groups())
        return None

    def predict(self, batch_data):
        batch_recs = []
        for item in batch_data:
            t_city = None
            # Try to get city from target_sid string
            meta = item.get('metadata', {})
            if 'target_sid' in meta and isinstance(meta['target_sid'], str):
                 try:
                    t_sid = tuple(int(x.strip()) for x in meta['target_sid'].replace('<','').replace('>','').split(','))
                    if t_sid in self.id_mgr.tree_map: t_city = self.id_mgr.tree_map[t_sid]['city']
                 except: pass
            
            # Prompt
            raw = item.get('instruction', '')
            base = raw.split("Response:")[0].strip() if "Response:" in raw else raw.strip()
            prompt = f"{base}\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
            
            # Generate
            trie = self.city_tries.get(t_city)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            processor = [TrieConstraintLogitsProcessor(inputs.input_ids.shape[1], trie)] if trie else []
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=32, num_beams=self.beams, num_return_sequences=self.beams, logits_processor=processor, pad_token_id=self.tokenizer.eos_token_id, early_stopping=True)
            
            cands = []
            seen = set()
            for seq in out:
                txt = "<" + self.tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pid = self.parse_output(txt)
                if pid and pid in self.id_mgr.sid_to_int:
                    iid = self.id_mgr.sid_to_int[pid]
                    if iid not in seen: cands.append(iid); seen.add(iid)
            batch_recs.append(cands)
        return batch_recs

# =========================================================
# 🥊 PinRec Wrapper (修复版)
# =========================================================
class PinRecWrapper:
    def __init__(self, config, id_manager):
        self.device = config['device']
        self.id_mgr = id_manager
        print(f"🥊 [PinRec] Loading...")
        
        # Load Item Tower
        item_path = os.path.join(config['grpo_ckpt'], "item_tower.bin")
        if not os.path.exists(item_path): item_path = os.path.join(config['sft_ckpt'], "item_tower.bin")
        state_dict = torch.load(item_path, map_location='cpu')
        
        max_shape = 0
        for v in state_dict.values():
            if v.dim()==2 and v.shape[0]>max_shape: max_shape = v.shape[0]
        self.vocab_size = max_shape
        print(f"PinRec Vocab: {self.vocab_size}")

        p_conf = PinRecConfig()
        if hasattr(p_conf, 'item_vocab_size'): p_conf.item_vocab_size = max_shape
        if hasattr(p_conf, 'vocab_size'): p_conf.vocab_size = max_shape
        
        self.item_tower = ItemTower(p_conf).to(self.device)
        self.item_tower.load_state_dict(state_dict, strict=False)
        self.item_tower.eval()
        
        self.user_tower = UserTower(p_conf).to(self.device)
        self.user_tower.load_state_dict(torch.load(os.path.join(config['grpo_ckpt'], "user_tower.bin"), map_location=self.device), strict=False)
        self.user_tower.eval()
        
        # Build Index
        print("Building Index...")
        bs = 2048
        embs = []
        with torch.no_grad():
            for i in range(0, max_shape, bs):
                ids = torch.arange(i, min(i+bs, max_shape), device=self.device)
                embs.append(F.normalize(self.item_tower(ids), p=2, dim=-1))
        self.index = torch.cat(embs, dim=0)

    def predict(self, batch_data):
        valid_batch = []
        indices = []
        
        for idx, s in enumerate(batch_data):
            h_final = []
            meta = s.get('metadata', {})
            
            # [核心修复] 尝试使用 Business IDs (String) 进行重映射
            # 假设 metadata 里有 'history_business_ids' 或者 'history_ids' 是字符串
            # 我们优先检查是否能用 String ID 映射
            
            raw_h = None
            if 'history_business_ids' in meta: raw_h = meta['history_business_ids']
            elif 'history_ids' in meta: raw_h = meta['history_ids']
            elif 'history_ids' in s: raw_h = s['history_ids']
            
            if raw_h:
                # 尝试转换
                if isinstance(raw_h[0], str):
                    # 它是字符串，查 str_to_int 表
                    h_final = [self.id_mgr.str_to_int[bid] for bid in raw_h if bid in self.id_mgr.str_to_int]
                else:
                    # 它是整数，我们暂时信任它，但如果它是错的也没办法
                    # 除非我们能验证。这里假设如果它是整数，就直接用 _safe_id
                    h_final = [x % self.vocab_size for x in raw_h]
            
            if h_final:
                temp = s.copy()
                temp['_h_ids'] = h_final
                # Action & Delta
                L = len(h_final)
                temp['_h_acts'] = meta.get('history_acts', [1]*L)
                temp['_h_deltas'] = meta.get('history_deltas', [0.0]*L)
                temp['_t_act'] = meta.get('target_1', {}).get('act', 1)
                valid_batch.append(temp)
                indices.append(idx)
        
        if not valid_batch: return [[]]*len(batch_data)

        # Batch Inference
        B = len(valid_batch)
        max_len = max(len(x['_h_ids']) for x in valid_batch)
        h_ids = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        h_acts = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        h_deltas = torch.zeros((B, max_len), dtype=torch.float, device=self.device)
        h_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        t_acts = []
        
        for i, item in enumerate(valid_batch):
            L = len(item['_h_ids'])
            h_ids[i, :L] = torch.tensor(item['_h_ids'], device=self.device)
            h_acts[i, :L] = torch.tensor(item['_h_acts'][:L], device=self.device)
            h_deltas[i, :L] = torch.tensor(item['_h_deltas'][:L], device=self.device)
            h_mask[i, :L] = 1
            t_acts.append(item['_t_act'])
            
        with torch.no_grad():
            h_vec = self.item_tower(h_ids.view(-1)).view(B, max_len, -1)
            t_in = torch.tensor(t_acts, device=self.device).unsqueeze(1)
            t_in = torch.cat([t_in, t_in], dim=1)
            d_in = torch.zeros((B, 2), device=self.device)
            
            u_vec = F.normalize(self.user_tower(h_vec, h_acts, h_deltas, h_mask, t_in, d_in)[:,0,:], p=2, dim=-1)
            scores = torch.matmul(u_vec, self.index.T)
            topk = torch.topk(scores, k=max(CONFIG['top_k_list']), dim=1).indices.cpu().tolist()
            
        res = [[]]*len(batch_data)
        for i, real_idx in enumerate(indices): res[real_idx] = topk[i]
        return res

# =========================================================
# 📊 Main Evaluation
# =========================================================
def evaluate_all():
    # 1. Global ID Manager
    id_mgr = IDManager(CONFIG['item_profiles'], CONFIG['sid_mapping'])
    
    # 2. Models
    models = {}
    if CONFIG['hier']['enabled']: models['HierGR'] = HierGRWrapper(CONFIG['hier'], id_mgr)
    if CONFIG['pinrec']['enabled']: models['PinRec'] = PinRecWrapper(CONFIG['pinrec'], id_mgr) # Pass id_mgr
    
    # 3. Data
    print(f"Loading data: {CONFIG['test_data']}")
    samples = []
    with open(CONFIG['test_data']) as f:
        for i, line in enumerate(f):
            if CONFIG['num_samples'] and i >= CONFIG['num_samples']: break
            if line.strip():
                d = json.loads(line)
                if d.get('task') == 'task_a_recommendation': samples.append(d)
    print(f"Loaded {len(samples)}.")

    # 4. Loop
    results = {m: defaultdict(int) for m in models}
    BATCH = 32
    
    for i in tqdm(range(0, len(samples), BATCH)):
        batch = samples[i:i+BATCH]
        
        # Ground Truth Extraction (Using ID Manager for reliability)
        targets = []
        for s in batch:
            truth = -1
            meta = s.get('metadata', {})
            
            # Try getting Business ID String first (Most reliable)
            if 'target_business_id' in meta and meta['target_business_id'] in id_mgr.str_to_int:
                truth = id_mgr.str_to_int[meta['target_business_id']]
            # Try target_1.id (Integer)
            elif 'target_1' in meta and 'id' in meta['target_1']:
                truth = meta['target_1']['id'] # Trusting integer if exists
            # Try semantic ID string
            elif 'target_sid' in meta and isinstance(meta['target_sid'], str):
                try:
                    t_sid = tuple(int(x.strip()) for x in meta['target_sid'].replace('<','').replace('>','').split(','))
                    if t_sid in id_mgr.sid_to_int: truth = id_mgr.sid_to_int[t_sid]
                except: pass
            
            targets.append(truth)
            
        for name, model in models.items():
            preds = model.predict(batch)
            for j, p_list in enumerate(preds):
                t = targets[j]
                if t == -1: continue
                for k in CONFIG['top_k_list']:
                    if t in p_list[:k]:
                        results[name][f'Hit@{k}'] += 1
                        rank = p_list[:k].index(t)
                        results[name][f'NDCG@{k}'] += 1.0/np.log2(rank+2)

    # 5. Report
    print("\n" + "="*60)
    print(f"🏆 Final Comparison (N={len(samples)})")
    print("="*60)
    data = []
    for name in models:
        row = {'Model': name}
        for k in CONFIG['top_k_list']:
            row[f'H@{k}'] = f"{results[name][f'Hit@{k}']/len(samples):.2%}"
            row[f'N@{k}'] = f"{results[name][f'NDCG@{k}']/len(samples):.4f}"
        data.append(row)
    print(pd.DataFrame(data).to_string(index=False))

if __name__ == '__main__':
    evaluate_all()