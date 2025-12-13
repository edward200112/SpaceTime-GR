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

# --- 尝试导入 PinRec 模型定义 ---
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    print("⚠️ 警告: 未找到 models/pinrec_ultimate_v2.py，请确保路径正确")

# =========================================================
# ⚙️ 全局配置
# =========================================================
CONFIG = {
    # 1. 数据集路径
    "test_data": "/workspace/data/processed/train_prompts.jsonl", 
    "sid_mapping": "/workspace/data/processed/sid_mapping.json",
    
    # [关键修复] 必须提供 item_profiles 来建立 String ID -> Int ID 的映射
    "item_profiles": "/workspace/data/processed/item_profiles.jsonl", 
    
    "num_samples": 500,       # 测试样本数 (设为 None 跑全量)
    "top_k_list": [1, 5, 10, 20],
    
    # 🥊 Model A: HierGR (生成式)
    "hier": {
        "enabled": True,
        "base_model": "/workspace/Qwen2_5-1.5B-Instruct",
        "sft_ckpt": "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000",
        "grpo_ckpt": "/workspace/data/grpo_v4_1_breadcrumbs/checkpoint-5000",
        "device": "cuda",
        "beams": 10
    },
    
    # 🥊 Model B: PinRec (判别式)
    "pinrec": {
        "enabled": True,
        "grpo_ckpt": "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-10000",
        "sft_ckpt": "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-48000",
        "device": "cuda"
    }
}

# =========================================================
# 🔧 工具类: Trie & LogitsProcessor (HierGR 专用)
# =========================================================
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

# =========================================================
# 🥊 选手 1: HierGR Wrapper
# =========================================================
class HierGRWrapper:
    def __init__(self, config, sid_map_path, item_profile_path):
        self.device = config['device']
        self.beams = config['beams']
        print(f"\n🥊 [HierGR] 初始化中...")
        
        # 1. 建立 ID 桥梁 (String ID -> Integer ID)
        print(f"Loading item profiles from {item_profile_path} to build ID bridge...")
        self.str_to_int = {}
        try:
            with open(item_profile_path, 'r') as f:
                for idx, line in enumerate(f):
                    if not line.strip(): continue
                    data = json.loads(line)
                    # 兼容不同字段名: business_id 或 id
                    bid = data.get('business_id') or data.get('id')
                    if bid:
                        self.str_to_int[str(bid)] = idx
            print(f"✅ Bridged {len(self.str_to_int)} items (String -> Int).")
        except FileNotFoundError:
            print(f"❌ Error: {item_profile_path} not found. Cannot map String IDs to Integers!")
            raise

        # 2. 加载 Semantic Mapping 并建立 Tuple -> Int 反向索引
        print(f"Loading SID mapping from {sid_map_path}...")
        with open(sid_map_path, 'r') as f:
            raw_map = json.load(f)
            
        self.sid_to_int = {} # (1,2,3,4) -> 10086 (Int)
        self.tree_map = {}   # (1,2,3,4) -> {'city': ...}
        self.city_sid_strings = defaultdict(list)
        
        match_count = 0
        for item_id_str, meta in raw_map.items():
            # [关键修复] 使用 str_to_int 进行转换，而不是 int()
            if item_id_str in self.str_to_int:
                int_id = self.str_to_int[item_id_str]
                full_code = tuple(int(x) for x in meta['full_sid'])
                
                self.sid_to_int[full_code] = int_id
                self.tree_map[full_code] = {'city': meta.get('city', 'Unknown')}
                
                sid_str = f"<{full_code[0]}, {full_code[1]}, {full_code[2]}, {full_code[3]}>"
                self.city_sid_strings[meta.get('city', 'Unknown')].append(sid_str)
                match_count += 1
            
        print(f"✅ Mapped {match_count} semantic IDs to integer IDs.")

        # 3. 加载模型
        print(f"Loading Tokenizer: {config['base_model']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        
        print(f"Loading Model & Merging Adapters...")
        model = AutoModelForCausalLM.from_pretrained(
            config['base_model'], torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, config['sft_ckpt'])
        model = model.merge_and_unload()
        
        try:
            model = PeftModel.from_pretrained(model, config['grpo_ckpt'])
        except Exception as e:
            print(f"Standard load failed, trying adapter_config fallback: {e}")
            adapter_config_path = os.path.join(config['grpo_ckpt'], "adapter_config.json")
            if os.path.exists(adapter_config_path):
                 model = PeftModel.from_pretrained(model, config['grpo_ckpt'])
            else:
                 raise e
        
        model.eval()
        self.model = model
        
        # 4. 构建 Trie
        self.build_city_tries()

    def build_city_tries(self):
        print("Building Tries for constrained generation...")
        self.city_tries = {}
        for city, strings in tqdm(self.city_sid_strings.items(), desc="Cities"):
            trie = Trie()
            if hasattr(self.tokenizer, 'encode_batch'):
                tokens_list = self.tokenizer.encode_batch(strings, add_special_tokens=False)
            else:
                tokens_list = [self.tokenizer.encode(s, add_special_tokens=False) for s in strings]
            for tokens in tokens_list: trie.insert(tokens)
            self.city_tries[city] = trie

    def parse_output(self, text):
        match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)>", text)
        if match: return tuple(int(g) for g in match.groups())
        return None

    def predict(self, batch_data):
        batch_recs = []
        for item in batch_data:
            # 1. 确定 Trie 约束 (City)
            t_city = None
            meta = item.get('metadata', {})
            
            if 'target_sid' in meta:
                t_raw = meta['target_sid']
                if isinstance(t_raw, str):
                     cleaned = t_raw.replace('<','').replace('>','').strip()
                     if cleaned:
                        try:
                            t_sid = tuple(int(x.strip()) for x in cleaned.split(','))
                            if t_sid in self.tree_map:
                                t_city = self.tree_map[t_sid]['city']
                        except: pass
            
            # 2. 构造 Prompt
            raw_inst = item.get('instruction', '')
            base_prompt = raw_inst.split("Response:")[0].strip() if "Response:" in raw_inst else raw_inst.strip()
            final_prompt = f"{base_prompt}\nOutput the semantic ID in the format <c0, c1, c2, suffix>.\nResponse: <"
            
            # 3. 准备 Trie
            trie = self.city_tries.get(t_city)
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            logits_processor = [TrieConstraintLogitsProcessor(prompt_len, trie)] if trie else []
            
            # 4. 生成 (Beam Search)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=32, num_beams=self.beams, num_return_sequences=self.beams,
                    logits_processor=logits_processor, pad_token_id=self.tokenizer.eos_token_id, early_stopping=True
                )
            
            # 5. 解析并转换为 Integer ID
            candidates = []
            seen = set()
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                text = "<" + self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pid_tuple = self.parse_output(text)
                
                if pid_tuple and pid_tuple in self.sid_to_int:
                    item_id = self.sid_to_int[pid_tuple]
                    if item_id not in seen:
                        candidates.append(item_id)
                        seen.add(item_id)
            
            batch_recs.append(candidates)
            
        return batch_recs

# =========================================================
# 🥊 选手 2: PinRec Wrapper
# =========================================================
class PinRecWrapper:
    def __init__(self, config):
        self.device = config['device']
        print(f"\n🥊 [PinRec] 初始化中...")
        
        # 1. 智能加载 Item Tower
        item_path = os.path.join(config['grpo_ckpt'], "item_tower.bin")
        if not os.path.exists(item_path):
            print(f"Item tower missing in GRPO, fallback to SFT: {config['sft_ckpt']}")
            item_path = os.path.join(config['sft_ckpt'], "item_tower.bin")
            
        state_dict = torch.load(item_path, map_location='cpu')
        
        # 探测物理权重
        max_shape = 0
        for k, v in state_dict.items():
            if v.dim() == 2 and v.shape[0] > max_shape:
                max_shape = v.shape[0]
        self.physical_vocab_size = max_shape
        print(f"PinRec Physical Vocab Size: {self.physical_vocab_size}")

        self.model_config = PinRecConfig()
        if hasattr(self.model_config, 'item_vocab_size'): self.model_config.item_vocab_size = self.physical_vocab_size
        if hasattr(self.model_config, 'vocab_size'): self.model_config.vocab_size = self.physical_vocab_size
        
        # Load Item Tower
        self.item_tower = ItemTower(self.model_config).to(self.device)
        self.item_tower.load_state_dict(state_dict, strict=False)
        self.item_tower.eval()
        
        # Load User Tower (GRPO)
        self.user_tower = UserTower(self.model_config).to(self.device)
        user_path = os.path.join(config['grpo_ckpt'], "user_tower.bin")
        self.user_tower.load_state_dict(torch.load(user_path, map_location=self.device), strict=False)
        self.user_tower.eval()
        
        self._build_index()

    def _safe_id(self, input_id):
        if input_id >= self.physical_vocab_size:
            return input_id % self.physical_vocab_size
        return input_id

    def _build_index(self):
        print(f"Building PinRec Index...")
        batch_size = 2048
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, self.physical_vocab_size, batch_size), desc="Indexing"):
                end = min(i + batch_size, self.physical_vocab_size)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                embs = self.item_tower(ids)
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs)
        self.item_index = torch.cat(all_embs, dim=0)

    def predict(self, batch_data):
        valid_indices = []
        valid_batch = []
        
        # 兼容性处理: 同时支持 history_ids 或 metadata.history_ids
        for idx, s in enumerate(batch_data):
            h_ids_list = None
            if 'history_ids' in s:
                h_ids_list = s['history_ids']
            elif 'metadata' in s and 'history_ids' in s['metadata']:
                h_ids_list = s['metadata']['history_ids']
            
            if h_ids_list:
                valid_indices.append(idx)
                temp = s.copy()
                temp['_h_ids'] = h_ids_list
                # 默认值处理
                meta = s.get('metadata', {})
                temp['_h_acts'] = s.get('history_acts', meta.get('history_acts', [1]*len(h_ids_list)))
                temp['_h_deltas'] = s.get('history_deltas', meta.get('history_deltas', [0.0]*len(h_ids_list)))
                
                # Target Action
                if 'target_1' in s: t_act = s['target_1']['act']
                elif 'target_1' in meta: t_act = meta['target_1']['act']
                else: t_act = 1
                temp['_t_act'] = t_act
                valid_batch.append(temp)
        
        if not valid_batch:
            return [[] for _ in range(len(batch_data))]

        # Tensor Preparation
        max_len = max(len(s['_h_ids']) for s in valid_batch)
        B = len(valid_batch)
        h_ids = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        h_acts = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        h_deltas = torch.zeros((B, max_len), dtype=torch.float, device=self.device)
        h_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        target_acts = []
        
        for i, s in enumerate(valid_batch):
            safe_hist = [self._safe_id(x) for x in s['_h_ids']]
            l = len(safe_hist)
            h_ids[i, :l] = torch.tensor(safe_hist, device=self.device)
            h_acts[i, :l] = torch.tensor(s['_h_acts'], device=self.device)
            h_deltas[i, :l] = torch.tensor(s['_h_deltas'], device=self.device)
            h_mask[i, :l] = 1
            target_acts.append(s['_t_act'])
            
        with torch.no_grad():
            flat_h_ids = h_ids.view(-1)
            flat_h_vecs = self.item_tower(flat_h_ids)
            h_vecs = flat_h_vecs.view(B, max_len, -1)
            
            t_acts_tensor = torch.tensor(target_acts, dtype=torch.long, device=self.device).unsqueeze(1)
            t_acts_input = torch.cat([t_acts_tensor, t_acts_tensor], dim=1)
            t_deltas_input = torch.zeros((B, 2), dtype=torch.float, device=self.device)
            
            user_preds = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, t_acts_input, t_deltas_input)
            user_vecs = F.normalize(user_preds[:, 0, :], p=2, dim=-1)
            
            scores = torch.matmul(user_vecs, self.item_index.T)
            max_k = max(CONFIG['top_k_list'])
            topk_ids = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy().tolist()
            
        final_results = [[] for _ in range(len(batch_data))]
        for i, real_idx in enumerate(valid_indices):
            final_results[real_idx] = topk_ids[i]
            
        return final_results

# =========================================================
# 📊 主评测逻辑
# =========================================================
def evaluate_all():
    # 1. 初始化模型
    models = {}
    if CONFIG['hier']['enabled']:
        # [关键] 传入 item_profiles 路径
        models['HierGR'] = HierGRWrapper(CONFIG['hier'], CONFIG['sid_mapping'], CONFIG['item_profiles'])
    
    if CONFIG['pinrec']['enabled']:
        models['PinRec'] = PinRecWrapper(CONFIG['pinrec'])
        
    if not models: 
        print("❌ 未启用任何模型。")
        return

    # 2. 加载数据
    print(f"\n📂 Loading Test Data: {CONFIG['test_data']}")
    samples = []
    with open(CONFIG['test_data'], 'r') as f:
        for i, line in enumerate(f):
            if CONFIG['num_samples'] and i >= CONFIG['num_samples']: break
            if line.strip():
                data = json.loads(line)
                if data.get('task') == 'task_a_recommendation':
                    samples.append(data)
    print(f"✅ Loaded {len(samples)} valid samples.")

    # 3. 运行评估
    results = {m_name: defaultdict(int) for m_name in models}
    BATCH_SIZE = 16 # Batch size
    
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Comparing"):
        batch = samples[i : i + BATCH_SIZE]
        
        # 统一提取 Ground Truth (Integer ID)
        targets = []
        for s in batch:
            truth = -1
            # 优先从 metadata 读 target_1.id (最可靠的 Integer ID)
            if 'metadata' in s and 'target_1' in s['metadata']:
                truth = s['metadata']['target_1']['id']
            
            # 如果没有 metadata ID，尝试从 HierGR 的 mapping 里找
            elif 'HierGR' in models:
                 meta = s.get('metadata', {})
                 if 'target_sid' in meta:
                     t_raw = meta['target_sid']
                     if isinstance(t_raw, str):
                        cleaned = t_raw.replace('<','').replace('>','').strip()
                        if cleaned:
                            try:
                                t_sid = tuple(int(x.strip()) for x in cleaned.split(','))
                                if t_sid in models['HierGR'].sid_to_int:
                                    truth = models['HierGR'].sid_to_int[t_sid]
                            except: pass
            
            targets.append(truth)

        # 开始推理
        for name, model in models.items():
            batch_preds = model.predict(batch)
            
            for j, pred_list in enumerate(batch_preds):
                truth = targets[j]
                if truth == -1: continue # 无法确定真值，跳过
                
                for k in CONFIG['top_k_list']:
                    if truth in pred_list[:k]:
                        results[name][f'Hit@{k}'] += 1
                        
                        rank = pred_list[:k].index(truth)
                        results[name][f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)

    # 4. 打印表格
    print("\n" + "="*60)
    print(f"🏆 终极对决结果 (N={len(samples)})")
    print("="*60)
    
    df_data = []
    for name in models:
        metrics = results[name]
        row = {'Model': name}
        for k in CONFIG['top_k_list']:
            row[f'Hit@{k}'] = f"{metrics[f'Hit@{k}'] / len(samples):.2%}"
            row[f'NDCG@{k}'] = f"{metrics[f'NDCG@{k}'] / len(samples):.4f}"
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    evaluate_all()