"""
Semantic Evaluation for Yelp18 using Qwen2.5-7B/1.5B GRPO
Requires: Preprocessed semantic mapping and jsonl data
"""

import os
import sys
import json
import torch
import numpy as np
import re
from math import log2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# 1. 配置区域
# ==========================================
class Config:
    # 你的模型路径 (可以是 7B 或 1.5B)
    base_model = "/workspace/Qwen2.5-7B-Instruct" 
    sft_ckpt   = "/workspace/data/llm_ckpt_sft_qwen2.5_7b_balanced"
    grpo_ckpt  = "/workspace/data/grpo_qwen2.5_7b_breadcrumbs/checkpoint-10000"
    
    # 预处理后的数据路径 (请修改为你实际保存的路径)
    map_file   = "./yelp18/id_to_semantic_map.json"
    test_file  = "./yelp18/test_semantic.jsonl"
    
    # 评估参数
    top_k = 20
    num_samples = 500       # 测试样本数
    max_history = 10        # 历史交互截断长度

# ==========================================
# 2. 语义评估器
# ==========================================
class SemanticYelpEvaluator:
    def __init__(self, config):
        self.conf = config
        self.load_mapping()
        self.load_model()
        
    def load_mapping(self):
        print(f"Loading Mapping: {self.conf.map_file}...")
        with open(self.conf.map_file, 'r') as f:
            self.id2sem = json.load(f)
        
        # 构建反向映射: Semantic String -> ID
        # 用于把模型生成的文字转回 ID 算分
        self.sem2id = {}
        for iid, info in self.id2sem.items():
            # info['semantic_str'] 应该是预处理时生成的类似 <Name> 的字符串
            sem_str = info.get('semantic_str', '').strip()
            if sem_str:
                self.sem2id[sem_str] = int(iid)
                
        print(f"Mapped {len(self.sem2id)} semantic items.")

    def load_model(self):
        print(f"Loading Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.conf.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        
        # 加载 Adapter
        try:
            model = PeftModel.from_pretrained(model, self.conf.sft_ckpt)
            model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, self.conf.grpo_ckpt)
        except Exception as e:
            print(f"Warning: Adapter loading issue ({e})")
            
        model.eval()
        self.model = model

    def get_metrics(self, pred_ids, true_ids):
        # 这里的输入都是数字 ID
        k = self.conf.top_k
        pred_set = pred_ids[:k]
        true_set = set(true_ids)
        
        hits = [1 if i in true_set else 0 for i in pred_set]
        num_hits = sum(hits)
        
        recall = num_hits / len(true_set) if len(true_set) > 0 else 0.0
        hit_rate = 1.0 if num_hits > 0 else 0.0
        
        idcg = sum([1.0 / log2(i + 2) for i in range(min(len(true_set), k))])
        dcg = sum([hits[i] / log2(i + 2) for i in range(len(hits))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return recall, ndcg, hit_rate

    def evaluate(self):
        print("Loading Test Data...")
        data = []
        with open(self.conf.test_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        if self.conf.num_samples:
            data = data[:self.conf.num_samples]
            
        metrics = {'recall': [], 'ndcg': [], 'hr': []}
        print(f"Evaluating {len(data)} samples...")
        
        for item in tqdm(data):
            # item 结构: {'user_id':..., 'history_ids': [...], 'target_ids': [...]}
            # 注意：预处理脚本需要把 history_ids 也放进去，或者我们在这里实时查表
            
            # 1. 构建语义 Prompt
            # 假设 train.txt 里的历史数据在这里可用，或者我们在预处理时已经把 history 转成了 semantic_list
            # 这里为了通用，假设我们只拿到了 history_ids (数字)
            # 我们需要查表转成语义
            
            # ⚠️ 这里需要你的预处理脚本配合，如果没有 history，你需要加载 train.txt
            # 假设 item 里有 'history_ids'
            hist_ids = item.get('history_ids', [])[-self.conf.max_history:]
            hist_strs = []
            for iid in hist_ids:
                s_iid = str(iid)
                if s_iid in self.id2sem:
                    hist_strs.append(self.id2sem[s_iid]['semantic_str'])
            
            history_text = ", ".join(hist_strs)
            
            # 构造和训练时一致的 Prompt
            prompt = (
                f"User History: {history_text}\n"
                f"Please recommend 20 suitable items.\n"
                f"Output format: <Item Name>\n"
                f"Response: <" 
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # 2. 生成 (Beam Search)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,       # 7B 推荐用 4
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
            output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 3. 解析与反向映射 (Semantic -> ID)
            # 假设模型输出: "Starbucks>, <Pizza Hut>, <KFC>"
            # 我们需要用正则提取 <> 里的内容
            candidates = re.findall(r"<(.*?)>", output_text) # 提取尖括号内容
            if not candidates:
                 # 兼容模型忘了输出尖括号的情况
                 candidates = output_text.split(',')
            
            pred_ids = []
            for cand in candidates:
                clean_cand = f"<{cand.strip()}>" # 补全格式去查表
                if clean_cand in self.sem2id:
                    pred_ids.append(self.sem2id[clean_cand])
                # 尝试模糊匹配或去括号匹配
                elif cand.strip() in self.sem2id: # 也许字典里没存尖括号
                     pred_ids.append(self.sem2id[cand.strip()])
            
            # 去重
            pred_ids = list(dict.fromkeys(pred_ids))
            
            # 4. 计算指标
            true_ids = item.get('target_ids', []) # 真实的数字 ID
            r, n, h = self.get_metrics(pred_ids, true_ids)
            
            metrics['recall'].append(r)
            metrics['ndcg'].append(n)
            metrics['hr'].append(h)
            
        print("\n" + "="*40)
        print(f"Yelp18 SEMANTIC Eval Results")
        print("="*40)
        print(f"Recall@20 : {np.mean(metrics['recall']):.4f}")
        print(f"NDCG@20   : {np.mean(metrics['ndcg']):.4f}")
        print(f"HitRate@20: {np.mean(metrics['hr']):.4f}")
        print("="*40)

if __name__ == "__main__":
    evaluator = SemanticYelpEvaluator(Config())
    evaluator.evaluate()