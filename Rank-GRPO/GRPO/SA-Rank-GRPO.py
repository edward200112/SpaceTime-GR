import os
import re
import math
import gzip
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
import time
from sklearn.neighbors import BallTree
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments 
from peft import PeftModel

# ================= 0. 核心配置 =================
@dataclass
class SARankConfig(TrainingArguments):
    # --- 1. 采样配置 (G=8) ---
    num_generations: int = field(default=8, metadata={"help": "Group Size G"})
    max_completion_length: int = field(default=256, metadata={"help": "Max gen len"})
    max_prompt_length: int = field(default=1024, metadata={"help": "Max prompt len"})
    
    # 显式覆盖父类默认值
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Grad accum"})
    
    # --- SA-Rank 核心算法参数 ---
    enable_geometric_mean: bool = True
    enable_adaptive_kde: bool = True
    
    w_rel: float = 1.0
    w_geo: float = 2.0
    w_div: float = 0.5
    w_pop: float = 0.5
    w_consist: float = 1.0
    w_dense: float = 0.5
    
    semantic_threshold: float = 0.05
    ips_gamma: float = 0.5
    ips_clip_m: float = 10.0
    kde_alpha: float = 1.0
    kde_k: int = 20
    consistency_penalty: float = -1.0
    
    raw_data_dir: str = "/workspace/data/GoogleRAW"
    poi_id_map_file: str = "./poi_semantic_ids.csv"
    train_data_file: str = "./SFT/sft_data/sft_balanced_train.jsonl"
    meta_files: List[str] = None
    
    # 学习参数
    beta: float = 0.04 # KL Penalty
    
    # [新增] 必须配置项
    remove_unused_columns: bool = False 
    
    def __post_init__(self):
        super().__post_init__()
        if self.meta_files is None:
            self.meta_files = [
                "meta-California.json.gz", "meta-New_York.json.gz", 
                "meta-New_Mexico.json.gz", "meta-Pennsylvania.json.gz"
            ]

# ================= 1. 数据基建 =================
class DataManager:
    def __init__(self, config: SARankConfig):
        self.cfg = config
        self.poi_db = {}     
        self.code2gmap = {}  
        self.gmap2code = {}  
        self.coords = []     
        self.spatial_tree = None
        self.propensity_map = {}

        print("🚀 Initializing Data Infrastructure...")
        self._load_id_mapping()
        self._load_raw_metadata()
        self._build_spatial_index()
        self._calc_propensity()

    def _load_id_mapping(self):
        print(f"📥 Loading ID Mapping...")
        if not os.path.exists(self.cfg.poi_id_map_file):
            print(f"⚠️ Warning: {self.cfg.poi_id_map_file} not found. Skipping ID map.")
            return
        df = pd.read_csv(self.cfg.poi_id_map_file)
        df['gmap_id'] = df['gmap_id'].astype(str)
        for _, row in df.iterrows():
            code = f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}"
            self.code2gmap[code] = row['gmap_id']
            self.gmap2code[row['gmap_id']] = code

    def _load_raw_metadata(self):
        print("📥 Loading Metadata...")
        for m_file in self.cfg.meta_files:
            path = os.path.join(self.cfg.raw_data_dir, m_file)
            if not os.path.exists(path): continue
            with gzip.open(path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        if gid in self.gmap2code:
                            code = self.gmap2code[gid]
                            lat, lon = d.get('latitude'), d.get('longitude')
                            if lat and lon:
                                cats = d.get('category')
                                cat_set = set(cats) if isinstance(cats, list) else (set([cats]) if cats else set())
                                self.poi_db[code] = {
                                    'loc': (float(lat), float(lon)),
                                    'cats': cat_set,
                                    'reviews': d.get('avg_rating', 0),
                                    'price': d.get('price', ''),
                                    'name': d.get('name', 'Unknown')
                                }
                    except: continue

    def _build_spatial_index(self):
        print("🏗️ Building Spatial Index (BallTree)...")
        for code, info in self.poi_db.items():
            self.coords.append([np.radians(info['loc'][0]), np.radians(info['loc'][1])])
        if self.coords:
            self.spatial_tree = BallTree(np.array(self.coords), metric='haversine')

    def _calc_propensity(self):
        print("📊 Pre-calculating IPS Weights...")
        counter = Counter()
        total = 0
        if os.path.exists(self.cfg.train_data_file):
            with open(self.cfg.train_data_file, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        gt = item.get('raw_target_code', '').strip()
                        if gt:
                            counter[gt] += 1
                            total += 1
                    except: pass
        
        vocab_size = len(self.code2gmap) if self.code2gmap else 1
        for code in self.code2gmap:
            self.propensity_map[code] = (counter.get(code, 0) + 1.0) / (total + vocab_size)

    def get_info(self, code): return self.poi_db.get(code)
    def get_propensity(self, code): return self.propensity_map.get(code, 1e-9)
    def get_bandwidth_h(self, lat, lon, k=20, alpha=1.0):
        if not self.spatial_tree: return 1.0
        u_rad = np.array([[np.radians(lat), np.radians(lon)]])
        dist, _ = self.spatial_tree.query(u_rad, k=min(k+1, len(self.coords)))
        d_k_km = dist[0][-1] * 6371.0
        return max(alpha * d_k_km, 0.1)

# ================= 2. 奖励系统 (核心修改区域) =================
class RewardSystem:
    def __init__(self, dm: DataManager, config: SARankConfig):
        self.dm = dm
        self.cfg = config

    def haversine_km(self, loc1, loc2):
        R = 6371.0
        dlat = math.radians(loc2[0] - loc1[0])
        dlon = math.radians(loc2[1] - loc1[1])
        a = math.sin(dlat/2)**2 + math.cos(math.radians(loc1[0])) * math.cos(math.radians(loc2[0])) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def calc_rewards(self, prompts_text, completions_text, generated_codes, gt_codes):
        rewards_rel, rewards_geo, rewards_pop = [], [], []
        rewards_consist, rewards_dense = [], []
        ips_weights = []
        
        all_cats = []
        for code in generated_codes:
            info = self.dm.get_info(code)
            if info: all_cats.extend(list(info['cats']))
        
        group_entropy = 0.0
        if all_cats:
            counts = Counter(all_cats)
            total = sum(counts.values())
            probs = np.array([c/total for c in counts.values()])
            group_entropy = -np.sum(probs * np.log(probs + 1e-9))

        for i in range(len(generated_codes)):
            pred_id = generated_codes[i]
            gt_id = gt_codes[i]
            prompt = prompts_text[i]
            reasoning_text = completions_text[i]
            
            user_loc = self._extract_last_poi_loc(prompt)
            pred_info = self.dm.get_info(pred_id)
            gt_info = self.dm.get_info(gt_id)
            
            # A. Relevance
            r_rel = 0.0
            pred_parts = pred_id.split()
            gt_parts = gt_id.split()
            
            if pred_id == gt_id:
                r_rel = 1.0
            elif len(pred_parts)>=2 and len(gt_parts)>=2 and pred_parts[:2] == gt_parts[:2]:
                r_rel = 0.2
            elif len(pred_parts)>=1 and len(gt_parts)>=1 and pred_parts[0] == gt_parts[0]:
                r_rel = 0.1

            # B. Spatial [关键修改：解开了 r_rel 的锁]
            r_geo = 0.0
            # 只要生成的 ID 是个有效的 POI (pred_info 存在)，就计算距离奖励！
            if user_loc and pred_info: 
                dist = self.haversine_km(user_loc, pred_info['loc'])
                h_u = self.dm.get_bandwidth_h(user_loc[0], user_loc[1], k=self.cfg.kde_k, alpha=self.cfg.kde_alpha)
                r_geo = math.exp(-(dist**2)/(2*h_u**2))
            
            # C. Consistency
            r_consist = 0.0
            if pred_info:
                text_lower = reasoning_text.lower()
                price = pred_info.get('price', '')
                if ('cheap' in text_lower or 'budget' in text_lower) and price in ['$$$', '$$$$']:
                    r_consist += self.cfg.consistency_penalty
                elif ('expensive' in text_lower or 'luxury' in text_lower) and price in ['$', '$$', 'free']:
                    r_consist += self.cfg.consistency_penalty

            # D. Dense Reasoning
            r_dense = 0.0
            if gt_info and any(c.lower() in reasoning_text.lower() for c in gt_info['cats']): r_dense += 0.3
            if "location" in reasoning_text.lower() or "5km" in reasoning_text.lower(): r_dense += 0.2
            if "Target:" in reasoning_text: r_dense += 0.5

            # E. Pop Debias
            r_pop = - math.log(pred_info['reviews'] + 1) if pred_info else 0.0

            # F. IPS Weight
            p_obs = self.dm.get_propensity(pred_id)
            w = min(1.0/p_obs, self.cfg.ips_clip_m) ** self.cfg.ips_gamma
            
            rewards_rel.append(r_rel)
            rewards_geo.append(r_geo)
            rewards_pop.append(r_pop)
            rewards_consist.append(r_consist)
            rewards_dense.append(r_dense)
            ips_weights.append(w)
            
        return rewards_rel, rewards_geo, [group_entropy]*len(generated_codes), rewards_pop, rewards_consist, rewards_dense, ips_weights

    def _extract_last_poi_loc(self, prompt):
        codes = re.findall(r"(\d+ \d+ \d+ \d+)", prompt)
        if codes:
            last_code = codes[-1]
            info = self.dm.get_info(last_code)
            if info: return info['loc']
        return None

# ================= 3. SA-Rank Trainer =================
class SARankTrainer(Trainer):
    def __init__(self, reward_system: RewardSystem, ref_model, **kwargs):
        super().__init__(**kwargs)
        self.rs = reward_system
        self.sa_cfg = kwargs.get('args')
        self.ref_model = ref_model 
        self.eval_metrics_buffer = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompts_text = inputs.get("prompts_text", []) 
        gt_codes = inputs.get("gt_codes", [])

        G = self.sa_cfg.num_generations
        
        was_training = model.training
        model.eval()
        
        original_use_cache = model.config.use_cache
        original_gc_enabled = model.is_gradient_checkpointing
        
        model.config.use_cache = True 
        if original_gc_enabled:
            model.gradient_checkpointing_disable()
            
        with torch.no_grad():
            repeated_prompt_ids = prompt_ids.repeat_interleave(G, dim=0)
            repeated_prompt_mask = prompt_mask.repeat_interleave(G, dim=0)

            generation_output = model.generate(
                input_ids=repeated_prompt_ids,
                attention_mask=repeated_prompt_mask,
                max_new_tokens=self.sa_cfg.max_completion_length,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                use_cache=True 
            )
            
        if self.state.global_step % 50 == 0 and was_training: 
            temp_text = self.processing_class.decode(generation_output[0], skip_special_tokens=True)
            gen_len = len(generation_output[0]) - repeated_prompt_ids.shape[1]
            print(f"\n[DEBUG Step {self.state.global_step}] Gen Len: {gen_len} tokens")
            print(f"[Preview]: {temp_text[-200:]}") 
        
        torch.cuda.synchronize() 
            
        model.config.use_cache = False
        if original_gc_enabled:
            model.gradient_checkpointing_enable()
        
        if was_training:
            model.train()
        else:
            model.eval()
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().requires_grad_(True)
            
        full_ids = generation_output
        full_mask = (full_ids != self.processing_class.pad_token_id).long()
        
        labels = full_ids.clone()
        prompt_len = repeated_prompt_ids.shape[1]
        labels[:, :prompt_len] = -100 
        labels[full_ids == self.processing_class.pad_token_id] = -100

        outputs = model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits 
        del outputs

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        per_token_logps = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        ).view(shift_labels.shape)

        completion_mask = (shift_labels != -100).float()
        per_token_logps = per_token_logps * completion_mask

        del logits, shift_logits
        
        with torch.no_grad():
            if self.ref_model.device != model.device:
                self.ref_model = self.ref_model.to(model.device)
            
            ref_outputs = self.ref_model(input_ids=full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]
            
            ref_per_token_logps = -F.cross_entropy(
                ref_logits.reshape(-1, ref_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            ).view(shift_labels.shape)
            
            ref_per_token_logps = ref_per_token_logps * completion_mask
            del ref_outputs, ref_logits

        reward_inputs = {
            "input_ids": full_ids,
            "prompts_text": prompts_text,
            "gt_codes": gt_codes
        }
        advantages, stats = self.compute_sa_advantages(reward_inputs)
        
        if not model.training:
            for k, v in stats.items():
                self.eval_metrics_buffer[k].append(v)

        per_token_advantages = advantages.unsqueeze(1).expand_as(per_token_logps) * completion_mask

        kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)
        kl = kl * completion_mask
        
        ratio = torch.exp(per_token_logps - ref_per_token_logps)
        
        epsilon_clip = 0.2
        surr1 = ratio * per_token_advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * per_token_advantages
        
        pg_loss = -torch.min(surr1, surr2).sum() / (completion_mask.sum() + 1e-9)
        kl_mean = kl.sum() / (completion_mask.sum() + 1e-9)
        
        loss = pg_loss + self.sa_cfg.beta * kl_mean
        
        if self.state.global_step % 50 == 0 and model.training:
            print(f"  [Stats] Ratio Max: {ratio.max().item():.2f}, Adv Max: {per_token_advantages.max().item():.2f}")

        if return_outputs:
            return loss, loss.unsqueeze(0)
        
        return loss

    def compute_sa_advantages(self, inputs):
        full_ids = inputs["input_ids"]
        prompts_text = inputs["prompts_text"]
        gt_codes = inputs["gt_codes"]
        
        G = self.sa_cfg.num_generations
        B = len(gt_codes)

        full_texts = self.processing_class.batch_decode(full_ids, skip_special_tokens=True)
        pred_codes = [self._extract_target(t) for t in full_texts]
        
        completions = []
        for i, text in enumerate(full_texts):
            if "assistant\n" in text:
                completions.append(text.split("assistant\n")[-1])
            else:
                completions.append(text)

        gt_codes_expanded = np.repeat(gt_codes, G).tolist()
        prompts_expanded = np.repeat(prompts_text, G).tolist()
        
        r_rel, r_geo, r_div, r_pop, r_consist, r_dense, w_ips = self.rs.calc_rewards(
            prompts_expanded, completions, pred_codes, gt_codes_expanded
        )
        
        dev = full_ids.device
        t_rel = torch.tensor(r_rel, device=dev, dtype=torch.float32)
        t_geo = torch.tensor(r_geo, device=dev, dtype=torch.float32)
        t_div = torch.tensor(r_div, device=dev, dtype=torch.float32)
        t_pop = torch.tensor(r_pop, device=dev, dtype=torch.float32)
        t_consist = torch.tensor(r_consist, device=dev, dtype=torch.float32)
        t_dense = torch.tensor(r_dense, device=dev, dtype=torch.float32)
        t_ips = torch.tensor(w_ips, device=dev, dtype=torch.float32)

        def group_norm(t):
            t_reshaped = t.view(B, G)
            mean = t_reshaped.mean(dim=1, keepdim=True)
            std = t_reshaped.std(dim=1, keepdim=True)
            normed = (t_reshaped - mean) / (std + 1e-9)
            return normed.view(-1)

        A_raw = (self.sa_cfg.w_rel * group_norm(t_rel) + 
                 self.sa_cfg.w_geo * group_norm(t_geo) + 
                 self.sa_cfg.w_div * group_norm(t_div) + 
                 self.sa_cfg.w_pop * group_norm(t_pop) + 
                 self.sa_cfg.w_consist * group_norm(t_consist) + 
                 self.sa_cfg.w_dense * group_norm(t_dense))
        
        stats = {
            "reward/rel": np.mean(r_rel),
            "reward/geo": np.mean(r_geo),
            "reward/pop": np.mean(r_pop),
            "reward/consist": np.mean(r_consist),
            "reward/dense": np.mean(r_dense),
            "reward/ips_weight": np.mean(w_ips)
        }
        
        return A_raw * t_ips, stats


    def _extract_target(self, text):
        m = re.search(r"Target:\s*([\d\s]+)", text)
        if m: return m.group(1).strip()
        return ""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_metrics_buffer.clear()
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        custom_metrics = {}
        if self.eval_metrics_buffer:
            for k, v_list in self.eval_metrics_buffer.items():
                custom_metrics[f"{metric_key_prefix}_{k}"] = sum(v_list) / len(v_list)
        output.update(custom_metrics)
        
        print("\n" + "="*30 + " CUSTOM EVAL METRICS " + "="*30)
        for k, v in custom_metrics.items():
            print(f"  📊 {k}: {v:.4f}")
        print("="*80 + "\n")
        
        self.log(custom_metrics)
        return output

# ================= 4. 自定义 Collator =================
@dataclass
class SADataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts_text = [f["prompts_text"] for f in features]
        gt_codes = [f["gt_codes"] for f in features]
        
        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]
        mask_list = [torch.tensor(f["attention_mask"]) for f in features]
        
        input_ids_flipped = [t.flip(0) for t in input_ids_list]
        mask_flipped = [t.flip(0) for t in mask_list]
        
        padded_ids_flipped = torch.nn.utils.rnn.pad_sequence(
            input_ids_flipped, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_mask_flipped = torch.nn.utils.rnn.pad_sequence(
            mask_flipped, batch_first=True, padding_value=0
        )
        
        final_input_ids = padded_ids_flipped.flip(1)
        final_attention_mask = padded_mask_flipped.flip(1)
        
        batch = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "prompts_text": prompts_text,
            "gt_codes": gt_codes,   
            "labels": final_input_ids.clone() 
        }
        return batch

# ================= 5. Main =================
def main():
    # 1. 核心配置: Stage 2
    cfg = SARankConfig(
        output_dir="./GRPO/output_sarank_stage2", # [修改] 避免覆盖 Stage 1
        remove_unused_columns=False,
        num_generations=4, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=5,
        max_steps=1000, # [修改] Stage 2 不用跑太久，500步足够观察效果
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # [关键] 降低学习率，微调 ckpt800
        learning_rate=5e-7,  
        warmup_ratio=0.1,
        
        prediction_loss_only=True,
        max_grad_norm=1.0,
    )
    
    dm = DataManager(cfg)
    rs = RewardSystem(dm, cfg)

    # [修改] 直接加载 Stage 1 训练好的合并模型作为 Policy
    base_model_path = "/workspace/Rank-GRPO/GRPO/output_sarank/checkpoint-800"
    
    # 原始 Base 模型路径 (仅用于 tokenizer 和 Ref Model 的初始化)
    original_base_path = "/workspace/Qwen2_5-1.5B-Instruct"
    # SFT Adapter 路径 (用于 Ref Model)
    sft_adapter_path = "/workspace/Rank-GRPO/SFT/sft_output_coin/checkpoint-44000"
    
    print(f"🔄 Loading Tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    # 1. 加载 Policy Model (使用 ckpt800)
    print(f"🔄 Loading Policy Model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    # 注意：这里不需要再加载 SFT adapter，因为 ckpt800 已经是 merge 过的了

    print("   🛑 Gradient Checkpointing DISABLED for speed (VRAM is sufficient).")
    model.gradient_checkpointing_disable() 
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    print("   🔓 Unfreezing model parameters for training...")
    for param in model.parameters():
        param.requires_grad = True

    # 2. 加载 Reference Model (使用 Original Base + SFT Adapter)
    #    这样 KL 约束的是让模型不要偏离 SFT 太多
    print("   ❄️ Loading Reference Model (Original Base + SFT)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        original_base_path, # 用原始 Qwen
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    if os.path.exists(sft_adapter_path):
        print("     + Loading SFT Adapter for Reference...")
        ref_model = PeftModel.from_pretrained(ref_model, sft_adapter_path)
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    ref_model.requires_grad_(False)

    processed_data_dir = "./processed_data_cache"
    if os.path.exists(processed_data_dir):
        dataset_dict = load_from_disk(processed_data_dir)
        train_ds = dataset_dict["train"]
        eval_ds = dataset_dict["test"]
    else:
        pass 
        
    if len(eval_ds) > 200:
        print(f"⚠️ Truncating Eval Dataset from {len(eval_ds)} to 200 for speed...")
        eval_ds = eval_ds.select(range(10))
    print(f"✅ Final Dataset Size - Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    data_collator = SADataCollator(tokenizer=tokenizer)

    trainer = SARankTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        reward_system=rs,
        ref_model=ref_model,
        tokenizer=tokenizer
    )
    
    print(f"🚀 Starting SA-Rank Stage 2 (Unlocked Geo Reward)...")
    trainer.train()
    trainer.save_model(cfg.output_dir)

if __name__ == "__main__":
    main()