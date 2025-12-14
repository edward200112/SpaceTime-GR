import torch
from tqdm import tqdm
import json
import os
import inspect
import numpy as np
from haversine import haversine

from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig

# ==========================================
# ⚙️ 1. 全局配置 & 自适应参数加载 (核心修复)
# ==========================================
CONFIG = {
    "base_model": "/workspace/Qwen2_5-1.5B-Instruct",
    "sft_ckpt": "/workspace/data/llm_ckpt_sft_v2_optimized/checkpoint-35000",
    "sid_mapping": "/workspace/data/processed/sid_mapping.json",
    "train_data": "/workspace/data/processed/train_prompts.jsonl",
    "output_dir": "/workspace/data/llm_ckpt_ppo_final",
    "device": "cuda"
}

def get_compatible_config():
    """
    动态检测 PPOConfig 支持哪些参数，自动过滤掉不支持的，防止报错。
    """
    # 1. 定义我们想要设置的理想参数
    desired_config = {
        "learning_rate": 1e-5,
        "batch_size": 32,
        "mini_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "ppo_epochs": 4,
        "seed": 42,
        "init_kl_coef": 0.2,
        "target_kl": 0.1,
        "adap_kl_ctrl": True,
        "optimize_cuda_cache": True,
    }

    # 2. 获取当前环境 PPOConfig 的实际参数签名
    signature = inspect.signature(PPOConfig.__init__)
    valid_keys = set(signature.parameters.keys())
    
    # 3. 过滤参数
    filtered_config = {}
    print("\n🔍 [Config Diagnostics] 检测 PPOConfig 支持的参数...")
    for k, v in desired_config.items():
        if k in valid_keys:
            filtered_config[k] = v
        else:
            print(f"⚠️ 警告: 当前 trl 版本不支持参数 '{k}'，已自动忽略。")
    
    # 4. 如果连最基本的 batch_size 都不支持，可能是 dataclass 形式，尝试直接赋值
    if not filtered_config and hasattr(PPOConfig, '__dataclass_fields__'):
         print("⚠️ 检测到 PPOConfig 是 Dataclass，尝试直接匹配字段...")
         valid_keys = set(PPOConfig.__dataclass_fields__.keys())
         for k, v in desired_config.items():
            if k in valid_keys:
                filtered_config[k] = v
    
    print(f"✅ 最终使用的有效配置: {filtered_config}\n")
    return PPOConfig(**filtered_config)

# 初始化 Config
ppo_config = get_compatible_config()

# LoRA 配置
lora_config = LoraConfig(
    r=64,               # <--- Changed from 16 to 64 to match SFT
    lora_alpha=128,     # <--- Changed from 32 to 128 (Standard is 2x rank)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# 🔧 2. 奖励函数 (同前)
# ==========================================
print("Loading ID Mapping for Reward Calculation...")
with open(CONFIG["sid_mapping"], 'r') as f:
    sid_map = json.load(f)
    tree_map = {}
    for k, v in sid_map.items():
        tree_map[tuple(v['full_sid'])] = v

def compute_reward_scalar(pred_text, target_str, meta_info):
    total_reward = 0.0
    # A. 格式
    pred_tuple = None
    try:
        content = pred_text.split("<")[-1].split(">")[0] if "<" in pred_text else pred_text
        parts = [int(x.strip()) for x in content.split(',')]
        if len(parts) == 4:
            pred_tuple = tuple(parts)
            total_reward += 0.2
        else: return -1.0
    except: return -1.0
    
    # B. 语义
    try:
        t_clean = target_str.replace('<','').replace('>','').strip()
        target_tuple = tuple(int(x) for x in t_clean.split(','))
    except: return 0.0
        
    if pred_tuple[0] == target_tuple[0]:
        total_reward += 0.2
        if pred_tuple[1] == target_tuple[1]:
            total_reward += 0.3
            if pred_tuple[2] == target_tuple[2]:
                total_reward += 1.5 
                if pred_tuple[3] == target_tuple[3]:
                    total_reward += 3.0
    
    # C. 地理
    if pred_tuple in tree_map:
        pred_meta = tree_map[pred_tuple]
        try:
            dist = haversine((meta_info['target_lat'], meta_info['target_lon']), (pred_meta['latitude'], pred_meta['longitude']))
            if dist <= 5.0: total_reward += 0.5
            elif dist <= 20.0: total_reward += 0.2
            elif dist > 50.0: total_reward -= 0.5
        except: pass
    else: total_reward -= 0.5
        
    return total_reward

# ==========================================
# 📊 3. 数据加载 (同前)
# ==========================================
def build_dataset(tokenizer, data_path):
    dataset = []
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item.get('task') == 'task_a_recommendation':
                instruction = item['instruction']
                query = instruction.split("Response:")[0].strip() + "\nResponse:" if "Response:" in instruction else instruction
                dataset.append({
                    "query": query,
                    "target_sid": item['metadata']['target_sid'],
                    "meta": item['metadata']
                })
    print(f"Loaded {len(dataset)} samples for PPO.")
    return dataset

def collator(data):
    return {
        "query": [d['query'] for d in data],
        "target_sid": [d['target_sid'] for d in data],
        "meta": [d['meta'] for d in data]
    }

# ==========================================
# 🚀 4. 主训练逻辑
# ==========================================
def train():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_dataset = build_dataset(tokenizer, CONFIG["train_data"])
    
    print("Loading SFT Model with Value Head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["base_model"],
        peft_config=lora_config,
        torch_dtype=torch.bfloat16,
        device_map=CONFIG["device"]
    )
    
    print(f"Loading SFT Adapter weights from {CONFIG['sft_ckpt']}...")
    model.pretrained_model.load_adapter(CONFIG["sft_ckpt"], adapter_name="default")
    
    ref_model = create_reference_model(model)

    # 这里的 config 使用的是上面 get_compatible_config 生成的对象
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=raw_dataset,
        data_collator=collator
    )

    generation_kwargs = {
        "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 32,
    }

    print("🚀 Starting PPO Loop...")
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"][0].to(CONFIG["device"]) for q in batch["query"]]
        
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        batch_rewards = []
        response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        for i, pred_text in enumerate(response_texts):
            target_str = batch["target_sid"][i]
            meta = batch["meta"][i]
            reward_val = compute_reward_scalar(pred_text, target_str, meta)
            batch_rewards.append(torch.tensor(reward_val, dtype=torch.float32))
        
        stats = ppo_trainer.step(query_tensors, response_tensors, batch_rewards)
        ppo_trainer.log_stats(stats, batch, batch_rewards)

    print(f"💾 Saving PPO model to {CONFIG['output_dir']}")
    ppo_trainer.save_pretrained(CONFIG["output_dir"])

if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train()