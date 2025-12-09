"""
Validate GRPO Training with t-SNE Visualization (Fixed Prompt Version)
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
import argparse
import re

# LLM imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))

from models.rqvae import RQVAE

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_rqvae_model(config):
    # ... (保持原来的代码不变) ...
    # 为了节省篇幅，这里省略 RQ-VAE 加载代码，请复用你之前的 load_rqvae_model
    rq_conf = config['rqvae']
    data_conf = config['data']
    args = argparse.Namespace(
        num_emb_list=rq_conf['num_emb_list'], e_dim=rq_conf['e_dim'],
        layers=rq_conf['layers'], dropout_prob=rq_conf['dropout_prob'], bn=False,
        loss_type=rq_conf['loss_type'], quant_loss_weight=rq_conf['quant_loss_weight'],
        kmeans_init=True, kmeans_iters=100, sk_epsilons=rq_conf['sk_epsilons'],
        sk_iters=50, beta=rq_conf['beta'], alpha=rq_conf.get('alpha', 1.0),
        n_clusters=rq_conf['n_clusters'], sample_strategy='all'
    )
    model = RQVAE(in_dim=rq_conf['in_dim'], **vars(args))
    ckpt_path = os.path.join(data_conf['rqvae_ckpt_dir'], 'best_collision_model.pth')
    print(f"Loading RQ-VAE from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_grpo_model(config, sft_path, grpo_path):
    # ... (保持原来的代码不变) ...
    base_path = config['llm']['model_name']
    print(f"Loading Base LLM: {base_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    
    print(f"Merging SFT Adapter: {sft_path}")
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()
    
    print(f"Loading GRPO Adapter: {grpo_path}")
    model = PeftModel.from_pretrained(model, grpo_path)
    model.eval()
    return model, tokenizer

def get_tsne_coordinates(model):
    # ... (保持原来的代码不变) ...
    print("Computing t-SNE layout...")
    all_codes = []
    layer_offsets = [0]
    for i, layer in enumerate(model.rq.vq_layers):
        weights = layer.embedding.weight.detach().cpu().numpy()
        all_codes.append(weights)
        if i < len(model.rq.vq_layers) - 1:
            layer_offsets.append(layer_offsets[-1] + len(weights))
    X = np.vstack(all_codes)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    return X_embedded, layer_offsets

def get_city_mapping(config):
    # ... (保持原来的代码不变) ...
    mapping_file = os.path.join(config['data']['processed_dir'], config['data']['sid_mapping_file'])
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    votes = defaultdict(lambda: defaultdict(Counter))
    global_city_counts = Counter()
    for item in data.values():
        city = item['city']
        global_city_counts[city] += 1
        for layer, code in enumerate(item['raw_codes']):
            votes[layer][code][city] += 1
    top_cities = [c for c, _ in global_city_counts.most_common(10)]
    code_city_map = {}
    for layer in votes:
        for code in votes[layer]:
            dominant = votes[layer][code].most_common(1)[0][0]
            code_city_map[(layer, code)] = dominant if dominant in top_cities else 'Other'
    return code_city_map, top_cities

# ==========================================
# [FIXED] 修复的部分：Generate IDs & Main
# ==========================================

def generate_ids(model, tokenizer, city, instruction):
    """
    让模型生成 ID (修复版)
    1. 使用 apply_chat_template 确保格式正确
    2. 增加对 <c0, c1, c2> 的解析鲁棒性
    """
    messages = [
        {"role": "user", "content": instruction}
    ]
    # 使用官方模板构建 Prompt
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64,  # 稍微增加长度以防万一
            temperature=0.1,    # [重要] 降低温度，让模型更确定地输出 ID，不瞎聊
            top_p=0.9
        )
    
    # 只解码新生成的部分
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 提取 <c0, c1, c2> (支持带空格或不带空格)
    # 匹配 <12, 34, 56> 或 <12,34,56>
    match = re.search(r"<(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", output_text)
    
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))], output_text
    return None, output_text

def plot_grpo_validation(X_embedded, layer_offsets, code_city_map, top_cities, test_cases, output_dir):
    # ... (保持原来的绘图逻辑不变) ...
    plt.figure(figsize=(18, 14))
    
    palette = sns.color_palette("bright", len(top_cities))
    city_to_color = {city: palette[i] for i, city in enumerate(top_cities)}
    city_to_color['Other'] = (0.8, 0.8, 0.8)
    markers = ['o', 's', '^']
    
    print("Drawing background map...")
    for idx in range(X_embedded.shape[0]):
        layer = 0
        while layer < len(layer_offsets) - 1 and idx >= layer_offsets[layer+1]:
            layer += 1
        code_idx = idx - layer_offsets[layer]
        city = code_city_map.get((layer, code_idx), 'Unused')
        if city == 'Unused': continue
        color = city_to_color.get(city, city_to_color['Other'])
        alpha = 0.6 if city in top_cities else 0.1
        size = 60 if city in top_cities else 20
        plt.scatter(
            X_embedded[idx, 0], X_embedded[idx, 1],
            c=[color], marker=markers[layer], s=size, alpha=alpha, edgecolors='none'
        )

    print("Overlaying GRPO predictions...")
    path_colors = ['black', 'blue', 'purple', 'darkgreen', 'darkred']
    legend_handles = []
    
    for i, case in enumerate(test_cases):
        city = case['city']
        ids = case['ids']
        if not ids: continue
        
        coords = []
        for layer, code in enumerate(ids):
            global_idx = layer_offsets[layer] + code
            if global_idx < X_embedded.shape[0]:
                coords.append(X_embedded[global_idx])
            else:
                print(f"Warning: Code {code} out of bounds for layer {layer}")
        
        coords = np.array(coords)
        if len(coords) < 2: continue
        
        line_color = path_colors[i % len(path_colors)]
        plt.plot(coords[:, 0], coords[:, 1], color=line_color, linewidth=3, linestyle='-', alpha=0.9, zorder=10)
        plt.arrow(coords[0, 0], coords[0, 1], coords[1, 0]-coords[0, 0], coords[1, 1]-coords[0, 1], color=line_color, head_width=0, length_includes_head=True, zorder=10)
        plt.scatter(coords[0, 0], coords[0, 1], c='white', s=300, marker='*', edgecolors='black', zorder=11)
        plt.scatter(coords[-1, 0], coords[-1, 1], c='white', s=150, marker='X', edgecolors='black', zorder=11)
        
        # 标签调整：为了避免文字重叠，稍微偏移一点
        offset_y = 1.5 if i % 2 == 0 else -1.5
        plt.text(coords[0, 0], coords[0, 1]+offset_y, f"{city}", fontsize=12, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='w', label='--- Background ---', marker='None')]
    for city in top_cities[:5]:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=city, markerfacecolor=city_to_color[city], markersize=8))
    legend_elements.append(Line2D([0], [0], color='w', label='--- GRPO Paths ---', marker='None'))
    legend_elements.append(Line2D([0], [0], marker='*', color='w', label='Start (Layer 0)', markerfacecolor='white', markeredgecolor='k', markersize=10))
    legend_elements.append(Line2D([0], [0], marker='X', color='w', label='End (Layer 2)', markerfacecolor='white', markeredgecolor='k', markersize=10))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(f"Visual Validation of GRPO Model (Checkpoint-8400)\nChecking if Predictions land in correct City Clusters", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "grpo_validation_tsne.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Validation Plot saved to: {save_path}")

def main():
    sft_ckpt = "/workspace/data/llm_ckpt/checkpoint-28000"
    grpo_ckpt = "/workspace/data/grpo_checkpoints/checkpoint-8400"
    
    config = load_config('./config/config.yaml')
    os.makedirs('data/visualization', exist_ok=True)
    
    rqvae = load_rqvae_model(config)
    X_embedded, layer_offsets = get_tsne_coordinates(rqvae)
    code_city_map, top_cities = get_city_mapping(config)
    
    llm, tokenizer = load_grpo_model(config, sft_ckpt, grpo_ckpt)
    
    test_cities = ['Philadelphia', 'Reno', 'Tampa', 'Tucson', 'Nashville']
    results = []
    
    print("\n=== Running GRPO Inference (Fixed Prompt) ===")
    for city in test_cities:
        # [FIXED] 这里的 Prompt 必须更具指令性，强制模型输出 ID
        instruction = (
            f"User is currently in {city}. "
            "Recommend a business that matches the user's preference. "
            "Output the semantic ID in the format <c0, c1, c2, suffix>."
        )
        print(f"Testing City: {city}...")
        
        ids, text = generate_ids(llm, tokenizer, city, instruction)
        if ids:
            print(f"  -> Generated IDs: {ids}")
            results.append({'city': city, 'ids': ids})
        else:
            print(f"  -> Failed to parse IDs. Output: {text}")
            
    plot_grpo_validation(X_embedded, layer_offsets, code_city_map, top_cities, results, 'data/visualization')

if __name__ == '__main__':
    main()