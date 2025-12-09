"""
Visualize RQ-VAE Codebooks by City (Geography-Aware t-SNE)
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

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))

from models.rqvae import RQVAE

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(config):
    """加载 RQ-VAE 模型权重 (用于获取 Codebook 坐标)"""
    rq_conf = config['rqvae']
    data_conf = config['data']
    device = torch.device('cpu') # 可视化不需要 GPU
    
    args = argparse.Namespace(
        num_emb_list=rq_conf['num_emb_list'],
        e_dim=rq_conf['e_dim'],
        layers=rq_conf['layers'],
        dropout_prob=rq_conf['dropout_prob'],
        bn=False,
        loss_type=rq_conf['loss_type'],
        quant_loss_weight=rq_conf['quant_loss_weight'],
        kmeans_init=True,
        kmeans_iters=rq_conf.get('kmeans_iters', 100),
        sk_epsilons=rq_conf['sk_epsilons'],
        sk_iters=rq_conf.get('sk_iters', 50),
        beta=rq_conf['beta'],
        alpha=rq_conf.get('alpha', 1.0),
        n_clusters=rq_conf['n_clusters'],
        sample_strategy='all'
    )
    
    model = RQVAE(
        in_dim=rq_conf['in_dim'], 
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        beta=args.beta,
        alpha=args.alpha,
        n_clusters=args.n_clusters,
        sample_strategy=args.sample_strategy
    )
    
    ckpt_path = os.path.join(data_conf['rqvae_ckpt_dir'], 'best_collision_model.pth')
    print(f"Loading weights from: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def compute_dominant_cities(config):
    """
    读取 sid_mapping.json，计算每个 Code 的主导城市。
    返回: 
        code_city_map: { (layer_idx, code_idx): 'Las Vegas' }
        top_cities: list of top N city names
    """
    mapping_file = os.path.join(config['data']['processed_dir'], config['data']['sid_mapping_file'])
    print(f"Reading mapping data from: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 统计每个 Code 被哪些城市使用了多少次
    # 结构: votes[layer][code_idx][city] = count
    votes = defaultdict(lambda: defaultdict(Counter))
    global_city_counts = Counter()
    
    for item in data.values():
        city = item['city']
        raw_codes = item['raw_codes'] # [c0, c1, c2]
        
        global_city_counts[city] += 1
        
        for layer_idx, code_val in enumerate(raw_codes):
            votes[layer_idx][code_val][city] += 1
            
    # 2. 选出 Top 10 城市用于染色，其他的归为 'Other'
    top_cities = [c for c, _ in global_city_counts.most_common(10)]
    print(f"Top 10 Cities: {top_cities}")
    
    # 3. 决定每个 Code 的归属
    code_city_map = {}
    
    # 遍历所有可能的层和码
    # 注意：这里我们遍历 votes 中记录的码。没被用过的码将不会在 map 中，后续画图时设为灰色。
    for layer_idx in votes:
        for code_val in votes[layer_idx]:
            # 获取该 Code 出现最多的城市
            most_common = votes[layer_idx][code_val].most_common(1)
            if most_common:
                dominant_city = most_common[0][0]
                # 只有当主导城市在 Top 10 里时，才记录具体名字，否则记为 'Other'
                if dominant_city in top_cities:
                    code_city_map[(layer_idx, code_val)] = dominant_city
                else:
                    code_city_map[(layer_idx, code_val)] = 'Other'
            else:
                code_city_map[(layer_idx, code_val)] = 'Unused'
                
    return code_city_map, top_cities

def visualize_by_city(model, code_city_map, top_cities, output_dir):
    print("Running t-SNE and plotting...")
    
    all_codes = []
    layer_labels = [] # 0, 1, 2
    code_indices = [] # 0~255
    
    # 提取权重
    for i, layer in enumerate(model.rq.vq_layers):
        weights = layer.embedding.weight.detach().cpu().numpy()
        all_codes.append(weights)
        layer_labels.extend([i] * len(weights))
        code_indices.extend(list(range(len(weights))))
        
    X = np.vstack(all_codes)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 准备绘图颜色
    # 使用 seaborn 调色板
    palette = sns.color_palette("bright", len(top_cities))
    city_to_color = {city: palette[i] for i, city in enumerate(top_cities)}
    city_to_color['Other'] = (0.7, 0.7, 0.7) # 灰色
    city_to_color['Unused'] = (0.9, 0.9, 0.9) # 极浅灰
    
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    
    # 形状映射
    markers = ['o', 's', '^'] # Layer 0: Circle, Layer 1: Square, Layer 2: Triangle
    
    # 为了图例整洁，我们需要手动维护图例句柄
    legend_elements = []
    seen_cities = set()
    
    # 绘制点
    # 建议先画 Unused 和 Other (背景)，再画 Top Cities (前景)
    z_orders = {'Unused': 0, 'Other': 1}
    for city in top_cities: z_orders[city] = 2
    
    for i in range(X.shape[0]):
        layer = layer_labels[i]
        code_idx = code_indices[i]
        
        # 获取该点的城市
        city = code_city_map.get((layer, code_idx), 'Unused')
        
        color = city_to_color.get(city, city_to_color['Other'])
        marker = markers[layer]
        z = z_orders.get(city, 1)
        
        # 调整大小：Top Cities 的点稍微大一点
        s = 80 if city in top_cities else 40
        alpha = 0.8 if city in top_cities else 0.3
        
        plt.scatter(
            X_embedded[i, 0], 
            X_embedded[i, 1], 
            color=color,
            marker=marker,
            s=s,
            alpha=alpha,
            zorder=z,
            edgecolors='w' if city in top_cities else 'none',
            linewidth=0.5
        )
        
    # 构建图例
    from matplotlib.lines import Line2D
    
    # 1. 城市颜色图例
    legend_elements.append(Line2D([0], [0], color='w', label='--- Cities ---', marker='None'))
    for city in top_cities:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=city,
                                      markerfacecolor=city_to_color[city], markersize=10))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Other/Mix',
                                  markerfacecolor=city_to_color['Other'], markersize=8))
    
    # 2. 层级形状图例
    legend_elements.append(Line2D([0], [0], color='w', label=' ', marker='None')) # Spacer
    legend_elements.append(Line2D([0], [0], color='w', label='--- Layers ---', marker='None'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Layer 0 (Coarse)', markerfacecolor='k', markersize=8))
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Layer 1 (Fine)', markerfacecolor='k', markersize=8))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', label='Layer 2 (Detail)', markerfacecolor='k', markersize=8))

    plt.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.title("RQ-VAE Codebook Geographical Distribution\n(Color=Dominant City, Shape=Layer)", fontsize=18)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'codebook_tsne_by_city.png')
    plt.savefig(out_path, dpi=300)
    print(f"✅ Visualization saved to: {out_path}")

def main():
    os.makedirs('data/visualization', exist_ok=True)
    config = load_config('./config/config.yaml')
    
    # 1. 加载模型
    model = load_model(config)
    
    # 2. 计算映射关系
    code_city_map, top_cities = compute_dominant_cities(config)
    
    # 3. 绘图
    visualize_by_city(model, code_city_map, top_cities, 'data/visualization')

if __name__ == '__main__':
    main()