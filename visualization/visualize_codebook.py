"""
Visualize RQ-VAE Codebooks using t-SNE (兼容性修复版)
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))

from models.rqvae import RQVAE

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(config):
    """加载 RQ-VAE 模型权重"""
    rq_conf = config['rqvae']
    data_conf = config['data']
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # 构造参数
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
    
    # 初始化模型
    # 注意：in_dim 必须与训练时一致 (770)
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
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Check your checkpoint path!")
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def visualize_codebooks(model, output_dir):
    print("Extracting Codebook weights...")
    
    all_codes = []
    layer_labels = []
    
    # RQ-VAE 的 Codebook 存储在 model.rq.vq_layers[i].embedding.weight 中
    for i, layer in enumerate(model.rq.vq_layers):
        # weights shape: [vocab_size, e_dim]
        weights = layer.embedding.weight.detach().cpu().numpy()
        all_codes.append(weights)
        # 为每个点打上层级标签
        layer_labels.extend([i] * len(weights))
        print(f"  Layer {i}: {weights.shape}")
        
    # 合并所有层的 Code 进行统一 t-SNE
    X = np.vstack(all_codes)
    y = np.array(layer_labels)
    
    print(f"Running t-SNE on {X.shape[0]} vectors (dim={X.shape[1]})...")
    
    # [FIX] 移除 n_iter 和 learning_rate，增强兼容性
    # 默认 n_iter=1000
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
        X_embedded = tsne.fit_transform(X)
    except TypeError:
        # 如果还是报错，尝试最简参数
        print("⚠️ Warning: Standard TSNE init failed, trying minimal args...")
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF'] # 不同层的颜色
    markers = ['o', 's', '^', 'D']
    
    for i in range(len(model.rq.vq_layers)):
        mask = (y == i)
        plt.scatter(
            X_embedded[mask, 0], 
            X_embedded[mask, 1], 
            c=colors[i % len(colors)], 
            label=f'Layer {i} (Codebook)',
            alpha=0.7,
            s=50,
            marker=markers[i % len(markers)]
        )
        
    plt.title("t-SNE Visualization of RQ-VAE Codebooks", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(output_dir, "codebook_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    # 简单的统计诊断
    print("\n=== Codebook Health Check ===")
    for i in range(len(all_codes)):
        layer_codes = all_codes[i]
        # 计算两两距离的平均值
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(layer_codes)
        # 排除对角线
        avg_dist = np.sum(dists) / (dists.shape[0] * (dists.shape[0]-1))
        print(f"Layer {i} Avg Distance: {avg_dist:.4f} (Larger is better, near 0 means collapse)")

def main():
    os.makedirs('data/visualization', exist_ok=True)
    config = load_config('./config/config.yaml')
    model = load_model(config)
    visualize_codebooks(model, 'data/visualization')

if __name__ == '__main__':
    main()