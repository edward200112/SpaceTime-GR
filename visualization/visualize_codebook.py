"""
Codebook Visualization for RQ-VAE Quality Assessment

可视化内容：
1. 三层 Codebook Embedding 的 t-SNE/UMAP 分布
2. 每层 Codebook 的使用频率分布
3. 层级聚类热力图
4. 商家在语义空间的分布（按 Cluster ID 着色）

使用时机：在 Step 2 完成后运行
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CodebookVisualizer:
    def __init__(self, config_path='./config/config.yaml'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = self.config['data']['processed_dir']
        self.rqvae_ckpt_dir = self.config['data']['rqvae_ckpt_dir']
        self.output_dir = './visualization/outputs'
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load checkpoint and SID mapping
        self.model = self.load_rqvae_model()
        self.sid_mapping = self.load_sid_mapping()
        
        print(f"Loaded {len(self.sid_mapping)} items with SIDs")
    
    def load_rqvae_model(self):
        """加载训练好的 RQ-VAE 模型"""
        print("\n=== Loading RQ-VAE Model ===")
        
        # Add RQ-VAE to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'RQ-VAE'))
        from models.rqvae import RQVAE
        
        # Model config
        rqvae_config = self.config['rqvae']
        model = RQVAE(
            in_dim=rqvae_config['in_dim'],
            num_emb_list=rqvae_config['num_emb_list'],
            e_dim=rqvae_config['e_dim'],
            layers=rqvae_config['layers'],
            dropout_prob=rqvae_config['dropout_prob'],
            loss_type=rqvae_config['loss_type'],
            quant_loss_weight=rqvae_config['quant_loss_weight'],
            kmeans_init=rqvae_config['kmeans_init'],
            kmeans_iters=rqvae_config['kmeans_iters'],
            sk_epsilons=rqvae_config['sk_epsilons'],
            sk_iters=rqvae_config['sk_iters'],
            alpha=rqvae_config['alpha'],
            beta=rqvae_config['beta']
        )
        
        # Load checkpoint
        ckpt_path = os.path.join(self.rqvae_ckpt_dir, 'best_collision_model.pth')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"Loaded RQ-VAE from: {ckpt_path}")
        return model
    
    def load_sid_mapping(self):
        """加载 SID 映射"""
        mapping_file = os.path.join(
            self.processed_dir,
            self.config['data']['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        return sid_mapping
    
    def extract_codebook_embeddings(self):
        """提取三层 Codebook 的 Embeddings"""
        print("\n=== Extracting Codebook Embeddings ===")
        
        codebooks = []
        for idx, vq_layer in enumerate(self.model.rq.vq_layers):
            codebook = vq_layer.get_codebook().detach().cpu().numpy()
            codebooks.append(codebook)
            print(f"Layer {idx}: {codebook.shape}")
        
        return codebooks
    
    def visualize_codebook_tsne(self, codebooks, perplexity=30):
        """使用 t-SNE 可视化三层 Codebook"""
        print("\n=== Visualizing Codebooks with t-SNE ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (codebook, ax) in enumerate(zip(codebooks, axes)):
            # t-SNE 降维
            if codebook.shape[0] > perplexity:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                embeddings_2d = tsne.fit_transform(codebook)
            else:
                # 如果 codebook 太小，使用 PCA
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(codebook)
            
            # 绘制散点图
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=np.arange(len(codebook)),
                cmap='tab20',
                s=100,
                alpha=0.7
            )
            
            ax.set_title(f'Layer {idx} Codebook (n={len(codebook)})', fontsize=14)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            
            # 标注部分点
            if len(codebook) <= 64:
                for i, (x, y) in enumerate(embeddings_2d):
                    ax.annotate(str(i), (x, y), fontsize=6, alpha=0.5)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'codebook_tsne.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    
    def visualize_codebook_umap(self, codebooks, n_neighbors=15):
        """使用 UMAP 可视化三层 Codebook"""
        print("\n=== Visualizing Codebooks with UMAP ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (codebook, ax) in enumerate(zip(codebooks, axes)):
            # UMAP 降维
            if codebook.shape[0] > n_neighbors:
                reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
                embeddings_2d = reducer.fit_transform(codebook)
            else:
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(codebook)
            
            # 绘制散点图
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=np.arange(len(codebook)),
                cmap='tab20',
                s=100,
                alpha=0.7
            )
            
            ax.set_title(f'Layer {idx} Codebook (n={len(codebook)})', fontsize=14)
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'codebook_umap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    
    def analyze_codebook_usage(self):
        """分析每层 Codebook 的使用频率"""
        print("\n=== Analyzing Codebook Usage ===")
        
        # 统计每层的使用频率
        layer_counters = [Counter(), Counter(), Counter()]
        
        for business_id, info in self.sid_mapping.items():
            sid = info['full_sid']  # 使用 'full_sid' 而不是 'sid'
            for layer_idx, code in enumerate(sid):
                layer_counters[layer_idx][code] += 1
        
        # 绘制使用频率分布
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (counter, ax) in enumerate(zip(layer_counters, axes)):
            codes = sorted(counter.keys())
            frequencies = [counter[c] for c in codes]
            
            ax.bar(codes, frequencies, alpha=0.7, color=f'C{idx}')
            ax.set_title(f'Layer {idx} Codebook Usage Distribution', fontsize=14)
            ax.set_xlabel('Code Index')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 统计信息
            total = sum(frequencies)
            used_codes = len([f for f in frequencies if f > 0])
            total_codes = self.config['rqvae']['num_emb_list'][idx]
            usage_rate = used_codes / total_codes * 100
            
            ax.text(
                0.95, 0.95,
                f'Used: {used_codes}/{total_codes} ({usage_rate:.1f}%)\n'
                f'Total items: {total:,}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10
            )
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'codebook_usage.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
        
        # 打印详细统计
        print("\n=== Codebook Usage Statistics ===")
        for idx, counter in enumerate(layer_counters):
            print(f"\nLayer {idx}:")
            print(f"  Total codes: {self.config['rqvae']['num_emb_list'][idx]}")
            print(f"  Used codes: {len(counter)}")
            print(f"  Usage rate: {len(counter)/self.config['rqvae']['num_emb_list'][idx]*100:.2f}%")
            print(f"  Most common codes: {counter.most_common(5)}")
    
    def visualize_cluster_heatmap(self):
        """可视化 Cluster ID (前两层) 的共现矩阵"""
        print("\n=== Visualizing Cluster Co-occurrence ===")
        
        num_codes_layer0 = self.config['rqvae']['num_emb_list'][0]
        num_codes_layer1 = self.config['rqvae']['num_emb_list'][1]
        
        # 构建共现矩阵
        cooccurrence = np.zeros((num_codes_layer0, num_codes_layer1))
        
        for business_id, info in self.sid_mapping.items():
            c0, c1 = info['cluster_id']
            cooccurrence[c0, c1] += 1
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cooccurrence,
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Items'},
            xticklabels=5,
            yticklabels=5,
            square=False
        )
        plt.title('Cluster ID Co-occurrence Matrix (Layer 0 vs Layer 1)', fontsize=14)
        plt.xlabel('Layer 1 Code')
        plt.ylabel('Layer 0 Code')
        
        output_path = os.path.join(self.output_dir, 'cluster_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    
    def visualize_business_distribution(self, sample_size=1000):
        """可视化商家在语义空间的分布（按 Cluster ID 着色）"""
        print("\n=== Visualizing Business Distribution ===")
        
        # 采样商家
        business_ids = list(self.sid_mapping.keys())
        if len(business_ids) > sample_size:
            import random
            business_ids = random.sample(business_ids, sample_size)
        
        # 提取商家的 embedding 和 cluster_id
        embeddings_file = os.path.join(
            self.config['data']['embeddings_dir'],
            self.config['data']['item_embeddings_file']
        )
        
        embeddings_data = torch.load(embeddings_file, weights_only=False)
        all_embeddings = embeddings_data['embeddings']
        all_business_ids = embeddings_data['business_ids']
        
        # 创建 business_id 到 embedding 的映射
        embeddings_dict = {bid: all_embeddings[i] for i, bid in enumerate(all_business_ids)}
        
        embeddings = []
        cluster_labels = []
        
        for business_id in business_ids:
            if business_id in embeddings_dict and business_id in self.sid_mapping:
                embeddings.append(embeddings_dict[business_id].cpu().numpy())
                c0, c1 = self.sid_mapping[business_id]['cluster_id']
                cluster_labels.append(c0 * 100 + c1)  # 组合标签
        
        embeddings = np.array(embeddings)
        cluster_labels = np.array(cluster_labels)
        
        # t-SNE 降维
        print(f"Running t-SNE on {len(embeddings)} businesses...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=cluster_labels,
            cmap='tab20',
            s=30,
            alpha=0.6
        )
        
        plt.colorbar(scatter, label='Cluster ID (Layer0*100 + Layer1)')
        plt.title(f'Business Distribution in Semantic Space (n={len(embeddings)})', fontsize=14)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'business_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    
    def generate_quality_report(self):
        """生成 Codebook 质量报告"""
        print("\n=== Generating Quality Report ===")
        
        # 统计信息
        layer_counters = [Counter(), Counter(), Counter()]
        
        for business_id, info in self.sid_mapping.items():
            sid = info['full_sid']  # 使用 'full_sid' 而不是 'sid'
            for layer_idx, code in enumerate(sid):
                layer_counters[layer_idx][code] += 1
        
        report = []
        report.append("=" * 60)
        report.append("RQ-VAE Codebook Quality Report")
        report.append("=" * 60)
        report.append("")
        
        # 每层统计
        for idx, counter in enumerate(layer_counters):
            total_codes = self.config['rqvae']['num_emb_list'][idx]
            used_codes = len(counter)
            usage_rate = used_codes / total_codes * 100
            
            frequencies = list(counter.values())
            avg_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            
            report.append(f"Layer {idx}:")
            report.append(f"  Total Codes: {total_codes}")
            report.append(f"  Used Codes: {used_codes} ({usage_rate:.2f}%)")
            report.append(f"  Average Frequency: {avg_freq:.2f} ± {std_freq:.2f}")
            report.append(f"  Most Common: {counter.most_common(3)}")
            report.append(f"  Least Common: {counter.most_common()[-3:]}")
            report.append("")
        
        # Collision 分析
        full_sids = set()
        for business_id, info in self.sid_mapping.items():
            sid_str = '-'.join(map(str, info['full_sid']))  # 使用 'full_sid'
            full_sids.add(sid_str)
        
        collision_rate = (len(self.sid_mapping) - len(full_sids)) / len(self.sid_mapping)
        
        report.append(f"Collision Analysis:")
        report.append(f"  Total Items: {len(self.sid_mapping)}")
        report.append(f"  Unique SIDs: {len(full_sids)}")
        report.append(f"  Collision Rate: {collision_rate*100:.4f}%")
        report.append("")
        
        # Cluster 统计
        cluster_counter = Counter()
        for business_id, info in self.sid_mapping.items():
            cluster_str = '-'.join(map(str, info['cluster_id']))
            cluster_counter[cluster_str] += 1
        
        report.append(f"Cluster Analysis:")
        report.append(f"  Total Clusters: {len(cluster_counter)}")
        report.append(f"  Average Items per Cluster: {np.mean(list(cluster_counter.values())):.2f}")
        report.append(f"  Top 5 Largest Clusters:")
        for cluster, count in cluster_counter.most_common(5):
            report.append(f"    Cluster {cluster}: {count} items")
        report.append("")
        
        report.append("=" * 60)
        
        # 保存报告
        report_text = '\n'.join(report)
        print(report_text)
        
        report_path = os.path.join(self.output_dir, 'quality_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {report_path}")
    
    def run(self):
        """运行所有可视化"""
        print("\n" + "="*60)
        print("RQ-VAE Codebook Visualization")
        print("="*60)
        
        # 1. 提取 Codebook Embeddings
        codebooks = self.extract_codebook_embeddings()
        
        # 2. t-SNE 可视化
        self.visualize_codebook_tsne(codebooks)
        
        # 3. UMAP 可视化
        self.visualize_codebook_umap(codebooks)
        
        # 4. 使用频率分析
        self.analyze_codebook_usage()
        
        # 5. Cluster 共现热力图
        self.visualize_cluster_heatmap()
        
        # 6. 商家分布可视化
        self.visualize_business_distribution()
        
        # 7. 质量报告
        self.generate_quality_report()
        
        print("\n" + "="*60)
        print("✓ All visualizations completed!")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    visualizer = CodebookVisualizer()
    visualizer.run()


if __name__ == '__main__':
    main()
