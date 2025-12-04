"""
Step 2: 语义 ID 生成 (最终修复版 v3)

功能：
1. 加载已保存的 Embeddings (跳过 BERT)
2. 加载已训练的 RQ-VAE Checkpoint (修复 Checkpoint 结构加载问题)
3. 修复 JSON int64 报错
"""

import json
import os
import torch
import numpy as np
import yaml
import sys
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add RQ-VAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))
from models.rqvae import RQVAE

class SemanticIDGenerator:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.rqvae_config = config['rqvae']
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
    def load_saved_data(self):
        """直接加载之前保存的 Embeddings，跳过 BERT"""
        print("\n=== Loading Saved Embeddings (Skipping BERT) ===")
        emb_file = os.path.join(
            self.data_config['embeddings_dir'],
            self.data_config['item_embeddings_file']
        )
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"找不到 {emb_file}，请检查是否跑过 Step 2 前半部分")
        
        # 加载 embeddings
        data = torch.load(emb_file, weights_only=False)
        embeddings = data['embeddings']
        business_ids = data['business_ids']
        metadata = data['metadata']
        
        print(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings, business_ids, metadata

    def load_trained_model(self, input_dim):
        """加载训练好的 RQ-VAE 模型"""
        print("\n=== Loading Trained RQ-VAE Checkpoint ===")
        
        # 1. 重建参数
        args = argparse.Namespace(
            num_emb_list=self.rqvae_config['num_emb_list'],
            e_dim=self.rqvae_config['e_dim'],
            layers=self.rqvae_config['layers'],
            dropout_prob=self.rqvae_config['dropout_prob'],
            bn=False,
            loss_type=self.rqvae_config['loss_type'],
            quant_loss_weight=self.rqvae_config['quant_loss_weight'],
            kmeans_init=True,
            kmeans_iters=self.rqvae_config.get('kmeans_iters', 100),
            sk_epsilons=self.rqvae_config['sk_epsilons'],
            sk_iters=self.rqvae_config.get('sk_iters', 50),
            beta=self.rqvae_config['beta'],
            alpha=self.rqvae_config.get('alpha', 1.0),
            n_clusters=self.rqvae_config['n_clusters'],
            sample_strategy='all'
        )
        
        # 2. 初始化模型
        model = RQVAE(
            in_dim=input_dim,
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
        
        # 3. 加载权重 (修复结构问题)
        ckpt_path = os.path.join(self.data_config['rqvae_ckpt_dir'], 'best_collision_model.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")
            
        print(f"Loading weights from: {ckpt_path}")
        
        # 加载 checkpoint 字典
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # [关键修复] 提取 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Detected full checkpoint dict, extracting 'state_dict'...")
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model

    def generate_sids(self, model, embeddings, business_ids, metadata):
        """生成 ID (包含 int64 修复)"""
        print("\n=== Generating SIDs & Resolving Collisions ===")
        
        model.eval()
        model.to(self.device)
        
        # K-Means Labels
        embs_weight = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
        labels = {}
        for idx, emb in enumerate(embs_weight):
            n_clusters = min(10, len(emb))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(emb)
            labels[str(idx)] = cluster_labels.tolist()
        
        # Inference
        batch_size = 2048 
        all_indices = []
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Inferencing"):
                batch = embeddings[i:i+batch_size].to(self.device)
                indices = model.get_indices(batch, labels, use_sk=False)
                all_indices.append(indices.cpu())
        
        all_indices = torch.cat(all_indices, dim=0).numpy()
        
        # Collision Resolution
        id_to_businesses = {}
        for i, (bid, raw_idx) in enumerate(zip(business_ids, all_indices)):
            raw_tuple = tuple(raw_idx)
            if raw_tuple not in id_to_businesses:
                id_to_businesses[raw_tuple] = []
            id_to_businesses[raw_tuple].append((i, bid))
            
        sid_mapping = {}
        cluster_levels = self.rqvae_config['cluster_levels']
        
        for raw_tuple, items in id_to_businesses.items():
            for suffix, (idx, bid) in enumerate(items):
                # Python int conversion fix
                raw_list = [int(x) for x in raw_tuple]
                
                full_sid = raw_list + [int(suffix)]
                cluster_id = raw_list[:cluster_levels]
                
                sid_str = '<' + ', '.join(map(str, full_sid)) + '>'
                cluster_str = '<' + ', '.join(map(str, cluster_id)) + '>'
                
                sid_mapping[bid] = {
                    'name': metadata[idx]['name'],
                    'city': metadata[idx]['city'],
                    'categories': metadata[idx]['categories'],
                    'latitude': metadata[idx]['latitude'],
                    'longitude': metadata[idx]['longitude'],
                    'raw_codes': raw_list,   
                    'suffix': int(suffix),   
                    'full_sid': full_sid, 
                    'cluster_id': cluster_id,
                    'sid_str': sid_str,
                    'cluster_str': cluster_str
                }
        
        # Save
        mapping_file = os.path.join(self.data_config['processed_dir'], self.data_config['sid_mapping_file'])
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(sid_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Success! Saved Unique SID mapping to {mapping_file}")

    def run(self):
        print("\n" + "="*60)
        print("Step 2: FAST RECOVERY MODE v3 (Checkpoint Fix)")
        print("="*60)
        
        # 1. Load Data
        embeddings, business_ids, metadata = self.load_saved_data()
        
        # 2. Load Model
        model = self.load_trained_model(input_dim=embeddings.shape[1])
        
        # 3. Generate & Save
        self.generate_sids(model, embeddings, business_ids, metadata)
        
        print("\n✓ Step 2 completed successfully!")

def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    generator = SemanticIDGenerator(config)
    generator.run()

if __name__ == '__main__':
    main()