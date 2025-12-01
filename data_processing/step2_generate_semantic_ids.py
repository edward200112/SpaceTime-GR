"""
Step 2: 语义 ID 生成 (Semantic ID Generation)

目标：训练 RQ-VAE 并生成分层语义 ID
输入：item_profiles.jsonl
输出：
  - item_embeddings.pt (商家的语义向量)
  - sid_mapping.json (business_id -> SID 映射)
  - RQ-VAE checkpoint

流程：
1. 使用 BERT 将 raw_text 转为 768 维向量
2. 训练 RQ-VAE (3层，每层64/128个codebook)
3. 生成 SID，定义前两层为 Cluster ID
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer
import sys

# Add RQ-VAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))


class SemanticIDGenerator:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.rqvae_config = config['rqvae']
        
        self.processed_dir = self.data_config['processed_dir']
        self.embeddings_dir = self.data_config['embeddings_dir']
        self.rqvae_ckpt_dir = self.data_config['rqvae_ckpt_dir']
        
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.rqvae_ckpt_dir, exist_ok=True)
        
        # BERT model for text encoding
        self.bert_model = None
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    def load_bert_model(self):
        """加载 BERT 模型用于文本编码"""
        print("\n=== Loading BERT Model ===")
        # 使用本地下载的模型
        model_name = './all-mpnet-base-v2'  # 768-dim 本地模型路径
        
        self.bert_model = SentenceTransformer(model_name)
        self.bert_model.to(self.device)
        print(f"Loaded BERT model: {model_name}")
        print(f"Embedding dimension: {self.bert_model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self):
        """生成商家的语义向量"""
        print("\n=== Generating Item Embeddings ===")
        
        # Load item profiles
        profile_file = os.path.join(
            self.processed_dir,
            self.data_config['item_profile_file']
        )
        
        business_ids = []
        texts = []
        metadata = []
        
        with open(profile_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading profiles"):
                profile = json.loads(line.strip())
                business_ids.append(profile['business_id'])
                texts.append(profile['raw_text'])
                metadata.append({
                    'business_id': profile['business_id'],
                    'name': profile['name'],
                    'city': profile['city'],
                    'categories': profile['categories'],
                    'latitude': profile['latitude'],
                    'longitude': profile['longitude']
                })
        
        print(f"Loaded {len(texts)} items")
        
        # Encode with BERT
        print("Encoding texts with BERT...")
        batch_size = 64
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.bert_model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )
            all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Save embeddings
        embeddings_file = os.path.join(
            self.embeddings_dir,
            self.data_config['item_embeddings_file']
        )
        torch.save({
            'embeddings': all_embeddings,
            'business_ids': business_ids,
            'metadata': metadata
        }, embeddings_file)
        
        print(f"Saved embeddings to {embeddings_file}")
        
        return all_embeddings, business_ids, metadata
    
    def train_rqvae(self, embeddings):
        """训练 RQ-VAE"""
        print("\n=== Training RQ-VAE ===")
        
        # Import RQ-VAE components
        from models.rqvae import RQVAE
        from trainer import Trainer
        from torch.utils.data import DataLoader
        import argparse
        
        # Prepare dataset
        # Save embeddings as .npy for RQ-VAE trainer
        temp_emb_file = os.path.join(self.embeddings_dir, 'temp_embeddings.npy')
        np.save(temp_emb_file, embeddings.numpy())
        
        # Create args for RQ-VAE
        args = argparse.Namespace(
            lr=self.rqvae_config['lr'],
            epochs=self.rqvae_config['epochs'],
            batch_size=self.rqvae_config['batch_size'],
            eval_step=self.rqvae_config['eval_step'],
            num_workers=self.config['hardware']['num_workers'],
            learner=self.rqvae_config.get('optimizer', 'AdamW'),
            data_path=self.embeddings_dir,
            weight_decay=self.rqvae_config['weight_decay'],
            dropout_prob=self.rqvae_config['dropout_prob'],
            bn=False,
            loss_type=self.rqvae_config['loss_type'],
            kmeans_init=self.rqvae_config['kmeans_init'],
            kmeans_iters=self.rqvae_config['kmeans_iters'],
            sk_epsilons=self.rqvae_config['sk_epsilons'],
            sk_iters=self.rqvae_config.get('sk_iters', 50),
            device=self.config['hardware']['device'],
            num_emb_list=self.rqvae_config['num_emb_list'],
            e_dim=self.rqvae_config['e_dim'],
            quant_loss_weight=self.rqvae_config['quant_loss_weight'],
            alpha=self.rqvae_config.get('alpha', 0.1),
            beta=self.rqvae_config['beta'],
            n_clusters=self.rqvae_config['n_clusters'],
            sample_strategy='all',
            layers=self.rqvae_config['layers'],
            ckpt_dir=self.rqvae_ckpt_dir
        )
        
        # Update in_dim based on actual embedding dimension
        actual_dim = embeddings.shape[1]
        
        # Build model
        model = RQVAE(
            in_dim=actual_dim,
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
        
        print(f"RQ-VAE Model:\n{model}")
        
        # Prepare data loader
        # Note: Need to create a simple dataset wrapper
        dataset = torch.utils.data.TensorDataset(
            embeddings,
            torch.arange(len(embeddings))  # dummy indices
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Train
        trainer = Trainer(args, model)
        best_loss, best_collision_rate = trainer.fit(data_loader)
        
        print(f"\nTraining completed!")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Best Collision Rate: {best_collision_rate:.4f}")
        
        return model
    
    def generate_sids(self, model, embeddings, business_ids, metadata):
        """生成语义 ID 并保存映射"""
        print("\n=== Generating Semantic IDs ===")
        
        model.eval()
        model.to(self.device)
        
        # Prepare labels (dummy, needed by model)
        embs = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
        labels = {}
        for idx, emb in enumerate(embs):
            # Simple clustering for labels
            from sklearn.cluster import KMeans
            n_clusters = min(10, len(emb))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(emb)
            labels[str(idx)] = cluster_labels.tolist()
        
        # Generate indices
        batch_size = 256
        all_indices = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Generating SIDs"):
                batch = embeddings[i:i+batch_size].to(self.device)
                indices = model.get_indices(batch, labels, use_sk=False)
                all_indices.append(indices.cpu())
        
        all_indices = torch.cat(all_indices, dim=0).numpy()
        print(f"Generated SIDs shape: {all_indices.shape}")
        
        # Build mapping
        sid_mapping = {}
        cluster_levels = self.rqvae_config['cluster_levels']
        
        for i, business_id in enumerate(business_ids):
            sid = all_indices[i].tolist()
            cluster_id = sid[:cluster_levels]  # 前两层作为 Cluster ID
            
            sid_mapping[business_id] = {
                'name': metadata[i]['name'],
                'city': metadata[i]['city'],
                'categories': metadata[i]['categories'],
                'latitude': metadata[i]['latitude'],
                'longitude': metadata[i]['longitude'],
                'full_sid': sid,
                'cluster_id': cluster_id,
                'sid_str': '<' + ', '.join(map(str, sid)) + '>',
                'cluster_str': '<' + ', '.join(map(str, cluster_id)) + '>'
            }
        
        # Save mapping
        mapping_file = os.path.join(
            self.processed_dir,
            self.data_config['sid_mapping_file']
        )
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(sid_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Saved SID mapping to {mapping_file}")
        
        # Statistics
        unique_clusters = len(set(tuple(v['cluster_id']) for v in sid_mapping.values()))
        unique_sids = len(set(tuple(v['full_sid']) for v in sid_mapping.values()))
        
        print(f"\nStatistics:")
        print(f"  Total items: {len(sid_mapping)}")
        print(f"  Unique Cluster IDs: {unique_clusters}")
        print(f"  Unique Full SIDs: {unique_sids}")
        print(f"  Collision rate: {1 - unique_sids/len(sid_mapping):.4f}")
    
    def run(self):
        """执行完整流程"""
        print("\n" + "="*60)
        print("Step 2: Generating Semantic IDs")
        print("="*60)
        
        # 1. Load BERT
        self.load_bert_model()
        
        # 2. Generate embeddings
        embeddings, business_ids, metadata = self.generate_embeddings()
        
        # 3. Train RQ-VAE
        model = self.train_rqvae(embeddings)
        
        # 4. Generate SIDs
        self.generate_sids(model, embeddings, business_ids, metadata)
        
        print("\n✓ Step 2 completed successfully!")


def main():
    # Load config
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Generate semantic IDs
    generator = SemanticIDGenerator(config)
    generator.run()


if __name__ == '__main__':
    main()
