"""
Step 2: 语义 ID 生成 (全自动鲁棒版 - 终极修复 PyTorch 兼容性)

修复内容：
1. [NEW] 添加 Monkey Patch 修复 TypeError: ReduceLROnPlateau.__init__() unexpected keyword 'verbose'
2. 包含之前的自动 Embedding 生成、自动训练、JSON 修复
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
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 🛠️ CRITICAL FIX: PyTorch Compatibility Patch
# 解决 PyTorch 新版本移除了 ReduceLROnPlateau 的 verbose 参数导致的报错
# ==========================================
import torch.optim.lr_scheduler
_original_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau

class PatchedReduceLROnPlateau(_original_ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        # 过滤掉不再支持的 verbose 参数
        if 'verbose' in kwargs:
            kwargs.pop('verbose')
        super().__init__(optimizer, **kwargs)

# 应用补丁：替换原生的类
torch.optim.lr_scheduler.ReduceLROnPlateau = PatchedReduceLROnPlateau
# ==========================================

# Add RQ-VAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RQ-VAE'))
from models.rqvae import RQVAE
from trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset

# 解决 Tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SemanticIDGenerator:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.rqvae_config = config['rqvae']
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
        # 路径定义
        self.emb_file = os.path.join(self.data_config['embeddings_dir'], self.data_config['item_embeddings_file'])
        self.ckpt_path = os.path.join(self.data_config['rqvae_ckpt_dir'], 'best_collision_model.pth')

    def get_embeddings(self):
        """获取 Embeddings (加载或生成)"""
        if os.path.exists(self.emb_file):
            print(f"\n=== Found saved embeddings at {self.emb_file} ===")
            # PyTorch 2.6 fix
            data = torch.load(self.emb_file, weights_only=False)
            return data['embeddings'], data['business_ids'], data['metadata']
        
        print("\n=== Embeddings not found, generating from scratch... ===")
        return self._generate_embeddings_from_scratch()

    def _generate_embeddings_from_scratch(self):
        # 简化的生成逻辑，复用之前的 Geo-Fusion 代码
        print("Loading BERT...")
        bert = SentenceTransformer('all-mpnet-base-v2').to(self.device)
        
        profile_file = os.path.join(self.data_config['processed_dir'], self.data_config['item_profile_file'])
        texts, coords, business_ids, metadata = [], [], [], []
        
        with open(profile_file, 'r') as f:
            for line in tqdm(f, desc="Loading profiles"):
                p = json.loads(line)
                texts.append(p['raw_text'])
                coords.append([p.get('latitude', 0), p.get('longitude', 0)])
                business_ids.append(p['business_id'])
                metadata.append(p)
                
        print("Encoding texts...")
        text_embs = bert.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        print("Processing coords...")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        coords_weighted = scaler.fit_transform(np.array(coords)) * 10.0
        
        fused = np.hstack([text_embs, coords_weighted])
        embeddings = torch.tensor(fused, dtype=torch.float32)
        
        # 保存
        os.makedirs(self.data_config['embeddings_dir'], exist_ok=True)
        torch.save({
            'embeddings': embeddings,
            'business_ids': business_ids,
            'metadata': metadata
        }, self.emb_file)
        
        return embeddings, business_ids, metadata

    def get_model(self, input_dim):
        """获取模型 (仅加载结构，训练逻辑分离)"""
        # 构建参数 (用于初始化模型结构)
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

        if os.path.exists(self.ckpt_path):
            print(f"\n=== Found Checkpoint: {self.ckpt_path} ===")
            try:
                # PyTorch 2.6 fix + dict fix
                checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("✓ Model loaded successfully.")
                return model, True # Loaded
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                print(">>> Will retrain model from scratch.")
                return model, False # Needs training
        else:
            print(f"\n=== Checkpoint not found at {self.ckpt_path} ===")
            print(">>> Will train RQ-VAE from scratch...")
            return model, False # Needs training

    def train_loop(self, model, embeddings, args):
        """实际的训练循环"""
        dataset = TensorDataset(embeddings, torch.arange(len(embeddings)))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        trainer = Trainer(args, model)
        trainer.fit(loader)
        return model

    def generate_sids_and_save(self, model, embeddings, business_ids, metadata):
        print("\n=== Generating SIDs & Resolving Collisions ===")
        model.eval()
        model.to(self.device)
        
        # KMeans Labels
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
            if raw_tuple not in id_to_businesses: id_to_businesses[raw_tuple] = []
            id_to_businesses[raw_tuple].append((i, bid))
            
        sid_mapping = {}
        cluster_levels = self.rqvae_config['cluster_levels']
        
        for raw_tuple, items in id_to_businesses.items():
            for suffix, (idx, bid) in enumerate(items):
                # JSON Int64 Fix
                raw_list = [int(x) for x in raw_tuple]
                
                full_sid = raw_list + [int(suffix)]
                cluster_id = raw_list[:cluster_levels]
                
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
                    'sid_str': '<' + ', '.join(map(str, full_sid)) + '>',
                    'cluster_str': '<' + ', '.join(map(str, cluster_id)) + '>'
                }
                
        out_file = os.path.join(self.data_config['processed_dir'], self.data_config['sid_mapping_file'])
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(sid_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved Unique SID mapping to {out_file}")

    def run(self):
        print("\n" + "="*60)
        print("Step 2: Auto-Hybrid Mode (Load or Train)")
        print("="*60)
        
        # 1. Get Embeddings
        embeddings, business_ids, metadata = self.get_embeddings()
        
        # 2. Check Checkpoint Status
        # get_model 返回 (model, is_loaded)
        model, is_loaded = self.get_model(embeddings.shape[1])
        
        # 3. Train if needed
        if not is_loaded:
            print("\n>>> Starting Training...")
            # 构建完整的训练参数
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
                sample_strategy='all',
                lr=self.rqvae_config['lr'],
                weight_decay=self.rqvae_config['weight_decay'],
                epochs=self.rqvae_config['epochs'],
                batch_size=self.rqvae_config['batch_size'],
                eval_step=self.rqvae_config['eval_step'],
                ckpt_dir=self.data_config['rqvae_ckpt_dir'],
                num_workers=4,
                
                # 之前缺失的参数已补全
                learner='AdamW', 
                device=self.device
            )
            
            model = self.train_loop(model, embeddings, args)
            
        # 4. Generate Results
        self.generate_sids_and_save(model, embeddings, business_ids, metadata)
        print("\n✓ Step 2 Completed!")

def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    SemanticIDGenerator(config).run()

if __name__ == '__main__':
    main()