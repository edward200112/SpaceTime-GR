import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ==========================================
# 1. 核心组件定义 (VectorQuantizer & Model)
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, 
                 restart_threshold=1.0, use_restart=True):
        super().__init__()
        self.n_e = n_e      # 码本大小 (Codebook Size)
        self.e_dim = e_dim  # 嵌入维度
        self.beta = beta    # 承诺损失系数
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        # Restart 机制
        self.use_restart = use_restart
        self.restart_threshold = restart_threshold
        self.register_buffer('cluster_size_ema', torch.zeros(n_e))
        self.decay = 0.99

    def forward(self, z):
        # z: [batch_size, e_dim]
        z_flattened = z.view(-1, self.e_dim)
        
        # 计算距离
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # Straight-Through Estimator
        z_q = z + (z_q - z).detach()
        
        if self.training and self.use_restart:
            self._codebook_restart(z_flattened, min_encoding_indices)
            
        return loss, z_q, min_encoding_indices

    def _codebook_restart(self, z_flattened, indices):
        encodings = torch.zeros(indices.shape[0], self.n_e, device=z_flattened.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_usage = encodings.sum(0)
        self.cluster_size_ema.mul_(self.decay).add_(avg_usage * (1 - self.decay))
        
        dead_codes = self.cluster_size_ema < self.restart_threshold
        if dead_codes.any():
            n_dead = dead_codes.sum().item()
            rand_idx = torch.randperm(z_flattened.shape[0], device=z_flattened.device)[:n_dead]
            replacements = z_flattened[rand_idx].detach()
            if replacements.shape[0] < n_dead:
                padding = torch.randn(n_dead - replacements.shape[0], self.e_dim, device=z_flattened.device)
                replacements = torch.cat([replacements, padding], dim=0)
            with torch.no_grad():
                self.embedding.weight.data[dead_codes] = replacements
                self.cluster_size_ema[dead_codes] = self.restart_threshold

    def calculate_diversity_loss(self, z, temp=1.0):
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        probs = F.softmax(-d / temp, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        return -entropy

class GeoConstrainedRQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, codebook_size, 
                 num_states, num_cities):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # VQ Layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(codebook_size, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Aux Predictors (Geo Constraints)
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_states)
        )
        self.city_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_cities)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, x, state_labels=None, city_labels=None):
        z = self.encoder(x)
        residual = z
        quantized_sum = 0
        total_vq_loss = 0.0
        total_div_loss = 0.0
        indices_list = []
        geo_loss = torch.tensor(0.0, device=x.device)
        
        for i, layer in enumerate(self.vq_layers):
            loss, z_q, indices = layer(residual)
            div_loss = layer.calculate_diversity_loss(residual)
            
            quantized_sum = quantized_sum + z_q
            residual = residual - z_q
            
            total_vq_loss += loss
            total_div_loss += div_loss
            indices_list.append(indices)
            
            if self.training and state_labels is not None:
                if i == 0: # Layer 1 -> State
                    pred_state = self.state_predictor(z_q)
                    geo_loss += F.cross_entropy(pred_state, state_labels)
                elif i == 1: # Layer 1+2 -> City
                    pred_city = self.city_predictor(quantized_sum)
                    geo_loss += F.cross_entropy(pred_city, city_labels)
        
        x_recon = self.decoder(quantized_sum)
        recon_loss = F.mse_loss(x_recon, x)
        
        return {
            'x_recon': x_recon,
            'indices': torch.stack(indices_list, dim=1),
            'recon_loss': recon_loss,
            'vq_loss': total_vq_loss,
            'div_loss': total_div_loss,
            'geo_loss': geo_loss
        }

# ==========================================
# 2. 训练主流程 (Training Pipeline)
# ==========================================

def train_rqvae():
    # --- 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {DEVICE}")
    
    BATCH_SIZE = 1024 
    EPOCHS = 30
    LR = 1e-3
    
    # [新增配置] 早停参数
    EARLY_STOPPING_PATIENCE = 5  # 如果连续 5 个 Epoch 验证集 Loss 不下降，则停止
    
    DATA_PATH = "./processed_data/train_data.pt"
    STATE_CLASSES = "./processed_data/state_classes.npy"
    CITY_CLASSES = "./processed_data/city_classes.npy"
    
    # --- 1. 加载数据 ---
    print("📥 Loading data...")
    # [关键修改] weights_only=False 
    data_dict = torch.load(DATA_PATH, weights_only=False)
    
    features = data_dict['features'] 
    state_ids = data_dict['state_ids']
    city_ids = data_dict['city_ids']
    gmap_ids = data_dict['gmap_ids'] 
    
    # 创建 Dataset
    full_dataset = TensorDataset(features, state_ids, city_ids)
    
    # 划分 Train (90%) / Val (10%)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📊 Dataset Split: Train={train_size}, Val={val_size}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # --- 2. 初始化模型 ---
    # 动态获取维度
    num_states = len(np.load(STATE_CLASSES, allow_pickle=True))
    num_cities = len(np.load(CITY_CLASSES, allow_pickle=True))
    input_dim = features.shape[1]
    
    print(f"🔧 Model Config: Input={input_dim}, States={num_states}, Cities={num_cities}")
    
    model = GeoConstrainedRQVAE(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,        
        codebook_size=512,   
        num_states=num_states,
        num_cities=num_cities
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- 3. 训练循环 ---
    best_val_loss = float('inf')
    epochs_no_improve = 0  # [新增] 记录未提升的次数
    
    for epoch in range(EPOCHS):
        # ==========================
        #      TRAINING PHASE
        # ==========================
        model.train()
        total_loss = 0
        total_recon = 0
        total_geo = 0
        
        lambda_geo = max(0.1, 1.0 - (epoch / (EPOCHS * 0.5)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch_x, batch_state, batch_city in pbar:
            batch_x = batch_x.to(DEVICE)
            batch_state = batch_state.to(DEVICE)
            batch_city = batch_city.to(DEVICE)
            
            optimizer.zero_grad()
            
            out = model(batch_x, state_labels=batch_state, city_labels=batch_city)
            
            l_recon = out['recon_loss']
            l_vq = out['vq_loss']
            l_div = out['div_loss']
            l_geo = out['geo_loss']
            
            loss = l_recon + 1.0 * l_vq + 0.1 * l_div + lambda_geo * l_geo
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += l_recon.item()
            total_geo += l_geo.item()
            
            pbar.set_postfix({'L': loss.item(), 'Geo': l_geo.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # ==========================
        #     VALIDATION PHASE
        # ==========================
        model.eval()
        val_loss = 0
        correct_state = 0
        correct_city = 0
        total_samples = 0
        
        print(f"🔍 Validating Epoch {epoch+1}...")
        
        with torch.no_grad():
            for bx, bs, bc in val_loader:
                bx, bs, bc = bx.to(DEVICE), bs.to(DEVICE), bc.to(DEVICE)
                
                # 1. Val Loss
                out = model(bx, state_labels=bs, city_labels=bc)
                val_loss += out['recon_loss'].item() + out['geo_loss'].item()
                
                # 2. Acc Calculation
                z = model.encoder(bx)
                _, z_q1, _ = model.vq_layers[0](z) 
                
                pred_state_ids = torch.argmax(model.state_predictor(z_q1), dim=1)
                correct_state += (pred_state_ids == bs).sum().item()
                
                residual = z - z_q1
                _, z_q2, _ = model.vq_layers[1](residual)
                z_sum_1_2 = z_q1 + z_q2
                
                pred_city_ids = torch.argmax(model.city_predictor(z_sum_1_2), dim=1)
                correct_city += (pred_city_ids == bc).sum().item()
                
                total_samples += bs.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        acc_state = (correct_state / total_samples) * 100
        acc_city = (correct_city / total_samples) * 100
        
        print(f"📉 Epoch {epoch+1} Summary:")
        print(f"   Train Loss:  {avg_train_loss:.4f}")
        print(f"   Val Loss:    {avg_val_loss:.4f}")
        print(f"   State Acc:   \033[92m{acc_state:.2f}%\033[0m") 
        print(f"   City Acc:    \033[94m{acc_city:.2f}%\033[0m") 
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Current LR: {current_lr:.6f}")
        
        # [早停逻辑]
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 # 重置计数器
            torch.save(model.state_dict(), "best_rqvae_model.pth")
            print("   💾 Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epochs.")
        
        print("-" * 50)
        
        # 触发早停
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
            break

    # --- 5. 最终推理：生成 ID (Full Inference) ---
    # 无论是跑完 EPOCHS 还是中间 break，都会执行到这里
    print("\n🎉 Training Finished! Generating Semantic IDs for ALL data...")
    model.load_state_dict(torch.load("best_rqvae_model.pth", weights_only=True)) 
    model.eval()
    
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    all_indices = []
    with torch.no_grad():
        for batch_x, _, _ in tqdm(full_loader, desc="Inference"):
            batch_x = batch_x.to(DEVICE)
            out = model(batch_x)
            all_indices.append(out['indices'].cpu()) 
            
    all_indices = torch.cat(all_indices, dim=0).numpy()
    
    print("💾 Saving ID mapping to poi_semantic_ids.csv ...")
    cols = [f'code_{i}' for i in range(4)]
    df_res = pd.DataFrame(all_indices, columns=cols)
    df_res['gmap_id'] = gmap_ids 
    df_res.to_csv("poi_semantic_ids.csv", index=False)
    print("✅ Done! File saved.")

if __name__ == "__main__":
    train_rqvae()