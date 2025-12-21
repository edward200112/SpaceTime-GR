import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 5.1 增强型向量量化器 (VectorQuantizer)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, 
                 restart_threshold=1.0, use_restart=True):
        super().__init__()
        self.n_e = n_e      # 码本大小
        self.e_dim = e_dim  # 嵌入维度
        self.beta = beta    # 承诺损失系数
        
        # 初始化码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        # Restart 机制参数
        self.use_restart = use_restart
        self.restart_threshold = restart_threshold
        # 注册缓冲区以追踪使用率 (不参与梯度更新)
        self.register_buffer('cluster_size_ema', torch.zeros(n_e))
        self.decay = 0.99

    def forward(self, z):
        """
        z: [batch_size, e_dim]
        返回: loss, z_q, encoding_indices
        """
        # 展平输入以便计算距离
        z_flattened = z.view(-1, self.e_dim)
        
        # 1. 计算距离 ||z - e||^2
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        # 2. 获取最近邻索引
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # 3. 计算量化损失 (VQ Loss + Commitment Loss)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # 4. Straight-Through Estimator (STE)
        z_q = z + (z_q - z).detach()
        
        # 5. Codebook Restart 逻辑 (仅在训练模式下)
        if self.training and self.use_restart:
            self._codebook_restart(z_flattened, min_encoding_indices)
            
        return loss, z_q, min_encoding_indices

    def _codebook_restart(self, z_flattened, indices):
        # One-hot encoding of indices
        encodings = torch.zeros(indices.shape[0], self.n_e, device=z_flattened.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        
        # 计算当前Batch每个码字的使用次数
        avg_usage = encodings.sum(0)
        
        # 更新EMA
        self.cluster_size_ema.mul_(self.decay).add_(avg_usage * (1 - self.decay))
        
        # 识别“死码”
        dead_codes = self.cluster_size_ema < self.restart_threshold
        
        if dead_codes.any():
            n_dead = dead_codes.sum().item()
            # 从当前Batch输入中随机采样作为替换
            rand_idx = torch.randperm(z_flattened.shape[0], device=z_flattened.device)[:n_dead]
            replacements = z_flattened[rand_idx].detach()
            
            # 边界情况：如果Batch太小不够采，补随机噪声
            if replacements.shape[0] < n_dead:
                padding = torch.randn(n_dead - replacements.shape[0], self.e_dim, device=z_flattened.device)
                replacements = torch.cat([replacements, padding], dim=0)
            
            # 执行替换 (In-place update)
            with torch.no_grad():
                self.embedding.weight.data[dead_codes] = replacements
                # 重置统计量
                self.cluster_size_ema[dead_codes] = self.restart_threshold

    def calculate_diversity_loss(self, z, temp=1.0):
        """计算熵损失以鼓励均匀分布"""
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Softmax over negative distance
        probs = F.softmax(-d / temp, dim=1)
        # Batch平均概率
        avg_probs = torch.mean(probs, dim=0)
        # Maximize Entropy <=> Minimize Negative Entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        return -entropy

# ==========================================
# 5.2 层级化RQ-VAE主模型 (Hierarchical RQ-VAE)
# ==========================================
class GeoConstrainedRQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, codebook_size, 
                 num_states, num_cities):
        super().__init__()
        
        # 1. 特征融合与编码器
        # input_dim 对应预处理输出的 concat 维度 (例如 512)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim) # [batch, hidden_dim]
        )
        
        # 2. 残差量化层堆叠
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(codebook_size, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 3. 辅助预测头 (Hierarchical Geo Constraints)
        # Level 1: State (仅基于第1层量化结果)
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_states)
        )
        
        # Level 2: City (基于前2层量化结果的累加)
        self.city_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_cities)
        )
        
        # 4. 解码器 (尝试恢复原始特征)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, x, state_labels=None, city_labels=None):
        # 编码
        z = self.encoder(x)
        
        residual = z
        quantized_sum = 0
        
        total_vq_loss = 0.0
        total_div_loss = 0.0
        indices_list = []
        
        geo_loss = torch.tensor(0.0, device=x.device)
        
        # 逐层量化循环
        for i, layer in enumerate(self.vq_layers):
            # 量化当前残差
            loss, z_q, indices = layer(residual)
            
            # 计算多样性损失
            div_loss = layer.calculate_diversity_loss(residual)
            
            # 累加路径
            quantized_sum = quantized_sum + z_q
            # 更新残差
            residual = residual - z_q
            
            # 收集结果
            total_vq_loss += loss
            total_div_loss += div_loss
            indices_list.append(indices)
            
            # === 地理约束计算 (仅在训练且有标签时) ===
            if self.training and state_labels is not None:
                # Level 1 Constraint: State
                if i == 0: 
                    # 使用第一层的 z_q 预测 State
                    pred_state = self.state_predictor(z_q)
                    geo_loss += F.cross_entropy(pred_state, state_labels)
                    
                # Level 2 Constraint: City
                elif i == 1: 
                    # 使用前两层的累积和预测 City
                    # 这里的逻辑是：Layer1定大域，Layer2定小域，两者之和应能精准定位City
                    pred_city = self.city_predictor(quantized_sum)
                    geo_loss += F.cross_entropy(pred_city, city_labels)
        
        # 重构
        x_recon = self.decoder(quantized_sum)
        recon_loss = F.mse_loss(x_recon, x)
        
        return {
            'x_recon': x_recon,
            'indices': torch.stack(indices_list, dim=1), # [Batch, D]
            'recon_loss': recon_loss,
            'vq_loss': total_vq_loss,
            'div_loss': total_div_loss,
            'geo_loss': geo_loss
        }