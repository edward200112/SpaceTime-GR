import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,
                 alpha=None):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

        # 使用传入的 alpha 或默认值
        if alpha is None:
            self.alpha = [1.1, 1.05, 1.0] + [1.0] * (len(n_e_list) - 3)
        elif isinstance(alpha, (int, float)):
            self.alpha = [alpha] * len(n_e_list)
        else:
            # 确保 alpha 列表长度匹配层数
            self.alpha = list(alpha) + [1.0] * max(0, len(n_e_list) - len(alpha))
        
        print(f"RQ-VAE α coefficients: {self.alpha[:len(n_e_list)]}")
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def vq_ini(self, x, max_samples=20000):
        """对每一层使用残差进行 K-means 初始化"""
        x_q = 0
        residual = x
        
        # 随机抽样（如果数据量大）
        if len(x) > max_samples:
            indices = torch.randperm(len(x))[:max_samples]
            sample_data = x[indices]
            print(f"K-means Init: Sampling {max_samples}/{len(x)} items for initialization")
        else:
            sample_data = x
            print(f"K-means Init: Using all {len(x)} items for initialization")
        
        # 对每一层的残差进行初始化
        residual_sample = sample_data
        for idx, quantizer in enumerate(self.vq_layers):
            if not quantizer.initted:
                print(f"  Layer {idx}: K-means on residual (shape: {residual_sample.shape})")
                quantizer.init_emb(residual_sample)
                
                # 计算下一层的残差（用于下一层初始化）
                with torch.no_grad():
                    # 获取量化后的向量
                    z_q = quantizer.embedding.weight
                    # 为每个样本找到最近的 codebook entry
                    distances = torch.cdist(residual_sample, z_q)
                    indices = distances.argmin(dim=1)
                    x_res = quantizer.embedding(indices)
                    # 计算残差（带 alpha 放大）
                    residual_sample = self.alpha[idx] * residual_sample - x_res  

    def forward(self, x, labels, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            
            x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk)
            residual_ = self.alpha[idx]*residual - x_res 
            x_q = x_q + x_res + (1-self.alpha[idx])*residual   
            residual = residual_
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
