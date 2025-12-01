import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import random
import numpy as np
import faiss


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu = 0.25,
                 beta = 1, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data, max_samples=None):
        """使用 K-means 初始化 codebook
        
        Args:
            data: 输入数据 (N, e_dim)
            max_samples: 最大采样数（None表示使用全部数据）
        """
        # 如果指定了采样数且数据量大于采样数，进行随机采样
        if max_samples is not None and len(data) > max_samples:
            indices = torch.randperm(len(data))[:max_samples]
            sample_data = data[indices]
        else:
            sample_data = data
        
        centers, _ = self.constrained_km_faiss(sample_data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted = True
    
    def constrained_km_faiss(self, data, n_clusters=10):
        # 将数据从PyTorch张量转为NumPy数组
        x = data.cpu().detach().numpy().astype('float32')

        # 初始化Faiss的KMeans对象
        kmeans = faiss.Kmeans(d=x.shape[1], k=n_clusters, niter=10, nredo=10, verbose=False)

        # 训练聚类模型
        kmeans.train(x)

        # 获取聚类中心和标签
        _, labels = kmeans.index.search(x, 1)
        labels = labels.flatten()

        # 将聚类中心转换回PyTorch张量
        t_centers = torch.from_numpy(kmeans.centroids)
        t_labels = torch.from_numpy(labels).tolist()

        return t_centers, t_labels

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def vq_init(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()


        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                indices = torch.argmin(d, dim=-1)
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        return x_q
    
    def forward(self,  x, label, idx, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = 'distance'    
        
        if _distance_flag == 'distance':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()
        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                if idx != -1:
                    indices = torch.argmin(d, dim=-1)
                else:
                    temp = 1.0
                    prob_dist = F.softmax(-d/temp, dim=1)  
                    indices = torch.multinomial(prob_dist, 1).squeeze()
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)


        x_q = self.embedding(indices).view(x.shape)


        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss


        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
