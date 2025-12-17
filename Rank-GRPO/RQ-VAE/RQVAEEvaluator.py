import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from collections import Counter
import os
from tqdm import tqdm

# ==========================================
# 1. 必须重新声明模型结构 (与训练代码一致)
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 必须注册这个 buffer，否则加载权重时会报 Unexpected key 错误
        self.register_buffer('cluster_size_ema', torch.zeros(n_e))

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = z + (z_q - z).detach()
        return None, z_q, min_encoding_indices

class GeoConstrainedRQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, codebook_size, num_states, num_cities):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(codebook_size, hidden_dim) for _ in range(num_layers)
        ])
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_states)
        )
        self.city_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_cities)
        )
        # 必须加上 decoder，尽管评估时不使用它
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, x):
        # 评估逻辑依然在 Evaluator 类中手动控制
        return self.encoder(x)

# ==========================================
# 2. 评估类定义
# ==========================================

class RQVAEEvaluator:
    def __init__(self, model, dataloader, device, output_dir="./evaluation_results"):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.model.eval()

    def run_full_evaluation(self):
        print("🚀 Starting Full Evaluation...")
        all_indices, all_z_q_layer1, all_states, all_cities = [], [], [], []
        pred_states, pred_cities = [], []

        print("Collecting inference data...")
        with torch.no_grad():
            for batch_x, batch_state, batch_city in tqdm(self.dataloader):
                batch_x = batch_x.to(self.device)
                z = self.model.encoder(batch_x)
                
                residual = z
                quantized_sum = 0
                batch_indices = []
                
                for i, layer in enumerate(self.model.vq_layers):
                    _, z_q, indices = layer(residual)
                    if i == 0:
                        all_z_q_layer1.append(z_q.cpu())
                        pred_states.append(torch.argmax(self.model.state_predictor(z_q), dim=1).cpu())
                    
                    quantized_sum += z_q
                    if i == 1:
                        pred_cities.append(torch.argmax(self.model.city_predictor(quantized_sum), dim=1).cpu())
                    
                    residual = residual - z_q
                    batch_indices.append(indices.cpu())
                
                all_indices.append(torch.stack(batch_indices, dim=1))
                all_states.append(batch_state)
                all_cities.append(batch_city)

        self.all_indices = torch.cat(all_indices, dim=0).numpy()
        self.all_z_q_layer1 = torch.cat(all_z_q_layer1, dim=0).numpy()
        self.ground_truth_states = torch.cat(all_states, dim=0).numpy()
        self.ground_truth_cities = torch.cat(all_cities, dim=0).numpy()
        self.pred_states = torch.cat(pred_states, dim=0).numpy()
        self.pred_cities = torch.cat(pred_cities, dim=0).numpy()
        
        metrics = {
            'perplexity': self._evaluate_perplexity(),
            'geo_accuracy': self._evaluate_geo_accuracy(),
            'collision_rate': self._evaluate_collision_rate()
        }
        self._visualize_tsne()
        return metrics

    def _evaluate_perplexity(self):
        print("📊 Calculating Codebook Perplexity...")
        num_layers = self.all_indices.shape[1]
        results = {}
        for d in range(num_layers):
            indices = self.all_indices[:, d]
            n_codes = self.model.vq_layers[d].n_e
            counts = np.bincount(indices, minlength=n_codes)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            ppl = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
            util = (len(probs) / n_codes) * 100
            print(f"  Layer {d+1}: PPL = {ppl:.2f}, Util = {util:.1f}%")
            results[f'layer_{d+1}'] = ppl
        return results

    def _evaluate_geo_accuracy(self):
        print("🎯 Calculating Geo-Prediction Accuracy...")
        acc_s = accuracy_score(self.ground_truth_states, self.pred_states)
        acc_c = accuracy_score(self.ground_truth_cities, self.pred_cities)
        print(f"  State Acc: {acc_s*100:.2f}%, City Acc: {acc_c*100:.2f}%")
        return {'state_acc': acc_s, 'city_acc': acc_c}

    def _evaluate_collision_rate(self):
        print("💥 Calculating ID Collision Rate...")
        id_tuples = [tuple(row) for row in self.all_indices]
        collision_rate = 1.0 - (len(set(id_tuples)) / len(id_tuples))
        print(f"  Collision Rate: {collision_rate*100:.4f}%")
        return collision_rate

    def _visualize_tsne(self, max_samples=2000):
        print("🎨 Generating t-SNE Visualization...")
        idx = np.random.choice(len(self.all_z_q_layer1), min(max_samples, len(self.all_z_q_layer1)), replace=False)
        tsne = TSNE(n_components=2, random_state=42).fit_transform(self.all_z_q_layer1[idx])
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=self.ground_truth_states[idx], palette="hsv", alpha=0.7)
        plt.title("t-SNE of Layer 1 (Colored by State)")
        plt.savefig(os.path.join(self.output_dir, "tsne_state.png"))
        plt.close()

# ==========================================
# 3. 执行脚本 (Main)
# ==========================================

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    MODEL_PATH = "./best_rqvae_model.pth"
    DATA_PATH = "./processed_data/train_data.pt"
    STATE_CLASSES = "./processed_data/state_classes.npy"
    CITY_CLASSES = "./processed_data/city_classes.npy"

    # 1. 加载元数据以获取模型参数
    print("Loading Metadata...")
    num_states = len(np.load(STATE_CLASSES, allow_pickle=True))
    num_cities = len(np.load(CITY_CLASSES, allow_pickle=True))
    
    # 2. 初始化模型
    model = GeoConstrainedRQVAE(
        input_dim=512, hidden_dim=256, num_layers=4, 
        codebook_size=512, num_states=num_states, num_cities=num_cities
    ).to(DEVICE)
    
    # 3. 加载训练好的权重
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    
    # 4. 加载验证数据
    print(f"Loading data from {DATA_PATH}...")
    data_dict = torch.load(DATA_PATH, weights_only=False)
    # 使用验证集部分数据进行评估 (假设最后10%是验证集)
    features = data_dict['features']
    val_split = int(len(features) * 0.9)
    val_dataset = TensorDataset(
        features[val_split:], 
        data_dict['state_ids'][val_split:], 
        data_dict['city_ids'][val_split:]
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


    # 5. 启动评估
    evaluator = RQVAEEvaluator(model, val_loader, DEVICE)
    results = evaluator.run_full_evaluation()
    
    # === 新增：打印最终总结 ===
    print("\n" + "="*50)
    print("✅ Evaluation Complete!")
    print(f"📄 Full Results: {results}")
    print(f"📊 t-SNE saved to: {evaluator.output_dir}/tsne_state.png")