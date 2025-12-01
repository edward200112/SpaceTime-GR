"""
RQ-VAE Trainer (Simplified for HierGR-SeqRec)
"""

import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm
import os


class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = self._setup_logger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.labels = {"0": [], "1": [], "2": [], "3": []}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        
        # 早停机制
        self.patience = 1000  # 1000 epochs 没有改善就停止
        self.counter = 0
        self.early_stop = False
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.model = self.model.to(self.device)
    
    def _setup_logger(self):
        """配置 logger 同时输出到控制台和文件"""
        logger = logging.getLogger('RQ-VAE-Trainer')
        logger.setLevel(logging.INFO)
        
        # 清除已有的 handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (可选)
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{log_dir}/rqvae_training.log', mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        # 使用 CosineAnnealingWarmRestarts 或 ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,      # 每次衰减到 50%
            patience=500,    # 500 epochs 没有改善就降低学习率
            verbose=True,
            min_lr=1e-6      # 最小学习率
        )
        return scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        """使用sklearn的KMeans进行聚类"""
        from sklearn.cluster import KMeans
        x = data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(x)
        t_centers = torch.from_numpy(kmeans.cluster_centers_)
        t_labels = kmeans.labels_.tolist()
        return t_centers, t_labels

    def vq_init(self, data_loader):
        """使用训练数据初始化向量量化器（K-means）"""
        self.model.eval()
        print("\n=== VQ K-means Initialization ===")
        
        # 收集所有数据的 encoder 输出
        all_encoded = []
        with torch.no_grad():
            for batch_data in data_loader:
                data = batch_data[0].to(self.device)
                encoded = self.model.encoder(data)
                all_encoded.append(encoded)
        
        all_encoded = torch.cat(all_encoded, dim=0)
        print(f"Total encoded data: {all_encoded.shape}")
        
        # 调用 RQ 的 K-means 初始化（会自动处理残差）
        self.model.rq.vq_ini(all_encoded, max_samples=20000)
        print("VQ K-means initialization completed!\n")

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        # 动态聚类（减少频率以提高稳定性）
        # 前 1000 epochs：每 10 epochs 聚类一次
        # 1000-3000 epochs：每 50 epochs 聚类一次
        # 3000+ epochs：每 100 epochs 聚类一次或不聚类
        should_cluster = False
        if epoch_idx < 1000:
            should_cluster = (epoch_idx % 10 == 0)
        elif epoch_idx < 3000:
            should_cluster = (epoch_idx % 50 == 0)
        else:
            should_cluster = (epoch_idx % 100 == 0)
        
        if should_cluster or epoch_idx == 0:
            embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=f"Train {epoch_idx}")

        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out = self.model(data, self.labels)

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, emb_idx, dense_out, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, 0, total_quant_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        iter_data = tqdm(valid_data, total=len(valid_data), ncols=100, desc=f"Evaluate")
        indices_set = set()

        num_sample = 0
        embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(f"Saving current: {ckpt_path}")

    def fit(self, data):
        self.vq_init(data)
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_time = training_end_time - training_start_time
            
            # 每个epoch都输出训练信息
            print(f"\n[Epoch {epoch_idx}] Loss: {train_loss:.4f} | Recon: {train_recon_loss:.4f} | Quant: {quant_loss:.4f} | Time: {train_time:.2f}s")
            
            self.logger.info(
                f"epoch {epoch_idx} training [time: {train_time:.2f}s, "
                f"train loss: {train_loss:.4f}, recon loss: {train_recon_loss:.4f}]"
            )

            # 更新最佳 loss 和早停计数
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.counter = 0  # 重置早停计数器
            else:
                self.counter += 1
            
            # 学习率调度（基于 loss）
            self.scheduler.step(train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != self.lr:  # 学习率发生变化时打印
                print(f"  Learning rate changed: {self.lr:.6f} → {current_lr:.6f}")
            
            # 早停检查
            if self.counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered after {self.patience} epochs without improvement")
                self.early_stop = True

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate, ckpt_file=self.best_collision_ckpt)

                valid_end_time = time()
                
                # 打印评估结果（重要）
                print(f"\n{'='*60}")
                print(f"[Epoch {epoch_idx}] EVALUATION")
                print(f"  Collision Rate: {collision_rate:.4f} (Best: {self.best_collision_rate:.4f})")
                print(f"  Eval Time: {valid_end_time - valid_start_time:.2f}s")
                print(f"{'='*60}\n")
                
                self.logger.info(
                    f"epoch {epoch_idx} evaluating [time: {valid_end_time - valid_start_time:.2f}s, "
                    f"collision_rate: {collision_rate:.4f}]"
                )
            
            # 检查早停
            if self.early_stop:
                print(f"\n✅ Training stopped early at epoch {epoch_idx}")
                print(f"   Best Loss: {self.best_loss:.4f}")
                print(f"   Best Collision Rate: {self.best_collision_rate:.4f}")
                break

        return self.best_loss, self.best_collision_rate
