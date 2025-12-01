# Codebook 可视化文档

## 使用时机

在 **Step 2 完成后**运行，用于评估 RQ-VAE Codebook 的质量。

```bash
# Step 2 完成后
python data_processing/step2_generate_semantic_ids.py

# 运行可视化
python visualization/visualize_codebook.py
```

---

## 生成的可视化结果

所有输出保存在 `visualization/outputs/` 目录：

### 1. `codebook_tsne.png`
**三层 Codebook 的 t-SNE 降维可视化**

- 展示每层 codebook embedding 在 2D 空间的分布
- 颜色表示不同的 code index
- 如果聚类紧密，说明 codebook 学习到了清晰的语义簇

**质量指标**：
- ✅ 好：各个点分散均匀，形成多个清晰的簇
- ❌ 差：所有点挤在一起，或者只有少数几个簇

---

### 2. `codebook_umap.png`
**三层 Codebook 的 UMAP 降维可视化**

- 与 t-SNE 类似，但 UMAP 更好地保留全局结构
- 用于验证 t-SNE 的结果

---

### 3. `codebook_usage.png`
**每层 Codebook 的使用频率分布**

- 柱状图显示每个 code 被多少个商家使用
- 右上角显示使用率统计

**质量指标**：
- ✅ 好：大部分 code 都被使用，使用率 > 80%
- ⚠️  一般：使用率 50%-80%，部分 code 未被使用
- ❌ 差：使用率 < 50%，大量 code 闲置（codebook collapse）

**改进方法**：
```yaml
# 如果使用率低，调整 Sinkhorn epsilon
rqvae:
  sk_epsilons: [0.005, 0.005, 0.003]  # 在前两层也启用约束
```

---

### 4. `cluster_heatmap.png`
**Cluster ID 共现热力图**

- 行：Layer 0 的 code
- 列：Layer 1 的 code
- 颜色深度：该 Cluster 包含的商家数量

**质量指标**：
- ✅ 好：热力图呈现分散的块状结构，每个块代表一个语义簇
- ❌ 差：只有少数几个格子有颜色（大量商家集中在少数 Cluster）

---

### 5. `business_distribution.png`
**商家在语义空间的分布**

- 采样 1000 个商家，用 t-SNE 降维
- 颜色表示 Cluster ID（Layer0*100 + Layer1）

**质量指标**：
- ✅ 好：相同颜色的点聚集在一起，不同颜色分开
- ❌ 差：颜色混杂，没有明显的聚类结构

---

### 6. `quality_report.txt`
**文本格式的质量报告**

包含：
- 每层 codebook 使用率
- Collision rate（碰撞率）
- 最大和最小的 Cluster
- 平均每个 Cluster 的商家数量

**示例**：
```
============================================================
RQ-VAE Codebook Quality Report
============================================================

Layer 0:
  Total Codes: 64
  Used Codes: 62 (96.88%)
  Average Frequency: 805.56 ± 234.12
  Most Common: [(12, 1250), (45, 1100), (8, 980)]

Layer 1:
  Total Codes: 64
  Used Codes: 64 (100.00%)
  Average Frequency: 780.23 ± 189.45

Collision Analysis:
  Total Items: 50000
  Unique SIDs: 49958
  Collision Rate: 0.0840%

Cluster Analysis:
  Total Clusters: 3968
  Average Items per Cluster: 12.60
  Top 5 Largest Clusters:
    Cluster 12-45: 85 items
    Cluster 8-23: 72 items
```

---

## 依赖安装

```bash
pip install umap-learn scikit-learn seaborn
```

---

## 质量评估标准

### 优秀的 Codebook

1. **使用率 > 90%**：几乎所有 code 都被使用
2. **Collision rate < 1%**：几乎没有重复的 SID
3. **均匀分布**：每个 code 的使用频率相近（方差小）
4. **清晰聚类**：t-SNE 可视化中形成明显的簇

### 需要改进的 Codebook

1. **使用率 < 50%**：大量 code 未被使用 → 增大 Sinkhorn epsilon
2. **Collision rate > 5%**：太多商家共享相同 SID → 增大 codebook 数量
3. **分布不均**：少数 code 占据大部分商家 → 调整 alpha 系数
4. **聚类模糊**：t-SNE 中点混在一起 → 延长训练 epoch

---

## 调参建议

### 问题 1: Codebook Collapse（使用率低）

**现象**：`codebook_usage.png` 中大量柱子为 0

**解决方案**：
```yaml
rqvae:
  # 启用 Sinkhorn 约束（所有层）
  sk_epsilons: [0.01, 0.01, 0.005]
  
  # 增大训练 epoch
  epochs: 10000
```

---

### 问题 2: Collision Rate 高

**现象**：`quality_report.txt` 中 Collision Rate > 5%

**解决方案**：
```yaml
rqvae:
  # 增大 codebook 数量
  num_emb_list: [128, 128, 128]  # 从 64 增加到 128
  
  # 或增加层数
  num_emb_list: [64, 64, 64, 64]  # 4 层
```

---

### 问题 3: 聚类质量差

**现象**：`business_distribution.png` 中颜色混杂

**解决方案**：
```yaml
rqvae:
  # 调整残差放大系数（让前两层学得更好）
  alpha: [1.2, 1.1, 1.0]  # 增大前两层的放大
  
  # 增大 embedding 维度
  e_dim: 64  # 从 32 增加到 64
```

---

## 与论文对比

参考 HierGR 论文的质量指标：

| 指标 | HierGR 论文 | 你的目标 |
|------|------------|---------|
| Codebook Usage | > 95% | > 90% |
| Collision Rate | < 0.1% | < 1% |
| 聚类清晰度 | 视觉上可分 | 同左 |

---

## 示例输出

运行后会看到：

```
============================================================
RQ-VAE Codebook Visualization
============================================================

=== Loading RQ-VAE Model ===
Loaded RQ-VAE from: ./data/rqvae_ckpt/best_collision_model.pth

=== Extracting Codebook Embeddings ===
Layer 0: (64, 32)
Layer 1: (64, 32)
Layer 2: (64, 32)

=== Visualizing Codebooks with t-SNE ===
Saved to: ./visualization/outputs/codebook_tsne.png

=== Visualizing Codebooks with UMAP ===
Saved to: ./visualization/outputs/codebook_umap.png

=== Analyzing Codebook Usage ===
Saved to: ./visualization/outputs/codebook_usage.png

Layer 0:
  Total codes: 64
  Used codes: 62
  Usage rate: 96.88%
  Most common codes: [(12, 1250), (45, 1100), (8, 980)]

...

✓ All visualizations completed!
Output directory: ./visualization/outputs
============================================================
```
