# HierGR-SeqRec 项目技术总结

## 核心创新点

### 1. 分层语义 ID 设计

**问题**：传统推荐系统中，Item ID 是随机分配的整数，没有语义信息。

**解决方案**：
- 使用 3 层 RQ-VAE 将商家编码为语义 ID，如 `<12, 45, 08>`
- **Cluster ID**：前两层 `<12, 45>` 表示语义簇（如"纽约高档日料"）
- **Full SID**：三层完整 ID 用于唯一标识

**优势**：
- **语义相似性**：相似商家有相似的 Cluster ID
- **在线展开**：预测 Cluster ID 后，可根据地理位置筛选具体商家
- **泛化能力**：新商家只需生成 SID，无需重新训练 LLM

---

### 2. 残差放大机制

**HierGR 的核心技术**：在 RQ-VAE 训练中使用残差放大系数 α。

```python
# 第 1 层：α=1.1，放大 10% 残差
residual = 1.1 * residual - x_res

# 第 2 层：α=1.05，放大 5% 残差
residual = 1.05 * residual - x_res

# 第 3 层：α=1.0，不放大
residual = 1.0 * residual - x_res
```

**作用**：
- 让早期层捕获主要信息（语义簇）
- 后续层处理细节差异
- 避免后续层学习退化

---

### 3. 滑动窗口 + 长期摘要

**问题**：用户历史可能很长（数百次交互），但 LLM 上下文有限。

**解决方案**：
```python
# 短期窗口（最近 10-15 个）
recent_history = interactions[-15:]

# 长期摘要（久远历史压缩）
if len(interactions) > 15:
    long_history = interactions[:-15]
    summary = generate_summary(long_history)
    # "User previously enjoyed Japanese Ramen and American BBQ"
```

**效果**：
- 保留近期意图（短期窗口）
- 记住长期偏好（摘要）
- 避免"Lost in the Middle"现象

---

### 4. 多任务训练

**任务 A：序列推荐**（主任务，70%）
```
Input: User History: [Starbucks] -> <5, 11>, [Cinema] -> <88, 21>
Output: <12, 45>
```

**任务 B：偏好摘要**（辅助任务，20%）
```
Input: User History: [Pizza 5★], [Burger 4★], [Taco 5★]
Output: The user enjoys casual fast-food dining.
```

**任务 C：ID 对齐**（辅助任务，10%）
```
Input: What is the ID for "Late night Italian pizza in NYC"?
Output: <12, 45>
```

**原理**：多任务学习强迫 LLM 同时理解：
1. ID 的语义含义
2. 用户的偏好模式
3. 序列的转移规律

---

## 技术栈对比

| 组件 | HierGR (原版) | HierGR-SeqRec (本项目) |
|------|---------------|----------------------|
| **输入** | 搜索词 (Query) | 用户历史序列 |
| **中间表示** | RQ-VAE (商品 → SID) | 同左 |
| **模型任务** | Query → Item SID | History → Next Cluster ID |
| **输出** | 单个商品 | Top-K 商品列表 |
| **地理过滤** | 无 | 基于用户位置筛选 |
| **数据集** | Product2Vec + 搜索日志 | Yelp Review |

---

## 关键超参数

### RQ-VAE 训练

```yaml
num_emb_list: [64, 64, 64]    # 64³ = 262K 种语义 ID
e_dim: 32                      # Codebook embedding 维度
alpha: [1.1, 1.05, 1.0]        # 残差放大系数
sk_epsilons: [0.0, 0.0, 0.003] # Sinkhorn 约束（只在最后一层）
epochs: 5000                   # 训练轮数
batch_size: 1024               # 批大小
```

**调参建议**：
- 商家数量 < 10K：`[32, 32, 32]`（32K 种 ID）
- 商家数量 10K-100K：`[64, 64, 64]`（262K 种 ID）
- 商家数量 > 100K：`[128, 128, 128]`（2M 种 ID）

### LLM 训练

```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
lora_r: 32                     # LoRA 秩
lora_alpha: 64                 # LoRA 缩放
epochs: 3                      # 训练轮数
batch_size: 8                  # 批大小
lr: 2.0e-5                     # 学习率
max_seq_length: 2048           # 最大序列长度
```

**调参建议**：
- 数据量 < 10K：`lora_r=16, epochs=5`
- 数据量 10K-100K：`lora_r=32, epochs=3`
- 数据量 > 100K：`lora_r=64, epochs=2`

---

## 数据流转

```
原始数据 (Yelp JSON)
    ↓
[Step 1] 商户画像构建
    ├─ business.json
    ├─ review.json
    └─ → item_profiles.jsonl
    ↓
[Step 2] RQ-VAE 训练
    ├─ BERT 编码 (768 维)
    ├─ RQ-VAE (3 层量化)
    └─ → sid_mapping.json
    ↓
[Step 3] 用户序列构建
    ├─ K-core 过滤
    ├─ 滑动窗口
    └─ → user_sequences.jsonl
    ↓
[Step 4] Prompt 构造
    ├─ 任务 A (70%)
    ├─ 任务 B (20%)
    ├─ 任务 C (10%)
    └─ → train_prompts.jsonl
    ↓
[Step 5] LLM 训练
    ├─ LoRA 微调
    └─ → llm_ckpt/
    ↓
[Step 6] 在线推理
    ├─ 用户历史 → Prompt
    ├─ LLM 预测 Cluster ID
    ├─ 展开 + 地理过滤
    └─ → Top-K 推荐
```

---

## 性能优化技巧

### 1. RQ-VAE 训练加速

**问题**：Yelp 有 150K 商家，训练慢。

**优化**：
```python
# 1. 使用 Faiss 加速 K-Means
import faiss
kmeans = faiss.Kmeans(d=embedding_dim, k=64, gpu=True)

# 2. 减少 epoch，提前停止
if collision_rate < 0.01:
    break

# 3. 使用更小的 MLP
layers: [512, 256, 128]  # 而不是 [2048, 1024, ...]
```

### 2. LLM 训练加速

```yaml
# 1. 使用 bf16（比 fp16 更稳定）
bf16: true

# 2. Gradient Checkpointing（节省显存）
gradient_checkpointing: true

# 3. 梯度累积（模拟大 batch）
batch_size: 4
gradient_accumulation_steps: 8  # 等效 batch=32

# 4. FlashAttention 2（2x 加速）
# 需要安装：pip install flash-attn
attn_implementation: "flash_attention_2"
```

### 3. 推理加速

```python
# 1. 量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True  # 8-bit 量化
)

# 2. 批量推理
prompts = [prompt1, prompt2, ...]  # 多个用户
outputs = model.generate(batch_prompts)

# 3. 缓存 Cluster Index
# 离线构建 cluster -> businesses 映射
# 在线查询时直接查表
```

---

## 与 MiniOneRec 的区别

| 维度 | MiniOneRec | HierGR-SeqRec |
|------|-----------|---------------|
| **SID 层数** | 3 层 | 3 层（相同） |
| **Cluster ID** | 无 | 有（前两层） |
| **残差放大** | 无 | 有（α > 1.0） |
| **地理信息** | 无 | 有（位置过滤） |
| **训练目标** | 预测完整 SID | 预测 Cluster ID |
| **在线展开** | 无需 | 需要（Cluster → Items） |
| **适用场景** | 电商推荐 | 本地生活推荐（LBS） |

---

## 未来改进方向

### 1. 引入强化学习

参考 MiniOneRec 的 GRPO 算法：
```python
# 奖励函数
reward = 0.7 * correctness + 0.2 * ndcg + 0.1 * diversity
```

### 2. 多模态扩展

结合商家图片：
```python
# 视觉 + 文本联合编码
image_emb = CLIP(business_image)
text_emb = BERT(business_text)
joint_emb = concat(image_emb, text_emb)
```

### 3. 动态 Cluster

当前 Cluster 是静态的（训练后固定），可改为：
```python
# 在线动态聚合
if user_location == "New York":
    cluster_12_45 = [Pizza_A, Pizza_B, ...]  # NYC
else:
    cluster_12_45 = [Pizza_C, Pizza_D, ...]  # LA
```

### 4. 多目标优化

```python
# 不只预测下一个，而是预测未来 3 个
Output: <12, 45>, <33, 10>, <88, 21>
```

---

## 总结

HierGR-SeqRec 成功地将 HierGR 的生成式检索思想应用到序列推荐任务：

1. **继承了 HierGR 的优势**：分层语义 ID、残差放大、Sinkhorn 约束
2. **针对序列推荐优化**：滑动窗口、长期摘要、多任务训练
3. **适配本地生活场景**：地理过滤、Cluster 展开、实时推荐

**适用场景**：
- 美团/大众点评：餐厅推荐
- 携程/去哪儿：酒店推荐
- 高德/百度地图：POI 推荐

**核心优势**：
- 语义可解释性（Cluster 有实际含义）
- 冷启动友好（新商家只需生成 SID）
- 可扩展性强（支持数百万商家）
