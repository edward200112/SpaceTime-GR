# 连锁店碰撞问题解决方案

## 问题诊断

### 现象
- **Collision Rate: 61.89%** - 远高于预期（目标 < 5%）
- 150,346 个商户只产生 57,300 个唯一 SID
- 平均每个 SID 对应 2.6 个商户

### 可能的根本原因

#### 🔴 **原因 1：连锁店导致的文本描述重复** ⭐⭐⭐⭐⭐

**问题**：
```
Starbucks 在 Yelp 数据集中可能有 500 家门店
如果 text_description 只包含：
  "Name: Starbucks. Category: Coffee."

那么这 500 家店的文本完全相同
↓
BERT 生成的 embedding 完全相同
↓
RQ-VAE 量化后的 SID 必然相同
↓
500 家店全部碰撞！
```

**影响**：
- 连锁店（McDonald's, Starbucks, Subway 等）在美国非常普遍
- 如果 10% 的商户是连锁店（每个连锁平均 50 家），就会贡献大量碰撞

#### ⚠️ **原因 2：RQ-VAE 训练不足**
- 当前训练可能还没收敛
- 需要等待完整的 5000 epochs

#### ⚠️ **原因 3：Codebook 容量不足**
- 当前 64³ = 262,144 种可能的 SID
- 理论上足够，但实际利用率只有 21.8%

---

## 诊断步骤

### Step 1：运行连锁店分析

```bash
cd /workspace
python data_processing/analyze_chain_stores.py
```

**输出示例**：
```
=== Top 20 Chain Stores ===
1.  Starbucks                           |  523 stores | Cities: Las Vegas(89), Phoenix(78), ...
2.  McDonald's                          |  412 stores | Cities: Las Vegas(65), Phoenix(52), ...
3.  Subway                              |  387 stores | Cities: Phoenix(71), Las Vegas(58), ...
...

=== Text Description Duplication ===
Duplicate text descriptions: 12,345 (8.2%)
Total businesses with duplicate texts: 45,678 (30.4%)

=== Collision Correlation ===
Chain store collisions: 28,000 (30.1% of all collisions)
Non-chain collisions: 65,046 (69.9% of all collisions)
```

**判断**：
- **如果 chain_collisions > 30%** → 连锁店是主要问题，必须修改 text_description
- **如果 chain_collisions < 10%** → 连锁店不是主因，专注于提高 RQ-VAE 质量

---

## 解决方案

### ✅ **方案 1：在 text_description 中添加唯一性信息** ⭐⭐⭐⭐⭐

#### 修改内容
已修改 `data_processing/step1_build_item_profile.py`：

**修改前**：
```python
profile_parts.append(f"Name: {name}.")
profile_parts.append(f"City: {city}, {state}.")
profile_parts.append(f"Categories: {categories}.")
```

**修改后**：
```python
import hashlib

# 生成唯一ID哈希
id_hash = hashlib.md5(business_id.encode()).hexdigest()[:6]

profile_parts.append(f"Name: {name}.")

# 添加地址（如果有）
if address:
    address_short = address[:50]
    profile_parts.append(f"Address: {address_short}.")

profile_parts.append(f"City: {city}, {state}.")

# 添加唯一ID哈希（强制区分）
profile_parts.append(f"ID: {id_hash}.")

profile_parts.append(f"Categories: {categories}.")
```

**效果**：
```
同一个 Starbucks 的不同门店：

门店 A: "Name: Starbucks. Address: 123 Main St. City: Las Vegas, NV. ID: a1b2c3. Category: Coffee."
门店 B: "Name: Starbucks. Address: 456 Strip Blvd. City: Las Vegas, NV. ID: d4e5f6. Category: Coffee."

↓
BERT 生成的 embedding 会有微小差异
↓
RQ-VAE 可以利用第三层（细粒度层）将它们区分开
```

#### 重新运行 Step 1

```bash
# 重新生成 item_profiles.jsonl
python data_processing/step1_build_item_profile.py
```

**注意**：
- 这会覆盖之前的 `item_profiles.jsonl`
- 需要重新运行 Step 2（重新训练 RQ-VAE）

---

### ✅ **方案 2：增大 Codebook 容量**

修改 `config/config.yaml`：

```yaml
rqvae:
  num_emb_list: [128, 128, 128]  # 从 [64, 64, 64] 增加
  epochs: 10000                   # 增加训练时间
```

**容量对比**：
```
当前: 64³ = 262,144
增大: 128³ = 2,097,152 (8倍容量)
```

---

### ✅ **方案 3：调整 α 系数和 Sinkhorn**

```yaml
rqvae:
  alpha: [1.2, 1.1, 1.0]          # 增强前两层的残差放大
  sk_epsilons: [0.005, 0.005, 0.003]  # 在前两层也启用约束
```

---

## 推荐执行顺序

### 🎯 **立即执行**（最重要）

1. **运行诊断**
   ```bash
   python data_processing/analyze_chain_stores.py
   ```

2. **查看结果**
   - 打开 `./data/chain_store_analysis.json`
   - 查看 `chain_store_rate`（连锁店占比）
   - 查看 `chain_collisions` 占比

### 📋 **根据诊断结果选择方案**

#### 情况 A：Chain Collisions > 30%
```bash
# 1. 重新生成 profiles（已添加唯一性信息）
python data_processing/step1_build_item_profile.py

# 2. 重新训练 RQ-VAE
python data_processing/step2_generate_semantic_ids.py

# 3. 等待训练完成后再次可视化
python visualization/visualize_codebook.py
```

**预期效果**：
- Collision Rate 降到 < 10%
- Text duplication rate 降到 < 5%

#### 情况 B：Chain Collisions < 10%
```bash
# 问题不在连锁店，而在 RQ-VAE 训练
# 方案：增大 Codebook + 继续训练

# 1. 修改 config.yaml
vim config/config.yaml
# 设置 num_emb_list: [128, 128, 128]
# 设置 epochs: 10000

# 2. 删除旧 checkpoint
rm -rf ./data/rqvae_ckpt/*

# 3. 重新训练
python data_processing/step2_generate_semantic_ids.py
```

---

## 验证效果

### 1. 重新分析连锁店

```bash
python data_processing/analyze_chain_stores.py
```

检查：
- `Duplicate text descriptions` 是否下降
- `Chain store collisions` 是否下降

### 2. 重新可视化 Codebook

```bash
python visualization/visualize_codebook.py
```

检查 `quality_report.txt`：
- `Collision Rate` 是否 < 5%
- `Codebook Usage` 是否仍然 > 90%

---

## 理论依据

### 为什么添加唯一性信息有效？

#### BERT Embedding 的特性
```python
# BERT 对微小文本差异敏感
text1 = "Name: Starbucks. City: Las Vegas. ID: a1b2c3."
text2 = "Name: Starbucks. City: Las Vegas. ID: d4e5f6."

embedding1 = bert.encode(text1)  # [768-dim vector]
embedding2 = bert.encode(text2)  # [768-dim vector]

cosine_similarity(embedding1, embedding2) ≈ 0.95
# 相似但不完全相同！
```

#### RQ-VAE 的分层设计
```
Layer 0 (粗粒度): 捕获大类别（"Coffee Shop"）
  ↓ embedding1 和 embedding2 在这层可能相同
  
Layer 1 (中粒度): 捕获子类别（"Starbucks"）
  ↓ 在这层可能仍然相同
  
Layer 2 (细粒度): 捕获具体差异（不同门店）
  ↓ ID 哈希值强制这层产生不同的 code
  ↓ [12, 34, 56] vs [12, 34, 57]
```

### α 系数的作用

```python
# α = [1.2, 1.1, 1.0]

Layer 0: residual1 = 1.2 * input - code0
Layer 1: residual2 = 1.1 * residual1 - code1
Layer 2: residual3 = 1.0 * residual2 - code2

# α > 1 放大残差 → 下一层有更多信息可用
# → 提高细粒度层的区分能力
```

---

## 预期结果

### 修改前
```
Text: "Name: Starbucks. Category: Coffee."
500 家 Starbucks → 500 个完全相同的 embedding
→ 500 家店共享 1 个 SID
→ Collision rate ↑↑↑
```

### 修改后
```
Text: "Name: Starbucks. Address: 123 Main St. ID: a1b2c3. Category: Coffee."
500 家 Starbucks → 500 个略有差异的 embedding
→ Layer 0-1 可能相同（都是 Starbucks）
→ Layer 2 不同（不同门店）
→ 500 家店 → 400-450 个不同的 SID
→ Collision rate: 10-20% (可接受)
```

---

## 常见问题

### Q1: 添加 ID 哈希会不会破坏语义？
**A**: 不会。BERT 会将 ID 视为"噪音"，主要语义仍来自名称和类别。但这个"噪音"足以让 RQ-VAE 的第三层区分不同门店。

### Q2: 如果还是碰撞率高怎么办？
**A**: 按优先级尝试：
1. 增大 Codebook: `num_emb_list: [128, 128, 128]`
2. 增加训练时间: `epochs: 10000`
3. 调整 α: `alpha: [1.3, 1.2, 1.0]`

### Q3: 需要重新训练 LLM 吗？
**A**: **不需要**。SID 改变后，只需重新运行 Step 3-4 构建新的训练数据即可。

---

## 总结

**核心思想**：在不改变语义的前提下，为每个商户注入"唯一性噪音"，让 RQ-VAE 的细粒度层能够区分连锁店的不同门店。

**关键修改**：
1. ✅ Step 1: 添加地址和 ID 哈希到 text_description
2. ⏳ 等待分析结果，决定是否需要额外调整 Codebook

**期望效果**：
- Collision Rate: 61.89% → < 5%
- 推荐精度显著提升
