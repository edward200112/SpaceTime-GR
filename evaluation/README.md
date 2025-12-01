# 模型评估工具集

本目录包含 HierGR-SeqRec 模型评估的完整工具链。

## 📂 文件说明

### 1. `evaluate_model.py` - 完整评估脚本 ⭐
对模型进行全面评估，计算 HR@K、NDCG@K、MRR 等指标。

**用法**:
```bash
python evaluate_model.py \
    --config ../config/config.yaml \
    --test_data ../data/processed/test_prompts.json \
    --batch_size 8 \
    --num_beams 5 \
    --top_k 5,10,20 \
    --output results.json
```

**输出示例**:
```
==================================================
评估结果:
==================================================
HR@5           : 0.3421
NDCG@5         : 0.2156
HR@10          : 0.4832
NDCG@10        : 0.2687
HR@20          : 0.6125
NDCG@20        : 0.3012
MRR            : 0.2845
==================================================
```

---

### 2. `quick_test.py` - 快速测试工具
交互式测试单个样本，适合快速验证模型是否正常工作。

**用法**:
```bash
# 交互式模式
python quick_test.py --config ../config/config.yaml

# 批量测试模式
python quick_test.py \
    --config ../config/config.yaml \
    --test_file ../data/processed/test_prompts.json \
    --mode batch
```

**交互式示例**:
```
> B_abc123,B_def456,B_ghi789

📝 格式化的历史:
1. [Starbucks] (Coffee) -> <3, 12>
2. [AMC Cinema] (Entertainment) -> <5, 8>
3. [Best Buy] (Electronics) -> <7, 3>

✅ 预测的 Cluster IDs (Top-5):
  1. <3, 15>
  2. <7, 4>
  3. <5, 9>
  ...

🎯 展开 <3, 15> 中的商家:
  1. [Apple Store] (Electronics) - Phoenix
  2. [Target] (Shopping) - Phoenix
  ...
```

---

### 3. `compare_results.py` - 结果对比工具
对比两个模型版本的评估结果，生成详细报告。

**用法**:
```bash
python compare_results.py \
    --baseline results_baseline.json \
    --finetuned results_finetuned.json \
    --output comparison_report.md
```

---

### 4. `generate_test_samples.py` - 测试数据生成器 🆕
从已处理的数据中快速生成测试样本。

**用法**:
```bash
python generate_test_samples.py \
    --config ../config/config.yaml \
    --num_samples 20 \
    --output test_samples.json
```

**功能**:
- 从用户序列中自动抽取测试样本
- 自动构造 prompt 和 target
- 无需手动标注数据

---

### 5. `EVALUATION_GUIDE.md` - 完整评估指南
详细的评估方法说明、指标解释、常见问题和优化建议。

---

### 6. `README.md` - 工具集说明文档
本文档，提供快速入门和工具使用说明

---

## 🚀 快速开始

### Step 0: 生成测试数据（如果没有）
如果你还没有测试文件，使用生成器快速创建：
```bash
python evaluation/generate_test_samples.py \
    --config ./config/config.yaml \
    --num_samples 20 \
    --output ./evaluation/test_samples.json
```

**前提条件**：需要先运行数据处理流程：
```bash
python data_processing/step1_build_item_profile.py
python data_processing/step2_generate_semantic_ids.py
python data_processing/step3_build_user_sequences.py
```

---

### Step 1: 快速验证（无需测试文件）⭐
最简单的方式 - 交互式测试：
```bash
python quick_test.py --config ../config/config.yaml
```
然后手动输入商家ID，立即看到预测结果。

---

### Step 2: 批量测试
使用生成的测试文件进行批量测试：
```bash
python quick_test.py \
    --config ../config/config.yaml \
    --test_file evaluation/test_samples.json \
    --mode batch
```

---

### Step 3: 完整评估
运行完整评估获取详细指标：
```bash
python evaluate_model.py \
    --config ../config/config.yaml \
    --test_data evaluation/test_samples.json \
    --output results.json
```

### Step 4: 对比分析（可选）
如果有多个模型版本，进行对比:
```bash
python compare_results.py \
    --baseline results_v1.json \
    --finetuned results_v2.json \
    --output comparison.md
```

---

## 📊 评估指标说明

### HR@K (Hit Rate @ K)
前 K 个推荐中是否包含目标商家。

**计算公式**:
```
HR@K = (命中样本数) / (总样本数)
```

**解释**: HR@10 = 0.48 表示 48% 的情况下，正确答案在前 10 个推荐中。

---

### NDCG@K (Normalized Discounted Cumulative Gain @ K)
考虑排序位置的命中率，排名越靠前得分越高。

**计算公式**:
```
NDCG@K = (1 / log2(rank + 1)) if rank <= K else 0
```

**解释**: NDCG@10 = 0.27 表示考虑位置后的归一化得分为 0.27。

---

### MRR (Mean Reciprocal Rank)
目标商家的平均倒数排名。

**计算公式**:
```
MRR = 平均(1 / rank)
```

**解释**: MRR = 0.28 表示平均排名约为 1/0.28 ≈ 3.6 位。

---

## 🎯 性能基准

| 指标 | 差 | 中等 | 良好 | 优秀 |
|------|-----|------|------|------|
| **HR@10** | <30% | 30-45% | 45-60% | >60% |
| **NDCG@10** | <0.15 | 0.15-0.25 | 0.25-0.35 | >0.35 |
| **MRR** | <0.20 | 0.20-0.30 | 0.30-0.40 | >0.40 |

---

## 🔧 常见问题

### Q: 评估速度慢怎么办？
A: 
- 减小 `--batch_size`
- 减小 `--num_beams`
- 在小样本上先测试

### Q: 显存不足？
A: 
- 减小 batch_size
- 使用 8bit 量化加载模型
- 使用 CPU 模式（慢但不需要显存）

### Q: 指标很低怎么办？
A: 
1. 检查数据格式是否正确
2. 检查模型是否加载成功
3. 查看训练 loss 是否收敛
4. 分析错误案例

---

## 📝 评估流程建议

1. ✅ 训练完成后，先查看训练/验证 loss
2. ✅ 使用 `quick_test.py` 手动测试几个样本
3. ✅ 运行 `evaluate_model.py` 进行完整评估
4. ✅ 如有多个版本，使用 `compare_results.py` 对比
5. ✅ 分析评估结果，制定优化方案
6. ✅ 记录实验配置和结果

---

## 📚 更多信息

详细使用说明请参考 `EVALUATION_GUIDE.md`。
