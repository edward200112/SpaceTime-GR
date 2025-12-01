# 模型微调准确性验证指南

本文档说明如何验证 HierGR-SeqRec 模型微调后的准确性。

---

## 一、评估方法概览

### 1. 训练时在线评估
在训练过程中自动进行，无需额外操作。

**配置位置**: `config/config.yaml`

```yaml
logging:
  log_interval: 100      # 每100步输出一次训练loss
  save_interval: 500     # 每500步进行一次验证并保存checkpoint
```

**特点**:
- ✅ 实时监控训练进度
- ✅ 自动保存最佳模型（基于 `eval_loss`）
- ✅ 防止过拟合
- ❌ 仅使用验证集的 loss，不计算业务指标

**查看训练日志**:
```bash
# 训练日志会显示
Step 500: train_loss=2.34, eval_loss=2.56
Step 1000: train_loss=1.98, eval_loss=2.31
...
```

---

### 2. 训练后离线评估 ⭐ 推荐

使用独立测试集全面评估模型性能。

**评估指标**:
- **HR@K (Hit Rate)**: 前K个推荐中是否包含目标商家
- **NDCG@K**: 考虑排序位置的命中率，排名越靠前得分越高
- **MRR (Mean Reciprocal Rank)**: 目标商家的平均倒数排名

---

## 二、使用评估脚本

### 准备工作

1. **确保测试数据格式正确**

测试数据 JSON 格式示例 (`data/processed/test_prompts.json`):
```json
[
  {
    "prompt": "Based on the user's visit history:\n1. [Starbucks] (Coffee) -> <3, 12>\n2. ...\n\nPredict next visit:",
    "target_business_id": "B_xyz123",
    "target_cluster_str": "<3, 15>"
  },
  ...
]
```

2. **确保模型已训练完成**

检查 checkpoint 目录:
```bash
ls ./data/llm_checkpoints/
# 应该看到: pytorch_model.bin, config.json, tokenizer_config.json 等
```

---

### 运行评估

#### 基础用法
```bash
cd HierGR-SeqRec

python evaluation/evaluate_model.py \
    --config ./config/config.yaml \
    --test_data ./data/processed/test_prompts.json \
    --batch_size 8 \
    --num_beams 5 \
    --top_k 5,10,20 \
    --output ./evaluation/results.json
```

#### 参数说明
- `--config`: 配置文件路径
- `--test_data`: 测试数据路径
- `--batch_size`: 批量推理大小，显存不足可减小
- `--num_beams`: Beam search 数量，越大越准确但越慢
- `--top_k`: 评估的 K 值列表，逗号分隔
- `--output`: 结果保存路径

---

### 解读评估结果

运行后会看到:
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

**指标解释**:
- **HR@5 = 0.3421**: 34.21% 的情况下，正确答案在前5个推荐中
- **NDCG@5 = 0.2156**: 考虑排序位置后的得分为 0.2156
- **MRR = 0.2845**: 平均来说，正确答案排在第 3.5 位 (1/0.2845 ≈ 3.5)

**性能标准参考**:
| 指标 | 差 | 中等 | 良好 | 优秀 |
|------|-----|------|------|------|
| HR@10 | <30% | 30-45% | 45-60% | >60% |
| NDCG@10 | <0.15 | 0.15-0.25 | 0.25-0.35 | >0.35 |
| MRR | <0.20 | 0.20-0.30 | 0.30-0.40 | >0.40 |

---

## 三、高级评估方法

### 1. 分类别评估
评估不同商家类别的性能差异:

```python
# 在 evaluate_model.py 中添加
def evaluate_by_category(self, test_data, categories):
    category_metrics = {}
    
    for category in categories:
        # 过滤该类别的测试数据
        category_samples = [s for s in test_data if s['category'] == category]
        
        # 评估
        metrics = self.evaluate(category_samples)
        category_metrics[category] = metrics
    
    return category_metrics
```

### 2. 错误案例分析
保存预测错误的案例进行分析:

```python
# 修改 calculate_metrics 方法，添加:
error_cases = []
for i, (pred, target) in enumerate(zip(predictions, targets)):
    if target not in pred[:10]:  # Top-10 miss
        error_cases.append({
            'index': i,
            'prompt': prompts[i],
            'predicted': pred[:10],
            'target': target
        })

# 保存错误案例
with open('error_analysis.json', 'w') as f:
    json.dump(error_cases, f, indent=2)
```

### 3. A/B 测试对比
对比不同模型版本:

```bash
# 评估基线模型
python evaluation/evaluate_model.py \
    --config config_baseline.yaml \
    --test_data test.json \
    --output results_baseline.json

# 评估微调模型
python evaluation/evaluate_model.py \
    --config config_finetuned.yaml \
    --test_data test.json \
    --output results_finetuned.json

# 对比结果
python tools/compare_results.py \
    --baseline results_baseline.json \
    --finetuned results_finetuned.json
```

---

## 四、常见问题

### Q1: 评估速度很慢怎么办？
**解决方案**:
- 减小 `--batch_size`（如改为 4 或 2）
- 减小 `--num_beams`（如改为 3）
- 使用采样策略而非 beam search
- 在测试集子集上快速验证

### Q2: 显存不足 (OOM)
**解决方案**:
```python
# 在 evaluate_model.py 开头添加
import torch
torch.cuda.empty_cache()

# 或使用 8bit 量化
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map='auto'
)
```

### Q3: HR 和 NDCG 都很低怎么办？
**诊断步骤**:
1. **检查数据质量**: 测试集格式是否正确
2. **检查模型加载**: 是否加载了正确的 checkpoint
3. **检查 SID 映射**: cluster_str 是否正确解析
4. **降低任务难度**: 先在简单数据上测试

**可能原因**:
- 训练数据不足或质量差
- 模型未充分训练
- Cluster 粒度过细，导致展开后候选商家少
- 测试集与训练集分布差异大

### Q4: 如何判断模型是否过拟合？
**对比指标**:
```
训练集 HR@10: 0.85  ✓ 很高
验证集 HR@10: 0.48  ✓ 中等
测试集 HR@10: 0.32  ✗ 偏低 -> 可能过拟合
```

**解决方案**:
- 增加训练数据
- 使用更强的正则化（增大 `weight_decay`）
- 早停（选择验证集表现最好的 checkpoint）
- 数据增强

---

## 五、持续监控与优化

### 建立评估流程
```bash
#!/bin/bash
# scripts/auto_evaluate.sh

echo "开始自动评估..."

# 评估最新模型
python evaluation/evaluate_model.py \
    --config config.yaml \
    --test_data data/processed/test_prompts.json \
    --output evaluation/results_$(date +%Y%m%d_%H%M%S).json

# 发送通知
echo "评估完成！查看结果: evaluation/results_*.json"
```

### 记录实验结果
创建实验日志表格:

| 日期 | 模型版本 | 训练步数 | HR@10 | NDCG@10 | MRR | 备注 |
|------|----------|----------|-------|---------|-----|------|
| 2024-11-28 | v1.0 | 10K | 0.42 | 0.23 | 0.26 | Baseline |
| 2024-11-29 | v1.1 | 15K | 0.48 | 0.27 | 0.29 | 增大学习率 |
| 2024-11-30 | v1.2 | 20K | 0.51 | 0.29 | 0.31 | 添加数据增强 |

---

## 六、快速检查清单

微调完成后，依次检查：

- [ ] 训练 loss 是否收敛
- [ ] 验证 loss 是否不再下降
- [ ] 训练/验证 loss 差距是否合理（<20%）
- [ ] 在小样本上手动测试推理是否正常
- [ ] 运行完整评估脚本
- [ ] HR@10 是否 >40%
- [ ] NDCG@10 是否 >0.20
- [ ] 与基线模型对比是否有提升
- [ ] 分析错误案例找出共性问题
- [ ] 记录实验结果和参数配置

---

## 参考资料

- **MiniOneRec 评估实现**: `../MiniOneRec/evaluate.py`
- **推荐系统评估综述**: [RecSys Evaluation Metrics](https://recsys.acm.org/)
- **NDCG 详解**: [Understanding NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
