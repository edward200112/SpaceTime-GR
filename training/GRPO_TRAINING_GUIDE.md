# GRPO Training Guide for HierGR-SeqRec

基于 MiniOneRec 实现的 GRPO (Group Relative Policy Optimization) 强化学习训练完整指南。

---

## 📚 核心概念

### 1. Constrained Beam Search
约束生成确保模型只输出有效的 Cluster IDs。

**工作原理**:
```python
# 构建哈希字典，存储有效的token序列
hash_dict = {
    "### Response": [<, token_id_for_numbers],
    "### Response\n<": [0,1,2,3...],  # 第一个数字
    "### Response\n<3": [","],          # 逗号
    "### Response\n<3,": [0,1,2...],    # 第二个数字
    ...
}

# 在生成时，约束logits只允许有效的tokens
constrained_processor = ConstrainedClusterLogitsProcessor(...)
```

### 2. GRPO训练流程
```
for each epoch:
    for each batch (包含N个prompts):
        1. 重复每个prompt M次 (通过RepeatRandomSampler)
        2. 生成 M 个completions (beam search or sampling)
        3. 计算每个completion的reward
        4. Group-wise reward normalization:
           advantages = (rewards - group_mean) / (group_std + eps)
        5. 计算GRPO loss:
           loss = -E[π/π_old * A] + β * KL(π||π_ref)
        6. 反向传播更新模型
```

### 3. Reward Functions

**Rule Reward (二元奖励)**:
```python
reward = 1.0 if completion == target else 0.0
```

**NDCG Reward (排序奖励)**:
```python
if target in completions:
    reward = 0.0 for target
    reward = -1/log2(rank+2) for others
else:
    reward = 0.0 for all
```

**Combined Reward**:
```python
reward = α * rule_reward + β * ndcg_reward
```

---

## 🚀 快速开始

### Step 1: 准备 SFT 模型
GRPO需要一个已经过SFT训练的模型作为起点：

```bash
# 如果还没有SFT模型，先运行SFT训练
python training/train_llm.py \
    --config ./config/config.yaml
```

### Step 2: 准备训练数据
数据格式（JSON）：
```json
[
  {
    "prompt": "Based on the user's visit history:\n1. [Starbucks] (Coffee) -> <3, 12>\n...\nPredict next visit:",
    "target_cluster_str": "<3, 15>"
  },
  ...
]
```

生成数据：
```bash
python data_processing/step4_construct_prompts.py
```

### Step 3: 运行 GRPO 训练

**基础训练（Rule Reward + Beam Search）**:
```bash
python training/train_grpo.py \
    --config ./config/config.yaml \
    --train_data ./data/processed/train_prompts.json \
    --eval_data ./data/processed/valid_prompts.json \
    --sft_model ./data/llm_checkpoints \
    --output_dir ./data/grpo_checkpoints \
    --num_generations 16 \
    --beta 0.04 \
    --reward_type rule \
    --use_beam_search \
    --test_during_training \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --num_epochs 1
```

**高级训练（Combined Reward + Sampling）**:
```bash
python training/train_grpo.py \
    --config ./config/config.yaml \
    --train_data ./data/processed/train_prompts.json \
    --eval_data ./data/processed/valid_prompts.json \
    --sft_model ./data/llm_checkpoints \
    --output_dir ./data/grpo_combined_checkpoints \
    --num_generations 16 \
    --beta 0.04 \
    --reward_type combined \
    --batch_size 4 \
    --temperature 1.0 \
    --num_epochs 1
```

---

## 🔧 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_generations` | 16 | 每个prompt生成的completion数量 |
| `--beta` | 0.04 | KL散度系数，控制与参考模型的偏离程度 |
| `--reward_type` | rule | 奖励类型: `rule`, `ndcg`, `combined` |
| `--use_beam_search` | False | 使用beam search而非sampling |
| `--test_during_training` | False | 训练时评估 |
| `--test_beam` | 20 | 测试时的beam size |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 批量大小 |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--learning_rate` | 1e-6 | 学习率（比SFT小） |
| `--num_epochs` | 1 | 训练轮数 |
| `--max_completion_length` | 20 | 最大生成长度 |
| `--temperature` | 1.0 | 采样温度 |

---

## 📊 监控训练

训练时会输出以下指标：

```
Step 100:
  loss: 2.34
  reward: 0.56          # 平均奖励
  reward_std: 0.23      # 奖励标准差
  kl: 0.012             # KL散度
  HR@5: 0.34            # Hit Rate @5
  NDCG@5: 0.21          # NDCG @5
  HR@10: 0.48
  NDCG@10: 0.27
  HR@20: 0.62
  NDCG@20: 0.31
```

**关键指标解释**:
- **reward**: 越高越好，接近1.0表示大部分生成都是正确的
- **kl**: KL散度，应保持较小（<0.05），否则模型偏离参考模型太远
- **HR@K / NDCG@K**: 测试集上的命中率和排序质量

---

## 🎯 最佳实践

### 1. 超参数调优

**Beta (KL系数)**:
- 太小（<0.01）: 模型可能过度优化reward，忘记原始知识
- 太大（>0.1）: 模型更新太保守，训练慢
- 推荐: 0.03 - 0.05

**Num Generations**:
- 太少（<8）: Group normalization不稳定
- 太多（>32）: 显存和计算开销大
- 推荐: 16

**Learning Rate**:
- 比SFT小1-2个数量级
- 推荐: 1e-6 ~ 5e-6

### 2. Beam Search vs Sampling

**Beam Search**:
- ✅ 生成质量更高
- ✅ 更稳定
- ❌ 多样性低
- 推荐用于: 任务需要准确答案时

**Sampling**:
- ✅ 多样性高
- ✅ 探索更多可能性
- ❌ 可能生成低质量样本
- 推荐用于: 需要探索多种推荐时

### 3. Reward设计

**Rule Reward**:
- 简单直接
- 适合: 只关心是否命中

**NDCG Reward**:
- 考虑排序位置
- 适合: 关心推荐列表质量

**Combined Reward**:
- 平衡准确性和排序
- 推荐用于生产环境

---

## 🔍 评估 GRPO 模型

### 使用约束生成评估
```bash
python evaluation/evaluate_model.py \
    --config ./config/config.yaml \
    --test_data ./data/processed/test_prompts.json \
    --batch_size 8 \
    --num_beams 20 \
    --use_constrained_generation \
    --output ./evaluation/grpo_results.json
```

### 对比 SFT vs GRPO
```bash
# 评估SFT模型
python evaluation/evaluate_model.py \
    --test_data test.json \
    --output sft_results.json \
    --use_constrained_generation

# 评估GRPO模型
python evaluation/evaluate_model.py \
    --test_data test.json \
    --output grpo_results.json \
    --use_constrained_generation

# 对比
python evaluation/compare_results.py \
    --baseline sft_results.json \
    --finetuned grpo_results.json \
    --output grpo_improvement.md
```

---

## 🐛 常见问题

### Q1: 显存不足 (OOM)
**解决方案**:
```bash
# 减小batch size和num_generations
--batch_size 2 \
--num_generations 8 \
--gradient_accumulation_steps 8

# 或使用梯度检查点
# 在config.yaml中设置:
llm:
  gradient_checkpointing: true
```

### Q2: Reward不增长
**可能原因**:
1. Learning rate太大或太小
2. Beta太大，限制了模型更新
3. Reward函数设计不合理

**诊断**:
```python
# 检查KL散度
if kl > 0.1:
    # Beta太小，减小learning rate
    pass
if kl < 0.001:
    # Beta太大，减小beta或增大learning rate
    pass
```

### Q3: 训练不稳定
**解决方案**:
1. 使用更小的learning rate
2. 增加warmup比例
3. 使用beam search而非sampling
4. 确保batch size能被num_generations整除

### Q4: 生成无效的Cluster IDs
**解决方案**:
```bash
# 确保使用约束生成
--use_beam_search  # 训练时

# 评估时
--use_constrained_generation  # 评估时
```

---

## 📈 预期效果

根据 MiniOneRec 的经验：

**SFT Baseline**:
- HR@10: 40-45%
- NDCG@10: 22-25%

**GRPO (Rule Reward)**:
- HR@10: 48-53% (+8%)
- NDCG@10: 25-28% (+3%)

**GRPO (Combined Reward)**:
- HR@10: 50-55% (+10%)
- NDCG@10: 27-30% (+5%)

---

## 🔗 参考资料

1. **DeepSeekMath论文**: [GRPO原始论文](https://arxiv.org/abs/2402.03300)
2. **MiniOneRec**: 本实现基于的开源项目
3. **TRL库**: [Hugging Face TRL](https://github.com/huggingface/trl)

---

## 📝 总结

GRPO训练流程：
1. ✅ 准备SFT模型
2. ✅ 准备训练数据（prompt + target）
3. ✅ 选择reward类型和超参数
4. ✅ 运行train_grpo.py
5. ✅ 监控指标（reward, KL, HR, NDCG）
6. ✅ 使用约束生成评估

关键点：
- 使用约束生成确保输出有效
- Beta控制KL散度
- Group-wise normalization稳定训练
- Beam search提高质量，sampling提高多样性
