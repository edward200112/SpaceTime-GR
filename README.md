# HierGR-SeqRec: 层级生成式序列推荐系统

基于 Yelp 数据集的生成式推荐系统，使用 RQ-VAE + LLM 实现层级语义 ID 生成。

HierGR-SeqRec/
├── config/
│   └── config.yaml                   # 全局配置文件
│
├── data/                             # 数据存储目录
│   ├── raw/                          # 原始Yelp数据
│   │   ├── yelp_academic_dataset_business.json
│   │   └── yelp_academic_dataset_review.json
│   ├── processed/                    # 处理后数据
│   │   ├── item_profiles.jsonl       # 商家画像
│   │   ├── item_embeddings.pt        # BERT嵌入
│   │   ├── sid_mapping.json          # 商家ID→语义ID映射
│   │   ├── user_sequences.jsonl      # 用户序列
│   │   ├── train_prompts.jsonl       # 训练Prompts
│   │   ├── train_prompts_balanced.jsonl  # 平衡采样版本
│   │   ├── valid_prompts.jsonl       # 验证集
│   │   ├── test_prompts.jsonl        # 测试集
│   │   └── category_weights.json     # 类别权重表
│   ├── embeddings/                   # 嵌入向量缓存
│   ├── rqvae_ckpt/                   # RQ-VAE模型检查点
│   ├── llm_ckpt_sft_v5_balanced/     # SFT训练检查点
│   └── grpo_v5_weighted/             # GRPO训练检查点
│
├── data_processing/                  # 数据处理流水线
│   ├── step1_build_item_profile.py   # 构建商家画像
│   ├── step2_generate_semantic_ids.py # 训练RQ-VAE + 生成SID
│   ├── step3_build_user_sequences.py # 构建用户行为序列
│   ├── step4_construct_prompts.py    # 构造多任务Prompts
│   ├── balance_dataset.py            # 长尾类别平衡采样
│   └── analyze_chain_stores.py       # 连锁店分析工具
│
├── RQ-VAE/                           # RQ-VAE核心实现
│   ├── models/
│   │   ├── rqvae.py                  # RQ-VAE模型定义
│   │   ├── quantizers.py             # 量化器（Sinkhorn-Knopp）
│   │   └── encoder_decoder.py        # 编码器-解码器
│   └── trainer.py                    # RQ-VAE训练器
│
├── training/                         # LLM训练模块
│   ├── train_sft_final.py            # SFT训练脚本（V5终极版）
│   ├── train_grpo_v5.py              # GRPO训练脚本（V5加权版）
│   ├── grpo_rewards_optimized.py     # 优化奖励函数
│   ├── constrained_logits_processor.py # Trie约束生成
│   ├── dataset.py                    # 数据集加载器
│   ├── merge_model.py                # LoRA模型合并工具
│   └── GRPO_TRAINING_GUIDE.md        # GRPO训练指南
│
├── inference/                        # 推理与评估
│   ├── new_evaluate.py               # 完整评估脚本
│   ├── evaluate_metrics.py           # 层级准确率统计
│   ├── validate_grpo_with_tsne.py    # t-SNE可视化验证
│   ├── check_sft_quality.py          # SFT质量检查
│   ├── check_cluster_purity.py       # 聚类纯度分析
│   ├── recommend.py                  # 在线推荐接口
│   └── trie_utils.py                 # Trie树工具类
│
├── visualization/                    # 可视化脚本
│   ├── plot_training_curves.py       # 训练曲线绘制
│   ├── plot_tsne.py                  # t-SNE降维可视化
│   └── plot_category_distribution.py # 类别分布图
│
├── evaluation/                       # 评估工具
│   ├── metrics.py                    # 评估指标（Hit@K, NDCG@K）
│   └── geo_utils.py                  # 地理距离计算
│
├── examples/                         # 示例脚本
│   ├── quick_demo.py                 # 快速演示
│   └── batch_inference.py            # 批量推理
│
├── run_pipeline.py                   # 全流程自动化脚本
├── inspect_data.py                   # 数据检查工具
├── requirements.txt                  # Python依赖
├── README.md                         # 项目文档
├── QUICKSTART.md                     # 快速开始指南
└── MODEL_PATHS.md                    # 模型路径配置说明


## 项目架构



```
用户历史序列 → RQ-VAE编码 → 4层语义ID → LLM生成 → 推荐结果
                ↓                    ↓
            地理信息融合          SFT + GRPO训练
```

### 核心组件
- **RQ-VAE**: 将商家编码为 4 层语义 ID `<c0, c1, c2, suffix>`
- **LLM**: Qwen2.5-1.5B + LoRA 微调
- **训练**: SFT（格式学习）+ GRPO（精准度优化）

---

## 最近更新（Training & Inference 模块）

### 🔧 训练模块改进 (`training/`)

#### 1. **`train_llm.py`** - SFT 训练脚本（已修复）

**关键修复：**
- ✅ **修复 `ValueError: model did not return a loss`**
  - 在 `tokenize_function` 中显式创建 `labels` 字段
  - `labels = input_ids.copy()` 用于 Causal LM 训练
  
- ✅ **修复 LoRA + Gradient Checkpointing 兼容性**
  - 添加 `model.enable_input_require_grads()` 
  - 确保梯度能正确反向传播到 LoRA 层

- ✅ **优化数据处理**
  - 使用 `DataCollatorForSeq2Seq` 自动处理 padding
  - 将 padding token 的 label 设为 -100（忽略 loss 计算）

**代码示例：**
```python
def tokenize_function(examples):
    model_inputs = self.tokenizer(
        examples["text"],
        truncation=True,
        max_length=self.llm_conf['max_seq_length'],
        padding=False 
    )
    # [关键] 显式创建 labels
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs
```

#### 2. **`train_grpo_v2.py`** - GRPO 优化版本（新增）

**改进点：**
- 从 SFT checkpoint 重新开始训练（而非继续训练）
- 增加 LoRA 秩：`r=64`（提升学习能力）
- 调整采样温度：`temperature=1.2`（鼓励探索不同 ID）
- 增加 KL 惩罚：`beta=0.04`（防止遗忘 SFT 知识）
- 使用优化的奖励函数 `grpo_rewards_optimized.py`

**使用方法：**
```bash
python training/train_grpo_v2.py
```

#### 3. **`grpo_rewards_optimized.py`** - 优化奖励函数（新增）

**三维度奖励系统：**

| 奖励类型 | 计算逻辑 | 权重分配 |
|---------|---------|---------|
| **Format Reward** | 格式正确: +0.1<br>有内容但格式错: -0.9<br>空输出: -1.0 | 基础 |
| **Geo Reward** | ≤5km: +0.3<br>≤20km: +0.1<br>≤50km: 0.0<br>>50km: -0.1 | 核心 |
| **Semantic Reward** | Layer0匹配: +0.1<br>Layer1匹配: +0.2<br>Layer2匹配: +1.0<br>完全匹配: +3.0 | 递进 |

**关键特性：**
- 支持 String 和 List 格式的 `target_sid` 解析
- 完美适配 Step2/Step4 生成的数据格式
- 使用 Haversine 公式计算地理距离

#### 4. **`constrained_logits_processor.py`** - 约束生成（新增）

**功能：**
- 基于 Trie 数据结构约束生成
- 确保模型只输出有效的 Cluster IDs
- 支持 Beam Search 的约束生成

**工作原理：**
```python
# 构建 Trie 索引所有有效 ID
trie.insert("<12, 34, 56, 0>")
trie.insert("<12, 34, 56, 1>")

# 生成时约束 logits
allowed_tokens = trie.get_next_tokens(current_prefix)
scores[~allowed_tokens] = -inf  # 屏蔽无效 token
```

#### 5. **`GRPO_TRAINING_GUIDE.md`** - 完整训练指南（新增）

包含：
- GRPO 核心概念（Group Relative Policy Optimization）
- 约束生成机制详解
- 超参数调优建议
- 常见问题排查

---

### 📊 推理模块改进 (`inference/`)

#### 1. **`new_evaluate.py`** - 量化评估脚本（新增）

**功能：**
- 加载 SFT + GRPO 模型
- 使用 Trie 约束生成确保输出有效
- 计算完整评估指标

**评估指标：**
- Hit@K, NDCG@K（标准推荐指标）
- 平均距离误差（地理准确性）
- 层级匹配准确率（Layer 0-3）

**使用方法：**
```bash
python inference/new_evaluate.py
```

**输出示例：**
```
FINAL RESULTS (N=500)
========================================
Mean Distance Error: 11.2 km
--------------------
Hit@1 : 0.3400 | NDCG@1 : 0.3400
Hit@5 : 0.5200 | NDCG@5 : 0.4100
Hit@10: 0.6100 | NDCG@10: 0.4350
--------------------
Hierarchical Accuracy (Top-1):
Layer 0 Match (City/Region): 78%
Layer 1 Match (District)   : 65%
Layer 2 Match (Category)   : 52%
Exact Match (Item)         : 34%
========================================
```

#### 2. **`validate_grpo_with_tsne.py`** - 可视化验证（新增）

**功能：**
- 使用 t-SNE 可视化 RQ-VAE 编码空间
- 验证 GRPO 预测是否落在正确的城市聚类
- 生成预测路径可视化图

**关键修复：**
- 修复 Prompt 构建问题
- 使用 `apply_chat_template` 确保格式正确
- 降低温度 `temperature=0.1` 让模型更确定地输出 ID

**使用方法：**
```bash
python inference/validate_grpo_with_tsne.py
```

**输出：** `data/visualization/grpo_validation_tsne.png`

#### 3. **`evaluate_metrics.py`** - 增强评估（改进）

**新增功能：**
- 层级准确率统计（Layer 0-3 逐层匹配）
- 更详细的评估报告
- 支持自定义测试集大小

**层级准确率说明：**
```python
# 只要 Top-1 预测的前 N 层匹配就算对
if pred[0] == target[0]:  # Layer 0 (城市/大区)
    layer_hits[0] += 1
    if pred[1] == target[1]:  # Layer 1 (街区)
        layer_hits[1] += 1
        if pred[2] == target[2]:  # Layer 2 (类别)
            layer_hits[2] += 1
            if pred[3] == target[3]:  # Layer 3 (商家)
                layer_hits[3] += 1  # 完全命中
```

#### 4. **`check_sft_quality.py`** - SFT 质量检查（改进）

**功能：**
- 快速验证 SFT 模型是否学会了新的 4 层 ID 格式
- 检查生成的 ID 是否在映射表中
- 提供详细的诊断信息

**使用方法：**
```bash
python inference/check_sft_quality.py
```

**输出示例：**
```
✅ Valid ID! Mapped to: Starbucks in Phoenix
   (Prediction is valid, even if not Ground Truth)
```

#### 5. **`check_cluster_purity.py`** - 聚类纯度分析（新增）

**功能：**
- 分析 Layer 2（类别层）的语义纯度
- 统计每个聚类的类别分布
- 验证 RQ-VAE 是否学到了有意义的聚类

#### 6. **`recommend.py`** - 在线推荐（改进）

**功能：**
- 完整的推荐流程实现
- 支持地理位置过滤
- Beam Search 生成多个候选

**使用方法：**
```bash
python inference/recommend.py \
    --user_history user_history.json \
    --user_location location.json \
    --top_k 10
```

---

## 快速开始

### 1. 环境安装
```bash
pip install torch transformers peft trl geopy bitsandbytes scikit-learn matplotlib seaborn
```

### 2. 数据处理
```bash
# Step 1-4: 数据预处理
python data_processing/step1_build_item_profile.py
python data_processing/step2_generate_semantic_ids.py
python data_processing/step3_build_user_sequences.py
python data_processing/step4_construct_prompts.py
```

### 3. 模型训练

#### SFT 训练（必须）
```bash
python training/train_llm.py
```

**检查 SFT 质量：**
```bash
python inference/check_sft_quality.py
```

#### GRPO 训练（可选，提升精准度）
```bash
# 推荐使用优化版本
python training/train_grpo_v2.py

# 或使用基础版本
python training/train_grpo.py
```

### 4. 模型评估
```bash
# 完整评估
python inference/new_evaluate.py

# 可视化验证
python inference/validate_grpo_with_tsne.py

# 聚类质量分析
python inference/check_cluster_purity.py
```

---

## 训练效果

### SFT 阶段
- ✅ 100% 学会 `<c0, c1, c2, suffix>` 格式
- ✅ 验证集推理格式正确率 100%

### GRPO 阶段（前 8% 训练）

| 指标 | 初始值 | 当前值 | 提升 |
|-----|-------|-------|-----|
| Format Reward | 0.50 | 0.50 | ✅ 保持完美 |
| Geo Reward | 0.20 | 0.44 | 🚀 +120% |
| Semantic Reward | 0.13 | 0.24 | 📈 +85% |
| 平均距离误差 | 16km | 11.2km | ✅ -30% |

---

## 核心技术突破

### 1. ID 冲突解决
- **问题**: RQ-VAE 碰撞率 98.1%
- **方案**: 添加 Unique Suffix（第4层）
- **结果**: 碰撞率降至 0.0%

### 2. 地理感知
- **问题**: 模型推荐距离用户很远的商家
- **方案**: 
  - RQ-VAE 编码时融合经纬度
  - GRPO 奖励函数加入地理距离
- **结果**: 平均误差从 16km 降至 11.2km

### 3. 训练稳定性
- **问题**: GRPO 训练不收敛
- **方案**:
  - 密集奖励替代稀疏奖励
  - 层级语义奖励递进
  - 约束生成确保输出有效
- **结果**: 训练稳定收敛

---

## 文件结构

```
HierGR-SeqRec/
├── training/
│   ├── train_llm.py              # [改进] SFT训练（修复labels问题）
│   ├── train_grpo.py             # GRPO训练基础版
│   ├── train_grpo_v2.py          # [新增] GRPO优化版
│   ├── grpo_rewards.py           # 原始奖励函数
│   ├── grpo_rewards_optimized.py # [新增] 优化奖励函数
│   ├── constrained_logits_processor.py  # [新增] 约束生成
│   ├── dataset.py                # 数据集处理
│   └── GRPO_TRAINING_GUIDE.md    # [新增] 训练指南
│
├── inference/
│   ├── new_evaluate.py           # [新增] 量化评估
│   ├── evaluate_metrics.py       # [改进] 增强评估
│   ├── validate_grpo_with_tsne.py # [新增] 可视化验证
│   ├── check_sft_quality.py      # [改进] SFT质量检查
│   ├── check_cluster_purity.py   # [新增] 聚类纯度分析
│   ├── recommend.py              # [改进] 在线推荐
│   └── trie_utils.py             # Trie工具
│
├── data_processing/              # 数据预处理脚本
├── config/                       # 配置文件
└── README.md                     # 本文档
```

---

## 常见问题

### Q1: SFT 训练报错 `ValueError: model did not return a loss`
**A**: 已在 `train_llm.py` 中修复，确保使用最新版本。关键是在 tokenize 时显式创建 `labels` 字段。

### Q2: LoRA + Gradient Checkpointing 不兼容
**A**: 已修复，添加了 `model.enable_input_require_grads()`。

### Q3: GRPO 训练不收敛
**A**: 使用 `train_grpo_v2.py` 和 `grpo_rewards_optimized.py`，采用密集奖励和更好的超参数。

### Q4: 生成的 ID 格式不对
**A**: 
1. 检查 SFT 是否训练充分（运行 `check_sft_quality.py`）
2. 使用约束生成（`constrained_logits_processor.py`）

### Q5: 如何可视化验证模型效果？
**A**: 运行 `python inference/validate_grpo_with_tsne.py`，查看生成的 t-SNE 图。

---

## 参考资料

- **GRPO 论文**: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- **MiniOneRec**: 本项目基于的开源框架
- **TRL 库**: [Hugging Face TRL](https://github.com/huggingface/trl)

---

## 致谢

本项目基于 MiniOneRec 框架，在 Yelp 数据集上进行了大量优化。感谢开源社区的贡献。

---

**最后更新**: 2024-12
**维护者**: [Your Name]
