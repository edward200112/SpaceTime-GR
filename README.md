# HierGR-SeqRec: Hierarchical Generative Retrieval for Sequential Recommendation

## 项目概述

HierGR-SeqRec 将 HierGR 的生成式检索方法改造为生成式序列推荐系统，应用于 Yelp 数据集。

### 核心思想

- **输入**：用户的历史签到/评论序列（而不是搜索词）
- **中间层**：HierGR 的 RQ-VAE 将商家转化为语义 ID（SIDs）
- **模型**：LLM 学习序列规律，预测下一个商家的 Cluster ID
- **输出**：结合当前地理位置（LBS），将预测的 Cluster ID 展开并过滤，推荐具体的商家

### 关键特性

1. **分层语义 ID**：使用 3 层 RQ-VAE，前两层 `<c0, c1>` 作为 Cluster ID
2. **滑动窗口机制**：结合短期窗口（最近10-15个）和长期语义摘要
3. **多任务训练**：
   - 任务 A：序列推荐（预测下一个 Cluster ID）
   - 任务 B：用户偏好摘要
   - 任务 C：语义 ID 对齐（Text ↔ ID）
4. **位置感知推荐**：Cluster ID 可在线展开并根据地理位置过滤

## 项目结构

```
HierGR-SeqRec/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包
├── config/
│   └── config.yaml                     # 全局配置
├── RQ-VAE/                             # 从 HierGR 继承的 RQ-VAE
│   ├── models/
│   │   ├── rq.py                       # 残差量化器
│   │   ├── vq.py                       # 向量量化器
│   │   ├── rqvae.py                    # RQ-VAE 模型
│   │   └── layers.py                   # 基础层
│   ├── trainer.py                      # RQ-VAE 训练器
│   └── main.py                         # RQ-VAE 训练入口
├── data_processing/
│   ├── step1_build_item_profile.py     # 商户画像构建
│   ├── step2_generate_semantic_ids.py  # 语义 ID 生成
│   ├── step3_build_user_sequences.py   # 用户序列构建（滑动窗口）
│   └── step4_construct_prompts.py      # 多任务 Prompt 构造
├── training/
│   ├── train_llm.py                    # LLM 训练脚本
│   └── dataset.py                      # 数据集加载器
├── inference/
│   └── recommend.py                    # 在线推理
└── data/                               # 数据目录（gitignore）
    ├── raw/                            # 原始 Yelp 数据
    ├── processed/                      # 处理后的数据
    ├── embeddings/                     # 商户语义向量
    ├── rqvae_ckpt/                     # RQ-VAE 模型检查点
    └── llm_ckpt/                       # LLM 模型检查点
```

## 使用流程

### 1. 数据准备

将 Yelp 数据集放入 `data/raw/` 目录：
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

### 2. 数据处理流水线

```bash
# Step 1: 构建商户画像（聚合 Name + Categories + Attributes + Top Reviews）
python data_processing/step1_build_item_profile.py

# Step 2: 生成语义向量并训练 RQ-VAE
python data_processing/step2_generate_semantic_ids.py

# Step 3: 构建用户序列（滑动窗口 + 长期摘要）
python data_processing/step3_build_user_sequences.py

# Step 4: 构造多任务 Prompt（任务 A/B/C 混合）
python data_processing/step4_construct_prompts.py
```

### 3. 模型训练

```bash
# 训练 LLM（基于 Qwen/Llama）
python training/train_llm.py --config config/config.yaml
```

### 4. 在线推理

```bash
# 给定用户历史和当前位置，推荐商家
python inference/recommend.py --user_id U123 --location "New York" --top_k 10
```

## 技术细节

### RQ-VAE 设计

- **3 层量化**：每层 64/128 个 codebook
- **Cluster ID**：前两层 `<c0, c1>` 表示语义簇（如"纽约高档日料"）
- **残差放大**：α = [1.1, 1.05, 1.0]，增强早期层表示能力
- **Sinkhorn 约束**：避免 codebook collapse

### 滑动窗口策略

- **短期窗口**：最近 10-15 个交互（保留详细信息）
- **长期摘要**：久远历史压缩为文本画像（如"User previously enjoyed diverse cuisines..."）
- **数据增强**：训练时滑动生成多条样本

### Prompt 设计

#### 任务 A：序列推荐
```
Instruction: Based on the user's visit history, predict the Semantic Cluster ID of the next place.
User History:
1. [Starbucks] (Coffee) -> <05, 11>
2. [AMC Cinema] (Entertainment) -> <88, 21>
Response: <12, 45>
```

#### 任务 B：偏好摘要
```
Instruction: Summarize the user's preferences.
User History:
1. [Joe's Pizza] (Rating: 5/5)
2. [Burger King] (Rating: 4/5)
Response: The user enjoys casual fast-food dining.
```

#### 任务 C：ID 对齐
```
Instruction: What is the Semantic Cluster ID for "Late night Italian pizza in NYC"?
Response: <12, 45>
```

## 依赖环境

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Sentence-Transformers (for BERT embeddings)
- FAISS (for K-Means)
- DeepSpeed (optional, for large-scale training)

## 引用

本项目结合了以下研究：
- **HierGR**: Hierarchical Generative Retrieval
- **MiniOneRec**: Open-Source Generative Recommendation Framework

## 许可证

MIT License
