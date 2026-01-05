# HierGR-SeqRec: 层级生成式序列推荐系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HierGR-SeqRec** 是一个基于**层级语义ID**和**生成式推荐**的深度学习框架，支持多种推荐模型架构和训练策略。项目包含从数据处理到模型训练、评估的完整流水线，特别专注于Yelp数据集上的序列推荐任务。

## 🎯 项目概述

### 核心特性
- **🔢 层级语义ID**: 使用RQ-VAE将物品编码为多层语义表示
- **🤖 双架构支持**: HierGR(生成式) + PinRec(判别式)
- **🚀 强化学习**: GRPO、NTP等多种强化学习优化方法
- **📊 完整评估**: Hit@K、NDCG@K等多种评估指标
- **🎨 可视化**: t-SNE、训练曲线、类别分布等可视化工具

### 技术栈
- **语言模型**: Qwen2.5、Llama等主流LLM
- **推荐模型**: SASRec、PinRec、Ultimate等
- **量化技术**: RQ-VAE、Vector Quantization
- **强化学习**: GRPO、NTP、Hard Mining

## 📁 项目结构

```
HierGR-SeqRec/
│
├── 📂 核心框架 (根目录)
│   ├── data_processing/        # 数据处理流水线
│   ├── training/              # 训练脚本集合
│   ├── inference/             # 推理与评估
│   ├── RQ-VAE/               # 语义ID生成器
│   ├── models/               # 模型定义
│   ├── evaluation/           # 评估工具
│   └── visualization/        # 可视化脚本
│
├── 📂 Rank-GRPO/             # 高级排序与强化学习
│   ├── GRPO/                # SA-Rank-GRPO实现
│   ├── HardMiningGRPONTP/   # 困难样本挖掘+NTP
│   ├── HardMiningGRPORerank/ # 重排序方法
│   ├── TeacherModel/        # SASRec教师模型
│   └── SFT/                 # 监督微调
│
├── 📂 分类副本 (可选使用)
│   ├── HierGR/              # 生成式模型分类
│   └── PinRec/              # 判别式模型分类
│
└── 📂 配置与文档
    ├── config/              # 全局配置
    ├── examples/            # 使用示例
    └── docs/               # 详细文档
```

## 🔧 核心组件详解

### 1. 数据处理流水线 (`data_processing/`)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `step1_build_item_profile.py` | 构建商家画像 | Yelp原始数据 | 商家特征文件 |
| `step2_generate_semantic_ids.py` | 训练RQ-VAE生成语义ID | 商家特征 | SID映射文件 |
| `step3_build_user_sequences.py` | 构建用户行为序列 | 用户历史数据 | 序列数据 |
| `step4_construct_prompts.py` | 构造训练Prompts | 序列+SID | 训练格式数据 |
| `balance_dataset.py` | 长尾类别平衡采样 | 原始训练数据 | 平衡后数据 |
| `analyze_chain_stores.py` | 连锁店分析工具 | 商家数据 | 分析报告 |

### 2. RQ-VAE语义编码 (`RQ-VAE/`)

```
RQ-VAE/
├── models/
│   ├── rqvae.py           # RQ-VAE主模型
│   ├── quantizers.py      # Sinkhorn-Knopp量化器
│   ├── encoder_decoder.py # 编码器-解码器
│   └── vq.py             # 向量量化基础组件
└── trainer.py            # RQ-VAE训练器
```

**核心技术**:
- **残差量化**: 3层量化 + 1层后缀，消除ID冲突
- **Sinkhorn-Knopp**: 优化码本分配平衡性
- **层级语义**: Region→District→Category→Unique

### 3. 训练脚本集合 (`training/`)

#### 🎯 HierGR (生成式模型)
| 脚本 | 阶段 | 特点 | 推荐度 |
|------|------|------|--------|
| `train_sft_final.py` | SFT | 监督微调最终版 | ⭐⭐⭐⭐⭐ |
| `train_grpo_v5.py` | RL | GRPO V5加权版 | ⭐⭐⭐⭐⭐ |
| `train_grpo_v4_3_logit_masking.py` | RL | Logit掩码约束 | ⭐⭐⭐⭐ |
| `constrained_logits_processor.py` | 工具 | Trie树约束生成 | ⭐⭐⭐⭐ |

#### 🎯 PinRec (判别式模型)
| 脚本 | 模型 | 特点 | 推荐度 |
|------|------|------|--------|
| `train_pinrec_sft_final.py` | PinRec | 双塔架构SFT | ⭐⭐⭐⭐⭐ |
| `train_pinrec_grpo_final.py` | PinRec | 双塔架构GRPO | ⭐⭐⭐⭐⭐ |
| `train_ultimate_v4_stable.py` | Ultimate | 稳定训练版本 | ⭐⭐⭐⭐ |

### 4. 推理与评估 (`inference/`)

#### 🔍 质量检查
- `check_sft_quality.py` - SFT模型质量检查
- `check_cluster_purity.py` - 聚类纯度分析
- `validate_grpo_with_tsne.py` - t-SNE可视化验证

#### 📊 性能评估
- `evaluate_final_v9.py` - 最新评估脚本 (推荐)
- `new_evaluate.py` - 综合评估工具
- `evaluate_metrics.py` - 层级准确率统计

#### 🎮 在线推理
- `recommend.py` - 在线推荐接口
- `demo_inference.py` - 演示脚本
- `trie_utils.py` - Trie树约束工具

### 5. Rank-GRPO高级组件

#### 🎯 GRPO核心 (`Rank-GRPO/GRPO/`)
- `SA-Rank-GRPO.py` - Self-Attention排序GRPO (26KB核心实现)
- `eval_grpo_v2.py` - GRPO专用评估

#### 🔍 困难样本挖掘 (`Rank-GRPO/HardMiningGRPONTP/`)
- `train_grpo_ntp.py` - NTP(Next Token Prediction)训练
- `reward_ntp_itemid.py` - 基于物品ID的奖励函数
- `build_teacher_topk.py` - 构建教师模型Top-K

#### 👨‍🏫 教师模型 (`Rank-GRPO/TeacherModel/`)
- `train_sasrec.py` - SASRec教师模型训练 (43KB大型脚本)
- `eval_sasrec_strict.py` - 严格评估
- `export_teacher.py` - 导出教师模型知识

## 🚀 快速开始

### 环境配置
```bash
# 克隆项目
git clone <repository-url>
cd HierGR-SeqRec

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖 (如需要)
pip install transformers accelerate deepspeed wandb
```

### 完整流水线
```bash
# 1. 数据处理
python data_processing/step1_build_item_profile.py
python data_processing/step2_generate_semantic_ids.py  
python data_processing/step3_build_user_sequences.py
python data_processing/step4_construct_prompts.py

# 2. 模型训练
# HierGR路线
python training/train_sft_final.py          # SFT阶段
python training/train_grpo_v5.py            # GRPO阶段

# PinRec路线  
python training/train_pinrec_sft_final.py   # PinRec SFT
python training/train_pinrec_grpo_final.py  # PinRec GRPO

# 3. 模型评估
python inference/evaluate_final_v9.py       # 综合评估
python inference/new_evaluate.py            # 详细评估

# 4. 模型推理
python inference/recommend.py               # 在线推荐
python examples/quick_demo.py               # 快速演示
```

### 高级功能
```bash
# Rank-GRPO高级训练
python Rank-GRPO/GRPO/SA-Rank-GRPO.py
python Rank-GRPO/HardMiningGRPONTP/train_grpo_ntp.py

# 教师模型训练
python Rank-GRPO/TeacherModel/train_sasrec.py

# 可视化分析
python visualization/plot_training_curves.py
python visualization/plot_tsne.py
```

## 📊 模型架构对比

| 模型 | 架构类型 | 编码方式 | 训练策略 | 推理方式 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| **HierGR** | 生成式 | 语义ID | SFT→GRPO | 自回归生成 | 序列建模、多样性推荐 |
| **PinRec** | 判别式 | 双塔嵌入 | SFT→GRPO | 相似度计算 | 大规模召回、实时推荐 |
| **Ultimate** | 混合式 | 语义+嵌入 | 多阶段训练 | 混合推理 | 复杂场景、高精度 |

## 🎯 性能指标

### 数据集统计
- **Yelp数据集**: 1.6M+ 评论, 160K+ 商家, 1.9M+ 用户
- **语义ID数量**: 256³ = 16.7M 理论空间，实际使用 ~100K
- **平均序列长度**: 10-15个交互

### 评估指标
- **Hit@K**: K=1,3,5,10的命中率
- **NDCG@K**: K=1,3,5,10的归一化累计增益
- **层级准确率**: 分层语义ID的准确率统计
- **地理相关性**: 基于地理距离的推荐质量

## 🛠️ 高级配置

### 模型路径配置 (`MODEL_PATHS.md`)
```yaml
base_model_path: "Qwen/Qwen2.5-1.5B-Instruct"
rqvae_checkpoint: "data/rqvae_ckpt/best_model.pth"
sft_checkpoint: "data/llm_ckpt_sft_v5_balanced/final"
grpo_checkpoint: "data/grpo_v5_weighted/final"
```

### 训练参数调优
```yaml
# SFT阶段
learning_rate: 5e-5
batch_size: 8-16
gradient_accumulation_steps: 8
max_seq_length: 512

# GRPO阶段  
learning_rate: 1e-6
beta: 0.04
temperature: 0.8
top_k: 50
```

## 📈 可视化与分析

### 训练监控
- `visualization/plot_training_curves.py` - 损失曲线、准确率曲线
- `wandb` 集成 - 实时训练监控
- 梯度范数、学习率衰减可视化

### 模型分析
- `visualization/plot_tsne.py` - 语义ID空间分布
- `visualization/plot_category_distribution.py` - 类别平衡性分析
- `inference/check_cluster_purity.py` - 聚类质量评估

## 🔧 故障排除

### 常见问题
1. **CUDA OOM**: 减小batch_size，增加gradient_accumulation_steps
2. **收敛缓慢**: 检查学习率设置，使用学习率调度器
3. **生成无效ID**: 检查Trie树约束，验证词汇表扩展

### 调试工具
- `inspect_data.py` - 数据完整性检查
- `inference/debug_eval_v2.py` - 评估过程调试
- `training/constrained_logits_processor.py` - 生成约束调试

## 📚 相关文档

- [快速开始指南](QUICKSTART.md)
- [项目结构说明](PROJECT_STRUCTURE.md)  
- [模型路径配置](MODEL_PATHS.md)
- [GRPO训练指南](training/GRPO_TRAINING_GUIDE.md)
- [评估工具说明](evaluation/EVALUATION_GUIDE.md)

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)  
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Yelp数据集** - 提供高质量的评论和商家数据
- **Transformers社区** - 预训练模型和工具
- **PyTorch团队** - 深度学习框架支持

---

**最后更新**: 2024-12-30  
**状态**: ✅ 生产就绪  
**版本**: v2.0

如有问题或建议，请提交Issue或联系维护团队。
