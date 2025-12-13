# HierGR-SeqRec 项目结构说明

## 📁 目录组织

本项目包含两个推荐模型的实现，**所有原始代码保持在根目录**，同时在 `HierGR/` 和 `PinRec/` 子目录中提供了分类整理的副本。

```
HierGR-SeqRec/
│
├── 📂 原始代码（根目录 - 保持不变）
│   ├── training/              # 所有训练脚本（HierGR + PinRec）
│   ├── inference/             # 所有推理脚本（HierGR + PinRec）
│   ├── models/                # 所有模型定义（主要是 PinRec）
│   ├── data_processing/       # 所有数据处理脚本
│   ├── RQ-VAE/                # RQ-VAE 语义 ID 生成器
│   ├── evaluation/            # 评估工具
│   ├── visualization/         # 可视化工具
│   └── config/                # 配置文件
│
├── 📂 HierGR/                 # 生成式模型（分类副本）
│   ├── training/              # HierGR 训练脚本
│   ├── inference/             # HierGR 推理脚本
│   ├── data_processing/       # HierGR 数据处理
│   ├── docs/                  # HierGR 文档
│   └── README.md              # HierGR 使用指南
│
├── 📂 PinRec/                 # 判别式模型（分类副本）
│   ├── models/                # PinRec 模型定义
│   ├── training/              # PinRec 训练脚本
│   ├── inference/             # PinRec 推理脚本
│   ├── data_processing/       # PinRec 数据处理
│   └── README.md              # PinRec 使用指南
│
├── 📂 backup/                 # 原始代码备份
│   ├── training/
│   ├── inference/
│   ├── models/
│   └── data_processing/
│
├── compare_models_unified.py  # 统一模型对比工具
├── README.md                  # 主文档
├── QUICKSTART.md              # 快速开始
└── requirements.txt           # 依赖列表
```

## 🎯 使用方式

### 方式 1：使用原始路径（推荐）

所有脚本保持原有路径，无需修改导入：

```bash
# HierGR 训练
python training/train_sft_final.py
python training/train_grpo_v3.py

# PinRec 训练
python training/train_pinrec_sft_final.py
python training/train_pinrec_grpo_final.py

# 评估
python inference/evaluate_final_v9.py
python inference/evaluate_pinrec_v7_debug.py
```

### 方式 2：使用分类副本（可选）

如果你想明确区分模型，可以使用子目录中的副本：

```bash
# HierGR 训练
python HierGR/training/train_sft_final.py
python HierGR/training/train_grpo_v3.py

# PinRec 训练
python PinRec/training/train_pinrec_sft_final.py
python PinRec/training/train_pinrec_grpo_final.py
```

## 📋 文件分类

### HierGR（生成式模型）文件

**训练脚本（在 training/ 目录）：**
- `train_sft_final.py` - SFT 训练（推荐）
- `train_sft_optimized.py` - SFT 优化版
- `train_llm.py` - 基础 LLM 训练
- `train_grpo.py` - GRPO 基础版
- `train_grpo_v2.py` - GRPO V2
- `train_grpo_v3.py` - GRPO V3（推荐）
- `train_grpo_v4.py` - GRPO V4
- `train_grpo_v4_1.py` - GRPO V4.1
- `train_grpo_v4_1_resume.py` - GRPO V4.1 恢复
- `train_grpo_v5.py` - GRPO V5
- `grpo_rewards.py` - 奖励函数
- `grpo_rewards_v3.py` - 奖励函数 V3
- `grpo_rewards_optimized.py` - 优化奖励函数
- `constrained_logits_processor.py` - 约束生成
- `merge_model.py` - LoRA 模型合并
- `GRPO_TRAINING_GUIDE.md` - GRPO 训练指南

**推理脚本（在 inference/ 目录）：**
- `check_sft_quality.py` - SFT 质量检查
- `check_sft_only.py` - 仅 SFT 质量检查
- `validate_grpo_with_tsne.py` - t-SNE 可视化
- `evaluate_final.py` - 最终评估
- `evaluate_final_v8.py` - 最终评估 V8
- `evaluate_final_v9.py` - 最终评估 V9
- `evaluate_final_extended.py` - 扩展评估
- `evaluate_sft_final.py` - SFT 评估
- `evaluate_sft_optimized.py` - SFT 优化评估
- `demo_inference.py` - 演示推理
- `trie_utils.py` - Trie 树工具

**数据处理（在 data_processing/ 目录）：**
- `step4_construct_prompts.py` - 构造训练 Prompts

---

### PinRec（判别式模型）文件

**模型定义（在 models/ 目录）：**
- `pinrec_ultimate_v2.py` - PinRec Ultimate V2（推荐）
- `pinrec_ultimate.py` - PinRec Ultimate V1
- `pinrec_llm.py` - PinRec LLM 版本

**训练脚本（在 training/ 目录）：**
- `train_pinrec_sft_final.py` - PinRec SFT（推荐）
- `train_pinrec_grpo_final.py` - PinRec GRPO（推荐）
- `train_pinrec_v7_final.py` - PinRec V7 + LogQ
- `train_pinrec_v6.py` - PinRec V6
- `train_ultimate_v4_stable.py` - Ultimate V4 稳定版
- `train_ultimate_v2.py` - Ultimate V2
- `train_ultimate_v2_logq.py` - Ultimate V2 + LogQ
- `train_ultimate.py` - Ultimate V1

**推理脚本（在 inference/ 目录）：**
- `evaluate_pinrec_v7_debug.py` - PinRec V7 评估调试
- `evaluate_pinrec_v6.py` - PinRec V6 评估
- `evaluate_pinrec_v5.py` - PinRec V5 评估
- `evaluate_pinrec_v3.py` - PinRec V3 评估
- `evaluate_ultimate_v2.py` - Ultimate V2 评估
- `evaluate_ultimate.py` - Ultimate V1 评估
- `evaluate_bulletproof.py` - 防弹评估
- `eval_grpo_aggresive.py` - GRPO 激进评估
- `debug_eval_v2.py` - 调试评估 V2

**数据处理（在 data_processing/ 目录）：**
- `step6_build_ultimate_data_v2.py` - 构建 Ultimate 数据 V2
- `step6_build_ultimate_data.py` - 构建 Ultimate 数据 V1
- `balance_sequences_for_pinrec.py` - 平衡序列
- `balance_ultimate_data.py` - 平衡 Ultimate 数据

---

### 共享文件

**数据处理（在 data_processing/ 目录）：**
- `step1_build_item_profile.py` - 构建商家画像
- `step2_generate_semantic_ids.py` - 生成语义 ID
- `step3_build_user_sequences.py` - 构建用户序列
- `balance_dataset.py` - 平衡数据集
- `analyze_chain_stores.py` - 连锁店分析
- `check_data_v2.py` - 数据检查
- `inspect_sft_data.py` - 检查 SFT 数据

**其他共享模块：**
- `RQ-VAE/` - RQ-VAE 语义 ID 生成器
- `evaluation/` - 评估工具
- `visualization/` - 可视化工具
- `compare_models_unified.py` - 统一模型对比

## 🔍 如何识别文件属于哪个模型？

### HierGR 文件特征：
- 文件名包含：`sft`, `grpo`, `llm`
- 导入 `AutoModelForCausalLM`（生成式模型）
- 使用 `generate()` 方法
- 涉及 Trie 树、约束生成
- 输出格式：`<c0, c1, c2, suffix>`

### PinRec 文件特征：
- 文件名包含：`pinrec`, `ultimate`
- 导入 `ItemTower`, `UserTower`
- 使用双塔架构
- 涉及哈希嵌入、时序编码
- 输出：相似度分数

## 📚 参考文档

- [主 README](README.md) - 项目总览
- [HierGR README](HierGR/README.md) - HierGR 详细说明
- [PinRec README](PinRec/README.md) - PinRec 详细说明
- [快速开始](QUICKSTART.md) - 快速上手指南

## ⚠️ 重要说明

1. **原始代码不变**：根目录下的所有代码保持原样，可以直接使用
2. **分类副本**：`HierGR/` 和 `PinRec/` 目录提供了分类整理的副本，方便查找
3. **备份完整**：`backup/` 目录包含所有原始文件的完整备份
4. **无需修改**：所有脚本的导入路径保持不变，可以直接运行

---

**更新时间**：2024-12-13  
**状态**：✅ 完成
