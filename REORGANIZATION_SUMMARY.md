# 代码重组总结

## 📋 重组概述

HierGR-SeqRec 项目包含两个推荐模型的实现。**所有原始代码保持在根目录不变**，同时在 `HierGR/` 和 `PinRec/` 子目录中提供了按模型分类的副本，方便查找和理解。

## 📁 新的目录结构

```
HierGR-SeqRec/
├── HierGR/                    # 生成式模型
│   ├── training/              # 训练脚本（11 个文件）
│   ├── inference/             # 推理评估（11 个文件）
│   ├── data_processing/       # 数据处理（1 个文件）
│   ├── docs/                  # 文档（1 个文件）
│   └── README.md              # HierGR 说明文档
│
├── PinRec/                    # 判别式模型
│   ├── models/                # 模型定义（3 个文件）
│   ├── training/              # 训练脚本（8 个文件）
│   ├── inference/             # 推理评估（9 个文件）
│   ├── data_processing/       # 数据处理（4 个文件）
│   └── README.md              # PinRec 说明文档
│
├── backup/                    # 原始代码备份
│   ├── training/              # 所有原始训练脚本
│   ├── inference/             # 所有原始推理脚本
│   ├── models/                # 所有原始模型文件
│   └── data_processing/       # 所有原始数据处理脚本
│
├── RQ-VAE/                    # 共享：RQ-VAE 模型
├── evaluation/                # 共享：评估工具
├── visualization/             # 共享：可视化工具
├── data_processing/           # 共享：通用数据处理
├── config/                    # 共享：配置文件
├── examples/                  # 共享：示例
├── data/                      # 共享：数据目录
│
├── compare_models_unified.py  # 统一模型对比工具
├── README.md                  # 主文档
├── QUICKSTART.md              # 快速开始
├── MODEL_PATHS.md             # 模型路径配置
└── requirements.txt           # 依赖列表
```

## 🔄 文件分类详情

### HierGR（生成式模型）- 23 个文件

#### 训练脚本（11 个）
1. `train_sft_final.py` - SFT 训练（推荐）
2. `train_sft_optimized.py` - SFT 优化版
3. `train_llm.py` - 基础 LLM 训练
4. `train_grpo.py` - GRPO 基础版
5. `train_grpo_v2.py` - GRPO V2
6. `train_grpo_v3.py` - GRPO V3（推荐）
7. `train_grpo_v4.py` - GRPO V4
8. `train_grpo_v4_1.py` - GRPO V4.1
9. `train_grpo_v4_1_resume.py` - GRPO V4.1 恢复
10. `train_grpo_v5.py` - GRPO V5
11. `merge_model.py` - LoRA 模型合并

#### 训练辅助（4 个）
1. `grpo_rewards.py` - 奖励函数
2. `grpo_rewards_v3.py` - 奖励函数 V3
3. `grpo_rewards_optimized.py` - 优化奖励函数
4. `constrained_logits_processor.py` - 约束生成

#### 推理评估（11 个）
1. `check_sft_quality.py` - SFT 质量检查
2. `check_sft_only.py` - 仅 SFT 质量检查
3. `validate_grpo_with_tsne.py` - t-SNE 可视化
4. `evaluate_final.py` - 最终评估
5. `evaluate_final_v8.py` - 最终评估 V8
6. `evaluate_final_v9.py` - 最终评估 V9
7. `evaluate_final_extended.py` - 扩展评估
8. `evaluate_sft_final.py` - SFT 评估
9. `evaluate_sft_optimized.py` - SFT 优化评估
10. `demo_inference.py` - 演示推理
11. `trie_utils.py` - Trie 树工具

#### 数据处理（1 个）
1. `step4_construct_prompts.py` - 构造训练 Prompts

#### 文档（1 个）
1. `GRPO_TRAINING_GUIDE.md` - GRPO 训练指南

---

### PinRec（判别式模型）- 24 个文件

#### 模型定义（3 个）
1. `pinrec_ultimate_v2.py` - PinRec Ultimate V2（推荐）
2. `pinrec_ultimate.py` - PinRec Ultimate V1
3. `pinrec_llm.py` - PinRec LLM 版本

#### 训练脚本（8 个）
1. `train_pinrec_sft_final.py` - PinRec SFT（推荐）
2. `train_pinrec_grpo_final.py` - PinRec GRPO（推荐）
3. `train_pinrec_v7_final.py` - PinRec V7 + LogQ
4. `train_pinrec_v6.py` - PinRec V6
5. `train_ultimate_v4_stable.py` - Ultimate V4 稳定版
6. `train_ultimate_v2.py` - Ultimate V2
7. `train_ultimate_v2_logq.py` - Ultimate V2 + LogQ
8. `train_ultimate.py` - Ultimate V1

#### 推理评估（9 个）
1. `evaluate_pinrec_v7_debug.py` - PinRec V7 评估调试
2. `evaluate_pinrec_v6.py` - PinRec V6 评估
3. `evaluate_pinrec_v5.py` - PinRec V5 评估
4. `evaluate_pinrec_v3.py` - PinRec V3 评估
5. `evaluate_ultimate_v2.py` - Ultimate V2 评估
6. `evaluate_ultimate.py` - Ultimate V1 评估
7. `evaluate_bulletproof.py` - 防弹评估
8. `eval_grpo_aggresive.py` - GRPO 激进评估
9. `debug_eval_v2.py` - 调试评估 V2

#### 数据处理（4 个）
1. `step6_build_ultimate_data_v2.py` - 构建 Ultimate 数据 V2
2. `step6_build_ultimate_data.py` - 构建 Ultimate 数据 V1
3. `balance_sequences_for_pinrec.py` - 平衡序列
4. `balance_ultimate_data.py` - 平衡 Ultimate 数据

---

### 共享文件（保留在根目录）

#### RQ-VAE（整个目录）
- 两个模型都使用的语义 ID 生成器

#### 数据处理（共享）
- `step1_build_item_profile.py` - 构建商家画像
- `step2_generate_semantic_ids.py` - 生成语义 ID
- `step3_build_user_sequences.py` - 构建用户序列
- `balance_dataset.py` - 平衡数据集
- `analyze_chain_stores.py` - 连锁店分析
- `check_data_v2.py` - 数据检查
- `inspect_sft_data.py` - 检查 SFT 数据

#### 评估工具（整个目录）
- `evaluation/` - 通用评估工具

#### 可视化（整个目录）
- `visualization/` - Codebook 可视化等

#### 统一对比工具
- `compare_models_unified.py` - 一键对比 HierGR vs PinRec

#### 配置和文档
- `config/` - 全局配置
- `examples/` - 示例文件
- `README.md` - 主文档
- `QUICKSTART.md` - 快速开始
- `MODEL_PATHS.md` - 模型路径
- `requirements.txt` - 依赖列表

---

## 🎯 使用指南

### 方式 1：使用原始路径（推荐）

所有脚本保持原有路径，无需修改：

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

# 对比两个模型
python compare_models_unified.py
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

---

## ✅ 重组优势

1. **原始代码不变**：根目录下所有代码保持原样，可以直接使用
2. **分类副本**：`HierGR/` 和 `PinRec/` 提供分类整理的副本，方便查找
3. **完整的备份**：所有原始代码保存在 `backup/` 目录
4. **独立的文档**：每个模型都有自己的 README 说明
5. **无需修改**：所有脚本的导入路径保持不变
6. **统一对比工具**：`compare_models_unified.py` 可以一键对比两个模型

---

## 🔍 查找文件

### 如果你想找某个文件：

1. **查看备份**：所有原始文件都在 `backup/` 目录
2. **查看分类文档**：`FILE_CLASSIFICATION.md` 有完整的文件分类列表
3. **查看各自 README**：
   - `HierGR/README.md` - HierGR 相关文件
   - `PinRec/README.md` - PinRec 相关文件

---

## 📝 注意事项

### 导入路径保持不变

**重要**：根目录下的所有脚本保持原有的导入路径，无需修改！

**根目录脚本（推荐使用）：**
```python
# 这些导入路径保持不变
from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
from transformers import AutoModelForCausalLM
```

**子目录副本脚本（如果使用）：**
```python
# 如果从 PinRec/ 子目录运行，导入路径也保持不变
from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
```

---

## 🔄 备份说明

所有原始文件已完整备份到 `backup/` 目录：

```
backup/
├── training/              # 所有训练脚本备份
├── inference/             # 所有推理脚本备份
├── models/                # 所有模型文件备份
└── data_processing/       # 所有数据处理脚本备份
```

如果需要恢复某个文件，可以从 `backup/` 目录复制。

---

## 📊 统计信息

- **总文件数**：47 个代码文件被重新组织
- **HierGR 文件**：23 个
- **PinRec 文件**：24 个
- **备份文件**：所有原始文件
- **共享文件**：保留在根目录

---

**重组完成时间**：2024-12-13  
**备份位置**：`HierGR-SeqRec/backup/`  
**状态**：✅ 完成
