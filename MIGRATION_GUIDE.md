# 项目使用指南

本项目包含两个推荐模型的实现。**所有原始代码保持在根目录不变**，无需修改任何导入路径。同时提供了分类副本方便查找。

## 📁 项目结构

```
HierGR-SeqRec/
├── 📂 原始代码（根目录 - 保持不变）
│   ├── training/              # 所有训练脚本
│   ├── inference/             # 所有推理脚本
│   ├── models/                # 所有模型定义
│   └── data_processing/       # 所有数据处理
│
├── 📂 HierGR/                 # 生成式模型（分类副本）
│   ├── training/              # HierGR 训练脚本副本
│   ├── inference/             # HierGR 推理脚本副本
│   └── README.md              # HierGR 使用指南
│
├── 📂 PinRec/                 # 判别式模型（分类副本）
│   ├── models/                # PinRec 模型定义副本
│   ├── training/              # PinRec 训练脚本副本
│   └── README.md              # PinRec 使用指南
│
└── 📂 backup/                 # 完整备份
```

## ✅ 重要说明

1. **原始代码不变**：根目录下的所有代码保持原样
2. **无需修改导入**：所有脚本的导入路径保持不变
3. **分类副本**：`HierGR/` 和 `PinRec/` 提供分类副本，方便查找
4. **完整备份**：`backup/` 目录包含所有原始文件

## 🎯 推荐使用方式

### 直接使用原始路径（推荐）

所有脚本保持原有路径，导入路径不变：

```python
# PinRec 脚本 - 导入保持不变
from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# HierGR 脚本 - 导入保持不变
from transformers import AutoModelForCausalLM
```

## 🚀 运行方式

### 方式 1：使用原始路径（推荐）

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

```bash
# HierGR 训练
python HierGR/training/train_sft_final.py
python HierGR/training/train_grpo_v3.py

# PinRec 训练
python PinRec/training/train_pinrec_sft_final.py
python PinRec/training/train_pinrec_grpo_final.py
```

## 📋 文件分类参考

### 如何识别文件属于哪个模型？

**HierGR（生成式）文件特征：**
- 文件名包含：`sft`, `grpo`, `llm`
- 导入 `AutoModelForCausalLM`
- 使用 `generate()` 方法
- 涉及 Trie 树、约束生成
- 输出格式：`<c0, c1, c2, suffix>`

**PinRec（判别式）文件特征：**
- 文件名包含：`pinrec`, `ultimate`
- 导入 `ItemTower`, `UserTower`
- 使用双塔架构
- 涉及哈希嵌入、时序编码
- 输出：相似度分数

详细分类列表请查看：
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 完整的文件分类
- [HierGR/README.md](HierGR/README.md) - HierGR 文件列表
- [PinRec/README.md](PinRec/README.md) - PinRec 文件列表

## 📊 数据路径更新

### 配置文件路径

**旧路径：**
```python
CONFIG_PATH = "./config/config.yaml"
```

**新路径（从 HierGR/PinRec 目录运行）：**
```python
CONFIG_PATH = "../config/config.yaml"
```

**新路径（从项目根目录运行）：**
```python
CONFIG_PATH = "./config/config.yaml"
```

### 数据文件路径

数据文件路径保持不变，因为 `data/` 目录仍在项目根目录：

```python
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
OUTPUT_DIR = "/workspace/data/llm_ckpt_sft_v2_optimized"
```

## 🚀 快速修复脚本

创建一个自动修复导入路径的脚本：

```python
# fix_imports.py
import os
import re

def fix_pinrec_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复 PinRec 模型导入
    content = re.sub(
        r"from models\.pinrec",
        r"from PinRec.models.pinrec",
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

# 修复所有 PinRec 脚本
for root, dirs, files in os.walk("PinRec"):
    for file in files:
        if file.endswith(".py"):
            fix_pinrec_imports(os.path.join(root, file))
```

## 🔍 常见问题

### Q1: 运行脚本时提示 "ModuleNotFoundError"

**问题：**
```
ModuleNotFoundError: No module named 'models'
```

**解决方案：**
1. 检查当前工作目录
2. 确保从正确的目录运行脚本
3. 更新 `sys.path.append()` 路径

### Q2: 找不到配置文件

**问题：**
```
FileNotFoundError: [Errno 2] No such file or directory: './config/config.yaml'
```

**解决方案：**
```python
# 使用绝对路径
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
```

### Q3: 找不到数据文件

**问题：**
```
FileNotFoundError: data/processed/train_prompts.jsonl
```

**解决方案：**
数据文件使用绝对路径：
```python
DATA_PATH = "/workspace/data/processed/train_prompts.jsonl"
```

## 📋 检查清单

在运行脚本前，请检查：

- [ ] 确认当前工作目录
- [ ] 检查导入路径是否正确
- [ ] 确认配置文件路径
- [ ] 确认数据文件路径
- [ ] 检查输出目录路径

## 🔄 回滚到旧结构

如果遇到问题，可以临时回滚：

```bash
# 从备份恢复
cp -r backup/training/* training/
cp -r backup/inference/* inference/
cp -r backup/models/* models/
cp -r backup/data_processing/* data_processing/
```

## 📚 参考文档

- [重组总结](REORGANIZATION_SUMMARY.md) - 完整的重组说明
- [HierGR README](HierGR/README.md) - HierGR 使用指南
- [PinRec README](PinRec/README.md) - PinRec 使用指南
- [主 README](README.md) - 项目总览

---

**更新时间**：2024-12-13  
**状态**：✅ 完成
