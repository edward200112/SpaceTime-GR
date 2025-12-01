# 模型下载指南

## 需要下载的模型

HierGR-SeqRec 项目需要 **2 个模型**：

### 1. BERT 模型（用于生成商家文本向量）

**使用场景**：Step 2 - 将商家文本转换为 768 维向量

**推荐模型**（二选一）：
- `sentence-transformers/all-mpnet-base-v2` (推荐，768维)
- `sentence-transformers/all-MiniLM-L6-v2` (轻量级，384维)

### 2. LLM 基座模型（用于序列推荐训练）

**使用场景**：Step 5 - LLM 训练

**推荐模型**（根据配置文件）：
- `Qwen/Qwen2.5-7B-Instruct` (默认)
- `meta-llama/Llama-3-8B-Instruct` (备选)

---

## 自动下载（推荐）

### 方式 1：运行时自动下载

**无需手动下载**，代码会自动从 Hugging Face 下载模型。

#### BERT 模型（Step 2）
```python
# step2_generate_semantic_ids.py 会自动下载
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# 自动下载到：~/.cache/huggingface/hub/
```

#### LLM 模型（Step 5）
```python
# train_llm.py 会自动下载
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
# 自动下载到：~/.cache/huggingface/hub/
```

**优点**：简单，自动管理
**缺点**：首次运行时需要联网，下载较慢

---

## 手动下载（离线环境）

如果服务器无法访问 Hugging Face，需要手动下载。

### 步骤 1: 下载 BERT 模型

#### 使用 Hugging Face CLI

```bash
# 安装 CLI 工具
pip install huggingface-hub

# 下载模型
huggingface-cli download sentence-transformers/all-mpnet-base-v2 \
  --local-dir ./models/all-mpnet-base-v2 \
  --local-dir-use-symlinks False
```

**模型文件结构**：
```
models/all-mpnet-base-v2/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── ...
```

**放置位置**：项目根目录下创建 `models/` 文件夹

**修改代码**：
编辑 `data_processing/step2_generate_semantic_ids.py` 第 53 行：
```python
# 原代码：
self.bert_model = SentenceTransformer(model_name)

# 改为：
self.bert_model = SentenceTransformer('./models/all-mpnet-base-v2')
```

---

### 步骤 2: 下载 LLM 模型

#### 使用 Hugging Face CLI

```bash
# 下载 Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

**模型文件结构**：
```
models/Qwen2.5-7B-Instruct/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── tokenizer_config.json
└── ...
```

**放置位置**：同样放在 `models/` 文件夹

**修改配置**：
编辑 `config/config.yaml` 第 84 行：
```yaml
# 原配置：
model_name: "Qwen/Qwen2.5-7B-Instruct"

# 改为：
model_name: "./models/Qwen2.5-7B-Instruct"
```

---

## 使用国内镜像加速

### 方法 1: HF-Mirror (推荐)

```bash
# 临时使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python data_processing/step2_generate_semantic_ids.py
```

### 方法 2: ModelScope

```bash
# 从 ModelScope 下载
pip install modelscope

# 下载 BERT
modelscope download --model_id=sentence-transformers/all-mpnet-base-v2 \
  --local_dir ./models/all-mpnet-base-v2

# 下载 Qwen
modelscope download --model_id=qwen/Qwen2.5-7B-Instruct \
  --local_dir ./models/Qwen2.5-7B-Instruct
```

---

## 推荐的项目结构

```
HierGR-SeqRec/
├── models/                                    # 手动下载的模型（可选）
│   ├── all-mpnet-base-v2/                     # BERT 模型
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   └── Qwen2.5-7B-Instruct/                   # LLM 模型
│       ├── config.json
│       ├── model-*.safetensors
│       └── ...
├── data/
│   ├── raw/                                   # Yelp 原始数据
│   ├── processed/                             # 处理后的数据
│   ├── embeddings/                            # 生成的向量
│   ├── rqvae_ckpt/                            # RQ-VAE 检查点
│   └── llm_ckpt/                              # LLM 微调后的 LoRA 权重
├── config/
├── data_processing/
└── ...
```

---

## 模型大小估算

| 模型 | 大小 | 用途 |
|------|------|------|
| sentence-transformers/all-mpnet-base-v2 | ~420 MB | BERT 文本编码 |
| sentence-transformers/all-MiniLM-L6-v2 | ~90 MB | BERT 文本编码（轻量） |
| Qwen/Qwen2.5-7B-Instruct | ~15 GB | LLM 训练 |
| meta-llama/Llama-3-8B-Instruct | ~16 GB | LLM 训练（备选） |

**总存储需求**：~16 GB（BERT + LLM）

---

## 常见问题

### Q1: 如何选择 BERT 模型？

- **all-mpnet-base-v2** (推荐)：768 维，效果最好，与 config.yaml 中的 `in_dim: 768` 匹配
- **all-MiniLM-L6-v2**：384 维，速度快，但需要修改配置：
  ```yaml
  rqvae:
    in_dim: 384  # 改为 384
  ```

### Q2: 如何选择 LLM 模型？

- **Qwen2.5-7B-Instruct** (推荐)：中文理解能力强，适合本地生活推荐
- **Llama-3-8B-Instruct**：开源社区支持更好，英文性能更强

### Q3: 可以使用更小的模型吗？

可以！如果显存有限：
- **BERT**：使用 `all-MiniLM-L6-v2` (90 MB)
- **LLM**：使用 `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) 或 `microsoft/phi-2` (2.7B)

修改 `config.yaml`：
```yaml
llm:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
```

### Q4: 模型文件存放在哪里？

**自动下载时**：
- Windows: `C:\Users\<用户名>\.cache\huggingface\hub\`
- Linux/Mac: `~/.cache/huggingface/hub/`

**手动下载时**：
- 建议放在项目根目录下的 `models/` 文件夹

### Q5: 如何验证模型下载成功？

```bash
# 查看缓存的模型
ls ~/.cache/huggingface/hub/

# 或查看手动下载的模型
ls ./models/
```

---

## 下载脚本

创建 `download_models.sh`（Linux/Mac）：

```bash
#!/bin/bash

echo "Downloading models for HierGR-SeqRec..."

# 创建模型目录
mkdir -p models

# 下载 BERT
echo "Downloading BERT model..."
huggingface-cli download sentence-transformers/all-mpnet-base-v2 \
  --local-dir ./models/all-mpnet-base-v2 \
  --local-dir-use-symlinks False

# 下载 LLM
echo "Downloading LLM model..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False

echo "✓ All models downloaded successfully!"
```

Windows PowerShell 版本 `download_models.ps1`：

```powershell
Write-Host "Downloading models for HierGR-SeqRec..."

# 创建模型目录
New-Item -ItemType Directory -Force -Path models

# 下载 BERT
Write-Host "Downloading BERT model..."
huggingface-cli download sentence-transformers/all-mpnet-base-v2 `
  --local-dir ./models/all-mpnet-base-v2 `
  --local-dir-use-symlinks False

# 下载 LLM
Write-Host "Downloading LLM model..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct `
  --local-dir ./models/Qwen2.5-7B-Instruct `
  --local-dir-use-symlinks False

Write-Host "✓ All models downloaded successfully!"
```

---

## 总结

### 快速开始（推荐）

**不需要手动下载**，直接运行代码即可：

```bash
# 首次运行会自动下载模型
python data_processing/step2_generate_semantic_ids.py
python training/train_llm.py
```

### 离线环境

如果无法访问 Hugging Face：

1. 在有网络的机器上下载模型
2. 复制 `models/` 文件夹到目标机器
3. 修改代码中的模型路径为本地路径

### 国内用户

使用镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者添加到 `~/.bashrc`（永久生效）。
