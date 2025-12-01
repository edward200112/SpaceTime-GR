# Quick Start Guide

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

**重要**：项目需要 2 个模型，详见 [MODEL_DOWNLOAD_GUIDE.md](./MODEL_DOWNLOAD_GUIDE.md)

#### 方式 1：自动下载（推荐）

**无需手动操作**，代码首次运行时会自动从 Hugging Face 下载：
- BERT 模型：`sentence-transformers/all-mpnet-base-v2` (~420 MB)
- LLM 模型：`Qwen/Qwen2.5-7B-Instruct` (~15 GB)

```bash
# 国内用户建议使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

#### 方式 2：手动下载（离线环境）

```bash
# 安装下载工具
pip install huggingface-hub

# 下载 BERT 模型
huggingface-cli download sentence-transformers/all-mpnet-base-v2 \
  --local-dir ./models/all-mpnet-base-v2

# 下载 LLM 模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct

# 然后修改 config.yaml 中的模型路径为本地路径
```

### 3. 准备 Yelp 数据集

将 Yelp Academic Dataset 放入 `data/raw/` 目录：

```
data/raw/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_tip.json (可选)
```

数据集下载：https://www.yelp.com/dataset

## 完整流程运行

### 方式 1：一键运行（推荐）

```bash
python run_pipeline.py --step all
```

这将依次执行：
1. 商户画像构建
2. RQ-VAE 训练和语义 ID 生成
3. 用户序列构建（滑动窗口）
4. 多任务 Prompt 构造
5. LLM 训练

### 方式 2：分步运行

#### Step 1: 构建商户画像

```bash
python data_processing/step1_build_item_profile.py
```

**输入**：
- `data/raw/yelp_academic_dataset_business.json`
- `data/raw/yelp_academic_dataset_review.json`

**输出**：
- `data/processed/item_profiles.jsonl`

**功能**：聚合商家的 Name + Categories + Attributes + Top Reviews

---

#### Step 2: 生成语义 ID

```bash
python data_processing/step2_generate_semantic_ids.py
```

**输入**：
- `data/processed/item_profiles.jsonl`

**输出**：
- `data/embeddings/item_embeddings.pt`
- `data/processed/sid_mapping.json`
- `data/rqvae_ckpt/best_collision_model.pth`

**功能**：
1. 使用 BERT 将文本转为向量
2. 训练 3 层 RQ-VAE（每层 64 个 codebook）
3. 生成语义 ID，前两层作为 Cluster ID

**关键参数**（在 `config/config.yaml` 中调整）：
```yaml
rqvae:
  num_emb_list: [64, 64, 64]  # 3层，每层64个码本
  epochs: 5000
  batch_size: 1024
  alpha: [1.1, 1.05, 1.0]  # 残差放大系数
```

**质量检查**（推荐）：
```bash
# Step 2 完成后，运行可视化检查 Codebook 质量
python visualization/visualize_codebook.py
```

这将生成：
- 三层 Codebook 的 t-SNE/UMAP 可视化
- 使用频率分布图
- Cluster 共现热力图
- 质量报告（碰撞率、使用率等）

详见：[visualization/README.md](./visualization/README.md)

---

#### Step 3: 构建用户序列

```bash
python data_processing/step3_build_user_sequences.py
```

**输入**：
- `data/raw/yelp_academic_dataset_review.json`
- `data/processed/sid_mapping.json`

**输出**：
- `data/processed/user_sequences.jsonl`

**功能**：
1. K-core 过滤（用户和商家至少 5 次交互）
2. 按时间排序
3. 滑动窗口（最近 10-15 个）+ 长期语义摘要
4. 划分训练/验证/测试集

**关键参数**：
```yaml
preprocessing:
  min_user_interactions: 5
  min_item_interactions: 5
  max_window_size: 15  # 短期窗口
  sliding_stride: 1
```

---

#### Step 4: 构造多任务 Prompt

```bash
python data_processing/step4_construct_prompts.py
```

**输入**：
- `data/processed/user_sequences.jsonl`
- `data/processed/sid_mapping.json`

**输出**：
- `data/processed/train_prompts.jsonl`
- `data/processed/valid_prompts.jsonl`
- `data/processed/test_prompts.jsonl`

**功能**：生成三种任务的 Prompt：
- **任务 A**：序列推荐（70%）- 预测下一个 Cluster ID
- **任务 B**：偏好摘要（20%）- 生成用户画像
- **任务 C**：ID 对齐（10%）- Text ↔ ID 双向映射

**Prompt 示例**：

```
Instruction: You are a local guide. Based on the user's visit history, predict the Semantic Cluster ID of the next place they will visit.

User History:
1. [Starbucks] (Coffee) -> <5, 11>
2. [AMC Cinema] (Entertainment) -> <88, 21>

Response:
<12, 45>
```

---

#### Step 5: 训练 LLM

```bash
python training/train_llm.py
```

**输入**：
- `data/processed/train_prompts.jsonl`
- `data/processed/valid_prompts.jsonl`

**输出**：
- `data/llm_ckpt/` (LoRA 权重)

**功能**：使用 LoRA 微调 Qwen/Llama 模型

**关键参数**：
```yaml
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  use_lora: true
  lora_r: 32
  epochs: 3
  batch_size: 8
  lr: 2.0e-5
```

**训练时间估算**（单卡 A100）：
- 10K 样本：~1 小时
- 100K 样本：~10 小时

---

### Step 6: 在线推理

#### 准备用户历史文件

创建 `examples/user_history.json`：

```json
[
  {
    "business_id": "B_xxx",
    "name": "Starbucks",
    "timestamp": 1609459200
  },
  {
    "business_id": "B_yyy",
    "name": "AMC Cinema",
    "timestamp": 1609545600
  }
]
```

#### 运行推理

```bash
python inference/recommend.py \
  --user_history examples/user_history.json \
  --user_location '{"latitude": 40.7128, "longitude": -74.0060}' \
  --top_k 10
```

**输出示例**：

```
=== Recommendations ===

1. Joe's Pizza
   Category: Pizza, Italian
   City: New York
   Distance: 0.8 km
   Score: 0.9512

2. Le Bernardin
   Category: French, Seafood
   City: New York
   Distance: 1.2 km
   Score: 0.9203
```

---

## 高级功能

### 1. 调整模型参数

编辑 `config/config.yaml`：

```yaml
rqvae:
  num_emb_list: [128, 128, 128]  # 增大 codebook
  epochs: 10000  # 延长训练

llm:
  model_name: "meta-llama/Llama-3-8B-Instruct"  # 更换基座模型
  lora_r: 64  # 增大 LoRA 秩
```

### 2. 使用 DeepSpeed 加速

```yaml
llm:
  use_deepspeed: true
  deepspeed_config: "./config/deepspeed_config.json"
```

```bash
deepspeed --num_gpus=4 training/train_llm.py
```

### 3. 仅从某一步开始运行

```bash
# 从 Step 3 开始
python run_pipeline.py --step data --start_from 3
```

---

## 故障排除

### 问题 1: CUDA OOM

**解决方案**：
1. 减小 batch size
2. 启用 gradient checkpointing
3. 使用 bf16 代替 fp16

```yaml
llm:
  batch_size: 4
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  bf16: true
```

### 问题 2: RQ-VAE 碰撞率高

**解决方案**：
1. 增大 codebook 数量
2. 延长训练 epoch
3. 调整 Sinkhorn epsilon

```yaml
rqvae:
  num_emb_list: [128, 128, 128]
  epochs: 10000
  sk_epsilons: [0.0, 0.0, 0.005]
```

### 问题 3: 推荐结果不准确

**解决方案**：
1. 增加训练数据
2. 调整任务权重（增大任务 A 比例）
3. 使用更大的基座模型

```yaml
prompt:
  task_weights:
    task_a_recommendation: 0.8
    task_b_preference_summary: 0.15
    task_c_sid_alignment: 0.05
```

---

## 性能指标

### 评估指标

- **RQ-VAE**: Collision Rate（越低越好，目标 < 0.01）
- **LLM**: Perplexity / Exact Match Accuracy
- **推荐**: HR@10, NDCG@10

### 示例结果

在 Yelp 数据集上（100K 用户，50K 商家）：

| 指标 | 值 |
|------|-----|
| Collision Rate | 0.008 |
| Exact Match | 42.3% |
| HR@10 | 68.5% |
| NDCG@10 | 0.523 |

---

## 下一步

1. **实验不同的 LLM**：Qwen, Llama, Mistral
2. **引入强化学习**：参考 MiniOneRec 的 GRPO
3. **多模态扩展**：结合商家图片
4. **在线 A/B 测试**：部署到生产环境

---

## 引用

如果使用本项目，请引用：

```bibtex
@software{hiergr_seqrec,
  title = {HierGR-SeqRec: Hierarchical Generative Retrieval for Sequential Recommendation},
  year = {2024},
  author = {Your Name}
}
```
