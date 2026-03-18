# SpaceTime-GR

一个面向下一地点（Next-POI）预测的时空生成式推荐项目，结合层级语义 ID、生成式建模、检索式排序和强化学习优化，基于 Yelp 数据构建完整实验链路。

## 项目概述

这个仓库关注的不是“用户下一次会点击什么”，而是更接近真实线下行为的问题：**用户下一步会去哪里**。

为了解决这个问题，项目将 POI 推荐重构为一个生成式任务：先通过 RQ-VAE 将商家压缩为**层级语义 ID**，再使用基于 Qwen 的语言模型根据用户历史行为生成下一个语义 ID，最后将预测出的语义簇展开为具体商家候选集合。

除了主线的生成式推荐路径之外，仓库还保留了并行的 **PinRec / Ultimate** 双塔检索路线，以及独立的 `Rank-GRPO/` 子目录，用于更复杂的排序、教师-学生强化学习和 hard-mining 实验。因此，这不是一个单脚本 demo，而是一个覆盖**数据处理、表征学习、监督训练、强化学习微调、推理、评估、调试与可视化**的端到端研究工程工作区。

## 为什么做这个项目

很多推荐系统项目往往只覆盖其中一个层面：

- 基础序列建模；
- 基础召回 / 检索；
- 或者单一训练循环。

这个项目尝试处理一个更难、更接近真实场景的问题：推荐质量不仅取决于**语义相关性**，还取决于**地理邻近性**和**时间行为模式**。

这个仓库重点实现以下内容：

1. 从原始评论数据出发，构建结构化推荐输入；
2. 为大规模 POI 词表学习紧凑的语义 ID 表示；
3. 将 LLM 微调为生成式推荐器；
4. 通过自定义奖励函数引入强化学习优化；
5. 保留并维护双塔检索基线，便于对照实验；
6. 从推荐指标、层级匹配、空间距离等多个维度进行评估。

## 项目的目的

- **端到端机器学习系统设计**：从原始 Yelp 数据到表征学习、Prompt 构造、训练、推理、评估的一整套链路。
- **LLM 工程能力**：LoRA 微调、语义 ID token 扩展、checkpoint 合并与推理接入。
- **推荐系统建模能力**：基于 RQ-VAE 的语义压缩建模，并将地理特征融入 POI 表示。
- **多范式建模能力**：同一仓库内同时维护生成式推荐、双塔排序与 RL 优化路径。
- **评估与诊断意识**：离线排序指标、层级语义匹配分析、空间距离评估、误差分析和可视化。
- **研究工程习惯**：保留多版本实验脚本、ablation 风格分支、调试工具和独立说明文档。

## 核心思路

```text
Yelp 商家数据 + 评论数据
            ↓
构建 POI 画像
            ↓
文本嵌入 + 经纬度联合编码
            ↓
RQ-VAE 学习层级语义 ID
            ↓
构建用户行为序列
            ↓
生成多任务 Prompt / 排序训练数据
            ↓
训练推荐模型
  ├─ HierGR：基于 LLM 的生成式推荐
  └─ PinRec / Ultimate：双塔排序路线
            ↓
可选的 GRPO / hard-mining 强化优化
            ↓
推理、评估、诊断、可视化
```

## 主要模块

### 1. 基于 RQ-VAE 的层级语义 ID

项目不会直接依赖原始 POI ID，而是先为每个商家学习一个结构化语义编码。RQ-VAE 流程会联合使用文本语义嵌入和经纬度特征，因此最终得到的表示同时包含**语义信息**和**空间信息**。

### 2. 生成式推荐主线（HierGR）

项目使用基于 Qwen 的因果语言模型，根据用户历史行为预测下一个语义 ID。推理流程支持：

- tokenizer 扩展语义 ID token；
- 加载 LoRA checkpoint；
- 将预测簇映射回具体商家候选。

### 3. 检索 / 排序路线（PinRec / Ultimate）

除了生成式路径，仓库中还包含一条完整的双塔推荐路线，例如：

- `models/pinrec_llm.py`
- `models/pinrec_ultimate.py`
- `models/pinrec_ultimate_v2.py`

这使得项目既可以做生成式实验，也可以做排序式对照。

### 4. 强化学习与排序实验

仓库中包含多种 GRPO 训练脚本、约束生成逻辑、稠密奖励实现，以及更大的 `Rank-GRPO/` 实验子树，用于教师-学生训练、hard-mining 和 reranking 场景。

## 亮点特性

- **4 步数据处理流水线**：将原始 Yelp 商家 / 评论 JSON 转换为可训练推荐数据。
- **位置感知的语义 ID 生成**：将文本嵌入与归一化经纬度联合用于语义量化。
- **生成式下一地点预测**：通过 LLM 微调实现 Next-POI 生成式推荐。
- **约束式解码机制**：通过 Trie / constrained logits processor 限制无效输出。
- **并行排序基线**：通过 PinRec 与 Ultimate 双塔模型保留检索式对照实验。
- **完整评估工具链**：覆盖 HR@K / NDCG@K / MRR、距离分析、聚类纯度检查与 t-SNE 验证。

## 技术栈

### 建模与训练
- PyTorch
- Transformers
- PEFT / LoRA
- Sentence Transformers
- RQ-VAE 风格量化建模

### 数据处理
- Datasets
- NumPy
- Pandas
- scikit-learn
- PyYAML
- tqdm

### 检索与向量工具
- FAISS（CPU）

### 可视化与分析
- matplotlib
- seaborn
- UMAP

## 仓库结构

```text
SpaceTime-GR/
├── config/
│   └── config.yaml                  # 主配置文件
├── data_processing/
│   ├── step1_build_item_profile.py  # 从 Yelp 数据构建 POI 画像
│   ├── step2_generate_semantic_ids.py
│   ├── step3_build_user_sequences.py
│   ├── step4_construct_prompts.py
│   └── README.md                    # 数据流程说明
├── RQ-VAE/
│   ├── models/
│   └── trainer.py                   # RQ-VAE 训练实现
├── models/
│   ├── pinrec_llm.py
│   ├── pinrec_ultimate.py
│   └── pinrec_ultimate_v2.py        # 双塔推荐模型变体
├── training/
│   ├── train_sft_final.py
│   ├── train_llm.py
│   ├── train_grpo_v3.py
│   ├── train_grpo_v5.py
│   ├── train_pinrec_sft_final.py
│   ├── train_pinrec_grpo_final.py
│   ├── train_ultimate_v4_stable.py
│   ├── constrained_logits_processor.py
│   ├── merge_model.py
│   └── GRPO_TRAINING_GUIDE.md
├── inference/
│   ├── recommend.py                 # 推荐推理入口
│   ├── new_evaluate.py
│   ├── evaluate_final_v9.py
│   ├── evaluate_metrics.py
│   ├── check_sft_quality.py
│   ├── check_cluster_purity.py
│   ├── analyze_errors.py
│   ├── demo_inference.py
│   └── validate_grpo_with_tsne.py
├── evaluation/
│   ├── evaluate_model.py            # 独立评估工具链
│   ├── quick_test.py
│   ├── compare_results.py
│   ├── EVALUATION_GUIDE.md
│   └── README.md
├── examples/
│   ├── user_history_example.json
│   └── user_location_example.json
├── Rank-GRPO/                       # 进阶排序 / reranking 实验
├── yelp18Eval/                      # 额外评估相关资源
├── requirements.txt
└── run_pipeline.py                  # 粗粒度流程入口
```

## 推荐阅读顺序

如果你是从“工程深度评估”的角度阅读这个仓库，建议按下面顺序看：

1. `config/config.yaml`：先理解路径、训练配置和整体假设；
2. `data_processing/README.md` + `step1~4`：理解原始 Yelp 数据如何变成训练输入；
3. `training/train_sft_final.py` 和 `training/train_grpo_v5.py`：查看生成式训练主线；
4. `models/pinrec_ultimate_v2.py` 和 `training/train_pinrec_sft_final.py`：查看排序 / 检索路线；
5. `inference/recommend.py` 和 `evaluation/evaluate_model.py`：查看推理与评估如何落地；
6. `Rank-GRPO/`：查看更复杂的排序与教师-学生实验。

## 环境要求

### 软件环境
- Python 3.8+
- 建议使用支持 CUDA 的环境进行训练 / 推理
- 需要本地可访问的 Qwen 基座模型 checkpoint（默认配置使用本地路径）

### 数据
- Yelp business JSON
- Yelp review JSON

### 模型与中间产物
如果要完整复现实验，一般还需要：

- 放在 `data/raw/` 下的原始 Yelp 数据；
- 处理后的中间文件，例如 `sid_mapping.json`；
- 本地基础 LLM checkpoint；
- SFT / GRPO / PinRec 等训练完成的 checkpoint（若要直接推理或评估）。

## 安装方式

```bash
git clone https://github.com/edward200112/SpaceTime-GR.git
cd SpaceTime-GR
pip install -r requirements.txt
```

## 配置说明

主配置文件位于 `config/config.yaml`。

当前配置的几个重要假设包括：

- 原始 Yelp 数据默认放在类似 `/workspace/data/raw` 的路径；
- 处理结果和 checkpoint 也默认写入 `/workspace/data/...`；
- 基础模型路径默认是 `/workspace/Qwen2_5-1.5B-Instruct`；
- LLM 训练默认采用 LoRA 微调；
- RQ-VAE 模块默认使用文本 + 地理特征融合表示；
- GRPO 模块配置了语义、地理、格式、命中等稠密奖励项。

在本地运行前，至少建议先修改以下字段：

```yaml
data:
  raw_dir: "/your/local/path/data/raw"
  processed_dir: "/your/local/path/data/processed"
  embeddings_dir: "/your/local/path/data/embeddings"
  rqvae_ckpt_dir: "/your/local/path/data/rqvae_ckpt"
  llm_ckpt_dir: "/your/local/path/data/llm_ckpt"

llm:
  model_name: "/your/local/path/Qwen2_5-1.5B-Instruct"

hardware:
  device: "cuda"
```

## 数据准备

先将 Yelp 原始文件放到 `data/raw/`：

```text
data/raw/
├── yelp_academic_dataset_business.json
└── yelp_academic_dataset_review.json
```

然后按顺序执行核心数据流水线：

```bash
python data_processing/step1_build_item_profile.py
python data_processing/step2_generate_semantic_ids.py
python data_processing/step3_build_user_sequences.py
python data_processing/step4_construct_prompts.py
```

`data_processing/` 下还包含一些可选的数据平衡脚本，例如：

```bash
python data_processing/balance_dataset.py
python data_processing/balance_sequences_for_pinrec.py
python data_processing/balance_ultimate_data.py
```

## 快速开始

### 方案 A：先理解流程

如果你暂时不想复现实验，只想先理解代码组织和流程，可以先运行：

```bash
python run_pipeline.py --step data
```

### 方案 B：基于已准备好的产物做推荐推理

在配置文件、处理后数据和 checkpoint 已准备好的前提下，可以直接运行：

```bash
python inference/recommend.py \
  --config ./config/config.yaml \
  --user_history ./examples/user_history_example.json \
  --user_location ./examples/user_location_example.json \
  --top_k 10
```

## 训练路径

这个仓库支持多条实验路线，而不是只有一个“官方唯一训练命令”。

### HierGR / 生成式路线

推荐先看：

```bash
python training/train_sft_final.py
python training/train_grpo_v5.py
```

仓库中还保留了多种实验版本：

```bash
python training/train_llm.py
python training/train_grpo.py
python training/train_grpo_v2.py
python training/train_grpo_v3.py
python training/train_grpo_v4.py
python training/train_grpo_v4_1.py
python training/train_grpo_v4_2_optimized.py
python training/train_grpo_v4_3_logit_masking.py
python training/train_grpo_v4_4_cot.py
```

### PinRec / Ultimate 排序路线

```bash
python training/train_pinrec_sft_final.py
python training/train_pinrec_grpo_final.py
python training/train_pinrec_v7_final.py
python training/train_ultimate_v4_stable.py
```

### 更大的 7B 实验

`training/` 目录中还包含针对 7B 模型的脚本，例如：

```bash
python training/7B_train_sft_optimized.py
python training/7B_train_grpo.py
python training/7B_train_GRPO_optimized_resume.py
```

## 评估方式

### 推荐效果评估

```bash
python evaluation/evaluate_model.py \
  --config ./config/config.yaml \
  --test_data ./data/processed/test_prompts.jsonl \
  --batch_size 8 \
  --num_beams 5 \
  --top_k 5,10,20 \
  --use_constrained_generation \
  --output ./evaluation/results.json
```

### 快速交互检查

```bash
python evaluation/quick_test.py --config ./config/config.yaml
```

### 推理侧评估与分析工具

```bash
python inference/new_evaluate.py
python inference/evaluate_final_v9.py
python inference/evaluate_metrics.py
python inference/check_sft_quality.py
python inference/check_cluster_purity.py
python inference/analyze_errors.py
python inference/validate_grpo_with_tsne.py
```

## 实验效果对比（离线评估）

> 说明：下表汇总了当前 README 中最值得招聘方关注的一组阶段性结果。除特别说明外，指标均来自离线评估；其中部分 GRPO 结果明确标注了 `N=500`。这些结果的重点不在于“是否已经超过工业级排序模型”，而在于：**项目已经证明 LLM 可以稳定学习地理约束，并开始学习层级语义 ID 的类别结构**。

| 实验版本 | 关键改动 | Mean Distance ↓ | Hit@1 ↑ | Hit@5 ↑ | Hit@10 ↑ | Layer 0（City）↑ | Layer 1（District）↑ | Layer 2（Category）↑ | Layer 3 / Exact ↑ | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GRPO V1（高 Geo Reward） | 强化地理奖励，优先学会“先到正确城市” | 8.3879 km | 0.20% | 1.80% | 2.40% | 95.80% | 64.60% | 2.80% | 0.20% | 地理命中已经很强，但语义层明显不足，模型更像“附近地点生成器” |
| GRPO V2（高 Semantic Reward） | 下调地理权重，尝试给语义留空间 | 5.0849 km | 0.00% | 2.00% | 3.00% | 96.00% | 73.00% | 7.40% | 0.00% | 语义层有提升，但模型仍会利用 Geo reward 走捷径 |
| GRPO V3（Gating + 强约束） | 对 Layer 2 预测错误施加惩罚，并尝试更强解码约束 | — | 0.80% | 1.80% | 2.00% | 93.80% | 34.40% | 2.60% | 0.80% | 证明“约束过强”会破坏层级学习，不能只靠硬规则修复生成问题 |
| SFT V2 | 优化 SFT 掩码与格式学习，提升格式遵从与基础语义对齐 | 4.5398 km | 1.00% | 2.00% | 3.40% | 94.40% | 78.40% | 9.80% | 1.00% | 说明仅靠更干净的监督训练，就能显著改善语义层和格式稳定性 |
| GRPO V4.1（1500 steps） | 面包屑奖励（Breadcrumbs）+ 更稳定的奖励塑形 | 4.4896 km | 1.20% | 2.40% | 3.60% | 95.00% | 78.80% | 10.80% | 1.20% | 当前 1.5B 路线的阶段性最佳平衡点之一 |
| GRPO V4.1（4800 steps） | 延长训练，继续优化层级语义对齐 | **4.4136 km** | **1.20%** | 2.20% | 3.40% | 94.80% | **79.60%** | **12.00%** | **1.20%** | 当前 README 中最值得展示的主结果：空间定位稳定，Category 层达到 12% |
| Qwen2.5-7B GRPO | 扩大模型容量，测试更强语言模型的层级语义学习能力 | 5.3531 km | 0.80% | 1.00% | 1.00% | 93.80% | 75.40% | **12.40%** | 0.80% | 更大模型带来了更高的 Layer 2，但整体召回并未同步提升 |
| Qwen2.5-7B GRPO（Beam=20） | 更强搜索策略，视作生成式召回实验 | 5.5003 km | 0.80% | 2.20% | 2.80% | 93.80% | 73.80% | 8.20% | 0.80% | 更适合解释为“生成式召回器”，而不是最终精排器 |

### 如何解读这些结果

- **最稳定的优势是地理层级学习**：多数版本都能把 Layer 0 保持在 94%~96%，说明模型已经能稳定学习城市/区域约束。
- **真正困难的是 Layer 2（Category）**：从 V1 的 2.8% 到 V4.1 的 12.0%，虽然绝对值仍不高，但已经验证了“奖励设计 + SFT 清洗 + 表征改进”确实能推动语义层进步。
- **项目最适合的定位不是直接替代 DLRM，而是 Generative Recall**：模型擅长把候选空间缩小到“正确城市 / 正确区域 / 接近类别”的范围，再交由传统排序模型做精排。

## 待完成 / 关键对照实验

| 实验项 | 当前状态 | 目标指标 / 预期价值 |
|---|---|---|
| 冷启动数据集：HierGR vs SASRec + HNSW | 待完成 | 严格对比 Recall@10，拿到“冷启动场景提升 X%”的定量结论 |
| Trie-Constrained Decoding（城市隔离 + 前缀树） | 方案明确，仍在调参 | 强制只生成 `sid_mapping.json` 中真实存在的 ID，降低幻觉与非法输出 |
| Dense Embedding + FAISS 检索 A/B | 在研 | 对比“离散层级 ID 生成”与“连续向量检索”两条路线谁更能缓解语义坍塌 |
| 生成式召回 + DLRM 精排 | 规划中 | 将 LLM 产出的 Top-K 语义候选交给传统排序模型重排，验证混合链路收益 |

## 可解释性分析摘要（RQ-VAE / GRPO）

这部分是这个项目非常适合放在招聘 README 里的亮点，因为它不仅展示了结果，还展示了**为什么结果会这样**。

### 1. RQ-VAE 没有发生 Codebook Collapse

基于导出的 Codebook 做 t-SNE 可视化后，可以看到：

- **Layer 0 / Layer 1** 的点更像“地理锚点（Geo-Anchors）”，不同城市在向量空间中被明显拉开；
- **Layer 2** 则更像“跨城市共享的语义云（Semantic Mixing）”，例如不同城市的同类餐饮 POI 会落在相近区域；
- 大量未使用或低频使用的 code 形成空白区域，说明 codebook 具备容量冗余和稀疏性，而不是被压缩成单一点或均匀噪声。

这意味着当前的 Geo-Text Fusion 策略至少完成了两件事：

1. 避免了早期出现的高碰撞问题；
2. 初步实现了 **Geo-Semantic Disentanglement（地理-语义解耦）**。

### 2. 聚类纯度证明 Layer 2 确实在学习类别

在聚类纯度分析中：

- 平均聚类纯度约为 **72.65%**；
- 例如 `Cluster (215, 55, 151)` 中 **84.3%** 是 `Food`；
- `Cluster (44, 122, 154)` 中 **100%** 是 `Food/Burgers`。

这说明 `<c0, c1, c2>` 并不只是一个压缩后的哈希 ID，而是在相当程度上保留了类别语义。

### 3. GRPO 学到的是“先定位，再找类别”

结合可视化和奖励分析，可以把当前 GRPO 行为概括为两步：

1. **Geo-Routing**：先用 Layer 0 / 1 把搜索空间锁定到正确城市与区域；
2. **Semantic Matching**：再尝试在局部空间中找到更符合用户意图的类别语义。

这也解释了为什么当前系统虽然 Exact Match 仍然不高，但已经具备了作为 **生成式召回模块** 的实用价值。

## 关键设计决策

### 1. 用层级语义 ID 替代原始 POI ID

相比巨大的平铺 item vocabulary，语义 ID 让输出空间更结构化、更可压缩，也让模型可以做“部分正确”的层级判断：即使没有命中精确 POI，也可以在城市层、区域层或类别层上分析模型是否接近正确答案。

### 2. 提前融合文本与地理信息

当前配置和数据处理说明都明确将文本嵌入与经纬度特征在语义量化之前进行融合。对 Next-POI 任务来说，这是一个很关键的工程选择，因为地理位置不是附加元数据，而是问题本身的一部分。

### 3. 同时保留生成式与检索式两条路线

项目没有只押注一种推荐范式，而是并行保留了生成式 LLM 路线和双塔排序路线。这让仓库更适合做实验、对照和调试，也更能体现建模判断力。

### 4. 使用约束解码而不是完全放任自由生成

仓库里有 Trie / constrained logits 相关逻辑，确保模型只生成合法的 cluster ID。这是一个非常实用的工程设计，能显著减少推理和评估阶段的无效输出。

### 5. 用奖励塑形替代过于稀疏的 RL 信号

GRPO 实验并不是只依赖单一的二值奖励，而是组合了格式正确性、语义对齐、地理接近度和命中行为等稠密奖励项。这比简单 sparse reward 更适合推荐场景下的强化学习优化。

## 如果在面试中讨论这个项目，我会重点讲什么

如果把这个项目作为面试作品讨论，我会重点展开以下问题：

- 语义 ID 空间是如何构造的，为什么它对推荐有效；
- 为什么仓库要同时维护 HierGR 与 PinRec / Ultimate 两条主线；
- 约束生成如何影响最终评估质量；
- GRPO 的奖励项是如何选择与权衡的；
- 使用大量实验脚本和绝对路径配置带来了哪些工程代价；
- 如果把它整理成更清晰的生产 / 研究双结构，我会如何重构。

## 取舍与局限

- 这个仓库明显是一个持续演进中的研究工作区，因此存在大量版本化脚本（如 `v2`、`v3`、`v4`、`v5`、`final`、`debug`、`7B`），而不是单一整洁入口。
- `run_pipeline.py` 适合帮助读者理解基础流程，但它并不能覆盖所有最新实验路径。
- 当前配置大量使用 `/workspace/...` 绝对路径，因此本地复现前需要手动调整路径。
- 完整复现依赖外部数据、本地 LLM checkpoint 以及若干处理后的中间产物，这些并未全部随仓库提供。
- `requirements.txt` 更适合作为起点，而不是所有实验场景都可直接锁定复现的最终环境文件。
- 仓库命名和模块命名体现了多个阶段的迭代（`SpaceTime-GR`、`HierGR-SeqRec`、`PinRec`、`Ultimate`、`Rank-GRPO`），后续若统一命名会更利于新读者理解。

## 下一步改进方向

如果继续完善这个项目，比较值得做的方向包括：

1. 统一仓库命名与文档表述；
2. 为其中一条主实验路线提供更干净的可复现配置；
3. 更明确地区分“研究实验区”和“稳定流程区”；
4. 提供环境文件或容器化配置；
5. 补充一个可复现的 benchmark 表格；
6. 暴露一条更清晰、面向评审者的一键 demo 路径。

## 如何快速评估这个仓库

- 想看**数据工程能力**：从 `data_processing/` 开始；
- 想看**生成式推荐能力**：从 `training/train_sft_final.py` 和 `inference/recommend.py` 开始；
- 想看**排序 / 检索能力**：从 `models/pinrec_ultimate_v2.py` 和 `training/train_pinrec_sft_final.py` 开始；
- 想看**RL / 奖励塑形设计**：从 `training/train_grpo_v5.py`、`training/grpo_rewards_optimized.py` 和 `Rank-GRPO/` 开始；
- 想看**评估成熟度**：从 `evaluation/evaluate_model.py` 和 `inference/validate_grpo_with_tsne.py` 开始。

## 仓库内相关文档

- [`data_processing/README.md`](./data_processing/README.md)
- [`training/GRPO_TRAINING_GUIDE.md`](./training/GRPO_TRAINING_GUIDE.md)
- [`evaluation/README.md`](./evaluation/README.md)
- [`evaluation/EVALUATION_GUIDE.md`](./evaluation/EVALUATION_GUIDE.md)

## License

当前顶层目录中没有看到明确的 `LICENSE` 文件。如果你希望这个项目的使用范围和授权方式更清晰，建议补充一个顶层 `LICENSE`。
