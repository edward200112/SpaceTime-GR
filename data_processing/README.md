项目采用 4 步数据处理流水线，将原始 Yelp 数据转换为适合训练的格式：

Step 1: 构建商家画像 (step1_build_item_profile.py)
功能：将原始 Yelp 商家和评论数据聚合成富文本描述

处理流程：

加载数据：读取 yelp_academic_dataset_business.json 和 yelp_academic_dataset_review.json
文本清洗：去除换行符、多余空格，处理乱码
评论筛选：按 useful 字段排序，保留前 3 条最有用的评论
属性自然语言化：将结构化属性转换为自然语言句子
富文本构建：按 Category -> Name -> Location -> Attributes -> Reviews 的顺序拼接
输出：item_profiles.jsonl，每行包含商家的富文本描述和地理坐标

Step 2: 生成语义ID (step2_generate_semantic_ids.py)
功能：使用 RQ-VAE 将商家编码为层级语义 ID

处理流程：

嵌入生成：使用 BERT (all-mpnet-base-v2) 对富文本编码得到 768 维向量
地理融合：将经纬度归一化后拼接，形成 770 维输入 (768d + 2d)
RQ-VAE 训练：4 层残差量化，每层 256 个 codebook
冲突解决：为相同的前 3 层 ID 添加唯一后缀，确保 0% 冲突率
映射保存：生成 business_id -> semantic_id 的映射关系
输出：sid_mapping.json，包含每个商家的 4 层语义 ID <c0, c1, c2, suffix>

Step 3: 构建用户序列 (step3_build_user_sequences.py)
功能：从用户评论历史构建时序交互序列

处理流程：

K-core 过滤：移除交互次数少于 5 次的用户和商家
时间排序：按时间戳对用户交互进行排序
滑动窗口：使用滑动窗口生成训练样本，窗口大小最大 15
长期摘要：为长历史用户生成偏好摘要
数据划分：按用户维度划分训练/验证/测试集
输出：train.jsonl, valid.jsonl, test.jsonl

Step 4: 构造训练提示 (step4_construct_prompts.py)
功能：将用户序列转换为多任务训练格式

处理流程：

Task A (推荐任务)：基于历史预测下一个访问的语义 ID
Task B (偏好总结)：总结用户偏好（辅助任务）
Task C (ID 对齐)：学习商家名称到语义 ID 的映射（辅助任务）
格式统一：使用 instruction-following 格式
输出：train_prompts.jsonl, valid_prompts.jsonl, test_prompts.jsonl

SFT 数据处理
数据增强策略 (train_sft_final.py)
核心特性：

动态历史增强：训练时随机丢弃 1-2 个历史记录，防止过拟合
Chat 模板：使用 apply_chat_template 转换为对话格式
标签掩码：只对输出部分计算损失，输入部分标签设为 -100
数据平衡：使用平衡采样的训练集
数据平衡处理 (balance_dataset.py)
策略：

按类别分组：根据语义 ID 的第 2 层（类别层）分组
目标对齐：将所有类别的样本数对齐到 75% 分位数
重采样：热门类别下采样，冷门类别上采样（最大 10 倍）
权重计算：为强化学习阶段计算类别权重
强化学习数据处理
GRPO 训练数据 (train_grpo_v3.py)
数据准备：

任务过滤：只使用 Task A（推荐任务）的数据
格式统一：添加统一的输出格式提示
元数据保留：保留目标商家的地理坐标用于奖励计算
奖励函数设计 (grpo_rewards_v3.py)
三维奖励系统：

格式奖励：

输出格式正确：+0.1
格式错误：-1.0
语义奖励（层级递进）：

Layer 0 匹配：+0.2
Layer 1 匹配：+0.3
Layer 2 匹配：+1.0
完全匹配：+2.0
地理奖励：

≤2km：+1.5
≤5km：+1.0
≤20km：+0.5
50km：-0.1

渐进式惩罚：

完全无效 ID：-1.0
前缀部分有效：-0.2 到 -0.6（根据有效层数）
关键技术特点
地理感知：在 RQ-VAE 输入中融合地理坐标，在奖励函数中加入距离惩罚
冲突解决：通过添加唯一后缀将 ID 冲突率从 98.1% 降至 0%
多任务学习：结合推荐、偏好总结、ID 对齐三个任务
渐进式奖励：对部分正确的预测给予适当奖励，避免梯度消失
数据平衡：通过重采样和加权解决长尾分布问题
这套数据处理流程实现了从原始 Yelp 数据到可训练格式的完整转换，支持 SFT 和 GRPO 两阶段训练，是一个相当完整和优化的推荐系统数据处理方案。