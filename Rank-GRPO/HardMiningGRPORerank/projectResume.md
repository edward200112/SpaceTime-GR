项目背景：生成式推荐 + 候选集约束

我在做生成式推荐（next POI / next place prediction），数据来自 美国 Google Maps Reviews 数据集。

meta-*.json.gz：地点信息（name/address/gmap_id/category/...）

review-*.json.gz：用户评论行为（user_id/time/rating/text/gmap_id/...）

任务：给定用户最近访问的地点序列（按时间从旧到新），预测下一次最可能去的一个地点。
由于全库 label 空间极大（item 数约 99 万级），我采用 “候选集 rerank” 形式：每条样本提供 K=50 候选地点，并保证 Ground Truth 在候选里。模型输出必须从候选中选 1 个并原样输出 Name (Category)。

训练流程概览
1) SFT 两阶段（已完成）

stage1：主要学习输出格式与指令遵循（只输出一个 Name (Cat)，不解释）

stage2：加入困难负样本（hard negatives），进一步强化候选内选择

2) 教师模型：过拟合 SASRec（用于候选内软引导）

我训练了一个过拟合的 SASRec当教师模型，在限定数据范围内预测候选集合的偏好分数，用于：

产生/筛选困难负样本（SFT 阶段用）

GRPO 阶段做 reward shaping（teacher shaping）

SASRec 规模：n_items ≈ 992,862，max_len=50，embed_dim=128，num_blocks=2，num_heads=2。

GRPO 阶段：在候选内做生成式选择（本质是 rerank）
核心思路

让 LLM 生成一个地点字符串（Name (Category)）

reward 负责：

解析输出是否符合格式

是否在候选集合中（in_candidates）

是否命中 target（exact 或 fold match）

结合 SASRec 在候选内的偏好分数做软引导（teacher shaping）

对多余文本、未知输出、前缀污染、括号不完整、未原样 copy、同 prompt 内重复输出做惩罚

评价指标（offline）：HR@1 / HR@10（候选集 rerank 口径）

GRPO 使用的数据格式（训练 / 验证 JSONL）

每条样本关键字段：

prompt：包含历史序列 + 规则 + 候选列表（K=50）

history_item_ids：用户历史 item id 序列（长度 <= 50）

target_item_id：真实下一个 item id

target_namecat：真实下一个地点的 Name (Cat)（保证出现在候选里）

candidate_namecats：候选 Name (Cat) 列表（长度 K）

candidate_item_ids：候选 item id 列表（长度 K，与 namecats 对齐）

候选列表在 prompt 末尾，要求模型：只能从候选里选 1 个，并原样只输出地点名(类别)，不要解释。

GRPO 训练代码结构（关键点）
训练入口：加载 base model + LoRA adapter（来自 SFT stage2），再用 TRL 的 GRPOTrainer 训练

base model：/workspace/Qwen2_5-1.5B-Instruct

adapter：SFT stage2 checkpoint（LoRA）

reward：在 HardMiningGRPO/reward_sasrec.py 实现

reward 逻辑：

用正则抽取第一行中的 Name (Cat)

candidate 内 membership：exact 优先、casefold 容错

match：exact 命中给 match_reward_exact，fold 命中给 match_reward_fold

teacher shaping：对候选 item_ids 用 SASRec 打分，按 teacher_mode（zscore/logprob/prob/rank）计算 reward，再乘 alpha

penalties：extra_text / unknown / prefix / incomplete / copy_penalty / duplicate_penalty 等

支持 debug：定期打印统计 + top/bottom examples，并可 dump 到 jsonl

当前遇到与处理过的问题（简述）

early steps in_candidates 比较低是预期现象（训练会拉高）

在线 eval（每 N step）曾因 logits 过大导致 OOM：需要 micro-batch / use_cache=False / 动态 padding 或改为训练后离线 eval

现在更倾向于：训练结束后用独立 eval 脚本算 HR@1/10

关键文件路径（务必在新对话贴上）
GRPO 训练

HardMiningGRPO/train_grpo.py （GRPO 训练入口）

HardMiningGRPO/reward_sasrec.py（reward 函数 + ResolverConfig + SasrecScorer）

GRPO 数据

训练集：./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl

验证集：./HardMiningGRPO/grpo_data_v2/grpo_val.cand.fixed_precise_v2.jsonl

SFT adapter（作为 GRPO 初始策略）

./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500

SASRec 教师模型

pkl：/workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl

ckpt：/workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt

GRPO 输出目录（示例）

./HardMiningGRPO/ckpt_grpo_candidates_best_v1
或当前 rerank 实验：./HardMiningGRPO/ckpt_grpo_candidates_rerank_v3

我现在将NTP问题转换为了分类问题，也就是说，我在SFT阶段对齐了语义并使用SASRec加入困难负样本，我现在的GRPO阶段训练了一个rerank模型

我现在想将代码转变回NTP任务，仍然继续之前的开放式回答的，让模型的定位为推荐系统中的recall或者rank。我应该怎么做
我当前的代码如下：
