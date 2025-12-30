# 转回NTP任务
把候选从“显式约束/输出空间”降级为“训练时的 teacher 软池子（reward shaping 用）”，最终 prompt 不再出现候选；
同时 reward 需要从“候选 membership reward”改成“全库实体解析 + 命中 GT + teacher 软引导”。


方案 A（推荐，工程最稳）：输出 gmap_id（或 item_id）作为主输出
训练/评估都清晰，避免 同名同类地点冲突（Name(Cat) 在全库一定会撞）。
推理时再用 id 查 meta 输出 Name(Cat)。
格式例：
0x123abc... 或 992862（你的 item_id）


最推荐的“训练日程”（避免一下子信号稀疏崩掉）：

Phase 1：Prompt 去候选，但 reward 仍用 K=50 作为 teacher-pool（密集信号）
  prompt_with_candidates=False
  reward：exists/hit 为主，teacher shaping 用 candidate pool
  这一步能让模型学会“没候选也能猜你要去哪里”，但训练仍稳定
Phase 2：把 teacher-pool 扩大（更像 recall）
  预计算每条样本的 teacher_top_item_ids（SASRec 全库 top200）
  reward shaping 从 K=50 切到 top200
Phase 3（可选）：输出从 Name(Cat) 切换到 item_id（彻底消除歧义）
  你可以先 SFT 一轮让格式切过去，再 GRPO

Phase 1：不需要重新生成训练数据（你现有 jsonl 就能跑）
只要包含：
prompt
history_item_ids
target_item_id
candidate_item_ids（用于 Phase1 shaping；如果缺失也能跑，但会更稀疏/更难训）
上面的 train_grpo_ntp.py 已经会自动把旧 prompt 里的候选段切掉，并追加 “只输出 item_id”。
Phase 2：强烈建议生成“新增字段”，但不需要重做 prompt
你需要在每条样本里加一个字段：
teacher_top_item_ids: List[int]（建议 100~300）
这一步的本质是：让 reward shaping 看到更接近全库 recall 的负样本空间。
否则 Phase2 名义上开始了，但实际上仍在用你原来的 K=50 pool（训练目标会被限制在那个分布里，难以成为真正 recall）。
Phase 3：不强制生成新数据，但如果没有 Phase2 的大 pool，Phase3 会更难训
因为 reward 更稀疏，只有 “命中 target_item_id” 才大回报。


如果你愿意，我可以继续把 **Phase2 的“teacher_top_item_ids 生成脚本”**也给你（关键是：你 SASRec 代码里有没有 predict_all/能否拿到 query 向量做 FAISS；我会按你现有 SASRec 实现写一个可跑的版本）。你只要贴一下 TeacherModel/SASRec.py 里 forward/predict_candidates 的实现片段（几十行就够）。

# phase1
python HardMiningGRPO/train_grpo_ntp.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_val.cand.fixed_precise_v2.jsonl \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir  ./HardMiningGRPO/ckpt_grpo_ntp_phase1 \
  --phase 1 \
  --strip_candidates_from_prompt \
  --use_chat_template




# phase2 
先算每条样本的 user_repr = log2feats(history)[:,-1,:]（[B,H]）
再用 FAISS（优先） 或 torch chunked topk（保底） 在全库 item embedding 上做 topK inner product 搜索
写回 JSONL：新增字段 teacher_top_item_ids: List[int]（长度 K，如 200），并保证 target_item_id 在其中

✅ GPU torch 分块 topk（不依赖 faiss，一定能跑）
✅ 如果环境装了 faiss，会自动用 faiss（更快）
✅ 流式读写 JSONL，不吃内存
✅ 自动确保 GT 在 topK 里

python HardMiningGRPO/build_teacher_topk.py \
  --input_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --output_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.phase2.teacher200.jsonl \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --topk 200 \
  --batch_size 2048 \
  --chunk_size 600000 \
  --device cuda \
  --item_emb_on_gpu \
  --score_dtype fp16 \
  --overwrite


python HardMiningGRPO/build_teacher_topk.py \
  --input_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_val.cand.fixed_precise_v2.jsonl \
  --output_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_val.phase2.teacher200.jsonl \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --topk 200 \
  --batch_size 1024 \
  --chunk_size 600000 \
  --device cuda \
  --item_emb_on_gpu \
  --score_dtype fp16 \
  --overwrite

训练代码：
python HardMiningGRPO/train_grpo_ntp.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_val.cand.fixed_precise_v2.jsonl \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir  ./HardMiningGRPO/ckpt_grpo_ntp_phase1 \
  --phase 1 \
  --strip_candidates_from_prompt \
  --use_chat_template
  --train_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.phase2.teacher200.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_val.phase2.teacher200.jsonl \
  --phase 2 \
  --teacher_pool_k 200


验证生成的数据
python HardMiningGRPO/verify_teacher_topk.py \
  --jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.phase2.teacher200.jsonl \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --topk 200


python HardMiningGRPO/verify_teacher_topk.py \
  --jsonl ./HardMiningGRPO/grpo_data_v2/grpo_val.phase2.teacher200.jsonl \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --topk 200



Phase1 要训练到什么程度才能进 Phase2？
Phase1 的目标本质是：把“开放式生成 item_id”变成稳定的“可解析、可落库”的行为。
只要没达到这个，Phase2 加 teacher shaping 只是在浪费梯度（一直罚格式/unknown）。
进入 Phase2 的硬条件（满足再进）

在 Phase1 的 reward debug 里（或你自己统计）：
    parse_rate ≥ 0.98
    extract_first_int() 能解析出数字的比例
    prefix_ok_rate ≥ 0.98
    第一行必须是纯数字（不要 “答案：123”）
    extra_text_rate ≤ 0.0
    不要多行解释
    exists_rate ≥ 0.98
    id ∈ [1, n_items]
    unknown_rate ≤ 0.02
    解析失败或越界很少
===
{'loss': -0.0009, 'grad_norm': 0.8077430725097656, 'learning_rate': 4.821497826131331e-06, 'num_tokens': 12297029.0, 'completions/mean_length': 4.755625, 'completions/min_length': 3.16, 'completions/max_length': 5.64, 'completions/clipped_ratio': 0.0, 'completions/mean_terminated_length': 4.755625, 'completions/min_terminated_length': 3.16, 'completions/max_terminated_length': 5.64, 'rewards/reward_fn/mean': -0.033404000997543336, 'rewards/reward_fn/std': 0.002927899292553775, 'reward': -0.033404000997543336, 'reward_std': 0.0012324198288843036, 'frac_reward_zero_std': 0.805, 'entropy': 1.7773398172855377, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
{'loss': 0.004, 'grad_norm': 0.0, 'learning_rate': 4.816394848033313e-06, 'num_tokens': 12640811.0, 'completions/mean_length': 4.88375, 'completions/min_length': 3.22, 'completions/max_length': 6.04, 'completions/clipped_ratio': 0.00125, 'completions/mean_terminated_length': 4.879899187088013, 'completions/min_terminated_length': 3.22, 'completions/max_terminated_length': 5.96, 'rewards/reward_fn/mean': -0.03332650117576122, 'rewards/reward_fn/std': 0.0028051279234932737, 'reward': -0.03332650117576122, 'reward_std': 0.0014054529479471966, 'frac_reward_zero_std': 0.79, 'entropy': 1.8149483811855316, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
{'loss': -0.0015, 'grad_norm': 1.6705952882766724, 'learning_rate': 4.811291869935295e-06, 'num_tokens': 12986653.0, 'completions/mean_length': 4.19625, 'completions/min_length': 2.68, 'completions/max_length': 5.1, 'completions/clipped_ratio': 0.0, 'completions/mean_terminated_length': 4.19625, 'completions/min_terminated_length': 2.68, 'completions/max_terminated_length': 5.1, 'rewards/reward_fn/mean': -0.03342700116336346, 'rewards/reward_fn/std': 0.0030697618750855325, 'reward': -0.03342700116336346, 'reward_std': 0.0013467424863483756, 'frac_reward_zero_std': 0.73, 'entropy': 1.8077222418785095, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
{'loss': -0.0023, 'grad_norm': 1.5364161729812622, 'learning_rate': 4.806188891837277e-06, 'num_tokens': 13333251.0, 'completions/mean_length': 4.34375, 'completions/min_length': 2.82, 'completions/max_length': 5.06, 'completions/clipped_ratio': 0.0, 'completions/mean_terminated_length': 4.34375, 'completions/min_terminated_length': 2.82, 'completions/max_terminated_length': 5.06, 'rewards/reward_fn/mean': -0.0333995009958744, 'rewards/reward_fn/std': 0.0025044096063356848, 'reward': -0.0333995009958744, 'reward_std': 0.0010298513039015233, 'frac_reward_zero_std': 0.75, 'entropy': 1.797992798089981, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
{'loss': 0.0017, 'grad_norm': 0.0, 'learning_rate': 4.801085913739259e-06, 'num_tokens': 13683003.0, 'completions/mean_length': 4.14, 'completions/min_length': 2.66, 'completions/max_length': 5.0, 'completions/clipped_ratio': 0.0, 'completions/mean_terminated_length': 4.14, 'completions/min_terminated_length': 2.66, 'completions/max_terminated_length': 5.0, 'rewards/reward_fn/mean': -0.03336150124669075, 'rewards/reward_fn/std': 0.0027668682148214428, 'reward': -0.03336150124669075, 'reward_std': 0.0011412121099419893, 'frac_reward_zero_std': 0.775, 'entropy': 1.744689666032791, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
{'loss': -0.0015, 'grad_norm': 3.118298292160034, 'learning_rate': 4.795982935641241e-06, 'num_tokens': 14021282.0, 'completions/mean_length': 3.784375, 'completions/min_length': 2.32, 'completions/max_length': 5.04, 'completions/clipped_ratio': 0.000625, 'completions/mean_terminated_length': 3.781794352531433, 'completions/min_terminated_length': 2.32, 'completions/max_terminated_length': 4.98, 'rewards/reward_fn/mean': -0.034070751070976256, 'rewards/reward_fn/std': 0.0037958259927108884, 'reward': -0.034070751070976256, 'reward_std': 0.001679502110928297, 'frac_reward_zero_std': 0.72, 'entropy': 1.69618141412735, 'clip_ratio/low_mean': 0.0, 'clip_ratio/low_min': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/high_max': 0.0, 'clip_ratio/region_mean': 0.0, 'epoch': 0.04}
===

进入phase2指令：
python HardMiningGRPO/train_grpo_ntp.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningGRPO/ckpt_grpo_ntp_phase1/checkpoint-2000/ \
  --train_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.phase2.teacher200.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_val.phase2.teacher200.jsonl \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir  ./HardMiningGRPO/ckpt_grpo_ntp_phase2_recall \
  --phase 2 \
  --strip_candidates_from_prompt \
  --use_chat_template \
  --teacher_pool_k 200 \
  --alpha 0.6 \
  --rank_shaping_weight 0.8 \
  --wrong_penalty 0.2 \
  --unknown_penalty 0.6 \
  --temperature 1.0 \
  --num_generations 8 \
  --max_length 1024 \
  --max_new_tokens 8 \
  --per_device_bs 16 \
  --grad_accum 2 \
  --lr 5e-6 \
  --num_train_epochs 1 \
  --logging_steps 50 \
  --save_steps 500 \
  --debug_log_every_steps 200



这个 Phase2 reward 不会因为 target 不在 teacher_top 而惩罚正确答案；但如果你希望 Phase2 也能“更直接提升 HR@1”，长期还是建议你把 teacher_top 生成脚本里“过滤 full_set 导致 target 被剔除”的逻辑修掉（至少别把 target 过滤掉）。你现在这个 reward 能先跑起来把策略往 recall 区域拉近。


根据phase1的指标分析后

这份 build_teacher_topk.py 现在并没有做任何 full_set / history 的过滤，所以你日志里 forced_target_at_last≈1.0 的根因并不是“过滤把 target 剔除”，而是更直接的事实：
SASRec 在全库 top200 里几乎从来检不回 target（target 的真实 teacher rank 远大于 200），所以 ensure_target_in_topk() 每次都会把最后一个位置强行改成 target。
你想要 Phase2 更直接提升 HR@1：该怎么改 teacher_top 的生成
    如果 teacher 本身检不回 target，那么“纯 top200 头部列表”对 HR@1 的帮助有限（甚至会让 shaping 更偏离）。长期更有效的是把 pool 改成：
    “头部 top + target 附近（teacher 分数邻域）的 hard negatives + target”
    这样：
    target 不会永远是 pool 里“最低分那个”
    模型输出落在 target 附近更容易得到 shaping，RL 更容易把 HR@1 往上推



# 重构数据

将phase1构建数据的代码改名为build_teacher_topk_old.py

python HardMiningGRPO/build_teacher_topk.py \
  --input_jsonl  ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --output_jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.phase2.teacher200.headnear.jsonl \
  --overwrite \
  --sasrec_pkl  /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --topk 200 \
  --pool_mode head_near \
  --head_k 80 --near_above_k 60 --near_below_k 59 \
  --filter_history \
  --batch_size 2048 --chunk_size 50000 \
  --item_emb_on_gpu \
  --score_dtype fp16

teacher_topk_hit_rate (target naturally in head topk) = 0.000046
0.000046 × 195966 ≈ 9 条样本→ 也就是 约 20 万条里，只有 ~9 条 target 能自然进 teacher 的 head topk

可能性 1（最常见，也最该先排查）：ID 映射或序列方向不一致
重点核对：

history_item_ids 的顺序：你的 SASRec 推理假设“最后一位是最新行为”（feats[:, -1, :]），且你是 left pad。
如果你的 GRPO 数据里 history 是“从新到旧”（反的），teacher 会非常惨。

target_item_id 是否真的是“history 的下一跳 item”（next-item）
如果你的 target 是别的定义（比如点击后下下跳、或者被重映射过），teacher hit 会非常低。
GRPO jsonl 的 item_id space 是否和 sasrec_dataset.pkl 的 item_id space 完全一致（同一套 remap）。

可能性 2：teacher 本身确实弱（但一般不会弱到这个量级）
如果 teacher 训练不足、或数据 domain 不一致，也会低，但通常不会“20 万里才 9 条”。



## 现在的 Phase2 debug 最大问题是：输出不在 teacher pool 时 reward 不变化。
check_teacher_alignment.py：
    history 正向 vs 反向
    id shift：0 / +1 / -1
    最终告诉你哪个组合 HR@10/HR@1 最好 —— 这通常直接指出是“方向错了”还是“ID 映射错了”。

1) 如果 reversed + shift0 远好于 forward + shift0✅ 序列方向反了
    → 你的 GRPO history_item_ids 很可能是 新→旧，而 SASRec 需要 旧→新（最后一位最新）。
    修法：生成 GRPO 数据时把 history 反转（或在训练前 map 的时候反转）。

2) 如果 shift-1 或 shift+1 远好于 shift0✅ off-by-one 映射问题
    → 常见于：
    SASRec 的 item_id 从 1 开始（0 是 padding），但你 GRPO 用了 0-based 的 id
    或你某处做了 +1 remap / 没做 remap
    修法：统一映射（建议从源头生成 GRPO jsonl 时就按 SASRec 的 remap 输出）。

3) 如果所有组合 HR@10 都接近随机（非常非常低）✅ ID space 根本不一致（最严重）
    → 说明 GRPO jsonl 里的 item_id 不是 SASRec 那套 remap id（可能是原始 poi_id / sku_id / hash id）。
    修法：必须把 GRPO 数据的 history/target/candidates 全部映射到 sasrec_dataset.pkl 的 id space。


4) bad_tgt_in_hist 很高✅ 数据泄漏/定义错误
    → target 出现在 history 里，说明你构造 next-item label 时对齐错了（或者 history 包含了 target）。
    修法：重新生成训练样本，保证 target_item_id 是 history 的下一跳，且不在 history 内。

python HardMiningGRPO/check_teacher_alignment.py \
  --jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec/sasrec_full_latest.pth \
  --max_samples 5000 \
  --num_neg 199 \
  --batch_size 512 \
  --device cuda


## 结果如下：
结论 1：ID 映射“没有明显 off-by-one”
    因为 shift -1 / +1 都显著变差，说明你现在用的 target_item_id 与 history_item_ids 在 Teacher 视角下是对齐的（不是整体 +1/-1 那种错位）。
结论 2：序列方向基本是“你现在的方向”（forward）更合理
    反转也能跑，但指标略差：说明 Teacher 更认可你当前 history 的时间顺序（最新在右侧、left pad、取最后位表征）这一套。
结论 3：Teacher 并非“完全不行”，但也绝对达不到“全库 top200 常命中”
    你 forward+shift0 在 200 个候选里 MeanRank≈40、HR@10≈0.29，说明 Teacher 对 target 的排序有一定区分度。
✅ 这更像是 Teacher 排序能力“中等”，而不是 ID/方向错。

额外：Teacher Model 自身是否“真的弱”？（把 teacher 和 jsonl 的问题拆开）
    用你现成的 SASRec eval（在 pkl 的 strict split 上）跑一次

python HardMiningGRPO/diagnose_teacher_fullcorpus.py \
  --jsonl ./HardMiningGRPO/grpo_data_v2/grpo_train.cand.fixed_precise_v2.jsonl \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data_new/sasrec_full_latest.pth \
  --sample_n 5000 \
  --batch_size 256 \
  --chunk_size 50000 \
  --K 1,10,50,200,1000 \
  --device cuda \
  --emb_on_gpu \
  --score_dtype fp16 \
  --show_chunk_pbar false









