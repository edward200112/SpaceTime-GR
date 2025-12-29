python HardMiningSFT/make_sft_jsonl_from_teacher.py \
  --sasrec_data_path ./SASRec_Data/sasrec_dataset.pkl \
  --sasrec_model_path ./SASRec_Data/sasrec_full_latest.pt \
  --raw_meta_dir /workspace/data/GoogleRAW \
  --output_jsonl ./HardMiningSFT/sft_data/google_sft_train.jsonl \
  --topk 200 --max_hist_text 5 --max_len 50 \
  --max_samples 2000000



 python ./TeacherModel/train_sasrec.py   --dataset_path ./SASRec_Data/sasrec_dataset.pkl   --output_dir ./SASRec_Data_new   --batch_size 4096   --lr 1e-4   --num_epochs 50   --num_negs 4   --pop_alpha 0.75   --do_fast_eval --fast_eval_every 5 --fast_eval_users 2000 --fast_eval_neg 99 --fast_eval_batch_size 256   --do_strict_eval --strict_eval_every 10 --strict_eval_users 2000 --strict_eval_neg 99 --strict_eval_batch_size 128   --early_stop --early_stop_metric NDCG@10 --early_stop_patience 5 --early_stop_min_delta 0.002   --pin_memory --num_workers 14

生成stage1的数据

python HardMiningSFT/make_sft_jsonl_unified.py \
  --stage stage1 \
  --sasrec_data_path ./SASRec_Data/sasrec_dataset.pkl \
  --raw_meta_dir /workspace/data/GoogleRAW \
  --output_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --max_hist_text 5 \
  --min_seq_len 2 \
  --max_samples 2000000

生成stage2 的数据
python HardMiningSFT/make_sft_jsonl_unified.py \
  --stage stage2 \
  --sasrec_data_path ./SASRec_Data/sasrec_dataset.pkl \
  --sasrec_model_path ./SASRec_Data/sasrec_full_latest.pt \
  --raw_meta_dir /workspace/data/GoogleRAW \
  --output_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k.jsonl \
  --device cuda \
  --infer_bs 1024 \
  --num_neg 199 --pop_top 200000 --oversample 8 \
  --p_hard 0.4 --p_semi 0.4 --p_easy 0.2 --semi_margin 1.0 \
  --neg_cap 5000 \
  --max_samples 800000


stage1 继续训练，使用2M数据（只加载 LoRA（不恢复 optimizer/scheduler），从200k->2M数据）
python HardMiningSFT/train_stage1_sft_2M.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage1_continue_from_lora_only \
  --init_from_adapter ./HardMiningSFT/ckpt_stage1/checkpoint-22000 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 24 --grad_accum 2 \
  --lr 2e-5 --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory


如果中断了，想“真正 resume”同一次训练（不丢步数、不覆盖）
python HardMiningSFT/train_stage1_sft_resume_or_lora.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage1_continue_from_lora_only \
  --resume_trainer \
  --max_length 1024 --num_epochs 1 \
  --batch_size 24 --grad_accum 2 \
  --lr 2e-5 --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory



给你一个独立 eval 脚本（Stage1：生成+统计）
  这个 eval 主要回答你“要不要继续 Stage1”：
  exact match（严格匹配）
  contains match（宽松匹配）
  无效输出率
  top1 输出是否塌缩（mode collapse）
  平均生成长度
python HardMiningSFT/eval_stage1_generate.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 32 --max_new_tokens 48



python HardMiningSFT/report_neg_repetition.py   --jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl   --topn 50
----------------------------------------

lines: 200000
top1  coverage:  0.0050%
top10 coverage:  0.0450%

top100 coverage: 0.3450%
========================================


python HardMiningSFT/train_stage1_sft.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_sft_train.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage1 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 16 --grad_accum 2 \
  --lr 2e-5 --save_steps 1000 --logging_steps 50



直接用 hard_level 权重（最直观）
比如你想 不要 hard++ 太强，同时 给 easy/medium 更多存在感：

TOKENIZERS_PARALLELISM=false \
python HardMiningSFT/train_stage2_coin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1 \
  --data_jsonl ./HardMiningSFT/sft_data/google_sft_train_batchcand.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_coin_mix \
  --max_length 1024 --num_epochs 1 \
  --batch_size 8 --grad_accum 2 \
  --lr 1e-5 --lambda_coin 0.1 --contrastive_margin 0.5 \
  --coin_weight_mode hard_level \
  --w_easy 1.5 --w_medium 1.2 --w_hard 1.0 --w_hard_plus 0.8 --w_hard_pp 0.6 \
  --save_steps 1000 --logging_steps 50

B) 用 gap 连续加权（更细粒度）
如果你的 jsonl 有 gap 或 teacher_gap：

TOKENIZERS_PARALLELISM=false \
python HardMiningSFT/train_stage2_coin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1 \
  --data_jsonl ./HardMiningSFT/sft_data/google_sft_train_batchcand.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_coin_gap \
  --max_length 1024 --num_epochs 1 \
  --batch_size 8 --grad_accum 2 \
  --lr 1e-5 --lambda_coin 0.1 \
  --coin_weight_mode gap \
  --gap_clip_min -10 --gap_clip_max -1 \
  --gap_w_min 0.6 --gap_w_max 1.4 \
  --save_steps 1000 --logging_steps 50



stage2 分层margin
分层设置 margin，就是把你现在 CoIN 里“负样本要被推开多远”的阈值 按难度分组来设，而不是所有样本都用同一个 contrastive_margin=0.5。
所以 margin 是“允许的最大相似度阈值”：

margin 越小 ⇒ 要求更严格（相似度稍微高一点就惩罚）⇒ push 更强、更激进
margin 越大 ⇒ 要求更宽松（相似度要非常高才惩罚）⇒ push 更弱、更稳

python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1 \
  --data_jsonl ./HardMiningSFT/sft_data/google_sft_train_batchcand.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin \
  --max_length 1024 --num_epochs 1 \
  --batch_size 8 --grad_accum 2 \
  --lr 2e-5 --lambda_coin 0.1 --default_margin 0.2 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.10 --m_medium 0.15 --m_hard 0.20 --m_hardpp 0.25 \
  --num_workers 2 --pin_memory





==============DEBUG=================
实验 1：只改 eval，把 PROMPT_RULE 拼进 prompt（验证 prompt mismatch 是否导致 Top1 collapse）

python HardMiningSFT/eval_stage1_generate_with_rule.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 32 --max_new_tokens 48

实验 2：可视化 assistant_start 断点（验证 mask 是否错位 / add_special_tokens 不一致）
python HardMiningSFT/debug_assistant_start.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n 5 --max_length 1024

实验 3：统计训练集 completion 的 Top-10（判断数据本身是否高度偏斜）
python HardMiningSFT/stats_top_completions.py \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --topk 20


(py312) root@af2099a629c5:/workspace/Rank-GRPO# python HardMiningSFT/stats_copy_last.py \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n 200000
Counted: 200000 (no_match=0)
GT == last_item_in_prompt: 7109/200000 (3.5545%)
GT appears somewhere in prompt: 8733/200000 (4.3665%)
(py312) root@af2099a629c5:/workspace/Rank-GRPO# python HardMiningSFT/stats_prompt_truncation.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --max_length 1024 --n 200000
Counted: 200000
max_length=1024, hit_max(>=max): 0/200000 (0.0000%)
P50: 74
P90: 83
P95: 86
P99: 92
Mean: 73.92, Max: 140
(py312) root@af2099a629c5:/workspace/Rank-GRPO# python HardMiningSFT/eval_stage1_nll.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_length 1024
N=2000, avg_NLL=1.408023, ppl=4.088, total_tokens=20690

那为什么 generation exact 还这么低、而且 Top1 常数很高？
这通常是典型的现象：
GT 序列“有一定概率”，但不是条件分布的 mode（argmax）
于是 greedy decoding 永远选一个全局最稳的答案（比如 “The Home Depot …”），看起来就像塌缩。
换句话说：你现在的任务在文本空间里是 高不确定性/多解（同样的短历史，下一跳可能很多），SFT 会学到一个“平滑的条件分布”，但 greedy 只会挑 mode，导致：
NLL/PPL 看起来不错（因为 GT 概率还行）
exact 很差（因为 GT 往往不是 argmax 那条路径）
Top1 常数很高（因为全局 mode 在很多条件下都赢）



实验 A：算一下 base model 的 NLL/PPL（不加载 adapter）对比

实验 B：做一个 Beam@K Oracle exact（GT 是否出现在 beam 的前 K 条里）

顺带一提：你现在这个任务用 “生成一个具体店名字符串” 来做 next-item 预测，本身就很难
因为：
店名/类别是自然语言、同名店很多、别名很多
next item 天然多解
用 greedy 生成等价于“取 mode”，很容易输出一个全局常见店
所以 stage2 用困难负样本做 ranking/contrastive 是方向正确的；stage1 生成 exact 不高并不一定代表 stage1 无效——关键要看 Oracle@K 或 ranking 指标。


对每个样本，用 beam search 生成 K 条候选答案；
只要 GT（标注的 completion） 出现在这 K 条中的任意一条，就算这个样本命中。
命中率就是 Oracle@K.


(py312) root@af2099a629c5:/workspace/Rank-GRPO# python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio



[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

BEAM ORACLE EVAL (K=10)
========================================

Eval samples:   2000

Oracle@10 hit: 97/2000 (0.0485)
----------------------------------------

Beam1 Top1 ratio: 451/2000 (0.2255)

Beam1 Top1 sample: McDonald's (Fast food restaurant)
========================================



(py312) root@af2099a629c5:/workspace/Rank-GRPO# for k in 1 5 10 20 50; do
  python HardMiningSFT/eval_stage1_beam_oracle.py \
    --base_model /workspace/Qwen2_5-1.5B-Instruct \
    --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
    --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
    --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams $k
done
[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

BEAM ORACLE EVAL (K=1)
========================================

Eval samples:   2000

Oracle@1 hit: 21/2000 (0.0105)
========================================

[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

BEAM ORACLE EVAL (K=5)
========================================

Eval samples:   2000

Oracle@5 hit: 62/2000 (0.0310)
========================================

[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

BEAM ORACLE EVAL (K=10)
========================================

Eval samples:   2000

Oracle@10 hit: 97/2000 (0.0485)
========================================

[INFO] eos_id_used=151645 (im_end_id=151645)
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

========================================

BEAM ORACLE EVAL (K=20)
========================================

Eval samples:   2000

Oracle@20 hit: 157/2000 (0.0785)
========================================

[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

BEAM ORACLE EVAL (K=50)
========================================

Eval samples:   2000

Oracle@50 hit: 237/2000 (0.1185)
========================================



----BASE MODEL NLL/PPL----
(py312) root@af2099a629c5:/workspace/Rank-GRPO# python HardMiningSFT/eval_stage1_beam_oracle_base.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10
[INFO] eos_id_used=151645 (im_end_id=151645)

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
========================================

[BASE] BEAM ORACLE EVAL (K=10)
========================================

Eval samples:   2000

Oracle@10 hit: 22/2000 (0.0110)
========================================




==============DEBUG=================




stage1+IPS
python HardMiningSFT/train_stage1_sft_ips_resume_or_lora.py \
  --model_id /workspace/Qwen2_5-1.5B-Instruct \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage1_ips_continue \
  --resume_trainer \
  --resume_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --ips_mode completion_freq --ips_beta 0.5 --ips_min 0.2 --ips_max 5.0 --ips_smoothing 1.0 \
  --extra_steps 2000 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 24 --grad_accum 2 \
  --lr 2e-5 --save_steps 500 --logging_steps 50 \
  --num_workers 2 --pin_memory



Stage1+IPS eval
# HardMiningSFT/train_stage1_sft_ips_resume_or_lora.py
# 尽管没用，但是加入ips证明了 能够降低top1 的指标，所以保留这份代码，但是会降低oracal指标
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio



运行指令（生成 stage2 数据，带 ips_weight）

python HardMiningSFT/make_sft_jsonl_unified.py \
  --stage stage2 \
  --sasrec_data_path ./SASRec_Data/sasrec_dataset.pkl \
  --sasrec_model_path ./SASRec_Data/sasrec_full_latest.pt \
  --raw_meta_dir /workspace/data/GoogleRAW \
  --output_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k.jsonl \
  --device cuda \
  --infer_bs 1024 \
  --num_neg 199 --pop_top 200000 --oversample 8 \
  --p_hard 0.4 --p_semi 0.4 --p_easy 0.2 --semi_margin 1.0 \
  --neg_cap 5000 \
  --max_samples 800000 \
  --ips_beta 0.5 --ips_smoothing 10.0 --ips_min 0.2 --ips_max 5.0 \
  --ips_count_tail_k 2


你生成完怎么快速 sanity check（建议你顺手做）

python - << 'PY'
import json, random
path = "./HardMiningSFT/sft_data/google_stage2_coin_800k.jsonl"
with open(path, "r", encoding="utf-8") as f:
    lines = [next(f) for _ in range(20)]
for ln in random.sample(lines, 5):
    d = json.loads(ln)
    print(d["ips_weight"], d["meta"].get("gt_freq"), d["completion"][:50])
PY


我们还可以把 ips_weight 做成 “正样本长尾 + hard_level 更强” 的组合，比如
ips_weight *= (1 + alpha * I[hard_level in hard/hard++])，让“长尾且困难”的样本更被重视。你先按这版跑起来，确认 ips_weight 生效后再加这个也不迟。

把“用 SASRec 候选集评测”的脚本写出来（输出 Recall@1/5/10/20、MRR）




stage2训练

现在就可以开训 stage2（推荐用 stage1+IPS 的 ckpt-35000 初始化）
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_ips_v2 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 12 --grad_accum 2 \
  --lr 2e-5 --lambda_coin 0.1 --default_margin 0.2 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.10 --m_medium 0.15 --m_hard 0.20 --m_hardpp 0.25 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory



断点接着练
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_ips_v2 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 8 --grad_accum 2 \
  --lr 2e-5 --lambda_coin 0.1 --default_margin 0.2 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.10 --m_medium 0.15 --m_hard 0.20 --m_hardpp 0.25 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory \
  --resume





python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_ips_v3/checkpoint-2000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio
========================================
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   2000
Oracle@10 hit: 76/2000 (0.0380)
Strict-format rate (Top1): 1983/2000 (0.9915)
----------------------------------------
Beam1 Top1 ratio: 767/2000 (0.3835)
Beam1 Top1 sample: Walmart Supercenter (Department store)
========================================



python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_ips_v2/checkpoint-8000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio
========================================
ckpt-8000
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   2000
Oracle@10 hit: 53/2000 (0.0265)
Strict-format rate (Top1): 1742/2000 (0.8710)
----------------------------------------
Beam1 Top1 ratio: 438/2000 (0.2190)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================



new stage2 training
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_ips_v3 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 10 --grad_accum 2 \
  --lr 2e-5 --lambda_coin 0.03 --default_margin 0.25 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.12 --m_medium 0.20 --m_hard 0.28 --m_hardpp 0.35 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory




新数据生成，✅ “去偏”力度够了，解决IPS 作用偏弱的主要原因，减小 beta + 提升 alpha：
python HardMiningSFT/make_sft_jsonl_unified.py \
  --stage stage2 \
  --sasrec_data_path ./SASRec_Data/sasrec_dataset.pkl \
  --sasrec_model_path ./SASRec_Data/sasrec_full_latest.pt \
  --raw_meta_dir /workspace/data/GoogleRAW \
  --output_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --device cuda \
  --infer_bs 1024 \
  --num_neg 199 --pop_top 200000 --oversample 8 \
  --p_hard 0.6 --p_semi 0.3 --p_easy 0.1 --semi_margin 1.0 \
  --hard_topM 10 \
  --neg_cap 5000 \
  --ips_alpha 0.8 --ips_beta 1 --ips_min 0.2 --ips_max 3.0 \
  --max_samples 800000


GT-weighted mean = 1.0000 ✅
说明你构建 IPS 的“全局归一化”是对的（整体 loss scale 不会被系统性放大/缩小）。
min = 0.0345, max = 3.9202 ✅
说明在“未 clip 前”，权重范围已经变宽了（相比上次 max≈1.57 强很多）。



Stage2 从头开始训（推荐你先跑 2k step 看效果）
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_ips_rule_hard_strongerIPS \
  --max_length 1024 --num_epochs 1 \
  --batch_size 10 --grad_accum 2 \
  --lr 2e-5 \
  --lambda_coin 0.03 --default_margin 0.25 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.10 --m_medium 0.15 --m_hard 0.25 --m_hardpp 0.35 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory
我把 hard/hard++ 的 margin 稍微拉高了一点（0.25/0.35），配合你更贴的 hard neg + 更强 IPS；同时 lambda_coin=0.03 保守，避免梯度又炸。

Stage2 继续训练（从 output_dir 里最新 checkpoint 自动 resume）

python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_ips_rule_hard_strongerIPS \
  --max_length 1024 --num_epochs 1 \
  --batch_size 16 --grad_accum 2 \
  --lr 2e-5 \
  --lambda_coin 0.03 --default_margin 0.25 \
  --w_easy 0.3 --w_medium 0.6 --w_hard 1.0 --w_hardpp 1.2 \
  --m_easy 0.10 --m_medium 0.15 --m_hard 0.25 --m_hardpp 0.35 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory \
  --resume


至少跑到 2k/5k step 再做一次 beam_oracle / top1 ratio 对比才有意义。
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_ips_rule_hard_strongerIPS/checkpoint-2000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio
===
结果如下
===

========================================
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   2000
Oracle@10 hit: 48/2000 (0.0240)
Strict-format rate (Top1): 1985/2000 (0.9925)
----------------------------------------
Beam1 Top1 ratio: 365/2000 (0.1825)
Beam1 Top1 sample: El Pollo Loco (Mexican restaurant)
========================================


# 分析
这份 eval 还有一个关键问题：它在测“你想要的指标”吗？
你现在用的是 eval_stage1_beam_oracle.py 这种“生成 exact hit”的方式。
但 stage2 的优化目标（你现在关 consistency）本质是：
  NTP（生成） + hinge(表示拉开)
  它优化的不是“beam 里出现 GT 的概率”这一项的直接上界；尤其 hinge 更像正则。
这份 eval 对阶段性趋势是有参考的，但不要期待它和 hinge 指标强一致。

更合理的对照：你至少要同时跑两套 eval：
1）在 stage1_pos_2m 上跑 Oracle@10（看泛化）
2）在 stage2_triplet 上跑 hinge 相关统计（你已经在 coin_debug 做了）

新train，拉低学习率 以及 warmup，因为上一次训练的loss和grad 有点高
从头跑（warmup + lr 降）
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 12 --grad_accum 2 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --lambda_coin 0.03 --default_margin 0.25 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory



继续训练（resume）
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_ips_continue/checkpoint-35000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5 \
  --max_length 1024 --num_epochs 1 \
  --batch_size 16 --grad_accum 2 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --lambda_coin 0.03 --default_margin 0.25 \
  --save_steps 1000 --logging_steps 50 \
  --num_workers 2 --pin_memory \
  --resume

评估
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5/checkpoint-8000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio

  

python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5/checkpoint-6000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio


python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_ips_rule_hard_strongerIPS/checkpoint-2000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio


新的eval
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5/checkpoint-8000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 2000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio


========================================
800k数据，ckpt-8000
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   2000
Oracle@10 hit (exact): 42/2000 (0.0210)
Strict-format rate (Top1): 1986/2000 (0.9930)
----------------------------------------
NameExact@1:        12/2000 (0.0060)
CatExact@1:         71/2000 (0.0355)
NameEditSim>=0.90:  13/2000 (0.0065)
NameEditSim>=0.80:  13/2000 (0.0065)
NameJaccard>=0.80:  13/2000 (0.0065)
Distinct@1:         0.2190
Entropy(Top1):      3.5253
----------------------------------------
Beam1 Top1 ratio: 644/2000 (0.3220)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================




========================================
2M数据，ckpt-8000
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   2000
Oracle@10 hit (exact): 65/2000 (0.0325)
Strict-format rate (Top1): 1982/2000 (0.9910)
----------------------------------------
NameExact@1:        22/2000 (0.0110)
CatExact@1:         71/2000 (0.0355)
NameEditSim>=0.90:  24/2000 (0.0120)
NameEditSim>=0.80:  24/2000 (0.0120)
NameJaccard>=0.80:  22/2000 (0.0110)
Distinct@1:         0.2950
Entropy(Top1):      4.0023
----------------------------------------
Beam1 Top1 ratio: 539/2000 (0.2695)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================



新的训练代码，从ckpt-8000 resume，把 rank loss 改成“不会依赖 sim_pos=1 的版本”，只影响尺度、不改变样本相对权重的处理
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5 \
  --resume \
  --lr 1e-5 \
  --lambda_coin 0.03 \
  --neg_tau 0.60 \
  --warmup_ratio 0.03 \
  --batch_size 4 --grad_accum 4 \
  --save_steps 1000 --logging_steps 50 \
  --attn_impl flash_attention_2


step=8100/8200 时把 coin_debug.jsonl

===
新测试，看看 top1 的 most_common 前 10,确认是不是被少数“超级热门”店名占领
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5/checkpoint-10000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio\
  --report_topk 10


========================================
坍塌（ckpt-10000废弃）
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 144/5000 (0.0288)
Strict-format rate (Top1): 4948/5000 (0.9896)
----------------------------------------
NameExact@1:        48/5000 (0.0096)
CatExact@1:         190/5000 (0.0380)
NameEditSim>=0.90:  49/5000 (0.0098)
NameEditSim>=0.80:  50/5000 (0.0100)
NameJaccard>=0.80:  49/5000 (0.0098)
Distinct@1:         0.1820
Entropy(Top1):      3.5868
----------------------------------------
Top10 Top1 predictions:
01. 1275/5000 (0.2550)  Taco Bell (Fast food restaurant)
02. 1054/5000 (0.2108)  The UPS Store (Shipping and mailing service)
03.  536/5000 (0.1072)  El Pollo Loco (Mexican restaurant)
04.  185/5000 (0.0370)  Safeway (Grocery store)
05.  169/5000 (0.0338)  Starbucks (Coffee shop)
06.  131/5000 (0.0262)  McDonald's (Fast food restaurant)
07.   93/5000 (0.0186)  Popeyes Louisiana Kitchen (Chicken restaurant)
08.   50/5000 (0.0100)  Walmart Supercenter (Department store)
09.   42/5000 (0.0084)  T-Mobile (Cell phone store)
10.   41/5000 (0.0082)  Subway (Sandwich shop)
----------------------------------------
Beam1 Top1 ratio: 1275/5000 (0.2550)
Beam1 Top1 sample: Taco Bell (Fast food restaurant)
========================================





python HardMiningSFT/train_stage2_coin_rankmargin.py   --base_model /workspace/Qwen2_5-1.5B-Instruct   --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000   --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl   --output_dir ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5   --resume   --lr 1e-5   --lambda_coin 0.04   --default_margin 0.20   --warmup_ratio 0.03   --batch_size 12 --grad_accum 4  --save_steps 1000 --logging_steps 50 --attn_impl flash_attention_2



new eval：
python HardMiningSFT/eval_stage1_beam_oracle.py   --base_model /workspace/Qwen2_5-1.5B-Instruct   --adapter ./HardMiningSFT/ckpt_stage2_rankmargin_strongerIPS_warmup_lr1e5/checkpoint-10000   --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl   --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10   --report_top1_ratio  --report_topk 10

========================================
ckpt-10000
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 194/5000 (0.0388)
Strict-format rate (Top1): 4958/5000 (0.9916)
----------------------------------------
NameExact@1:        73/5000 (0.0146)
CatExact@1:         259/5000 (0.0518)
NameEditSim>=0.90:  74/5000 (0.0148)
NameEditSim>=0.80:  74/5000 (0.0148)
NameJaccard>=0.80:  73/5000 (0.0146)
Distinct@1:         0.3018
Entropy(Top1):      4.5478
----------------------------------------
Top10 Top1 predictions:
01. 1151/5000 (0.2302)  Taco Bell (Fast food restaurant)
02.  709/5000 (0.1418)  The UPS Store (Shipping and mailing service)
03.  417/5000 (0.0834)  El Pollo Loco (Mexican restaurant)
04.  182/5000 (0.0364)  Safeway (Grocery store)
05.  150/5000 (0.0300)  Starbucks (Coffee shop)
06.  114/5000 (0.0228)  McDonald's (Fast food restaurant)
07.   86/5000 (0.0172)  Popeyes Louisiana Kitchen (Chicken restaurant)
08.   54/5000 (0.0108)  Walmart Supercenter (Department store)
09.   50/5000 (0.0100)  Wendy's (Fast food restaurant)
10.   39/5000 (0.0078)  T-Mobile (Cell phone store)
----------------------------------------
Beam1 Top1 ratio: 1151/5000 (0.2302)
Beam1 Top1 sample: Taco Bell (Fast food restaurant)


## 为什么在2m数据上更好
主要来自 数据分布 + 评估方式 两件事的叠加，而不一定代表模型更“会推荐”。
你的 eval 的核心是 exact match（Oracle@K / NameExact@1）。这类指标对数据分布非常敏感：

stage1（2m）是纯正样本，completion 更“干净”、噪声更小、表达更统一（例如连锁店名 + 类别这种规范格式）。
stage2（800k）引入 hard negatives + 采样策略 + IPS，数据更复杂：
  completion 的“真值”可能更长尾、更细碎
  prompt/历史/候选空间更复杂
  如果生成 hard negative 的规则或 meta 拼接存在微小不一致，exact-match 会立刻吃亏（比如括号类别略有差异也算错）
所以同一个模型在 stage1 上更容易拿到“看起来更好”的 exact-match 分数。

你 top10 里总是 Taco Bell / UPS / McD / Starbucks，这说明数据里强头部存在。
如果 stage1 数据对这些头部连锁出现频次更高，那么模型只要“偏向头部答案”，在 stage1 上就能吃到不少分（尤其 Oracle@10 这种只要某个 beam 撞对就算命中）。




## 检查dropout
“dropout 很小/为 0 导致 sim_pos≈1、hinge 不触发”是什么意思？你要不要先检查？
你现在的 self-positive ranking 本质上要用：
  sim_pos = cos(pos_repr_view1, pos_repr_view2)
  sim_neg = cos(pos_repr_view1, neg_repr)
  hinge = ReLU(margin - (sim_pos - sim_neg))
其中 pos_repr_view1/view2 需要是“同一个输入的两次不同视角”（views），不然 sim_pos 永远接近 1。

1、模型配置 dropout=0 或极小
  很多现代大模型为了稳定和可复现，会把 dropout 设为 0（尤其推理型/指令模型经常这样）。
  如果 dropout=0，即使 model.train()，两次 forward 也是几乎一致的。
2、训练时确实是 train()，但你用的路径没有任何随机算子
  比如没有 dropout、没有随机 mask，forward 是确定性的。





# self-positive ranking 为什么“几乎没训练信号”
你用“同一个 input 再 forward 一次”当正对，理论上应靠 dropout 产生差异；但 Qwen2.5 这类模型很可能 dropout=0 或极小，导致两次 forward 几乎一致 ⇒ sim_pos ≈ 1，从而 hinge 很难触发。



既然 2m 指标更好，为什么不能在 GRPO 上用更大的数据集训练？
  更现实的工程观点：大数据 ≠ 更好，通常是“分层采样 + 高信息子集”更好
  在推荐里，最有效的训练数据往往不是全量，而是：
  覆盖头部与长尾的分层采样
  偏向不确定/困难样本的高信息子集
  对曝光偏置做校正（你用 IPS 就是这个方向）



# SASRec 会不会是问题？需要检查吗？
GroundTruth（completion）是从真实交互里取的

1) False Negative（伪负例）比例过高

困难负样本本质上更像“用户可能也喜欢但没点过”的项目。
如果 neg 真的很像 pos，你在 stage2 的训练里等于在教模型：
> 同样合理的店名/类别也要当负例压下去

2) 负样本分布过于头部化
如果 SASRec hard negative 采样策略（pop_top、oversample、neg_cap 等）导致 neg 过于集中在头部连锁店，模型会频繁看到 “TacoBell/UPS/ElPollo” 作为负例/或正例，同时 NTP 又天然偏高频 token，最终很容易塌缩到少数答案。
 
>所以需要检查：stage2 数据里 completion 和 negative_completion 的 TopK 分布是不是已经非常头部。


3) stage2 数据本身 head 极强

即使 SASRec 完全没问题，如果 800k stage2 的 completion 自身就头部极强，模型“输出头部答案”在 NTP 上就是最优策略，CoIN 也很难扳回来。


✅ 结论：不是说 SASRec 一定有问题，但你应该检查的是：
  stage2 数据（completion/negative_completion）的分布是否过头部
  hard negative 是否太“真”/太集中
  这两项比“GT 是否正确”更关键。



# 检查数据（困难负样本是否为伪困难） + Dropout（验证 sim_pos≈1 的根因）
数据分布检查（最重要） + dropout/视角检查（验证 sim_pos≈1 的根因）。
===

python - <<'PY'
import json
from collections import Counter
import random

def topk(path, field, k=20, nmax=200000):
    c = Counter()
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= nmax: break
            ex = json.loads(line)
            v = ex.get(field, "")
            if v: c[v] += 1
    total = sum(c.values())
    print(f"\n[{path}] field={field}, total_counted={total}")
    for i,(x,cnt) in enumerate(c.most_common(k), 1):
        print(f"{i:02d}. {cnt}/{total} ({cnt/total:.4f})  {x[:120]}")
    return c

stage1 = "./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl"
stage2 = "./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl"

topk(stage1, "completion", k=10, nmax=200000)
topk(stage2, "completion", k=10, nmax=200000)
topk(stage2, "negative_completion", k=10, nmax=200000)
PY

===
[./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl] field=completion, total_counted=200000
01. 1940/200000 (0.0097)  McDonald's (Fast food restaurant)
02. 1606/200000 (0.0080)  Starbucks (Coffee shop)
03. 1169/200000 (0.0058)  In-N-Out Burger (Hamburger restaurant)
04. 1115/200000 (0.0056)  Walmart Supercenter (Department store)
05. 1000/200000 (0.0050)  Jack in the Box (Fast food restaurant)
06. 787/200000 (0.0039)  Costco Wholesale (Warehouse store)
07. 749/200000 (0.0037)  The Home Depot (Home improvement store)
08. 732/200000 (0.0037)  Taco Bell (Fast food restaurant)
09. 589/200000 (0.0029)  Denny's (Diner)
10. 563/200000 (0.0028)  Subway (Sandwich shop)

[./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl] field=completion, total_counted=200000
01. 1940/200000 (0.0097)  McDonald's (Fast food restaurant)
02. 1606/200000 (0.0080)  Starbucks (Coffee shop)
03. 1169/200000 (0.0058)  In-N-Out Burger (Hamburger restaurant)
04. 1115/200000 (0.0056)  Walmart Supercenter (Department store)
05. 1000/200000 (0.0050)  Jack in the Box (Fast food restaurant)
06. 787/200000 (0.0039)  Costco Wholesale (Warehouse store)
07. 749/200000 (0.0037)  The Home Depot (Home improvement store)
08. 732/200000 (0.0037)  Taco Bell (Fast food restaurant)
09. 589/200000 (0.0029)  Denny's (Diner)
10. 563/200000 (0.0028)  Subway (Sandwich shop)

[./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl] field=negative_completion, total_counted=200000
01. 4491/200000 (0.0225)  McDonald's (Fast food restaurant)
02. 2202/200000 (0.0110)  Walmart Supercenter (Department store)
03. 1664/200000 (0.0083)  In-N-Out Burger (Hamburger restaurant)
04. 1225/200000 (0.0061)  Denny's (Diner)
05. 1182/200000 (0.0059)  Best Buy (Electronics store)
06. 1101/200000 (0.0055)  Costco Wholesale (Warehouse store)
07. 967/200000 (0.0048)  The Home Depot (Home improvement store)
08. 966/200000 (0.0048)  Wendy's (Fast food restaurant)
09. 920/200000 (0.0046)  Chick-fil-A (Fast food restaurant)
10. 782/200000 (0.0039)  Jack in the Box (Fast food restaurant)

===
你要观察的点：

stage2 的 completion Top1/Top3 是否已经很夸张（比如 Top1>0.15、Top3>0.30）
stage2 的 negative_completion 是否也被同样的少数连锁店占据
如果这两者本身就极头部，你模型塌缩基本是“数据/目标函数共同决定”的。

## 分析：
这意味着：你在 stage2 eval 上看到的“输出塌缩到 TacoBell/UPS/ElPollo 占比 20%~30%”，不是因为正样本数据本身就这么头部。
负样本里 top1（McD）是 2.25%。比正样本更头部一些，但还远没到你 eval 时那种 20%~30% 的头部占比。

B) 检查 negative 是否经常等于 positive（数据 bug 快速排雷）
===
python - <<'PY'
import json

path = "./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl"
n = 0
same = 0
empty_neg = 0
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        ex = json.loads(line)
        n += 1
        pos = ex.get("completion","")
        neg = ex.get("negative_completion","")
        if not neg:
            empty_neg += 1
        if pos and neg and pos.strip() == neg.strip():
            same += 1
        if n >= 200000:
            break

print("checked:", n)
print("negative empty:", empty_neg, empty_neg/n)
print("pos==neg:", same, same/n)
PY
===
若 pos==neg 比例明显 > 0（比如 >0.1% 都值得查生成逻辑），那会严重干扰 CoIN。
checked: 200000
negative empty: 0 0.0
pos==neg: 124 0.00062
===

C) 检查 Qwen2.5 的 dropout 是否为 0 / 很小（验证你是否需要 view-dropout）

===
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model="/workspace/Qwen2_5-1.5B-Instruct"
m = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)
cfg = m.config

keys = [k for k in dir(cfg) if "drop" in k.lower()]
print("Config dropout-like keys:", keys)
for k in keys:
    try:
        print(k, "=", getattr(cfg,k))
    except Exception:
        pass

# scan modules for Dropout layers and list unique p
ps=set()
for mod in m.modules():
    if mod.__class__.__name__.lower() == "dropout":
        ps.add(float(mod.p))
print("Unique nn.Dropout p:", sorted(ps))
PY

===
如果你看到 Unique nn.Dropout p: [0.0] 或者 config 里 dropout 都接近 0，
那“同 input forward 两次”确实会让 sim_pos≈1，hinge 很难触发。此时 必须加 view-dropout。
Config dropout-like keys: ['attention_dropout']
attention_dropout = 0.0
Unique nn.Dropout p: []

===
D) 直接测 sim_pos 的实际分布（最直接）
===
python - <<'PY'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model="/workspace/Qwen2_5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)
m.train()

text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nTaco Bell (Fast food restaurant)<|im_end|>"
enc = tok([text]*16, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    o1 = m(**enc, output_hidden_states=True)
    o2 = m(**enc, output_hidden_states=True)
    h1 = o1.hidden_states[-1].float().mean(dim=1)
    h2 = o2.hidden_states[-1].float().mean(dim=1)
    sim = F.cosine_similarity(h1, h2, dim=-1)

print("sim_pos mean:", sim.mean().item(), "min:", sim.min().item(), "max:", sim.max().item())
PY
===
如果这里输出仍然接近 0.99+，那你不需要犹豫：直接上 view-dropout。
`torch_dtype` is deprecated! Use `dtype` instead!
sim_pos mean: 1.0 min: 1.0 max: 1.0

===

## 总结
**SASRec一点毛病没有**
# 问题分析解决过程
在 Stage2 的 CoIN/self-positive ranking 训练中，我们最初采用“同一 input 进行两次 forward 得到两份表示作为正对（pos_view1/pos_view2）”，期望依赖模型内部 dropout 的随机性让两次表示产生差异，从而使 `sim_pos = cos(pos_view1, pos_view2) < 1`，再通过 `hinge = ReLU(margin - (sim_pos - sim_neg))` 产生稳定梯度来推开困难负样本；但训练日志中出现 `contrastive_loss`/`hinge_pos_rate` 经常为 0、`sim_pos_mean` 接近 1 的异常现象。进一步用验证脚本检查发现：Qwen2.5-1.5B 的 `attention_dropout=0.0`，模型中不存在 `nn.Dropout` 层（`Unique nn.Dropout p: []`），并且实测同一输入两次 forward 的 `sim_pos` 恒为 1.0（mean/min/max 全为 1.0），导致正对相似度几乎固定为 1、hinge 大多数样本不触发，使 CoIN 项实际“没有训练信号”，训练退化为仅靠 NTP 目标优化。解决方案是在表示层显式引入 **view-dropout**：对 `pos_repr` 进行两次独立的 `F.dropout(..., training=True)` 生成两个视角，再计算 `sim_pos` 与 `sim_neg` 并继续使用 margin hinge，从而在底模 dropout=0 的情况下也能强制产生 `sim_pos < 1`、提高 `hinge_pos_rate`，让对比/排序损失重新有效工作并用于抑制输出塌缩。




新的训练指令：
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_vdrop_run1 \
  --resume \
  --lr 1e-5 \
  --lambda_coin 0.04 \
  --default_margin 0.20 \
  --warmup_ratio 0.03 \
  --batch_size 12 --grad_accum 4 \ 
  --save_steps 1000 --logging_steps 50 \
  --attn_impl flash_attention_2


新的eval代码：
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run1/checkpoint-12000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10

python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run1/checkpoint-12000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10


新指令：
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_vdrop_run2_m0p30_v0p15 \
  --resume \
  --lr 1e-5 \
  --lambda_coin 0.04 \
  --default_margin 0.30 \
  --view_dropout 0.15 \
  --warmup_ratio 0.03 \
  --batch_size 12 --grad_accum 4 \
  --save_steps 1000 --logging_steps 50 \
  --attn_impl flash_attention_2


eval

python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run2_m0p30_v0p15/checkpoint-14000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10


  python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run2_m0p30_v0p15/checkpoint-14000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10


========================================
800k
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 145/5000 (0.0290)
Strict-format rate (Top1): 4949/5000 (0.9898)
----------------------------------------
NameExact@1:        49/5000 (0.0098)
CatExact@1:         193/5000 (0.0386)
NameEditSim>=0.90:  50/5000 (0.0100)
NameEditSim>=0.80:  52/5000 (0.0104)
NameJaccard>=0.80:  49/5000 (0.0098)
Distinct@1:         0.1856
Entropy(Top1):      3.5334
----------------------------------------
Top10 Top1 predictions:
01. 1555/5000 (0.3110)  The UPS Store (Shipping and mailing service)
02.  941/5000 (0.1882)  Taco Bell (Fast food restaurant)
03.  439/5000 (0.0878)  El Pollo Loco (Mexican restaurant)
04.  177/5000 (0.0354)  Starbucks (Coffee shop)
05.  176/5000 (0.0352)  Safeway (Grocery store)
06.   91/5000 (0.0182)  McDonald's (Fast food restaurant)
07.   82/5000 (0.0164)  Walmart Supercenter (Department store)
08.   63/5000 (0.0126)  Dollar Tree (Dollar store)
09.   48/5000 (0.0096)  Popeyes Louisiana Kitchen (Chicken restaurant)
10.   36/5000 (0.0072)  The Home Depot (Home improvement store)
----------------------------------------
Beam1 Top1 ratio: 1555/5000 (0.3110)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================



========================================
2m
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 197/5000 (0.0394)
Strict-format rate (Top1): 4954/5000 (0.9908)
----------------------------------------
NameExact@1:        72/5000 (0.0144)
CatExact@1:         238/5000 (0.0476)
NameEditSim>=0.90:  73/5000 (0.0146)
NameEditSim>=0.80:  74/5000 (0.0148)
NameJaccard>=0.80:  72/5000 (0.0144)
Distinct@1:         0.3074
Entropy(Top1):      4.5053
----------------------------------------
Top10 Top1 predictions:
01. 1098/5000 (0.2196)  The UPS Store (Shipping and mailing service)
02.  905/5000 (0.1810)  Taco Bell (Fast food restaurant)
03.  350/5000 (0.0700)  El Pollo Loco (Mexican restaurant)
04.  177/5000 (0.0354)  Starbucks (Coffee shop)
05.  155/5000 (0.0310)  Safeway (Grocery store)
06.   89/5000 (0.0178)  McDonald's (Fast food restaurant)
07.   80/5000 (0.0160)  Walmart Supercenter (Department store)
08.   74/5000 (0.0148)  Dollar Tree (Dollar store)
09.   50/5000 (0.0100)  Popeyes Louisiana Kitchen (Chicken restaurant)
10.   31/5000 (0.0062)  Wendy's (Fast food restaurant)
----------------------------------------
Beam1 Top1 ratio: 1098/5000 (0.2196)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================



margin=0.30 + view_dropout=0.15 已经把 CoIN 拉活了（hinge_pos_rate≈0.25 出现了）

数据里的 coin_margin 覆盖了 default_margin（你 JSONL 里可能有 coin_margin 或 hard_level 映射给了较小 margin）。
做法 1（最快）： 重新生成 stage2 数据时别写 coin_margin，或者让 coin_margin 的映射整体上移（hard/hard++ 更大）。


检查确认 stage2 JSONL 是否有 coin_margin 字段（因为它会覆盖 --default_margin）。
python - << 'PY'
import json, itertools
p="./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl"
keys=set()
for i,line in zip(range(2000), open(p,'r',encoding='utf-8')):
    keys |= set(json.loads(line).keys())
print("has coin_margin:", "coin_margin" in keys)
print("has hard_level:", "hard_level" in keys)
PY

===
has hard_level: True
===

这个输出把“--default_margin 为什么没真正生效”彻底解释清楚了：

coin_margin 字段确实没有 ✅
但你 有 hard_level ✅
而你在 preprocess() 里写的是：

hls = examples.get("hard_level", ["hard"] * len(prompts))
out["coin_margin"] = [m_map.get(str(h), float(args.default_margin)) for h in hls]


也就是说：只要 hard_level 存在，绝大多数样本都会走 m_map（m_easy/m_medium/m_hard/m_hardpp），--default_margin 只在“hard_level 不在 map 里”时才会用到。
所以你 coin_debug 里 coin_margin_mean≈0.21~0.23 完全符合：默认 m_hard=0.20、m_hardpp=0.25 的加权平均。

你要让 margin=0.30 真正生效，正确做法是改 m_*，不是改 default_margin

✅ 推荐先改这两个（最关键）

--m_hard 0.30
--m_hardpp 0.35

继续训练：
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_vdrop_run3_lc0p06_v0p10_mhard0p30 \
  --resume \
  --lr 1e-5 \
  --lambda_coin 0.06 \
  --view_dropout 0.10 \
  --m_hard 0.30 \
  --m_hardpp 0.35 \
  --warmup_ratio 0.03 \
  --batch_size 12 --grad_accum 4 \
  --save_steps 1000 --logging_steps 50 \
  --attn_impl flash_attention_2


eval指令：
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run3_lc0p06_v0p10_mhard0p30/checkpoint-14000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10


python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_vdrop_run3_lc0p06_v0p10_mhard0p30/checkpoint-14000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10



你这次 margin 真的生效了（不是在调空气）

证据在 coin_debug：
coin_margin_mean ≈ 0.30~0.325 ✅（说明 m_hard / m_hardpp 已经覆盖到样本）
sim_pos_mean ≈ 0.89~0.91 ✅（view-dropout 有效，已经不再是 1.0）
hinge_pos_rate ≈ 0.17~0.33 ✅（对比项会触发，不是“死的”）
hinge_mean ≈ 0.01~0.03（强度偏温和）
所以：训练目标层面现在是健康的，不是代码没跑、dropout没起作用、margin没用。


如果硬上GRPO：更稳妥的选择反而是你之前那条表现更“均衡”的（例如你贴过的 vdrop_run1 ckpt-12000：stage2 Top1=0.225，Entropy=3.718 更好一些）。




从头训练stage2

在你现在的现象里，最影响 stage2 Oracle 的通常是两件事：
一上来就强 CoIN/大 margin 会把模型推向“保守高频答案”，Oracle 很难涨


Phase-1：只做 NTP 适配（不启用 CoIN）
目的：让模型先适配 stage2 的 prompt / completion 分布，避免一开始就被对比约束拉向高频塌缩。
2k~4k step 级别，够看趋势
stage2 的困难负样本/对比项更像“判别约束”，它不一定直接提升“exact string match”

python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32 \
  --num_epochs 1 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --batch_size 8 --grad_accum 4 \
  --lambda_coin 0.0 \
  --view_dropout 0.0 \
  --save_steps 500 --logging_steps 50 \
  --attn_impl flash_attention_2

======
Phase-2：打开 CoIN（但别太猛，目标是“轻推”Oracle，不是强判别）
为了 Oracle@10，我建议 CoIN 温和：
lambda_coin 不要一上来 0.10（容易把生成推向保守高频），先用 0.04~0.06
view_dropout 保持 0.10
m_hard/m_hardpp 不要比你现在更大（你已经验证 margin 生效后仍塌缩），建议先用 0.25 / 0.30（更温和）

python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32 \
  --resume \
  --num_epochs 1 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --batch_size 8 --grad_accum 4 \
  --lambda_coin 0.06 \
  --view_dropout 0.10 \
  --m_hard 0.25 --m_hardpp 0.30 \
  --save_steps 1000 --logging_steps 50 \
  --attn_impl flash_attention_2


======
eval指令


stage2（800k）
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo/checkpoint-XXXX \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio --report_topk 10
stage1（2m，用来确认没退化太多）
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo/checkpoint-XXXX \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio --report_topk 10




python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-2500 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10

python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-2500 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio \
  --report_topk 10


ckpt2500
========================================
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 162/5000 (0.0324)
Strict-format rate (Top1): 4945/5000 (0.9890)
----------------------------------------
NameExact@1:        54/5000 (0.0108)
CatExact@1:         188/5000 (0.0376)
NameEditSim>=0.90:  55/5000 (0.0110)
NameEditSim>=0.80:  55/5000 (0.0110)
NameJaccard>=0.80:  54/5000 (0.0108)
Distinct@1:         0.1814
Entropy(Top1):      3.8382
----------------------------------------
Top10 Top1 predictions:
01. 1004/5000 (0.2008)  The UPS Store (Shipping and mailing service)
02.  607/5000 (0.1214)  El Pollo Loco (Mexican restaurant)
03.  562/5000 (0.1124)  Taco Bell (Fast food restaurant)
04.  373/5000 (0.0746)  Starbucks (Coffee shop)
05.  272/5000 (0.0544)  Safeway (Grocery store)
06.  257/5000 (0.0514)  H&R Block (Tax preparation service)
07.  182/5000 (0.0364)  Dollar Tree (Dollar store)
08.  102/5000 (0.0204)  Popeyes Louisiana Kitchen (Chicken restaurant)
09.  100/5000 (0.0200)  Wendy's (Fast food restaurant)
10.   81/5000 (0.0162)  Subway (Sandwich shop)
----------------------------------------
Beam1 Top1 ratio: 1004/5000 (0.2008)
Beam1 Top1 sample: The UPS Store (Shipping 



========================================
2M
BEAM ORACLE EVAL (K=10)
========================================
Eval samples:   5000
Oracle@10 hit (exact): 187/5000 (0.0374)
Strict-format rate (Top1): 4956/5000 (0.9912)
----------------------------------------
NameExact@1:        63/5000 (0.0126)
CatExact@1:         212/5000 (0.0424)
NameEditSim>=0.90:  64/5000 (0.0128)
NameEditSim>=0.80:  65/5000 (0.0130)
NameJaccard>=0.80:  63/5000 (0.0126)
Distinct@1:         0.2960
Entropy(Top1):      4.6975
----------------------------------------
Top10 Top1 predictions:
01.  694/5000 (0.1388)  The UPS Store (Shipping and mailing service)
02.  540/5000 (0.1080)  Taco Bell (Fast food restaurant)
03.  486/5000 (0.0972)  El Pollo Loco (Mexican restaurant)
04.  356/5000 (0.0712)  Starbucks (Coffee shop)
05.  243/5000 (0.0486)  Safeway (Grocery store)
06.  235/5000 (0.0470)  H&R Block (Tax preparation service)
07.  183/5000 (0.0366)  Dollar Tree (Dollar store)
08.   97/5000 (0.0194)  Popeyes Louisiana Kitchen (Chicken restaurant)
09.   71/5000 (0.0142)  Wendy's (Fast food restaurant)
10.   58/5000 (0.0116)  McDonald's (Fast food restaurant)
----------------------------------------
Beam1 Top1 ratio: 694/5000 (0.1388)
Beam1 Top1 sample: The UPS Store (Shipping and mailing service)
========================================



下一步
python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-33000 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32 \
  --resume \
  --num_epochs 1 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --batch_size 8 --grad_accum 4 \
  --lambda_coin 0.04 \
  --view_dropout 0.10 \
  --m_hard 0.25 --m_hardpp 0.30 \
  --save_steps 500 --logging_steps 50 \
  --attn_impl flash_attention_2


eval
python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-4500 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio --report_topk 10

python HardMiningSFT/eval_stage1_beam_oracle.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-4500 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --n_eval 5000 --bs 16 --max_new_tokens 48 --num_beams 10 \
  --report_top1_ratio --report_topk 10




python HardMiningSFT/train_stage2_coin_rankmargin.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --stage1_ckpt ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-2500 \
  --data_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --output_dir ./HardMiningSFT/ckpt_stage2_coinweak_from2500 \
  --num_epochs 1 \
  --max_length 1024 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 5e-6 \
  --warmup_ratio 0.02 \
  --save_steps 500 \
  --logging_steps 50 \
  --lambda_coin 0.01 \
  --view_dropout 0.05 \
  --neg_tau 0.55 \
  --attn_impl flash_attention_2 \
  --pin_memory \
  --resume
