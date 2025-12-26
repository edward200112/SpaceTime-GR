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
  --adapter ./HardMiningSFT/ckpt_stage1_continue_from_lora_only/checkpoint-13000 \
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




