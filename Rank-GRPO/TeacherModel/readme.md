check data quality
python TeacherModel/check_hard_negative_quality.py   --sample_n 50000 --batch_size 1024  --k_top 500  --k_gt 200   --k_neg 50   --gap_th 0.05    --dist_km 10 




python ./TeacherModel/train_sasrec.py  --dataset_path ./SASRec_Data/sasrec_dataset.pkl    --output_dir ./SASRec_Data    --batch_size 4096     --lr 1e-4    --num_epochs 50    --num_negs 4    --pop_alpha 0.75    --do_eval    --eval_every 1    --eval_users 5000  --eval_neg 199    --eval_batch_size 256  --pin_memory --num_workers 14


python TeacherModel/train_sasrec.py \
  --dataset_path /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --output_dir /workspace/Rank-GRPO/SASRec_Data \
  --resume_path /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --batch_size 4096 \
  --lr 5e-5 \
  --num_epochs 20 \
  --num_negs 6 \
  --pop_alpha 0.75 \
  --do_fast_eval --fast_eval_every 10 --fast_eval_users 2000 --fast_eval_neg 99 --fast_eval_batch_size 512 --fast_eval_fixed_users \
  --do_strict_eval --strict_eval_every 20 --strict_eval_users 2000 --strict_eval_neg 99 --strict_eval_batch_size 512 \
  --early_stop --early_stop_metric NDCG@10 --early_stop_patience 5 --early_stop_min_delta 0.002 \
  --pin_memory --num_workers 14



python TeacherModel/calc_density_from_sasrec_pkl.py --pkl ./SASRec_Data/sasrec_dataset.pkl 


生成GRPO数据：

重新构建 namecat 映射（你这一步之前是 0，现在一定不再是 0）
python TeacherModel/build_namecat_maps_from_meta.py \
  --csv ./poi_semantic_ids.csv \
  --meta_files \
    /workspace/data/GoogleRAW/meta-California.json.gz \
    /workspace/data/GoogleRAW/meta-New_York.json.gz \
    /workspace/data/GoogleRAW/meta-New_Mexico.json.gz \
    /workspace/data/GoogleRAW/meta-Pennsylvania.json.gz \
  --out_dir ./SASRec_Data \
  --max_ids_per_key 50

映射到 SASRec item_id（推荐先用 unique_only，GRPO 最稳）
python HardMiningGRPO/build_namecat2item_ids.py \
  --pkl ./SASRec_Data/sasrec_dataset.pkl \
  --namecat2gmap ./SASRec_Data/namecat2gmap_ids.json \
  --out ./SASRec_Data/namecat2item_ids_unique.json \
  --unique_only


如果你希望保留多映射（后续做 disambiguation / 采样），再跑：
python HardMiningGRPO/build_namecat2item_ids.py \
  --pkl ./SASRec_Data/sasrec_dataset.pkl \
  --namecat2gmap ./SASRec_Data/namecat2gmap_ids.json \
  --out ./SASRec_Data/namecat2item_ids_disambiguation.json


python - <<'PY'
import json
path = "/workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_disambiguation.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print("type:", type(data))
print("len:", len(data))
for i, (k, v) in enumerate(data.items()):
    print(f"{i}: {k!r} -> {v!r}  (value_type={type(v).__name__})")
    if i >= 4:
        break
PY




python /workspace/Rank-GRPO/TeacherModel/train_sasrec.py \
  --dataset_path /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --output_dir   /workspace/Rank-GRPO/SASRec_Cont \
  --resume_path  /workspace/Rank-GRPO/SASRec_Cont/seed.pth \
  --batch_size 4096 \
  --lr 1e-4 \
  --num_epochs 300 \
  --num_negs 4 --pop_alpha 0.75 \
  --do_fast_eval --fast_eval_every 20 --fast_eval_users 2000 --fast_eval_neg 99 --fast_eval_batch_size 512 \
  --save_every 5 --save_best \
  --early_stop --early_stop_metric NDCG@10 --early_stop_patience 5 --early_stop_min_delta 0.001 \
  --pin_memory --num_workers 14



大batch有什么好处
in-batch negatives + 梯度统计稳定性 + 吞吐。

perm = torch.randperm(B)
inbatch = pos_ids[perm]   # 用别人的 pos 当我的 neg

这意味着：对每个用户、每个位置，你的负样本来自“其他用户真实点击过的 item”（只是对当前用户来说是负的）。这类负样本通常比纯随机负样本更难（harder），训练信号更强。


batch 越大：

同一 step 内出现的“别人的真实点击 item”越多样，分布更接近真实数据分布
更可能抽到与你当前正样本“语义相近/热度相近”的 item（更 hard）
训练更像在做“区分正样本 vs 真实世界里常见的候选”


2) 大 batch 的第二个收益：梯度更稳定（更像“用更大的样本估计期望”）

你的 loss（BCEWithLogitsLoss）在非零位置上做 mean，batch 大了以后：
每步参与 loss 的 token 数更多（大概是 B * 有效序列长度）
梯度方差更小，更新更平滑
通常更不容易出现“某一步梯度特别怪导致指标抖动”
这点对序列推荐也常见：小 batch 容易训练不稳定/波动大，大 batch 往往更稳。


python evaluate_sasrec_metrics.py \
  --dataset_path /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --weights_path /workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth \
  --do_fast --fast_users 2000 --fast_neg 99 --fast_bs 256 \
  --do_strict --strict_users 2000 --strict_neg 99 --strict_bs 1024 \
  --max_len 50 --embed_dim 128 --num_blocks 2 --num_heads 2 --dropout 0.2 \
  --eval_ks 10,50,100




📥 Loading dataset: /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl
✂️ Building strict splits ...
✅ users=4767973  n_items=992862
🏗️ Initializing SASRec ...
🔄 Loading weights: /workspace/Rank-GRPO/SASRec_Cont/sasrec_full_latest.pth
✅ Loaded.

⚡ FAST eval (uniform negatives) ...
[FAST] done in 0.63s
{
  "total": 2000,
  "HR@10": 0.879,
  "NDCG@10": 0.5376149725924332,
  "HR@50": 0.9905,
  "NDCG@50": 0.5640935367994954,
  "HR@100": 1.0,
  "NDCG@100": 0.5656655478127447
}
{
  "total": 2000,
  "HR@10": 0.872,
  "NDCG@10": 0.5218007241245267,
  "HR@50": 0.985,
  "NDCG@50": 0.548443918966576,
  "HR@100": 1.0,
  "NDCG@100": 0.5509070310494065
}

🧪 STRICT eval (popularity negatives) ...
[STRICT] done in 0.01s
{
  "total": 0,
  "HR@10": 0.0,
  "NDCG@10": 0.0,
  "HR@50": 0.0,
  "NDCG@50": 0.0,
  "HR@100": 0.0,
  "NDCG@100": 0.0
}
{
  "total": 0,
  "HR@10": 0.0,
  "NDCG@10": 0.0,
  "HR@50": 0.0,
  "NDCG@50": 0.0,
  "HR@100": 0.0,
  "NDCG@100": 0.0
}

📌 ALL RESULTS:
{
  "fast_valid": {
    "total": 2000,
    "HR@10": 0.879,
    "NDCG@10": 0.5376149725924332,
    "HR@50": 0.9905,
    "NDCG@50": 0.5640935367994954,
    "HR@100": 1.0,
    "NDCG@100": 0.5656655478127447
  },
  "fast_test": {
    "total": 2000,
    "HR@10": 0.872,
    "NDCG@10": 0.5218007241245267,
    "HR@50": 0.985,
    "NDCG@50": 0.548443918966576,
    "HR@100": 1.0,
    "NDCG@100": 0.5509070310494065
  },
  "strict_valid": {
    "total": 0,
    "HR@10": 0.0,
    "NDCG@10": 0.0,
    "HR@50": 0.0,
    "NDCG@50": 0.0,
    "HR@100": 0.0,
    "NDCG@100": 0.0
  },
  "strict_test": {
    "total": 0,
    "HR@10": 0.0,
    "NDCG@10": 0.0,
    "HR@50": 0.0,
    "NDCG@50": 0.0,
    "HR@100": 0.0,
    "NDCG@100": 0.0
  }
}