check data quality
python TeacherModel/check_hard_negative_quality.py   --sample_n 50000 --batch_size 1024  --k_top 500  --k_gt 200   --k_neg 50   --gap_th 0.05    --dist_km 10 




python ./TeacherModel/train_sasrec.py  --dataset_path ./SASRec_Data/sasrec_dataset.pkl    --output_dir ./SASRec_Data    --batch_size 4096     --lr 1e-4    --num_epochs 50    --num_negs 4    --pop_alpha 0.75    --do_eval    --eval_every 1    --eval_users 5000  --eval_neg 199    --eval_batch_size 256  --pin_memory --num_workers 14



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
