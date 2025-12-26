check data quality
python TeacherModel/check_hard_negative_quality.py   --sample_n 50000 --batch_size 1024  --k_top 500  --k_gt 200   --k_neg 50   --gap_th 0.05    --dist_km 10 




python ./TeacherModel/train_sasrec.py  --dataset_path ./SASRec_Data/sasrec_dataset.pkl    --output_dir ./SASRec_Data    --batch_size 4096     --lr 1e-4    --num_epochs 50    --num_negs 4    --pop_alpha 0.75    --do_eval    --eval_every 1    --eval_users 5000  --eval_neg 199    --eval_batch_size 256  --pin_memory --num_workers 14



python TeacherModel/calc_density_from_sasrec_pkl.py --pkl ./SASRec_Data/sasrec_dataset.pkl 