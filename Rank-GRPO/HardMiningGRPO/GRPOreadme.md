指令如下：
你的 stage2 起点 LoRA：./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-4500
GRPO 数据：./HardMiningSFT/sft_data/grpo_train.jsonl
SASRec：./SASRec/sasrec_ckpt.pt + ./SASRec/sasrec_config.json
映射：./SASRec/poi_text2id.json


python HardMiningSFT/train_grpo_sasrec.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --init_lora ./HardMiningSFT/ckpt_stage2_retrain_for_grpo_bs32/checkpoint-4500 \
  --data_jsonl ./HardMiningSFT/sft_data/grpo_train.jsonl \
  --sasrec_ckpt ./SASRec/sasrec_ckpt.pt \
  --sasrec_config ./SASRec/sasrec_config.json \
  --poi_text2id ./SASRec/poi_text2id.json \
  --output_dir ./HardMiningSFT/ckpt_grpo_sasrec_run1 \
  --prompt_batch_size 2 \
  --group_size 8 \
  --max_new_tokens 48 \
  --temperature 0.9 \
  --top_p 0.9 \
  --alpha_sasrec 0.10 \
  --beta_kl 0.05 \
  --lr 2e-6 \
  --max_steps 2000 \
  --grad_accum 1 \
  --save_steps 200 \
  --log_steps 20 \
  --attn_impl flash_attention_2



A. 检查 SFT 命中率
B. 把 SFT jsonl 转成 GRPO jsonl
C. SASRec reward 模块（供 train_grpo.py 调用）


python HardMiningGRPO/check_namecat_hit_from_sft.py \
  --sft_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --namecat2item_unique ./SASRec_Data/namecat2item_ids_unique.json \
  --field completion --n 200000

python HardMiningGRPO/check_namecat_hit_from_sft.py \
  --sft_jsonl ./HardMiningSFT/sft_data/google_stage1_pos_2m.jsonl \
  --namecat2item_unique ./SASRec_Data/namecat2item_ids_unique.json \
  --field completion --n 200000


python HardMiningGRPO/make_grpo_data_from_sft.py \
  --in_jsonl ./HardMiningSFT/sft_data/google_stage2_coin_800k_ips_rule_hard_strongerIPS.jsonl \
  --out_jsonl ./HardMiningSFT/grpo_data/grpo_stage2_from_sft.jsonl \
  --namecat2item_unique ./SASRec_Data/namecat2item_ids_unique.json \
  --max_hist 50

DEBUG版本”
python HardMiningGRPO/train_grpo.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v1/grpo_train.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v1/grpo_val.jsonl \
  --namecat2item_disamb ./SASRec_Data/namecat2item_ids_disambiguation.json \
  --sasrec_pkl ./SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data_new/sasrec_full_latest.pt \
  --output_dir ./HardMiningGRPO/ckpt_grpo_v1_debug_old \
  --per_device_bs 16 --grad_accum 2 \
  --lr 5e-6 --num_generations 12 \
  --alpha 0.3 --n_neg_sample 256 --format_bonus 0.05 --softmax_temp 1.0 \
  --sasrec_max_len 50 --sasrec_embed_dim 128 --sasrec_num_blocks 2 --sasrec_num_heads 2 --sasrec_dropout 0.2 \
  --debug_log_every_steps 20 \
  --debug_num_show 5 \
  --debug_dump_jsonl ./HardMiningGRPO/ckpt_grpo_v1_debug_old/debug_samples.jsonl\
  --debug_print_full_completion



训练指令
python HardMiningGRPO/train_grpo.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v1/grpo_train.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v1/grpo_val.jsonl \
  --namecat2item_disamb /workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_disambiguation.json \
  --name2item_disamb /workspace/Rank-GRPO/SASRec_Data/name2item_ids_disambiguation.json \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir ./HardMiningGRPO/ckpt_grpo_v4_canon_namefallback \
  --per_device_bs 16 --grad_accum 2 \
  --lr 5e-6 --num_generations 8 \
  --alpha 0.3 --n_neg_sample 256 --format_bonus 0.05 --softmax_temp 1.0 \
  --extra_text_penalty 0.05 --unknown_penalty 0.05 \
  --max_new_tokens 12 \
  --sasrec_max_len 50 --sasrec_embed_dim 128 --sasrec_num_blocks 2 --sasrec_num_heads 2 --sasrec_dropout 0.2 \
  --debug_log_every_steps 20 --debug_num_show 5 \
  --debug_dump_jsonl ./HardMiningGRPO/ckpt_grpo_v4_canon_namefallback/debug_samples.jsonl


全量统计：有多少条 prompt 含“候选/只能从下面选/选项”等关键词： 
python - <<'PY'
import json, re
path="./HardMiningGRPO/grpo_data_v1/grpo_train.filtered.jsonl"

pat = re.compile(r"(候选|选项|只能从|请从以下|candidates|options|choose from)", re.IGNORECASE)

tot = 0
hit = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        tot += 1
        p = json.loads(line).get("prompt","")
        if pat.search(p):
            hit += 1

print("total:", tot)
print("has_candidate_like_prompt:", hit, "rate:", hit / tot if tot else 0)
PY

===
total: 195966
has_candidate_like_prompt: 26 rate: 0.00013267607646224344
===





抽样打印：如果命中，把那条 prompt 尾部 500 字打印出来看看是不是候选列表:
python - <<'PY'
import json, re
path="./HardMiningGRPO/grpo_data_v1/grpo_train.filtered.jsonl"

pat = re.compile(r"(候选|选项|只能从|请从以下|candidates|options|choose from)", re.IGNORECASE)

shown = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        p = o.get("prompt","")
        if pat.search(p):
            print("="*80)
            print("target:", o.get("target_namecat"))
            print("prompt(tail 500 chars):")
            print(p[-500:].replace("\n","\\n"))
            shown += 1
            if shown >= 5:
                break
print("shown:", shown)
PY



================================================================================
target: Royal Coach Diner (Diner)
prompt(tail 500 chars):
lamo Drafthouse Cinema Yonkers (Movie theater)\n9. New York Aquarium (Aquarium)\n10. IHOP (Restaurant)\n11. Applebee's Grill + Bar (Restaurant)\n12. Yonkers Gateway Center (Shopping mall)\n13. T Swirl Crepe (Crêperie)\n14. Duck Donuts (Donut shop)\n15. Hampton Inn & Suites Newburgh Stewart Airport (Hotel)\n16. Rockefeller Park (Park)\n17. Columbus Diner (Diner)\n18. Creative Auto Options (Auto parts store)\n19. McDonald's (Fast food restaurant)\n20. The Home Depot (Home improvement store)\n只输出一个地点名(类别)，不要解释\n
================================================================================
target: Northeast Regional Library (Public library)
prompt(tail 500 chars):
(Shopping mall)\n8. The Fresh Grocer of LaSalle (Grocery store)\n9. Neshaminy Mall (Shopping mall)\n10. AMC Philadelphia Mills 14 (Movie theater)\n11. Pennypack Park (City park)\n12. 2300 Arena (Event venue)\n13. Eye Options (Eye care center)\n14. Old Navy (Clothing store)\n15. ROOSEVELT PLAZA (Shopping mall)\n16. Dollar General (Dollar store)\n17. Greyhound: Bus Station (Bus company)\n18. Dollar Tree (Dollar store)\n19. Great Northeast Plaza (Shopping mall)\n20. Save A Lot (Grocery store)\n只输出一个地点名(类别)，不要解释\n
================================================================================
target: Ready Coffee (Coffee stand)
prompt(tail 500 chars):
你将看到用户最近访问的地点列表（按时间从旧到新），请预测用户下一次最可能去的一个地点。\n历史：\n1. Dutchess Animal Clinic (Veterinarian)\n2. Mid Hudson Subaru (Subaru dealer)\n3. County Fare (American restaurant)\n4. Meadowbrook Farm (Produce market)\n5. Village Creamery (Ice cream shop)\n6. SHORTHillS - Restaurant & Diner (Restaurant)\n7. Broad Options The Jewelry Store (Jewelry store)\n只输出一个地点名(类别)，不要解释\n
================================================================================
target: Mariner Finance (Loan agency)
prompt(tail 500 chars):
9. McDonald's (Fast food restaurant)\n10. Crab Shack II (Seafood restaurant)\n11. Reen's Delicatessen (Deli)\n12. Millevoi's Tire & Automotive Center, Bensalem (Auto repair shop)\n13. Live! Casino & Hotel Philadelphia (Hotel)\n14. Advance Auto Parts (Auto parts store)\n15. Ron's Caribbean Cafe (Jamaican restaurant)\n16. Aston Village Nails (Nail salon)\n17. Parx Casino (Casino)\n18. IHOP (Restaurant)\n19. Colonial Nissan Inc (Nissan dealer)\n20. Bella Maria Tomato Pies (Pizza restaurant)\n只输出一个地点名(类别)，不要解释\n
================================================================================
target: Burger King (Restaurant)
prompt(tail 500 chars):
op)\n8. McDonald's (Fast food restaurant)\n9. Newport Diagnostic Center (Diagnostic center)\n10. Edison Park (Park)\n11. Taco Bell (Fast food restaurant)\n12. McDonald's (Fast food restaurant)\n13. Irvine Spectrum Center (Shopping mall)\n14. BoxLunch (Gift shop)\n15. Village Market Los Olivos (Grocery store)\n16. Irvine Subaru (Subaru dealer)\n17. Dunkin' (Coffee shop)\n18. VCA Irvine Boulevard Animal Hospital (Animal hospital)\n19. Petco (Pet supply store)\n20. Sootha Coffee (Coffee shop)\n只输出一个地点名(类别)，不要解释\n
shown: 5



python HardMiningGRPO/train_grpo.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v1/grpo_train.filtered.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v1/grpo_val.filtered.jsonl \
  --namecat2item_disamb /workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_disambiguation.json \
  --name2item_disamb /workspace/Rank-GRPO/SASRec_Data/name2item_ids_disambiguation.json \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir ./HardMiningGRPO/ckpt_grpo_v4_canon_namefallback \
  --per_device_bs 16 --grad_accum 2 \
  --lr 5e-6 --num_generations 8 \
  --alpha 0.3 --n_neg_sample 256 --format_bonus 0.05 --softmax_temp 1.0 \
  --extra_text_penalty 0.10 --unknown_penalty 0.05 \
  --max_new_tokens 10 \
  --sasrec_max_len 50 --sasrec_embed_dim 128 --sasrec_num_blocks 2 --sasrec_num_heads 2 --sasrec_dropout 0.2 \
  --debug_log_every_steps 20 --debug_num_show 5 \
  --debug_dump_jsonl ./HardMiningGRPO/ckpt_grpo_v4_canon_namefallback/debug_samples.jsonl



你目前训练里“生成空间没收紧”的主要原因是两类：

prompt 没用 Qwen 的 chat template 包起来（跟 stage2 不一致），模型更容易把历史列表当成“可继续写的文本”，于是出现 Human:/乱尾巴。
reward 解析太严格（要求整行完全匹配 Name (Cat)），一旦同一行后面跟了尾巴（比如 Pingpong Coin Op (Restaurant) 24hr Lock），就会解析失败，resolved_rate 直接掉。


下面给你一套“想清楚再改”的 完整版本（train + reward），核心改动：
train：把 raw prompt 用 tokenizer.apply_chat_template(..., add_generation_prompt=True) 包成 Qwen 格式（等价于你 stage2 的 <|im_start|>user ... <|im_start|>assistant\n），并把 eos_token_id 设好，尽量让生成在 <|im_end|> 处停下。
reward：改成“在第一行里搜索第一个 Name (Cat) 子串”，并对尾巴/多行做惩罚；同时支持：
  namecat2item_ids_disambiguation.json（list）
  name2item_ids_disambiguation.json（name-only fallback，用 SASRec disambiguate



⚠️ candidate_like=22 基本是误报

你之前命中的样例里有店名：Creative Auto Options
你的关键词统计里包含 options，所以把店名里的 Options 当成“候选/选项”命中了。
结论：你的数据 prompt 并没有候选列表（仍然是自由生成）。

✅ bad_target_map=73265 (37.38%) 是“用 unique 字典检查”导致的

你检查脚本用的是：
namecat2item_ids_unique.json（只包含全局唯一的 Name(Cat)）
而像：
Best Buy (Electronics store)
Safeway (Grocery store)
Burger King (Restaurant)
这类连锁店在全美会出现很多次，天然是 ambiguous，所以不会出现在 unique.json 里——于是你的脚本会得到 mapped: None，并计入 bad_target_map。
所以这个 37% 不能说明你的 grpo_train 有错，只说明：
你的 target_namecat 里有大量“非唯一地点名”，这在真实数据里是正常现象。


检查逻辑应当是：

target_namecat 必须存在于 disamb dict
target_item_id 必须在该 key 对应的候选 list 里
===
python - <<'PY'
import json, random

data_path = "./HardMiningGRPO/grpo_data_v1/grpo_train.jsonl"  # 或 filtered
disamb_path = "/workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_disambiguation.json"

with open(disamb_path,"r",encoding="utf-8") as f:
    dis = json.load(f)

total = 0
missing_key = 0
not_in_list = 0
ambiguous = 0

miss_samples=[]
notin_samples=[]

with open(data_path,"r",encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        total += 1
        key = o.get("target_namecat","")
        tgt = o.get("target_item_id", None)

        cands = dis.get(key)
        if cands is None:
            missing_key += 1
            if len(miss_samples) < 5:
                miss_samples.append((key, tgt))
            continue

        if isinstance(cands, list) and len(cands) > 1:
            ambiguous += 1

        if tgt is not None and int(tgt) not in set(map(int, cands)):
            not_in_list += 1
            if len(notin_samples) < 5:
                notin_samples.append((key, tgt, cands[:10], len(cands)))

print("total:", total)
print("missing_key_in_disamb:", missing_key, "rate:", missing_key/total)
print("target_not_in_disamb_list:", not_in_list, "rate:", not_in_list/total)
print("ambiguous_keys:", ambiguous, "rate:", ambiguous/total)

if miss_samples:
    print("\n[missing_key samples]")
    for s in miss_samples:
        print("-", s)

if notin_samples:
    print("\n[target_not_in_list samples]")
    for s in notin_samples:
        print("-", s[0], "tgt=", s[1], "cand_len=", s[3], "cand_head=", s[2])
PY



# ✅ 所以最推荐的训练策略仍然是：
用 disamb(top50) + ensure_target_in_candidates=True + namecat match 主导 + item bonus。
全量 mapping 更适合 推理/离线评估（没有 target 时，为了解决 disamb 截断的召回问题）。
你真正需要的“全量 namecat→item_ids”，可以自己从你已有的文件无损构建
你已经确认了：sasrec_dataset.pkl 里
item2id: gmap_id → item_id（非常关键）
gmap_id2namecat.json: gmap_id → "Name (Cat)"
所以你可以把它们 join 一下，得到真正的全量 mapping（同时还能顺手生成 name-only mapping）
# namecat2item_ids_all.json
saved: /workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_all.json keys: 769907 missing_gmap2namecat: 4406
saved: /workspace/Rank-GRPO/SASRec_Data/name2item_ids_all.json keys: 741778





# 最终GRPO（无地理信息）
python HardMiningGRPO/train_grpo.py \
  --base_model /workspace/Qwen2_5-1.5B-Instruct \
  --adapter ./HardMiningSFT/ckpt_stage2_coinweak_from2500/checkpoint-17500 \
  --train_jsonl ./HardMiningGRPO/grpo_data_v1/grpo_train.filtered.jsonl \
  --eval_jsonl  ./HardMiningGRPO/grpo_data_v1/grpo_val.filtered.jsonl \
  --namecat2item_disamb /workspace/Rank-GRPO/SASRec_Data/namecat2item_ids_disambiguation.json \
  --name2item_disamb /workspace/Rank-GRPO/SASRec_Data/name2item_ids_disambiguation.json \
  --gmap_id2namecat /workspace/Rank-GRPO/SASRec_Data/gmap_id2namecat.json \
  --sasrec_pkl /workspace/Rank-GRPO/SASRec_Data/sasrec_dataset.pkl \
  --sasrec_ckpt /workspace/Rank-GRPO/SASRec_Data/sasrec_full_latest.pt \
  --output_dir ./HardMiningGRPO/ckpt_grpo_final_namecat_main \
  --per_device_bs 16 --grad_accum 2 \
  --num_generations 8 \
  --lr 5e-6 \
  --alpha 0.3 --softmax_temp 1.0 --n_neg_sample 256 \
  --format_bonus 0.05 --item_match_bonus 0.2 \
  --extra_text_penalty 0.05 --unknown_penalty 0.05 \
  --ensure_target_in_candidates --max_disamb_candidates 64 \
  --max_new_tokens 12 \
  --use_chat_template \
  --debug_log_every_steps 20 --debug_num_show 5 \
  --debug_dump_jsonl ./HardMiningGRPO/ckpt_grpo_final_namecat_main/debug_samples.jsonl

===
match_namecat_rate 逐步升高（这是主信号）
unknown_rate 降低（resolver 找不到候选变少）
extra_text_rate 降低（输出变干净）
resolved_rate 稳定上升（因为有 target 注入 + disamb）
===





