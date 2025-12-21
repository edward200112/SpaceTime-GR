# 文件分类方案

## HierGR（生成式模型）

### 训练脚本
- training/train_sft_final.py ✅
- training/train_sft_optimized.py ✅
- training/train_llm.py ✅
- training/train_grpo.py ✅
- training/train_grpo_v2.py ✅
- training/train_grpo_v3.py ✅
- training/train_grpo_v4.py ✅
- training/train_grpo_v4_1.py ✅
- training/train_grpo_v4_1_resume.py ✅
- training/train_grpo_v5.py ✅
- training/merge_model.py ✅
- training/constrained_logits_processor.py ✅
- training/grpo_rewards.py ✅
- training/grpo_rewards_v3.py ✅
- training/grpo_rewards_optimized.py ✅
- training/GRPO_TRAINING_GUIDE.md ✅

### 推理/评估脚本
- inference/check_sft_quality.py ✅
- inference/check_sft_only.py ✅
- inference/validate_grpo_with_tsne.py ✅
- inference/evaluate_final.py ✅
- inference/evaluate_final_v8.py ✅
- inference/evaluate_final_v9.py ✅
- inference/evaluate_final_extended.py ✅
- inference/evaluate_sft_final.py ✅
- inference/evaluate_sft_optimized.py ✅
- inference/demo_inference.py ✅
- inference/trie_utils.py ✅
- inference/recommend.py ✅（如果是生成式的）

### 数据处理
- data_processing/step4_construct_prompts.py ✅（生成 prompt 格式）

---

## PinRec（判别式模型）

### 模型定义
- models/pinrec_ultimate.py ✅
- models/pinrec_ultimate_v2.py ✅
- models/pinrec_llm.py ✅

### 训练脚本
- training/train_ultimate.py ✅
- training/train_ultimate_v2.py ✅
- training/train_ultimate_v2_logq.py ✅
- training/train_ultimate_v4_stable.py ✅
- training/train_pinrec_v6.py ✅
- training/train_pinrec_v7_final.py ✅
- training/train_pinrec_sft_final.py ✅
- training/train_pinrec_grpo_final.py ✅

### 推理/评估脚本
- inference/evaluate_ultimate.py ✅
- inference/evaluate_ultimate_v2.py ✅
- inference/evaluate_pinrec_v3.py ✅
- inference/evaluate_pinrec_v5.py ✅
- inference/evaluate_pinrec_v6.py ✅
- inference/evaluate_pinrec_v7_debug.py ✅
- inference/eval_grpo_aggresive.py ✅
- inference/evaluate_bulletproof.py ✅
- inference/debug_eval_v2.py ✅

### 数据处理
- data_processing/step6_build_ultimate_data.py ✅
- data_processing/step6_build_ultimate_data_v2.py ✅
- data_processing/balance_sequences_for_pinrec.py ✅
- data_processing/balance_ultimate_data.py ✅

---

## 共享文件（保留在根目录）

### RQ-VAE（两个模型都用）
- RQ-VAE/ ✅（整个目录）

### 数据处理（共享）
- data_processing/step1_build_item_profile.py ✅
- data_processing/step2_generate_semantic_ids.py ✅
- data_processing/step3_build_user_sequences.py ✅
- data_processing/balance_dataset.py ✅
- data_processing/analyze_chain_stores.py ✅
- data_processing/check_data_v2.py ✅
- data_processing/inspect_sft_data.py ✅

### 评估工具（共享）
- evaluation/ ✅（整个目录）
- compare_models_unified.py ✅（对比两个模型）

### 可视化（共享）
- visualization/ ✅（整个目录）

### 配置和文档
- config/ ✅
- examples/ ✅
- README.md ✅
- QUICKSTART.md ✅
- MODEL_PATHS.md ✅
- requirements.txt ✅
- run_pipeline.py ✅
- inspect_data.py ✅

### 数据目录
- data/ ✅（保留在根目录，两个模型共享）

---

## 推理/评估脚本（需要进一步判断）
- inference/new_evaluate.py ❓（需要查看内容）
- inference/evaluate_v3.py ❓
- inference/evaluate_metrics.py ❓
- inference/analyze_errors.py ❓
- inference/check_cluster_purity.py ❓
