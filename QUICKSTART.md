# HierGR-SeqRec Quick Start Guide

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

### Install Dependencies

```bash
cd HierGR-SeqRec
pip install -r requirements.txt
```

## 🚀 Quick Start (5 Steps)

### Step 0: Prepare Data & Models

#### 0.1 Download Yelp Dataset
Download the Yelp Academic Dataset and place files in `data/raw/`:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_tip.json` (optional)

```bash
mkdir -p data/raw
# Place your Yelp dataset files here
```

#### 0.2 Download Pre-trained Language Model
The system uses **Qwen2.5-1.5B-Instruct** by default. Download it to your workspace:

```bash
# Option 1: Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /workspace/Qwen2_5-1.5B-Instruct

# Option 2: Using Python
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/workspace/Qwen2_5-1.5B-Instruct')"
```

**Important**: Update the model path in `config/config.yaml`:
```yaml
llm:
  model_name: "/workspace/Qwen2_5-1.5B-Instruct"  # Update this path
```

### Step 1: Build Item Profiles

Construct rich text descriptions for each business by aggregating:
- Name, Categories, Attributes
- Location (City, State, Postal Code, Address)
- Top reviews and tips

```bash
python data_processing/step1_build_item_profile.py
```

**Output**: `data/processed/item_profiles.jsonl`

### Step 2: Generate Semantic IDs

Train RQ-VAE model to convert business embeddings into hierarchical semantic IDs:

```bash
python data_processing/step2_generate_semantic_ids.py
```

**What happens**:
1. Generates BERT embeddings for item profiles
2. Trains 3-layer RQ-VAE model
3. Assigns unique semantic IDs to each business
4. Resolves ID collisions with suffixes

**Outputs**:
- `data/embeddings/item_embeddings.pt` - BERT embeddings
- `data/rqvae_ckpt/best_collision_model.pth` - Trained RQ-VAE model
- `data/processed/sid_mapping.json` - Business ID to Semantic ID mapping

### Step 3: Build User Sequences

Create user interaction sequences with sliding window:

```bash
python data_processing/step3_build_user_sequences.py
```

**Features**:
- Sliding window (last 15 interactions)
- Long-term preference summary
- K-core filtering (min 5 interactions per user/item)
- Train/Valid/Test split

**Outputs**:
- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_meta.json`

### Step 4: Construct Multi-Task Prompts

Generate training prompts for three tasks:

```bash
python data_processing/step4_construct_prompts.py
```

**Tasks**:
- **Task A (80%)**: Next-item prediction with location context
- **Task B (10%)**: User preference summarization
- **Task C (10%)**: Text ↔ Semantic ID alignment

**Outputs**:
- `data/processed/train_prompts.jsonl`
- `data/processed/valid_prompts.jsonl`
- `data/processed/test_prompts.jsonl`

### Step 5: Train LLM

Fine-tune the language model with LoRA:

```bash
python training/train_llm.py
```

**Training Details**:
- Base Model: Qwen2.5-1.5B-Instruct
- Method: LoRA (r=64, alpha=128)
- Epochs: 3
- Batch Size: 8 × 4 (gradient accumulation)
- Mixed Precision: BF16

**Output**: `data/llm_ckpt/checkpoint-XXXXX/`

## 🎯 Inference & Recommendation

### Generate Recommendations

```bash
python inference/recommend.py \
  --user_history examples/user_history_example.json \
  --location "New York" \
  --top_k 10
```

### Check Model Quality

```bash
python inference/check_sft_quality.py
```

## 🔧 Run Complete Pipeline

Use the automated pipeline runner:

```bash
# Run everything
python run_pipeline.py --step all

# Run only data processing
python run_pipeline.py --step data

# Resume from Step 3
python run_pipeline.py --step data --start_from 3

# Run only LLM training
python run_pipeline.py --step llm
```

## 📊 Evaluation

### Quick Test

```bash
python evaluation/quick_test.py
```

### Full Evaluation

```bash
# Generate test samples
python evaluation/generate_test_samples.py

# Run evaluation
python evaluation/evaluate_model.py

# Compare results
python evaluation/compare_results.py
```

See `evaluation/EVALUATION_GUIDE.md` for detailed metrics.

## 🎨 Visualization

Visualize the learned semantic codebook:

```bash
python visualization/visualize_codebook.py
```

**Output**: `visualization/outputs/` (UMAP plots, cluster analysis)

## ⚙️ Configuration

All settings are in `config/config.yaml`:

### Key Parameters

```yaml
# Data paths
data:
  raw_dir: "/workspace/data/raw"
  processed_dir: "/workspace/data/processed"
  rqvae_ckpt_dir: "/workspace/data/rqvae_ckpt"
  llm_ckpt_dir: "/workspace/data/llm_ckpt"

# RQ-VAE settings
rqvae:
  num_emb_list: [256, 256, 256]  # Codebook sizes
  cluster_levels: 2               # Use first 2 layers as Cluster ID
  epochs: 2000

# LLM settings
llm:
  model_name: "/workspace/Qwen2_5-1.5B-Instruct"
  use_lora: true
  lora_r: 64
  epochs: 3
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: Yelp dataset not found"
**Solution**: Download Yelp dataset and place in `data/raw/`

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `config/config.yaml`:
```yaml
rqvae:
  batch_size: 1024  # Reduce from 2048
llm:
  batch_size: 4     # Reduce from 8
```

### Issue: "Model checkpoint not found"
**Solution**: Ensure Step 2 completed successfully and check `data/rqvae_ckpt/`

### Issue: "Labels not found in training"
**Solution**: This is fixed in the current version. Ensure you're using the latest `train_llm.py`

## 📁 Directory Structure After Setup

```
HierGR-SeqRec/
├── data/
│   ├── raw/                          # Your Yelp dataset
│   │   ├── yelp_academic_dataset_business.json
│   │   └── yelp_academic_dataset_review.json
│   ├── processed/                    # Processed data
│   │   ├── item_profiles.jsonl
│   │   ├── sid_mapping.json
│   │   ├── train.jsonl
│   │   └── train_prompts.jsonl
│   ├── embeddings/                   # BERT embeddings
│   │   └── item_embeddings.pt
│   ├── rqvae_ckpt/                   # RQ-VAE model
│   │   └── best_collision_model.pth
│   └── llm_ckpt/                     # Fine-tuned LLM
│       └── checkpoint-XXXXX/
└── /workspace/Qwen2_5-1.5B-Instruct/ # Base LLM (external)
```

## 🚀 Advanced: GRPO Reinforcement Learning

After SFT training, you can further optimize with GRPO:

```bash
python training/train_grpo.py
```

See `training/GRPO_TRAINING_GUIDE.md` for details.

## 📚 Additional Resources

- **Evaluation Guide**: `evaluation/EVALUATION_GUIDE.md`
- **GRPO Training**: `training/GRPO_TRAINING_GUIDE.md`
- **Visualization**: `visualization/README.md`
- **Main README**: `README.md`

## 💡 Tips

1. **Start Small**: Test with a subset of data first
2. **Monitor GPU**: Use `nvidia-smi` to track memory usage
3. **Checkpoints**: Training auto-saves every 500 steps
4. **Resume Training**: Pipeline automatically resumes from last checkpoint
5. **Experiment**: Adjust `config.yaml` for your dataset size

## 📞 Need Help?

Check the detailed documentation in each module's README or the main project README.

---

**Estimated Time**: 
- Data Processing (Steps 1-4): 2-4 hours (depends on dataset size)
- RQ-VAE Training (Step 2): 4-8 hours
- LLM Training (Step 5): 6-12 hours

**Total**: ~12-24 hours for complete pipeline on Yelp dataset
