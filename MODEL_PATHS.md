# Model Download Locations - HierGR-SeqRec

## 📦 Model Storage Overview

This document explains where all models are downloaded and stored in the HierGR-SeqRec project.

## 🤖 Pre-trained Language Model (LLM)

### Default Model: Qwen2.5-1.5B-Instruct

**Download Location**: `/workspace/Qwen2_5-1.5B-Instruct/`

**Configured in**: `config/config.yaml`
```yaml
llm:
  model_name: "/workspace/Qwen2_5-1.5B-Instruct"
```

**How to Download**:

```bash
# Method 1: Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /workspace/Qwen2_5-1.5B-Instruct

# Method 2: Python script
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/workspace/Qwen2_5-1.5B-Instruct'); \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/workspace/Qwen2_5-1.5B-Instruct')"

# Method 3: Auto-download (happens automatically when training starts)
# The model will be downloaded to the path specified in config.yaml
```

**Used in**:
- `training/train_llm.py` - Line 52: `model_name = self.llm_conf['model_name']`
- `training/train_grpo.py` - For reinforcement learning
- `inference/recommend.py` - For generating recommendations

**Alternative Models**:
You can use other models by updating `config.yaml`:
- `Qwen/Qwen2.5-3B-Instruct` (larger, better performance)
- `meta-llama/Llama-2-7b-chat-hf` (requires access token)
- Any Hugging Face causal LM model

## 🧠 BERT Embedding Model

### Model: sentence-transformers/all-MiniLM-L6-v2

**Download Location**: Automatically cached by `sentence-transformers` library

**Default Cache**: `~/.cache/torch/sentence_transformers/`

**Used in**: `data_processing/step2_generate_semantic_ids.py`

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# Auto-downloads to cache on first use
```

**Output Embeddings Saved to**: `data/embeddings/item_embeddings.pt`

**Configured in**: `config/config.yaml`
```yaml
data:
  embeddings_dir: "/workspace/data/embeddings"
  item_embeddings_file: "item_embeddings.pt"
```

## 🔢 RQ-VAE Model (Trained by You)

### Residual Quantized Variational AutoEncoder

**Training Location**: `data_processing/step2_generate_semantic_ids.py`

**Saved to**: `data/rqvae_ckpt/best_collision_model.pth`

**Configured in**: `config/config.yaml`
```yaml
data:
  rqvae_ckpt_dir: "/workspace/data/rqvae_ckpt"
```

**Model Architecture**:
- Input: 770-dim (768 BERT + 2 geo coordinates)
- Layers: [512, 256, 128, 64]
- Codebooks: 3 layers × 256 codes each
- Output: 3-level semantic IDs

**Checkpoint Contents**:
```python
{
    'state_dict': model.state_dict(),
    'epoch': epoch,
    'collision_rate': collision_rate,
    'config': {...}
}
```

**Loading Example** (from `step2_generate_semantic_ids.py`):
```python
checkpoint = torch.load('data/rqvae_ckpt/best_collision_model.pth')
model.load_state_dict(checkpoint['state_dict'])
```

## 🎓 Fine-tuned LLM (Trained by You)

### LoRA-adapted Qwen Model

**Training Script**: `training/train_llm.py`

**Saved to**: `data/llm_ckpt/checkpoint-XXXXX/`

**Configured in**: `config/config.yaml`
```yaml
data:
  llm_ckpt_dir: "/workspace/data/llm_ckpt/checkpoint-14500"
```

**Checkpoint Structure**:
```
data/llm_ckpt/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-14500/        # Final checkpoint
│   ├── adapter_config.json  # LoRA configuration
│   ├── adapter_model.bin    # LoRA weights
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── tokenizer.json
└── training_args.bin
```

**What's Saved**:
- LoRA adapter weights (not full model)
- Tokenizer configuration
- Training arguments
- Optimizer state (if resuming)

**Loading for Inference**:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("/workspace/Qwen2_5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "data/llm_ckpt/checkpoint-14500")
```

## 🎯 GRPO Reinforcement Learning Model (Optional)

### Group Relative Policy Optimization

**Training Script**: `training/train_grpo.py`

**Saved to**: `data/grpo_checkpoints/`

**Configured in**: `config/config.yaml`
```yaml
grpo:
  use_rl: true
  learning_rate: 1.0e-6
```

**Checkpoint Location**: `data/grpo_checkpoints/best_model/`

## 📊 Complete Directory Structure

```
HierGR-SeqRec/
├── data/
│   ├── embeddings/
│   │   └── item_embeddings.pt              # BERT embeddings (Step 2)
│   ├── rqvae_ckpt/
│   │   └── best_collision_model.pth        # Trained RQ-VAE (Step 2)
│   ├── llm_ckpt/
│   │   └── checkpoint-XXXXX/               # Fine-tuned LLM (Step 5)
│   │       ├── adapter_config.json
│   │       └── adapter_model.bin
│   └── grpo_checkpoints/
│       └── best_model/                     # RL-optimized model (Optional)
│
└── /workspace/Qwen2_5-1.5B-Instruct/       # Base LLM (External)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 🔄 Model Flow

```
1. Download Base LLM
   ↓
   /workspace/Qwen2_5-1.5B-Instruct/

2. Generate BERT Embeddings (Step 2)
   ↓
   data/embeddings/item_embeddings.pt

3. Train RQ-VAE (Step 2)
   ↓
   data/rqvae_ckpt/best_collision_model.pth

4. Fine-tune LLM with LoRA (Step 5)
   ↓
   data/llm_ckpt/checkpoint-XXXXX/

5. (Optional) GRPO RL Training
   ↓
   data/grpo_checkpoints/best_model/
```

## 💾 Storage Requirements

| Model | Size | Location |
|-------|------|----------|
| Base LLM (Qwen2.5-1.5B) | ~3 GB | `/workspace/Qwen2_5-1.5B-Instruct/` |
| BERT Embeddings | ~500 MB | `data/embeddings/` |
| RQ-VAE Model | ~10 MB | `data/rqvae_ckpt/` |
| LoRA Adapters | ~100 MB | `data/llm_ckpt/` |
| GRPO Checkpoints | ~100 MB | `data/grpo_checkpoints/` |
| **Total** | **~4 GB** | |

## 🔧 Changing Model Paths

To use different paths, update `config/config.yaml`:

```yaml
data:
  embeddings_dir: "/your/custom/path/embeddings"
  rqvae_ckpt_dir: "/your/custom/path/rqvae_ckpt"
  llm_ckpt_dir: "/your/custom/path/llm_ckpt"

llm:
  model_name: "/your/custom/path/base_model"
```

## 🚨 Important Notes

1. **Base LLM is NOT trained**: Only LoRA adapters are trained, saving disk space
2. **BERT model auto-downloads**: First run of Step 2 will download BERT
3. **Checkpoints auto-save**: Training saves every 500 steps
4. **Resume training**: Pipeline automatically finds latest checkpoint
5. **Disk space**: Ensure 50GB+ free space for full pipeline

## 📝 Model Loading Code Reference

### In `step2_generate_semantic_ids.py` (Lines 60-100)
```python
# Load BERT embeddings
emb_file = os.path.join(
    self.data_config['embeddings_dir'],
    self.data_config['item_embeddings_file']
)
data = torch.load(emb_file, weights_only=False)

# Load RQ-VAE checkpoint
ckpt_path = os.path.join(
    self.data_config['rqvae_ckpt_dir'], 
    'best_collision_model.pth'
)
checkpoint = torch.load(ckpt_path, map_location=self.device)
model.load_state_dict(checkpoint['state_dict'])
```

### In `train_llm.py` (Lines 40-60)
```python
# Load base model
model_name = self.llm_conf['model_name']  # From config.yaml
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,  # "/workspace/Qwen2_5-1.5B-Instruct"
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Save fine-tuned model
trainer.save_model(self.output_dir)  # data/llm_ckpt/
```

## ✅ Verification

Check if models are properly downloaded:

```bash
# Check base LLM
ls -lh /workspace/Qwen2_5-1.5B-Instruct/

# Check BERT embeddings
ls -lh data/embeddings/item_embeddings.pt

# Check RQ-VAE
ls -lh data/rqvae_ckpt/best_collision_model.pth

# Check fine-tuned LLM
ls -lh data/llm_ckpt/checkpoint-*/
```

---

**Summary**: 
- **Base LLM**: `/workspace/Qwen2_5-1.5B-Instruct/` (download manually or auto)
- **BERT**: Auto-cached by `sentence-transformers`
- **RQ-VAE**: `data/rqvae_ckpt/` (trained in Step 2)
- **Fine-tuned LLM**: `data/llm_ckpt/` (trained in Step 5)
