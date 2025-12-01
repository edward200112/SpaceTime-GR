"""
LLM Training Script for HierGR-SeqRec

Support for:
- LoRA fine-tuning
- Full fine-tuning
- DeepSpeed (optional)
- Multi-task training (Task A/B/C)
"""

import os
import sys
import yaml
import json
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dataset import prepare_datasets, DataCollatorForPromptDataset


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def load_sid_tokens(config, logger):
    """Load all unique Cluster IDs (SIDs) from sid_mapping.json"""
    data_config = config['data']
    processed_dir = data_config['processed_dir']
    mapping_file = os.path.join(processed_dir, data_config['sid_mapping_file'])
    
    if not os.path.exists(mapping_file):
        logger.warning(f"SID mapping file not found: {mapping_file}")
        return []
    
    logger.info(f"Loading SID tokens from {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        sid_mapping = json.load(f)
    
    # Extract all unique cluster_str tokens
    unique_tokens = set()
    for item_data in sid_mapping.values():
        cluster_str = item_data['cluster_str']  # Format: "<0, 12>"
        unique_tokens.add(cluster_str)
    
    sid_tokens = sorted(list(unique_tokens))
    logger.info(f"Found {len(sid_tokens)} unique SID tokens")
    logger.info(f"Example SID tokens: {sid_tokens[:5]}")
    
    return sid_tokens


def load_model_and_tokenizer(config, logger):
    """Load base model and tokenizer"""
    llm_config = config['llm']
    model_name = llm_config['model_name']
    
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Record original vocab size
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    
    # Load and add SID tokens to vocabulary
    sid_tokens = load_sid_tokens(config, logger)
    if sid_tokens:
        logger.info(f"Adding {len(sid_tokens)} SID tokens to tokenizer...")
        num_added = tokenizer.add_tokens(sid_tokens)
        logger.info(f"Successfully added {num_added} new tokens to vocabulary")
        logger.info(f"New vocabulary size: {len(tokenizer)}")
    else:
        logger.warning("No SID tokens found. Model will use subword tokenization for SIDs.")
    
    # Load model with Flash Attention 2 for acceleration
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if llm_config['bf16'] else torch.float16,
            device_map='auto',
            attn_implementation="flash_attention_2",  # Enable Flash Attention 2
            use_cache=False  # Disable cache for training (required by gradient checkpointing)
        )
        logger.info("Flash Attention 2 enabled successfully")
    except Exception as e:
        logger.warning(f"Flash Attention 2 not available: {e}")
        logger.info("Falling back to standard attention")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if llm_config['bf16'] else torch.float16,
            device_map='auto',
            use_cache=False  # Disable cache for training
        )
    
    # Resize embeddings to match new vocabulary size
    if len(tokenizer) > original_vocab_size:
        logger.info(f"Resizing model embeddings from {original_vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Model embeddings resized successfully")
    
    logger.info(f"Model configuration: {model.config}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, original_vocab_size


def setup_lora(model, config, logger, original_vocab_size=None):
    """Setup LoRA for efficient fine-tuning"""
    llm_config = config['llm']
    
    if not llm_config['use_lora']:
        logger.info("LoRA disabled, using full fine-tuning")
        return model
    
    logger.info("Setting up LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=llm_config['lora_r'],
        lora_alpha=llm_config['lora_alpha'],
        lora_dropout=llm_config['lora_dropout'],
        target_modules=llm_config['target_modules'],
        inference_mode=False,
        modules_to_save=["embed_tokens", "lm_head"]  # 确保 embedding 和 lm_head 可训练
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 如果添加了新 token，确保新的 embedding 参数可训练
    if original_vocab_size is not None:
        embedding_layer = model.get_input_embeddings()
        if hasattr(embedding_layer, 'original_module'):
            embedding_layer = embedding_layer.original_module
        
        current_vocab_size = embedding_layer.weight.shape[0]
        if current_vocab_size > original_vocab_size:
            logger.info(f"Ensuring new token embeddings ({original_vocab_size} -> {current_vocab_size}) are trainable")
            # 新增的 embedding 参数已经通过 modules_to_save 设置为可训练
    
    return model


def setup_training_args(config, output_dir):
    """Setup training arguments"""
    llm_config = config['llm']
    hardware_config = config['hardware']
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=llm_config['epochs'],
        per_device_train_batch_size=llm_config['batch_size'],
        per_device_eval_batch_size=llm_config['batch_size'],
        gradient_accumulation_steps=llm_config['gradient_accumulation_steps'],
        learning_rate=llm_config['lr'],
        weight_decay=llm_config['weight_decay'],
        warmup_ratio=llm_config['warmup_ratio'],
        lr_scheduler_type=llm_config['lr_scheduler'],
        logging_steps=config['logging']['log_interval'],
        save_steps=config['logging']['save_interval'],
        eval_strategy="steps",
        eval_steps=config['logging']['save_interval'],
        save_strategy="steps",  # 按步数保存
        save_total_limit=3,  # 只保留1个最佳checkpoint
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 以eval_loss作为最佳模型的评判标准
        greater_is_better=False,  # eval_loss越小越好
        save_only_model=False,  # 保存完整checkpoint（包含optimizer状态，便于恢复训练）
        fp16=llm_config['fp16'],
        bf16=llm_config['bf16'],
        gradient_checkpointing=llm_config['gradient_checkpointing'],
        optim=llm_config['optimizer'],
        report_to="none",  # Can change to "wandb" if needed
        seed=hardware_config['seed'],
        dataloader_num_workers=hardware_config['num_workers'],
        dataloader_pin_memory=hardware_config['pin_memory'],
        remove_unused_columns=False
    )
    
    # DeepSpeed config (optional)
    if llm_config.get('use_deepspeed', False):
        training_args.deepspeed = llm_config['deepspeed_config']
    
    return training_args


def main():
    # Load config
    config_path = './config/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting LLM training for HierGR-SeqRec")
    
    # Set random seed
    torch.manual_seed(config['hardware']['seed'])
    
    # Load model and tokenizer
    model, tokenizer, original_vocab_size = load_model_and_tokenizer(config, logger)
    
    # Setup LoRA
    model = setup_lora(model, config, logger, original_vocab_size)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, valid_dataset = prepare_datasets(config, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForPromptDataset(
        tokenizer=tokenizer,
        max_length=config['llm']['max_seq_length']
    )
    
    # Training arguments
    output_dir = config['data']['llm_ckpt_dir']
    training_args = setup_training_args(config, output_dir)
    
    # Log training configuration to verify
    logger.info("=" * 50)
    logger.info("Training Configuration:")
    logger.info(f"  - eval_steps: {training_args.eval_steps}")
    logger.info(f"  - save_steps: {training_args.save_steps}")
    logger.info(f"  - logging_steps: {training_args.logging_steps}")
    logger.info(f"  - eval_strategy: {training_args.eval_strategy}")
    logger.info("=" * 50)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Check for existing checkpoint to resume training
    checkpoint_dir = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            # Sort by step number and get the latest
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            checkpoint_dir = os.path.join(output_dir, latest_checkpoint)
            logger.info(f"Found existing checkpoint: {checkpoint_dir}")
            logger.info("Resuming training from checkpoint...")
            logger.info("NOTE: Using CURRENT training args (not checkpoint's old config)")
            logger.info(f"  - Current eval_steps will be: {training_args.eval_steps}")
            
            # CRITICAL: Modify trainer_state.json in checkpoint to use current config
            # This prevents the checkpoint's old config from overriding our new settings
            trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                logger.info(f"Modifying {trainer_state_file} to use current training config...")
                with open(trainer_state_file, 'r', encoding='utf-8') as f:
                    trainer_state = json.load(f)
                
                # Log old values
                old_eval_steps = trainer_state.get('eval_steps', 'N/A')
                old_save_steps = trainer_state.get('save_steps', 'N/A')
                logger.info(f"  - Old eval_steps in checkpoint: {old_eval_steps}")
                logger.info(f"  - Old save_steps in checkpoint: {old_save_steps}")
                
                # Update with current config
                trainer_state['eval_steps'] = training_args.eval_steps
                trainer_state['save_steps'] = training_args.save_steps
                trainer_state['logging_steps'] = training_args.logging_steps
                
                # Save modified state back
                with open(trainer_state_file, 'w', encoding='utf-8') as f:
                    json.dump(trainer_state, f, indent=2)
                
                logger.info(f"  ✓ Updated eval_steps to: {training_args.eval_steps}")
                logger.info(f"  ✓ Updated save_steps to: {training_args.save_steps}")
                logger.info("  ✓ trainer_state.json modified successfully")
        else:
            logger.info("No checkpoint found, starting training from scratch")
    
    # Train (automatically resume if checkpoint exists)
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    
    # Verify actual configuration used during training
    logger.info("=" * 50)
    logger.info("Training completed with configuration:")
    logger.info(f"  - Actual eval_steps used: {trainer.args.eval_steps}")
    logger.info(f"  - Total steps: {trainer.state.global_step}")
    logger.info("=" * 50)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    trainer.save_state()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {output_dir}")


if __name__ == '__main__':
    main()
