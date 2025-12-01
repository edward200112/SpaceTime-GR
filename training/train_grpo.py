"""
GRPO Training Script for HierGR-SeqRec

使用 Group Relative Policy Optimization 进行强化学习训练
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
from datasets import Dataset
from transformers import set_seed

from grpo_trainer import HierGRSeqRecGRPOTrainer
from reward_functions import create_reward_function
from trl import GRPOConfig


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_sid_mapping(config):
    """加载 SID mapping"""
    mapping_file = os.path.join(
        config['data']['processed_dir'],
        config['data']['sid_mapping_file']
    )
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataset(data_file: str, prompt_format: str = "task_a"):
    """
    准备训练数据
    
    Args:
        data_file: 数据文件路径
        prompt_format: prompt格式（task_a, task_b, task_c）
    
    Returns:
        dataset: HuggingFace Dataset对象
        prompt2target: prompt到target的映射
    """
    # 加载数据（支持 .json 和 .jsonl 格式）
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        if data_file.endswith('.jsonl'):
            # JSONL 格式：每行一个 JSON 对象
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            # JSON 格式：整个文件是一个 JSON 数组
            data = json.load(f)
    
    # 构建prompt2target映射
    prompt2target = {}
    prompts = []
    
    for sample in data:
        # 兼容两种数据格式
        if 'prompt' in sample:
            # 格式1: 直接包含 prompt 和 target_cluster_str
            prompt = sample['prompt']
            target_cluster_str = sample.get('target_cluster_str', '')
        else:
            # 格式2: 包含 instruction 和 output
            prompt = sample['instruction']
            target_cluster_str = sample.get('output', '')
        
        # 只保留推荐任务 (task_a_recommendation)
        task = sample.get('task', '')
        if task and 'recommendation' not in task:
            continue
        
        prompts.append(prompt)
        prompt2target[prompt] = target_cluster_str
    
    # 创建Dataset
    dataset = Dataset.from_dict({"prompt": prompts})
    
    return dataset, prompt2target


def main():
    parser = argparse.ArgumentParser(description='GRPO Training for HierGR-SeqRec')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='配置文件路径')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--eval_data', type=str, help='验证数据路径（可选）')
    parser.add_argument('--sft_model', type=str, required=True, help='SFT模型路径（GRPO的初始模型）')
    parser.add_argument('--output_dir', type=str, default='./data/grpo_checkpoints', help='输出目录')
    
    # GRPO特定参数
    parser.add_argument('--num_generations', type=int, default=8, help='每个prompt生成的数量')
    parser.add_argument('--beta', type=float, default=0.04, help='KL散度系数')
    parser.add_argument('--reward_type', type=str, default='rule', choices=['rule', 'ndcg', 'combined'], help='奖励类型')
    parser.add_argument('--use_beam_search', action='store_true', help='使用beam search而非sampling')
    parser.add_argument('--test_during_training', action='store_true', help='训练时进行评估')
    parser.add_argument('--test_beam', type=int, default=20, help='测试时的beam size')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (必须是num_generations的倍数，默认=num_generations)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--max_completion_length', type=int, default=20, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    
    args = parser.parse_args()
    
    # 验证参数：batch_size 必须是 num_generations 的倍数
    if args.batch_size % args.num_generations != 0:
        suggested_batch_size = args.num_generations
        raise ValueError(
            f"batch_size ({args.batch_size}) must be a multiple of num_generations ({args.num_generations}). "
            f"Suggested: --batch_size {suggested_batch_size} --gradient_accumulation_steps {max(1, args.batch_size * args.gradient_accumulation_steps // suggested_batch_size)}"
        )
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting GRPO training for HierGR-SeqRec")
    
    # 设置随机种子
    set_seed(42)
    
    # 加载配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # 加载 SID mapping
    logger.info("Loading SID mapping...")
    sid_mapping = load_sid_mapping(config)
    logger.info(f"Loaded {len(sid_mapping)} items")
    
    # 准备数据集
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset, train_prompt2target = prepare_dataset(args.train_data)
    logger.info(f"Training samples: {len(train_dataset)}")
    
    if args.eval_data:
        logger.info(f"Loading eval data from {args.eval_data}")
        eval_dataset, eval_prompt2target = prepare_dataset(args.eval_data)
        logger.info(f"Eval samples: {len(eval_dataset)}")
        # 合并prompt2target
        prompt2target = {**train_prompt2target, **eval_prompt2target}
    else:
        eval_dataset = None
        prompt2target = train_prompt2target
    
    # 创建reward函数
    logger.info(f"Creating reward function: {args.reward_type}")
    reward_func = create_reward_function(
        reward_type=args.reward_type,
        num_generations=args.num_generations
    )
    
    # GRPO配置
    # 确保 eval_batch_size 是 num_generations 的倍数
    eval_batch_size = max(1, args.num_generations // 4)  # 使用较小的 eval batch
    if eval_batch_size < args.num_generations:
        eval_batch_size = args.num_generations
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,  # 必须 >= num_generations
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        logging_steps=10,  # 每10步记录一次
        save_steps=500,
        eval_steps=None,  # GRPO不使用默认evaluation
        eval_strategy="no",  # GRPO在训练时已计算metrics
        save_strategy="steps",
        save_total_limit=3,
        bf16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=False,  # GRPO不能使用gradient checkpointing！
        report_to="none",  # 改为 "wandb" 如果需要
        seed=42,
    )
    
    logger.info("Initializing GRPO Trainer...")
    
    # 创建trainer
    trainer = HierGRSeqRecGRPOTrainer(
        model=args.sft_model,
        sid_mapping=sid_mapping,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prompt2target=prompt2target,
        use_beam_search=args.use_beam_search,
        test_during_training=args.test_during_training,
        test_beam=args.test_beam,
    )
    
    # 检查是否有checkpoint可以恢复
    checkpoint_dir = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            # 使用最新的checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            checkpoint_dir = os.path.join(args.output_dir, latest_checkpoint)
            logger.info(f"Found existing checkpoint: {checkpoint_dir}")
            logger.info("Will resume training from checkpoint...")
    
    # 训练
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    
    # 保存模型
    logger.info("Saving final model...")
    trainer.save_model(args.output_dir)
    trainer.save_state()
    
    # 保存metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {args.output_dir}")
    
    # 打印最终指标
    logger.info("=" * 50)
    logger.info("Final Training Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
