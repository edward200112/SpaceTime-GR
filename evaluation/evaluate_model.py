"""
模型评估脚本 - HierGR-SeqRec

支持的评估指标：
1. HR@K (Hit Rate): 前K个推荐中是否命中目标
2. NDCG@K (Normalized Discounted Cumulative Gain): 考虑排序位置的命中率
3. MRR (Mean Reciprocal Rank): 目标的平均倒数排名

特性：
- 支持约束beam search（Constrained Generation）
- 只生成有效的 Cluster IDs

用法:
python evaluate_model.py \
    --config ./config/config.yaml \
    --test_data ./data/processed/test_prompts.json \
    --batch_size 8 \
    --num_beams 5 \
    --top_k 5,10,20 \
    --use_constrained_generation \
    --output results.json
"""

import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from peft import PeftModel
import math

# 添加training目录到路径以导入constrained processor
sys.path.append(os.path.join(os.path.dirname(__file__), '../training'))
from constrained_logits_processor import (
    ConstrainedClusterLogitsProcessor,
    build_cluster_hash_dict,
    create_prefix_allowed_tokens_fn,
)


class ModelEvaluator:
    def __init__(self, config_path: str, use_constrained_generation: bool = False):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        self.model, self.tokenizer = self.load_model()
        self.sid_mapping = self.load_sid_mapping()
        
        self.use_constrained_generation = use_constrained_generation
        
        # 如果使用约束生成，构建hash dict
        if use_constrained_generation:
            print("构建约束生成哈希字典...")
            self.hash_dict, self.prefix_index, self.get_hash = build_cluster_hash_dict(
                sid_mapping=self.sid_mapping['mapping'],
                tokenizer=self.tokenizer,
                model_type=self.model.config.model_type
            )
            self.prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(
                self.hash_dict,
                self.get_hash
            )
            print("约束生成已启用")
        else:
            self.prefix_allowed_tokens_fn = None
        
        print("评估系统初始化完成")
    
    def load_model(self):
        """加载训练好的模型"""
        llm_config = self.config['llm']
        base_model_name = llm_config['model_name']
        ckpt_dir = self.config['data']['llm_ckpt_dir']
        
        print(f"从 {base_model_name} 加载基础模型...")
        
        # 1. 加载 tokenizer（从基础模型）
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side='left'  # Decoder-only 模型使用左填充
        )
        
        # 设置 pad_token（如果没有的话）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"设置 pad_token 为 eos_token: {tokenizer.eos_token}")
        
        # 记录原始词汇表大小
        original_vocab_size = len(tokenizer)
        print(f"原始词汇表大小: {original_vocab_size}")
        
        # 2. 加载 SID tokens 并扩展词汇表（与训练时保持一致）
        sid_tokens = self.load_sid_tokens()
        if sid_tokens:
            print(f"添加 {len(sid_tokens)} 个 SID tokens 到词汇表...")
            num_added = tokenizer.add_tokens(sid_tokens)
            print(f"成功添加 {num_added} 个新 tokens")
            print(f"新词汇表大小: {len(tokenizer)}")
        
        # 3. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if llm_config['bf16'] else torch.float16,
            device_map='auto'
        )
        
        # 4. 调整模型 embeddings 大小
        if len(tokenizer) > original_vocab_size:
            print(f"调整模型 embeddings: {original_vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # 5. 加载 LoRA 权重
        if llm_config['use_lora']:
            print(f"从 {ckpt_dir} 加载 LoRA 权重...")
            model = PeftModel.from_pretrained(model, ckpt_dir)
            model = model.merge_and_unload()
            print("LoRA 权重合并完成")
        
        model.eval()
        return model, tokenizer
    
    def load_sid_tokens(self):
        """加载所有唯一的 SID tokens（与训练时相同的逻辑）"""
        data_config = self.config['data']
        processed_dir = data_config['processed_dir']
        mapping_file = os.path.join(processed_dir, data_config['sid_mapping_file'])
        
        if not os.path.exists(mapping_file):
            print(f"警告: SID mapping 文件不存在: {mapping_file}")
            return []
        
        print(f"从 {mapping_file} 加载 SID tokens...")
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        # 提取所有唯一的 cluster_str tokens
        unique_tokens = set()
        for item_data in sid_mapping.values():
            cluster_str = item_data['cluster_str']  # Format: "<0, 12>"
            unique_tokens.add(cluster_str)
        
        sid_tokens = sorted(list(unique_tokens))
        print(f"找到 {len(sid_tokens)} 个唯一的 SID tokens")
        print(f"示例 SID tokens: {sid_tokens[:5]}")
        
        return sid_tokens
    
    def load_sid_mapping(self):
        """加载 SID 映射"""
        mapping_file = os.path.join(
            self.config['data']['processed_dir'],
            self.config['data']['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        # 创建 cluster_str 到 business_id 的映射
        cluster_to_biz = {}
        for biz_id, info in sid_mapping.items():
            cluster_str = info['cluster_str']
            if cluster_str not in cluster_to_biz:
                cluster_to_biz[cluster_str] = []
            cluster_to_biz[cluster_str].append(biz_id)
        
        return {
            'mapping': sid_mapping,
            'cluster_to_biz': cluster_to_biz
        }
    
    def predict_batch(self, prompts: List[str], num_beams: int = 5) -> List[List[str]]:
        """批量预测（支持约束生成）"""
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['llm']['max_seq_length']
        ).to(self.device)
        
        # 准备生成配置
        generation_config = GenerationConfig(
            max_new_tokens=20,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 如果使用约束生成，添加logits processor
        logits_processor = None
        if self.use_constrained_generation and self.prefix_allowed_tokens_fn:
            constrained_processor = ConstrainedClusterLogitsProcessor(
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                num_beams=num_beams,
                model_type=self.model.config.model_type
            )
            logits_processor = LogitsProcessorList([constrained_processor])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor
            )
        
        # Decode
        predictions = []
        batch_size = len(prompts)
        
        for i in range(batch_size):
            beam_predictions = []
            for j in range(num_beams):
                idx = i * num_beams + j
                decoded = self.tokenizer.decode(
                    outputs[idx][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # 解析 cluster_str
                cluster_str = self.parse_cluster_str(decoded)
                if cluster_str:
                    beam_predictions.append(cluster_str)
            
            predictions.append(beam_predictions)
        
        return predictions
    
    def parse_cluster_str(self, text: str) -> str:
        """从生成的文本中解析 cluster_str"""
        import re
        # 匹配 <数字, 数字> 格式
        match = re.search(r'<(\d+),\s*(\d+)>', text)
        if match:
            return f"<{match.group(1)}, {match.group(2)}>"
        return None
    
    def expand_cluster(self, cluster_str: str) -> List[str]:
        """将 cluster_str 展开为具体的 business_id 列表"""
        return self.sid_mapping['cluster_to_biz'].get(cluster_str, [])
    
    def calculate_metrics(
        self,
        predictions: List[List[str]],  # [batch_size, num_beams] cluster_strs
        targets: List[str],  # [batch_size] business_ids
        topk_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """计算评估指标"""
        
        num_samples = len(targets)
        
        # 初始化指标
        hr = {k: 0 for k in topk_list}
        ndcg = {k: 0 for k in topk_list}
        mrr_sum = 0
        
        for i, (pred_clusters, target_biz) in enumerate(zip(predictions, targets)):
            # 展开所有预测的 clusters 为 business_ids
            predicted_bizs = []
            for cluster_str in pred_clusters:
                bizs = self.expand_cluster(cluster_str)
                predicted_bizs.extend(bizs)
            
            # 去重并保持顺序
            seen = set()
            unique_predicted_bizs = []
            for biz in predicted_bizs:
                if biz not in seen:
                    seen.add(biz)
                    unique_predicted_bizs.append(biz)
            
            # 查找目标位置
            try:
                rank = unique_predicted_bizs.index(target_biz) + 1  # 1-indexed
            except ValueError:
                rank = float('inf')  # 未找到
            
            # 计算 HR@K 和 NDCG@K
            for k in topk_list:
                if rank <= k:
                    hr[k] += 1
                    ndcg[k] += 1.0 / math.log2(rank + 1)
            
            # 计算 MRR
            if rank != float('inf'):
                mrr_sum += 1.0 / rank
        
        # 归一化
        metrics = {}
        for k in topk_list:
            metrics[f'HR@{k}'] = hr[k] / num_samples
            metrics[f'NDCG@{k}'] = ndcg[k] / num_samples
        
        metrics['MRR'] = mrr_sum / num_samples
        
        return metrics
    
    def evaluate(
        self,
        test_data_path: str,
        batch_size: int = 8,
        num_beams: int = 5,
        topk_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """评估模型"""
        
        # 加载测试数据（JSONL 格式，每行一个 JSON 对象）
        print(f"加载测试数据: {test_data_path}")
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    test_data.append(json.loads(line))
        
        print(f"测试样本数: {len(test_data)}")
        
        # 准备数据
        prompts = []
        targets = []
        
        for sample in test_data:
            # 从实际数据格式中提取字段
            prompts.append(sample['instruction'])  # instruction 字段是 prompt
            targets.append(sample['metadata']['target_business_id'])  # target 在 metadata 中
        
        # 批量预测
        all_predictions = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        print("开始评估...")
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(prompts))
            
            batch_prompts = prompts[start_idx:end_idx]
            batch_preds = self.predict_batch(batch_prompts, num_beams)
            all_predictions.extend(batch_preds)
        
        # 计算指标
        metrics = self.calculate_metrics(all_predictions, targets, topk_list)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='评估 HierGR-SeqRec 模型')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='配置文件路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_beams', type=int, default=5, help='Beam search 数量')
    parser.add_argument('--top_k', type=str, default='5,10,20', help='Top-K 列表，用逗号分隔')
    parser.add_argument('--use_constrained_generation', action='store_true', help='使用约束生成（推荐）')
    parser.add_argument('--output', type=str, default='./evaluation/results.json', help='结果保存路径')
    
    args = parser.parse_args()
    
    # 解析 top_k
    topk_list = [int(k) for k in args.top_k.split(',')]
    
    # 创建评估器
    evaluator = ModelEvaluator(
        args.config,
        use_constrained_generation=args.use_constrained_generation
    )
    
    # 评估
    metrics = evaluator.evaluate(
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        topk_list=topk_list
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    print("=" * 50)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'test_data': args.test_data,
            'num_beams': args.num_beams,
            'batch_size': args.batch_size,
            'topk_list': topk_list
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
