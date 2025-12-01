"""
快速测试脚本 - 手动验证模型推理

用法:
python quick_test.py --config ./config/config.yaml
"""

import os
import sys
import json
import yaml
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re


class QuickTester:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        print("加载模型中...")
        self.model, self.tokenizer = self.load_model()
        print("加载 SID 映射中...")
        self.sid_mapping = self.load_sid_mapping()
        print("\n✅ 初始化完成！\n")
    
    def load_model(self):
        """加载模型"""
        llm_config = self.config['llm']
        base_model_name = llm_config['model_name']
        ckpt_dir = self.config['data']['llm_ckpt_dir']
        
        # 1. 从基础模型加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side='left'  # Decoder-only 模型使用左填充
        )
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 记录原始词汇表大小
        original_vocab_size = len(tokenizer)
        
        # 2. 加载 SID tokens 并扩展词汇表
        sid_tokens = self.load_sid_tokens()
        if sid_tokens:
            num_added = tokenizer.add_tokens(sid_tokens)
            print(f"添加了 {num_added} 个 SID tokens (词汇表: {original_vocab_size} -> {len(tokenizer)})")
        
        # 3. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if llm_config['bf16'] else torch.float16,
            device_map='auto'
        )
        
        # 4. 调整模型 embeddings
        if len(tokenizer) > original_vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # 5. 加载 LoRA 权重
        if llm_config['use_lora']:
            model = PeftModel.from_pretrained(model, ckpt_dir)
            model = model.merge_and_unload()
        
        model.eval()
        return model, tokenizer
    
    def load_sid_tokens(self):
        """加载所有唯一的 SID tokens"""
        data_config = self.config['data']
        processed_dir = data_config['processed_dir']
        mapping_file = os.path.join(processed_dir, data_config['sid_mapping_file'])
        
        if not os.path.exists(mapping_file):
            return []
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        # 提取所有唯一的 cluster_str tokens
        unique_tokens = set()
        for item_data in sid_mapping.values():
            cluster_str = item_data['cluster_str']
            unique_tokens.add(cluster_str)
        
        return sorted(list(unique_tokens))
    
    def load_sid_mapping(self):
        """加载 SID 映射"""
        mapping_file = os.path.join(
            self.config['data']['processed_dir'],
            self.config['data']['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_history_from_ids(self, business_ids: list) -> str:
        """从 business_id 列表格式化历史"""
        lines = []
        for idx, biz_id in enumerate(business_ids, 1):
            if biz_id not in self.sid_mapping:
                print(f"⚠️  警告: business_id {biz_id} 不在映射中，已跳过")
                continue
            
            info = self.sid_mapping[biz_id]
            name = info['name']
            category = info['categories'].split(',')[0].strip() if info['categories'] else 'Unknown'
            cluster_str = info['cluster_str']
            
            lines.append(f"{idx}. [{name}] ({category}) -> {cluster_str}")
        
        return "\n".join(lines)
    
    def create_prompt(self, history_text: str) -> str:
        """创建推理提示"""
        template = self.config['prompt']['task_a_template']
        prompt = template.format(
            longterm_summary="",
            history=history_text
        ).strip()
        return prompt
    
    def predict(self, prompt: str, num_beams: int = 5) -> list:
        """预测"""
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config['llm']['max_seq_length']
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        predictions = []
        for output in outputs:
            decoded = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            cluster_str = self.parse_cluster_str(decoded)
            if cluster_str:
                predictions.append(cluster_str)
        
        return predictions
    
    def parse_cluster_str(self, text: str) -> str:
        """解析 cluster_str"""
        match = re.search(r'<(\d+),\s*(\d+)>', text)
        if match:
            return f"<{match.group(1)}, {match.group(2)}>"
        return None
    
    def expand_cluster(self, cluster_str: str) -> list:
        """展开 cluster 为具体商家"""
        businesses = []
        for biz_id, info in self.sid_mapping.items():
            if info['cluster_str'] == cluster_str:
                businesses.append({
                    'business_id': biz_id,
                    'name': info['name'],
                    'category': info['categories'].split(',')[0].strip() if info['categories'] else 'Unknown',
                    'city': info['city']
                })
        return businesses
    
    def interactive_test(self):
        """交互式测试"""
        print("=" * 60)
        print("🔍 HierGR-SeqRec 快速测试工具")
        print("=" * 60)
        print("\n输入示例:")
        print("  方式1 - 输入 business_id 列表 (逗号分隔):")
        print("    B_abc123,B_def456,B_ghi789")
        print("\n  方式2 - 输入完整历史文本:")
        print("    1. [Starbucks] (Coffee) -> <3, 12>")
        print("    2. [AMC Cinema] (Entertainment) -> <5, 8>")
        print("\n输入 'quit' 退出\n")
        print("=" * 60)
        
        while True:
            print("\n请输入用户历史 (或 'quit' 退出):")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            # 判断输入类型
            if ',' in user_input and not '[' in user_input:
                # 方式1: business_id 列表
                business_ids = [x.strip() for x in user_input.split(',')]
                history_text = self.format_history_from_ids(business_ids)
            else:
                # 方式2: 直接输入历史文本
                history_text = user_input
            
            print("\n📝 格式化的历史:")
            print(history_text)
            
            # 创建 prompt
            prompt = self.create_prompt(history_text)
            
            print("\n🔮 模型推理中...")
            
            # 预测
            predictions = self.predict(prompt, num_beams=5)
            
            if not predictions:
                print("❌ 模型未生成有效预测")
                continue
            
            print(f"\n✅ 预测的 Cluster IDs (Top-{len(predictions)}):")
            for i, cluster_str in enumerate(predictions, 1):
                print(f"  {i}. {cluster_str}")
            
            # 展开第一个 cluster
            print(f"\n🎯 展开 {predictions[0]} 中的商家:")
            businesses = self.expand_cluster(predictions[0])
            
            if businesses:
                for i, biz in enumerate(businesses[:10], 1):  # 只显示前10个
                    print(f"  {i}. [{biz['name']}] ({biz['category']}) - {biz['city']}")
                if len(businesses) > 10:
                    print(f"  ... 还有 {len(businesses) - 10} 个商家")
            else:
                print("  (该 cluster 为空)")
            
            print("\n" + "-" * 60)
    
    def batch_test_from_file(self, test_file: str):
        """从文件批量测试"""
        print(f"从 {test_file} 加载测试样本...")
        
        # 读取 JSONL 格式（每行一个 JSON 对象）
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_samples.append(json.loads(line))
        
        print(f"共 {len(test_samples)} 个测试样本\n")
        
        correct = 0
        total = len(test_samples)
        
        for i, sample in enumerate(test_samples[:10], 1):  # 只测试前10个
            print(f"\n{'='*60}")
            print(f"测试样本 {i}/{min(10, total)}")
            print(f"{'='*60}")
            
            # 从实际 JSONL 格式提取字段
            prompt = sample['instruction']  # instruction 字段是 prompt
            target_biz = sample['metadata']['target_business_id']  # target 在 metadata 中
            target_cluster = sample['output']  # output 字段是目标 cluster
            
            print(f"目标: {target_cluster}")
            if target_biz in self.sid_mapping:
                target_name = self.sid_mapping[target_biz]['name']
                print(f"目标商家: {target_name} ({target_biz})")
            
            # 预测
            predictions = self.predict(prompt, num_beams=5)
            
            print(f"预测: {predictions}")
            
            # 检查是否命中
            if target_cluster in predictions:
                rank = predictions.index(target_cluster) + 1
                print(f"✅ 命中! (排名 {rank})")
                correct += 1
            else:
                print(f"❌ 未命中")
        
        print(f"\n{'='*60}")
        print(f"准确率 (Top-5): {correct}/{min(10, total)} = {correct/min(10, total)*100:.2f}%")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='HierGR-SeqRec 快速测试')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='配置文件')
    parser.add_argument('--test_file', type=str, help='测试文件（可选，用于批量测试）')
    parser.add_argument('--mode', type=str, default='interactive', choices=['interactive', 'batch'],
                        help='测试模式: interactive 或 batch')
    
    args = parser.parse_args()
    
    tester = QuickTester(args.config)
    
    if args.mode == 'interactive':
        tester.interactive_test()
    elif args.mode == 'batch':
        if args.test_file:
            tester.batch_test_from_file(args.test_file)
        else:
            print("❌ 批量测试需要提供 --test_file 参数")
            print("\n💡 提示：")
            print("   1. 使用交互式模式: python quick_test.py")
            print("   2. 生成测试文件: python evaluation/generate_test_samples.py")
            print("   3. 然后运行: python quick_test.py --test_file evaluation/test_samples.json --mode batch")


if __name__ == '__main__':
    main()
