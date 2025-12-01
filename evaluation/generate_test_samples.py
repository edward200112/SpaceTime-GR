"""
快速生成测试样本

从 SID mapping 和用户序列中抽取少量样本用于测试
不需要完整的数据处理流程
"""

import json
import os
import random
import yaml
import argparse


def generate_test_samples(config_path: str, num_samples: int = 20, output_path: str = None):
    """生成测试样本"""
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    prompt_config = config['prompt']
    
    # 路径
    processed_dir = data_config.get('processed_dir', './data/processed')
    sid_mapping_file = os.path.join(processed_dir, data_config['sid_mapping_file'])
    user_sequences_file = os.path.join(processed_dir, data_config['user_sequences_file'])
    
    # 检查文件是否存在
    if not os.path.exists(sid_mapping_file):
        print(f"❌ 错误: SID mapping 文件不存在: {sid_mapping_file}")
        print("\n请先运行数据处理流程:")
        print("  1. python data_processing/step1_build_item_profile.py")
        print("  2. python data_processing/step2_generate_semantic_ids.py")
        return None
    
    if not os.path.exists(user_sequences_file):
        print(f"❌ 错误: 用户序列文件不存在: {user_sequences_file}")
        print("\n请先运行:")
        print("  python data_processing/step3_build_user_sequences.py")
        return None
    
    # 加载数据
    print("加载 SID mapping...")
    with open(sid_mapping_file, 'r', encoding='utf-8') as f:
        sid_mapping = json.load(f)
    
    print("快速采样用户序列（流式读取）...")
    
    # 🚀 优化：使用流式采样，不加载整个文件
    # Reservoir Sampling算法：保持固定大小的样本池
    sampled_sequences = []
    total_lines = 0
    valid_count = 0
    
    # 目标：采集 num_samples * 3 个候选（确保足够）
    target_samples = num_samples * 3
    
    with open(user_sequences_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            seq = json.loads(line)
            
            # 过滤：至少4个商家（3个历史 + 1个目标）
            if len(seq['sequence']) < 4:
                continue
            
            valid_count += 1
            
            # Reservoir sampling
            if len(sampled_sequences) < target_samples:
                sampled_sequences.append(seq)
            else:
                # 随机替换
                j = random.randint(0, valid_count - 1)
                if j < target_samples:
                    sampled_sequences[j] = seq
            
            # 早停：如果已经扫描了足够多的行
            if total_lines >= 100000:  # 最多扫描10万行
                break
    
    print(f"✅ 扫描了 {total_lines} 行，找到 {valid_count} 个有效序列")
    print(f"✅ 采样了 {len(sampled_sequences)} 个候选序列")
    
    if len(sampled_sequences) == 0:
        print("❌ 没有足够长的序列用于测试")
        return None
    
    # 最终随机选择
    num_samples = min(num_samples, len(sampled_sequences))
    sampled_sequences = random.sample(sampled_sequences, num_samples)
    
    print(f"\n生成 {num_samples} 个测试样本...")
    
    # 构造测试样本
    test_samples = []
    task_a_template = prompt_config['task_a_template']
    
    for seq in sampled_sequences:
        sequence = seq['sequence']
        
        # 使用前 N-1 个作为历史，最后一个作为目标
        history_items = sequence[:-1]
        target_item = sequence[-1]
        
        target_biz_id = target_item['business_id']
        
        # 检查目标是否在 SID mapping 中
        if target_biz_id not in sid_mapping:
            continue
        
        # 格式化历史
        history_lines = []
        for idx, item in enumerate(history_items[-5:], 1):  # 只用最后5个
            biz_id = item['business_id']
            if biz_id not in sid_mapping:
                continue
            
            info = sid_mapping[biz_id]
            name = info['name']
            category = info['categories'].split(',')[0].strip() if info['categories'] else 'Unknown'
            cluster_str = info['cluster_str']
            
            history_lines.append(f"{idx}. [{name}] ({category}) -> {cluster_str}")
        
        if not history_lines:
            continue
        
        history_text = "\n".join(history_lines)
        
        # 创建 prompt
        prompt = task_a_template.format(
            longterm_summary="",
            history=history_text
        ).strip()
        
        # 目标信息
        target_info = sid_mapping[target_biz_id]
        target_cluster_str = target_info['cluster_str']
        
        test_samples.append({
            'prompt': prompt,
            'target_business_id': target_biz_id,
            'target_cluster_str': target_cluster_str,
            'target_name': target_info['name'],
            'history_length': len(history_lines)
        })
    
    print(f"✅ 成功生成 {len(test_samples)} 个测试样本")
    
    # 保存
    if output_path is None:
        os.makedirs('./evaluation', exist_ok=True)
        output_path = './evaluation/test_samples.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    
    print(f"测试样本已保存至: {output_path}")
    
    # 显示示例
    print("\n=== 示例测试样本 ===")
    example = test_samples[0]
    print(f"\n历史记录长度: {example['history_length']}")
    print(f"目标: {example['target_name']} -> {example['target_cluster_str']}")
    print(f"\nPrompt:\n{example['prompt'][:200]}...")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='生成测试样本')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='配置文件')
    parser.add_argument('--num_samples', type=int, default=20, help='生成样本数量')
    parser.add_argument('--output', type=str, default='./evaluation/test_samples.json', help='输出文件')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    
    output_path = generate_test_samples(
        config_path=args.config,
        num_samples=args.num_samples,
        output_path=args.output
    )
    
    if output_path:
        print("\n" + "="*60)
        print("✅ 完成！现在可以运行评估:")
        print(f"   python evaluation/evaluate_model.py --test_data {output_path}")
        print("\n或者快速测试:")
        print(f"   python evaluation/quick_test.py --test_file {output_path} --mode batch")
        print("="*60)


if __name__ == '__main__':
    main()
