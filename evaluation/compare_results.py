"""
模型评估结果对比工具

用法:
python compare_results.py \
    --baseline results_baseline.json \
    --finetuned results_finetuned.json \
    --output comparison_report.md
"""

import json
import argparse
from typing import Dict, List
from datetime import datetime


class ResultComparator:
    def __init__(self):
        self.metric_order = ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20', 'MRR']
    
    def load_result(self, file_path: str) -> Dict:
        """加载评估结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_improvement(self, baseline: float, current: float) -> tuple:
        """计算提升百分比"""
        if baseline == 0:
            return 0, "N/A"
        
        absolute_diff = current - baseline
        relative_diff = (current - baseline) / baseline * 100
        
        return absolute_diff, relative_diff
    
    def generate_comparison_table(self, baseline: Dict, finetuned: Dict) -> str:
        """生成对比表格"""
        baseline_metrics = baseline['metrics']
        finetuned_metrics = finetuned['metrics']
        
        # 表头
        lines = []
        lines.append("| 指标 | Baseline | Fine-tuned | 绝对提升 | 相对提升 | 状态 |")
        lines.append("|------|----------|------------|----------|----------|------|")
        
        # 数据行
        for metric in self.metric_order:
            if metric not in baseline_metrics:
                continue
            
            baseline_val = baseline_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            
            abs_diff, rel_diff = self.calculate_improvement(baseline_val, finetuned_val)
            
            # 判断状态
            if rel_diff > 5:
                status = "✅ 显著提升"
            elif rel_diff > 0:
                status = "⬆️ 小幅提升"
            elif rel_diff == 0:
                status = "➡️ 无变化"
            elif rel_diff > -5:
                status = "⬇️ 小幅下降"
            else:
                status = "❌ 显著下降"
            
            if isinstance(rel_diff, float):
                lines.append(
                    f"| {metric} | {baseline_val:.4f} | {finetuned_val:.4f} | "
                    f"{abs_diff:+.4f} | {rel_diff:+.2f}% | {status} |"
                )
            else:
                lines.append(
                    f"| {metric} | {baseline_val:.4f} | {finetuned_val:.4f} | "
                    f"{abs_diff:+.4f} | {rel_diff} | {status} |"
                )
        
        return "\n".join(lines)
    
    def generate_summary(self, baseline: Dict, finetuned: Dict) -> str:
        """生成总结"""
        baseline_metrics = baseline['metrics']
        finetuned_metrics = finetuned['metrics']
        
        improvements = []
        degradations = []
        
        for metric in self.metric_order:
            if metric not in baseline_metrics:
                continue
            
            _, rel_diff = self.calculate_improvement(
                baseline_metrics[metric],
                finetuned_metrics[metric]
            )
            
            if isinstance(rel_diff, float):
                if rel_diff > 0:
                    improvements.append((metric, rel_diff))
                elif rel_diff < 0:
                    degradations.append((metric, rel_diff))
        
        lines = []
        lines.append("## 📊 总结\n")
        
        if improvements:
            lines.append("### ✅ 提升的指标\n")
            for metric, diff in sorted(improvements, key=lambda x: -x[1]):
                lines.append(f"- **{metric}**: +{diff:.2f}%")
            lines.append("")
        
        if degradations:
            lines.append("### ❌ 下降的指标\n")
            for metric, diff in sorted(degradations, key=lambda x: x[1]):
                lines.append(f"- **{metric}**: {diff:.2f}%")
            lines.append("")
        
        if not improvements and not degradations:
            lines.append("所有指标无显著变化。\n")
        
        # 综合评价
        avg_improvement = sum([d for _, d in improvements]) / len(improvements) if improvements else 0
        
        lines.append("### 🎯 综合评价\n")
        if avg_improvement > 10:
            lines.append("**结论**: 模型微调效果显著，强烈建议使用微调后的模型。")
        elif avg_improvement > 5:
            lines.append("**结论**: 模型微调效果良好，建议使用微调后的模型。")
        elif avg_improvement > 0:
            lines.append("**结论**: 模型微调有小幅提升，可考虑使用微调后的模型。")
        elif avg_improvement == 0:
            lines.append("**结论**: 模型微调无明显效果，建议检查训练配置。")
        else:
            lines.append("**结论**: 模型微调导致性能下降，不建议使用。请检查：")
            lines.append("- 训练数据质量")
            lines.append("- 是否过拟合（检查训练/验证loss）")
            lines.append("- 学习率是否过大")
            lines.append("- 是否训练步数不足")
        
        return "\n".join(lines)
    
    def generate_config_info(self, baseline: Dict, finetuned: Dict) -> str:
        """生成配置信息"""
        lines = []
        lines.append("## ⚙️ 实验配置\n")
        
        lines.append("### Baseline")
        lines.append(f"- 测试数据: `{baseline.get('test_data', 'N/A')}`")
        lines.append(f"- Beam size: {baseline.get('num_beams', 'N/A')}")
        lines.append(f"- Batch size: {baseline.get('batch_size', 'N/A')}")
        lines.append("")
        
        lines.append("### Fine-tuned")
        lines.append(f"- 测试数据: `{finetuned.get('test_data', 'N/A')}`")
        lines.append(f"- Beam size: {finetuned.get('num_beams', 'N/A')}")
        lines.append(f"- Batch size: {finetuned.get('batch_size', 'N/A')}")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_report(
        self,
        baseline_path: str,
        finetuned_path: str,
        output_path: str = None
    ):
        """生成完整对比报告"""
        
        # 加载结果
        baseline = self.load_result(baseline_path)
        finetuned = self.load_result(finetuned_path)
        
        # 生成报告
        report_lines = []
        report_lines.append(f"# 模型评估对比报告")
        report_lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("---\n")
        
        # 配置信息
        report_lines.append(self.generate_config_info(baseline, finetuned))
        report_lines.append("---\n")
        
        # 对比表格
        report_lines.append("## 📈 详细对比\n")
        report_lines.append(self.generate_comparison_table(baseline, finetuned))
        report_lines.append("\n---\n")
        
        # 总结
        report_lines.append(self.generate_summary(baseline, finetuned))
        report_lines.append("\n---\n")
        
        # 建议
        report_lines.append("## 💡 优化建议\n")
        
        # 分析主要问题
        hr10_baseline = baseline['metrics'].get('HR@10', 0)
        hr10_finetuned = finetuned['metrics'].get('HR@10', 0)
        
        if hr10_finetuned < 0.3:
            report_lines.append("### 🔴 性能较低")
            report_lines.append("- 增加训练数据量")
            report_lines.append("- 检查数据质量和标注准确性")
            report_lines.append("- 尝试更大的模型或更长的训练时间")
            report_lines.append("- 调整 Cluster 粒度，减少过细的聚类")
        elif hr10_finetuned < 0.45:
            report_lines.append("### 🟡 性能中等")
            report_lines.append("- 尝试数据增强技术")
            report_lines.append("- 调整学习率和训练步数")
            report_lines.append("- 考虑使用更复杂的提示模板")
        else:
            report_lines.append("### 🟢 性能良好")
            report_lines.append("- 模型表现已达到较高水平")
            report_lines.append("- 可考虑在更多场景下测试泛化能力")
            report_lines.append("- 进行在线 A/B 测试验证实际效果")
        
        report = "\n".join(report_lines)
        
        # 输出
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"对比报告已保存至: {output_path}")
        
        print("\n" + report)
        
        return report


def main():
    parser = argparse.ArgumentParser(description='对比模型评估结果')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline 结果文件')
    parser.add_argument('--finetuned', type=str, required=True, help='Fine-tuned 结果文件')
    parser.add_argument('--output', type=str, default='comparison_report.md', help='报告输出路径')
    
    args = parser.parse_args()
    
    comparator = ResultComparator()
    comparator.generate_report(
        baseline_path=args.baseline,
        finetuned_path=args.finetuned,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
