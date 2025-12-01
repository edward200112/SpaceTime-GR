"""
Complete Pipeline Runner for HierGR-SeqRec

运行完整的数据处理和训练流程
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print status"""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print("="*60)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        sys.exit(1)
    else:
        print(f"\n✓ Completed: {description}")


def main():
    parser = argparse.ArgumentParser(description='Run HierGR-SeqRec Pipeline')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'data', 'rqvae', 'llm', 'inference'],
                       help='Which step to run')
    parser.add_argument('--start_from', type=int, default=1,
                       help='Start from which data processing step (1-4)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HierGR-SeqRec Pipeline Runner")
    print("="*60)
    
    # Step 1-4: Data Processing
    if args.step in ['all', 'data']:
        if args.start_from <= 1:
            run_command(
                "python data_processing/step1_build_item_profile.py",
                "Step 1: Building Item Profiles"
            )
        
        if args.start_from <= 2:
            run_command(
                "python data_processing/step2_generate_semantic_ids.py",
                "Step 2: Generating Semantic IDs (RQ-VAE Training)"
            )
        
        if args.start_from <= 3:
            run_command(
                "python data_processing/step3_build_user_sequences.py",
                "Step 3: Building User Sequences"
            )
        
        if args.start_from <= 4:
            run_command(
                "python data_processing/step4_construct_prompts.py",
                "Step 4: Constructing Multi-Task Prompts"
            )
    
    # Step 5: LLM Training
    if args.step in ['all', 'llm']:
        run_command(
            "python training/train_llm.py",
            "Step 5: Training LLM"
        )
    
    # Step 6: Inference Example
    if args.step == 'inference':
        print("\n" + "="*60)
        print("To run inference, use:")
        print("python inference/recommend.py --user_history examples/user_history.json --top_k 10")
        print("="*60)
    
    print("\n" + "="*60)
    print("✓ Pipeline Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
