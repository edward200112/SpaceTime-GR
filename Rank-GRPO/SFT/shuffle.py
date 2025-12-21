import random
import json

input_file = "SFT/sft_data/sft_enhanced_train.jsonl"
output_file = "SFT/sft_data/sft_shuffled_final.jsonl"

print("Reading data...")
with open(input_file, 'r') as f:
    lines = f.readlines()

print(f"Total samples captured: {len(lines)}")
print("Shuffling...")
random.shuffle(lines)

print("Writing shuffled data...")
with open(output_file, 'w') as f:
    f.writelines(lines)

print("Done! Use this file for training.")