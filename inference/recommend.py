"""
Online Recommendation Inference

输入：用户历史 + 当前位置
输出：推荐的商家列表

流程：
1. 加载训练好的 LLM 模型
2. 格式化用户历史为 Prompt
3. 预测 Cluster ID
4. 将 Cluster ID 展开为具体商家
5. 根据用户位置过滤
6. 返回 Top-K 推荐
"""

import os
import sys
import json
import yaml
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from typing import List, Dict, Tuple


class HierGRSeqRecInference:
    def __init__(self, config_path: str = './config/config.yaml'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model()
        
        # Load SID mapping
        self.sid_mapping = self.load_sid_mapping()
        
        # Build cluster index
        self.cluster_index = self.build_cluster_index()
        
        print("Inference system initialized successfully!")
    
    def load_model(self):
        """Load trained LLM model"""
        llm_config = self.config['llm']
        base_model_name = llm_config['model_name']
        ckpt_dir = self.config['data']['llm_ckpt_dir']
        
        print(f"Loading base model from {base_model_name}...")
        
        # 1. Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side='left'  # Use left-padding for decoder-only models
        )
        
        # Set pad_token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Record original vocab size
        original_vocab_size = len(tokenizer)
        print(f"Original vocabulary size: {original_vocab_size}")
        
        # 2. Load SID tokens and expand vocabulary (consistent with training)
        sid_tokens = self.load_sid_tokens()
        if sid_tokens:
            print(f"Adding {len(sid_tokens)} SID tokens to vocabulary...")
            num_added = tokenizer.add_tokens(sid_tokens)
            print(f"Successfully added {num_added} new tokens")
            print(f"New vocabulary size: {len(tokenizer)}")
        
        # 3. Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if llm_config['bf16'] else torch.float16,
            device_map='auto'
        )
        
        # 4. Resize model embeddings
        if len(tokenizer) > original_vocab_size:
            print(f"Resizing model embeddings: {original_vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # 5. Load LoRA weights if used
        if llm_config['use_lora']:
            print(f"Loading LoRA weights from {ckpt_dir}...")
            model = PeftModel.from_pretrained(model, ckpt_dir)
            model = model.merge_and_unload()  # Merge LoRA weights into base model
            print("LoRA weights merged successfully")
        
        model.eval()
        
        return model, tokenizer
    
    def load_sid_tokens(self):
        """Load all unique SID tokens (same logic as training)"""
        data_config = self.config['data']
        processed_dir = data_config['processed_dir']
        mapping_file = os.path.join(processed_dir, data_config['sid_mapping_file'])
        
        if not os.path.exists(mapping_file):
            print(f"Warning: SID mapping file not found: {mapping_file}")
            return []
        
        print(f"Loading SID tokens from {mapping_file}...")
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        # Extract all unique cluster_str tokens
        unique_tokens = set()
        for item_data in sid_mapping.values():
            cluster_str = item_data['cluster_str']  # Format: "<0, 12>"
            unique_tokens.add(cluster_str)
        
        sid_tokens = sorted(list(unique_tokens))
        print(f"Found {len(sid_tokens)} unique SID tokens")
        print(f"Example SID tokens: {sid_tokens[:5]}")
        
        return sid_tokens
    
    def load_sid_mapping(self):
        """Load SID mapping"""
        mapping_file = os.path.join(
            self.config['data']['processed_dir'],
            self.config['data']['sid_mapping_file']
        )
        
        with open(mapping_file, 'r', encoding='utf-8') as f:
            sid_mapping = json.load(f)
        
        print(f"Loaded {len(sid_mapping)} items with SIDs")
        return sid_mapping
    
    def build_cluster_index(self):
        """Build cluster to businesses mapping"""
        cluster_index = {}
        
        for business_id, info in self.sid_mapping.items():
            cluster_tuple = tuple(info['cluster_id'])
            if cluster_tuple not in cluster_index:
                cluster_index[cluster_tuple] = []
            cluster_index[cluster_tuple].append({
                'business_id': business_id,
                'name': info['name'],
                'city': info['city'],
                'categories': info['categories'],
                'latitude': info['latitude'],
                'longitude': info['longitude']
            })
        
        print(f"Built cluster index with {len(cluster_index)} clusters")
        return cluster_index
    
    def format_history(self, history: List[Dict]) -> str:
        """Format user history for prompt"""
        history_lines = []
        
        for idx, item in enumerate(history, 1):
            # Get cluster_str from SID mapping
            business_id = item['business_id']
            if business_id not in self.sid_mapping:
                continue
            
            cluster_str = self.sid_mapping[business_id]['cluster_str']
            name = self.sid_mapping[business_id]['name']
            category = self.sid_mapping[business_id]['categories'].split(',')[0].strip()
            
            line = f"[{name}] ({category}) -> {cluster_str}"
            history_lines.append(f"{idx}. {line}")
        
        return '\n'.join(history_lines)
    
    def create_prompt(self, history: List[Dict], longterm_summary: str = None) -> str:
        """Create inference prompt"""
        template = self.config['prompt']['task_a_template']
        
        longterm_text = ""
        if longterm_summary:
            longterm_text = f"User Profile: {longterm_summary}\n"
        
        history_text = self.format_history(history)
        
        prompt = template.format(
            longterm_summary=longterm_text,
            history=history_text
        ).strip()
        
        return prompt
    
    def predict_cluster(self, prompt: str, num_beams: int = 3) -> List[Tuple[List[int], float]]:
        """Predict cluster ID using beam search"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config['llm']['max_seq_length']
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,  # Enough for "<12, 45>"
                num_beams=num_beams,
                num_return_sequences=num_beams,
                temperature=self.config['inference']['temperature'],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        predictions = []
        for output in outputs:
            decoded = self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse cluster ID from text like "<12, 45>"
            cluster_id = self.parse_cluster_id(decoded)
            if cluster_id:
                predictions.append((cluster_id, 1.0 / num_beams))  # Simple uniform score
        
        return predictions
    
    def parse_cluster_id(self, text: str) -> List[int]:
        """Parse cluster ID from generated text"""
        import re
        
        # Match pattern like <12, 45> or <12,45>
        match = re.search(r'<(\d+),\s*(\d+)>', text)
        if match:
            return [int(match.group(1)), int(match.group(2))]
        
        return None
    
    def expand_cluster(self, cluster_id: List[int], user_location: Dict = None) -> List[Dict]:
        """Expand cluster ID to specific businesses"""
        cluster_tuple = tuple(cluster_id)
        
        if cluster_tuple not in self.cluster_index:
            return []
        
        businesses = self.cluster_index[cluster_tuple]
        
        # Filter by location if provided
        if user_location and self.config['inference']['location_filter']:
            max_distance = self.config['inference']['max_distance_km']
            businesses = self.filter_by_location(businesses, user_location, max_distance)
        
        return businesses
    
    def filter_by_location(self, businesses: List[Dict], user_location: Dict, max_distance_km: float) -> List[Dict]:
        """Filter businesses by distance from user location"""
        from math import radians, sin, cos, sqrt, atan2
        
        filtered = []
        user_lat = user_location['latitude']
        user_lon = user_location['longitude']
        
        for biz in businesses:
            if biz['latitude'] is None or biz['longitude'] is None:
                continue
            
            # Haversine distance
            R = 6371  # Earth radius in km
            
            lat1, lon1 = radians(user_lat), radians(user_lon)
            lat2, lon2 = radians(biz['latitude']), radians(biz['longitude'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            if distance <= max_distance_km:
                biz['distance_km'] = distance
                filtered.append(biz)
        
        # Sort by distance
        filtered.sort(key=lambda x: x['distance_km'])
        
        return filtered
    
    def recommend(self, user_history: List[Dict], user_location: Dict = None, top_k: int = 10) -> List[Dict]:
        """Main recommendation function"""
        # Create prompt
        prompt = self.create_prompt(user_history)
        
        print("\n=== Recommendation Process ===")
        print(f"User history: {len(user_history)} items")
        if user_location:
            print(f"User location: {user_location}")
        
        # Predict cluster IDs
        predicted_clusters = self.predict_cluster(prompt, num_beams=self.config['inference']['beam_size'])
        
        if not predicted_clusters:
            print("No valid cluster predicted!")
            return []
        
        print(f"\nPredicted clusters: {predicted_clusters}")
        
        # Expand clusters and aggregate results
        all_candidates = []
        for cluster_id, score in predicted_clusters:
            businesses = self.expand_cluster(cluster_id, user_location)
            for biz in businesses:
                biz['cluster_score'] = score
                all_candidates.append(biz)
        
        print(f"Found {len(all_candidates)} candidate businesses")
        
        # Remove duplicates (same business from different clusters)
        seen = set()
        unique_candidates = []
        for biz in all_candidates:
            if biz['business_id'] not in seen:
                seen.add(biz['business_id'])
                unique_candidates.append(biz)
        
        # Sort by cluster score and distance
        unique_candidates.sort(key=lambda x: (
            -x['cluster_score'],
            x.get('distance_km', float('inf'))
        ))
        
        # Return top-k
        return unique_candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description='HierGR-SeqRec Inference')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Config file path')
    parser.add_argument('--user_history', type=str, required=True, help='User history file (JSON)')
    parser.add_argument('--user_location', type=str, help='User location JSON (optional)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Load inference system
    recommender = HierGRSeqRecInference(args.config)
    
    # Load user history
    with open(args.user_history, 'r', encoding='utf-8') as f:
        user_history = json.load(f)
    
    # Load user location (optional)
    user_location = None
    if args.user_location:
        with open(args.user_location, 'r', encoding='utf-8') as f:
            user_location = json.load(f)
    
    # Recommend
    recommendations = recommender.recommend(user_history, user_location, args.top_k)
    
    # Print results
    print("\n=== Recommendations ===")
    for idx, rec in enumerate(recommendations, 1):
        print(f"\n{idx}. {rec['name']}")
        print(f"   Category: {rec['categories']}")
        print(f"   City: {rec['city']}")
        if 'distance_km' in rec:
            print(f"   Distance: {rec['distance_km']:.2f} km")
        print(f"   Score: {rec['cluster_score']:.4f}")


if __name__ == '__main__':
    main()
