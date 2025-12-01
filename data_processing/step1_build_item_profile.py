"""
Step 1: 商户画像构建 (Item Profile Construction)

目标：为 RQ-VAE 提供高质量的语义输入
输入：business.json, review.json, tip.json
输出：item_profiles.jsonl - 每个商家的富文本描述

逻辑：聚合 Name + City + Categories + Attributes + Top Reviews
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm
import yaml


class ItemProfileBuilder:
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        
        # Paths
        self.raw_dir = self.data_config['raw_dir']
        self.processed_dir = self.data_config['processed_dir']
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Data
        self.businesses = {}
        self.business_reviews = defaultdict(list)
        self.business_tips = defaultdict(list)
    
    def load_businesses(self):
        """加载商家基本信息"""
        print("\n=== Loading Business Data ===")
        business_file = os.path.join(self.raw_dir, self.data_config['business_file'])
        
        with open(business_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading businesses"):
                business = json.loads(line.strip())
                self.businesses[business['business_id']] = business
        
        print(f"Loaded {len(self.businesses)} businesses")
    
    def load_reviews(self):
        """加载评论数据，为每个商家提取 Top-K 有用评论"""
        print("\n=== Loading Review Data ===")
        review_file = os.path.join(self.raw_dir, self.data_config['review_file'])
        
        skipped_count = 0
        success_count = 0
        
        with open(review_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading reviews"), 1):
                try:
                    review = json.loads(line.strip())
                    business_id = review['business_id']
                    
                    # 只保留评分和有用性信息
                    self.business_reviews[business_id].append({
                        'text': review['text'],
                        'useful': review.get('useful', 0),
                        'stars': review['stars']
                    })
                    success_count += 1
                except json.JSONDecodeError as e:
                    skipped_count += 1
                    if skipped_count <= 10:  # 只打印前 10 个错误
                        print(f"\nWarning: Skipping malformed JSON at line {line_num}: {str(e)[:100]}")
                except Exception as e:
                    skipped_count += 1
                    if skipped_count <= 10:
                        print(f"\nWarning: Error at line {line_num}: {str(e)[:100]}")
        
        print(f"\nLoaded {success_count} reviews for {len(self.business_reviews)} businesses")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} malformed lines ({skipped_count/success_count*100:.2f}%)")
        
        # 为每个商家按 useful 排序，保留 Top-K
        top_k = self.preprocess_config['top_reviews_count']
        for business_id in self.business_reviews:
            self.business_reviews[business_id] = sorted(
                self.business_reviews[business_id],
                key=lambda x: x['useful'],
                reverse=True
            )[:top_k]
    
    def load_tips(self):
        """加载 Tip 数据（可选）"""
        print("\n=== Loading Tip Data ===")
        tip_file = os.path.join(self.raw_dir, "yelp_academic_dataset_tip.json")
        
        if not os.path.exists(tip_file):
            print("Tip file not found, skipping...")
            return
        
        with open(tip_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading tips"):
                tip = json.loads(line.strip())
                business_id = tip['business_id']
                self.business_tips[business_id].append({
                    'text': tip['text'],
                    'likes': tip.get('likes', 0)
                })
        
        print(f"Loaded tips for {len(self.business_tips)} businesses")
        
        # 按 likes 排序
        for business_id in self.business_tips:
            self.business_tips[business_id] = sorted(
                self.business_tips[business_id],
                key=lambda x: x['likes'],
                reverse=True
            )[:3]
    
    def extract_key_attributes(self, attributes):
        """提取关键属性"""
        if not attributes or attributes == 'None':
            return []
        
        # 关键属性列表
        key_attrs = [
            'RestaurantsPriceRange2', 'Ambience', 'GoodForKids',
            'RestaurantsTakeOut', 'RestaurantsDelivery', 'OutdoorSeating',
            'WiFi', 'Alcohol', 'NoiseLevel', 'RestaurantsAttire',
            'HasTV', 'Caters', 'GoodForGroups'
        ]
        
        extracted = []
        for attr in key_attrs:
            if attr in attributes and attributes[attr] not in [None, 'None', 'False']:
                value = attributes[attr]
                
                # 格式化属性值
                if attr == 'RestaurantsPriceRange2':
                    price_map = {'1': '$', '2': '$$', '3': '$$$', '4': '$$$$'}
                    value = price_map.get(str(value), value)
                    extracted.append(f"Price: {value}")
                elif attr == 'Ambience' and isinstance(value, dict):
                    # Ambience 是字典，提取 True 的键
                    amb = [k for k, v in value.items() if v in [True, 'True']]
                    if amb:
                        extracted.append(f"Ambience: {', '.join(amb)}")
                elif value in [True, 'True']:
                    # 布尔属性直接添加属性名
                    extracted.append(attr.replace('Restaurants', '').replace('GoodFor', 'Good for '))
                else:
                    extracted.append(f"{attr}: {value}")
        
        return extracted
    
    def build_item_profile(self, business_id, business):
        """为单个商家构建富文本描述"""
        import hashlib
        
        # 1. 基础信息
        name = business.get('name', 'Unknown')
        city = business.get('city', 'Unknown')
        state = business.get('state', '')
        address = business.get('address', '')
        categories = business.get('categories', '')
        
        if not categories:
            categories = 'General'
        
        # 生成唯一性标识（用于区分连锁店）
        # 方法1：使用 business_id 的哈希后缀
        id_hash = hashlib.md5(business_id.encode()).hexdigest()[:6]
        
        profile_parts = []
        
        # Name and Location（添加地址和唯一ID）
        profile_parts.append(f"Name: {name}.")
        
        # 添加地址信息（如果有）以区分连锁店
        if address:
            # 截断地址避免过长
            address_short = address[:50] if len(address) > 50 else address
            profile_parts.append(f"Address: {address_short}.")
        
        profile_parts.append(f"City: {city}, {state}.")
        
        # 添加唯一ID哈希（强制区分）
        profile_parts.append(f"ID: {id_hash}.")
        
        profile_parts.append(f"Categories: {categories}.")
        
        # 2. 属性信息
        attributes = business.get('attributes', {})
        key_attrs = self.extract_key_attributes(attributes)
        if key_attrs:
            profile_parts.append(f"Attributes: {'; '.join(key_attrs)}.")
        
        # 3. Top Reviews（众包语义）
        reviews = self.business_reviews.get(business_id, [])
        if reviews:
            highlights = []
            for review in reviews:
                text = review['text']
                # 截断过长的评论
                max_len = self.preprocess_config['max_review_length']
                if len(text) > max_len:
                    text = text[:max_len] + "..."
                highlights.append(text)
            
            profile_parts.append(f"Highlights: {' '.join(highlights)}")
        
        # 4. Tips（可选）
        tips = self.business_tips.get(business_id, [])
        if tips:
            tip_texts = [tip['text'] for tip in tips]
            profile_parts.append(f"Tips: {' '.join(tip_texts)}")
        
        # 拼接成完整描述
        raw_text = ' '.join(profile_parts)
        
        return {
            'business_id': business_id,
            'name': name,
            'city': city,
            'state': state,
            'categories': categories,
            'latitude': business.get('latitude'),
            'longitude': business.get('longitude'),
            'stars': business.get('stars', 0),
            'review_count': business.get('review_count', 0),
            'raw_text': raw_text
        }
    
    def build_all_profiles(self):
        """构建所有商家的画像"""
        print("\n=== Building Item Profiles ===")
        
        output_file = os.path.join(
            self.processed_dir,
            self.data_config['item_profile_file']
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for business_id, business in tqdm(self.businesses.items(), desc="Building profiles"):
                profile = self.build_item_profile(business_id, business)
                f.write(json.dumps(profile, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(self.businesses)} item profiles to {output_file}")
    
    def run(self):
        """执行完整流程"""
        print("\n" + "="*60)
        print("Step 1: Building Item Profiles")
        print("="*60)
        
        self.load_businesses()
        self.load_reviews()
        self.load_tips()
        self.build_all_profiles()
        
        print("\n✓ Step 1 completed successfully!")


def main():
    # Load config
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Build item profiles
    builder = ItemProfileBuilder(config)
    builder.run()


if __name__ == '__main__':
    main()
