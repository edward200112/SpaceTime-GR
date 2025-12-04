"""
Step 1: 商户画像构建 (Item Profile Construction) - Optimized

优化点：
1. 移除 Hash ID (避免对语义模型产生噪声干扰)
2. 强化地理差异性 (加入 Postal Code 和自然语言地址，解决连锁店冲突)
3. 属性自然语言化 (将 key-value 转换为通顺的句子)
4. 数据清洗 (去除评论中的换行符和乱码)
"""

import json
import os
import re
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
    
    def clean_text(self, text):
        """清洗文本：去除换行、多余空格"""
        if not text:
            return ""
        # 替换换行符为空格
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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
        """加载评论数据"""
        print("\n=== Loading Review Data ===")
        review_file = os.path.join(self.raw_dir, self.data_config['review_file'])
        
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading reviews"):
                try:
                    review = json.loads(line.strip())
                    business_id = review['business_id']
                    
                    # 简单的预清洗
                    cleaned_text = self.clean_text(review['text'])
                    
                    self.business_reviews[business_id].append({
                        'text': cleaned_text,
                        'useful': review.get('useful', 0),
                        'stars': review['stars']
                    })
                except:
                    continue
        
        # 排序并保留 Top-K
        top_k = self.preprocess_config.get('top_reviews_count', 3) # 默认取3条，太多会稀释语义
        for business_id in self.business_reviews:
            # 优先选择 useful 且 长度适中 的评论（过短的信息量少，过长的可能跑题）
            # 这里简单策略：按 useful 降序
            self.business_reviews[business_id] = sorted(
                self.business_reviews[business_id],
                key=lambda x: x['useful'],
                reverse=True
            )[:top_k]
    
    def load_tips(self):
        """加载 Tip 数据"""
        print("\n=== Loading Tip Data ===")
        tip_file = os.path.join(self.raw_dir, "yelp_academic_dataset_tip.json")
        
        if not os.path.exists(tip_file):
            return
        
        with open(tip_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading tips"):
                try:
                    tip = json.loads(line.strip())
                    business_id = tip['business_id']
                    self.business_tips[business_id].append({
                        'text': self.clean_text(tip['text']),
                        'likes': tip.get('likes', 0)
                    })
                except:
                    continue
        
        # 按 likes 排序，取 Top 3
        for business_id in self.business_tips:
            self.business_tips[business_id] = sorted(
                self.business_tips[business_id],
                key=lambda x: x['likes'],
                reverse=True
            )[:3]
    
    def format_attribute(self, key, value):
        """将属性转换为自然语言句子，增强语义理解"""
        if value in ['None', 'False', None]:
            return None
            
        key_map = {
            'RestaurantsPriceRange2': lambda v: f"Price range is {len(str(v)) * '$'} ({['Cheap', 'Moderate', 'Expensive', 'Luxury'][int(v)-1] if str(v).isdigit() and 1<=int(v)<=4 else v})" if str(v).isdigit() else None,
            'WiFi': lambda v: f"Wi-Fi is {v}" if v != "u'no'" and v != "'no'" else "No Wi-Fi",
            'Alcohol': lambda v: f"Serves {v}" if v != "u'none'" and v != "'none'" else "No Alcohol",
            'OutdoorSeating': lambda v: "Has outdoor seating" if v == "True" else None,
            'RestaurantsDelivery': lambda v: "Offers delivery" if v == "True" else None,
            'RestaurantsTakeOut': lambda v: "Offers takeout" if v == "True" else None,
            'GoodForKids': lambda v: "Good for kids" if v == "True" else None,
            'HasTV': lambda v: "Has TV" if v == "True" else None,
            'NoiseLevel': lambda v: f"Noise level is {v}",
            'Ambience': lambda v: f"Ambience is {', '.join([k for k, val in eval(v).items() if val])}" if isinstance(v, str) and '{' in v else None
        }

        if key in key_map:
            try:
                return key_map[key](str(value))
            except:
                return f"{key}: {value}"
        return None

    def build_item_profile(self, business_id, business):
        """
        构建富文本描述
        格式策略：[Category] -> [Name] -> [Location Details] -> [Features] -> [Vibe/Reviews]
        这种顺序符合由粗到细的语义逻辑
        """
        
        # 1. 核心身份 (Category + Name)
        name = self.clean_text(business.get('name', 'Unknown'))
        categories = self.clean_text(business.get('categories', 'General Place'))
        
        # 2. 地理语义 (这是解决 "Collision" 的关键)
        # 必须包含 Postal Code，因为它是最强的区域语义特征
        city = business.get('city', '')
        state = business.get('state', '')
        address = self.clean_text(business.get('address', ''))
        postal_code = business.get('postal_code', '')
        
        location_desc = f"Located in {city}, {state}"
        if postal_code:
            location_desc += f" (Zip: {postal_code})"
        if address:
            location_desc += f", at {address}"
        location_desc += "."

        # 3. 属性特征 (自然语言化)
        attributes = business.get('attributes', {})
        attr_sentences = []
        if attributes:
            # 选取最具语义区分度的属性
            target_attrs = ['RestaurantsPriceRange2', 'Ambience', 'NoiseLevel', 'Alcohol', 'WiFi', 'OutdoorSeating']
            for k, v in attributes.items():
                if k in target_attrs:
                    sent = self.format_attribute(k, v)
                    if sent: attr_sentences.append(sent)
        
        attr_text = ". ".join(attr_sentences) + "." if attr_sentences else ""
        
        # 4. 众包评价 (Reviews & Tips)
        # 混合 Review 和 Tip，Tip 通常包含非常具体的 location 提示 (e.g., "Entrance is around the corner")
        reviews = self.business_reviews.get(business_id, [])
        tips = self.business_tips.get(business_id, [])
        
        feedback_texts = []
        # 取最有用的1条 Tip (通常短且包含关键信息)
        if tips:
            feedback_texts.append(f"Tip: {tips[0]['text']}")
        
        # 取 Top 2 Reviews
        for i, rev in enumerate(reviews[:2]):
            text = rev['text']
            # 截断过长评论，保留前 50 个词，避免覆盖其他特征的权重
            words = text.split()
            if len(words) > 50:
                text = " ".join(words[:50]) + "..."
            feedback_texts.append(f"Review: {text}")

        feedback_str = " ".join(feedback_texts)

        # === 最终拼接 ===
        # 移除 "ID: hash" 这种噪声
        # 模板：Category -> Name -> Location -> Attributes -> Feedback
        raw_text = f"Category: {categories}. Name: {name}. {location_desc} {attr_text} {feedback_str}"
        
        # 清理多余空格
        raw_text = self.clean_text(raw_text)

        return {
            'business_id': business_id,
            'name': name,
            'city': city,
            'state': state,
            'categories': categories,
            'latitude': business.get('latitude'),
            'longitude': business.get('longitude'), # 保留给后续 RL 做 Geo Reward 计算
            'stars': business.get('stars', 0),
            'raw_text': raw_text
        }
    
    def build_all_profiles(self):
        print("\n=== Building Item Profiles ===")
        output_file = os.path.join(self.processed_dir, self.data_config['item_profile_file'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for business_id, business in tqdm(self.businesses.items(), desc="Building profiles"):
                profile = self.build_item_profile(business_id, business)
                f.write(json.dumps(profile, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(self.businesses)} item profiles to {output_file}")
    
    def run(self):
        print("\n" + "="*60)
        print("Step 1: Building Item Profiles (Optimized)")
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
    builder = ItemProfileBuilder(config)
    builder.run()

if __name__ == '__main__':
    main()