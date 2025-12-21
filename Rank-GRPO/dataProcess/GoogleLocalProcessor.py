import gzip
import json
import torch
import numpy as np
import pandas as pd
import usaddress
import openlocationcode
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

class MultiStatePreprocessor:
    def __init__(self, meta_file_paths, save_dir="./processed_data"):
        self.meta_file_paths = meta_file_paths
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 1. 初始化模型
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.state_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        
        # 2. 定义严格的州名映射表 (标准化核心)
        # 将所有可能的变体映射到统一的 Full Name
        self.state_mapping = {
            # California
            'CA': 'California', 'CALIFORNIA': 'California', 'CALIF': 'California',
            # New York
            'NY': 'New York', 'NEW YORK': 'New York', 'N.Y.': 'New York', 'NEW YORK STATE': 'New York',
            # New Mexico
            'NM': 'New Mexico', 'NEW MEXICO': 'New Mexico', 'N.M.': 'New Mexico', 'N MEX': 'New Mexico',
            # Pennsylvania
            'PA': 'Pennsylvania', 'PENNSYLVANIA': 'Pennsylvania', 'PENN': 'Pennsylvania', 'PENNA': 'Pennsylvania'
        }
        
        # 定义我们只关心的目标州 (白名单)
        self.target_states = {'California', 'New York', 'New Mexico', 'Pennsylvania'}

    def normalize_state(self, raw_state):
        """
        核心清洗逻辑：将各种乱七八糟的输入归一化为标准名
        """
        if not raw_state or not isinstance(raw_state, str):
            return None
        
        # 1. 预处理：去空格，转大写，去标点
        clean_s = raw_state.strip().upper().replace('.', '')
        
        # 2. 查表
        return self.state_mapping.get(clean_s, None)

    def extract_state_via_regex(self, address_str):
        """
        Fallback 策略：当 usaddress 解析失败时，使用正则暴力提取
        匹配逻辑：逗号 + 空格 + (两字符大写缩写 或 州全名) + 空格 + (邮编 或 结尾)
        """
        # 针对四个州的特定正则模式
        patterns = [
            r',\s*(CA|CALIFORNIA)\b', 
            r',\s*(NY|NEW\s*YORK)\b', 
            r',\s*(NM|NEW\s*MEXICO)\b', 
            r',\s*(PA|PENNSYLVANIA)\b'
        ]
        
        address_upper = address_str.upper()
        for pat in patterns:
            match = re.search(pat, address_upper)
            if match:
                # 提取匹配到的组（例如 'NY' 或 'NEW YORK'）
                return match.group(1)
        return None

    def parse_address_strict(self, address_str):
        """
        双重解析流程
        返回: (Normalized State, Formatted City)
        """
        if not address_str or not isinstance(address_str, str):
            return "Unknown", "Unknown"
        
        final_state = None
        final_city = "Unknown"
        
        # --- 策略 A: usaddress 语义解析 ---
        try:
            parsed, val_type = usaddress.tag(address_str)
            
            # 尝试获取 State
            raw_state = parsed.get('StateName')
            if raw_state:
                final_state = self.normalize_state(raw_state)
            
            # 尝试获取 City
            final_city = parsed.get('PlaceName', "Unknown").strip().title()
            
        except usaddress.RepeatedLabelError:
            # 地址格式太烂，usaddress 崩溃，转入 fallback
            pass
        except Exception:
            pass

        # --- 策略 B: Regex Fallback (如果 A 没找到合法的州) ---
        if final_state is None:
            raw_state_regex = self.extract_state_via_regex(address_str)
            if raw_state_regex:
                final_state = self.normalize_state(raw_state_regex)

        # --- 最终校验 ---
        # 如果经过两轮努力，State 依然不在我们的 4 个目标里，标记为 Invalid
        if final_state not in self.target_states:
            return "Invalid", "Unknown"
            
        return final_state, final_city

    def get_plus_code(self, lat, lon):
        try:
            return openlocationcode.encode(lat, lon)
        except:
            return "00000000+00"

    def sinusoidal_embedding(self, values, dim=32):
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        pos_enc_a = np.sin(np.outer(values, inv_freq))
        pos_enc_b = np.cos(np.outer(values, inv_freq))
        return np.hstack([pos_enc_a, pos_enc_b])

    def load_and_merge_data(self):
        all_data = []
        print(f"🔄 Starting to load {len(self.meta_file_paths)} state files...")
        
        for file_path in self.meta_file_paths:
            if not os.path.exists(file_path):
                print(f"    ❌ File not found: {file_path}")
                continue
                
            print(f"  - Loading {file_path} ...")
            file_data = []
            try:
                with gzip.open(file_path, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if 'address' in record and record['address']:
                            file_data.append(record)
                print(f"    Loaded {len(file_data)} raw records.")
                all_data.extend(file_data)
            except Exception as e:
                print(f"    ⚠️ Error loading {file_path}: {e}")
                
        return pd.DataFrame(all_data)

    def process(self):
        # 1. 加载数据
        df = self.load_and_merge_data()
        if len(df) == 0:
            raise ValueError("No data loaded! Check file paths.")
            
        print(f"Processing {len(df)} records with Strict Address Parsing...")
        
        states = []
        cities = []
        plus_codes = []
        valid_indices = [] # 记录有效行的索引
        
        # 2. 清洗循环
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # === 严格解析 ===
            state, city = self.parse_address_strict(row.get('address', ''))
            
            # 如果解析结果是 "Invalid"，说明这条数据不在我们的4个州里，或者是脏数据
            if state == "Invalid":
                continue
            
            # 记录有效数据
            states.append(state)
            cities.append(city)
            valid_indices.append(idx)
            
            # Plus Code
            lat = row.get('latitude')
            lon = row.get('longitude')
            if lat is not None and lon is not None:
                pc = self.get_plus_code(lat, lon)
            else:
                pc = "00000000+00"
            plus_codes.append(pc)

        # 3. 重建 DataFrame (只保留清洗后有效的数据)
        print(f"Filtering complete. Keeping {len(valid_indices)} / {len(df)} records.")
        
        df_clean = df.loc[valid_indices].copy()
        df_clean['normalized_state'] = states
        df_clean['parsed_city'] = cities
        df_clean['plus_code'] = plus_codes
        
        # 打印清洗后的分布，让你检查是否符合预期
        print("\n📊 Final State Distribution:")
        print(df_clean['normalized_state'].value_counts())
        
        if len(df_clean) == 0:
             raise ValueError("All data was filtered out! Check if regex/mapping matches your data format.")

        # 4. 标签编码
        print("\nEncoding Labels...")
        # 这里 fit 之后，classes_ 顺序固定。建议检查一下是否是4个
        df_clean['state_id'] = self.state_encoder.fit_transform(df_clean['normalized_state'])
        df_clean['city_id'] = self.city_encoder.fit_transform(df_clean['parsed_city'])
        
        print(f"Found {len(self.state_encoder.classes_)} unique states: {self.state_encoder.classes_}")
        
        # 保存 Classes
        np.save(f"{self.save_dir}/state_classes.npy", self.state_encoder.classes_)
        np.save(f"{self.save_dir}/city_classes.npy", self.city_encoder.classes_)
        
        # 5. 特征构建
        print("Constructing Multimodal Features...")
        prompts = []
        for _, row in df_clean.iterrows():
            cats = ", ".join(row.get('category', [])) if isinstance(row.get('category'), list) else str(row.get('category'))
            desc = row.get('description') or ""
            # Prompt 强化地理信息
            prompt = f"Location: {row['parsed_city']}, {row['normalized_state']}. Category: {cats}. Name: {row['name']}. {desc}"
            prompts.append(prompt)
            
        text_embeddings = self.text_encoder.encode(prompts, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
        
        lats = df_clean['latitude'].fillna(0).values
        lons = df_clean['longitude'].fillna(0).values
        lat_emb = self.sinusoidal_embedding(lats, dim=32)
        lon_emb = self.sinusoidal_embedding(lons, dim=32)
        geo_embeddings = torch.tensor(np.hstack([lat_emb, lon_emb]), dtype=torch.float32)
        
        collab_embeddings = torch.zeros((len(df_clean), 64))
        
        fusion_input = torch.cat([text_embeddings.cpu(), geo_embeddings, collab_embeddings], dim=1)
        
        # 6. 保存
        save_path = f"{self.save_dir}/train_data.pt"
        print(f"Saving to {save_path} ...")
        torch.save({
            'features': fusion_input,
            'state_ids': torch.tensor(df_clean['state_id'].values, dtype=torch.long),
            'city_ids': torch.tensor(df_clean['city_id'].values, dtype=torch.long),
            'gmap_ids': df_clean['gmap_id'].values,
            'state_names': df_clean['normalized_state'].values
        }, save_path)
        
        print("✅ Data Preprocessing Complete!")

if __name__ == "__main__":
    # 1. 定义文件夹路径
    raw_data_dir = "/workspace/data/GoogleRAW"
    
    # 2. 拼接完整的文件路径
    files = [
        os.path.join(raw_data_dir, 'meta-California.json.gz'),
        os.path.join(raw_data_dir, 'meta-New_York.json.gz'),
        os.path.join(raw_data_dir, 'meta-New_Mexico.json.gz'),
        os.path.join(raw_data_dir, 'meta-Pennsylvania.json.gz')
    ]
    
    # 3. 初始化并运行
    # 你也可以自定义 save_dir，例如存放在 workspace 下
    processor = MultiStatePreprocessor(files, save_dir="/workspace/processed_data")
    processor.process()