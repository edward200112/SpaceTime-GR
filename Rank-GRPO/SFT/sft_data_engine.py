import json
import gzip
import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.neighbors import BallTree
import random

# ================= 核心路径配置 =================
# 1. 原始数据文件夹
RAW_DATA_DIR = "/workspace/data/GoogleRAW"

# 2. 文件列表
REVIEW_FILENAMES = [
    'review-California.json.gz', 
    'review-New_York.json.gz', 
    'review-New_Mexico.json.gz', 
    'review-Pennsylvania.json.gz'
]

META_FILENAMES = [
    'meta-California.json.gz', 
    'meta-New_York.json.gz', 
    'meta-New_Mexico.json.gz', 
    'meta-Pennsylvania.json.gz'
]

# 生成完整路径
REVIEW_FILES = [os.path.join(RAW_DATA_DIR, f) for f in REVIEW_FILENAMES]
META_FILES = [os.path.join(RAW_DATA_DIR, f) for f in META_FILENAMES]

# 3. RQ-VAE 生成的 ID 映射表
ID_MAPPING_FILE = "./poi_semantic_ids.csv"

# 4. 输出路径
OUTPUT_DIR = "./SFT/sft_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_enhanced_train.jsonl")

# ================= 超参数 =================
MIN_HISTORY_LEN = 3         # 过滤短序列
MAX_HISTORY_LEN = 20        # 截断长度
IPS_POWER = 0.5             # IPS 平滑系数

# 难负样本定义
SPATIAL_RADIUS_KM = 2.0     # 保持原设定的 2.0km
CATEGORY_DIST_THRESHOLD = 50.0 # 类别难负：50km 以外

class SFTDataEngine:
    def __init__(self):
        self.poi_meta = {}      # {gmap_id: {lat, lon, categories, name, rating}}
        self.gmap2code = {}     # {gmap_id: "12 34 56 78"}
        self.code2gmap = {}     # {"12 34 56 78": gmap_id}
        
        # 空间索引
        self.spatial_tree = None 
        self.gmap_ids_list = [] # 对应 Tree 索引的 gmap_id 列表
        
        # 全局计数 (用于 IPS)
        self.global_poi_counter = Counter()

    # ------------------------------------------------------------------------
    # 1. 基础工具：Haversine 距离
    # ------------------------------------------------------------------------
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 # 地球半径 (km)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # ------------------------------------------------------------------------
    # 2. 加载数据与构建索引
    # ------------------------------------------------------------------------
    def load_data(self):
        print(f"📥 Phase 1: Loading ID Mappings from {ID_MAPPING_FILE}...")
        if not os.path.exists(ID_MAPPING_FILE):
            raise FileNotFoundError(f"❌ Missing ID mapping file: {ID_MAPPING_FILE}. Please run RQ-VAE inference first.")
            
        id_df = pd.read_csv(ID_MAPPING_FILE)
        # 确保 gmap_id 是字符串
        id_df['gmap_id'] = id_df['gmap_id'].astype(str)
        
        for _, row in tqdm(id_df.iterrows(), total=len(id_df)):
            code_str = f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}"
            gid = row['gmap_id']
            self.gmap2code[gid] = code_str
            self.code2gmap[code_str] = gid
        
        print("📥 Phase 2: Loading Metadata (Lat/Lon/Category)...")
        coords = [] # 用于构建 BallTree
        valid_gids = []
        
        for m_file in META_FILES:
            if not os.path.exists(m_file):
                print(f"   ⚠️ Warning: Meta file not found: {m_file}")
                continue
            print(f"   Reading {m_file}...")
            with gzip.open(m_file, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        
                        # 只处理在我们 Semantic ID 体系内的 POI
                        if gid in self.gmap2code:
                            lat = d.get('latitude')
                            lon = d.get('longitude')
                            cats = d.get('category')
                            
                            # 必须有坐标才能做空间采样
                            if lat and lon:
                                # 处理类别：转为 set 方便计算交集
                                cat_set = set(cats) if isinstance(cats, list) else (set([cats]) if cats else set())
                                
                                self.poi_meta[gid] = {
                                    'lat': float(lat),
                                    'lon': float(lon),
                                    'cats': cat_set,
                                    'name': d.get('name', ''),
                                    'rating': d.get('avg_rating', 0.0)
                                }
                                # 为 BallTree 准备数据 (sklearn 需要弧度)
                                coords.append([math.radians(lat), math.radians(lon)])
                                valid_gids.append(gid)
                    except Exception:
                        continue
                        
        if not coords:
            raise ValueError("No valid POI metadata loaded! Check your raw data paths.")

        # 构建 BallTree (空间索引)
        print(f"🏗️ Building Spatial BallTree for {len(coords)} POIs...")
        # metric='haversine' 输入必须是弧度
        self.spatial_tree = BallTree(np.array(coords), metric='haversine')
        self.gmap_ids_list = valid_gids # 索引映射: Tree Index -> Gmap ID

    # ------------------------------------------------------------------------
    # 3. 核心：难负采样 (CoIN Strategy - 修复版)
    # ------------------------------------------------------------------------
    def mine_hard_negative(self, target_gid):
        """
        [修复] 统一逻辑：先获取 neg_gid (Gmap ID)，最后统一转换为 Code
        """
        if target_gid not in self.poi_meta:
            return self._random_negative_code() # 辅助函数，直接返回Code

        target_meta = self.poi_meta[target_gid]
        target_lat_rad = math.radians(target_meta['lat'])
        target_lon_rad = math.radians(target_meta['lon'])
        
        # 50% 概率选空间难负，50% 概率选类别难负
        strategy = 'spatial' if random.random() < 0.5 else 'category'
        
        neg_gid = None
        
        # --- 策略 A: 空间难负 (Spatial Hard Negative) ---
        if strategy == 'spatial':
            radius_rad = SPATIAL_RADIUS_KM / 6371.0
            
            indices = self.spatial_tree.query_radius([[target_lat_rad, target_lon_rad]], r=radius_rad)[0]
            
            if len(indices) > 1: # 排除自己
                # 随机选一个非自己的邻居
                neg_idx = random.choice(indices)
                cand_gid = self.gmap_ids_list[neg_idx]
                
                if cand_gid != target_gid:
                    neg_gid = cand_gid

        # --- 策略 B: 类别难负 (Category Hard Negative) ---
        if neg_gid is None and target_meta['cats']:
            for _ in range(20):
                rand_gid = random.choice(self.gmap_ids_list)
                if rand_gid == target_gid: continue
                
                cand_meta = self.poi_meta[rand_gid]
                
                # 条件1: 类别有重叠
                has_overlap = not target_meta['cats'].isdisjoint(cand_meta['cats'])
                
                if has_overlap:
                    # 条件2: 距离要远
                    dist = self.haversine_distance(
                        target_meta['lat'], target_meta['lon'],
                        cand_meta['lat'], cand_meta['lon']
                    )
                    if dist > CATEGORY_DIST_THRESHOLD:
                        neg_gid = rand_gid
                        break

        # Fallback: 如果以上都失败，使用随机负采样
        if neg_gid is None:
            # 这里调用 _random_negative_id 返回 ID，以便下面统一处理
            neg_gid = self._random_negative_id(exclude_gid=target_gid)
            
        return self.gmap2code[neg_gid]

    def _random_negative_id(self, exclude_gid=None):
        """
        [修复] 只返回 Gmap ID，不返回 Code
        """
        while True:
            gid = random.choice(self.gmap_ids_list)
            if gid != exclude_gid:
                return gid

    def _random_negative_code(self):
        """辅助：如果target本身不在meta里，直接随机一个code"""
        gid = random.choice(self.gmap_ids_list)
        return self.gmap2code[gid]

    # ------------------------------------------------------------------------
    # 4. 核心：Prompt 增强 (Intent Analysis)
    # ------------------------------------------------------------------------
    def enhance_prompt(self, history_gids, original_prompt):
        """
        分析用户历史意图，动态修改 Prompt
        """
        # 1. 统计历史类别偏好
        cat_counter = Counter()
        for h_gid in history_gids:
            if h_gid in self.poi_meta:
                cat_counter.update(self.poi_meta[h_gid]['cats'])
        
        # 2. 提取 Top 意图
        if cat_counter:
            top_cats = [c for c, _ in cat_counter.most_common(3)] # 取前3个
            intent_str = ", ".join(top_cats)
            
            # 3. 嵌入到 Prompt
            insertion = f"User history reflects preference for [{intent_str}]. Considering sequential patterns, "
            new_prompt = original_prompt.replace("Predict Next POI:", insertion + "Predict Next POI:")
            return new_prompt
        
        return original_prompt

    # ------------------------------------------------------------------------
    # 5. 主流程：构建序列与 IPS
    # ------------------------------------------------------------------------
    def run_pipeline(self):
        self.load_data()
        
        print("🔄 Phase 3: Processing Reviews & Building Sequences...")
        user_history = defaultdict(list)
        
        # 1. 读取评论，按用户聚合
        for r_file in REVIEW_FILES:
            if not os.path.exists(r_file): 
                print(f"   ⚠️ Review file not found: {r_file}")
                continue
            print(f"   Reading {r_file}...")
            with gzip.open(r_file, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        uid = d['user_id']
                        gid = d['gmap_id']
                        time = d['time']
                        
                        # 数据清洗：只保留存在于 Semantic ID 体系的 POI
                        if gid in self.gmap2code:
                            user_history[uid].append((time, gid))
                            self.global_poi_counter[gid] += 1
                    except: continue

        # 2. 计算 IPS 权重 (全局流行度倒数)
        print("⚖️ Calculating IPS Weights...")
        total_interactions = sum(self.global_poi_counter.values())
        if total_interactions == 0:
             raise ValueError("No interactions found! Check review data loading.")

        ips_map = {}
        for gid, count in self.global_poi_counter.items():
            prob = count / total_interactions
            # IPS 公式: 1 / p^0.5
            weight = 1.0 / (math.pow(prob, IPS_POWER) + 1e-9)
            ips_map[gid] = weight
            
        # 归一化 IPS 到 [0.5, 3.0]
        w_values = list(ips_map.values())
        if w_values:
            w_min, w_max = min(w_values), max(w_values)
            for gid in ips_map:
                ips_map[gid] = 0.5 + (ips_map[gid] - w_min) / (w_max - w_min) * 2.5

        # 3. 生成训练样本
        print("🚀 Phase 4: Generating Enhanced SFT Samples...")
        sft_samples = []
        
        for uid, hist in tqdm(user_history.items()):
            # 按时间排序
            hist.sort(key=lambda x: x[0])
            poi_seq = [x[1] for x in hist]
            
            if len(poi_seq) < MIN_HISTORY_LEN: continue
            
            L = len(poi_seq)
            for i in range(MIN_HISTORY_LEN, L):
                target_gid = poi_seq[i]
                
                # Reflect 机制
                if target_gid in self.poi_meta and self.poi_meta[target_gid]['rating'] < 2.0:
                    continue

                input_gids = poi_seq[max(0, i - MAX_HISTORY_LEN) : i]
                
                # 构造 ID 字符串序列
                input_codes = [self.gmap2code[g] for g in input_gids]
                target_code = self.gmap2code[target_gid]
                
                # 基础 Prompt
                input_str = " -> ".join(input_codes)
                base_prompt = f"User History: {input_str}\nPredict Next POI:"
                
                # --- Step A: 意图增强 (Adjust) ---
                enhanced_prompt = self.enhance_prompt(input_gids, base_prompt)
                
                # --- Step B: 难负采样 (CoIN) ---
                # [关键修复] mine_hard_negative 内部已修复，这里调用安全
                neg_code = self.mine_hard_negative(target_gid)
                
                # --- Step C: IPS 权重 ---
                sample_weight = ips_map.get(target_gid, 1.0)
                
                sft_samples.append({
                    "prompt": enhanced_prompt,
                    "completion": target_code,
                    "negative_completion": neg_code,
                    "ips_weight": sample_weight,
                    "target_gmap_id": target_gid 
                })

        print(f"🎉 Generated {len(sft_samples)} High-Quality Samples.")
        
        # 4. 保存
        print(f"💾 Saving to {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, 'w') as f:
            for item in sft_samples:
                f.write(json.dumps(item) + "\n")
        
        print("✅ Data Engine Finished.")

# ================= 运行入口 =================
if __name__ == "__main__":
    engine = SFTDataEngine()
    engine.run_pipeline()