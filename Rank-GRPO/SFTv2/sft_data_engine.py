# # SFTv2 enhanced version
import json
import gzip
import pandas as pd
import numpy as np
import os
import math
import random
from datetime import datetime  # [新增] 用于时间解析
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.neighbors import BallTree

# ================= 核心路径配置 =================
RAW_DATA_DIR = "/workspace/data/GoogleRAW"

REVIEW_FILES = [
    os.path.join(RAW_DATA_DIR, f) for f in [
        'review-California.json.gz', 'review-New_York.json.gz', 
        'review-New_Mexico.json.gz', 'review-Pennsylvania.json.gz'
    ]
]

META_FILES = [
    os.path.join(RAW_DATA_DIR, f) for f in [
        'meta-California.json.gz', 'meta-New_York.json.gz', 
        'meta-New_Mexico.json.gz', 'meta-Pennsylvania.json.gz'
    ]
]

ID_MAPPING_FILE = "./poi_semantic_ids.csv"
OUTPUT_DIR = "./SFT/sft_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_enhanced_train.jsonl")

# ================= 超参数 =================
MIN_HISTORY_LEN = 3
MAX_HISTORY_LEN = 20
IPS_POWER = 0.5
SPATIAL_RADIUS_KM = 0.5      # 0.5km 严格限制
CATEGORY_DIST_THRESHOLD = 50.0 
MIN_RATING_THRESHOLD = 2.0

# [6.1] CoIN 指令模版池 (增强了时空约束)
# 显式加入 Current Time 和 Location Constraint
INSTRUCTION_TEMPLATES = [
    "Current Time: {time}. Location Constraint: 5km.\nUser History: {hist}\nPredict Next POI:",
    "Context: {time}, within 5km radius.\nTrajectory: {hist}.\nRecommend the next location.",
    "Given the user's past visits: {hist}.\nTime: {time}.\nConstraint: 5km.\nWhere will they go next?",
    "Sequence: {hist}.\nEnvironment: {time}, nearby (5km).\nNext stop prediction:",
]

class SFTDataEngine:
    def __init__(self):
        self.poi_meta = {}      
        self.gmap2code = {}     
        self.code2gmap = {}     
        self.spatial_tree = None 
        self.gmap_ids_list = [] 
        self.global_poi_counter = Counter()
        self.witg_graph = defaultdict(Counter)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def load_data(self):
        print(f"📥 Phase 1: Loading ID Mappings from {ID_MAPPING_FILE}...")
        if not os.path.exists(ID_MAPPING_FILE):
            raise FileNotFoundError(f"❌ Missing ID mapping file: {ID_MAPPING_FILE}. Please run RQ-VAE inference first.")
            
        id_df = pd.read_csv(ID_MAPPING_FILE)
        id_df['gmap_id'] = id_df['gmap_id'].astype(str)
        
        for _, row in tqdm(id_df.iterrows(), total=len(id_df)):
            code_str = f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}"
            gid = row['gmap_id']
            self.gmap2code[gid] = code_str
            self.code2gmap[code_str] = gid
        
        print("📥 Phase 2: Loading Metadata (Lat/Lon/Category)...")
        coords = [] 
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
                        if gid in self.gmap2code:
                            lat = d.get('latitude')
                            lon = d.get('longitude')
                            cats = d.get('category')
                            if lat and lon:
                                cat_set = set(cats) if isinstance(cats, list) else (set([cats]) if cats else set())
                                self.poi_meta[gid] = {
                                    'lat': float(lat), 'lon': float(lon),
                                    'cats': cat_set,
                                    'name': d.get('name', 'Unknown'),
                                    'rating': d.get('avg_rating', 0.0)
                                }
                                coords.append([math.radians(lat), math.radians(lon)])
                                valid_gids.append(gid)
                    except: continue
                        
        if not coords:
            raise ValueError("No valid POI metadata loaded!")

        print(f"🏗️ Building Spatial BallTree for {len(coords)} POIs...")
        self.spatial_tree = BallTree(np.array(coords), metric='haversine')
        self.gmap_ids_list = valid_gids 

    def build_witg(self, user_history):
        print("🕸️ Building Global Weighted Item Transition Graph (WITG)...")
        for uid, hist in tqdm(user_history.items()):
            if len(hist) < 2: continue
            for i in range(len(hist) - 1):
                curr_node = hist[i][1]
                next_node = hist[i+1][1]
                self.witg_graph[curr_node][next_node] += 1

    def mine_hard_negative(self, target_gid, context_gid=None):
        neg_gid = None
        if target_gid in self.poi_meta:
            target_meta = self.poi_meta[target_gid]
            rand_val = random.random()
            
            # --- 策略 A: 空间邻近 (500m) ---
            if rand_val < 0.4: 
                target_lat_rad = math.radians(target_meta['lat'])
                target_lon_rad = math.radians(target_meta['lon'])
                radius_rad = SPATIAL_RADIUS_KM / 6371.0
                indices = self.spatial_tree.query_radius([[target_lat_rad, target_lon_rad]], r=radius_rad)[0]
                if len(indices) > 1:
                    for _ in range(5):
                        neg_idx = random.choice(indices)
                        cand_gid = self.gmap_ids_list[neg_idx]
                        if cand_gid != target_gid:
                            neg_gid = cand_gid
                            break
            
            # --- 策略 B: GNNO (图结构) ---
            elif rand_val < 0.8 and context_gid is not None and neg_gid is None:
                neighbors = self.witg_graph.get(context_gid, Counter())
                if neighbors:
                    top_neighbors = [g for g, c in neighbors.most_common(10)]
                    for _ in range(5):
                        cand_gid = random.choice(top_neighbors)
                        if cand_gid != target_gid and cand_gid in self.gmap2code:
                            neg_gid = cand_gid
                            break

            # --- 策略 C: 类别难负 ---
            if neg_gid is None and target_meta['cats']:
                for _ in range(10):
                    rand_gid = random.choice(self.gmap_ids_list)
                    if rand_gid == target_gid: continue
                    cand_meta = self.poi_meta[rand_gid]
                    if not target_meta['cats'].isdisjoint(cand_meta['cats']):
                        neg_gid = rand_gid
                        break

        if neg_gid is None:
            neg_gid = self._random_negative(exclude_gid=target_gid)

        return self.gmap2code[neg_gid]

    def _random_negative(self, exclude_gid=None):
        while True:
            gid = random.choice(self.gmap_ids_list)
            if gid != exclude_gid: return gid

    def enhance_prompt(self, history_gids, original_prompt):
        cat_counter = Counter()
        for h_gid in history_gids:
            if h_gid in self.poi_meta:
                cat_counter.update(self.poi_meta[h_gid]['cats'])
        
        if cat_counter:
            top_cats = [c for c, _ in cat_counter.most_common(3)]
            intent_str = ", ".join(top_cats)
            insertion = f"User history reflects preference for [{intent_str}]. Considering sequential patterns, "
            new_prompt = original_prompt.replace("Predict Next POI:", insertion + "Predict Next POI:")
            return new_prompt
        return original_prompt

    # [修改] 增加 time_desc 参数，支持 6.1 时空上下文
    def get_coin_prompts(self, input_codes, time_desc):
        hist_str = " -> ".join(input_codes)
        t1, t2 = random.sample(INSTRUCTION_TEMPLATES, 2)
        # 填充模板中的 {time} 和 {hist}
        return t1.format(hist=hist_str, time=time_desc), t2.format(hist=hist_str, time=time_desc)

    # [新增] 6.1 时空上下文: 解析时间戳
    def get_time_desc(self, timestamp):
        """
        Unix Timestamp (ms) -> Natural Language
        e.g. "Friday Evening", "Sunday Morning"
        """
        try:
            dt = datetime.fromtimestamp(timestamp / 1000.0)
            day_str = dt.strftime("%A") # Monday, Tuesday...
            hour = dt.hour
            
            if 5 <= hour < 12: period = "Morning"
            elif 12 <= hour < 17: period = "Afternoon"
            elif 17 <= hour < 22: period = "Evening"
            else: period = "Late Night"
            
            return f"{day_str} {period}"
        except:
            return "Unknown Time"

    # [新增] 6.2 思维链 (CoT) Completion 构造
    def construct_cot_completion(self, target_gid, time_desc):
        """
        Format: Reasoning: User likes [Category], usually visits at [Time]. -> Target: [ID]
        """
        target_code = self.gmap2code[target_gid]
        target_meta = self.poi_meta.get(target_gid, {})
        
        # 获取类别信息
        cats = list(target_meta.get('cats', []))
        main_cat = cats[0] if cats else "Unknown Category"
        
        # 构造推理过程
        reasoning = f"Reasoning: User sequential pattern implies intent for [{main_cat}]. Context matches {time_desc}."
        
        # 拼接最终输出
        return f"{reasoning} -> Target: {target_code}"

    # ------------------------------------------------------------------------
    # 5. 主流程
    # ------------------------------------------------------------------------
    def run_pipeline(self):
        self.load_data()
        
        print("🔄 Phase 3: Processing Reviews & Building Sequences...")
        user_history = defaultdict(list)
        
        for r_file in REVIEW_FILES:
            if not os.path.exists(r_file): continue
            with gzip.open(r_file, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        if gid in self.gmap2code:
                            user_history[d['user_id']].append((d['time'], gid))
                            self.global_poi_counter[gid] += 1
                    except: continue

        self.build_witg(user_history)

        print("⚖️ Calculating IPS Weights...")
        total_interactions = sum(self.global_poi_counter.values())
        if total_interactions == 0:
             raise ValueError("No interactions found!")

        ips_map = {}
        for gid, count in self.global_poi_counter.items():
            prob = count / total_interactions
            weight = 1.0 / (math.pow(prob, IPS_POWER) + 1e-9)
            ips_map[gid] = weight
            
        w_values = list(ips_map.values())
        w_min, w_max = min(w_values), max(w_values)
        for gid in ips_map:
            ips_map[gid] = 0.5 + (ips_map[gid] - w_min) / (w_max - w_min) * 2.5

        print("🚀 Generating Enhanced SFT Samples (CoT + Context + Mask)...")
        sft_samples = []
        
        for uid, hist in tqdm(user_history.items()):
            hist.sort(key=lambda x: x[0])
            poi_seq = [x[1] for x in hist]
            time_seq = [x[0] for x in hist] # 记录时间序列
            
            if len(poi_seq) < MIN_HISTORY_LEN: continue
            
            L = len(poi_seq)
            for i in range(MIN_HISTORY_LEN, L):
                target_gid = poi_seq[i]
                context_gid = poi_seq[i-1]
                target_time = time_seq[i] # 目标时间
                
                # Reflect
                if target_gid in self.poi_meta and self.poi_meta[target_gid]['rating'] < MIN_RATING_THRESHOLD:
                    continue

                input_gids = poi_seq[max(0, i - MAX_HISTORY_LEN) : i]
                input_codes = [self.gmap2code[g] for g in input_gids]
                
                # [6.1] 获取自然语言时间描述
                time_desc = self.get_time_desc(target_time)
                
                # [3.3 & 6.1] CoIN Prompt Pair (传入 time_desc)
                prompt_a, prompt_b = self.get_coin_prompts(input_codes, time_desc)
                
                # Adjust (意图增强)
                prompt_a = self.enhance_prompt(input_gids, prompt_a)
                
                # [6.2] CoT Completion (Reasoning + Target)
                cot_completion = self.construct_cot_completion(target_gid, time_desc)
                
                # 难负挖掘
                neg_code = self.mine_hard_negative(target_gid, context_gid)
                
                # [课程学习 Mask]
                # 标记：我们有一个4层的ID结构。
                # CoT部分 + ID前2层 (Coarse) -> 标记为 1 (Learnable in Stage 1)
                # ID后2层 (Fine) -> 标记为 0 (Masked in Stage 1)
                # 由于这是文本生成，我们在 dataset 中记录 "Target ID String"，
                # Trainer 的 Collator 会根据这个字符串在 Tokenized 序列中的位置生成掩码。
                target_code_str = self.gmap2code[target_gid] # e.g. "12 34 56 78"
                
                sft_samples.append({
                    "prompt": prompt_a,
                    "prompt_augment": prompt_b,
                    "completion": cot_completion, # 包含推理过程
                    "negative_completion": neg_code,
                    "ips_weight": ips_map.get(target_gid, 1.0),
                    "target_gmap_id": target_gid,
                    # [新增] 用于课程学习的辅助字段
                    "raw_target_code": target_code_str 
                })

        print(f"🎉 Generated {len(sft_samples)} High-Quality Samples.")
        
        print(f"💾 Saving to {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, 'w') as f:
            for item in sft_samples:
                f.write(json.dumps(item) + "\n")
        
        print("✅ Data Engine Finished.")

# ================= 运行入口 =================
if __name__ == "__main__":
    engine = SFTDataEngine()
    engine.run_pipeline()