import json
import gzip
import pandas as pd
import numpy as np
import os
import math
import random
import gc
import multiprocessing
from datetime import datetime
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.neighbors import BallTree

# ================= 核心路径配置 =================
RAW_DATA_DIR = "/workspace/data/GoogleRAW"

# 地区定义
REGION_CONFIG = [
    {'file': 'review-California.json.gz', 'meta': 'meta-California.json.gz', 'name': 'California'},
    {'file': 'review-New_York.json.gz',   'meta': 'meta-New_York.json.gz',   'name': 'New_York'},
    {'file': 'review-New_Mexico.json.gz', 'meta': 'meta-New_Mexico.json.gz', 'name': 'New_Mexico'},
    {'file': 'review-Pennsylvania.json.gz','meta': 'meta-Pennsylvania.json.gz','name': 'Pennsylvania'}
]

REVIEW_FILES = [os.path.join(RAW_DATA_DIR, r['file']) for r in REGION_CONFIG]
META_FILES = [os.path.join(RAW_DATA_DIR, r['meta']) for r in REGION_CONFIG]

ID_MAPPING_FILE = "./poi_semantic_ids.csv"
OUTPUT_DIR = "./SFT/sft_data"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_balanced_train.jsonl")

# ================= 超参数 =================
MIN_HISTORY_LEN = 3 
MAX_HISTORY_LEN = 20 
IPS_POWER = 0.5 
SPATIAL_RADIUS_KM = 0.5 
CATEGORY_DIST_THRESHOLD = 50.0 
MIN_RATING_THRESHOLD = 2.0

# [均衡采样] 每个地区最多保留 30万 条
TARGET_SAMPLES_PER_REGION = 300000 

INSTRUCTION_TEMPLATES = [
    "Current Time: {time}. Location Constraint: 5km.\nUser History: {hist}\nPredict Next POI:",
    "Context: {time}, within 5km radius.\nTrajectory: {hist}.\nRecommend the next location.",
    "Given the user's past visits: {hist}.\nTime: {time}.\nConstraint: 5km.\nWhere will they go next?",
    "Sequence: {hist}.\nEnvironment: {time}, nearby (5km).\nNext stop prediction:",
]

# [全局变量]
GLOBAL_ENGINE = None

def _worker_process_batch(user_batch):
    engine = GLOBAL_ENGINE
    local_results = []
    
    for uid, hist in user_batch:
        poi_seq = [x[1] for x in hist]
        time_seq = [x[0] for x in hist]
        
        if len(poi_seq) < MIN_HISTORY_LEN: continue
        
        L = len(poi_seq)
        for i in range(MIN_HISTORY_LEN, L):
            target_gid = poi_seq[i]
            target_time = time_seq[i]
            context_gid = poi_seq[i-1]
            
            # 获取 Region
            region_name = engine.gid_to_region.get(target_gid, "Unknown")
            
            if target_gid in engine.poi_meta and engine.poi_meta[target_gid]['rating'] < MIN_RATING_THRESHOLD:
                continue

            input_gids = poi_seq[max(0, i - MAX_HISTORY_LEN) : i]
            input_codes = [engine.gmap2code[g] for g in input_gids]
            target_code_str = engine.gmap2code[target_gid]
            
            time_desc = engine.get_time_desc(target_time)
            prompt_a, prompt_b = engine.get_coin_prompts(input_codes, time_desc)
            prompt_a = engine.enhance_prompt(input_gids, prompt_a)
            cot_completion = engine.construct_cot_completion(target_gid, time_desc)
            neg_code = engine.mine_hard_negative(target_gid, context_gid)
            
            item = {
                "prompt": prompt_a,
                "prompt_augment": prompt_b,
                "completion": cot_completion,
                "negative_completion": neg_code,
                "ips_weight": engine.ips_map.get(target_gid, 1.0),
                "target_gmap_id": target_gid,
                "raw_target_code": target_code_str
            }
            local_results.append((region_name, json.dumps(item)))
            
    return local_results

class SFTDataEngine:
    def __init__(self):
        self.poi_meta = {} 
        self.gmap2code = {} 
        self.code2gmap = {} 
        self.spatial_tree = None 
        self.gmap_ids_list = [] 
        self.global_poi_counter = Counter()
        self.witg_graph = defaultdict(Counter)
        self._spatial_cache = {} 
        self.ips_map = {} 
        self.gid_to_region = {} 

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def fast_distance_check(self, lat1, lon1, lat2, lon2, threshold_km):
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lon1 - lon2)
        deg_threshold = threshold_km / 110.0
        if lat_diff > deg_threshold or lon_diff > deg_threshold: return True
        if lat_diff < deg_threshold * 0.1 and lon_diff < deg_threshold * 0.1: return False
        return self.haversine_distance(lat1, lon1, lat2, lon2) > threshold_km

    def load_data(self):
        print(f"📥 Phase 1: Loading ID Mappings...")
        if not os.path.exists(ID_MAPPING_FILE): raise FileNotFoundError("Missing ID mapping file.")
        id_df = pd.read_csv(ID_MAPPING_FILE)
        id_df['gmap_id'] = id_df['gmap_id'].astype(str)
        for _, row in tqdm(id_df.iterrows(), total=len(id_df)):
            code_str = f"{row['code_0']} {row['code_1']} {row['code_2']} {row['code_3']}"
            self.gmap2code[row['gmap_id']] = code_str
            self.code2gmap[code_str] = row['gmap_id']
        
        print("📥 Phase 2: Loading Metadata & Mapping Regions...")
        coords = [] 
        valid_gids = []
        for r_conf in REGION_CONFIG:
            meta_path = os.path.join(RAW_DATA_DIR, r_conf['meta'])
            if not os.path.exists(meta_path): continue
            print(f"   Reading Metadata for {r_conf['name']}...")
            with gzip.open(meta_path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        gid = d['gmap_id']
                        if gid in self.gmap2code:
                            self.gid_to_region[gid] = r_conf['name']
                            lat = d.get('latitude')
                            lon = d.get('longitude')
                            cats = d.get('category')
                            if lat and lon:
                                cat_set = set(cats) if isinstance(cats, list) else (set([cats]) if cats else set())
                                self.poi_meta[gid] = {
                                    'lat': float(lat), 'lon': float(lon),
                                    'cats': cat_set, 'name': d.get('name', 'Unknown'),
                                    'rating': d.get('avg_rating', 0.0)
                                }
                                coords.append([math.radians(lat), math.radians(lon)])
                                valid_gids.append(gid)
                    except: continue
                        
        if not coords: raise ValueError("No valid POI metadata loaded!")
        print(f"🏗️ Building Spatial BallTree for {len(coords)} POIs...")
        self.spatial_tree = BallTree(np.array(coords), metric='haversine')
        self.gmap_ids_list = valid_gids 

    def build_witg(self, user_history):
        print("🕸️ Building WITG Graph...")
        for uid, hist in tqdm(user_history.items()):
            if len(hist) < 2: continue
            for i in range(len(hist) - 1):
                self.witg_graph[hist[i][1]][hist[i+1][1]] += 1

    def mine_hard_negative(self, target_gid, context_gid=None):
        if target_gid not in self.poi_meta: return self._random_negative_code()
        target_meta = self.poi_meta[target_gid]
        rand_val = random.random()
        neg_gid = None
        
        if rand_val < 0.4: # Spatial
            if target_gid in self._spatial_cache:
                indices = self._spatial_cache[target_gid]
            else:
                target_lat_rad = math.radians(target_meta['lat'])
                target_lon_rad = math.radians(target_meta['lon'])
                radius_rad = SPATIAL_RADIUS_KM / 6371.0
                indices = self.spatial_tree.query_radius([[target_lat_rad, target_lon_rad]], r=radius_rad)[0]
                # [安全优化] 缓存限制降低到 10万，防止 OOM
                if len(self._spatial_cache) > 100000: self._spatial_cache.clear()
                self._spatial_cache[target_gid] = indices
            
            if len(indices) > 1:
                for _ in range(5):
                    neg_idx = np.random.choice(indices) 
                    cand_gid = self.gmap_ids_list[neg_idx]
                    if cand_gid != target_gid:
                        neg_gid = cand_gid
                        break
        
        elif rand_val < 0.8 and context_gid is not None and neg_gid is None: # GNNO
            neighbors = self.witg_graph.get(context_gid, Counter())
            if neighbors:
                top_neighbors = [g for g, c in neighbors.most_common(10)]
                for _ in range(5):
                    cand_gid = random.choice(top_neighbors)
                    if cand_gid != target_gid and cand_gid in self.gmap2code:
                        neg_gid = cand_gid
                        break

        if neg_gid is None and target_meta['cats']: # Category
            for _ in range(10):
                rand_gid = random.choice(self.gmap_ids_list)
                if rand_gid == target_gid: continue
                cand_meta = self.poi_meta[rand_gid]
                if not target_meta['cats'].isdisjoint(cand_meta['cats']):
                    is_far = self.fast_distance_check(
                        target_meta['lat'], target_meta['lon'],
                        cand_meta['lat'], cand_meta['lon'],
                        CATEGORY_DIST_THRESHOLD
                    )
                    if is_far:
                        neg_gid = rand_gid
                        break

        if neg_gid is None: neg_gid = self._random_negative(exclude_gid=target_gid)
        return self.gmap2code[neg_gid]

    def _random_negative(self, exclude_gid=None):
        while True:
            gid = random.choice(self.gmap_ids_list)
            if gid != exclude_gid: return gid

    def _random_negative_code(self):
        gid = random.choice(self.gmap_ids_list)
        return self.gmap2code[gid]

    def enhance_prompt(self, history_gids, original_prompt):
        cat_counter = Counter()
        for h_gid in history_gids:
            if h_gid in self.poi_meta:
                cat_counter.update(self.poi_meta[h_gid]['cats'])
        if cat_counter:
            top_cats = [c for c, _ in cat_counter.most_common(3)]
            intent_str = ", ".join(top_cats)
            return original_prompt.replace("Predict Next POI:", f"User history reflects preference for [{intent_str}]. Considering sequential patterns, Predict Next POI:")
        return original_prompt

    def get_coin_prompts(self, input_codes, time_desc):
        hist_str = " -> ".join(input_codes)
        t1, t2 = random.sample(INSTRUCTION_TEMPLATES, 2)
        return t1.format(hist=hist_str, time=time_desc), t2.format(hist=hist_str, time=time_desc)

    def get_time_desc(self, timestamp):
        try:
            dt = datetime.fromtimestamp(timestamp / 1000.0)
            day_str = dt.strftime("%A")
            hour = dt.hour
            if 5 <= hour < 12: period = "Morning"
            elif 12 <= hour < 17: period = "Afternoon"
            elif 17 <= hour < 22: period = "Evening"
            else: period = "Late Night"
            return f"{day_str} {period}"
        except: return "Unknown Time"

    def construct_cot_completion(self, target_gid, time_desc):
        target_code = self.gmap2code[target_gid]
        target_meta = self.poi_meta.get(target_gid, {})
        cats = list(target_meta.get('cats', []))
        main_cat = cats[0] if cats else "Unknown"
        reasoning = f"Reasoning: User sequential pattern implies intent for [{main_cat}]. Context matches {time_desc}."
        return f"{reasoning} -> Target: {target_code}"

    def run_pipeline(self):
        self.load_data()
        
        print("🔄 Phase 3: Processing Reviews...")
        user_history = defaultdict(list)
        for r_file in REVIEW_FILES:
            if not os.path.exists(r_file): continue
            print(f"   Reading {r_file}...")
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
        total = sum(self.global_poi_counter.values())
        if total == 0: raise ValueError("No interactions found!")
        
        raw_ips = {g: 1.0/(math.pow(c/total, IPS_POWER)+1e-9) for g, c in self.global_poi_counter.items()}
        w_vals = list(raw_ips.values())
        w_min, w_max = min(w_vals), max(w_vals)
        self.ips_map = {g: 0.5 + (raw_ips[g] - w_min) / (w_max - w_min) * 2.5 for g in raw_ips}

        print("🚀 Phase 4: Generating Samples & Balancing Regions (Safe Mode)...")
        
        temp_files = {}
        for r in REGION_CONFIG:
            fname = os.path.join(OUTPUT_DIR, f"temp_{r['name']}.jsonl")
            temp_files[r['name']] = open(fname, 'w')
        temp_files["Unknown"] = open(os.path.join(OUTPUT_DIR, "temp_Unknown.jsonl"), 'w')

        global GLOBAL_ENGINE
        GLOBAL_ENGINE = self
        
        # [内存优化] 删除大对象后再 Fork
        # user_history 如果不删，fork后虽然是CoW，但内存压力依然大
        # 为了多进程能处理，我们必须把 user_history 转成 list。
        print("📦 Preparing batch list...")
        user_items = list(user_history.items())
        print("🗑️ Freeing memory (user_history dict)...")
        del user_history 
        gc.collect()
        try: gc.freeze()
        except: pass

        # Shuffle users to mix workload
        random.shuffle(user_items)
        
        # [安全优化] 4 Workers, Batch 2000
        batch_size = 2000
        batches = [user_items[i:i + batch_size] for i in range(0, len(user_items), batch_size)]
        num_workers = 4 
        
        print(f"🔥 Spawning {num_workers} workers (Batch=2000)...")
        
        total_counts = defaultdict(int)

        with multiprocessing.Pool(processes=num_workers) as pool:
            for batch_result in tqdm(pool.imap_unordered(_worker_process_batch, batches), total=len(batches)):
                for region, line in batch_result:
                    if region in temp_files:
                        temp_files[region].write(line + "\n")
                        total_counts[region] += 1
                    else:
                        temp_files["Unknown"].write(line + "\n")
                        total_counts["Unknown"] += 1
                        
        for f in temp_files.values():
            f.close()
            
        print("\n📊 Raw Generation Complete. Region Counts:")
        for r, c in total_counts.items():
            print(f"   - {r}: {c}")

        print("\n⚖️ Phase 5: Balancing & Merging Regions...")
        
        if TARGET_SAMPLES_PER_REGION:
            target_n = TARGET_SAMPLES_PER_REGION
        else:
            counts_valid = [c for r, c in total_counts.items() if r != "Unknown" and c > 0]
            target_n = min(counts_valid) if counts_valid else 0
        
        print(f"   Target samples per region: {target_n}")

        final_samples = []
        for r_name in temp_files.keys():
            temp_path = os.path.join(OUTPUT_DIR, f"temp_{r_name}.jsonl")
            if not os.path.exists(temp_path): continue
            
            print(f"   Processing {r_name}...")
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            random.shuffle(lines)
            if len(lines) > target_n:
                final_samples.extend(lines[:target_n])
            else:
                final_samples.extend(lines)
            os.remove(temp_path)
            
        print(f"🎲 Final Global Shuffle of {len(final_samples)} samples...")
        random.shuffle(final_samples)
        
        print(f"💾 Saving Balanced Data to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            f.writelines(final_samples)
            
        print("✅ Data Engine Finished Successfully!")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass 
    engine = SFTDataEngine()
    engine.run_pipeline()