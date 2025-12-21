import sys
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

# --- 导入模型 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# =========================================================
# ⚙️ 路径配置 (自动适配)
# =========================================================
# 1. 目标：你要测的 GRPO Checkpoint
CHECKPOINT_PATH = "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-10000"

# 2. 借用：SFT Checkpoint (用于加载 Item Tower)
# 如果 GRPO 没存 item_tower.bin，代码会自动来这里找
SFT_CHECKPOINT_PATH = "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-48000"

TEST_DATA = "/workspace/data/processed/train_balanced_pinrec.jsonl" 
BATCH_SIZE = 128
TOP_K_EVAL = [1, 5, 10, 50]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BulletproofEvaluator:
    def __init__(self, checkpoint_path, sft_path):
        self.device = DEVICE
        print(f"🚀 Target Model (User): {checkpoint_path}")
        print(f"📦 Source Model (Item): {sft_path}")
        
        # --- 1. 智能加载 Item Tower ---
        # 优先在 GRPO 目录找，找不到去 SFT 目录找
        item_path = os.path.join(checkpoint_path, "item_tower.bin")
        if not os.path.exists(item_path):
            print(f"⚠️ item_tower.bin missing in GRPO. Falling back to SFT path...")
            item_path = os.path.join(sft_path, "item_tower.bin")
            
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"❌ Missing item_tower.bin in both paths!")
            
        print(f"✅ Loading Item Tower from: {item_path}")
        state_dict = torch.load(item_path, map_location='cpu')
        
        # 自动探测物理权重大小
        max_shape = 0
        for k, v in state_dict.items():
            if v.dim() == 2 and v.shape[0] > max_shape:
                max_shape = v.shape[0]
        
        self.physical_vocab_size = max_shape
        print(f"📏 Auto-Detected Vocab Size: {self.physical_vocab_size}")
        
        # 初始化 Config
        self.config = PinRecConfig()
        # 强制覆盖所有相关 size 参数
        if hasattr(self.config, 'item_vocab_size'): self.config.item_vocab_size = self.physical_vocab_size
        if hasattr(self.config, 'vocab_size'): self.config.vocab_size = self.physical_vocab_size
        if hasattr(self.config, 'item_size'): self.config.item_size = self.physical_vocab_size
        
        # 加载 Item Tower
        self.item_tower = ItemTower(self.config).to(self.device)
        self.item_tower.load_state_dict(state_dict, strict=False)
        self.item_tower.eval()
        
        # --- 2. 加载 User Tower (GRPO 训练的核心) ---
        self.user_tower = UserTower(self.config).to(self.device)
        user_path = os.path.join(checkpoint_path, "user_tower.bin")
        if os.path.exists(user_path):
            print(f"✅ Loading User Tower from: {user_path}")
            self.user_tower.load_state_dict(torch.load(user_path, map_location=self.device), strict=False)
        else:
            print(f"❌ User tower weights not found at {user_path}")
            exit(1)
            
        self.user_tower.eval()

    def _safe_id(self, input_id):
        # 动态取模，防止越界
        if input_id >= self.physical_vocab_size:
            return input_id % self.physical_vocab_size
        return input_id

    def build_item_index(self):
        print(f"🏗️ Building Item Index for {self.physical_vocab_size} items...")
        batch_size = 2048
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, self.physical_vocab_size, batch_size)):
                end = min(i + batch_size, self.physical_vocab_size)
                ids = torch.arange(i, end, dtype=torch.long, device=self.device)
                embs = self.item_tower(ids)
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.cpu())
        self.item_index = torch.cat(all_embs, dim=0).to(self.device)
        print(f"✅ Index Built: {self.item_index.shape}")

    def evaluate(self, test_file, num_samples=2000):
        print(f"🧪 Evaluating {num_samples} samples...")
        samples = []
        try:
            with open(test_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= num_samples: break
                    if line.strip(): samples.append(json.loads(line))
        except FileNotFoundError:
            print(f"❌ Test file not found: {test_file}")
            return
        
        metrics = {'global': defaultdict(int), 'click': defaultdict(int), 'save': defaultdict(int)}
        counts = {'global': 0, 'click': 0, 'save': 0}
        ranks = []
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(samples), BATCH_SIZE)):
                batch = samples[batch_start : batch_start + BATCH_SIZE]
                B = len(batch)
                
                # --- Collate ---
                max_len = max(len(s['history_ids']) for s in batch)
                h_ids = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                h_acts = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                h_deltas = torch.zeros((B, max_len), dtype=torch.float, device=self.device)
                h_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
                
                target_ids, target_acts, target_ids_2 = [], [], []
                
                for i, s in enumerate(batch):
                    safe_hist = [self._safe_id(x) for x in s['history_ids']]
                    h_ids[i, :len(safe_hist)] = torch.tensor(safe_hist, device=self.device)
                    h_acts[i, :len(safe_hist)] = torch.tensor(s['history_acts'], device=self.device)
                    h_deltas[i, :len(safe_hist)] = torch.tensor(s['history_deltas'], device=self.device)
                    h_mask[i, :len(safe_hist)] = 1
                    target_ids.append(self._safe_id(s['target_1']['id']))
                    target_acts.append(s['target_1']['act'])
                    target_ids_2.append(self._safe_id(s['target_2']['id']))

                flat_h_ids = h_ids.view(-1)
                flat_h_vecs = self.item_tower(flat_h_ids)
                h_vecs = flat_h_vecs.view(B, max_len, -1)
                
                # Outcome Conditioning Inputs
                t_acts_tensor = torch.tensor(target_acts, dtype=torch.long, device=self.device).unsqueeze(1)
                t_acts_input = torch.cat([t_acts_tensor, t_acts_tensor], dim=1)
                t_deltas_tensor = torch.zeros((B, 2), dtype=torch.float, device=self.device)
                
                # User Tower Forward
                user_preds = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, t_acts_input, t_deltas_tensor)
                user_vec = F.normalize(user_preds[:, 0, :], p=2, dim=-1)
                
                # Retrieval
                scores = torch.matmul(user_vec, self.item_index.T)
                
                for i in range(B):
                    truth = target_ids[i]
                    truth_2 = target_ids_2[i]
                    act = target_acts[i]
                    
                    # Rank
                    truth_score = scores[i, truth].item()
                    rank = (scores[i] > truth_score).sum().item() + 1
                    ranks.append(rank)
                    
                    # Recall (Relaxed)
                    top_k = torch.topk(scores[i], k=max(TOP_K_EVAL)).indices.tolist()
                    preds_set = set(top_k)
                    
                    for k in TOP_K_EVAL:
                        hit = (truth in preds_set) or (truth_2 in preds_set)
                        v = 1 if hit else 0
                        metrics['global'][k] += v
                        if act == 1: metrics['save'][k] += v
                        else: metrics['click'][k] += v
                    
                    counts['global'] += 1
                    if act == 1: counts['save'] += 1
                    else: counts['click'] += 1

        print("\n" + "="*50)
        print(f"📊 EVAL REPORT: {os.path.basename(CHECKPOINT_PATH)}")
        print("="*50)
        print(f"Physical Vocab Size: {self.physical_vocab_size}")
        print(f"Avg Rank: {np.mean(ranks):.1f}")
        
        def show(name, key):
            n = counts[key]
            if n == 0: return
            print(f"\n🔹 {name} (N={n})")
            for k in TOP_K_EVAL:
                print(f"   Recall@{k:<2}: {metrics[key][k]/n:.2%}")

        show("Global", 'global')
        show("Click", 'click')
        show("Save", 'save')
        print("="*50)

if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_PATH):
        evaluator = BulletproofEvaluator(CHECKPOINT_PATH, SFT_CHECKPOINT_PATH)
        evaluator.build_item_index()
        evaluator.evaluate(TEST_DATA, num_samples=2000)
    else:
        print(f"❌ Target Checkpoint not found: {CHECKPOINT_PATH}")