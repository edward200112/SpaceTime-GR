import sys
import os
import torch
import torch.nn.functional as F
import json

# --- [Fix] Add parent directory to system path so we can import 'models' ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    # Fallback if running from root
    from pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower

# ================= Configuration =================
# 🏆 Using your best GRPO model (Checkpoint-4000)
USER_CKPT = "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-10000/user_tower.bin"
# Using the frozen Item Tower from SFT
ITEM_CKPT = "/workspace/data/pinrec_ckpt_sft_final_v3/checkpoint-48000/item_tower.bin"

# Must match training config
MAX_VOCAB_SIZE = 150346 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10

class PinRecInference:
    def __init__(self):
        print(f"🚀 Initializing PinRec Inference Engine on {DEVICE}...")
        
        # 1. Config
        self.config = PinRecConfig()
        self.config.item_vocab_size = MAX_VOCAB_SIZE
        self.config.vocab_size = MAX_VOCAB_SIZE
        self.config.item_size = MAX_VOCAB_SIZE
        
        # 2. Load Item Tower & Build Index
        print("📦 Loading Item Tower & Building Index (This may take a moment)...")
        self.item_tower = ItemTower(self.config).to(DEVICE)
        self.item_tower.load_state_dict(torch.load(ITEM_CKPT, map_location=DEVICE))
        self.item_tower.eval()
        
        # Build full index
        self.index = self._build_index()
        
        # 3. Load User Tower
        print(f"👤 Loading User Tower from {USER_CKPT}...")
        self.user_tower = UserTower(self.config).to(DEVICE)
        self.user_tower.load_state_dict(torch.load(USER_CKPT, map_location=DEVICE))
        self.user_tower.eval()
        
        print("✅ Ready to recommend!")

    def _build_index(self):
        # Pre-compute all item embeddings
        batch_size = 2048
        all_embs = []
        with torch.no_grad():
            for i in range(0, MAX_VOCAB_SIZE, batch_size):
                end = min(i + batch_size, MAX_VOCAB_SIZE)
                ids = torch.arange(i, end, dtype=torch.long, device=DEVICE)
                embs = self.item_tower(ids)
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs)
        return torch.cat(all_embs, dim=0)

    def recommend(self, history_ids, target_action=1):
        """
        history_ids: List[int], user history item IDs
        target_action: int, 0=Click, 1=Save (What do you want to optimize for?)
        """
        # Preprocessing
        seq_len = len(history_ids)
        h_ids = torch.tensor([history_ids], dtype=torch.long, device=DEVICE) # [1, L]
        
        # Fake auxiliary features (In prod, use real values)
        h_acts = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE) # Assume history is all Saves
        h_deltas = torch.zeros((1, seq_len), dtype=torch.float, device=DEVICE) # Assume recent
        h_mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)
        
        # Condition
        # Tell the model: "I want items the user will [target_action]"
        t_act_input = torch.tensor([[target_action, target_action]], dtype=torch.long, device=DEVICE)
        t_delta_input = torch.zeros((1, 2), dtype=torch.float, device=DEVICE) 
        
        # Inference
        with torch.no_grad():
            # Item Tower encodes history
            flat_h_vecs = self.item_tower(h_ids.view(-1))
            h_vecs = flat_h_vecs.view(1, seq_len, -1)
            
            # User Tower generates interest vector
            user_preds = self.user_tower(h_vecs, h_acts, h_deltas, h_mask, t_act_input, t_delta_input)
            user_vec = F.normalize(user_preds[:, 0, :], p=2, dim=-1) # Take first head
            
            # Retrieval (Dot Product)
            scores = torch.matmul(user_vec, self.index.T).squeeze()
            
            # Top-K
            topk_scores, topk_ids = torch.topk(scores, k=TOP_K)
            
        return topk_ids.cpu().tolist(), topk_scores.cpu().tolist()

# --- Demo Main ---
if __name__ == "__main__":
    engine = PinRecInference()
    
    # Simulate a few user cases
    test_cases = [
        # Case A: User who likes items 100, 101, 102
        [100, 101, 102],
        # Case B: Cold start user (1 item)
        [5000],
        # Case C: Active user
        [10, 20, 30, 40, 50, 60, 70, 80]
    ]
    
    print("\n" + "="*50)
    for i, hist in enumerate(test_cases):
        print(f"\n🔍 User {i+1} History: {hist}")
        
        # Mode 1: Predict Save (Optimized for Repin)
        recs, scores = engine.recommend(hist, target_action=1)
        print(f"   👉 Recs (Predict Save): {recs}")
        
        # Mode 2: Predict Click (Optimized for CTR)
        recs_click, _ = engine.recommend(hist, target_action=0)
        print(f"   👉 Recs (Predict Click): {recs_click}")
        
        if recs != recs_click:
            print("   ✨ Outcome Conditioning Works! (Different recs for different goals)")
        else:
            print("   (Recs are same, user interest might be very strong/narrow)")
    print("="*50)