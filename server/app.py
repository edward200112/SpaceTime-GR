import streamlit as st
import torch
import torch.nn.functional as F
import json
import os
import sys

# --- 1. 路径设置 (确保能导入 models) ---
# ✅ 正确：添加上一级目录 (项目根目录)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.pinrec_ultimate_v2 import PinRecConfig, ItemTower, UserTower
except ImportError:
    st.error("❌ 找不到 models 模块，请确保 app.py 在项目根目录或 inference 目录下运行。")

# ================= 配置 =================
USER_CKPT = "../data/pinrec_ckpt_grpo_aggressive/checkpoint-4000/user_tower.bin"
ITEM_CKPT = "../data/pinrec_ckpt_sft_final_v3/checkpoint-48000/item_tower.bin"
DATA_PATH = "../data/processed/train_balanced_pinrec.jsonl"
MAX_VOCAB_SIZE = 150346
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 资源加载函数 (带缓存) =================

@st.cache_resource
def load_metadata():
    """读取 jsonl 构建 ID -> Info 映射"""
    meta_dict = {}
    if not os.path.exists(DATA_PATH):
        return meta_dict
    
    with open(DATA_PATH, 'r') as f:
        # 只读取前 50000 行或者全量读取（取决于内存）
        # 这里为了演示，我们尝试读取一部分构建映射，或者全量
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 提取历史和 Target 的元数据
            # 假设数据里有 'target_1': {'id': 123, 'title': '...', 'image': '...'}
            # 如果你的数据里没有 title/image 字段，这里会使用默认值
            
            def add_item(item_obj):
                iid = item_obj.get('id')
                if iid:
                    # 如果数据里没有真实 url，用 placeholder 图片代替
                    img = item_obj.get('image', f"https://placehold.co/200x300?text=Item+{iid}")
                    title = item_obj.get('title', f"Item {iid}")
                    meta_dict[iid] = {"image": img, "title": title}

            add_item(data.get('target_1', {}))
            add_item(data.get('target_2', {}))
            for h in data.get('history_ids', []):
                 # 历史 ID 可能没有详情，只能暂存 ID
                 if h not in meta_dict:
                     meta_dict[h] = {"image": f"https://placehold.co/200x300?text=Hist+{h}", "title": f"History {h}"}
                     
    return meta_dict

@st.cache_resource
def load_models():
    """加载模型并构建索引"""
    config = PinRecConfig()
    config.item_vocab_size = MAX_VOCAB_SIZE
    config.vocab_size = MAX_VOCAB_SIZE
    
    # Load Item Tower
    item_tower = ItemTower(config).to(DEVICE)
    item_tower.load_state_dict(torch.load(ITEM_CKPT, map_location=DEVICE))
    item_tower.eval()
    
    # Build Index
    batch_size = 2048
    all_embs = []
    with torch.no_grad():
        for i in range(0, MAX_VOCAB_SIZE, batch_size):
            end = min(i + batch_size, MAX_VOCAB_SIZE)
            ids = torch.arange(i, end, dtype=torch.long, device=DEVICE)
            embs = item_tower(ids)
            embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs)
    item_index = torch.cat(all_embs, dim=0)
    
    # Load User Tower
    user_tower = UserTower(config).to(DEVICE)
    user_tower.load_state_dict(torch.load(USER_CKPT, map_location=DEVICE))
    user_tower.eval()
    
    return item_tower, user_tower, item_index

# ================= 页面逻辑 =================

st.set_page_config(page_title="PinRec Visual Demo", layout="wide")
st.title("📌 PinRec: Generative Recommendation Demo")
st.caption(f"Running on {DEVICE} | Model: GRPO Checkpoint-4000")

# 1. 加载资源
with st.spinner("Loading Models & Metadata..."):
    item_tower, user_tower, item_index = load_models()
    metadata = load_metadata()
st.success("System Ready!")

# 2. 侧边栏配置
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top K Recommendation", 5, 20, 10)
    user_input = st.text_input("User History IDs (comma separated)", "100, 101, 102")

# 3. 推理逻辑
if st.button("🚀 Generate Recommendation"):
    try:
        hist_ids = [int(x.strip()) for x in user_input.split(',')]
        seq_len = len(hist_ids)
        
        # 构造输入
        h_ids = torch.tensor([hist_ids], dtype=torch.long, device=DEVICE)
        h_acts = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)
        h_deltas = torch.zeros((1, seq_len), dtype=torch.float, device=DEVICE)
        h_mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)
        
        # --- 显示历史 ---
        st.subheader("📜 User History")
        cols = st.columns(min(len(hist_ids), 8))
        for idx, col in enumerate(cols):
            if idx < len(hist_ids):
                iid = hist_ids[idx]
                info = metadata.get(iid, {"image": "https://placehold.co/150", "title": f"Unknown {iid}"})
                with col:
                    st.image(info['image'], use_container_width=True)
                    st.caption(f"ID: {iid}")

        # --- 定义推荐函数 ---
        def get_recs(target_act):
            t_act_in = torch.tensor([[target_act, target_act]], dtype=torch.long, device=DEVICE)
            t_delta_in = torch.zeros((1, 2), dtype=torch.float, device=DEVICE)
            
            with torch.no_grad():
                flat_h_vecs = item_tower(h_ids.view(-1))
                h_vecs = flat_h_vecs.view(1, seq_len, -1)
                user_preds = user_tower(h_vecs, h_acts, h_deltas, h_mask, t_act_in, t_delta_in)
                user_vec = F.normalize(user_preds[:, 0, :], p=2, dim=-1)
                scores = torch.matmul(user_vec, item_index.T).squeeze()
                top_scores, top_ids = torch.topk(scores, k=top_k)
            return top_ids.cpu().tolist(), top_scores.cpu().tolist()

        # --- 获取两种结果 ---
        recs_save, _ = get_recs(1) # Save
        recs_click, _ = get_recs(0) # Click

        st.divider()

        # --- 展示结果 (左右分栏) ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❤️ Predicted SAVE (Repin)")
            st.info("Goal: Items the user is likely to collect.")
            c1_grid = st.columns(3) # 3列网格
            for i, iid in enumerate(recs_save):
                info = metadata.get(iid, {"image": "https://placehold.co/200x300", "title": f"Item {iid}"})
                with c1_grid[i % 3]:
                    st.image(info['image'], caption=f"{info['title']} ({iid})")

        with col2:
            st.subheader("👆 Predicted CLICK (CTR)")
            st.info("Goal: Items the user is likely to click.")
            c2_grid = st.columns(3)
            for i, iid in enumerate(recs_click):
                info = metadata.get(iid, {"image": "https://placehold.co/200x300", "title": f"Item {iid}"})
                with c2_grid[i % 3]:
                    st.image(info['image'], caption=f"{info['title']} ({iid})")

    except Exception as e:
        st.error(f"Error: {e}")