import torch
import torch.nn as nn
import os
import sys

# --- 1. Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from models.pinrec_ultimate_v2 import PinRecConfig, UserTower
except ImportError:
    print("❌ Cannot import model. Check path.")
    sys.exit(1)

# ================= Config =================
USER_CKPT = "/workspace/data/pinrec_ckpt_grpo_aggressive/checkpoint-10000/user_tower.bin"
OUTPUT_ONNX = "/workspace/data/pinrec_user_tower.onnx"
DEVICE = "cpu" # 必须 CPU

# ==========================================
# 🛡️ Wrapper: 隔离一切复杂性
# ==========================================
class ExportWrapper(nn.Module):
    """
    这个 Wrapper 的作用是'清洗' UserTower 的输出。
    HuggingFace 模型默认会返回 (last_hidden_state, past_key_values, ...)。
    ONNX 导出器经常在处理 past_key_values 时崩溃 (unordered_map::at)。
    我们在这里强制只返回一个 Tensor，切断所有其他输出路径。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, h_vecs, h_acts, h_deltas, h_mask, t_act, t_delta):
        # 调用原始模型
        # UserTower 内部应该已经处理了输出选择，但为了保险，
        # 如果 UserTower 返回的是 tuple，我们在这里解包。
        output = self.model(h_vecs, h_acts, h_deltas, h_mask, t_act, t_delta)
        
        # 如果输出是 tuple/list (常见于 HF 模型)，只取第一个元素
        if isinstance(output, (tuple, list)):
            return output[0]
        elif hasattr(output, 'last_hidden_state'): # ModelOutput 对象
            return output.last_hidden_state
        return output

def export_model():
    print(f"🚀 Starting Sanitized Export to {OUTPUT_ONNX}...")
    
    # 1. Init Model
    config = PinRecConfig()
    config.item_vocab_size = 150346
    config.vocab_size = 150346
    config.item_size = 384

    raw_model = UserTower(config)
    
    # [关键配置 1] 强制关闭 KV Cache
    # 这是解决 unordered_map::at 的核心！ONNX Tracer 处理不了动态增长的 Cache
    if hasattr(raw_model, 'llm') and hasattr(raw_model.llm, 'config'):
        print("ℹ️  Disabling KV Cache & Forcing Eager Attention...")
        raw_model.llm.config.use_cache = False 
        raw_model.llm.config._attn_implementation = "eager"

    # Load weights
    state_dict = torch.load(USER_CKPT, map_location="cpu")
    raw_model.load_state_dict(state_dict)
    
    # Wrap model
    model = ExportWrapper(raw_model)
    model.to(DEVICE)
    model.eval()
    
    # 2. Dummy Inputs (1024 dim)
    # 你的 checkpoint 里的 Linear 层输入是 1024，必须匹配
    INPUT_DIM = 1024 
    print(f"ℹ️  Using Input Dim: {INPUT_DIM}")

    dummy_batch = 1
    dummy_seq = 10
    
    args = (
        torch.randn(dummy_batch, dummy_seq, INPUT_DIM),
        torch.ones(dummy_batch, dummy_seq, dtype=torch.long),
        torch.zeros(dummy_batch, dummy_seq, dtype=torch.float),
        torch.ones(dummy_batch, dummy_seq, dtype=torch.long),
        torch.ones(dummy_batch, 2, dtype=torch.long),
        torch.zeros(dummy_batch, 2, dtype=torch.float)
    )
    
    input_names = ["h_vecs", "h_acts", "h_deltas", "h_mask", "t_act", "t_delta"]
    output_names = ["user_embedding"]
    
    # 3. Export
    try:
        torch.onnx.export(
            model,
            args,
            OUTPUT_ONNX,
            input_names=input_names,
            output_names=output_names,
            opset_version=14, # 14 比较稳定
            do_constant_folding=True,
            # 暂时不加 dynamic_axes，先求跑通
            # dynamic_axes={...} 
        )
        print(f"✅ SUCCESS! Model exported to: {OUTPUT_ONNX}")
        
        # 4. Verify
        if os.path.exists(OUTPUT_ONNX):
            size = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
            print(f"🎉 File Size: {size:.2f} MB")
        else:
            print("❌ File not found after export!")
            
    except Exception as e:
        print(f"\n❌ Export Failed: {e}")
        # 如果还是失败，打印更详细的信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_model()