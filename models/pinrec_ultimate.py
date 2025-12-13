import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np  # 必须导入 numpy

class PinRecConfig:
    # 路径配置
    base_model = "/workspace/Qwen2_5-1.5B-Instruct"
    content_feat_path = "/workspace/data/processed_pinrec/item_content_feats.npy"
    
    # 维度
    embedding_dim = 1024       # 对齐维度
    content_dim = 384          # all-MiniLM-L6-v2 的输出维度
    hash_bucket_size = 50000
    num_hash_tables = 4
    
    # 时间桶配置 (PinRec 论文参数)
    max_time_delta = 365 * 24 * 3600 # 1年
    num_time_buckets = 128     # 时间离散化桶数
    
    # LoRA
    use_lora = True

class TimeDeltaEncoder(nn.Module):
    """
    论文核心：Log-Distance Temporal Quantization
    将秒级时间差映射到 Embedding
    """
    def __init__(self, num_buckets, hidden_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.time_embedding = nn.Embedding(num_buckets, hidden_dim)
        
    def _bucketize(self, delta_secs):
        # 避免 log(0)
        delta_secs = torch.clamp(delta_secs, min=1.0)
        # Log bucket: index = floor(scale * log(delta))
        # 简单的对数分桶策略
        val = torch.log(delta_secs)
        # 归一化到 [0, num_buckets-1]
        # 假设最大时间差是 e^18 (约 1.8年)
        scale = (self.num_buckets - 1) / 18.0
        indices = (val * scale).long()
        return torch.clamp(indices, 0, self.num_buckets - 1)
        
    def forward(self, delta_secs):
        indices = self._bucketize(delta_secs)
        return self.time_embedding(indices)

class ItemTower(nn.Module):
    """
    OmniSage Implementation: Hash ID + Pre-computed Content Features
    """
    def __init__(self, config):
        super().__init__()
        # 1. 加载预计算的内容特征 (Frozen, save GPU memory)
        print("Loading Item Content Features...")
        # 确保 numpy 已导入
        content_matrix = torch.from_numpy(np.load(config.content_feat_path))
        num_items, raw_dim = content_matrix.shape
        
        # 注册为 buffer (不更新原始 BERT 向量)
        self.register_buffer("content_feats", content_matrix)
        
        # 投影层 (把 BERT 向量映射到模型维度)
        self.content_proj = nn.Linear(raw_dim, config.embedding_dim)
        
        # 2. Hash Embeddings
        self.hash_tables = nn.ModuleList([
            nn.Embedding(config.hash_bucket_size, config.embedding_dim)
            for _ in range(config.num_hash_tables)
        ])
        for i in range(config.num_hash_tables):
            self.register_buffer(f'salt_{i}', torch.randint(0, 100000, (1,)))
            
        self.norm = nn.LayerNorm(config.embedding_dim)
        
    def forward(self, item_ids):
        # A. Content Part
        # item_ids -> content vectors
        raw_content = self.content_feats[item_ids] # [B, 384]
        c_embed = self.content_proj(raw_content)   # [B, 1024]
        
        # B. Hash Part
        h_embed = 0
        for i, tbl in enumerate(self.hash_tables):
            salt = getattr(self, f'salt_{i}')
            idx = (item_ids * salt + i) % tbl.num_embeddings
            h_embed += tbl(idx)
            
        # Fusion
        return self.norm(c_embed + h_embed)

class UserTower(nn.Module):
    """
    PinRec User Tower with Multi-Query Generation
    """
    def __init__(self, config):
        super().__init__()
        # Backbone
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16, # 模型权重为 bfloat16
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        if config.use_lora:
            peft = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                r=64, lora_alpha=128, 
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ]
            )
            self.llm = get_peft_model(self.llm, peft)
            
        hidden_size = self.llm.config.hidden_size
        
        # --- Innovation: Query Condition Embeddings ---
        self.outcome_embed = nn.Embedding(2, hidden_size) # 0=Click, 1=Save
        self.time_encoder = TimeDeltaEncoder(config.num_time_buckets, hidden_size)
        
        # Base Query Token (Learned constant)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Output Projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.embedding_dim)
        )
        
        self.item_to_llm = nn.Linear(config.embedding_dim, hidden_size)
        
    def forward(self, history_vecs, outcomes, time_deltas):
        """
        history_vecs: [B, Seq, Embed_Dim]
        outcomes: [B, Num_Queries]
        time_deltas: [B, Num_Queries]
        """
        batch_size = history_vecs.shape[0]
        
        # 1. Embed History to LLM dimension
        inputs_embeds = self.item_to_llm(history_vecs) # [B, Seq, H]
        
        # 2. Construct Query Tokens
        num_queries = outcomes.shape[1]
        
        # Base Query: [B, NQ, H]
        queries = self.query_token.expand(batch_size, num_queries, -1)
        
        # Add Conditions
        cond_outcome = self.outcome_embed(outcomes) # [B, NQ, H]
        cond_time = self.time_encoder(time_deltas)  # [B, NQ, H]
        
        queries = queries + cond_outcome + cond_time
        
        # 3. Concat: [History, Query1, Query2]
        full_embeds = torch.cat([inputs_embeds, queries], dim=1) # [B, Seq+NQ, H]
        
        # 4. Attention Mask
        seq_len = inputs_embeds.shape[1]
        total_len = full_embeds.shape[1]
        attention_mask = torch.ones((batch_size, total_len), device=history_vecs.device)
        
        # --- 关键修改开始 ---
        # 显式转换输入数据类型以匹配 LLM 权重 (Fix Float vs BFloat16 error)
        # 获取 LLM 的 dtype (通常是 torch.bfloat16)
        target_dtype = self.llm.dtype 
        full_embeds = full_embeds.to(target_dtype)
        
        # 5. LLM Forward
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 6. Extract Query Outputs
        last_hidden = outputs.hidden_states[-1] # [B, Total, H]
        query_outputs = last_hidden[:, seq_len:, :] # [B, NQ, H]
        
        # 将输出转回 float32，因为 projector 默认是 float32，且为了保持输出嵌入的通用性
        query_outputs = query_outputs.to(torch.float32)
        # --- 关键修改结束 ---
        
        # 7. Project to Common Space
        user_vecs = self.projector(query_outputs) # [B, NQ, 1024]
        return F.normalize(user_vecs, p=2, dim=-1)