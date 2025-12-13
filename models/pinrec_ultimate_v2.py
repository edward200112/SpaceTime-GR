import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

class PinRecConfig:
    base_model = "/workspace/Qwen2_5-1.5B-Instruct"
    # 注意路径要改
    content_feat_path = "/workspace/data/processed_pinrec_v2/item_content_feats.npy"
    
    embedding_dim = 1024 
    content_dim = 384
    
    hash_bucket_size = 50000
    num_hash_tables = 2 # 减少一点减少显存
    
    # 时间桶
    num_time_buckets = 128
    
    use_lora = True

class TimeDeltaEncoder(nn.Module):
    def __init__(self, num_buckets, hidden_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.time_embedding = nn.Embedding(num_buckets, hidden_dim)
        
    def forward(self, delta_secs):
        # log(0) safety
        delta_secs = torch.clamp(delta_secs, min=1.0)
        val = torch.log(delta_secs)
        # Log-scale bucketization
        scale = (self.num_buckets - 1) / 18.0 # approx e^18 seconds max
        indices = (val * scale).long()
        indices = torch.clamp(indices, 0, self.num_buckets - 1)
        return self.time_embedding(indices)

class ItemTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("Loading Item Content Features...")
        content_matrix = torch.from_numpy(np.load(config.content_feat_path))
        self.register_buffer("content_feats", content_matrix)
        
        self.content_proj = nn.Linear(config.content_dim, config.embedding_dim)
        
        self.hash_tables = nn.ModuleList([
            nn.Embedding(config.hash_bucket_size, config.embedding_dim)
            for _ in range(config.num_hash_tables)
        ])
        for i in range(config.num_hash_tables):
            self.register_buffer(f'salt_{i}', torch.randint(0, 100000, (1,)))
            
        self.norm = nn.LayerNorm(config.embedding_dim)
        
    def forward(self, item_ids):
        # 0 is padding, but hash embedding handles 0 fine usually.
        # content_feats[0] should be valid.
        
        raw_content = self.content_feats(item_ids) if isinstance(self.content_feats, nn.Embedding) \
                      else self.content_feats[item_ids]
                      
        c_embed = self.content_proj(raw_content)
        
        h_embed = 0
        for i, tbl in enumerate(self.hash_tables):
            salt = getattr(self, f'salt_{i}')
            idx = (item_ids * salt + i) % tbl.num_embeddings
            h_embed += tbl(idx)
            
        return self.norm(c_embed + h_embed)

class UserTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Backbone
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        if config.use_lora:
            peft = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                r=32, lora_alpha=64, 
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.llm = get_peft_model(self.llm, peft)
            
        hidden_size = self.llm.config.hidden_size
        
        # --- Paper Alignment: History Feature Encoders ---
        # 1. Item Projector
        self.item_to_llm = nn.Linear(config.embedding_dim, hidden_size)
        
        # 2. Action Type Embedding (Click=0, Save=1, Padding=?? Let's use 0 for pad usually)
        self.action_embed = nn.Embedding(3, hidden_size) # 0, 1, 2(special)
        
        # 3. Time Delta Encoder (Shared for History and Query)
        self.time_encoder = TimeDeltaEncoder(config.num_time_buckets, hidden_size)
        
        # --- Query Side ---
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.outcome_embed = nn.Embedding(2, hidden_size)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.embedding_dim)
        )
        
    def forward(self, 
                history_vecs,      # [B, Seq, Embed_Dim] from ItemTower
                history_acts,      # [B, Seq]
                history_deltas,    # [B, Seq]
                history_mask,      # [B, Seq] 1=valid, 0=pad
                
                query_outcomes,    # [B, NQ]
                query_deltas       # [B, NQ]
               ):
        
        B, Seq, _ = history_vecs.shape
        NQ = query_outcomes.shape[1]
        dtype = self.llm.dtype # bfloat16
        
        # === 1. Construct History Input (PinnerFormer Style) ===
        # E_input = E_item + E_action + E_time
        
        feat_item = self.item_to_llm(history_vecs) # [B, Seq, H]
        feat_act  = self.action_embed(history_acts)
        feat_time = self.time_encoder(history_deltas)
        
        inputs_embeds = feat_item + feat_act + feat_time
        
        # === 2. Construct Query Input ===
        # Query = Token + E_outcome + E_time
        q_token = self.query_token.expand(B, NQ, -1)
        q_outcome = self.outcome_embed(query_outcomes)
        q_time = self.time_encoder(query_deltas)
        
        query_embeds = q_token + q_outcome + q_time
        
        # === 3. Concat ===
        full_embeds = torch.cat([inputs_embeds, query_embeds], dim=1) # [B, Seq+NQ, H]
        full_embeds = full_embeds.to(dtype) # Ensure bfloat16
        
        # === 4. Attention Mask ===
        # Mask 应该覆盖 History (padding=0) 和 Query (visible=1)
        # history_mask: [B, Seq]
        query_mask = torch.ones((B, NQ), device=history_mask.device)
        attention_mask = torch.cat([history_mask, query_mask], dim=1)
        
        # === 5. Forward ===
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # === 6. Extract Query Outputs ===
        last_hidden = outputs.hidden_states[-1]
        query_outputs = last_hidden[:, Seq:, :] # Only take the Query part
        
        # === 7. Project ===
        # Cast to float32 for projection and normalization (Stable training)
        query_outputs = query_outputs.float()
        user_vecs = self.projector(query_outputs)
        
        return F.normalize(user_vecs, p=2, dim=-1)