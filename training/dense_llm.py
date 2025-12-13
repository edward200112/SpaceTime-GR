import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class HashEmbedding(nn.Module):
    """
    PinRec 风格的 Hash Embedding，用于节省显存
    """
    def __init__(self, num_items, embedding_dim, num_hash_tables=2, hash_bucket_size=50000):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_hash_tables = num_hash_tables
        self.hash_bucket_size = hash_bucket_size
        
        # 核心：只维护很小的 Embedding Tables
        self.tables = nn.ModuleList([
            nn.Embedding(hash_bucket_size, embedding_dim) 
            for _ in range(num_hash_tables)
        ])
        
        # 初始化
        for tbl in self.tables:
            nn.init.xavier_normal_(tbl.weight)
            
    def forward(self, item_ids):
        # item_ids: [batch_size]
        final_embed = 0
        
        # 模拟多重哈希
        # 在实际工程中，通常使用固定的 hash 函数。这里为了演示用简单的取模偏移
        # 也可以用 sklearn.utils.murmurhash3_32 等
        for i, table in enumerate(self.tables):
            # 简单的哈希逻辑： (ID + 盐值) % 桶大小
            hashed_idx = (item_ids + i * 1234567) % self.hash_bucket_size
            final_embed += table(hashed_idx)
            
        return final_embed # [batch_size, dim]

class DenseLLMRec(nn.Module):
    def __init__(self, base_model_path, item_embedding_dim=1024):
        super().__init__()
        # 1. 加载基座 (Qwen2.5)
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.config = self.llm.config
        
        # 2. 替换 Head
        # 原来的 lm_head 是 [hidden_size, vocab_size]
        # 我们需要 [hidden_size, item_embedding_dim]
        self.llm.lm_head = nn.Identity() # 移除原有的 head
        
        self.projector = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, item_embedding_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # 1. LLM Forward
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 2. 获取最后一个 Token 的隐状态 (EOS Token 或者是最后一个有效 Token)
        # 简单起见，取最后一个 token
        last_hidden_state = outputs.hidden_states[-1] # [batch, seq, hidden]
        
        # 提取序列最后一个位置的向量
        # 注意：这里需要根据 attention_mask 找到真正的最后一个 token
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_embeds = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
        # 3. 映射到 Item 空间
        user_vector = self.projector(last_token_embeds) # [batch, item_dim]
        
        # 4. 归一化 (PinRec 强调 Cosine Similarity)
        user_vector = F.normalize(user_vector, p=2, dim=1)
        
        return user_vector