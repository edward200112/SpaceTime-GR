import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
import math

class PinRecConfig:
    def __init__(self):
        # 基础配置
        self.base_model = "/workspace/Qwen2_5-1.5B-Instruct"
        self.embedding_dim = 1024       # 最终对齐的向量维度
        self.max_seq_len = 1024
        
        # --- Hash Embedding (PinRec Item Tower) ---
        self.num_hash_tables = 4        # 论文中使用多个小表组合
        self.hash_bucket_size = 50000   # 每个表的大小 (50k * 4 = 200k 参数，却能表示无限ID)
        self.item_content_dim = 768     # 假设有 BERT 预提取的 Content Embedding
        
        # --- Multi-Horizon (多 Token 预测) ---
        # 定义我们想预测的时间跨度，例如：立即、短期、长期
        self.horizons = ['immediate', 'short_term', 'long_term'] 
        
        # --- Outcome Conditioning (结果条件) ---
        # 模拟不同的业务目标
        self.outcomes = {'click': 0, 'save': 1, 'purchase': 2}
        
        # 训练参数
        self.temperature = 0.07         # InfoNCE 温度
        self.use_lora = True

# ==============================================================================
# 1. Item Tower: Hash Embedding + Content Fusion
#    对应论文：Learned ID Embeddings (Compositional) + OmniSage
# ==============================================================================
class HashItemEncoder(nn.Module):
    def __init__(self, config: PinRecConfig):
        super().__init__()
        self.config = config
        
        # 1. Compositional Hash Embeddings
        # 不存储巨大的 Embedding Table，而是用 k 个小表求和
        self.hash_tables = nn.ModuleList([
            nn.Embedding(config.hash_bucket_size, config.embedding_dim)
            for _ in range(config.num_hash_tables)
        ])
        
        # 2. Content Projection (OmniSage)
        # 将 BERT/ResNet 提取的物品特征投影到相同空间
        self.content_proj = nn.Linear(config.item_content_dim, config.embedding_dim)
        
        # 3. Fusion Layer
        self.fusion_norm = nn.LayerNorm(config.embedding_dim)
        
        # 初始化哈希盐值 (固定随机数)，用于将 ID 映射到不同 bucket
        # 注册为 buffer，不参与梯度更新，但随模型保存
        for i in range(config.num_hash_tables):
            self.register_buffer(f'hash_salt_{i}', torch.randint(0, 1000000, (1,)))

    def forward(self, item_ids, content_embeds=None):
        """
        item_ids: LongTensor [batch_size] - 原始 Integer ID
        content_embeds: Tensor [batch_size, content_dim] - 预训练的文本/图像特征
        """
        # A. Hash ID Embedding
        hash_embed = 0
        for i, table in enumerate(self.hash_tables):
            salt = getattr(self, f'hash_salt_{i}')
            # 简单的多重哈希逻辑： (ID * Salt + i) % Bucket
            # 论文中可能使用了更复杂的 murmurhash，但这里线性同余足够模拟效果
            hashed_idx = (item_ids * salt + i) % self.config.hash_bucket_size
            hash_embed = hash_embed + table(hashed_idx)
            
        # B. Content Embedding (如果有)
        if content_embeds is not None:
            c_embed = self.content_proj(content_embeds)
            final_embed = hash_embed + c_embed
        else:
            final_embed = hash_embed
            
        return self.fusion_norm(final_embed)

# ==============================================================================
# 2. User Tower: LLM Backbone + Outcome Injector + Multi-Head Projector
# ==============================================================================
class PinRecLLM(nn.Module):
    def __init__(self, config: PinRecConfig):
        super().__init__()
        self.config = config
        
        # --- Backbone ---
        print(f"Loading Backbone: {config.base_model}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        if config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, # 注意这里不是 CAUSAL_LM
                r=64, lora_alpha=128, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # --- A. Outcome Condition Embeddings ---
        # 这是一个可学习的 Embedding，用于表示 [Goal: Click], [Goal: Save]
        # 我们将在 forward 中把它 prepend 到 input_embeds 前面
        self.outcome_embeddings = nn.Embedding(len(config.outcomes), self.llm.config.hidden_size)
        
        # --- B. Multi-Horizon Projectors (多时间步预测) ---
        # 论文核心：并行生成未来不同时间步的表征
        # 我们不只是取最后一个 token 映射一次，而是用不同的 Head 映射成不同的向量
        self.heads = nn.ModuleDict({
            horizon: nn.Sequential(
                nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.config.hidden_size, config.embedding_dim) # 压缩到 1024
            ) for horizon in config.horizons
        })
        
    def forward(self, input_ids, attention_mask, outcome_ids):
        """
        input_ids: [B, Seq] - 用户历史 Prompt
        outcome_ids: [B] - 想要预测的目标类型 (0=Click, 1=Save...)
        """
        # 1. 获取 LLM 的原生 Word Embeddings
        # 我们需要介入 Embedding 层来插入 Condition Token
        inputs_embeds = self.llm.get_input_embeddings()(input_ids) # [B, Seq, H]
        
        # 2. 获取 Outcome Condition Embedding
        cond_embeds = self.outcome_embeddings(outcome_ids).unsqueeze(1) # [B, 1, H]
        
        # 3. 拼接: [Condition] + [History]
        # 这样 Attention 机制会让整个序列都“看到”这个 Condition
        inputs_embeds = torch.cat([cond_embeds, inputs_embeds], dim=1)
        
        # 扩展 Attention Mask (为 Condition 腾出位置)
        batch_size = input_ids.shape[0]
        ones = torch.ones((batch_size, 1), device=input_ids.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([ones, attention_mask], dim=1)
        
        # 4. LLM Forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 5. Pooling (取最后一个有效 Token)
        # 注意：因为前面加了 1 个 token，长度变了，seq_len 要 +1
        # 但我们通常只需要取序列实际结束的位置
        last_hidden_state = outputs.hidden_states[-1] # [B, Seq+1, H]
        
        # 找到每个 sample 的最后一个真实 token 位置
        # mask sum 是真实长度，index = sum - 1. 
        # 但因为我们在前面 concat 了 1 位，所以 index = (sum + 1) - 1 = sum
        seq_indices = attention_mask.sum(dim=1) - 1
        # 保护边界
        seq_indices = seq_indices.clamp(max=last_hidden_state.shape[1]-1)
        
        final_token_state = last_hidden_state[torch.arange(batch_size), seq_indices] # [B, H]
        
        # 6. Multi-Horizon Projection
        # 并行输出多个向量：user_vec_immediate, user_vec_short, user_vec_long
        user_vectors = {}
        for horizon, head in self.heads.items():
            vec = head(final_token_state)
            user_vectors[horizon] = F.normalize(vec, p=2, dim=1) # L2 Normalize 是 Dense Retrieval 标配
            
        return user_vectors