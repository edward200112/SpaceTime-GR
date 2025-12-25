import torch
import torch.nn as nn
import numpy as np

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = args.device

        # Embedding dimensions
        self.item_emb = nn.Embedding(self.item_num + 1, args.embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len, args.embed_dim)
        self.emb_dropout = nn.Dropout(p=args.dropout)

        # Transformer Blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.embed_dim, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.embed_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = nn.MultiheadAttention(args.embed_dim, args.num_heads, args.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.embed_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.embed_dim, args.dropout)
            self.forward_layers.append(new_fwd_layer)
    @torch.no_grad()
    def predict_candidates(self, input_ids: torch.LongTensor, candidate_ids: torch.LongTensor):
        """
        input_ids: [B, L]
        candidate_ids: [B, C]
        return: scores [B, C]
        """
        # 1) 得到 sequence hidden features: [B, L, H]
        # 常见 SASRec 实现有 log2feats()；如果你不是这个名字，下面我给了 fallback 提示
        feats = self.log2feats(input_ids)  # [B, L, H]

        # 2) 取最后一个位置的表征（因为你是 left pad，最后位就是最新）
        user_repr = feats[:, -1, :]  # [B, H]

        # 3) 取候选 item embedding: [B, C, H]
        # 常见实现是 item_emb 或 item_embedding
        if hasattr(self, "item_emb"):
            item_emb = self.item_emb(candidate_ids)
        elif hasattr(self, "item_embedding"):
            item_emb = self.item_embedding(candidate_ids)
        else:
            raise AttributeError("Cannot find item embedding layer: expected self.item_emb or self.item_embedding")

        # 4) dot product -> [B, C]
        scores = (user_repr.unsqueeze(1) * item_emb).sum(-1)
        return scores
    def log2feats(self, log_seqs):
        # log_seqs is tensor on device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # ✅ Fast torch positions on device (no numpy, no cpu->gpu copy)
        B, L = log_seqs.size()
        positions = torch.arange(L, device=log_seqs.device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)

        # Masking
        timeline_mask = (log_seqs == 0)  # BoolTensor
        seqs = seqs * (~timeline_mask.unsqueeze(-1))  # broadcast

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)

            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        log_feats = self.last_layernorm(seqs)
        return log_feats


    def forward(self, log_seqs, pos_seqs, neg_seqs):
        # Training Logic
        log_feats = self.log2feats(log_seqs) 
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict_full(self, log_seqs):
        """
        用于 Teacher 生成：计算所有 Item 的分数
        """
        log_feats = self.log2feats(log_seqs) 
        final_feat = log_feats[:, -1, :] # Take last step embedding (Batch, Dim)
        
        # item_emb.weight shape: (num_items+1, dim)
        # We want to skip padding (index 0) usually, but simple matmul is faster
        all_item_embs = self.item_emb.weight 
        
        # (Batch, Dim) x (Dim, Num_Items) -> (Batch, Num_Items)
        logits = torch.matmul(final_feat, all_item_embs.t())
        
        return logits