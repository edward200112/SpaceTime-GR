import torch
import torch.nn as nn

class SASRec(torch.nn.Module):
    def __init__(self, item_num, geo_num, args):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.geo_num = geo_num

        self.item_emb = nn.Embedding(self.item_num + 1, args.embed_dim, padding_idx=0)
        self.geo_emb  = nn.Embedding(self.geo_num  + 1, args.embed_dim, padding_idx=0)

        self.pos_emb = nn.Embedding(args.max_len, args.embed_dim)
        self.emb_dropout = nn.Dropout(p=args.dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.embed_dim, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.embed_dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(args.embed_dim, args.num_heads, args.dropout))
            self.forward_layernorms.append(nn.LayerNorm(args.embed_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.embed_dim, args.dropout))

    def log2feats(self, item_seqs, geo_seqs):
        # item_seqs, geo_seqs: [B, L]
        seqs = self.item_emb(item_seqs) + self.geo_emb(geo_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        B, L = item_seqs.size()
        positions = torch.arange(L, device=item_seqs.device).unsqueeze(0).expand(B, L)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (item_seqs == 0)  # padding mask
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        tl = L
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=item_seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)        # [L,B,H]
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)        # [B,L,H]

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        return self.last_layernorm(seqs)

    def forward(self, log_items, log_geos, pos_items, neg_items):
        log_feats = self.log2feats(log_items, log_geos)
        pos_embs = self.item_emb(pos_items)
        neg_embs = self.item_emb(neg_items)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def predict_full(self, log_items, log_geos):
        log_feats = self.log2feats(log_items, log_geos)
        final_feat = log_feats[:, -1, :]
        logits = torch.matmul(final_feat, self.item_emb.weight.t())
        return logits
