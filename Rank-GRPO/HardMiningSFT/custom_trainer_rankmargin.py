import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer


class CoINSFTTrainerRankMargin(SFTTrainer):
    """
    Stage2 (B + C):
    - C: 使用标准 margin-ranking 形式： max(0, margin - (sim_pos - sim_neg))
         目标：sim_pos >= sim_neg + margin
    - B: 支持样本级 margin（inputs["coin_margin"]），也支持样本级对比权重（inputs["coin_weight"]）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 默认参数（若样本没提供 coin_margin/coin_weight，则用这些）
        self.default_margin = 0.20
        self.lambda_coin = 0.10

        # 对比损失组件权重（可按需改）
        self.use_consistency = True
        self.use_negative_rank = True
        self.consistency_weight = 1.0
        self.negative_weight = 1.0

    def _masked_mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_state: [B, L, H]
        attention_mask: [B, L]
        return: [B, H]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B,L,1]
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)  # [B,H]
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)  # [B,1]
        return sum_embeddings / denom

    def _to_device_tensor(self, x, device, dtype=None):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x.to(device)
        else:
            t = torch.tensor(x, device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # -------- 1) pop custom fields --------
        ips_weights = inputs.pop("ips_weight", None)

        neg_input_ids = inputs.pop("negative_input_ids", None)
        neg_attention_mask = inputs.pop("negative_attention_mask", None)

        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)

        # 新增：样本级 margin / weight（可选）
        coin_margin = inputs.pop("coin_margin", None)  # shape [B] or scalar/list
        coin_weight = inputs.pop("coin_weight", None)  # shape [B] or scalar/list

        # -------- 2) forward (main NTP) --------
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        # -------- 3) IPS-weighted NTP loss --------
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        padding_mask = (shift_labels != -100).to(token_losses.dtype)

        if ips_weights is not None:
            ips_weights = self._to_device_tensor(ips_weights, token_losses.device, dtype=token_losses.dtype)
            ips_weights = ips_weights.view(-1, 1)  # [B,1]
            weighted = token_losses * padding_mask * ips_weights
        else:
            weighted = token_losses * padding_mask

        ntp_loss = weighted.sum() / (padding_mask.sum() + 1e-9)

        # -------- 4) CoIN contrastive (B + C) --------
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device, dtype=ntp_loss.dtype)

        need_coin = (aug_input_ids is not None) or (neg_input_ids is not None)
        if need_coin:
            # [B,H] anchor
            pos_hidden = outputs.hidden_states[-1]
            pos_repr = self._masked_mean_pooling(pos_hidden, inputs["attention_mask"])

            B = pos_repr.size(0)

            # coin_margin: [B]
            if coin_margin is None:
                margin = torch.full((B,), float(self.default_margin), device=pos_repr.device, dtype=pos_repr.dtype)
            else:
                margin = self._to_device_tensor(coin_margin, pos_repr.device, dtype=pos_repr.dtype)
                if margin.numel() == 1:
                    margin = margin.view(1).repeat(B)
                else:
                    margin = margin.view(-1)
                    if margin.size(0) != B:
                        # 兜底：不匹配就用默认
                        margin = torch.full((B,), float(self.default_margin), device=pos_repr.device, dtype=pos_repr.dtype)

            # coin_weight: [B]
            if coin_weight is None:
                w = torch.ones((B,), device=pos_repr.device, dtype=pos_repr.dtype)
            else:
                w = self._to_device_tensor(coin_weight, pos_repr.device, dtype=pos_repr.dtype)
                if w.numel() == 1:
                    w = w.view(1).repeat(B)
                else:
                    w = w.view(-1)
                    if w.size(0) != B:
                        w = torch.ones((B,), device=pos_repr.device, dtype=pos_repr.dtype)

            # 4A) consistency: 1 - cos(pos, aug)
            sim_pos = None
            if self.use_consistency and aug_input_ids is not None:
                aug_outputs = model(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    output_hidden_states=True,
                )
                aug_hidden = aug_outputs.hidden_states[-1]
                aug_repr = self._masked_mean_pooling(aug_hidden, aug_attention_mask)

                sim_pos = F.cosine_similarity(pos_repr, aug_repr, dim=-1)  # [B]
                cons_vec = (1.0 - sim_pos)  # [B]
            else:
                cons_vec = torch.zeros((B,), device=pos_repr.device, dtype=pos_repr.dtype)

            # 4B) negative rank loss (C): max(0, margin - (sim_pos - sim_neg))
            if self.use_negative_rank and neg_input_ids is not None:
                neg_outputs = model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True,
                )
                neg_hidden = neg_outputs.hidden_states[-1]
                neg_repr = self._masked_mean_pooling(neg_hidden, neg_attention_mask)

                sim_neg = F.cosine_similarity(pos_repr, neg_repr, dim=-1)  # [B]

                # 如果没有 aug（sim_pos None），用 pos-pos 作为正对（cos=1）
                if sim_pos is None:
                    sim_pos_use = torch.ones_like(sim_neg)
                else:
                    sim_pos_use = sim_pos

                # 标准 margin ranking hinge
                # want: sim_pos - sim_neg >= margin
                neg_vec = torch.clamp(margin - (sim_pos_use - sim_neg), min=0.0)  # [B]
            else:
                neg_vec = torch.zeros((B,), device=pos_repr.device, dtype=pos_repr.dtype)

            coin_vec = self.consistency_weight * cons_vec + self.negative_weight * neg_vec  # [B]
            # 样本级权重（难度配比的入口）
            coin_vec = coin_vec * w

            contrastive_loss = coin_vec.mean()

        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        return (total_loss, outputs) if return_outputs else total_loss
