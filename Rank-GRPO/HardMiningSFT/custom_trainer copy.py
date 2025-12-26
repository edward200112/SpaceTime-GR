import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer


class CoINSFTTrainer(SFTTrainer):
    """
    CoIN = NTP(Next token prediction) + lambda_coin * ContrastiveLoss
    这里新增：
      - 支持输入字段 coin_weight（shape [B] 或 list[float]），对每个样本的对比损失做加权
      - 支持不同 hard_level 对应不同 coin_weight（由外部 collator/preprocess 决定）
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CoIN 参数
        self.contrastive_margin = 0.5
        self.lambda_coin = 0.1  # base weight (会再乘以 coin_weight)

    def _masked_mean_pooling(self, hidden_state, attention_mask):
        """对 Sequence 做 Mean Pooling 得到句向量"""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _to_1d_tensor(self, x, device, dtype=torch.float32):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x.to(device=device, dtype=dtype)
        else:
            t = torch.tensor(x, device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.view(1)
        return t

    def _weighted_mean(self, vec: torch.Tensor, w: torch.Tensor):
        """
        vec: [B]
        w:   [B] (non-negative)
        return: scalar
        """
        w = torch.clamp(w, min=0.0)
        denom = torch.sum(w) + 1e-9
        return torch.sum(vec * w) / denom

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1) 提取自定义字段
        ips_weights = inputs.pop("ips_weight", None)
        neg_input_ids = inputs.pop("negative_input_ids", None)
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)

        # ✅ 新增：样本级对比权重（由 stage2 的 preprocess/collator 提供）
        coin_weight = inputs.pop("coin_weight", None)

        # 2) 正向传播 (Main Task: Next Token Prediction)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True
        )

        # 3) 计算 NTP Loss (IPS reweight)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        token_losses = token_losses.view(shift_labels.size())

        padding_mask = (shift_labels != -100).to(token_losses.dtype)

        if ips_weights is not None:
            ips_weights = self._to_1d_tensor(ips_weights, device=token_losses.device, dtype=token_losses.dtype)
            ips_weights = ips_weights.view(-1, 1)
            weighted_losses = token_losses * padding_mask * ips_weights
        else:
            weighted_losses = token_losses * padding_mask

        ntp_loss = weighted_losses.sum() / (padding_mask.sum() + 1e-9)

        # 4) CoIN 对比损失（样本级加权）
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device)

        # coin_weight default = 1
        if coin_weight is None:
            coin_w = torch.ones((inputs["input_ids"].size(0),), device=ntp_loss.device, dtype=torch.float32)
        else:
            coin_w = self._to_1d_tensor(coin_weight, device=ntp_loss.device, dtype=torch.float32)
            if coin_w.numel() != inputs["input_ids"].size(0):
                # 容错：如果维度不对，退化为全1
                coin_w = torch.ones((inputs["input_ids"].size(0),), device=ntp_loss.device, dtype=torch.float32)

        # 仅当有负样本或增强样本时计算
        if aug_input_ids is not None or neg_input_ids is not None:
            pos_hidden = outputs.hidden_states[-1]
            pos_repr = self._masked_mean_pooling(pos_hidden, inputs["attention_mask"])  # [B,H]

            total_terms = 0

            # A) 一致性 loss (Augmentation): per-sample
            if aug_input_ids is not None:
                aug_outputs = model(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    output_hidden_states=True
                )
                aug_hidden = aug_outputs.hidden_states[-1]
                aug_repr = self._masked_mean_pooling(aug_hidden, aug_attention_mask)

                # 1 - cos: [B]
                cons_vec = 1.0 - F.cosine_similarity(pos_repr, aug_repr, dim=-1)
                consistency_loss = self._weighted_mean(cons_vec, coin_w)
                contrastive_loss = contrastive_loss + consistency_loss
                total_terms += 1

            # B) 负样本 hinge loss: per-sample
            if neg_input_ids is not None:
                neg_outputs = model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True
                )
                neg_hidden = neg_outputs.hidden_states[-1]
                neg_repr = self._masked_mean_pooling(neg_hidden, neg_attention_mask)

                sim_neg = F.cosine_similarity(pos_repr, neg_repr, dim=-1)  # [B]
                hinge_vec = torch.clamp(sim_neg - self.contrastive_margin, min=0.0)  # [B]
                neg_loss = self._weighted_mean(hinge_vec, coin_w)
                contrastive_loss = contrastive_loss + neg_loss
                total_terms += 1

            # 平均一下 term，避免同时有 aug+neg 时 scale 变大
            if total_terms > 1:
                contrastive_loss = contrastive_loss / float(total_terms)

        # 5) 总 Loss：样本级权重已体现在 contrastive_loss 内
        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        return (total_loss, outputs) if return_outputs else total_loss
