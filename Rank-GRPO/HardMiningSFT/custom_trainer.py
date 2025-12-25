import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer

class CoINSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CoIN 参数
        self.contrastive_margin = 0.5
        self.lambda_coin = 0.1 # 对比损失的权重

    def _masked_mean_pooling(self, hidden_state, attention_mask):
        """对 Sequence 进行 Mean Pooling 得到句向量"""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 提取自定义字段
        ips_weights = inputs.pop("ips_weight", None)
        neg_input_ids = inputs.pop("negative_input_ids", None) 
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)

        # 2. 正向传播 (Main Task: Next Token Prediction)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True # 需要 Hidden States 做对比学习
        )
        
        # 3. 计算 NTP Loss
        # 这里的 loss 是 batch 的 mean，我们需要根据 IPS 重新加权
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # Mask Padding
        padding_mask = (shift_labels != -100).to(token_losses.dtype)
        
        # IPS 加权
        if ips_weights is not None:
            if not isinstance(ips_weights, torch.Tensor):
                ips_weights = torch.tensor(ips_weights, device=token_losses.device)
            else:
                ips_weights = ips_weights.to(token_losses.device)
            
            ips_weights = ips_weights.view(-1, 1)
            weighted_losses = token_losses * padding_mask * ips_weights
        else:
            weighted_losses = token_losses * padding_mask
            
        ntp_loss = weighted_losses.sum() / (padding_mask.sum() + 1e-9)

        # 4. CoIN 对比损失 (Contrastive Loss)
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device)
        
        # 仅当有负样本或增强样本时计算
        if aug_input_ids is not None or neg_input_ids is not None:
            # 获取 Anchor (正样本) 的表征
            # 使用最后一层的 hidden state 进行 pooling
            pos_hidden = outputs.hidden_states[-1]
            pos_repr = self._masked_mean_pooling(pos_hidden, inputs["attention_mask"])

            # A. 一致性 Loss (Augmentation)
            if aug_input_ids is not None:
                aug_outputs = model(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    output_hidden_states=True
                )
                aug_hidden = aug_outputs.hidden_states[-1]
                aug_repr = self._masked_mean_pooling(aug_hidden, aug_attention_mask)
                # Cosine Distance: 1 - CosSim
                consistency_loss = 1.0 - F.cosine_similarity(pos_repr, aug_repr).mean()
                contrastive_loss += consistency_loss

            # B. 负样本 Loss (Hard Negative from SASRec)
            # 目标：让正样本表征和负样本表征的距离 至少大于 margin
            if neg_input_ids is not None:
                neg_outputs = model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True
                )
                neg_hidden = neg_outputs.hidden_states[-1]
                neg_repr = self._masked_mean_pooling(neg_hidden, neg_attention_mask)
                
                # 计算相似度
                sim_neg = F.cosine_similarity(pos_repr, neg_repr)
                # Hinge Loss: max(0, sim - margin)
                # 我们希望 sim 越小越好 (即距离越远越好)
                neg_loss = torch.mean(torch.clamp(sim_neg - self.contrastive_margin, min=0))
                contrastive_loss += neg_loss

        # 5. 总 Loss
        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss