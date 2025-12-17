import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import Trainer

class CoINSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CoIN 参数
        self.contrastive_margin = 0.5
        self.beta = 0.1 # 对比 Loss 的权重

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs 包含:
        - input_ids: 正样本序列 (Prompt + Positive Completion)
        - attention_mask
        - labels
        - ips_weight: 来自 DataCollator 的额外字段 (我们将在 collator 中处理)
        - negative_input_ids: 负样本序列 (Prompt + Negative Completion) (需自定义处理传入)
        """
        
        # 1. 提取额外信息 (IPS 权重 和 负样本)
        # 注意：HuggingFace 的 Trainer 会自动移除它不认识的列。
        # 我们需要在 Dataset 中把这些信息 pack 进去，或者使用自定义 DataCollator。
        # 为了简化实现且不破坏 Trainer 逻辑，我们假设 DataCollator 已经处理好了 inputs 字典。
        
        ips_weights = inputs.pop("ips_weight", None) # [Batch]
        neg_input_ids = inputs.pop("negative_input_ids", None) 
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        
        # 2. 正向传播 (Positive Sample) -> NTP Loss
        # HuggingFace 模型 forward 默认返回 Causal LM Loss
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True # 需要 Hidden States 做对比学习
        )
        
        # 原始 CrossEntropy Loss (Scalar mean)
        # 我们需要手动应用 IPS Weight。
        # 标准 outputs.loss 是已经 mean 过的。为了加权，我们需要取出 logits 手动算。
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift for Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算 Token-level Loss (不 reduce)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # Mask 掉 padding (labels == -100)
        padding_mask = (shift_labels != -100).float()
        
        # 应用 IPS Weight (Sample-level)
        # ips_weights: [Batch] -> 扩展到 [Batch, Seq]
        if ips_weights is not None:
            # 确保在同一设备
            ips_weights = ips_weights.to(token_losses.device).view(-1, 1)
            final_token_losses = token_losses * padding_mask * ips_weights
        else:
            final_token_losses = token_losses * padding_mask
            
        ntp_loss = final_token_losses.sum() / padding_mask.sum()

        # 3. 对比学习 Loss (CoIN)
        contrastive_loss = 0.0
        if neg_input_ids is not None:
            # 获取 Positive 的 EOS token 之前的 hidden state (代表整个序列的语义)
            # 简单起见，取最后一个 hidden state
            # outputs.hidden_states 是一个 tuple，取最后一层 [-1]
            pos_hidden = outputs.hidden_states[-1] # [Batch, Seq, Dim]
            
            # 获取序列最后一个有效 token 的向量 (使用 attention_mask)
            # pos_last_idx = inputs["attention_mask"].sum(dim=1) - 1
            # pos_repr = pos_hidden[torch.arange(pos_hidden.size(0)), pos_last_idx]
            # 为了计算方便，取 mean pooling 或者 max pooling
            pos_repr = torch.mean(pos_hidden, dim=1) 

            # 前向传播 Negative Sample (No Gradients needed for negative encoder usually, but here we train single model)
            # 我们希望模型认为 Negative Sample 的概率低，或者其 Embedding 离 Positive 远
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )
            neg_hidden = neg_outputs.hidden_states[-1]
            neg_repr = torch.mean(neg_hidden, dim=1)
            
            # 计算 Cosine Similarity
            sim = F.cosine_similarity(pos_repr, neg_repr)
            
            # Loss = max(0, sim - margin)
            # 如果相似度 > margin，产生 Loss，强迫拉远
            contrastive_loss = torch.mean(torch.clamp(sim - self.contrastive_margin, min=0))

        # 4. 总 Loss
        total_loss = ntp_loss + self.beta * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss