import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl

# [新增] 课程学习回调：用于更新模型的训练进度状态
class CurriculumCallback(TrainerCallback):
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 计算当前进度 (0.0 ~ 1.0)
        current_step = state.global_step
        max_steps = state.max_steps
        if max_steps > 0:
            progress = current_step / max_steps
        else:
            progress = 0.0
        
        # 将进度注入模型对象，供 compute_loss 使用
        if hasattr(kwargs['model'], 'curriculum_progress'):
            kwargs['model'].curriculum_progress = progress
        else:
            # 动态绑定属性
            setattr(kwargs['model'], 'curriculum_progress', progress)

class CoINSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CoIN 参数
        self.contrastive_margin = 0.5
        self.lambda_coin = 0.1 
        # 课程学习参数：前 30% 的步数只学粗粒度
        self.coarse_stage_ratio = 0.3 

    # [优化] 更加鲁棒的 Pooling 方法，解决 Padding 干扰问题
    def _masked_mean_pooling(self, hidden_state, attention_mask):
        """
        hidden_state: [Batch, Seq, Dim]
        attention_mask: [Batch, Seq]
        """
        mask_expanded = attention_mask.unsqueeze(-1).float() # [B, S, 1]
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1) # [B, D]
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) # [B, 1] 防止除零
        return sum_embeddings / sum_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs 扩展字段:
        - augment_input_ids: CoIN 增强指令 (Prompt B)
        - hierarchy_mask: 课程学习 Mask (1=Coarse, 0=Fine)
        """
        
        # 1. 提取所有自定义字段
        ips_weights = inputs.pop("ips_weight", None)
        
        # CoIN 相关
        neg_input_ids = inputs.pop("negative_input_ids", None) 
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        aug_input_ids = inputs.pop("augment_input_ids", None) # [新增] Prompt B
        aug_attention_mask = inputs.pop("augment_attention_mask", None)
        
        # 课程学习相关
        hierarchy_mask = inputs.pop("hierarchy_mask", None) 

        # 2. 正向传播 (Prompt A)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True 
        )
        
        # 3. 计算 NTP Loss (手动计算以应用 IPS 和 Curriculum)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # 基础 Mask (去除 Padding)
        padding_mask = (shift_labels != -100).float()
        
        # --- [课程学习逻辑] ---
        # 检查当前进度
        current_progress = getattr(model, 'curriculum_progress', 1.0)
        
        # 如果在第一阶段 (Coarse Stage) 且有层级 Mask
        if current_progress < self.coarse_stage_ratio and hierarchy_mask is not None:
            # hierarchy_mask 需要 shift 以对齐 labels
            # 假设 mask 是 [1, 1, 1, 0, 0] (Prompt+CoT+L1+L2=1, L3+L4=0)
            shift_hierarchy = hierarchy_mask[..., 1:].contiguous()
            
            # 最终 Mask = 非Padding AND 粗粒度允许
            final_mask = padding_mask * shift_hierarchy
        else:
            # 第二阶段或无 Mask：全量训练
            final_mask = padding_mask

        # --- [IPS 加权] ---
        if ips_weights is not None:
            ips_weights = ips_weights.to(token_losses.device).view(-1, 1)
            weighted_losses = token_losses * final_mask * ips_weights
        else:
            weighted_losses = token_losses * final_mask
            
        ntp_loss = weighted_losses.sum() / (final_mask.sum() + 1e-9)

        # 4. CoIN 对比损失 (Consistency + Negative)
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device)
        
        # 获取 Prompt A 的表征 (Anchor)
        # 使用优化的 Masked Mean Pooling
        pos_hidden = outputs.hidden_states[-1]
        pos_repr = self._masked_mean_pooling(pos_hidden, inputs["attention_mask"])

        # A. 一致性 Loss (Prompt A vs Prompt B)
        if aug_input_ids is not None:
            aug_outputs = model(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                output_hidden_states=True
            )
            aug_hidden = aug_outputs.hidden_states[-1]
            aug_repr = self._masked_mean_pooling(aug_hidden, aug_attention_mask)
            
            # Maximize Similarity => Minimize (1 - Cosine)
            consistency_loss = 1.0 - F.cosine_similarity(pos_repr, aug_repr).mean()
            contrastive_loss += consistency_loss

        # B. 负样本 Loss (Prompt A vs Negative Item)
        if neg_input_ids is not None:
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )
            neg_hidden = neg_outputs.hidden_states[-1]
            neg_repr = self._masked_mean_pooling(neg_hidden, neg_attention_mask)
            
            # Hinge Loss: Push away if similarity > margin
            sim_neg = F.cosine_similarity(pos_repr, neg_repr)
            neg_loss = torch.mean(torch.clamp(sim_neg - self.contrastive_margin, min=0))
            contrastive_loss += neg_loss

        # 5. 总 Loss
        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss