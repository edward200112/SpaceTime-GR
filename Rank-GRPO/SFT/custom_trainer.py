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
        self.coarse_stage_ratio = 0.1 

    # [优化] 更加鲁棒的 Pooling 方法
    def _masked_mean_pooling(self, hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float() # [B, S, 1]
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1) # [B, D]
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) # [B, 1]
        return sum_embeddings / sum_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 提取所有自定义字段
        ips_weights = inputs.pop("ips_weight", None)
        neg_input_ids = inputs.pop("negative_input_ids", None) 
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)
        hierarchy_mask = inputs.pop("hierarchy_mask", None) 

        # 2. 正向传播 (Prompt A)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True 
        )
        
        # 3. 计算 NTP Loss
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # 基础 Mask (去除 Padding)
        padding_mask = (shift_labels != -100).to(token_losses.dtype)
        
        # --- [课程学习逻辑] ---
        current_progress = getattr(model, 'curriculum_progress', 1.0)
        
        # 如果在第一阶段 (Coarse Stage) 且有层级 Mask
        if current_progress < self.coarse_stage_ratio and hierarchy_mask is not None:
            # 确保 hierarchy_mask 在正确的设备上
            if not isinstance(hierarchy_mask, torch.Tensor):
                hierarchy_mask = torch.tensor(hierarchy_mask, device=token_losses.device)
            else:
                hierarchy_mask = hierarchy_mask.to(token_losses.device)
            
            # Shift hierarchy_mask (对齐 Labels)
            shift_hierarchy = hierarchy_mask[..., 1:].contiguous().to(token_losses.dtype)
            
            # 最终 Mask = 非Padding AND 粗粒度允许
            # 如果 shift_hierarchy 某位是 0 (Masked)，乘积就是 0，该 Loss 被忽略
            final_mask = padding_mask * shift_hierarchy
        else:
            # 第二阶段或无 Mask：全量训练
            final_mask = padding_mask

        # --- [IPS 加权] ---
        if ips_weights is not None:
            if not isinstance(ips_weights, torch.Tensor):
                ips_weights = torch.tensor(ips_weights, device=token_losses.device)
            else:
                ips_weights = ips_weights.to(token_losses.device)
            
            ips_weights = ips_weights.view(-1, 1)
            weighted_losses = token_losses * final_mask * ips_weights
        else:
            weighted_losses = token_losses * final_mask
            
        ntp_loss = weighted_losses.sum() / (final_mask.sum() + 1e-9)

        # 4. CoIN 对比损失
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device)
        
        # 获取 Anchor 表征
        pos_hidden = outputs.hidden_states[-1]
        pos_repr = self._masked_mean_pooling(pos_hidden, inputs["attention_mask"])

        # A. 一致性 Loss
        if aug_input_ids is not None:
            aug_outputs = model(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                output_hidden_states=True
            )
            aug_hidden = aug_outputs.hidden_states[-1]
            aug_repr = self._masked_mean_pooling(aug_hidden, aug_attention_mask)
            consistency_loss = 1.0 - F.cosine_similarity(pos_repr, aug_repr).mean()
            contrastive_loss += consistency_loss

        # B. 负样本 Loss
        if neg_input_ids is not None:
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )
            neg_hidden = neg_outputs.hidden_states[-1]
            neg_repr = self._masked_mean_pooling(neg_hidden, neg_attention_mask)
            
            sim_neg = F.cosine_similarity(pos_repr, neg_repr)
            neg_loss = torch.mean(torch.clamp(sim_neg - self.contrastive_margin, min=0))
            contrastive_loss += neg_loss

        # 5. 总 Loss
        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss