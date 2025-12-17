import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl

# [4.1.1] 课程学习回调函数
class CurriculumCallback(TrainerCallback):
    def __init__(self, coarse_ratio=0.3):
        self.coarse_ratio = coarse_ratio # 前 30% 的 step 只学粗粒度

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 计算当前进度
        progress = state.global_step / state.max_steps
        # 将进度存入 model config 或其他可访问的地方，供 compute_loss 使用
        if hasattr(kwargs['model'], 'curriculum_progress'):
            kwargs['model'].curriculum_progress = progress
        else:
            # 动态给 model 绑定属性
            setattr(kwargs['model'], 'curriculum_progress', progress)

class CoINSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_margin = 0.5
        self.lambda_coin = 0.1
        # [4.1.1] 课程学习阈值
        self.coarse_stage_ratio = 0.3 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 提取字段
        ips_weights = inputs.pop("ips_weight", None)
        neg_input_ids = inputs.pop("negative_input_ids", None)
        neg_attention_mask = inputs.pop("negative_attention_mask", None)
        aug_input_ids = inputs.pop("augment_input_ids", None)     # CoIN Prompt B
        aug_attention_mask = inputs.pop("augment_attention_mask", None) # CoIN Mask B
        
        # [4.1.1] 获取课程学习 Mask (由 DataCollator 预先计算好的层级 Mask)
        # sft_data_engine 需要生成这个 mask: [1, 1, 0, 0] 对应 [Region, City, District, Item]
        hierarchy_mask = inputs.pop("hierarchy_mask", None) 

        # 2. 正向传播
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True
        )
        
        # 3. 计算 NTP Loss (手动计算以支持 IPS 和 Curriculum)
        logits = outputs.logits
        labels = inputs["labels"]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())
        
        # Padding Mask
        padding_mask = (shift_labels != -100).float()
        
        # --- [4.1.1] 课程学习逻辑 ---
        # 检查当前训练进度
        current_progress = getattr(model, 'curriculum_progress', 1.0)
        
        # 如果处于 Coarse Stage (第一阶段) 且提供了层级 Mask
        if current_progress < self.coarse_stage_ratio and hierarchy_mask is not None:
            # hierarchy_mask: [Batch, SeqLen]
            # 这里的 Mask 应该是: 对于 ID 的前两位为 1, 后两位为 0
            # 我们需要对其进行 shift 以匹配 labels
            shift_hierarchy = hierarchy_mask[..., 1:].contiguous()
            
            # 只有粗粒度 Token 参与 Loss 计算
            token_mask = padding_mask * shift_hierarchy
        else:
            # 第二阶段 (细节精修): 全量 Token 参与
            token_mask = padding_mask

        # --- [4.1.2] IPS 加权 ---
        if ips_weights is not None:
            ips_weights = ips_weights.to(token_losses.device).view(-1, 1)
            weighted_losses = token_losses * token_mask * ips_weights
        else:
            weighted_losses = token_losses * token_mask
            
        ntp_loss = weighted_losses.sum() / (token_mask.sum() + 1e-9)

        # 4. 计算 CoIN 对比损失 (Contrastive Loss)
        # 3.3 节要求：语义等价的指令 (prompt vs prompt_augment) 应该有一致的 Hidden Representation
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device)
        
        # 我们需要在 collator 中把 prompt_augment 也传进来
        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)
        
        if aug_input_ids is not None:
            # A. 获取 Prompt A 的语义向量 (Positive A)
            pos_hidden = outputs.hidden_states[-1] 
            # Mean Pooling
            pos_mask = inputs["attention_mask"].unsqueeze(-1)
            pos_repr = (pos_hidden * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-9)
            
            # B. 获取 Prompt B (Augment) 的语义向量 (Positive B)
            aug_outputs = model(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                output_hidden_states=True
            )
            aug_hidden = aug_outputs.hidden_states[-1]
            aug_mask = aug_attention_mask.unsqueeze(-1)
            aug_repr = (aug_hidden * aug_mask).sum(dim=1) / (aug_mask.sum(dim=1) + 1e-9)
            
            # C. 这里的 CoIN 目标是最大化一致性 (Maximize Cosine Similarity)
            # Loss = 1 - CosineSim(A, B)
            consistency_loss = 1.0 - F.cosine_similarity(pos_repr, aug_repr).mean()
            
            # D. 同时保持 Item 负采样对比 (原逻辑)
            # 如果有 neg_input_ids，则还要推远 Negative
            item_contrast_loss = 0.0
            if neg_input_ids is not None:
                neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True)
                neg_repr = (neg_outputs.hidden_states[-1] * neg_attention_mask.unsqueeze(-1)).sum(dim=1) / (neg_attention_mask.sum(dim=1) + 1e-9)
                # Hinge Loss: push away negative
                item_contrast_loss = torch.mean(torch.clamp(F.cosine_similarity(pos_repr, neg_repr) - 0.5, min=0))
            
            contrastive_loss = consistency_loss + item_contrast_loss

        # 5. 总 Loss: L_total = L_SFT + lambda * L_InfoNCE
        total_loss = ntp_loss + self.lambda_coin * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss