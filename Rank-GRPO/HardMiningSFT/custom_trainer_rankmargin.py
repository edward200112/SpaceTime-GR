# HardMiningSFT/custom_trainer_rankmargin.py
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import SFTTrainer


class CoINSFTTrainerRankMargin(SFTTrainer):
    """
    Stage2 (NTP + CoIN)

    ✅ CoIN negative-rank: self-positive margin ranking（不依赖 sim_pos=1）
       sim_pos = cos(pos_view1, pos_view2)
       sim_neg = cos(pos_view1, neg_repr)
       hinge  = ReLU(margin - (sim_pos - sim_neg))

    ✅ 关键：view-dropout 在 repr 上制造两视角（即使 base model dropout=0 也能工作）

    ✅ IPS 权重 batch 均值归一化：ips = ips / mean(ips)（只影响尺度，不改相对权重）

    - assistant-only pooling
    - debug 每 50 step 写 coin_debug.jsonl
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_margin = 0.20
        self.lambda_coin = 0.10

        # rank config
        self.use_consistency = False
        self.use_negative_rank = True

        # ✅ view-dropout 概率：用来制造 self-positive 两视角
        self.view_dropout = 0.10  # 建议 0.05~0.20

        self.consistency_weight = 1.0
        self.negative_weight = 1.0

        # debug logging
        self.debug_every = 50
        self._last_debug_step = -1
        self._debug_path = None
        self._debug_fh = None

    # -------------------------
    # Helpers
    # -------------------------
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

    def _masked_mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # [B,L,1]
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)       # [B,H]
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)               # [B,1]
        return sum_embeddings / denom

    def _masked_mean_pooling_from_start(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        start_idx,
    ) -> torch.Tensor:
        """
        只池化 start_idx 之后的 token（assistant token）
        start_idx: [B] / list / scalar / None
        """
        if start_idx is None:
            return self._masked_mean_pooling(hidden_state, attention_mask)

        device = hidden_state.device
        B, L, _ = hidden_state.shape

        if not isinstance(start_idx, torch.Tensor):
            start_idx = torch.tensor(start_idx, device=device)
        start_idx = start_idx.to(device=device).long().view(-1, 1)  # [B,1]
        if start_idx.numel() == 1:
            start_idx = start_idx.repeat(B, 1)

        pos = torch.arange(L, device=device).view(1, -1)            # [1,L]
        mask2 = attention_mask.long() * (pos >= start_idx).long()   # [B,L]
        mask2 = mask2.unsqueeze(-1).float()                         # [B,L,1]

        sum_embeddings = torch.sum(hidden_state * mask2, dim=1)     # [B,H]
        denom = torch.clamp(mask2.sum(dim=1), min=1e-9)             # [B,1]
        return sum_embeddings / denom

    def _ensure_debug_writer(self):
        if self._debug_fh is not None:
            return
        out_dir = getattr(self.args, "output_dir", None) or "."
        os.makedirs(out_dir, exist_ok=True)
        self._debug_path = os.path.join(out_dir, "coin_debug.jsonl")
        self._debug_fh = open(self._debug_path, "a", encoding="utf-8", buffering=1)

    def _get_lr_safe(self):
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                return float(self.optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            pass
        return None

    def _maybe_log_debug(self, payload: dict):
        gs = int(getattr(self.state, "global_step", 0) or 0)
        if gs <= 0:
            return
        if (gs % self.debug_every) != 0:
            return
        if gs == self._last_debug_step:
            return

        self._last_debug_step = gs
        self._ensure_debug_writer()

        payload = dict(payload)
        payload["ts"] = time.time()
        payload["global_step"] = gs
        payload["lr"] = self._get_lr_safe()

        self._debug_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # -------------------------
    # Core
    # -------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 0) assistant starts for representation pooling
        main_start = inputs.pop("assistant_start", None)
        aug_start = inputs.pop("augment_assistant_start", None)
        neg_start = inputs.pop("negative_assistant_start", None)

        # 1) pop custom fields
        ips_weights = inputs.pop("ips_weight", None)

        neg_input_ids = inputs.pop("negative_input_ids", None)
        neg_attention_mask = inputs.pop("negative_attention_mask", None)

        aug_input_ids = inputs.pop("augment_input_ids", None)
        aug_attention_mask = inputs.pop("augment_attention_mask", None)

        coin_margin = inputs.pop("coin_margin", None)  # [B] / scalar
        coin_weight = inputs.pop("coin_weight", None)  # [B] / scalar

        # 2) forward (main NTP)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        # 3) IPS-weighted NTP loss (batch-mean normalization)
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

        ips_mean_raw = None
        ips_mean_normed = None

        if ips_weights is not None:
            ips = self._to_device_tensor(ips_weights, token_losses.device, dtype=token_losses.dtype)
            ips = ips.view(-1, 1)  # [B,1]

            try:
                ips_mean_raw = float(ips.mean().detach().float().cpu().item())
            except Exception:
                ips_mean_raw = None

            ips = ips / (ips.mean() + 1e-9)

            try:
                ips_mean_normed = float(ips.mean().detach().float().cpu().item())
            except Exception:
                ips_mean_normed = None

            weighted = token_losses * padding_mask * ips
        else:
            weighted = token_losses * padding_mask

        ntp_loss = weighted.sum() / (padding_mask.sum() + 1e-9)

        # 4) CoIN contrastive
        contrastive_loss = torch.tensor(0.0, device=ntp_loss.device, dtype=ntp_loss.dtype)

        dbg = {}
        need_coin = (aug_input_ids is not None) or (neg_input_ids is not None)
        if need_coin:
            pos_hidden = outputs.hidden_states[-1]
            pos_repr = self._masked_mean_pooling_from_start(
                pos_hidden, inputs["attention_mask"], main_start
            )
            B = pos_repr.size(0)

            # margin: [B]
            if coin_margin is None:
                margin = torch.full((B,), float(self.default_margin), device=pos_repr.device, dtype=pos_repr.dtype)
            else:
                margin = self._to_device_tensor(coin_margin, pos_repr.device, dtype=pos_repr.dtype)
                if margin.numel() == 1:
                    margin = margin.view(1).repeat(B)
                else:
                    margin = margin.view(-1)
                    if margin.size(0) != B:
                        margin = torch.full((B,), float(self.default_margin), device=pos_repr.device, dtype=pos_repr.dtype)

            # weight: [B]
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

            # 4A) consistency（默认 False，保留）
            sim_cons = None
            if self.use_consistency and aug_input_ids is not None:
                aug_outputs = model(
                    input_ids=aug_input_ids,
                    attention_mask=aug_attention_mask,
                    output_hidden_states=True,
                )
                aug_hidden = aug_outputs.hidden_states[-1]
                aug_repr = self._masked_mean_pooling_from_start(
                    aug_hidden, aug_attention_mask, aug_start
                )
                sim_cons = F.cosine_similarity(pos_repr, aug_repr, dim=-1)
                cons_vec = (1.0 - sim_cons)
            else:
                cons_vec = torch.zeros((B,), device=pos_repr.device, dtype=pos_repr.dtype)

            # 4B) negative rank —— self-positive margin ranking + view-dropout
            sim_pos = None
            sim_neg = None
            neg_vec = torch.zeros((B,), device=pos_repr.device, dtype=pos_repr.dtype)

            if self.use_negative_rank and neg_input_ids is not None:
                neg_outputs = model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True,
                )
                neg_hidden = neg_outputs.hidden_states[-1]
                neg_repr = self._masked_mean_pooling_from_start(
                    neg_hidden, neg_attention_mask, neg_start
                )

                # ✅ view-dropout：制造 pos 两视角（training=True 强制启用）
                p = float(self.view_dropout)
                pos_view1 = F.dropout(pos_repr, p=p, training=True)
                pos_view2 = F.dropout(pos_repr, p=p, training=True)

                sim_pos = F.cosine_similarity(pos_view1, pos_view2, dim=-1)   # [B]
                sim_neg = F.cosine_similarity(pos_view1, neg_repr, dim=-1)    # [B]

                # hinge: want sim_pos >= sim_neg + margin
                neg_vec = torch.clamp(margin - (sim_pos - sim_neg), min=0.0)  # [B]

            coin_vec = self.consistency_weight * cons_vec + self.negative_weight * neg_vec
            coin_vec = coin_vec * w
            contrastive_loss = coin_vec.mean()

            # debug
            try:
                dbg["batch_B"] = int(B)
                dbg["lambda_coin"] = float(self.lambda_coin)
                dbg["default_margin"] = float(self.default_margin)
                dbg["view_dropout"] = float(self.view_dropout)

                dbg["coin_margin_mean"] = float(margin.mean().detach().float().cpu().item())
                dbg["coin_weight_mean"] = float(w.mean().detach().float().cpu().item())

                dbg["consistency_mean"] = float(cons_vec.mean().detach().float().cpu().item())
                dbg["hinge_mean"] = float(neg_vec.mean().detach().float().cpu().item())
                dbg["hinge_pos_rate"] = float((neg_vec > 0).float().mean().detach().cpu().item())

                if sim_cons is not None:
                    dbg["sim_cons_mean"] = float(sim_cons.mean().detach().float().cpu().item())
                if sim_pos is not None:
                    dbg["sim_pos_mean"] = float(sim_pos.mean().detach().float().cpu().item())
                if sim_neg is not None:
                    dbg["sim_neg_mean"] = float(sim_neg.mean().detach().float().cpu().item())
                if (sim_pos is not None) and (sim_neg is not None):
                    dbg["sim_gap_mean"] = float((sim_pos - sim_neg).mean().detach().float().cpu().item())

                if ips_mean_raw is not None:
                    dbg["ips_mean_raw"] = float(ips_mean_raw)
                if ips_mean_normed is not None:
                    dbg["ips_mean_normed"] = float(ips_mean_normed)
            except Exception:
                pass

        total_loss = ntp_loss + self.lambda_coin * contrastive_loss

        try:
            self._maybe_log_debug({
                "ntp_loss": float(ntp_loss.detach().float().cpu().item()),
                "contrastive_loss": float(contrastive_loss.detach().float().cpu().item()),
                "total_loss": float(total_loss.detach().float().cpu().item()),
                **dbg,
            })
        except Exception:
            pass

        return (total_loss, outputs) if return_outputs else total_loss



