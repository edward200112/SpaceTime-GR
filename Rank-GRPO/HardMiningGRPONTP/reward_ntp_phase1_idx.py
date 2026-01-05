# reward_ntp_phase1_idx.py
# Phase1: model outputs candidate index (1..K) as pure digits.
# Reward maps idx -> candidate_item_ids[idx-1], then compute:
#   - format bonus (pure digits)
#   - exists bonus (idx in range)
#   - correct reward / wrong penalty / unknown penalty
#   - SASRec rank shaping within candidate pool (optional, recommended)
#
# Designed to be robust to TRL/GRPO reward function calling conventions.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import json
import torch

ID_STRICT_RE = re.compile(r"^\s*(\d+)\s*$")
ID_FIRST_RE = re.compile(r"(\d+)")

@dataclass
class ResolverConfig:
    # rewards
    format_bonus: float = 0.05
    exists_bonus: float = 0.10
    correct_reward: float = 2.0
    wrong_penalty: float = 0.3
    unknown_penalty: float = 0.6

    # shaping
    teacher_rank_weight: float = 0.2   # multiply shaping score
    alpha: float = 0.6                 # shaping curve
    teacher_clip: float = 5.0          # clip shaping contribution
    pool_in_bonus: float = 0.02        # (optional) bonus if idx valid (exists already covers)

    # penalties for bad format
    extra_text_penalty: float = 0.05   # if not strict digits (has extra)
    prefix_penalty: float = 0.05       # if digits not starting from beginning (weak)
    duplicate_penalty: float = 0.02    # not used (single number), kept for compatibility

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


class SasrecScorer:
    """
    Wrap SASRec model to score candidate pools.
    Expect sasrec.predict_candidates(hist_ids[B,L], cand_ids[B,K]) -> scores[B,K]
    """
    def __init__(self, sasrec_model, n_items: int, device: str = "cuda"):
        self.model = sasrec_model
        self.n_items = int(n_items)
        self.device = device

    @torch.no_grad()
    def score_candidates_batch(
        self,
        history_ids: List[List[int]],
        cand_ids: List[List[int]],
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores: [B, Kmax] float32
          mask:   [B, Kmax] bool (True where valid candidate)
        """
        B = len(history_ids)
        Kmax = max(len(x) for x in cand_ids) if B > 0 else 0
        if Kmax <= 0:
            return torch.empty((B, 0), device=self.device), torch.empty((B, 0), device=self.device, dtype=torch.bool)

        # pad history left with 0
        hist = []
        for h in history_ids:
            h = [int(x) for x in (h or []) if x is not None]
            if len(h) >= max_len:
                h = h[-max_len:]
            else:
                h = [0] * (max_len - len(h)) + h
            hist.append(h)
        hist_t = torch.tensor(hist, dtype=torch.long, device=self.device)

        # pad candidates with 0 (mask them out later)
        cand_pad = []
        mask = []
        for cs in cand_ids:
            cs = [int(x) for x in (cs or []) if x is not None]
            m = [True] * len(cs) + [False] * (Kmax - len(cs))
            cs2 = cs + [0] * (Kmax - len(cs))
            cand_pad.append(cs2)
            mask.append(m)
        cand_t = torch.tensor(cand_pad, dtype=torch.long, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        scores = self.model.predict_candidates(hist_t, cand_t).to(torch.float32)  # [B,Kmax]
        # mask padded candidates
        scores = scores.masked_fill(~mask_t, -1e9)
        return scores, mask_t

    @torch.no_grad()
    def rank_in_pool_batch(
        self,
        history_ids: List[List[int]],
        cand_ids: List[List[int]],
        chosen_pos: List[Optional[int]],  # index in cand_ids[b], 0-based
        max_len: int,
    ) -> List[Optional[int]]:
        """
        Return 1-based rank of chosen candidate by SASRec score within its pool.
        If chosen_pos[b] is None -> None
        """
        scores, mask = self.score_candidates_batch(history_ids, cand_ids, max_len=max_len)
        if scores.numel() == 0:
            return [None for _ in chosen_pos]

        # ranks per row
        # order: descending
        order = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(order.size(1), device=order.device).unsqueeze(0).expand_as(order))

        out: List[Optional[int]] = []
        for b, pos in enumerate(chosen_pos):
            if pos is None:
                out.append(None)
                continue
            # if pos beyond row len -> None
            if pos < 0 or pos >= scores.size(1) or (not bool(mask[b, pos].item())):
                out.append(None)
                continue
            out.append(int(ranks[b, pos].item()) + 1)
        return out


def _extract_idx(completion: Any) -> Tuple[Optional[int], bool, bool]:
    """
    Returns: (idx, strict_ok, has_extra)
      strict_ok: whole completion is only digits (allow surrounding spaces)
      has_extra: not strict digits but contains a number somewhere (or extra text)
    """
    if completion is None:
        return None, False, False
    s = completion
    if isinstance(s, dict):
        # try common fields
        for k in ("content", "text", "completion", "generated_text"):
            if k in s:
                s = s[k]
                break
    s = str(s)

    m = ID_STRICT_RE.match(s)
    if m:
        return int(m.group(1)), True, False

    m2 = ID_FIRST_RE.search(s)
    if not m2:
        return None, False, False
    return int(m2.group(1)), False, True


def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # datasets may give tuple
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _get_batch_field(kwargs: Dict[str, Any], name: str, B: int) -> Any:
    v = kwargs.get(name, None)
    if v is None:
        return None
    # sometimes torch tensor
    if torch.is_tensor(v):
        v = v.detach().cpu().tolist()
    # If scalar repeated?
    if not isinstance(v, list):
        return [v] * B
    return v


def make_reward_fn(scorer: SasrecScorer, cfg: ResolverConfig, sasrec_max_len: int = 50):
    step = {"i": 0}
    dump_f = None
    if cfg.debug_dump_jsonl:
        dump_f = open(cfg.debug_dump_jsonl, "w", encoding="utf-8")

    @torch.no_grad()
    def reward_fn(*args, **kwargs):
        """
        Robust to TRL calling styles:
          - reward_fn(prompts, completions, **batch_fields)
          - reward_fn(completions=completions, **batch_fields)
        """
        completions = kwargs.get("completions", None)
        if completions is None and len(args) >= 2:
            completions = args[1]
        if completions is None and len(args) == 1:
            completions = args[0]
        completions = _safe_list(completions)

        B = len(completions)

        histories = _get_batch_field(kwargs, "history_item_ids", B)
        targets = _get_batch_field(kwargs, "target_item_id", B)
        cand_ids = _get_batch_field(kwargs, "candidate_item_ids", B)

        if histories is None or targets is None or cand_ids is None:
            # If fields missing, return zeros but make it obvious.
            return [0.0 for _ in range(B)]

        # parse idx + locate chosen position in candidate list
        idx_list: List[Optional[int]] = []
        strict_list: List[bool] = []
        extra_list: List[bool] = []
        chosen_pos: List[Optional[int]] = []
        pred_item_ids: List[Optional[int]] = []

        for b in range(B):
            idx, strict_ok, has_extra = _extract_idx(completions[b])
            idx_list.append(idx)
            strict_list.append(strict_ok)
            extra_list.append(has_extra)

            cands_b = [int(x) for x in (_safe_list(cand_ids[b]))]
            if idx is None or idx < 1 or idx > len(cands_b):
                chosen_pos.append(None)
                pred_item_ids.append(None)
            else:
                pos0 = idx - 1
                chosen_pos.append(pos0)
                pred_item_ids.append(int(cands_b[pos0]))

        # teacher rank shaping
        ranks = scorer.rank_in_pool_batch(
            history_ids=[_safe_list(h) for h in histories],
            cand_ids=[_safe_list(c) for c in cand_ids],
            chosen_pos=chosen_pos,
            max_len=sasrec_max_len,
        )

        rewards: List[float] = []
        dbg_rows = []

        for b in range(B):
            tgt = int(targets[b])
            idx = idx_list[b]
            strict_ok = strict_list[b]
            has_extra = extra_list[b]
            pred_id = pred_item_ids[b]

            r = 0.0
            # format
            if idx is not None:
                r += cfg.format_bonus
            if strict_ok:
                # strict digits => no extra penalty
                pass
            else:
                # has number but also extra/prefix
                if idx is not None:
                    r -= cfg.extra_text_penalty
                else:
                    # no number at all
                    r -= cfg.prefix_penalty

            # exists / unknown
            if pred_id is None:
                r -= cfg.unknown_penalty
                rewards.append(float(r))
                if dump_f:
                    dbg_rows.append({"tgt": tgt, "out": idx, "pred_id": pred_id, "reward": r, "reason": "unknown"})
                continue
            else:
                r += cfg.exists_bonus

            # correct / wrong
            correct = (pred_id == tgt)
            if correct:
                r += cfg.correct_reward
            else:
                r -= cfg.wrong_penalty

            # shaping by rank (lower rank is better)
            rk = ranks[b]
            if rk is not None:
                # convert rank -> [0,1], best=1
                Kb = len(_safe_list(cand_ids[b]))
                denom = max(1, Kb - 1)
                norm = 1.0 - float(rk - 1) / float(denom)
                # alpha curve
                shape = (norm ** cfg.alpha) * cfg.teacher_rank_weight
                # clip
                if shape > cfg.teacher_clip:
                    shape = cfg.teacher_clip
                if shape < -cfg.teacher_clip:
                    shape = -cfg.teacher_clip
                r += shape

            rewards.append(float(r))

            if dump_f:
                dbg_rows.append({
                    "tgt": tgt,
                    "idx": idx,
                    "pred_id": pred_id,
                    "correct": int(correct),
                    "rank_in_pool": rk,
                    "strict": int(strict_ok),
                    "has_extra": int(has_extra),
                    "reward": r,
                })

        # debug print
        step["i"] += 1
        if cfg.debug_log_every_steps and (step["i"] % cfg.debug_log_every_steps == 0):
            # small snapshot
            show = min(cfg.debug_num_show, B)
            print(f"[DEBUG reward-phase1-idx] step={step['i']} B={B}")
            for i in range(show):
                print("  ex", i, "idx=", idx_list[i], "pred=", pred_item_ids[i], "tgt=", int(targets[i]),
                      "rank=", ranks[i], "r=", rewards[i],
                      "strict=", strict_list[i], "extra=", extra_list[i])
                if cfg.debug_print_full_completion:
                    print("    completion=", repr(completions[i]))

        if dump_f and dbg_rows:
            for row in dbg_rows:
                dump_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            dump_f.flush()

        return rewards

    return reward_fn
