# HardMiningGRPO/reward_sasrec.py
import json
import re
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import torch

NAMECAT_FIND_RE = re.compile(r"([^\n\r\(\)]{1,200})\s*\(\s*([^\n\r\(\)]{1,120})\s*\)")


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = (
        s.replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    s = " ".join(s.split())
    return s


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("content", "text", "completion", "generated_text"):
            if k in x:
                return str(x[k])
    return str(x)


def extract_first_namecat(completion_text: str) -> Tuple[Optional[str], Optional[str], str, str, bool]:
    """
    从第一行中找第一个 Name (Cat) 子串。
    return: name, cat, first_line, trail, prefix_ok
    """
    t = _to_text(completion_text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)

    m = NAMECAT_FIND_RE.search(first)
    if not m:
        return None, None, first, "", False

    name = norm_text(m.group(1))
    cat = norm_text(m.group(2))
    prefix_ok = (first[:m.start()].strip() == "")
    trail = norm_text(first[m.end():])  # 同行尾巴
    return name, cat, first, trail, prefix_ok


def canon_key(name: str, cat: str) -> str:
    name = norm_text(name)
    cat = norm_text(cat)
    # 常见英式拼写
    cat = cat.replace("centre", "center")
    return f"{name} ({cat})"


def parse_namecat_keys(text: str) -> Tuple[str, str, bool]:
    """
    解析一个 "Name (Cat)" 字符串，返回：
    - key_exact: canonical 后的 key（保留大小写）
    - key_fold:  key_exact.casefold()（容错匹配用）
    - ok: 是否解析成功
    """
    t = norm_text(text)
    m = NAMECAT_FIND_RE.search(t)
    if not m:
        return "", "", False
    name = norm_text(m.group(1))
    cat = norm_text(m.group(2))
    k = canon_key(name, cat)
    return k, k.casefold(), True


@dataclass
class ResolverConfig:
    # reward weights
    format_bonus: float = 0.02
    match_reward: float = 1.0

    # shaping
    alpha: float = 0.1
    softmax_temp: float = 1.0

    # penalties
    extra_text_penalty: float = 0.05
    unknown_penalty: float = 0.05
    prefix_penalty: float = 0.05

    # ✅ 新增：容错命中候选但不是“原样输出”时的惩罚（逼模型 copy）
    copy_penalty: float = 0.02

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


class SasrecScorer:
    """只用于在候选集合内做 soft shaping（可关掉 alpha=0）。"""
    def __init__(self, sasrec_model, device: str = "cuda"):
        self.sasrec = sasrec_model.to(device)
        self.sasrec.eval()
        self.device = device

    @torch.no_grad()
    def score_candidates(self, history: List[int], candidate_ids: List[int]) -> torch.Tensor:
        if not history:
            return torch.zeros(len(candidate_ids), device=self.device)
        hist = torch.tensor(history, dtype=torch.long, device=self.device).unsqueeze(0)
        cand = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        scores = self.sasrec.predict_candidates(hist, cand).squeeze(0)
        return scores


def make_reward_fn(sasrec_scorer: SasrecScorer, cfg: ResolverConfig):
    state = {"last_logged_step": None, "_fallback_step": 0}

    def reward_fn(prompts, completions, **kwargs):
        histories: List[List[int]] = kwargs["history_item_ids"]
        targets_item: List[int] = kwargs["target_item_id"]
        targets_nc: List[str] = kwargs["target_namecat"]

        cand_namecats: List[List[str]] = kwargs["candidate_namecats"]
        cand_item_ids: List[List[int]] = kwargs["candidate_item_ids"]

        step = kwargs.get("step", None)
        if step is None:
            step = state["_fallback_step"]
            state["_fallback_step"] += 1
        step = int(step)

        n = len(completions)

        # stats
        cnt_prefix_ok = cnt_in_cands = cnt_in_exact = cnt_in_fold = 0
        cnt_match = cnt_unknown = cnt_extra = 0
        sum_reward = sum_fmt = sum_match = sum_soft = sum_pen = 0.0

        top_heap = []
        bot_heap = []
        seq = 0

        rewards = []
        for i, (comp, hist, tgt_item, tgt_nc, cands_nc, cands_it) in enumerate(
            zip(completions, histories, targets_item, targets_nc, cand_namecats, cand_item_ids)
        ):
            comp_text = _to_text(comp)
            lines = comp_text.splitlines()
            first_line = norm_text(lines[0] if lines else comp_text)

            name, cat, _, trail, prefix_ok = extract_first_namecat(comp_text)
            has_namecat = (name is not None and cat is not None)

            if prefix_ok:
                cnt_prefix_ok += 1

            # format bonus（只有输出里找得到 Name(Cat) 才给）
            r_fmt = float(cfg.format_bonus) if has_namecat else 0.0

            # extra text detection
            extra = False
            if has_namecat:
                if trail.strip():
                    extra = True
                if len(lines) > 1 and any(norm_text(x) for x in lines[1:]):
                    extra = True

            pen = 0.0
            if extra:
                pen -= float(cfg.extra_text_penalty)
                cnt_extra += 1

            if has_namecat and not prefix_ok:
                pen -= float(cfg.prefix_penalty)

            # target keys (casefold for match)
            tgt_key_exact, tgt_key_fold, tgt_ok = parse_namecat_keys(tgt_nc)

            # output keys
            out_key_exact = canon_key(name, cat) if has_namecat else ""
            out_key_fold = out_key_exact.casefold() if out_key_exact else ""

            # build candidate maps (exact + fold)
            cand_exact2idx: Dict[str, int] = {}
            cand_fold2idx: Dict[str, int] = {}
            cand_exact_list: List[str] = []
            cand_fold_list: List[str] = []
            for j, s in enumerate(cands_nc):
                k_exact, k_fold, ok = parse_namecat_keys(s)
                if not ok:
                    # 理论上不会发生（你校验过格式）
                    k_exact = norm_text(s)
                    k_fold = k_exact.casefold()
                cand_exact_list.append(k_exact)
                cand_fold_list.append(k_fold)
                # 若有重复 key，保留第一次即可
                if k_exact and k_exact not in cand_exact2idx:
                    cand_exact2idx[k_exact] = j
                if k_fold and k_fold not in cand_fold2idx:
                    cand_fold2idx[k_fold] = j

            r_match = 0.0
            r_soft = 0.0
            chosen_idx = None
            in_candidates = False
            via = "none"  # exact / fold / none

            if not has_namecat:
                cnt_unknown += 1
                pen -= float(cfg.unknown_penalty)
                r = r_fmt + pen
                rewards.append(float(r))
            else:
                # membership: exact first, then fold
                if out_key_exact and out_key_exact in cand_exact2idx:
                    chosen_idx = cand_exact2idx[out_key_exact]
                    in_candidates = True
                    via = "exact"
                    cnt_in_exact += 1
                elif out_key_fold and out_key_fold in cand_fold2idx:
                    chosen_idx = cand_fold2idx[out_key_fold]
                    in_candidates = True
                    via = "fold"
                    cnt_in_fold += 1
                    # ✅ 容错命中但不是原样输出，轻惩罚逼它 copy
                    pen -= float(cfg.copy_penalty)

                if not in_candidates or chosen_idx is None:
                    cnt_unknown += 1
                    pen -= float(cfg.unknown_penalty)
                    r = r_fmt + pen
                    rewards.append(float(r))
                else:
                    cnt_in_cands += 1

                    # main reward: namecat match（用 fold 比较，避免大小写坑）
                    if tgt_ok and out_key_fold and out_key_fold == tgt_key_fold:
                        r_match = float(cfg.match_reward)
                        cnt_match += 1

                    # soft shaping within candidates
                    if float(cfg.alpha) > 0 and cands_it and len(cands_it) == len(cands_nc):
                        cand_ids = [int(x) for x in cands_it]
                        scores = sasrec_scorer.score_candidates(hist, cand_ids) / float(cfg.softmax_temp)
                        probs = torch.softmax(scores, dim=0)
                        # chosen_idx 必须在范围内
                        if 0 <= int(chosen_idx) < len(cand_ids):
                            r_soft = float(probs[int(chosen_idx)].item())
                            sum_soft += r_soft

                    r = r_fmt + r_match + float(cfg.alpha) * r_soft + pen
                    rewards.append(float(r))

            sum_reward += float(rewards[-1])
            sum_fmt += float(r_fmt)
            sum_match += float(r_match)
            sum_pen += float(pen)

            rec = {
                "reward": float(rewards[-1]),
                "via": via,
                "first": first_line,
                "trail": trail,
                "out": out_key_exact,
                "out_fold": out_key_fold,
                "tgt": tgt_nc,
                "tgt_fold": tgt_key_fold,
                "tgt_item": int(tgt_item),
                "fmt": float(r_fmt),
                "match": float(r_match),
                "soft": float(r_soft),
                "pen": float(pen),
                "in_candidates": bool(in_candidates),
                "prefix_ok": bool(prefix_ok),
                "full_completion": comp_text if cfg.debug_print_full_completion else "",
            }

            seq += 1
            key_top = (float(rec["reward"]), seq, rec)
            if len(top_heap) < int(cfg.debug_num_show):
                heapq.heappush(top_heap, key_top)
            else:
                if float(rec["reward"]) > top_heap[0][0]:
                    heapq.heapreplace(top_heap, key_top)

            key_bot = (-float(rec["reward"]), seq, rec)
            if len(bot_heap) < int(cfg.debug_num_show):
                heapq.heappush(bot_heap, key_bot)
            else:
                if -float(rec["reward"]) > bot_heap[0][0]:
                    heapq.heapreplace(bot_heap, key_bot)

        # debug
        if cfg.debug_log_every_steps > 0:
            if (state["last_logged_step"] is None or step != state["last_logged_step"]) and (step % cfg.debug_log_every_steps == 0):
                state["last_logged_step"] = step

                def d(a, b): return a / b if b else 0.0
                print("=" * 90)
                print(f"[DEBUG reward] step={step} n={n}")
                print(
                    f"  prefix_ok_rate={d(cnt_prefix_ok,n):.3f} "
                    f"in_candidates_rate={d(cnt_in_cands,n):.3f} "
                    f"(exact_in={cnt_in_exact}, fold_in={cnt_in_fold}) "
                    f"match_namecat_rate={d(cnt_match,n):.3f} "
                    f"unknown_rate={d(cnt_unknown,n):.3f} "
                    f"extra_text_rate={d(cnt_extra,n):.3f}"
                )
                print(
                    f"  mean_reward={d(sum_reward,n):.4f} "
                    f"mean_fmt={d(sum_fmt,n):.4f} "
                    f"mean_match={d(sum_match,n):.4f} "
                    f"mean_soft(in_cands)={d(sum_soft,max(1,cnt_in_cands)):.4f} "
                    f"mean_penalty={d(sum_pen,n):.4f}"
                )

                top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                bot_sorted = sorted(bot_heap, key=lambda x: x[0])

                print("\n  [TOP examples]")
                for rr, _, rec in top_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} inCand={int(rec['in_candidates'])} "
                        f"first='{rec['first'][:80]}' trail='{rec['trail'][:40]}' "
                        f"out='{rec['out'][:60]}' tgt='{str(rec['tgt'])[:60]}' "
                        f"(fmt={rec['fmt']:.2f}, match={rec['match']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )

                print("\n  [BOTTOM examples]")
                for neg_rr, _, rec in bot_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} inCand={int(rec['in_candidates'])} "
                        f"first='{rec['first'][:80]}' trail='{rec['trail'][:40]}' "
                        f"out='{rec['out'][:60]}' tgt='{str(rec['tgt'])[:60]}' "
                        f"(fmt={rec['fmt']:.2f}, match={rec['match']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )
                print("=" * 90)

        return rewards

    return reward_fn
