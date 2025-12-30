# HardMiningGRPO/reward_sasrec.py
import json
import re
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
from collections import defaultdict, Counter

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


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("content", "text", "completion", "generated_text", "prompt"):
            if k in x:
                return str(x[k])
    return str(x)


def _has_incomplete_paren(s: str) -> bool:
    s = s or ""
    return (s.count("(") != s.count(")")) or (("(" in s) ^ (")" in s))


def extract_first_namecat(completion_text: str) -> Tuple[Optional[str], Optional[str], str, str, bool, bool]:
    """
    return: name, cat, first_line, trail, prefix_ok, incomplete
    """
    t = _to_text(completion_text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)

    incomplete = _has_incomplete_paren(first)

    m = NAMECAT_FIND_RE.search(first)
    if not m:
        return None, None, first, "", False, incomplete

    name = norm_text(m.group(1))
    cat = norm_text(m.group(2))
    prefix_ok = (first[:m.start()].strip() == "")
    trail = norm_text(first[m.end():])
    return name, cat, first, trail, prefix_ok, incomplete


def canon_key(name: str, cat: str) -> str:
    name = norm_text(name)
    cat = norm_text(cat)
    cat = cat.replace("centre", "center")
    return f"{name} ({cat})"


def parse_namecat_keys(text: str) -> Tuple[str, str, bool]:
    """
    返回 canonical key（保留大小写）和 casefold key
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
    # formatting / membership
    format_bonus: float = 0.05
    in_candidates_bonus: float = 0.05

    # ✅ HR 主目标：选中 GT 才高回报
    correct_reward: float = 2.0          # chosen == GT
    wrong_penalty: float = 0.3           # chosen in candidates but != GT
    unknown_penalty: float = 0.6         # not in candidates (or cannot parse)
    rank_shaping_weight: float = 0.2     # wrong 时：按 teacher-rank 距离 GT 给 0..w 的 shaping

    # teacher
    softmax_temp: float = 1.0            # 仅用于 SASRec predict 的温度（不一定要用，但保留）

    # penalties
    extra_text_penalty: float = 0.05
    prefix_penalty: float = 0.05
    incomplete_penalty: float = 0.10
    copy_penalty: float = 0.08
    duplicate_penalty: float = 0.02

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


class SasrecScorer:
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
        scores = self.sasrec.predict_candidates(hist, cand).squeeze(0)  # [K]
        return scores


def _rank_positions_desc(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: [K]
    return ranks: [K], best rank = 0
    """
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=scores.device)
    return ranks


def make_reward_fn(sasrec_scorer: SasrecScorer, cfg: ResolverConfig):
    state = {"last_logged_step": None, "_fallback_step": 0}

    def reward_fn(prompts, completions, **kwargs):
        histories: List[List[int]] = kwargs["history_item_ids"]
        targets_nc: List[str] = kwargs["target_namecat"]

        cand_namecats: List[List[str]] = kwargs["candidate_namecats"]
        cand_item_ids: List[List[int]] = kwargs["candidate_item_ids"]

        # ✅ 来自 train_grpo 的字段（你已经打印过 TARGET_POS_HIST，说明有）
        target_pos_list: List[int] = kwargs["target_pos"]

        step = kwargs.get("step", None)
        if step is None:
            step = state["_fallback_step"]
            state["_fallback_step"] += 1
        step = int(step)

        n = len(completions)

        # group duplicates within same prompt
        prompt_keys = [norm_text(_to_text(p)) for p in prompts]
        out_first_lines = []
        out_fold_keys = []
        for comp in completions:
            comp_text = _to_text(comp)
            lines = comp_text.splitlines()
            first_line = norm_text(lines[0] if lines else comp_text)
            out_first_lines.append(first_line)
            name, cat, *_ = extract_first_namecat(comp_text)
            if name is not None and cat is not None:
                out_fold_keys.append(canon_key(name, cat).casefold())
            else:
                out_fold_keys.append("")

        group_to_indices = defaultdict(list)
        for i, pk in enumerate(prompt_keys):
            group_to_indices[pk].append(i)

        dup_count_map = {}
        unique_rate_map = {}
        for pk, idxs in group_to_indices.items():
            ctr = Counter()
            for i in idxs:
                key = out_fold_keys[i] or out_first_lines[i] or "<EMPTY>"
                ctr[key] += 1
            for i in idxs:
                key = out_fold_keys[i] or out_first_lines[i] or "<EMPTY>"
                dup_count_map[i] = ctr[key]
            unique_rate_map[pk] = float(len(ctr)) / max(1, len(idxs))

        # stats
        cnt_prefix_ok = cnt_in_cands = 0
        cnt_in_exact = cnt_in_fold = 0
        cnt_correct = 0
        cnt_unknown = cnt_extra = cnt_incomplete = 0
        sum_reward = sum_fmt = sum_in = sum_core = sum_shape = sum_pen = 0.0
        sum_rankdist = 0.0
        cnt_rankdist = 0

        top_heap = []
        bot_heap = []
        seq = 0

        rewards = []
        dump_recs = []

        for i, (comp, hist, tgt_nc, cands_nc, cands_it, tgt_pos) in enumerate(
            zip(completions, histories, targets_nc, cand_namecats, cand_item_ids, target_pos_list)
        ):
            comp_text = _to_text(comp)
            lines = comp_text.splitlines()
            first_line = norm_text(lines[0] if lines else comp_text)

            name, cat, _, trail, prefix_ok, incomplete = extract_first_namecat(comp_text)
            has_namecat = (name is not None and cat is not None)

            r_fmt = float(cfg.format_bonus) if has_namecat else 0.0
            r_in = 0.0
            r_core = 0.0
            r_shape = 0.0
            pen = 0.0

            if prefix_ok:
                cnt_prefix_ok += 1
            else:
                if has_namecat:
                    pen -= float(cfg.prefix_penalty)

            if incomplete:
                cnt_incomplete += 1
                pen -= float(cfg.incomplete_penalty)

            extra = False
            if has_namecat:
                if trail.strip():
                    extra = True
                if len(lines) > 1 and any(norm_text(x) for x in lines[1:]):
                    extra = True
            if extra:
                pen -= float(cfg.extra_text_penalty)
                cnt_extra += 1

            out_key_exact = canon_key(name, cat) if has_namecat else ""
            out_key_fold = out_key_exact.casefold() if out_key_exact else ""

            # candidate maps
            cand_exact2idx: Dict[str, int] = {}
            cand_fold2idx: Dict[str, int] = {}
            cand_ids: List[int] = []
            for j, (s, it) in enumerate(zip(cands_nc, cands_it)):
                k_exact, k_fold, ok = parse_namecat_keys(s)
                if not ok:
                    k_exact = norm_text(s)
                    k_fold = k_exact.casefold()
                if k_exact and k_exact not in cand_exact2idx:
                    cand_exact2idx[k_exact] = j
                if k_fold and k_fold not in cand_fold2idx:
                    cand_fold2idx[k_fold] = j
                cand_ids.append(int(it))

            chosen_idx = None
            in_candidates = False
            via = "none"

            if not has_namecat:
                cnt_unknown += 1
                r_core -= float(cfg.unknown_penalty)
            else:
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
                    pen -= float(cfg.copy_penalty)

                if not in_candidates or chosen_idx is None:
                    cnt_unknown += 1
                    r_core -= float(cfg.unknown_penalty)
                else:
                    cnt_in_cands += 1
                    r_in = float(cfg.in_candidates_bonus)

                    # ✅ HR 主目标：选中 target_pos 才算对
                    correct = (int(chosen_idx) == int(tgt_pos))
                    if correct:
                        cnt_correct += 1
                        r_core += float(cfg.correct_reward)
                        # correct 时可不给 shaping（也行给一点常数；这里给 0）
                    else:
                        r_core -= float(cfg.wrong_penalty)

                        # ✅ teacher shaping：按 teacher-rank 距离 GT 给 0..w（越近越高）
                        if float(cfg.rank_shaping_weight) > 0 and cand_ids:
                            scores = sasrec_scorer.score_candidates(hist, cand_ids) / float(cfg.softmax_temp)
                            ranks = _rank_positions_desc(scores)  # best=0
                            rg = int(ranks[int(tgt_pos)].item()) if 0 <= int(tgt_pos) < ranks.numel() else ranks.numel() - 1
                            rc = int(ranks[int(chosen_idx)].item())
                            K = int(ranks.numel())
                            dist = abs(rc - rg) / max(1, K - 1)  # 0..1
                            # shaping: 近 => 接近 w；远 => 0
                            r_shape += float(cfg.rank_shaping_weight) * float(1.0 - dist)

                            sum_rankdist += float(dist)
                            cnt_rankdist += 1

            # duplicate penalty inside group
            dup_cnt = int(dup_count_map.get(i, 1))
            if dup_cnt > 1:
                pen -= float(cfg.duplicate_penalty) * float(dup_cnt - 1)

            r = r_fmt + r_in + r_core + r_shape + pen
            rewards.append(float(r))

            sum_reward += float(r)
            sum_fmt += float(r_fmt)
            sum_in += float(r_in)
            sum_core += float(r_core)
            sum_shape += float(r_shape)
            sum_pen += float(pen)

            rec = {
                "reward": float(r),
                "via": via,
                "first": first_line,
                "trail": trail,
                "out": out_key_exact,
                "out_fold": out_key_fold,
                "tgt": tgt_nc,
                "tgt_pos": int(tgt_pos),
                "chosen_idx": int(chosen_idx) if chosen_idx is not None else -1,
                "correct": bool(chosen_idx is not None and int(chosen_idx) == int(tgt_pos)),
                "fmt": float(r_fmt),
                "in": float(r_in),
                "core": float(r_core),
                "shape": float(r_shape),
                "pen": float(pen),
                "in_candidates": bool(in_candidates),
                "prefix_ok": bool(prefix_ok),
                "incomplete": bool(incomplete),
                "dup_cnt": int(dup_cnt),
                "full_completion": comp_text if cfg.debug_print_full_completion else "",
            }
            dump_recs.append(rec)

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

        if cfg.debug_dump_jsonl:
            try:
                with open(cfg.debug_dump_jsonl, "a", encoding="utf-8") as f:
                    for rec in dump_recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

        if cfg.debug_log_every_steps > 0:
            if (state["last_logged_step"] is None or step != state["last_logged_step"]) and (step % cfg.debug_log_every_steps == 0):
                state["last_logged_step"] = step

                def d(a, b): return a / b if b else 0.0
                mean_unique = sum(unique_rate_map.values()) / max(1, len(unique_rate_map))
                mean_rankdist = (sum_rankdist / cnt_rankdist) if cnt_rankdist else 0.0

                print("=" * 90)
                print(f"[DEBUG reward] step={step} n={n}")
                print(
                    f"  prefix_ok_rate={d(cnt_prefix_ok,n):.3f} "
                    f"in_candidates_rate={d(cnt_in_cands,n):.3f} "
                    f"(exact_in={cnt_in_exact}, fold_in={cnt_in_fold}) "
                    f"correct_rate={d(cnt_correct,n):.3f} "
                    f"unknown_rate={d(cnt_unknown,n):.3f} "
                    f"extra_text_rate={d(cnt_extra,n):.3f} "
                    f"incomplete_rate={d(cnt_incomplete,n):.3f} "
                    f"group_unique_rate≈{mean_unique:.3f} "
                    f"mean_rankdist(wrong_only)≈{mean_rankdist:.3f}"
                )
                print(
                    f"  mean_reward={d(sum_reward,n):.4f} "
                    f"mean_fmt={d(sum_fmt,n):.4f} "
                    f"mean_in={d(sum_in,n):.4f} "
                    f"mean_core={d(sum_core,n):.4f} "
                    f"mean_shape={d(sum_shape,n):.4f} "
                    f"mean_penalty={d(sum_pen,n):.4f}"
                )

                top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                bot_sorted = sorted(bot_heap, key=lambda x: x[0])

                print("\n  [TOP examples]")
                for rr, _, rec in top_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} inCand={int(rec['in_candidates'])} "
                        f"correct={int(rec['correct'])} dup={rec['dup_cnt']} "
                        f"chosen={rec['chosen_idx']} tgtpos={rec['tgt_pos']} "
                        f"first='{rec['first'][:80]}' "
                        f"(fmt={rec['fmt']:.2f}, in={rec['in']:.2f}, core={rec['core']:.2f}, "
                        f"shape={rec['shape']:.2f}, pen={rec['pen']:.2f})"
                    )

                print("\n  [BOTTOM examples]")
                for neg_rr, _, rec in bot_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} inCand={int(rec['in_candidates'])} "
                        f"correct={int(rec['correct'])} dup={rec['dup_cnt']} "
                        f"chosen={rec['chosen_idx']} tgtpos={rec['tgt_pos']} "
                        f"first='{rec['first'][:80]}' "
                        f"(fmt={rec['fmt']:.2f}, in={rec['in']:.2f}, core={rec['core']:.2f}, "
                        f"shape={rec['shape']:.2f}, pen={rec['pen']:.2f})"
                    )
                print("=" * 90)

        return rewards

    return reward_fn
