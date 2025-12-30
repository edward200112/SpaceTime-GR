# HardMiningGRPO/reward_ntp_itemid.py
import json
import re
import heapq
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
from collections import defaultdict, Counter

import torch


ID_FIND_RE = re.compile(r"(-?\d+)")


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
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


def extract_first_item_id(completion_text: str) -> Tuple[Optional[int], str, str, bool, bool]:
    """
    return: item_id, first_line, rest_text, prefix_ok, has_extra
    prefix_ok: first_line 在数字前不含非空前缀（允许 'id:' 这种会被视为 prefix）
    """
    t = _to_text(completion_text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)
    rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    m = ID_FIND_RE.search(first)
    if not m:
        return None, first, rest, False, bool(rest)

    prefix = first[:m.start()].strip()
    prefix_ok = (prefix == "")  # 只要数字前有东西，就算 prefix 污染（更严格）
    try:
        item_id = int(m.group(1))
    except Exception:
        item_id = None

    # extra: 数字后还有内容 或 多行
    tail = first[m.end():].strip()
    has_extra = bool(tail) or bool(rest)
    return item_id, first, rest, prefix_ok, has_extra


@dataclass
class ResolverConfig:
    # phase / pool
    phase: int = 1
    pool_mode: str = "candidate"   # candidate | teacher_top
    teacher_pool_k: int = 200

    # base rewards
    format_bonus: float = 0.05
    exists_bonus: float = 0.10

    correct_reward: float = 2.0
    wrong_penalty: float = 0.3
    unknown_penalty: float = 0.6

    # teacher shaping
    rank_shaping_weight: float = 0.2
    alpha: float = 0.6
    teacher_clip: float = 5.0
    pool_in_bonus: float = 0.02  # 输出落在 pool 里给一点小 bonus（Phase3 建议 0）

    # penalties
    extra_text_penalty: float = 0.05
    prefix_penalty: float = 0.05
    duplicate_penalty: float = 0.02

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


class SasrecScorer:
    def __init__(self, sasrec_model, n_items: int, device: str = "cuda"):
        self.sasrec = sasrec_model.to(device)
        self.sasrec.eval()
        self.n_items = int(n_items)
        self.device = device

    @torch.no_grad()
    def score_candidates(self, history: List[int], candidate_ids: List[int]) -> torch.Tensor:
        if not history or not candidate_ids:
            return torch.zeros(len(candidate_ids), device=self.device)
        hist = torch.tensor(history, dtype=torch.long, device=self.device).unsqueeze(0)
        cand = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        scores = self.sasrec.predict_candidates(hist, cand).squeeze(0)  # [K]
        return scores


def _rank_positions_desc(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=scores.device)
    return ranks


def _dedup_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_reward_fn(sasrec_scorer: SasrecScorer, cfg: ResolverConfig):
    state = {"last_logged_step": None, "_fallback_step": 0}

    def reward_fn(prompts, completions, **kwargs):
        histories: List[List[int]] = kwargs["history_item_ids"]
        target_ids: List[int] = kwargs["target_item_id"]

        candidate_ids_list: List[List[int]] = kwargs.get("candidate_item_ids", [[] for _ in completions])
        teacher_top_list: List[List[int]] = kwargs.get("teacher_top_item_ids", [[] for _ in completions])

        step = kwargs.get("step", None)
        if step is None:
            step = state["_fallback_step"]
            state["_fallback_step"] += 1
        step = int(step)

        n = len(completions)

        # group duplicates within same prompt
        prompt_keys = [norm_text(_to_text(p)) for p in prompts]
        out_keys = []
        out_first_lines = []
        for comp in completions:
            comp_text = _to_text(comp)
            item_id, first_line, _, _, _ = extract_first_item_id(comp_text)
            out_first_lines.append(first_line)
            out_keys.append(str(item_id) if item_id is not None else "")

        group_to_indices = defaultdict(list)
        for i, pk in enumerate(prompt_keys):
            group_to_indices[pk].append(i)

        dup_count_map = {}
        unique_rate_map = {}
        for pk, idxs in group_to_indices.items():
            ctr = Counter()
            for i in idxs:
                key = out_keys[i] or out_first_lines[i] or "<EMPTY>"
                ctr[key] += 1
            for i in idxs:
                key = out_keys[i] or out_first_lines[i] or "<EMPTY>"
                dup_count_map[i] = ctr[key]
            unique_rate_map[pk] = float(len(ctr)) / max(1, len(idxs))

        # stats
        cnt_parse = cnt_exists = cnt_correct = 0
        cnt_prefix_bad = cnt_extra = cnt_unknown = 0
        cnt_in_pool = 0
        sum_reward = sum_fmt = sum_exist = sum_core = sum_shape = sum_pen = 0.0
        sum_rankdist = 0.0
        cnt_rankdist = 0

        top_heap = []
        bot_heap = []
        seq = 0

        rewards = []
        dump_recs = []

        for i, (comp, hist, tgt_id, cand_ids, top_ids) in enumerate(
            zip(completions, histories, target_ids, candidate_ids_list, teacher_top_list)
        ):
            comp_text = _to_text(comp)
            item_id, first_line, rest, prefix_ok, has_extra = extract_first_item_id(comp_text)

            r_fmt = 0.0
            r_exist = 0.0
            r_core = 0.0
            r_shape = 0.0
            pen = 0.0

            # format bonus
            if item_id is not None:
                cnt_parse += 1
                r_fmt = float(cfg.format_bonus)
            else:
                cnt_unknown += 1
                r_core -= float(cfg.unknown_penalty)

            # penalties
            if item_id is not None and (not prefix_ok):
                cnt_prefix_bad += 1
                pen -= float(cfg.prefix_penalty)

            if item_id is not None and has_extra:
                cnt_extra += 1
                pen -= float(cfg.extra_text_penalty)

            # exists check (合法 item_id)
            exists = False
            if item_id is not None:
                # 允许 1..n_items（按你的 SASRec/数据习惯可调整）
                if 1 <= int(item_id) <= int(sasrec_scorer.n_items):
                    exists = True
                    cnt_exists += 1
                    r_exist = float(cfg.exists_bonus)
                else:
                    cnt_unknown += 1
                    r_core -= float(cfg.unknown_penalty)

            # core hit reward
            correct = False
            if exists:
                if int(item_id) == int(tgt_id):
                    correct = True
                    cnt_correct += 1
                    r_core += float(cfg.correct_reward)
                else:
                    r_core -= float(cfg.wrong_penalty)

            # teacher shaping (wrong only or both都可；这里 wrong 时更有意义)
            if exists and float(cfg.rank_shaping_weight) > 0.0:
                # pick pool
                if cfg.pool_mode == "teacher_top" and isinstance(top_ids, list) and len(top_ids) > 0:
                    pool = list(top_ids)[: int(cfg.teacher_pool_k)]
                else:
                    pool = list(cand_ids) if isinstance(cand_ids, list) else []

                pool = [int(x) for x in pool if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
                # ensure target in pool
                pool.append(int(tgt_id))
                pool = _dedup_keep_order(pool)

                in_pool = int(item_id) in set(pool)
                if in_pool:
                    cnt_in_pool += 1
                    r_shape += float(cfg.pool_in_bonus)

                # append output id to pool to compute rank
                if int(item_id) not in set(pool):
                    pool.append(int(item_id))

                # score pool
                try:
                    scores = sasrec_scorer.score_candidates(hist, pool)
                    # clip
                    if float(cfg.teacher_clip) > 0:
                        scores = torch.clamp(scores, -float(cfg.teacher_clip), float(cfg.teacher_clip))
                    ranks = _rank_positions_desc(scores)  # best=0

                    # rank distance shaping
                    id2pos = {pid: j for j, pid in enumerate(pool)}
                    if int(tgt_id) in id2pos and int(item_id) in id2pos:
                        rt = int(ranks[id2pos[int(tgt_id)]].item())
                        ro = int(ranks[id2pos[int(item_id)]].item())
                        K = int(ranks.numel())
                        dist = abs(ro - rt) / max(1, K - 1)  # 0..1
                        # 近=>高 shaping
                        r_shape += float(cfg.alpha) * float(cfg.rank_shaping_weight) * float(1.0 - dist)

                        sum_rankdist += float(dist)
                        cnt_rankdist += 1
                except Exception:
                    pass

            # duplicate penalty
            dup_cnt = int(dup_count_map.get(i, 1))
            if dup_cnt > 1:
                pen -= float(cfg.duplicate_penalty) * float(dup_cnt - 1)

            r = r_fmt + r_exist + r_core + r_shape + pen
            rewards.append(float(r))

            sum_reward += float(r)
            sum_fmt += float(r_fmt)
            sum_exist += float(r_exist)
            sum_core += float(r_core)
            sum_shape += float(r_shape)
            sum_pen += float(pen)

            rec = {
                "reward": float(r),
                "first": first_line,
                "item_id": int(item_id) if item_id is not None else -1,
                "tgt_id": int(tgt_id),
                "correct": bool(correct),
                "fmt": float(r_fmt),
                "exist": float(r_exist),
                "core": float(r_core),
                "shape": float(r_shape),
                "pen": float(pen),
                "prefix_ok": bool(prefix_ok),
                "has_extra": bool(has_extra),
                "dup_cnt": int(dup_cnt),
                "pool_mode": cfg.pool_mode,
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
                print(f"[DEBUG reward-itemid] step={step} n={n} phase={cfg.phase} pool_mode={cfg.pool_mode}")
                print(
                    f"  parse_rate={d(cnt_parse,n):.3f} exists_rate={d(cnt_exists,n):.3f} "
                    f"correct_rate={d(cnt_correct,n):.3f} unknown_rate={d(cnt_unknown,n):.3f} "
                    f"in_pool_rate={d(cnt_in_pool,n):.3f} "
                    f"prefix_bad_rate={d(cnt_prefix_bad,n):.3f} extra_rate={d(cnt_extra,n):.3f} "
                    f"group_unique_rate≈{mean_unique:.3f} mean_rankdist≈{mean_rankdist:.3f}"
                )
                print(
                    f"  mean_reward={d(sum_reward,n):.4f} mean_fmt={d(sum_fmt,n):.4f} mean_exist={d(sum_exist,n):.4f} "
                    f"mean_core={d(sum_core,n):.4f} mean_shape={d(sum_shape,n):.4f} mean_pen={d(sum_pen,n):.4f}"
                )

                top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                bot_sorted = sorted(bot_heap, key=lambda x: x[0])

                print("\n  [TOP examples]")
                for rr, _, rec in top_sorted:
                    print(f"    r={rec['reward']:.4f} id={rec['item_id']} tgt={rec['tgt_id']} "
                          f"correct={int(rec['correct'])} dup={rec['dup_cnt']} first='{rec['first'][:80]}'")

                print("\n  [BOTTOM examples]")
                for neg_rr, _, rec in bot_sorted:
                    print(f"    r={rec['reward']:.4f} id={rec['item_id']} tgt={rec['tgt_id']} "
                          f"correct={int(rec['correct'])} dup={rec['dup_cnt']} first='{rec['first'][:80]}'")
                print("=" * 90)

        return rewards

    return reward_fn
