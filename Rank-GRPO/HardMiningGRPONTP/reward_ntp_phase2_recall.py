# HardMiningGRPO/reward_ntp_phase2_recall.py
import json
import re
import heapq
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
from collections import defaultdict, Counter

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
    prefix_ok: 数字前不允许有非空前缀（严格）
    """
    t = _to_text(completion_text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)
    rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    m = ID_FIND_RE.search(first)
    if not m:
        return None, first, rest, False, bool(rest)

    prefix = first[:m.start()].strip()
    prefix_ok = (prefix == "")
    try:
        item_id = int(m.group(1))
    except Exception:
        item_id = None

    tail = first[m.end():].strip()
    has_extra = bool(tail) or bool(rest)
    return item_id, first, rest, prefix_ok, has_extra

def _dedup_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

@dataclass
class ResolverConfig:
    # pool
    teacher_pool_k: int = 200

    # base rewards
    format_bonus: float = 0.05
    exists_bonus: float = 0.10

    correct_reward: float = 2.0      # 命中 target
    wrong_penalty: float = 0.20      # 合法但没命中
    unknown_penalty: float = 0.60    # 解析失败或越界

    # ✅ recall teacher shaping（dense）
    alpha: float = 0.6               # 总系数
    teacher_rank_weight: float = 0.6 # 形状项强度：越大越“贴 teacher top”
    out_of_teacher_penalty: float = 0.02  # 仅 wrong 时：不在 teacher_top 给一点点惩罚（可设 0 更开放）

    # penalties
    extra_text_penalty: float = 0.05
    prefix_penalty: float = 0.05
    duplicate_penalty: float = 0.02

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


def make_reward_fn(n_items: int, cfg: ResolverConfig):
    """
    Phase2 recall reward:
      - correct: +correct_reward（不施加 teacher out-of-pool 惩罚，避免 teacher_top 没包含 target 时伤害学习）
      - wrong but exists:
          teacher shaping = alpha * teacher_rank_weight * (1 - rank/(K-1))  if in teacher_top[:K]
                           - alpha * out_of_teacher_penalty                if not in teacher_top[:K] (可设0)
      - parse/exists/格式/多余文本/前缀/重复输出 仍然约束输出干净
    """
    state = {"last_logged_step": None, "_fallback_step": 0}

    def reward_fn(prompts, completions, **kwargs):
        target_ids: List[int] = kwargs["target_item_id"]
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
        cnt_in_teacher = 0
        sum_reward = sum_fmt = sum_exist = sum_core = sum_shape = sum_pen = 0.0
        sum_rank = 0.0
        cnt_rank = 0

        top_heap, bot_heap = [], []
        seq = 0

        rewards = []
        dump_recs = []

        K = int(cfg.teacher_pool_k)

        for i, (comp, tgt_id, top_ids) in enumerate(zip(completions, target_ids, teacher_top_list)):
            comp_text = _to_text(comp)
            item_id, first_line, rest, prefix_ok, has_extra = extract_first_item_id(comp_text)

            r_fmt = 0.0
            r_exist = 0.0
            r_core = 0.0
            r_shape = 0.0
            pen = 0.0

            # format
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

            # exists
            exists = False
            if item_id is not None:
                if 1 <= int(item_id) <= int(n_items):
                    exists = True
                    cnt_exists += 1
                    r_exist = float(cfg.exists_bonus)
                else:
                    cnt_unknown += 1
                    r_core -= float(cfg.unknown_penalty)

            correct = False
            if exists:
                if int(item_id) == int(tgt_id):
                    correct = True
                    cnt_correct += 1
                    r_core += float(cfg.correct_reward)
                    # ✅ correct 时不做 teacher out-of-pool 惩罚，避免 target 不在 teacher_top 时伤害学习
                else:
                    r_core -= float(cfg.wrong_penalty)

                    # ✅ dense teacher shaping（更 recall）
                    # 只看 teacher_top 前 K（通常是按 teacher 分数降序）
                    top = list(top_ids)[:K] if isinstance(top_ids, list) else []
                    top = [int(x) for x in top if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())]
                    top = _dedup_keep_order(top)

                    id2rank = {pid: r for r, pid in enumerate(top)}  # best=0
                    if int(item_id) in id2rank and len(top) >= 2:
                        cnt_in_teacher += 1
                        rk = int(id2rank[int(item_id)])
                        score01 = 1.0 - (rk / max(1, (len(top) - 1)))  # 1..0
                        r_shape += float(cfg.alpha) * float(cfg.teacher_rank_weight) * float(score01)
                        sum_rank += float(rk)
                        cnt_rank += 1
                    else:
                        # 不在 teacher_top：给很小惩罚（可设 0 更开放）
                        r_shape -= float(cfg.alpha) * float(cfg.out_of_teacher_penalty)

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
                mean_rank = (sum_rank / cnt_rank) if cnt_rank else -1.0

                print("=" * 90)
                print(f"[DEBUG reward-phase2-recall] step={step} n={n} K={K}")
                print(
                    f"  parse_rate={d(cnt_parse,n):.3f} exists_rate={d(cnt_exists,n):.3f} "
                    f"correct_rate={d(cnt_correct,n):.6f} unknown_rate={d(cnt_unknown,n):.3f} "
                    f"in_teacher_rate(wrong_only)={d(cnt_in_teacher,n):.3f} mean_teacher_rank={mean_rank:.2f} "
                    f"prefix_bad_rate={d(cnt_prefix_bad,n):.3f} extra_rate={d(cnt_extra,n):.3f} "
                    f"group_unique_rate≈{mean_unique:.3f}"
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
