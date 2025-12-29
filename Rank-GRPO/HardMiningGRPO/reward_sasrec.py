import json
import re
import random
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

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
    cat = cat.replace("centre", "center")
    return f"{name} ({cat})"

def canonize_namecat_text(s: str) -> str:
    """把 target_namecat / 输出都规范到同一种 key 格式，减少因为空格/符号导致的假不匹配。"""
    name, cat, first, _, ok = extract_first_namecat(s)
    if name is None or cat is None:
        return norm_text(s)
    return canon_key(name, cat)

@dataclass
class ResolverConfig:
    n_neg_sample: int = 256
    softmax_temp: float = 1.0
    alpha: float = 0.3
    format_bonus: float = 0.05

    item_match_bonus: float = 0.2  # ✅ 只有 namecat match 时才会给

    extra_text_penalty: float = 0.05
    unknown_penalty: float = 0.05
    prefix_penalty: float = 0.0

    max_disamb_candidates: int = 64
    ensure_target_in_candidates: bool = False  # ✅ 注意：也要“条件注入”，见 resolver 实现

    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False


class SasrecResolver:
    def __init__(
        self,
        sasrec_model,
        n_items: int,
        namecat2item_disamb: Dict[str, List[int]],
        name2item_disamb: Dict[str, List[int]],
        namecat2item_all: Optional[Dict[str, List[int]]] = None,
        name2item_all: Optional[Dict[str, List[int]]] = None,
        device: str = "cuda",
    ):
        self.sasrec = sasrec_model.to(device)
        self.sasrec.eval()
        self.n_items = int(n_items)

        self.namecat2item_disamb = namecat2item_disamb
        self.name2item_disamb = name2item_disamb

        self.namecat2item_all = namecat2item_all or {}
        self.name2item_all = name2item_all or {}

        self.device = device

    @torch.no_grad()
    def _sasrec_score_candidates(self, history: List[int], candidate_ids: List[int]) -> torch.Tensor:
        if len(history) == 0:
            return torch.zeros(len(candidate_ids), device=self.device)

        hist = torch.tensor(history, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,L]
        cand = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,C]
        scores = self.sasrec.predict_candidates(hist, cand).squeeze(0)  # [C]
        return scores

    def _target_belongs(self, all_list: Optional[List[int]], target_item_id: Optional[int], fallback_list: Optional[List[int]]) -> bool:
        if target_item_id is None:
            return False
        tid = int(target_item_id)
        if all_list:
            return tid in set(map(int, all_list))
        if fallback_list:
            return tid in set(map(int, fallback_list))
        return False

    def _prepare_candidates(
        self,
        disamb_list: Optional[List[int]],
        all_list: Optional[List[int]],
        target_item_id: Optional[int],
        cfg: ResolverConfig,
    ) -> Optional[List[int]]:
        if not disamb_list and not all_list:
            return None

        # 优先用 disamb（更小）；如果 disamb 很小且缺 target，但 all_list 证明 target 属于该 key，
        # 就从 all_list 补一些（避免 top50 截断）。
        base = list(map(int, disamb_list)) if disamb_list else []
        allv = list(map(int, all_list)) if all_list else []

        tid = int(target_item_id) if target_item_id is not None else None
        belongs = self._target_belongs(allv, tid, base)

        # ✅ 条件注入：只有 belongs 才允许注入 target
        if cfg.ensure_target_in_candidates and belongs and tid is not None:
            if tid not in base:
                base.append(tid)

        # 如果 disamb 为空，就直接用 all_list（但要截断）
        if not base and allv:
            base = allv

        # 如果 all_list 很大（>max），但需要保证 target 有机会参与：
        # 这里策略：保留 base（含 target）+ 从 allv 里随机补齐到 max
        if len(base) > cfg.max_disamb_candidates:
            # 截断时优先保留 target
            if tid is not None and tid in base:
                keep = [tid]
                rest = [x for x in base if x != tid]
                random.shuffle(rest)
                keep.extend(rest[: max(0, cfg.max_disamb_candidates - 1)])
                base = keep
            else:
                random.shuffle(base)
                base = base[: cfg.max_disamb_candidates]
        else:
            # base 不满 max，且 allv 存在的话可适度补充（但没必要太大）
            if allv and len(base) < cfg.max_disamb_candidates:
                pool = [x for x in allv if x not in set(base)]
                random.shuffle(pool)
                base.extend(pool[: max(0, cfg.max_disamb_candidates - len(base))])

        return base

    @torch.no_grad()
    def resolve_to_item_id(
        self,
        completion_text: str,
        history_item_ids: List[int],
        target_item_id: Optional[int],
        cfg: ResolverConfig,
    ) -> Tuple[Optional[int], str, bool, str, str]:
        """
        return:
          pred_item_id or None,
          key,
          prefix_ok,
          via in {"exact","name_only","none"},
          trail
        """
        name, cat, first_line, trail, prefix_ok = extract_first_namecat(completion_text)
        if name is None or cat is None:
            return None, first_line, False, "none", ""

        key = canon_key(name, cat)

        disamb = self.namecat2item_disamb.get(key)
        allv = self.namecat2item_all.get(key)
        cands = self._prepare_candidates(disamb, allv, target_item_id, cfg)

        if cands:
            if len(cands) == 1:
                return int(cands[0]), key, prefix_ok, "exact", trail
            scores = self._sasrec_score_candidates(history_item_ids, cands)
            best_idx = int(torch.argmax(scores).item())
            return int(cands[best_idx]), key, prefix_ok, "exact", trail

        # name-only fallback
        name_only = norm_text(name)
        disamb2 = self.name2item_disamb.get(name_only)
        allv2 = self.name2item_all.get(name_only)
        cands2 = self._prepare_candidates(disamb2, allv2, target_item_id, cfg)

        if cands2:
            if len(cands2) == 1:
                return int(cands2[0]), name_only, prefix_ok, "name_only", trail
            scores = self._sasrec_score_candidates(history_item_ids, cands2)
            best_idx = int(torch.argmax(scores).item())
            return int(cands2[best_idx]), name_only, prefix_ok, "name_only", trail

        return None, key, prefix_ok, "none", trail

    @torch.no_grad()
    def sasrec_soft_reward(self, history: List[int], pred_item_id: int, target_item_id: int, cfg: ResolverConfig) -> float:
        pool = {int(pred_item_id), int(target_item_id)}
        while len(pool) < 2 + cfg.n_neg_sample:
            x = random.randint(1, self.n_items)
            pool.add(x)
        pool_list = list(pool)

        scores = self._sasrec_score_candidates(history, pool_list) / float(cfg.softmax_temp)
        probs = torch.softmax(scores, dim=0)
        pred_pos = pool_list.index(int(pred_item_id))
        return float(probs[pred_pos].item())


def make_reward_fn(resolver: SasrecResolver, cfg: ResolverConfig):
    state = {"last_logged_step": None, "_fallback_step": 0}

    def reward_fn(prompts, completions, **kwargs):
        histories: List[List[int]] = kwargs["history_item_ids"]
        targets: List[int] = kwargs["target_item_id"]
        target_namecats: Optional[List[str]] = kwargs.get("target_namecat", None)

        step = kwargs.get("step", None)
        if step is None:
            step = state["_fallback_step"]
            state["_fallback_step"] += 1
        step = int(step)

        n = len(completions)

        cnt_prefix_ok = cnt_resolved = cnt_match_nc = cnt_itemB = cnt_unknown = cnt_extra = 0
        via_cnt = {"exact": 0, "name_only": 0, "none": 0}

        sum_reward = sum_fmt = sum_match_nc = sum_itemB = sum_soft = sum_pen = 0.0

        top_heap = []
        bot_heap = []
        seq = 0

        rewards = []
        for i, (comp, hist, tgt) in enumerate(zip(completions, histories, targets)):
            comp_text = _to_text(comp)
            lines = comp_text.splitlines()
            first_line = norm_text(lines[0] if lines else comp_text)

            tgt = int(tgt)
            tgt_nc = target_namecats[i] if (target_namecats is not None and i < len(target_namecats)) else ""
            tgt_key = canonize_namecat_text(tgt_nc) if tgt_nc else ""

            name, cat, _, trail, prefix_ok = extract_first_namecat(comp_text)
            has_namecat = (name is not None and cat is not None)
            out_key = canon_key(name, cat) if has_namecat else ""

            if prefix_ok:
                cnt_prefix_ok += 1

            r_fmt = float(cfg.format_bonus) if has_namecat else 0.0

            extra = False
            if has_namecat:
                if trail.strip():
                    extra = True
                if len(lines) > 1 and any(norm_text(x) for x in lines[1:]):
                    extra = True

            pen = 0.0
            if not prefix_ok and has_namecat and float(cfg.prefix_penalty) > 0:
                pen -= float(cfg.prefix_penalty)

            if extra:
                pen -= float(cfg.extra_text_penalty)
                cnt_extra += 1

            pred_item, _, _, via, _ = resolver.resolve_to_item_id(
                comp_text, hist, tgt, cfg
            )
            via_cnt[via] = via_cnt.get(via, 0) + 1

            # ✅ namecat match 为主
            r_match_nc = 0.0
            if has_namecat and tgt_key:
                if norm_text(out_key) == norm_text(tgt_key):
                    r_match_nc = 1.0
                    cnt_match_nc += 1

            r_itemB = 0.0
            r_soft = 0.0

            if pred_item is None:
                if has_namecat:
                    cnt_unknown += 1
                    pen -= float(cfg.unknown_penalty)
                r = r_fmt + r_match_nc + pen
            else:
                cnt_resolved += 1

                # ✅ 关键：item bonus 必须绑定 namecat match，防止“钻候选注入”的漏洞
                if r_match_nc > 0.5 and int(pred_item) == int(tgt):
                    r_itemB = float(cfg.item_match_bonus)
                    cnt_itemB += 1

                r_soft = resolver.sasrec_soft_reward(hist, int(pred_item), int(tgt), cfg)
                r = r_fmt + r_match_nc + r_itemB + float(cfg.alpha) * float(r_soft) + pen

            rewards.append(float(r))

            sum_reward += float(r)
            sum_fmt += float(r_fmt)
            sum_match_nc += float(r_match_nc)
            sum_itemB += float(r_itemB)
            sum_soft += float(r_soft) if pred_item is not None else 0.0
            sum_pen += float(pen)

            rec = {
                "reward": float(r),
                "via": via,
                "first": first_line,
                "trail": trail,
                "out_key": out_key,
                "tgt_key": tgt_key,
                "pred_item": pred_item,
                "tgt_item": int(tgt),
                "fmt": float(r_fmt),
                "match_nc": float(r_match_nc),
                "itemB": float(r_itemB),
                "soft": float(r_soft) if pred_item is not None else 0.0,
                "pen": float(pen),
                "full_completion": comp_text if cfg.debug_print_full_completion else "",
            }

            seq += 1
            key_top = (float(r), seq, rec)
            if len(top_heap) < cfg.debug_num_show:
                heapq.heappush(top_heap, key_top)
            else:
                if float(r) > top_heap[0][0]:
                    heapq.heapreplace(top_heap, key_top)

            key_bot = (-float(r), seq, rec)
            if len(bot_heap) < cfg.debug_num_show:
                heapq.heappush(bot_heap, key_bot)
            else:
                if -float(r) > bot_heap[0][0]:
                    heapq.heapreplace(bot_heap, key_bot)

        if cfg.debug_log_every_steps > 0:
            if (state["last_logged_step"] is None or step != state["last_logged_step"]) and (step % cfg.debug_log_every_steps == 0):
                state["last_logged_step"] = step

                def d(a, b): return a / b if b else 0.0
                print("=" * 90)
                print(f"[DEBUG reward] step={step} n={n}")
                print(
                    f"  prefix_ok_rate={d(cnt_prefix_ok,n):.3f} "
                    f"resolved_rate={d(cnt_resolved,n):.3f} "
                    f"match_namecat_rate={d(cnt_match_nc,n):.3f} "
                    f"item_bonus_hit_rate={d(cnt_itemB,n):.3f} "
                    f"unknown_rate={d(cnt_unknown,n):.3f} "
                    f"extra_text_rate={d(cnt_extra,n):.3f}"
                )
                print(f"  via: exact={via_cnt.get('exact',0)} name_only={via_cnt.get('name_only',0)} none={via_cnt.get('none',0)}")
                print(
                    f"  mean_reward={d(sum_reward,n):.4f} "
                    f"mean_fmt={d(sum_fmt,n):.4f} "
                    f"mean_match_nc={d(sum_match_nc,n):.4f} "
                    f"mean_itemB={d(sum_itemB,n):.4f} "
                    f"mean_soft(resolved)={d(sum_soft,max(1,cnt_resolved)):.4f} "
                    f"mean_penalty={d(sum_pen,n):.4f}"
                )

                top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                bot_sorted = sorted(bot_heap, key=lambda x: x[0])

                print("\n  [TOP examples]")
                for rr, _, rec in top_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} first='{rec['first'][:80]}' "
                        f"trail='{rec['trail'][:40]}' out='{rec['out_key'][:60]}' tgt='{rec['tgt_key'][:60]}' "
                        f"pred={rec['pred_item']} tgt_id={rec['tgt_item']} "
                        f"(fmt={rec['fmt']:.2f}, nc={rec['match_nc']:.1f}, itemB={rec['itemB']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )

                print("\n  [BOTTOM examples]")
                for neg_rr, _, rec in bot_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} first='{rec['first'][:80]}' "
                        f"trail='{rec['trail'][:40]}' out='{rec['out_key'][:60]}' tgt='{rec['tgt_key'][:60]}' "
                        f"pred={rec['pred_item']} tgt_id={rec['tgt_item']} "
                        f"(fmt={rec['fmt']:.2f}, nc={rec['match_nc']:.1f}, itemB={rec['itemB']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )
                print("=" * 90)

        return rewards

    return reward_fn
