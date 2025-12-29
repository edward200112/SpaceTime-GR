import json
import re
import random
import heapq
import pickle
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
    # TRL completions 可能是 str / dict / list[dict]
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return _to_text(x[0])
    if isinstance(x, dict):
        for k in ("content", "text", "completion", "generated_text"):
            if k in x:
                return str(x[k])
    return str(x)

def extract_first_namecat(completion_text: str) -> Tuple[Optional[str], Optional[str], str, str, bool, bool]:
    """
    从第一行中找第一个 Name (Cat) 子串。
    return: name, cat, first_line, trail, prefix_ok, has_extra_text
    """
    t = _to_text(completion_text)
    lines = t.splitlines()
    first = norm_text(lines[0] if lines else t)

    m = NAMECAT_FIND_RE.search(first)
    if not m:
        # 没找到 Name(Cat)，prefix_ok=False, extra=False
        return None, None, first, "", False, False

    name = norm_text(m.group(1))
    cat = norm_text(m.group(2))

    prefix_ok = (first[:m.start()].strip() == "")
    trail = norm_text(first[m.end():])  # 同行尾巴

    has_extra = False
    if trail.strip():
        has_extra = True
    if len(lines) > 1 and any(norm_text(x) for x in lines[1:]):
        has_extra = True

    return name, cat, first, trail, prefix_ok, has_extra

def canon_key(name: str, cat: str) -> str:
    name = norm_text(name)
    cat = norm_text(cat)
    cat = cat.replace("centre", "center")
    return f"{name} ({cat})"

@dataclass
class ResolverConfig:
    # reward components
    n_neg_sample: int = 256
    softmax_temp: float = 1.0
    alpha: float = 0.3

    format_bonus: float = 0.05
    item_match_bonus: float = 0.2

    extra_text_penalty: float = 0.05
    unknown_penalty: float = 0.05
    prefix_penalty: float = 0.0

    # disamb speed control
    max_disamb_candidates: int = 64
    ensure_target_in_candidates: bool = True

    # debug
    debug_log_every_steps: int = 0
    debug_num_show: int = 5
    debug_dump_jsonl: str = ""
    debug_print_full_completion: bool = False

class SasrecResolver:
    """
    resolve completion -> pred_item_id
    支持：
      - namecat exact candidates
      - name-only fallback candidates
      - (训练) 强制把 target_item_id 注入候选，修复 top50 截断导致 item-match 不可达
      - 可选：从 sasrec_dataset.pkl 的 id2item + gmap_id2namecat 推导 target_namecat（防止数据缺字段）
    """
    def __init__(
        self,
        sasrec_model,
        n_items: int,
        namecat2item_disamb: Dict[str, List[int]],
        name2item_disamb: Dict[str, List[int]],
        sasrec_pkl_path: str,
        gmap_id2namecat: Optional[Dict[str, str]] = None,
        device: str = "cuda",
    ):
        self.sasrec = sasrec_model.to(device)
        self.sasrec.eval()
        self.n_items = int(n_items)
        self.namecat2item_disamb = namecat2item_disamb
        self.name2item_disamb = name2item_disamb
        self.device = device

        # load id2item (item_id -> gmap_id) from pkl for optional target_namecat derivation
        self.id2item = None
        try:
            with open(sasrec_pkl_path, "rb") as f:
                obj = pickle.load(f)
            self.id2item = obj.get("id2item", None)
        except Exception:
            self.id2item = None

        self.gmap_id2namecat = gmap_id2namecat

    def target_item_to_namecat(self, target_item_id: int) -> str:
        if self.id2item is None or self.gmap_id2namecat is None:
            return ""
        try:
            gmap = self.id2item.get(int(target_item_id), "")
            if not gmap:
                return ""
            return self.gmap_id2namecat.get(gmap, "") or ""
        except Exception:
            return ""

    @torch.no_grad()
    def _sasrec_score_candidates(self, history: List[int], candidate_ids: List[int]) -> torch.Tensor:
        if not history:
            return torch.zeros(len(candidate_ids), device=self.device)
        hist = torch.tensor(history, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,L]
        cand = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,C]
        scores = self.sasrec.predict_candidates(hist, cand).squeeze(0)  # [C]
        return scores

    def _prepare_candidates(
        self,
        base: Optional[List[int]],
        target_item_id: Optional[int],
        cfg: ResolverConfig,
    ) -> Optional[List[int]]:
        if not base:
            return None
        cands = [int(x) for x in base]

        if cfg.ensure_target_in_candidates and target_item_id is not None:
            tid = int(target_item_id)
            if tid not in cands:
                cands.append(tid)

        if len(cands) > int(cfg.max_disamb_candidates):
            tid = int(target_item_id) if target_item_id is not None else None
            keep = []
            if tid is not None and tid in cands:
                keep.append(tid)
                cands = [x for x in cands if x != tid]
            random.shuffle(cands)
            keep.extend(cands[: max(0, int(cfg.max_disamb_candidates) - len(keep))])
            cands = keep

        return cands

    @torch.no_grad()
    def resolve_to_item_id(
        self,
        completion_text: str,
        history_item_ids: List[int],
        target_item_id: Optional[int],
        cfg: ResolverConfig,
    ) -> Tuple[Optional[int], str, bool, str, str, bool, bool]:
        """
        return:
          pred_item_id or None,
          key (canon namecat or name),
          prefix_ok,
          via in {"exact","name_only","none"},
          trail,
          has_namecat,
          has_extra_text
        """
        name, cat, first_line, trail, prefix_ok, has_extra = extract_first_namecat(completion_text)
        if name is None or cat is None:
            return None, first_line, False, "none", "", False, False

        key = canon_key(name, cat)

        base = self.namecat2item_disamb.get(key)
        cands = self._prepare_candidates(base, target_item_id, cfg)
        if cands:
            if len(cands) == 1:
                return int(cands[0]), key, prefix_ok, "exact", trail, True, has_extra
            scores = self._sasrec_score_candidates(history_item_ids, cands)
            best_idx = int(torch.argmax(scores).item())
            return int(cands[best_idx]), key, prefix_ok, "exact", trail, True, has_extra

        base2 = self.name2item_disamb.get(norm_text(name))
        cands2 = self._prepare_candidates(base2, target_item_id, cfg)
        if cands2:
            if len(cands2) == 1:
                return int(cands2[0]), norm_text(name), prefix_ok, "name_only", trail, True, has_extra
            scores = self._sasrec_score_candidates(history_item_ids, cands2)
            best_idx = int(torch.argmax(scores).item())
            return int(cands2[best_idx]), norm_text(name), prefix_ok, "name_only", trail, True, has_extra

        return None, key, prefix_ok, "none", trail, True, has_extra

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

        # stats
        cnt_prefix_ok = cnt_resolved = cnt_match_nc = cnt_match_item = cnt_unknown = cnt_extra = 0
        via_cnt = {"exact": 0, "name_only": 0, "none": 0}

        sum_reward = sum_fmt = sum_match_nc = sum_match_item = sum_soft = sum_pen = 0.0

        top_heap = []
        bot_heap = []
        seq = 0

        rewards = []
        for i, (comp, hist, tgt) in enumerate(zip(completions, histories, targets)):
            comp_text = _to_text(comp)
            tgt = int(tgt)

            # target_namecat: 优先用数据字段，否则用 target_item_id 推导
            tgt_nc = ""
            if target_namecats is not None and i < len(target_namecats):
                tgt_nc = norm_text(target_namecats[i])
            if not tgt_nc:
                tgt_nc = norm_text(resolver.target_item_to_namecat(tgt))

            pred_item, out_key, prefix_ok, via, trail, has_namecat, has_extra = resolver.resolve_to_item_id(
                comp_text, hist, tgt, cfg
            )

            if prefix_ok:
                cnt_prefix_ok += 1

            via_cnt[via] = via_cnt.get(via, 0) + 1

            r_fmt = float(cfg.format_bonus) if has_namecat else 0.0

            pen = 0.0
            if has_extra:
                pen -= float(cfg.extra_text_penalty)
                cnt_extra += 1
            if has_namecat and (not prefix_ok) and float(cfg.prefix_penalty) > 0:
                pen -= float(cfg.prefix_penalty)

            r_match_nc = 0.0
            # namecat match（主信号）
            if has_namecat and tgt_nc:
                if norm_text(out_key) == norm_text(tgt_nc):
                    r_match_nc = 1.0
                    cnt_match_nc += 1

            r_match_item = 0.0
            r_soft = 0.0

            if pred_item is None:
                if has_namecat:
                    cnt_unknown += 1
                    pen -= float(cfg.unknown_penalty)
                r = r_fmt + r_match_nc + pen
            else:
                cnt_resolved += 1
                if int(pred_item) == int(tgt):
                    r_match_item = float(cfg.item_match_bonus)
                    cnt_match_item += 1
                r_soft = resolver.sasrec_soft_reward(hist, int(pred_item), int(tgt), cfg)
                r = r_fmt + r_match_nc + r_match_item + float(cfg.alpha) * float(r_soft) + pen

            r = float(r)
            rewards.append(r)

            sum_reward += r
            sum_fmt += float(r_fmt)
            sum_match_nc += float(r_match_nc)
            sum_match_item += float(r_match_item)
            sum_soft += float(r_soft) if pred_item is not None else 0.0
            sum_pen += float(pen)

            # debug record
            lines = comp_text.splitlines()
            first_line = norm_text(lines[0] if lines else comp_text)

            rec = {
                "reward": r,
                "via": via,
                "first": first_line,
                "trail": trail,
                "out_key": out_key,
                "tgt_namecat": tgt_nc,
                "pred_item": pred_item,
                "tgt_item": int(tgt),
                "fmt": float(r_fmt),
                "match_nc": float(r_match_nc),
                "match_item_bonus": float(r_match_item),
                "soft": float(r_soft) if pred_item is not None else 0.0,
                "pen": float(pen),
                "full_completion": comp_text if cfg.debug_print_full_completion else "",
            }

            seq += 1
            key_top = (r, seq, rec)
            if len(top_heap) < cfg.debug_num_show:
                heapq.heappush(top_heap, key_top)
            else:
                if r > top_heap[0][0]:
                    heapq.heapreplace(top_heap, key_top)

            key_bot = (-r, seq, rec)
            if len(bot_heap) < cfg.debug_num_show:
                heapq.heappush(bot_heap, key_bot)
            else:
                if -r > bot_heap[0][0]:
                    heapq.heapreplace(bot_heap, key_bot)

        # debug print
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
                    f"item_bonus_hit_rate={d(cnt_match_item,n):.3f} "
                    f"unknown_rate={d(cnt_unknown,n):.3f} "
                    f"extra_text_rate={d(cnt_extra,n):.3f}"
                )
                print(f"  via: exact={via_cnt.get('exact',0)} name_only={via_cnt.get('name_only',0)} none={via_cnt.get('none',0)}")
                print(
                    f"  mean_reward={d(sum_reward,n):.4f} "
                    f"mean_fmt={d(sum_fmt,n):.4f} "
                    f"mean_match_nc={d(sum_match_nc,n):.4f} "
                    f"mean_item_bonus={d(sum_match_item,n):.4f} "
                    f"mean_soft(resolved)={d(sum_soft,max(1,cnt_resolved)):.4f} "
                    f"mean_penalty={d(sum_pen,n):.4f}"
                )

                top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                bot_sorted = sorted(bot_heap, key=lambda x: x[0])

                print("\n  [TOP examples]")
                for rr, _, rec in top_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} first='{rec['first'][:80]}' "
                        f"trail='{rec['trail'][:40]}' out='{rec['out_key'][:60]}' tgt='{rec['tgt_namecat'][:60]}' "
                        f"pred={rec['pred_item']} tgt_id={rec['tgt_item']} "
                        f"(fmt={rec['fmt']:.2f}, nc={rec['match_nc']:.1f}, itemB={rec['match_item_bonus']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )

                print("\n  [BOTTOM examples]")
                for neg_rr, _, rec in bot_sorted:
                    print(
                        f"    r={rec['reward']:.4f} via={rec['via']} first='{rec['first'][:80]}' "
                        f"trail='{rec['trail'][:40]}' out='{rec['out_key'][:60]}' tgt='{rec['tgt_namecat'][:60]}' "
                        f"pred={rec['pred_item']} tgt_id={rec['tgt_item']} "
                        f"(fmt={rec['fmt']:.2f}, nc={rec['match_nc']:.1f}, itemB={rec['match_item_bonus']:.1f}, soft={rec['soft']:.3f}, pen={rec['pen']:.2f})"
                    )
                print("=" * 90)

        # optional dump
        if cfg.debug_dump_jsonl:
            try:
                with open(cfg.debug_dump_jsonl, "a", encoding="utf-8") as f:
                    # 只 dump top/bot，不写全量
                    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
                    bot_sorted = sorted(bot_heap, key=lambda x: x[0])
                    payload = {
                        "step": step,
                        "top": [rec for _, _, rec in top_sorted],
                        "bottom": [rec for _, _, rec in bot_sorted],
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

        return rewards

    return reward_fn
