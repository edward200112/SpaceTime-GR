# HardMiningGRPO/eval_rerank_hr_fast.py
import os
import sys
import argparse
import random
from typing import Any, Dict, List, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from HardMiningGRPO.reward_sasrec import parse_namecat_keys

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
RULE_FALLBACK = "只输出一个地点名(类别)，不要解释"

CAND_HEADER_CANON = "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）："
CAND_HEADERS = [
    CAND_HEADER_CANON,
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）:",
    "候选地点（只能从下列候选中选 1 个，并原样输出；不要输出其他文字）",
    "候选地点：",
    "候选地点:",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_ckpt", required=True)
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--max_length", type=int, default=1280)
    ap.add_argument("--max_new_tokens", type=int, default=32)

    ap.add_argument("--eval_max_samples", type=int, default=2000, help="<=0 means full eval set")
    ap.add_argument("--prompt_bs", type=int, default=1, help="建议=1；>1 仅影响数据读取，不会做真正batch cache复用")
    ap.add_argument("--cand_bs", type=int, default=2, help="候选chunk大小（会自动降以避免OOM）")
    ap.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--shuffle_candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--print_target_pos_hist", action="store_true")
    ap.add_argument("--out_json", type=str, default="")
    return ap.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat_prompt(tok, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return USER_PREFIX_TEMPLATE.format(prompt=user_text)


def _get_field(ex: Dict[str, Any], keys, required=True, default=None):
    for k in keys:
        if k in ex:
            return ex[k]
    if required:
        raise KeyError(f"missing required field, tried: {keys}")
    return default


def _det_shuffle_pairs(pairs: List[Tuple[Any, Any]], seed: int):
    rng = random.Random(seed)
    rng.shuffle(pairs)


def _find_target_pos(target_namecat: str, candidate_namecats: List[str]) -> int:
    _, tgt_fold, ok = parse_namecat_keys(target_namecat)
    if not ok:
        tgt_fold = str(target_namecat).strip().casefold()

    for i, s in enumerate(candidate_namecats):
        _, k_fold, ok2 = parse_namecat_keys(s)
        if ok2:
            if k_fold == tgt_fold:
                return i
        else:
            if str(s).strip().casefold() == tgt_fold:
                return i
    return -1


def _build_candidate_block(cands: List[str]) -> str:
    lines = [CAND_HEADER_CANON]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def _rewrite_prompt_candidates(raw_prompt: str, cand_namecats: List[str]) -> str:
    p = (raw_prompt or "").rstrip()
    new_block = _build_candidate_block(cand_namecats)

    hit_pos = -1
    for hdr in CAND_HEADERS:
        pos = p.find(hdr)
        if pos != -1:
            hit_pos = pos
            break

    if hit_pos != -1:
        prefix = p[:hit_pos].rstrip()
        return (prefix + "\n" + new_block).rstrip()

    return (p + "\n" + new_block).rstrip()


def _repeat_past(past_key_values, n: int):
    """
    兼容 Transformers 新 Cache API：
    - Qwen2 返回的是 Cache 对象（有 get_seq_length），不能返回 tuple
    - 正确做法：Cache -> legacy tuple -> repeat -> Cache.from_legacy_cache()
    """
    if past_key_values is None or n == 1:
        return past_key_values

    # ✅ 新版 Cache：有 to_legacy_cache / from_legacy_cache
    if hasattr(past_key_values, "to_legacy_cache") and hasattr(past_key_values.__class__, "from_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()  # tuple(layers)
        rep_layers = []
        for layer in legacy:
            # layer 一般是 (k, v)，也可能有更多张量，统一处理
            rep = []
            for t in layer:
                if torch.is_tensor(t):
                    rep.append(t.repeat((n,) + (1,) * (t.dim() - 1)))
                else:
                    rep.append(t)
            rep_layers.append(tuple(rep))
        rep_legacy = tuple(rep_layers)
        return past_key_values.__class__.from_legacy_cache(rep_legacy)

    # fallback：如果是老式 tuple cache（理论上 Qwen2 不会走到这里）
    out = []
    for layer in past_key_values:
        rep = []
        for t in layer:
            if torch.is_tensor(t):
                rep.append(t.repeat((n,) + (1,) * (t.dim() - 1)))
            else:
                rep.append(t)
        out.append(tuple(rep))
    return tuple(out)



@torch.no_grad()
def score_one_prompt_with_cache(
    model,
    tok,
    prompt_text: str,
    cand_texts: List[str],
    max_total_len: int,
    cand_bs: int,
    device: str,
    length_norm: bool = True,
):
    """
    对一个 prompt 的 K 个候选，输出 [K] 分数（越大越好）
    用 KV cache：prompt forward 一次，然后候选只喂短token段。
    """
    pad_id = int(tok.pad_token_id)

    # encode candidates first to know max cand len
    cand_ids_list = [tok.encode(c, add_special_tokens=False) for c in cand_texts]
    cand_lens = [len(x) for x in cand_ids_list]
    max_cand_len = max(cand_lens) if cand_lens else 1

    # encode prompt & truncate from left to leave space for candidate
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    max_prompt_len = max(1, max_total_len - max_cand_len)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_attn = torch.ones_like(prompt_ids_t, dtype=torch.long, device=device)

    # prompt forward once (cache)
    outp = model(input_ids=prompt_ids_t, attention_mask=prompt_attn, use_cache=True)
    past = outp.past_key_values
    prompt_len = prompt_ids_t.size(1)

    last_logp = torch.log_softmax(outp.logits[:, -1, :], dim=-1)  # [1, V]

    K = len(cand_ids_list)
    scores = torch.empty((K,), dtype=torch.float32)

    bs = max(1, int(cand_bs))
    i = 0
    while i < K:
        cur = min(bs, K - i)
        chunk = cand_ids_list[i:i + cur]
        lens = cand_lens[i:i + cur]

        try:
            # first token probs from prompt last logits
            first_tokens = torch.tensor([x[0] if len(x) > 0 else pad_id for x in chunk],
                                        dtype=torch.long, device=device)  # [cur]
            first_lp = last_logp.expand(cur, -1).gather(1, first_tokens.view(-1, 1)).squeeze(1)  # [cur]

            # handle candidates with len==1 separately (no "rest")
            max_len = max(lens) if lens else 1
            if max_len <= 1:
                total_lp = first_lp
                if length_norm:
                    total_lp = total_lp / torch.tensor(lens, device=device, dtype=torch.float32).clamp_min(1.0)
                scores[i:i + cur] = total_lp.detach().float().cpu()
                i += cur
                continue

            # build padded cand ids [cur, max_len]
            cand_pad = []
            for ids in chunk:
                ids = ids[:max_len]
                if len(ids) < max_len:
                    ids = ids + [pad_id] * (max_len - len(ids))
                cand_pad.append(ids)
            cand_pad = torch.tensor(cand_pad, dtype=torch.long, device=device)  # [cur, max_len]

            # input is cand[:-1], labels is cand[1:]
            inp = cand_pad[:, :-1].contiguous()   # [cur, max_len-1]
            lab = cand_pad[:, 1:].contiguous()    # [cur, max_len-1]

            # mask out padding positions in labels
            # valid positions are < (len-1)
            label_mask = torch.zeros_like(lab, dtype=torch.bool)
            for r, L in enumerate(lens):
                if L >= 2:
                    label_mask[r, :L-1] = True
            lab = lab.masked_fill(~label_mask, -100)

            # attention_mask should cover (past + current)
            cand_attn = (inp != pad_id).long()  # [cur, max_len-1] (left padding not used here, right padding)
            full_attn = torch.cat(
                [torch.ones((cur, prompt_len), dtype=torch.long, device=device), cand_attn],
                dim=1
            )  # [cur, prompt_len + max_len-1]

            past_rep = _repeat_past(past, cur)
            outc = model(
                input_ids=inp,
                attention_mask=full_attn,
                past_key_values=past_rep,
                use_cache=False,
            )
            logits = outc.logits  # [cur, max_len-1, V]

            # NLL over remaining tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                lab.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(cur, -1)

            rest_nll = loss.sum(dim=1)  # [cur]
            rest_lp = -rest_nll

            total_lp = first_lp + rest_lp  # include first token prob

            if length_norm:
                denom = torch.tensor(lens, device=device, dtype=torch.float32).clamp_min(1.0)
                total_lp = total_lp / denom

            scores[i:i + cur] = total_lp.detach().float().cpu()
            i += cur

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[EVAL] OOM, reduce cand_bs -> {bs}", flush=True)

    return scores


@torch.no_grad()
def eval_hr(model, tok, eval_ds, max_samples, prompt_bs, cand_bs, max_total_len, device, length_norm=True):
    n = len(eval_ds)
    if max_samples and max_samples > 0:
        n = min(n, int(max_samples))

    hr1 = 0
    hr10 = 0
    total = 0

    for st in tqdm(range(0, n, prompt_bs), desc="eval"):
        ed = min(n, st + prompt_bs)
        batch = eval_ds.select(range(st, ed))

        for prompt, cands, tgt_pos in zip(batch["prompt"], batch["candidate_namecats"], batch["target_pos"]):
            sc = score_one_prompt_with_cache(
                model=model,
                tok=tok,
                prompt_text=prompt,
                cand_texts=cands,
                max_total_len=max_total_len,
                cand_bs=cand_bs,
                device=device,
                length_norm=length_norm,
            )
            sc_t = torch.tensor(sc)
            top10 = torch.topk(sc_t, k=min(10, sc_t.numel())).indices.tolist()
            top1 = top10[0]

            if top1 == int(tgt_pos):
                hr1 += 1
            if int(tgt_pos) in top10:
                hr10 += 1
            total += 1

    return {"hr1": hr1 / max(1, total), "hr10": hr10 / max(1, total), "samples": float(total)}


def main():
    args = parse_args()
    seed_everything(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, args.adapter_ckpt, is_trainable=False)
    model.eval()

    eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    def format_ex(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw = _get_field(ex, ["prompt"])
        raw = (raw or "").rstrip()
        if "只输出一个地点名" not in raw:
            raw = (raw + "\n" + RULE_FALLBACK).rstrip()

        cand_nc = _get_field(ex, ["candidate_namecats", "candidates_namecat", "candidates_namecats"])
        cand_it = _get_field(ex, ["candidate_item_ids", "candidates_item_ids", "candidates_ids"])
        pairs = list(zip(cand_nc, cand_it))

        if args.shuffle_candidates:
            _det_shuffle_pairs(pairs, seed=args.seed + int(idx))

        cand_nc2, cand_it2 = zip(*pairs)
        cand_nc2 = list(cand_nc2)

        tgt_nc = _get_field(ex, ["target_namecat"])
        tgt_pos = _find_target_pos(tgt_nc, cand_nc2)
        if tgt_pos < 0:
            raise ValueError("[BAD DATA] target_namecat not found in candidate_namecats.")

        raw2 = _rewrite_prompt_candidates(raw, cand_nc2)
        prompt_final = build_chat_prompt(tok, raw2) if args.use_chat_template else raw2

        return {
            "prompt": prompt_final,
            "candidate_namecats": cand_nc2,
            "target_pos": int(tgt_pos),
        }

    num_proc = max(1, (os.cpu_count() or 8) // 2)
    eval_ds = eval_ds.map(format_ex, with_indices=True, remove_columns=eval_ds.column_names, num_proc=num_proc)

    if args.print_target_pos_hist:
        cnt = Counter(eval_ds["target_pos"])
        total = sum(cnt.values())
        top10 = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n[TARGET_POS_HIST] eval total={total} top10(pos,count,ratio):")
        for pos, c in top10:
            print(f"  pos={pos:>3d}  count={c:>8d}  ratio={c/total:.4f}")

    device = str(next(model.parameters()).device)
    max_total_len = int(args.max_length) + int(args.max_new_tokens)

    metrics = eval_hr(
        model=model,
        tok=tok,
        eval_ds=eval_ds,
        max_samples=args.eval_max_samples,
        prompt_bs=args.prompt_bs,
        cand_bs=args.cand_bs,
        max_total_len=max_total_len,
        device=device,
        length_norm=args.length_norm,
    )

    print("\n" + "=" * 80)
    print(f"[EVAL DONE] adapter={args.adapter_ckpt}")
    print(f"  HR@1  = {metrics['hr1']:.6f}")
    print(f"  HR@10 = {metrics['hr10']:.6f}")
    print(f"  samples = {int(metrics['samples'])}")
    print("=" * 80, flush=True)

    if args.out_json:
        import json
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"adapter_ckpt": args.adapter_ckpt, "eval_jsonl": args.eval_jsonl, **metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[OK] wrote: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
