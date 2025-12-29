# HardMiningGRPO/train_grpo.py
import os
import sys
import argparse
import random
from typing import Any, Dict, List, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import PeftModel

from trl import GRPOTrainer, GRPOConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TeacherModel.SASRec import SASRec
from HardMiningGRPO.reward_sasrec import SasrecScorer, ResolverConfig, make_reward_fn, parse_namecat_keys

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
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)

    ap.add_argument("--sasrec_pkl", required=True)
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)

    # GRPO
    ap.add_argument("--max_length", type=int, default=1280)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--num_train_epochs", type=int, default=1)

    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)

    # ✅ HR eval (callback)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--eval_max_samples", type=int, default=2000, help="<=0 means use full eval set.")
    ap.add_argument("--eval_prompt_bs", type=int, default=4, help="how many prompts per eval batch.")
    ap.add_argument("--eval_score_bs", type=int, default=64, help="how many (prompt,cand) sequences per forward.")
    ap.add_argument("--eval_length_norm", action=argparse.BooleanOptionalAction, default=True)

    # ✅ 去捷径：打散 candidates，并同步重写 prompt 的候选段
    ap.add_argument("--shuffle_candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print_target_pos_hist", action="store_true")

    # ✅ reward cfg（匹配你当前 reward_sasrec.py 的 ResolverConfig）
    ap.add_argument("--format_bonus", type=float, default=0.05)
    ap.add_argument("--in_candidates_bonus", type=float, default=0.10)

    ap.add_argument("--match_reward_exact", type=float, default=0.25)
    ap.add_argument("--match_reward_fold", type=float, default=0.03)

    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--softmax_temp", type=float, default=1.0)
    ap.add_argument("--teacher_mode", type=str, default="zscore", choices=["zscore", "logprob", "prob", "rank"])
    ap.add_argument("--teacher_clip", type=float, default=5.0)

    ap.add_argument("--extra_text_penalty", type=float, default=0.05)
    ap.add_argument("--unknown_penalty", type=float, default=0.10)
    ap.add_argument("--prefix_penalty", type=float, default=0.05)
    ap.add_argument("--incomplete_penalty", type=float, default=0.10)
    ap.add_argument("--copy_penalty", type=float, default=0.08)
    ap.add_argument("--duplicate_penalty", type=float, default=0.02)

    # sasrec arch
    ap.add_argument("--sasrec_embed_dim", type=int, default=128)
    ap.add_argument("--sasrec_num_blocks", type=int, default=2)
    ap.add_argument("--sasrec_num_heads", type=int, default=2)
    ap.add_argument("--sasrec_dropout", type=float, default=0.2)
    ap.add_argument("--sasrec_max_len", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    # debug
    ap.add_argument("--debug_log_every_steps", type=int, default=0)
    ap.add_argument("--debug_num_show", type=int, default=5)
    ap.add_argument("--debug_dump_jsonl", type=str, default="")
    ap.add_argument("--debug_print_full_completion", action="store_true")

    ap.add_argument("--use_chat_template", action="store_true")

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


def load_sasrec_from_ckpt(
    sasrec_pkl: str,
    sasrec_ckpt: str,
    device: str,
    max_len: int = 50,
    embed_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
):
    import pickle

    with open(sasrec_pkl, "rb") as f:
        obj = pickle.load(f)
    n_items = int(obj["n_items"])

    try:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(sasrec_ckpt, map_location="cpu")

    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state_dict = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state_dict = ckpt_obj["model_state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            state_dict = ckpt_obj["model"]
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    class _Args:
        pass

    a = _Args()
    a.device = device
    a.max_len = int(max_len)
    a.embed_dim = int(embed_dim)
    a.num_blocks = int(num_blocks)
    a.num_heads = int(num_heads)
    a.dropout = float(dropout)

    sasrec = SASRec(item_num=n_items, args=a).to(device)
    sasrec.load_state_dict(state_dict, strict=True)
    sasrec.eval()
    for p in sasrec.parameters():
        p.requires_grad_(False)

    print(f"[OK] loaded SASRec: n_items={n_items}, max_len={a.max_len}, dim={a.embed_dim}, "
          f"blocks={a.num_blocks}, heads={a.num_heads}, dropout={a.dropout}")
    return sasrec


# -------------------------
# HR@K evaluation utilities
# -------------------------

@torch.no_grad()
def _score_sequences_avg_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    length_norm: bool = True,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, L, V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, Lm1 = shift_labels.shape
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(B, Lm1)

    mask = (shift_labels != -100).float()
    tok_cnt = mask.sum(dim=1).clamp_min(1.0)
    nll = (loss * mask).sum(dim=1)

    if length_norm:
        return -(nll / tok_cnt)
    return -nll


def _encode_prompt_and_candidate(tok, prompt: str, cand: str):
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(cand, add_special_tokens=False)
    return p_ids, c_ids


@torch.no_grad()
def score_candidates_for_prompts(
    model,
    tok,
    prompts: List[str],
    candidates: List[List[str]],
    max_total_len: int,
    score_bs: int,
    device: str,
    length_norm: bool = True,
) -> torch.Tensor:
    B = len(prompts)
    K = len(candidates[0])
    assert all(len(x) == K for x in candidates), "Eval expects fixed K per sample."

    flat_input_ids = []
    flat_labels = []
    flat_attn = []

    for b in range(B):
        prompt = prompts[b]
        for k in range(K):
            cand = candidates[b][k]
            p_ids, c_ids = _encode_prompt_and_candidate(tok, prompt, cand)

            ids = p_ids + c_ids

            # truncate from left; keep candidate tail
            if len(ids) > max_total_len:
                if len(c_ids) <= max_total_len:
                    ids = ids[-max_total_len:]
                    ids = ids[:-len(c_ids)] + c_ids
                else:
                    ids = c_ids[-max_total_len:]

            cand_len = min(len(c_ids), len(ids))
            prompt_len = len(ids) - cand_len

            labels = [-100] * len(ids)
            for t in range(prompt_len, len(ids)):
                labels[t] = ids[t]

            flat_input_ids.append(ids)
            flat_labels.append(labels)

    # ✅ 左 padding（与你 tok.padding_side="left" 对齐）
    pad_id = int(tok.pad_token_id)
    max_len = max(len(x) for x in flat_input_ids)
    for i in range(len(flat_input_ids)):
        ids = flat_input_ids[i]
        lab = flat_labels[i]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids = [pad_id] * pad_n + ids
            lab = [-100] * pad_n + lab
            attn = [0] * pad_n + attn
        flat_input_ids[i] = ids
        flat_labels[i] = lab
        flat_attn.append(attn)

    input_ids = torch.tensor(flat_input_ids, dtype=torch.long, device=device)
    labels = torch.tensor(flat_labels, dtype=torch.long, device=device)
    attn = torch.tensor(flat_attn, dtype=torch.long, device=device)

    scores = []
    N = input_ids.size(0)
    for s in range(0, N, score_bs):
        e = min(N, s + score_bs)
        sc = _score_sequences_avg_logprob(
            model,
            input_ids[s:e],
            attn[s:e],
            labels[s:e],
            length_norm=length_norm,
        )
        scores.append(sc.detach().float().cpu())

    scores = torch.cat(scores, dim=0)  # [B*K]
    return scores.view(B, K)


@torch.no_grad()
def compute_hr_at_k(
    model,
    tok,
    eval_ds,
    max_samples: int,
    prompt_bs: int,
    score_bs: int,
    max_total_len: int,
    device: str,
    length_norm: bool = True,
) -> Dict[str, float]:
    n = len(eval_ds)
    if max_samples and max_samples > 0:
        n = min(n, int(max_samples))

    hr1_hit = 0
    hr10_hit = 0
    total = 0

    was_train = model.training
    model.eval()

    for st in range(0, n, prompt_bs):
        ed = min(n, st + prompt_bs)
        batch = eval_ds.select(range(st, ed))

        prompts = list(batch["prompt"])
        candidates = list(batch["candidate_namecats"])
        tgt_pos = torch.tensor(batch["target_pos"], dtype=torch.long)

        sc = score_candidates_for_prompts(
            model=model,
            tok=tok,
            prompts=prompts,
            candidates=candidates,
            max_total_len=max_total_len,
            score_bs=score_bs,
            device=device,
            length_norm=length_norm,
        )

        top10 = torch.topk(sc, k=min(10, sc.size(1)), dim=1).indices
        top1 = top10[:, 0]

        hr1_hit += int((top1 == tgt_pos).sum().item())
        hr10_hit += int((top10 == tgt_pos.unsqueeze(1)).any(dim=1).sum().item())
        total += (ed - st)

    if was_train:
        model.train()

    return {
        "eval_hr1": hr1_hit / max(1, total),
        "eval_hr10": hr10_hit / max(1, total),
        "eval_samples": float(total),
    }


class GRPOTrainerWithHREval(GRPOTrainer):
    def __init__(self, *args, tokenizer=None, eval_max_samples=0, eval_prompt_bs=4, eval_score_bs=64,
                 eval_length_norm=True, max_total_len=1408, **kwargs):
        super().__init__(*args, **kwargs)
        self._hr_tok = tokenizer
        self._eval_max_samples = int(eval_max_samples)
        self._eval_prompt_bs = int(eval_prompt_bs)
        self._eval_score_bs = int(eval_score_bs)
        self._eval_length_norm = bool(eval_length_norm)
        self._max_total_len = int(max_total_len)

    def evaluate(self, eval_dataset=None, **kwargs):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if ds is None:
            return {}

        device = next(self.model.parameters()).device
        metrics = compute_hr_at_k(
            model=self.model,
            tok=self._hr_tok,
            eval_ds=ds,
            max_samples=self._eval_max_samples,
            prompt_bs=self._eval_prompt_bs,
            score_bs=self._eval_score_bs,
            max_total_len=self._max_total_len,
            device=str(device),
            length_norm=self._eval_length_norm,
        )
        # ✅ 显式打印 + log，保证你一定能看到
        print(f"[EVAL @ step={self.state.global_step}] {metrics}", flush=True)
        try:
            self.log(metrics)
        except Exception:
            pass
        return metrics


import torch.distributed as dist
from transformers import TrainerCallback

class EvalEveryNStepsCallback(TrainerCallback):
    """每 N 个 global_step 手动触发 trainer.evaluate()（不依赖 GRPOConfig 支持 eval 参数）"""
    def __init__(self, trainer, every_n_steps: int):
        self.trainer = trainer
        self.every = int(every_n_steps)
        self._last = -1

    def on_step_end(self, args, state, control, **kwargs):
        # 多卡时只让 rank0 eval（单卡也兼容）
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return control

        gs = int(getattr(state, "global_step", 0))
        if self.every > 0 and gs > 0 and (gs % self.every == 0) and (gs != self._last):
            self._last = gs
            _ = self.trainer.evaluate()  # evaluate() 里我们会 print eval_hr1/10
        return control



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    generation_batch_size = int(args.per_device_bs) * int(args.grad_accum)
    if generation_batch_size % int(args.num_generations) != 0:
        raise ValueError(
            f"[BAD CONFIG] per_device_bs*grad_accum={generation_batch_size} "
            f"must be divisible by num_generations={args.num_generations}."
        )

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cuda",
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.print_trainable_parameters()

    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(int(tok.eos_token_id))
    for t in ["<|im_end|>", "</s>"]:
        try:
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(int(tid))
        except Exception:
            pass
    eos_ids = sorted(set(eos_ids)) if eos_ids else None
    if eos_ids is not None:
        model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = int(tok.pad_token_id)

    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    def format_ex(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw = _get_field(ex, ["prompt"])
        raw = (raw or "").rstrip()
        if "只输出一个地点名" not in raw:
            raw = (raw + "\n" + RULE_FALLBACK).rstrip()

        cand_nc = _get_field(ex, ["candidate_namecats", "candidates_namecat", "candidates_namecats"])
        cand_it = _get_field(ex, ["candidate_item_ids", "candidates_item_ids", "candidates_ids"])
        if not isinstance(cand_nc, list) or not isinstance(cand_it, list) or len(cand_nc) != len(cand_it) or len(cand_nc) == 0:
            raise ValueError(f"[BAD DATA] candidate lists invalid: len(namecats)={len(cand_nc)} len(item_ids)={len(cand_it)}")

        pairs = list(zip(cand_nc, cand_it))
        if args.shuffle_candidates:
            _det_shuffle_pairs(pairs, seed=args.seed + int(idx))

        cand_nc2, cand_it2 = zip(*pairs)
        cand_nc2 = list(cand_nc2)
        cand_it2 = [int(x) for x in cand_it2]

        tgt_nc = _get_field(ex, ["target_namecat"])
        tgt_pos = _find_target_pos(tgt_nc, cand_nc2)
        if tgt_pos < 0:
            raise ValueError("[BAD DATA] target_namecat not found in candidate_namecats. Ensure GT is included in candidates.")

        raw2 = _rewrite_prompt_candidates(raw, cand_nc2)

        return {
            "prompt": build_chat_prompt(tok, raw2) if args.use_chat_template else raw2,
            "prompt_raw": raw2,
            "history_item_ids": _get_field(ex, ["history_item_ids"]),
            "target_item_id": int(_get_field(ex, ["target_item_id"])),
            "target_namecat": tgt_nc,
            "candidate_namecats": cand_nc2,
            "candidate_item_ids": cand_it2,
            "target_pos": int(tgt_pos),  # ✅ eval HR 用
        }

    num_proc = max(1, (os.cpu_count() or 8) // 2)
    train_ds = train_ds.map(format_ex, with_indices=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    eval_ds = eval_ds.map(format_ex, with_indices=True, remove_columns=eval_ds.column_names, num_proc=num_proc)

    if args.print_target_pos_hist:
        def _hist(ds, name: str):
            cnt = Counter(ds["target_pos"])
            total = sum(cnt.values())
            top10 = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n[TARGET_POS_HIST] {name} total={total} top10(pos,count,ratio):")
            for pos, c in top10:
                print(f"  pos={pos:>3d}  count={c:>8d}  ratio={c/total:.4f}")
        _hist(train_ds, "train")
        _hist(eval_ds, "eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sasrec = load_sasrec_from_ckpt(
        sasrec_pkl=args.sasrec_pkl,
        sasrec_ckpt=args.sasrec_ckpt,
        device=device,
        max_len=args.sasrec_max_len,
        embed_dim=args.sasrec_embed_dim,
        num_blocks=args.sasrec_num_blocks,
        num_heads=args.sasrec_num_heads,
        dropout=args.sasrec_dropout,
    )
    scorer = SasrecScorer(sasrec_model=sasrec, device=device)

    r_cfg = ResolverConfig(
        format_bonus=float(args.format_bonus),
        in_candidates_bonus=float(args.in_candidates_bonus),

        match_reward_exact=float(args.match_reward_exact),
        match_reward_fold=float(args.match_reward_fold),

        alpha=float(args.alpha),
        softmax_temp=float(args.softmax_temp),
        teacher_mode=str(args.teacher_mode),
        teacher_clip=float(args.teacher_clip),

        extra_text_penalty=float(args.extra_text_penalty),
        unknown_penalty=float(args.unknown_penalty),
        prefix_penalty=float(args.prefix_penalty),
        incomplete_penalty=float(args.incomplete_penalty),
        copy_penalty=float(args.copy_penalty),
        duplicate_penalty=float(args.duplicate_penalty),

        debug_log_every_steps=int(args.debug_log_every_steps),
        debug_num_show=int(args.debug_num_show),
        debug_dump_jsonl=str(args.debug_dump_jsonl),
        debug_print_full_completion=bool(args.debug_print_full_completion),
    )
    reward_fn = make_reward_fn(scorer, r_cfg)

    # ✅ 注意：这里不再传 evaluation_strategy/eval_steps
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.per_device_bs),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.num_train_epochs),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=int(args.seed),

        max_prompt_length=int(args.max_length),
        max_completion_length=int(args.max_new_tokens),
        num_generations=int(args.num_generations),
        temperature=float(args.temperature),
    )

    max_total_len = int(args.max_length) + int(args.max_new_tokens)

    trainer = GRPOTrainerWithHREval(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        reward_funcs=reward_fn,

        tokenizer=tok,
        eval_max_samples=args.eval_max_samples,
        eval_prompt_bs=args.eval_prompt_bs,
        eval_score_bs=args.eval_score_bs,
        eval_length_norm=args.eval_length_norm,
        max_total_len=max_total_len,
    )

    # ✅ 关键：显式注册 callback（不依赖 config）
    trainer.add_callback(EvalEveryNStepsCallback(trainer, args.eval_steps))


    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("✅ GRPO done:", args.output_dir)


if __name__ == "__main__":
    main()
