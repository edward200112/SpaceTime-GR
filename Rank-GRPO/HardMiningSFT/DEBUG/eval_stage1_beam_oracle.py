# HardMiningSFT/eval_stage1_beam_oracle.py
import re
import math
import argparse
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
IM_END = "<|im_end|>"
IM_START_ASSIST = "<|im_start|>assistant"

SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")
PAIR_RE = re.compile(r"^(.*)\((.*)\)$")  # name (category)


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = SPECIAL_PAT.sub("", s)
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:，。；：")
    return s


def extract_answer(decoded: str) -> str:
    if IM_START_ASSIST in decoded:
        decoded = decoded.split(IM_START_ASSIST, 1)[1]
    if IM_END in decoded:
        decoded = decoded.split(IM_END, 1)[0]
    decoded = norm_text(decoded)

    # 截断掉可能的解释片段
    for sep in [" Answer:", " Explanation:", " Because", " - ", "：", "。"]:
        if sep in decoded:
            decoded = decoded.split(sep, 1)[0].strip()

    return norm_text(decoded)


def is_strict_format(s: str) -> bool:
    s = (s or "").strip()
    return bool(PAIR_RE.match(s))


def split_name_cat(s: str):
    s = (s or "").strip()
    m = PAIR_RE.match(s)
    if not m:
        return s.strip(), ""
    name = m.group(1).strip()
    cat = m.group(2).strip()
    return name, cat


def levenshtein_sim(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    dist = dp[lb]
    return 1.0 - dist / max(la, lb)


def token_jaccard(a: str, b: str) -> float:
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)

    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--bs", type=int, default=16)

    ap.add_argument("--num_beams", type=int, default=10)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])

    ap.add_argument("--report_top1_ratio", action="store_true")
    # ✅ 新增：打印 top-k Top1 预测
    ap.add_argument("--report_topk", type=int, default=0,
                    help="Print top-k most common Top1 predictions (count + ratio). e.g. 10")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 你环境里提示 torch_dtype deprecated -> 用 dtype
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_id = im_end_id if isinstance(im_end_id, int) and im_end_id >= 0 else tokenizer.eos_token_id
    print(f"[INFO] eos_id_used={eos_id} (im_end_id={im_end_id})")

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_eval, len(ds))))

    oracle_hit = 0
    strict_ok = 0

    # New metrics
    name_exact_1 = 0
    cat_exact_1 = 0
    name_edit90 = 0
    name_edit80 = 0
    name_jac80 = 0

    top1_counter = Counter()
    top1_list = []
    top1_sample = None

    for st in range(0, len(ds), args.bs):
        batch = ds[st: st + args.bs]
        prompts = batch["prompt"]
        gts = batch["completion"]

        PROMPT_RULE = "只输出一个地点名(类别)，不要解释"
        inputs = [USER_PREFIX_TEMPLATE.format(prompt=p.rstrip() + "\n" + PROMPT_RULE) for p in prompts]

        enc = tokenizer(
            inputs,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        use_amp = model.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model.generate(
                **enc,
                do_sample=False,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)

        B = len(prompts)
        K = args.num_beams
        for i in range(B):
            gt = norm_text(gts[i])
            gt_name, gt_cat = split_name_cat(gt)

            beams_i = decoded[i * K: (i + 1) * K]
            preds = [extract_answer(x) for x in beams_i]
            preds_norm = [norm_text(x) for x in preds]

            # Oracle@K exact match
            if any(p == gt for p in preds_norm):
                oracle_hit += 1

            # Top1
            top1 = preds_norm[0] if preds_norm else ""
            if top1:
                top1_counter[top1] += 1
                top1_list.append(top1)
                if top1_sample is None:
                    top1_sample = top1

            if is_strict_format(top1):
                strict_ok += 1

            # New: name/category match & fuzzy
            p1_name, p1_cat = split_name_cat(top1)

            if p1_name and gt_name:
                if p1_name == gt_name:
                    name_exact_1 += 1
                s = levenshtein_sim(p1_name, gt_name)
                if s >= 0.90:
                    name_edit90 += 1
                if s >= 0.80:
                    name_edit80 += 1
                if token_jaccard(p1_name, gt_name) >= 0.80:
                    name_jac80 += 1

            if p1_cat and gt_cat and (p1_cat == gt_cat):
                cat_exact_1 += 1

    n = len(ds)
    distinct_1 = len(set(top1_list)) / max(1, len(top1_list))
    ent = entropy_from_counter(top1_counter)

    print("========================================")
    print(f"BEAM ORACLE EVAL (K={args.num_beams})")
    print("========================================")
    print(f"Eval samples:   {n}")
    print(f"Oracle@{args.num_beams} hit (exact): {oracle_hit}/{n} ({oracle_hit/n:.4f})")
    print(f"Strict-format rate (Top1): {strict_ok}/{n} ({strict_ok/n:.4f})")
    print("----------------------------------------")
    print(f"NameExact@1:        {name_exact_1}/{n} ({name_exact_1/n:.4f})")
    print(f"CatExact@1:         {cat_exact_1}/{n} ({cat_exact_1/n:.4f})")
    print(f"NameEditSim>=0.90:  {name_edit90}/{n} ({name_edit90/n:.4f})")
    print(f"NameEditSim>=0.80:  {name_edit80}/{n} ({name_edit80/n:.4f})")
    print(f"NameJaccard>=0.80:  {name_jac80}/{n} ({name_jac80/n:.4f})")
    print(f"Distinct@1:         {distinct_1:.4f}")
    print(f"Entropy(Top1):      {ent:.4f}")

    # ✅ 新增：Top-K Top1 预测列表
    if args.report_topk and args.report_topk > 0 and top1_counter:
        k = int(args.report_topk)
        print("----------------------------------------")
        print(f"Top{min(k, len(top1_counter))} Top1 predictions:")
        for rank, (pred, cnt) in enumerate(top1_counter.most_common(k), start=1):
            ratio = cnt / n
            show = pred[:120] + ("..." if len(pred) > 120 else "")
            print(f"{rank:02d}. {cnt:4d}/{n} ({ratio:.4f})  {show}")

    # 保留你原来的 top1 ratio 输出
    if args.report_top1_ratio:
        if top1_counter:
            top1_pred, top1_cnt = top1_counter.most_common(1)[0]
            print("----------------------------------------")
            print(f"Beam1 Top1 ratio: {top1_cnt}/{n} ({top1_cnt/n:.4f})")
            show = top1_pred[:120] + ("..." if len(top1_pred) > 120 else "")
            print(f"Beam1 Top1 sample: {show}")

    print("========================================")


if __name__ == "__main__":
    main()
