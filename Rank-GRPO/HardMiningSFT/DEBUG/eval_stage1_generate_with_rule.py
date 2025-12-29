# HardMiningSFT/eval_stage1_generate_with_rule.py
import re
import argparse
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
IM_END = "<|im_end|>"
IM_START_ASSIST = "<|im_start|>assistant"

# ✅ 跟 stage1 训练保持一致
PROMPT_RULE = "只输出一个地点名(类别)，不要解释"

SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")

def add_rule_to_prompt(p: str) -> str:
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    return p + "\n" + PROMPT_RULE

def norm_text(s: str) -> str:
    s = s.strip()
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
    for sep in [" Answer:", " Explanation:", " Because", " - ", "：", "。"]:
        if sep in decoded:
            decoded = decoded.split(sep, 1)[0].strip()
    return norm_text(decoded)

def is_strict_format(s: str) -> bool:
    s = s.strip()
    return bool(re.match(r"^.+\(.+\)$", s))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2","sdpa","eager"])
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # ✅ 用 torch_dtype
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()

    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_id = im_end_id if isinstance(im_end_id, int) and im_end_id >= 0 else tokenizer.eos_token_id
    print(f"[INFO] im_end_id={im_end_id}, eos_id_used={eos_id}")

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_eval, len(ds))))

    exact = exact_norm = contains = strict_ok = invalid = 0
    out_counter = Counter()
    lens = []

    for st in range(0, len(ds), args.bs):
        batch = ds[st: st + args.bs]
        prompts = [add_rule_to_prompt(p) for p in batch["prompt"]]  # ✅ 加规则
        gts = batch["completion"]

        inputs = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        enc = tokenizer(inputs, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(model.device)

        use_amp = model.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.generate(
                **enc,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)

        for dec, gt in zip(decoded, gts):
            pred = extract_answer(dec)
            gt0 = norm_text(gt)
            if pred == "" or len(pred) < 2:
                invalid += 1
                continue

            out_counter[pred] += 1
            lens.append(len(pred.split()))
            if is_strict_format(pred):
                strict_ok += 1
            if pred == gt0:
                exact += 1
            if pred.lower() == gt0.lower():
                exact_norm += 1
            if (gt0 in pred) or (pred in gt0):
                contains += 1

    n = len(ds)
    top1_pred, top1_cnt = out_counter.most_common(1)[0] if out_counter else ("", 0)

    print("========================================")
    print("STAGE1 GENERATION EVAL (with PROMPT_RULE)")
    print("========================================")
    print(f"Eval samples:         {n}")
    print(f"Exact match:          {exact}/{n} ({exact/n:.4f})")
    print(f"Exact(norm):          {exact_norm}/{n} ({exact_norm/n:.4f})")
    print(f"Contains match:       {contains}/{n} ({contains/n:.4f})")
    print(f"Strict-format rate:   {strict_ok}/{n} ({strict_ok/n:.4f})")
    print(f"Invalid outputs:      {invalid}/{n} ({invalid/n:.4f})")
    if lens:
        print(f"Avg output words:     {sum(lens)/len(lens):.2f}")
    print("----------------------------------------")
    print(f"Top1 output ratio:    {top1_cnt}/{n} ({top1_cnt/n:.4f})")
    if top1_pred:
        show = top1_pred[:120] + ("..." if len(top1_pred) > 120 else "")
        print(f"Top1 output sample:   {show}")
    print("========================================")

if __name__ == "__main__":
    main()
