# HardMiningSFT/eval_stage1_generate_v2.py
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

# 更强的清洗：去掉各种特殊 token
SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")

def norm_text(s: str) -> str:
    s = s.strip()
    s = SPECIAL_PAT.sub("", s)          # remove <|...|>
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # 去掉尾部标点/多余引号
    s = s.strip(" \t\r\n\"'`.,;:，。；：")
    return s

def extract_answer(decoded: str) -> str:
    """
    1) 取 assistant 块
    2) 截断到 <|im_end|> 之前
    3) 只取“第一句/第一段”（避免啰嗦导致 exact 永远=0）
    """
    # 只取最后一个 assistant block（更安全）
    if IM_START_ASSIST in decoded:
        decoded = decoded.split(IM_START_ASSIST, 1)[1]

    if IM_END in decoded:
        decoded = decoded.split(IM_END, 1)[0]

    decoded = norm_text(decoded)

    # 进一步截断：遇到明显的解释性分隔符就停（可按需扩展）
    for sep in [" Answer:", " Explanation:", " Because", " - ", "：", "。"]:
        if sep in decoded:
            decoded = decoded.split(sep, 1)[0].strip()

    return norm_text(decoded)

def is_strict_format(s: str) -> bool:
    """
    你当前 completion 是 "Name (Category)"，用一个宽松 regex 判定
    """
    s = s.strip()
    # 至少包含一对括号
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
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()

    # 关键：把停止符设置成 <|im_end|>（如果 tokenizer 里存在的话）
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_id = im_end_id if isinstance(im_end_id, int) and im_end_id >= 0 else tokenizer.eos_token_id

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_eval, len(ds))))

    exact = 0
    exact_norm = 0
    contains = 0
    strict_ok = 0
    invalid = 0

    out_counter = Counter()
    lens = []

    for st in range(0, len(ds), args.bs):
        batch = ds[st: st + args.bs]
        prompts = batch["prompt"]
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

            # 更宽松的 exact：忽略大小写 & 多余空格/标点
            if pred.lower() == gt0.lower():
                exact_norm += 1

            if (gt0 in pred) or (pred in gt0):
                contains += 1

    n = len(ds)
    top1_pred, top1_cnt = out_counter.most_common(1)[0] if out_counter else ("", 0)

    print("========================================")
    print("STAGE1 GENERATION EVAL (v2)")
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
