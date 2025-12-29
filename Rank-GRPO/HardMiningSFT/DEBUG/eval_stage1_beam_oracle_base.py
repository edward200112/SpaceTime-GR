# HardMiningSFT/eval_stage1_beam_oracle_base.py
import re
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
IM_END = "<|im_end|>"
IM_START_ASSIST = "<|im_start|>assistant"
PROMPT_RULE = "只输出一个地点名(类别)，不要解释"
SPECIAL_PAT = re.compile(r"<\|[^>]+\|>")

def add_rule(p: str) -> str:
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    return p + "\n" + PROMPT_RULE

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = SPECIAL_PAT.sub("", s)
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n\"'`.,;:，。；：")

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=10)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2","sdpa","eager"])
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()

    im_end_id = tok.convert_tokens_to_ids(IM_END)
    eos_id = im_end_id if isinstance(im_end_id, int) and im_end_id >= 0 else tok.eos_token_id
    print(f"[INFO] eos_id_used={eos_id} (im_end_id={im_end_id})")

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_eval, len(ds))))

    oracle_hit = 0
    for st in range(0, len(ds), args.bs):
        batch = ds[st: st + args.bs]
        prompts = [add_rule(p) for p in batch["prompt"]]
        gts = [norm_text(x) for x in batch["completion"]]

        inputs = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        enc = tok(inputs, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(model.device)

        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(out, skip_special_tokens=False)

        B = len(prompts)
        K = args.num_beams
        for i in range(B):
            gt = gts[i]
            cand_texts = decoded[i*K:(i+1)*K]
            cands = [extract_answer(t) for t in cand_texts]
            if gt in cands:
                oracle_hit += 1

    n = len(ds)
    print("========================================")
    print(f"[BASE] BEAM ORACLE EVAL (K={args.num_beams})")
    print("========================================")
    print(f"Eval samples:   {n}")
    print(f"Oracle@{args.num_beams} hit: {oracle_hit}/{n} ({oracle_hit/n:.4f})")
    print("========================================")

if __name__ == "__main__":
    main()
