# HardMiningSFT/eval_stage1_nll_base.py
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

USER_PREFIX_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SUFFIX = "<|im_end|>"
PROMPT_RULE = "只输出一个地点名(类别)，不要解释"

def add_rule(p: str) -> str:
    p = (p or "").rstrip()
    if PROMPT_RULE in p:
        return p
    return p + "\n" + PROMPT_RULE

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2","sdpa","eager"])
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).eval()

    ds = load_dataset("json", data_files=args.data_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_eval, len(ds))))

    total_loss = 0.0
    total_tokens = 0

    for st in range(0, len(ds), args.bs):
        batch = ds[st: st + args.bs]
        prompts = [add_rule(p) for p in batch["prompt"]]
        comps = batch["completion"]

        prefix = [USER_PREFIX_TEMPLATE.format(prompt=p) for p in prompts]
        full = [pr + c + SUFFIX for pr, c in zip(prefix, comps)]

        enc_full = tok(full, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(model.device)
        enc_prefix = tok(prefix, truncation=True, max_length=args.max_length, padding=False, add_special_tokens=False)

        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]

        labels = input_ids.clone()

        # mask: prefix
        for i, ids in enumerate(enc_prefix["input_ids"]):
            s = len(ids)
            labels[i, :s] = -100

        # mask: padding (用 attention_mask 更稳)
        labels[attn == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss

        n_tok = (labels != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    avg_nll = total_loss / max(1, total_tokens)
    ppl = float(torch.exp(torch.tensor(avg_nll)).cpu())
    print(f"[BASE] N={len(ds)}, avg_NLL={avg_nll:.6f}, ppl={ppl:.3f}, total_tokens={total_tokens}")

if __name__ == "__main__":
    main()
