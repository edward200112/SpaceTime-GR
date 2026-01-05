import os
import json
import re
import random
import argparse
from typing import List, Tuple, Optional, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import deque


# ----------------------------
# Parsing utils
# ----------------------------
def parse_output(text: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Expect model output like: "123, 4, 56, 789" (may include "Response:" and <>)
    """
    text = text.replace("Response:", "").replace("<", "").replace(">", "")
    m = re.search(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text.strip())
    return tuple(int(x) for x in m.groups()) if m else None


def parse_target(target_raw) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(target_raw, (list, tuple)):
        try:
            return tuple(int(x) for x in target_raw)
        except:
            return None
    if isinstance(target_raw, str):
        clean = target_raw.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
        try:
            return tuple(int(x.strip()) for x in clean.split(","))
        except:
            return None
    return None


def build_prompt(item: dict) -> Optional[Dict]:
    """
    Compatible with your train_prompts.jsonl format:
      item['task'] == 'task_a_recommendation'
      item['instruction']
      item['metadata']['target_sid']
    """
    if item.get("task") != "task_a_recommendation":
        return None

    meta = item.get("metadata", {})
    raw_inst = (item.get("instruction") or "").strip()
    if not raw_inst:
        return None

    if "Response:" in raw_inst:
        prompt_text = raw_inst.split("Response:")[0].strip()
    else:
        prompt_text = raw_inst

    suffix = "Output the semantic ID in the format <c0, c1, c2, suffix>."
    final_prompt = f"{prompt_text}\n{suffix}\nResponse: <"

    return {
        "prompt": final_prompt,
        "target_sid": meta.get("target_sid"),
    }


# ----------------------------
# Streaming sample from huge JSONL
# ----------------------------
def reservoir_sample_jsonl(path: str, sample_size: int, seed: int = 42, max_read_lines: int = -1) -> List[Dict]:
    """
    Reservoir sampling over usable items (after filtering & prompt building).
    This avoids loading 2.2M lines into memory.
    """
    rng = random.Random(seed)
    sample = []
    seen = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_read_lines > 0 and idx >= max_read_lines:
                break
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except:
                continue

            x = build_prompt(item)
            if x is None:
                continue

            seen += 1
            if len(sample) < sample_size:
                sample.append(x)
            else:
                j = rng.randint(1, seen)
                if j <= sample_size:
                    sample[j - 1] = x

    return sample


def head_jsonl(path: str, limit: int) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except:
                continue
            x = build_prompt(item)
            if x is None:
                continue
            data.append(x)
            if len(data) >= limit:
                break
    return data


# ----------------------------
# Model loading
# ----------------------------
def load_base_and_tokenizer(base_dir: str):
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def merge_adapter_into_base(base_model, adapter_dir: str):
    """
    base -> load adapter -> merge -> return pure HF model (but may still retain some peft attrs in-memory)
    We'll later "save and reload" to fully clean it.
    """
    m = PeftModel.from_pretrained(base_model, adapter_dir)
    m = m.merge_and_unload()
    m.eval()
    return m


def save_and_reload_pure_model(model, tokenizer, out_dir: str):
    """
    Save merged weights and reload as a clean AutoModelForCausalLM to avoid 'multiple adapters' warnings.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    reloaded = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    reloaded.eval()
    return reloaded


def load_adapter_on_model(model, adapter_dir: str):
    """
    Load a PEFT adapter without merging (for evaluation).
    """
    m = PeftModel.from_pretrained(model, adapter_dir)
    m.eval()
    return m


# ----------------------------
# Recall@20 evaluation (generation-based)
# ----------------------------
@torch.inference_mode()
def recall_at_k_generate(
    model,
    tokenizer,
    data: List[Dict],
    k: int = 20,
    beams: int = 40,
    batch_size: int = 1,
    max_new_tokens: int = 24,
):
    """
    For each prompt, use beam search to get top 'beams' sequences.
    Parse semantic ID from each, keep unique predictions in order, take top-k.
    Hit if target_sid is in top-k unique.
    """
    total = 0
    hit = 0
    invalid_pred = 0
    not_enough_unique = 0

    pbar = tqdm(range(0, len(data), batch_size), desc="eval", unit="batch")

    for i in pbar:
        batch = data[i : i + batch_size]
        prompts = [x["prompt"] for x in batch]
        targets = [parse_target(x["target_sid"]) for x in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        enc = {kk: vv.to(model.device) for kk, vv in enc.items()}
        true_lens = enc["attention_mask"].sum(dim=1).tolist()  # padded input len

        gen = model.generate(
            **enc,
            do_sample=False,
            num_beams=beams,
            num_return_sequences=beams,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # gen shape: (batch_size * beams, seq_len)
        for b in range(len(batch)):
            tgt = targets[b]
            if tgt is None:
                # skip bad target
                continue

            seqs = gen[b * beams : (b + 1) * beams]
            preds = []

            for s in seqs:
                out_ids = s[true_lens[b]:]  # generated portion
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                sid = parse_output(text)
                if sid is not None:
                    preds.append(sid)

            if len(preds) == 0:
                invalid_pred += 1
                total += 1
                pbar.set_postfix({"recall@20": f"{(hit/total if total else 0):.4f}", "invalid": invalid_pred})
                continue

            uniq = []
            seen = set()
            for p in preds:
                if p not in seen:
                    uniq.append(p)
                    seen.add(p)
                if len(uniq) >= k:
                    break

            if len(uniq) < k:
                not_enough_unique += 1

            if tgt in uniq:
                hit += 1
            total += 1

        pbar.set_postfix({"recall@20": f"{(hit/total if total else 0):.4f}", "invalid": invalid_pred})

    return {
        "total": total,
        "hit": hit,
        f"recall@{k}": hit / total if total else 0.0,
        "invalid_pred_count": invalid_pred,
        "not_enough_unique_topk_count": not_enough_unique,
        "beams": beams,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/workspace/Qwen2_5-1.5B-Instruct")
    parser.add_argument("--sft_adapter", type=str, default="/workspace/recalltest/model/checkpoint-35000")
    parser.add_argument("--grpo_adapter", type=str, default="/workspace/recalltest/model/checkpoint-5000")
    parser.add_argument("--data", type=str, default="/workspace/recalltest/data/train_prompts.jsonl")
    parser.add_argument("--tmp_merged_dir", type=str, default="/workspace/recalltest/tmp_merged_sft")

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--beams", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=24)

    # sampling
    parser.add_argument("--eval_samples", type=int, default=200, help="How many samples to evaluate")
    parser.add_argument("--sample_mode", type=str, default="random", choices=["random", "head"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_read_lines", type=int, default=-1, help="limit how many lines to scan (optional)")

    args = parser.parse_args()

    # 1) Load eval data (streaming sample)
    if args.sample_mode == "head":
        eval_data = head_jsonl(args.data, limit=args.eval_samples)
    else:
        eval_data = reservoir_sample_jsonl(
            args.data,
            sample_size=args.eval_samples,
            seed=args.seed,
            max_read_lines=args.max_read_lines,
        )

    if not eval_data:
        raise RuntimeError("No usable samples loaded from jsonl. Check task/instruction/metadata format.")

    print(f"[Data] eval_samples={len(eval_data)} mode={args.sample_mode}")

    # 2) Load base + tokenizer
    base, tok = load_base_and_tokenizer(args.base_model)

    # 3) Evaluate SFT (Base + SFT adapter merged & cleaned)
    print("\n=== Evaluating SFT (Base -> SFT adapter -> merge -> reload) ===")
    merged_sft = merge_adapter_into_base(base, args.sft_adapter)
    sft_clean = save_and_reload_pure_model(merged_sft, tok, args.tmp_merged_dir + "_SFT_ONLY")

    sft_stats = recall_at_k_generate(
        sft_clean,
        tok,
        eval_data,
        k=args.k,
        beams=args.beams,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print("[SFT]", json.dumps(sft_stats, ensure_ascii=False, indent=2))

    # 4) Evaluate GRPO (Base -> SFT merge -> reload pure -> GRPO adapter)
    print("\n=== Evaluating GRPO (Base -> SFT merge -> reload -> GRPO adapter) ===")
    # reload base fresh (avoid any in-memory leftovers)
    base2, _ = load_base_and_tokenizer(args.base_model)
    merged_sft2 = merge_adapter_into_base(base2, args.sft_adapter)
    merged_clean = save_and_reload_pure_model(merged_sft2, tok, args.tmp_merged_dir + "_SFT_MERGED")

    grpo_model = load_adapter_on_model(merged_clean, args.grpo_adapter)

    grpo_stats = recall_at_k_generate(
        grpo_model,
        tok,
        eval_data,
        k=args.k,
        beams=args.beams,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print("[GRPO]", json.dumps(grpo_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
