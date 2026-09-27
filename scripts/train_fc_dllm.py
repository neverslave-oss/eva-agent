#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.45",
#   "datasets>=2.20",
#   "peft>=0.12",
#   "torch>=2.3",
#   "bitsandbytes",
# ]
# ///
"""
train_fc_dllm.py — probe SFT for the Nemotron-Labs-Diffusion-3B FC adapter.

ROOT-CAUSE CONTEXT (2026-09-27, verified): Nemotron-Labs-Diffusion-3B is a
DIFFUSION language model (dllm_paradigm="bidirectional" in its config.json), NOT
an autoregressive LM. TRL's SFTTrainer drives AR next-token CE loss, which
fundamentally mis-trains a bidirectional diffusion model — the reason v3..v9
adapters never learned to emit tool arguments (loss "moved", behavior never
changed).

This trainer instead calls the model's NATIVE forward with `labels` + `loss_mask`:
  modeling_nemotron_labs_diffusion.py forward:
    when dlm_paradigm != 'autoregressive' → LLaDA-style diffusion loss on the
    masked positions (CE on labels[masked_indices], masked via
    mask_token_id=100). loss_mask controls WHICH positions are eligible to be
    masked for denoising:
      masked_indices[loss_mask == 0] = 0
  So we pass loss_mask = 1 on assistant/answer turns (the tool-call + args),
  0 elsewhere (system/schema/user scaffolding) → the model learns to denoise
  EXACTLY the tools' argument-emission tokens.

NVIDIA's official guidance (dLLM SFT): mode=mdlm, mask_token_id=100, eps=0.001,
dataset.unshifted=true, dataset.mask_history=true (== supervise only answer turn,
the dLLM equivalent of `mask_history`). LoRA: dim 16, alpha 16. Their recipe is
8-way torchrun, so this is a single-GPU (16GB 4090) local adaptation keeping the
MECHANISM, not the multi-node harness.

Usage (probe):
  <conda-python> scripts/train_fc_dllm.py \
      --dataset ~/.kernel-evolving/workspace/artifacts/trajectories/nemotron_fc_v8.jsonl \
      --model /mnt/e/models/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B \
      --output-dir ~/.kernel-evolving/workspace/artifacts/finetune/nemotron_fc_v10 \
      --steps 200 --lr 5e-4
  --dry-run: build + validate one batch without loading the model (GPU check).
"""

import argparse, json, os, sys, time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset


def reduce_tensor(x):
    """Recursively reduce a scalar / nested tuple/list / dict of tensors to a
    single scalar tensor. Nemotron's dLLM forward returns out.loss as a nested
    tuple of per-component losses (AR + diffusion), so we collapse it."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        leaves = [e for e in (reduce_tensor(i) for i in x) if e is not None]
        if not leaves:
            return None
        return sum(leaves) / len(leaves)  # per-component mean (stable scalar)
    if isinstance(x, dict):
        leaves = [e for e in (reduce_tensor(v) for v in x.values()) if e is not None]
        if not leaves:
            return None
        return sum(leaves) / len(leaves)
    return None

ASSISTANT_TAG = "<|im_start|>assistant"
IM_END = "<|im_end|>"


def load_records(path: str) -> list:
    recs = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d.get("text"), str) and d["text"].strip():
                recs.append(d["text"])
    print(f"Loaded {len(recs)} rendered records from {path}", flush=True)
    return recs


def assistant_spans(text: str):
    """Return (start_char, end_char) of each <|im_start|>assistant ... <|im_end|>."""
    spans = []
    search_from = 0
    while True:
        a = text.find(ASSISTANT_TAG, search_from)
        if a == -1:
            break
        b = text.find(IM_END, a)
        if b == -1:
            b = len(text)
        spans.append((a, b + len(IM_END)))
        search_from = b + len(IM_END)
    return spans


def tokenize_with_mask(tokenizer, text: str, max_len: int = 4096):
    """Tokenize rendered text; return (input_ids, attention_mask, loss_mask).
    loss_mask[i]=1 when token i falls inside an assistant span (trainable)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc.get("offset_mapping", [])
    if not offsets:
        raise RuntimeError(
            "fast tokenizer offset_mapping not available — required for assistant-span masking"
        )
    spans = assistant_spans(text)
    loss_mask = [0] * len(ids)
    for (sa, sb) in spans:
        if sb <= sa:
            continue
        for i, (ts, te) in enumerate(offsets):
            if te is None:
                continue
            if te > sa and ts < sb:  # token overlaps assistant span
                loss_mask[i] = 1
    # Truncate / pad to fixed max_len for batching.
    if len(ids) > max_len:
        ids = ids[:max_len]
        loss_mask = loss_mask[:max_len]
    n = len(ids)
    if n < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - n)
        loss_mask = loss_mask + [0] * (max_len - n)
    attn = [1] * n + [0] * (max_len - n)
    return ids, attn, loss_mask


def build_dataset(records, tokenizer, max_len):
    rows = []
    for t in records:
        ids, attn, lm = tokenize_with_mask(tokenizer, t, max_len)
        rows.append({"input_ids": ids, "attention_mask": attn, "loss_mask": lm})
    ds = Dataset.from_list(rows)
    trainable = sum(sum(r["loss_mask"]) for r in ds)
    total = len(ds) * max_len
    print(f"[loss-mask] trainable tokens {trainable}/{total} ({100*trainable/max(total,1):.1f}%)", flush=True)
    if trainable == 0:
        raise RuntimeError("0 trainable tokens — assistant spans not detected")
    return ds


def collate(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "loss_mask": torch.tensor([b["loss_mask"] for b in batch], dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    records = load_records(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ds = build_dataset(records, tokenizer, args.max_seq_len)

    # ---- model + LoRA (diffusion blocks' projections, per NVIDIA target '*_proj') ----
    if args.dry_run:
        print("DRY-RUN OK: dataset built, tokenizer ready. Skipping model load.", flush=True)
        return

    print("Loading Nemotron-Labs-Diffusion-3B (4-bit, LoRA)...", flush=True)
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModel.from_pretrained(args.model, quantization_config=bnb,
                                      device_map="auto", trust_remote_code=True)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = min(args.steps, len(dl) * 5)  # cap; don't exceed a few epochs
    model.train()
    step = 0
    opt.zero_grad()
    accum = 0
    running_loss = 0.0
    it = iter(dl)
    t0 = time.time()
    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Native dLLM forward: labels = full input_ids, loss computed on masked
        # positions; loss_mask restricts those positions to assistant spans.
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            loss_mask=batch["loss_mask"],
        )
        raw_loss = out.loss if hasattr(out, "loss") else out[0]
        loss = reduce_tensor(raw_loss)
        if loss is None or not torch.is_tensor(loss):
            raise RuntimeError(f"unresolvable loss: {raw_loss!r}")
        # Normalize sum-over-masked-tokens into a per-token mean for a stable metric.
        num_masked = batch["loss_mask"].sum().clamp(min=1.0)
        per_token = loss / num_masked
        per_token.backward()
        accum += 1
        running_loss += per_token.detach().item() * num_masked.item()
        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            accum = 0
            step += 1
            if step % 5 == 0 or step == total_steps:
                mean_loss = running_loss / (5 * args.grad_accum * 1.0)
                print(
                    f"step {step}/{total_steps}  masked-CE-per-token={mean_loss:.4f}  "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
                running_loss = 0.0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}", flush=True)
    print("Verify: load with PeftModel.from_pretrained(base, output_dir), restart "
          "start.sh --config=configs/06-local-nemotron-fc.yaml (adapter_path -> v10), "
          "then run ONE live call and eyeball raw emission.", flush=True)


if __name__ == "__main__":
    main()
