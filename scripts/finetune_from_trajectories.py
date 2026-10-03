#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "trl>=0.9",
#   "transformers>=4.45",
#   "datasets>=2.20",
#   "peft>=0.12",
#   "torch>=2.3",
#   "trackio",
#   "huggingface_hub",
# ]
# ///
"""
finetune_from_trajectories.py
==============================
Fine-tunes Gemma 4 E2B-it (or its MTP drafter) on kernel-evolving task trajectories.

Method: SFT on PASS teacher trajectories (positive examples).
        DPO when Gemma FAIL examples are available on the same tasks.

Run on HuggingFace Jobs (H100, free tier with Pro):
  uv run scripts/finetune_from_trajectories.py --dataset path/to/export.jsonl

Run locally (CPU/GPU):
  uv run scripts/finetune_from_trajectories.py --dataset path/to/export.jsonl --local

Dataset format (from trajectory_collector.export_jsonl):
  Each line: {"id", "ts", "task", "provider", "model", "messages": [...], "artifacts", "critic_score", "verdict"}
  messages: [{"role": "user"}, {"role": "assistant", "tool_calls": [...]}, {"role": "tool"}, {"role": "assistant"}]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _reduce_loss(loss):
    """Reduce a nested/tuple loss to a scalar tensor.

    Nemotron-Labs-Diffusion-3B's custom remote-code forward returns a nested
    loss (a tuple/list of per-component losses) instead of a plain scalar
    tensor. TRL/transformers unwrap the model outputs as ``loss = outputs[0]``,
    so the trainer would pass that structure to ``accelerator.backward()`` and
    crash on ``loss / gradient_accumulation_steps``. Recursively collect every
    tensor leaf and mean them into a single scalar (identity for a plain tensor).

    NOTE: we MEAN (not sum) the leaves. The previous ``sum(leaves)`` produced a
    loss scaled by sequence length x component count, so ``train_loss`` landed at
    ~1.1e4 on every run regardless of data (structurally pinned, useless for
    tuning) and silently multiplied every gradient by that same factor (a hidden
    LR multiplier). Averaging yields a true per-token/per-component mean.
    """
    if isinstance(loss, torch.Tensor) or not isinstance(loss, (tuple, list)):
        return loss
    leaves = []
    stack = list(loss)
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            leaves.append(item)
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
    if not leaves:
        return loss[0] if loss else loss
    return sum(leaves) / len(leaves)


def _make_trainer_cls(base, normalize_by_tokens=False):
    """Return an ``SFTTrainer`` subclass whose ``compute_loss`` collapses any
    nested/tuple loss from the model forward into a single scalar tensor, and
    (optionally) re-normalizes a SUM-over-tokens loss into a true per-token mean.

    ``normalize_by_tokens=True`` is used for custom architectures (e.g.
    Nemotron-Labs-Diffusion-3B via trust_remote_code) whose forward returns a
    loss that is the raw sum over tokens rather than a per-token mean. In that
    case Trainer treats the huge scalar as-is: loss lands in the thousands and
    token accuracy stays ~0.45 because gradients are scaled by the token count
    (a hidden LR multiplier). ``_reduce_loss`` below only collapses nested
    leaves into one scalar — it does NOT divide by token count, so the huge
    value survives. Here we divide by the number of valid (non-ignored) labels
    to get a true per-token cross-entropy mean.
    """
    IGNORE = -100

    class TupleLossCompatTrainer(base):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            loss = _reduce_loss(loss)
            if normalize_by_tokens and loss is not None and labels is not None:
                valid = (labels != IGNORE).sum().float()
                # True per-token mean of a summed loss. Guard: only apply when
                # there are actual label tokens to divide by.
                if valid > 0:
                    loss = loss / valid
            if return_outputs:
                return loss, outputs
            return loss
    return TupleLossCompatTrainer


# ── Model family presets (generalized for Gemma / Qwen / Nemotron) ─────────
# Each family defines how the base is loaded, how the tokenizer is resolved, and
# which LoRA target modules are trainable. Kept as data so a new supported local
# model is a one-line addition.
FAMILY_PRESETS = {
    "qwen": {
        "load_class": "image_text_to_text",  # AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration (has lm_head -> loss)
        "lora_targets": r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        "lora_r": 16,
        "lora_alpha": 32,
        "quantize": True,
        "assistant_role": "assistant",        # Qwen chat template uses 'assistant'
        "task_type": None,                    # multimodal arch — no CAUSAL_LM task_type
    },
    "gemma": {
        "load_class": "image_text_to_text",   # AutoModelForImageTextToText
        "lora_targets": r"model\.language_model\..*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$",
        "lora_r": 16,
        "lora_alpha": 32,
        "quantize": True,
        "assistant_role": "model",            # Gemma4 chat template uses 'model'
    },
    "nemotron": {
        "load_class": "auto",                 # AutoModel (custom remote, trust_remote_code)
        "lora_targets": ["o_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # NVIDIA linear_spec targets o_proj; adding attention+MLP projections for adapt capacity
        "lora_r": 128,
        "lora_alpha": 512,
        "quantize": True,
    },
}


def detect_family(model_path: str) -> str:
    """Infer model family from its path/name. Defaults to gemma."""
    p = (model_path or "").lower()
    if "qwen" in p:
        return "qwen"
    if "nemotron" in p:
        return "nemotron"
    return "gemma"


def load_dataset_from_jsonl(path: str) -> list:
    """Load JSONL export from trajectory_collector."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(records)} trajectories from {path}")
    return records


# Tools list mirroring inference exactly — must stay in sync with src/tools.py TOOLS.
# Training applies_chat_template with tools= so the model sees the SAME token format
# as it does at inference. The previous bug: tool_calls were serialised as plain JSON
# text content without tools= in apply_chat_template, so the model learned to emit
# raw JSON blobs instead of native Gemma4 function-call tokens.
_TRAINING_TOOLS = [
    {"type": "function", "function": {
        "name": "exec_shell",
        "description": "Execute a shell command and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "http_get",
        "description": "Make an HTTP GET request and return the response body.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "integer"},
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "run_skill",
        "description": "Run a named skill with a natural-language input.",
        "parameters": {"type": "object", "properties": {
            "skill_name": {"type": "string"},
            "input": {"type": "string"},
        }, "required": ["skill_name", "input"]},
    }},
]


def _normalise_tool_calls(tool_calls) -> list:
    """Ensure tool_calls are in {type, id, function: {name, arguments}} format.
    The trajectory DB may store arguments as a JSON string or a dict — normalise both.
    """
    out = []
    for tc in (tool_calls or []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append({
            "type": "function",
            "id": tc.get("id", f"call_{len(out)}"),
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return out


def records_to_sft_dataset(records: list, tokenizer, assistant_role: str = "model") -> "Dataset":
    """Convert trajectory records to SFT format using chat template.

    Two input modes are supported:
      1. records with a pre-rendered "text" field (output of
         map_trajectories_to_nemotron.py / any tokenizer.pre-rendered export) —
         passed through as-is. This is what Nemotron FC training consumes, since
         its custom template + <tool_call> XML is rendered at map time.
      2. records with a "messages" list (standard export) — rendered here with
         apply_chat_template(tools=_TRAINING_TOOLS) so the model sees the exact
         inference prompt format. This is the Gemma/Qwen path.

    assistant_role: which role string the target's chat template accepts for the
        assistant turn — Gemma uses "model", Qwen uses "assistant".
    """
    from datasets import Dataset

    def format_record(r):
        # Mode 1: pre-rendered text passthrough (Nemotron mapper output).
        if "text" in r and isinstance(r.get("text"), str) and r["text"].strip():
            return {"text": r["text"], "task": r.get("task", ""), "critic_score": r.get("critic_score", 0)}

        messages = r.get("messages", [])
        if not messages:
            return None

        formatted = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            tool_calls = m.get("tool_calls")

            if role == "assistant":
                if tool_calls:
                    # Native tool-call turn: pass tool_calls as structured data.
                    formatted.append({
                        "role": assistant_role,
                        "content": content,
                        "tool_calls": _normalise_tool_calls(tool_calls),
                    })
                else:
                    formatted.append({"role": assistant_role, "content": content})

            elif role == "tool":
                # Tool result turn — Gemma4 expects role="tool" with tool_responses list.
                # trajectory_collector exports as {role:tool, name:..., content:...}
                # clean_trajectories.py preserves the same shape.
                tool_responses = m.get("tool_responses")
                if tool_responses:
                    formatted.append({"role": "tool", "tool_responses": tool_responses})
                else:
                    # Convert legacy {name, content} export format → tool_responses
                    tool_name = m.get("name", "tool")
                    result = content or ""
                    formatted.append({"role": "tool", "tool_responses": [
                        {"name": tool_name, "response": {"result": result}}
                    ]})

            else:
                # user / system — pass through as-is
                formatted.append({"role": role, "content": content})

        try:
            text = tokenizer.apply_chat_template(
                formatted,
                tools=_TRAINING_TOOLS,   # <-- THE FIX: match inference token format
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text, "task": r.get("task", ""), "critic_score": r.get("critic_score", 0)}
        except Exception as e:
            print(f"  [skip] chat template error: {e}")
            return None

    rows = [r for r in (format_record(rec) for rec in records) if r is not None]
    skipped = len(records) - len(rows)
    if skipped:
        print(f"  ({skipped} records skipped — chat template errors)")
    print(f"Formatted {len(rows)}/{len(records)} records for SFT")
    return Dataset.from_list(rows)


def _token_ids_with_offsets(tokenizer, text: str):
    """Tokenize a rendered text, returning (input_ids, char_offsets) where
    char_offsets is a parallel list of (start, end) unicode offsets per token.
    Falls back to a no-offset tokenization if the fast tokenizer is unavailable.
    """
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return enc["input_ids"], enc.get("offset_mapping", [])
    except (TypeError, ValueError):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids, []


def build_assistant_masked_dataset(records: list, tokenizer, base_texts: list) -> "Dataset":
    """Build a pre-tokenized SFT dataset whose loss is masked to ASSISTANT turns
    only (i.e. the tool-call emissions and final answers), zeroing out the large
    system/schema/user scaffolding that dominates a full-sequence loss.

    The rendered Nemotron text uses <|im_start|>assistant ... <|im_end|> blocks.
    We keep labels (not -100) only for tokens that fall inside an assistant block,
    so the model is forced to learn the argument-emission tokens instead of
    coasting on memorizing the repeated tool-schema preamble.

    Returns a datasets.Dataset with input_ids + labels columns for use with
    SFTTrainer(dataset_kwargs={"skip_prepare_dataset": True}).
    """
    from datasets import Dataset

    ASSISTANT_TAG = "<|im_start|>assistant"
    IM_END = "<|im_end|>"

    rows = []
    trainable_total = 0
    total_tokens = 0
    for text in base_texts:
        ids, offsets = _token_ids_with_offsets(tokenizer, text)
        n = len(ids)
        labels = [-100] * n
        # Find assistant spans as (start_char, end_char) directly in the raw text,
        # robust to how the <|im_start|>/<|im_end|> sentinels tokenize.
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
        # Map each span's char range onto token indices via offset_mapping.
        if offsets:
            for (sa, sb) in spans:
                if sb <= sa:
                    continue
                for i, (ts, te) in enumerate(offsets):
                    if te is None:
                        continue
                    if te > sa and ts < sb:  # token overlaps the assistant span
                        labels[i] = ids[i]
        else:
            # No offset map available: fall back to full-text token + role decode.
            text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            i = 0
            while i < n:
                seg = tokenizer.decode(text_ids[i:i + 2])
                if seg.startswith("<|im_start|>assistant"):
                    k = i
                    while k < n and "<|im_end|>" not in tokenizer.decode(text_ids[k:k + 1]):
                        labels[k] = ids[k]
                        k += 1
                    if k < n:
                        labels[k] = ids[k]
                    i = k + 1
                    continue
                i += 1
        trainable = sum(1 for x in labels if x != -100)
        trainable_total += trainable
        total_tokens += n
        rows.append({"input_ids": ids, "labels": labels, "task": ""})

    frac = (trainable_total / max(total_tokens, 1)) * 100
    print(
        f"[loss-mask] assistant-only: {trainable_total} trainable tokens / "
        f"{total_tokens} total ({frac:.1f}% of sequence)",
        flush=True,
    )
    if trainable_total == 0:
        raise RuntimeError(
            "loss-mask produced 0 trainable tokens — assistant spans not detected. "
            "Check the rendered assistant marker (current assumption: "
            "'<|im_start|>assistant'). Aborting rather than training a no-op."
        )
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL export from trajectory_collector")
    parser.add_argument("--model", default=None, help="Model path or HF repo (default: from config.yaml)")
    parser.add_argument("--drafter", action="store_true", help="Fine-tune MTP drafter instead of full model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (keep at 1 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length. All 90 augmented failing-tool records are ~3750-3850 tokens, so 4096 covers them fully (0 truncation).")
    parser.add_argument("--local", action="store_true", help="Run locally instead of HF Jobs")
    parser.add_argument("--output-dir", default=str(Path.home() / ".kernel-evolving/workspace/artifacts/finetune"))
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push adapter to")
    parser.add_argument("--min-score", type=float, default=0.7, help="Min critic score to include")
    parser.add_argument("--loss-mask", action="store_true", help="Mask loss to assistant turns only (zero out system/schema/user tokens), so the model is forced to learn the tool-call argument emission instead of memorizing the repeated schema preamble. Uses a pre-tokenized input_ids+labels dataset (skip_prepare_dataset=True).")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not model_path:
        try:
            import yaml
            cfg_candidates = [
                Path(__file__).parent.parent / "config.yaml",
                Path.home() / ".openclaw/workspace/repositories/kernel-evolving/config.yaml",
            ]
            for cfg_path in cfg_candidates:
                if cfg_path.exists():
                    cfg = yaml.safe_load(cfg_path.read_text())
                    if args.drafter:
                        model_path = cfg["model"].get("drafter_path")
                        print(f"Using drafter: {model_path}")
                    else:
                        model_path = cfg["model"]["path"]
                        print(f"Using main model: {model_path}")
                    break
        except Exception as e:
            print(f"Could not read config.yaml: {e}")

    if not model_path:
        model_path = "google/gemma-4-E2B-it"
        print(f"Falling back to: {model_path}")

    print(f"\n{'='*60}")
    print(f"Kernel-Evolving Trajectory Fine-Tuner")
    print(f"Model:   {model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs:  {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"{'='*60}\n")

    import torch
    from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model
    # Resolve family + tokenizer path from the chosen model.
    family = detect_family(model_path)
    preset = FAMILY_PRESETS[family]
    print(f"Model family: {family} (preset: {preset})")
    # Custom architectures (Nemotron trust_remote_code) return a SUM-over-tokens
    # loss; normalize it into a true per-token mean or token accuracy stays pinned
    # ~0.45 and loss lands in the thousands. Qwen/Gemma return a mean already.
    normalize = family == "nemotron"
    SFTTrainer = _make_trainer_cls(SFTTrainer, normalize_by_tokens=normalize)

    # Legacy E2B default: keep the env override for backwards compatibility.
    E2B_MODEL = os.environ.get(
        "KERNEL_EVO_E2B_MODEL",
        os.path.expanduser("~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
    )
    # Nemotron uses its own tokenizer (it has no multimodal processor); the chat
    # template lives in the tokenizer. Gemma uses AutoProcessor with chat template.
    tokenizer_path = model_path if family == "nemotron" else model_path
    try:
        if family in ("gemma", "qwen"):
            tokenizer = AutoProcessor.from_pretrained(tokenizer_path)
            if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer: {tokenizer_path} (chat_template: {'yes' if getattr(tokenizer,'chat_template',None) else 'no'})")

    # Legacy --drafter flag: only meaningful for Gemma's MTP drafter. If the target
    # has no standalone chat template (e.g. drafter_path), fall back to E2B-it.
    if args.drafter and family == "gemma" and model_path != E2B_MODEL:
        print("NOTE: Drafter is a speculative decoder (no standalone SFT). Training E2B-it with LoRA instead.")
        print("      The drafter benefits indirectly from improved E2B-it weights.")
        model_path = E2B_MODEL
        family = detect_family(model_path)
        preset = FAMILY_PRESETS[family]

    # Load dataset
    records = load_dataset_from_jsonl(args.dataset)
    # Filter by min critic score
    records = [r for r in records if (r.get("critic_score") or 0) >= args.min_score]
    print(f"After score filter (>={args.min_score}): {len(records)} records")
    if not records:
        print("ERROR: No records meet the score threshold. Lower --min-score or collect more trajectories.")
        sys.exit(1)

    dataset = records_to_sft_dataset(records, tokenizer, assistant_role=preset.get("assistant_role", "model"))

    # Optional assistant-only loss mask: rebuild as a pre-tokenized input_ids+labels
    # dataset whose labels are -100 outside assistant tool-call/answer turns, so the
    # model is forced to learn the argument-emission tokens instead of the repeated
    # system/schema preamble (which dominates the full-sequence loss/accuracy).
    if args.loss_mask:
        base_texts = [r["text"] for r in dataset]
        masked_ds = build_assistant_masked_dataset(records, tokenizer, base_texts)
        print(f"[loss-mask] built masked dataset with {len(masked_ds)} records", flush=True)
        # Sanity: count trainable (non -100) tokens across the whole dataset.
        try:
            trainable = sum(1 for lab in masked_ds["labels"] for x in lab if x != -100)
            total = sum(len(lab) for lab in masked_ds["labels"])
            print(f"[loss-mask] trainable tokens {trainable}/{total} ({100*trainable/max(total,1):.1f}%)", flush=True)
        except Exception as e:
            print(f"[loss-mask] sanity count skipped: {e}", flush=True)
        dataset = masked_ds

    # Load model with LoRA
    print("Loading model...")
    from transformers import BitsAndBytesConfig
    dtype = torch.float16

    # Try 4-bit quantisation if bitsandbytes available
    bnb_config = None
    try:
        import bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantisation (bitsandbytes)")
    except ImportError:
        print("bitsandbytes not available — loading in float16 (may need more VRAM)")

    load_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        quantization_config=bnb_config,
    )

    if preset["load_class"] == "auto":
        # AutoModel path — used by Nemotron (custom remote code) and Qwen3.5
        # (Qwen3_5ForConditionalGeneration multimodal arch). Nemotron additionally
        # needs trust_remote_code.
        if family == "nemotron":
            load_kwargs["trust_remote_code"] = True
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
    elif preset["load_class"] == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # LoRA config — family-aware target regex (Gemma/Qwen/Nemotron from presets).
    lora_kwargs = {
        "r": preset["lora_r"],
        "lora_alpha": preset["lora_alpha"],
        "target_modules": preset["lora_targets"],
        "lora_dropout": 0.05,
        "bias": "none",
    }
    # task_type: None for multimodal/conditional-generation archs (Qwen3.5) which
    # PEFT can't auto-map; CAUSAL_LM for standard decoders (Gemma/Nemotron).
    if preset.get("task_type"):
        lora_kwargs["task_type"] = preset["task_type"]
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable grad checkpointing before applying PEFT (required for memory efficiency)
    model.enable_input_require_grads()

    # SFT training — memory-safe config for 5B model on 16GB VRAM
    # batch=1 + grad_accum=16 = effective batch 16; grad_checkpointing halves VRAM usage
    sft_config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,          # MUST be 1 — batch=4 OOMs on logit alloc
        gradient_accumulation_steps=16,         # effective batch size = 16
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_len,              # was hardcoded 512 → truncated arg-bearing multi-step tails; now configurable (default 768)
        fp16=False,
        bf16=True,                               # RTX 4090 supports BF16; model weights are BF16
        gradient_checkpointing=True,             # recompute activations — saves ~50% VRAM
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to=["trackio"] if not args.local else ["none"],
        dataloader_pin_memory=False,             # avoid extra VRAM pinning
        torch_empty_cache_steps=1,               # free cache every step — needed for larger datasets
    )
    if args.loss_mask:
        # Pre-tokenized input_ids+labels dataset: skip TRL's re-tokenization and
        # its dataset_text_field (there is no plain text column to tokenize).
        sft_config_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    else:
        sft_config_kwargs["dataset_text_field"] = "text"
    sft_config = SFTConfig(**sft_config_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting SFT training on {len(dataset)} examples...")
    trainer.train()

    print(f"\nSaving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to HuggingFace: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Published: https://huggingface.co/{args.push_to_hub}")

    print("\nDone. To load the adapter:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")


if __name__ == "__main__":
    main()
