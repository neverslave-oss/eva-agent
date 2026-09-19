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
        "lora_targets": ["o_proj"],           # NVIDIA's own linear_spec adapter targets o_proj
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL export from trajectory_collector")
    parser.add_argument("--model", default=None, help="Model path or HF repo (default: from config.yaml)")
    parser.add_argument("--drafter", action="store_true", help="Fine-tune MTP drafter instead of full model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (keep at 1 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--local", action="store_true", help="Run locally instead of HF Jobs")
    parser.add_argument("--output-dir", default=str(Path.home() / ".kernel-evolving/workspace/artifacts/finetune"))
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push adapter to")
    parser.add_argument("--min-score", type=float, default=0.7, help="Min critic score to include")
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
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,          # MUST be 1 — batch=4 OOMs on logit alloc
        gradient_accumulation_steps=16,         # effective batch size = 16
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_length=512,                          # 1024 OOMs with 494-example dataset; 512 fits
        fp16=False,
        bf16=True,                               # RTX 4090 supports BF16; model weights are BF16
        gradient_checkpointing=True,             # recompute activations — saves ~50% VRAM
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to=["trackio"] if not args.local else ["none"],
        dataset_text_field="text",
        dataloader_pin_memory=False,             # avoid extra VRAM pinning
        torch_empty_cache_steps=1,               # free cache every step — needed for larger datasets
    )

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
