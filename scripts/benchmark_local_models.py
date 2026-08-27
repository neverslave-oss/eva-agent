#!/usr/bin/env python3
"""
benchmark_local_models.py — benchmark the local GPU models served by
kernel-evolving's model server across the modalities each model supports.

Uses core.inference.model_client (the same client the agent uses) to call the
running model server on /tmp/kernel_evolving_model.sock. Measures wall-clock
latency and (for text) throughput in tokens/sec.

Modalities benchmarked per model (from the HF model cards):
  - Gemma 4 E2B-it      : text, tool-calling, vision, audio
  - Qwen3.5-0.8B        : text, tool-calling, vision
  - Nemotron-Diffusion  : text (fast diffusion / self-speculation)
  - Qwen2.5-Omni-3B     : text, tool-calling, vision, audio, speech-gen (if loadable)
  - Janus-Pro-7B        : text, vision, image-generation (if loadable)

Usage:
  python scripts/benchmark_local_models.py [--text] [--tools] [--vision] [--audio]
                                            [--models gemma,qwen,nemotron]
  (no flags = run all modalities on all configured models)

Requires the model server to be running (start.sh). Run from the repo root so
the `src` package is importable.
"""

import argparse
import json
import os
import sys
import time

# Allow importing src.* modules from the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

# Test assets
TEST_IMAGE = "/tmp/bench_assets/test_chart.png"
TEST_AUDIO = "/home/pacificDev/.openclaw/workspace/repositories/olly-voice-server/media/voice-samples/fabio-en-phonetic.wav"

TEXT_PROMPT = "Explain in 3 short sentences what a transformer neural network is."
TOOL_PROMPT = "What is 15% of 240? Use the calculator tool to compute it."
VISION_PROMPT = "Describe this image. What shapes and colors do you see, and what text is written?"
AUDIO_PROMPT = "Transcribe the following speech segment in English into English text."

# A simple tool definition for the tool-calling benchmark.
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "math expression"}},
            "required": ["expression"],
        },
    },
}


def _import_client():
    """Import the model_client module (must be done after sys.path is set)."""
    import core.inference.model_client as mc
    return mc


def _measure(fn, *args, **kwargs):
    """Run fn, return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt


def _tok_rate(result, dt):
    """Rough tokens/sec from a text result (whitespace tokens)."""
    if not result or isinstance(result, str) and result.startswith("["):
        return 0.0
    tokens = max(1, len(str(result).split()))
    return round(tokens / dt, 2) if dt > 0 else 0.0


def bench_text(mc, slot: str, label: str) -> dict:
    """Basic text inference via infer_local (or infer for primary)."""
    messages = [{"role": "user", "content": TEXT_PROMPT}]
    try:
        if slot == "primary":
            result, dt = _measure(mc.infer, messages, 128)
        else:
            result, dt = _measure(mc.infer_local, messages, 128, slot)
    except Exception as e:
        return {"model": label, "modality": "text", "status": "error", "error": str(e)}
    return {
        "model": label, "modality": "text", "status": "ok",
        "latency_s": round(dt, 3), "tok_s": _tok_rate(result, dt),
        "output": str(result)[:200],
    }


def bench_tools(mc, slot: str, label: str) -> dict:
    """Tool-calling inference via infer_with_tools."""
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    try:
        result, dt = _measure(mc.infer_with_tools, messages, [CALC_TOOL], max_steps=5)
    except Exception as e:
        return {"model": label, "modality": "tools", "status": "error", "error": str(e)}
    return {
        "model": label, "modality": "tools", "status": "ok",
        "latency_s": round(dt, 3), "tok_s": _tok_rate(result, dt),
        "output": str(result)[:200],
    }


def bench_vision(mc, slot: str, label: str) -> dict:
    """Vision inference via infer_with_image (force_local=True)."""
    if not os.path.exists(TEST_IMAGE):
        return {"model": label, "modality": "vision", "status": "error", "error": "no test image"}
    try:
        result, dt = _measure(mc.infer_with_image, TEST_IMAGE, VISION_PROMPT, 128, True)
    except Exception as e:
        return {"model": label, "modality": "vision", "status": "error", "error": str(e)}
    return {
        "model": label, "modality": "vision", "status": "ok",
        "latency_s": round(dt, 3), "tok_s": _tok_rate(result, dt),
        "output": str(result)[:200],
    }


def bench_audio(mc, slot: str, label: str) -> dict:
    """Audio/STT inference via infer_with_audio."""
    if not os.path.exists(TEST_AUDIO):
        return {"model": label, "modality": "audio", "status": "error", "error": "no test audio"}
    try:
        result, dt = _measure(mc.infer_with_audio, TEST_AUDIO, AUDIO_PROMPT, 128, [], "stt")
    except Exception as e:
        return {"model": label, "modality": "audio", "status": "error", "error": str(e)}
    return {
        "model": label, "modality": "audio", "status": "ok",
        "latency_s": round(dt, 3), "tok_s": _tok_rate(result, dt),
        "output": str(result)[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark local GPU models")
    parser.add_argument("--text", action="store_true", help="run text benchmark")
    parser.add_argument("--tools", action="store_true", help="run tool-calling benchmark")
    parser.add_argument("--vision", action="store_true", help="run vision benchmark")
    parser.add_argument("--audio", action="store_true", help="run audio/STT benchmark")
    parser.add_argument("--models", default="gemma,qwen,nemotron",
                        help="comma list of gemma, qwen, nemotron, omni, janus")
    parser.add_argument("--json", action="store_true", help="output raw JSON")
    args = parser.parse_args()

    run_text = args.text or not (args.tools or args.vision or args.audio)
    run_tools = args.tools
    run_vision = args.vision
    run_audio = args.audio

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    # Map model key -> (slot name, label)
    MODEL_SLOTS = {
        "gemma": ("audio", "Gemma 4 E2B-it (audio slot)"),
        "qwen": ("tool_calling", "Qwen3.5-0.8B (tool_calling slot)"),
        "nemotron": ("primary", "Nemotron-Diffusion 3B (primary)"),
        "omni": ("omni", "Qwen2.5-Omni-3B"),
        "janus": ("janus", "Janus-Pro-7B"),
    }

    mc = _import_client()
    if not mc.is_server_running():
        print("ERROR: model server not running. Start it with ./start.sh first.")
        sys.exit(1)

    results = []
    print(f"=== Local model benchmark ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"Image: {TEST_IMAGE} | Audio: {TEST_AUDIO}\n")

    for key in models:
        if key not in MODEL_SLOTS:
            print(f"SKIP unknown model '{key}'")
            continue
        slot, label = MODEL_SLOTS[key]
        print(f"--- {label} (slot={slot}) ---")

        if run_text:
            r = bench_text(mc, slot, label)
            results.append(r)
            print(f"  text : {r.get('status')} | {r.get('latency_s','-')}s | {r.get('tok_s','-')} tok/s | {str(r.get('output',''))[:80]}")
        if run_tools:
            r = bench_tools(mc, slot, label)
            results.append(r)
            print(f"  tools: {r.get('status')} | {r.get('latency_s','-')}s | {str(r.get('output',''))[:80]}")
        if run_vision:
            r = bench_vision(mc, slot, label)
            results.append(r)
            print(f"  vision: {r.get('status')} | {r.get('latency_s','-')}s | {str(r.get('output',''))[:80]}")
        if run_audio:
            r = bench_audio(mc, slot, label)
            results.append(r)
            print(f"  audio: {r.get('status')} | {r.get('latency_s','-')}s | {str(r.get('output',''))[:80]}")
        print()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=== Summary ===")
        print(f"{'Model':<28}{'Modality':<9}{'Status':<8}{'Latency(s)':<12}{'tok/s'}")
        for r in results:
            print(f"{r.get('model',''):<28}{r.get('modality',''):<9}{r.get('status',''):<8}"
                  f"{str(r.get('latency_s','-')):<12}{r.get('tok_s','-')}")


if __name__ == "__main__":
    main()
