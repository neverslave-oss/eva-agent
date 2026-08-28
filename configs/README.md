# Local Model Benchmark — kernel-evolving

Benchmark of kernel-evolving behavior through its HTTP API (`POST /message`),
switching configs via `./start.sh --config=configs/<file>`. Each run sends a
battery of real agent-path messages (text, tool calling, follow-up, memory).

Run: `python3 scripts/benchmark_via_api.py --label <config-label> --models "<desc>"`

## Results (2026-08-27)

| Config | Primary model | task_inference | thought_slot | Avg latency | Tool support |
|--------|---------------|----------------|--------------|-------------|--------------|
| `01-baseline-cloud` | Nemotron-3B (4bit) | `hf` (cloud DeepSeek V4 Flash) | Gemma (audio) | **3.68s** | ✅ clean |
| `02-local-gemma-primary` | Gemma 4 E2B (4bit) | `local` | Gemma (audio) | **118.4s** | ✅ works, very slow |
| `03-local-qwen-primary` | Qwen3.5-0.8B | `local` | Qwen (tool_calling) | **4.29s** | ⚠️ no native, falls back |

### Per-test latency (s)

| Test | 01-cloud | 02-gemma-local | 03-qwen-local |
|------|----------|----------------|---------------|
| text | 3.51 | 23.80 | 4.65 |
| tool_calc | 6.10 | 17.98 | 6.98 |
| tool_followup | 1.46 | 246.44 | 1.78 |
| memory_context | 3.66 | 185.45 | 3.76 |

## Insights

1. **Cloud baseline (01) is fastest and cleanest** — Nemotron primary + cloud
   task_inference gives ~3.7s avg with reliable native tool calling.

2. **Local Gemma (02) is 32× slower** — the full agent path through local Gemma
   (2.3B) takes 118s avg; a multi-step tool follow-up took 246s. Not usable for
   interactive chat, but the model is capable (tools + vision + audio).

3. **Local Qwen3.5 (03) is fast (4.3s)** but reports **no native tool-calling**
   support in the model server — it falls back to computing directly (e.g.
   `bc`/manual arithmetic) rather than using the calculator tool. Fast but
   less capable for agentic tool use.

## Next steps (per user plan)

- Test **Qwen2.5-Omni-3B** and **Janus-Pro-7B** as primary/thought slots.
- Pull recent **7B function-calling Qwen models** (via kernel-evolving `/pull`)
  into `/mnt/e/models` (KERNEL_EVO_HF_HUB) and install them as slots.
- Validate **think-at-rest** works when inference is cloud: thoughts run locally
  on the most efficient capable local model (likely Qwen3.5 for speed, or Omni
  for capability).
- Extend the benchmark to cover: a skill invocation, a routine, think-at-rest,
  and a real scaffolding task (landing page).
