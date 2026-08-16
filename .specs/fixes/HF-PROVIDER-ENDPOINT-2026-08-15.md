# HF Provider Endpoint Bug — "inference unavailable" (2026-08-15)

## Overview

With `providers.task_inference: hf`, all chat/task inference returned
`(inference unavailable)`. The server log showed:

```
[provider] hf infer_with_tools failed for task_inference: <urlopen error [Errno -5] No address associated with hostname> — trying next in chain
[provider] local infer_with_tools returned error — trying next in chain: [model_client error] [Errno 2] No such file or directory
[provider] openrouter infer_with_tools failed for task_inference: HTTP Error 400: Bad Request — trying next in chain
[provider] copilot infer_with_tools failed for task_inference: HTTP Error 400: Bad Request — trying next in chain
[provider] all providers exhausted for infer_with_tools/task_inference.
```

The fallback chain also failed (local model server not running in cloud mode;
openrouter/copilot rejected the HF model slug), so the whole request fell through
to `(inference unavailable)`.

## Root cause

`src/core/inference/provider.py` builds the Hugging Face request against the
**legacy, deprecated host**:

```python
f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
```

`api-inference.huggingface.co` no longer resolves → `[Errno -5] No address
associated with hostname`. The current OpenAI-compatible endpoint is the **HF
Router**:

```
https://router.huggingface.co/v1/chat/completions
```

with the provider selected by a **`:provider` suffix on the model id** (per the
DeepInfra docs), e.g. `deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra`.

Two additional gotchas found while testing:
- The router **blocks urllib's default User-Agent (403)** — a real `User-Agent`
  header is required.
- `deepseek-ai/DeepSeek-V4-Flash-0731` is served by `novita`, `together`,
  `fireworks-ai`, `deepinfra` — **not** `hf-inference`. `deepinfra` is cheapest
  ($0.08 in vs $0.14 for together) and works via the model-id suffix.

### Provider selection (verified live 2026-08-15)

| provider | router-root + `:suffix` |
|---|---|
| `deepinfra` | ✅ works (cheapest) |
| `together` | ✅ works |
| (no suffix) | ✅ works (default provider) |
| `hf-inference` | ❌ "Model not supported" (doesn't serve this model) |

Note: `deepinfra`/`novita`/`fireworks-ai` did NOT work when the provider was put
in the URL path (`/deepinfra/v1/...`) — only the model-id `:suffix` form works.

## Fix

In `src/core/inference/provider.py`:

1. `_hf_router_provider()` — resolve provider name:
   env `HF_ROUTER_PROVIDER` → config `providers.hf_provider` → default `deepinfra`.
2. `_hf_model(model)` — append `:{provider}` suffix to the model id.
3. `_hf_base_url()` — return `https://router.huggingface.co/v1`.
4. `_call_hf` / `_hf_tool_loop` — POST to `{base}/chat/completions` with the
   suffixed model id and a real `User-Agent` header.

Also added `providers.hf_provider: deepinfra` to `config.yaml` and documented
`HF_ROUTER_PROVIDER` in `.env.example` / `README.md`.

Note: DeepSeek-V4-Flash is a reasoning model — it emits reasoning tokens and may
return empty `content` if `max_tokens` is too small. The provider uses
`max_tokens: 8192`, which is sufficient.

## Verification

- `_call_hf` live test → `'hello world'` via `deepinfra`.
- `_hf_tool_loop` live test → model issued a tool call and produced a final reply.
- Provider unit tests: 3 pre-existing failures in `test_provider_infer_with_tools.py`
  (local→openai fallback; unrelated to HF, present on `main` before this change —
  confirmed by stashing the change and re-running).

## Status

[x] Completed
