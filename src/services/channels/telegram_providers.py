"""telegram_providers.py — cloud/modal provider + model pickers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains the
guided /cloud and /modal provider/model picker flows, the short local-repo
token registry, and the provider model catalog fetch. Kept behavior-identical;
telegram_bot.py re-imports these names.
"""
import json
import requests
from telegram_config import CONFIG_PATH

def _bot_module():
    # The bot module is always registered under its dotted package name, in
    # production (api.py: import services.channels.telegram_bot) and in the
    # test harness (spec_from_file_location with the same name). Resolve via
    # sys.modules so test patches on bot.send_buttons/send_message apply.
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _sb(*a, **k):
    return _bot_module().send_buttons(*a, **k)


def _sm(*a, **k):
    return _bot_module().send_message(*a, **k)



def _get_current_model_label() -> str:
    """Return human-readable label of the currently loaded model."""
    try:
        from core.inference.model_client import health as _mch
        h = _mch()
        label = h.get("model", "")
        if h.get("nemotron"):
            label = f"{label} [Nemotron]"
        if label:
            return label
    except Exception:
        pass
    # Fallback: read from config.yaml
    try:
        import yaml as _y
        cfg = _y.safe_load(open(CONFIG_PATH))
        return cfg.get("model", {}).get("name", "local model")
    except Exception:
        return "local model"


# (VRAM-fit / curated-slot helpers moved to model_helpers.py — see issue #3)
# ── Short callback tokens (XP7) ──────────────────────────────────────────────
# Telegram inline callback_data is limited to 64 bytes, so embedding a full HF
# repo_id (e.g. "google/gemma-4-E2B-it") can exceed the limit and Telegram
# rejects the button with BUTTON_DATA_INVALID. We instead use short opaque
# tokens in callback_data and resolve them to repo ids here, server-side.

_LOCAL_REPO_TOKENS: dict = {}  # token -> repo_id


def _register_repo_token(repo_id: str) -> str:
    """Return a short token for a repo_id, registering it if not already present.

    Tokens are stable per repo_id within this process: 'lm0', 'lm1', ...
    """
    for tok, rid in _LOCAL_REPO_TOKENS.items():
        if rid == repo_id:
            return tok
    tok = f"lm{len(_LOCAL_REPO_TOKENS)}"
    _LOCAL_REPO_TOKENS[tok] = repo_id
    return tok


def _resolve_repo_token(token: str) -> str:
    """Resolve a short token back to a repo_id, or return the token unchanged."""
    return _LOCAL_REPO_TOKENS.get(token, token)


# ── Cloud model catalog for guided /cloud flow ─────────────────────────────
# Models are fetched dynamically from /provider/models API endpoint.
# This gives live model lists from OpenRouter and reasonable defaults
# for OpenAI, Anthropic, HuggingFace, and Copilot.


def _fetch_provider_models(provider: str, capability: str = "text") -> list:
    """Fetch model list for a provider+capability from the API endpoint.
    Falls back to minimal defaults if API unavailable."""
    import requests as _req, yaml as _yaml
    try:
        with open(CONFIG_PATH) as _pf:
            _cfg_p = _yaml.safe_load(_pf)
        _api_port = _cfg_p.get('api', {}).get('port', 8779)
        r = _req.get(f"http://localhost:{_api_port}/provider/models",
                      params={"provider": provider, "capability": capability},
                      timeout=5)
        if r.ok:
            data = r.json()
            return data.get("models", {}).get(provider, [])
    except Exception as e:
        print(f"[cloud] model fetch error for {provider}/{capability}: {e}", flush=True)
    # Fallback: minimal sensible defaults
    _FALLBACK = {
        "openai":     ["gpt-5.4", "gpt-4.1", "gpt-4o"],
        "openrouter": ["deepseek/deepseek-v4-flash", "google/gemma-4-26b-a4b-it", "google/gemini-2.5-pro"],
        "anthropic":  ["claude-sonnet-4-6", "claude-opus-4-5"],
        "hf":         ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-32B"],
        "copilot":    ["claude-sonnet-4.5", "gpt-5.4"],
    }
    return _FALLBACK.get(provider, [])

_CLOUD_CALL_TYPES = ["task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"]
_CALL_TYPE_LABELS = {
    "task_inference": "Chat (task inference)",
    "synthesis": "Skill synthesis",
    "critic": "Critic / verification",
    "planning": "Task planning",
    "trajectory_teacher": "Teacher trajectories",
    "vision": "Vision (images)",
    "stt": "Speech-to-Text (STT)",
    "tts": "Text-to-Speech (TTS / voice clone)",
}


def _show_cloud_provider_picker(chat_id: str):
    """Step 1: pick a cloud provider."""
    buttons = [
        [{"text": "\U0001f916 OpenAI",      "callback_data": "cloud_provider_openai"}],
        [{"text": "\U0001f310 OpenRouter",   "callback_data": "cloud_provider_openrouter"}],
        [{"text": "\U0001f9e0 Anthropic",    "callback_data": "cloud_provider_anthropic"}],
        [{"text": "\U0001f917 HuggingFace",  "callback_data": "cloud_provider_hf"}],
        [{"text": "\u26a1 Copilot",          "callback_data": "cloud_provider_copilot"}],
        [{"text": "\u2190 Back",            "callback_data": "/start"}],
    ]
    _sb(chat_id, "\u2601\ufe0f *Cloud mode*\n\nPick a provider for all capabilities. After that, you'll choose a model per call type.", buttons)


def _show_modal_provider_picker(chat_id: str, provider: str, call_type_idx: int):
    """Step for vision/stt/tts: pick a provider (or keep local) before model selection."""
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    buttons = [
        [{"text": "\U0001f916 OpenAI",      "callback_data": f"cmprov_{ct}|openai"}],
        [{"text": "\U0001f310 OpenRouter",   "callback_data": f"cmprov_{ct}|openrouter"}],
        [{"text": "\U0001f9e0 Anthropic",    "callback_data": f"cmprov_{ct}|anthropic"}],
        [{"text": "\U0001f917 HuggingFace",  "callback_data": f"cmprov_{ct}|hf"}],
        [{"text": "\u26a1 Copilot",          "callback_data": f"cmprov_{ct}|copilot"}],
        [{"text": "\U0001f512 Keep local",    "callback_data": f"cmprov_{ct}|local"}],
        [{"text": "\u2190 Back",            "callback_data": "/cloud"}],
    ]
    _sb(chat_id, f"\u2601\ufe0f ({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)}) Pick provider for *{label}*:", buttons)


def _fetch_models_for_ct(provider: str, ct: str) -> list:
    """Return model list for a call type, fetching dynamically."""
    cap_map = {"vision": "vision", "stt": "audio", "tts": "text"}  # TTS is text generation
    cap = cap_map.get(ct, "text")
    return _fetch_provider_models(provider, cap)


def _show_modal_model_picker(chat_id: str, modal_provider: str, call_type_idx: int):
    """Show model picker for vision/stt/tts under a specific provider."""
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    models = _bot_module()._fetch_models_for_ct(modal_provider, ct)
    if not models:
        # No dedicated models for this modality+provider — use default for provider
        _apply_modal_model(chat_id, modal_provider, ct, None, call_type_idx)
        return
    buttons = []
    for m in models:
        cb = f"cmm_{modal_provider}|{ct}|{m}"
        if len(cb.encode("utf-8")) > 64:
            cb = cb[:60]
        buttons.append([{"text": f"\U0001f4e1 {m}", "callback_data": cb}])
    buttons.append([{"text": "\u23ed Use default", "callback_data": f"cmm_{modal_provider}|{ct}|default"}])
    buttons.append([{"text": "\u2190 Back", "callback_data": "/cloud"}])
    step_label = f"({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)})"
    _sb(chat_id, f"\u2601\ufe0f {step_label} Pick model for *{label}* ({modal_provider}):", buttons)


def _apply_modal_model(chat_id: str, modal_provider: str, call_type: str, model: str | None, call_type_idx: int):
    """Set provider+model for vision/stt/tts, then advance."""
    import yaml as _yaml, json as _json_p, urllib.request as _ur
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    body = {call_type: modal_provider}
    if model:
        body["model_override"] = {call_type: model}
    body["persist"] = True
    payload = _json_p.dumps(body).encode()
    try:
        req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                          data=payload, headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=5).read()
    except Exception as _e:
        print(f"[cloud] modal model apply error: {_e}", flush=True)
    # Advance to next call type (use main provider from first 5 steps for the rest)
    if call_type_idx + 1 >= len(_CLOUD_CALL_TYPES):
        _finish_cloud_setup(chat_id, modal_provider)
    else:
        # Re-read main provider for subsequent steps
        _main_prov = "openrouter"
        try:
            with open(CONFIG_PATH) as _pf:
                _cfg_p2 = _yaml.safe_load(_pf)
            _routing = (_cfg_p2.get("providers", {}) or {})
            for _ct in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
                _v = _routing.get(_ct)
                if _v and _v != "local":
                    _main_prov = _v
                    break
        except Exception:
            pass
        _show_cloud_model_picker(chat_id, _main_prov, call_type_idx + 1)


def _show_cloud_model_picker(chat_id: str, provider: str, call_type_idx: int):
    """Show model picker for a specific call type."""
    if call_type_idx >= len(_CLOUD_CALL_TYPES):
        _finish_cloud_setup(chat_id, provider)
        return
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    # For vision/stt/tts, pick provider first (can differ from main provider)
    if ct in ("vision", "stt", "tts"):
        _show_modal_provider_picker(chat_id, provider, call_type_idx)
        return
    models = _fetch_provider_models(provider, "text")
    if not models:
        _show_cloud_model_picker(chat_id, provider, call_type_idx + 1)
        return
    buttons = []
    for m in models:
        cb = f"cm_{provider}|{ct}|{m}"
        # Cap to 64 bytes to avoid Telegram silently dropping the entire message
        if len(cb.encode("utf-8")) > 64:
            cb = cb[:60]  # truncate if somehow too long
        buttons.append([{"text": f"\U0001f4e1 {m}", "callback_data": cb}])
    buttons.append([{"text": "\u23ed Use default for this", "callback_data": f"cm_{provider}|{ct}|default"}])
    buttons.append([{"text": "\u2190 Back to providers", "callback_data": "/cloud"}])
    step_label = f"({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)})"
    _sb(chat_id, f"\u2601\ufe0f {step_label} Pick model for *{label}*:", buttons)


def _apply_cloud_model(chat_id: str, provider: str, call_type: str, model: str | None):
    """Set model override for one call type, then advance to next."""
    import yaml as _yaml, json as _json_p, urllib.request as _ur
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    body = {call_type: provider}
    if model:
        body["model_override"] = {call_type: model}
    body["persist"] = True
    payload = _json_p.dumps(body).encode()
    try:
        req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                          data=payload, headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=5).read()
    except Exception as _e:
        print(f"[cloud] model override error: {_e}", flush=True)
    try:
        idx = _CLOUD_CALL_TYPES.index(call_type)
    except ValueError:
        idx = 0
    _show_cloud_model_picker(chat_id, provider, idx + 1)


def _finish_cloud_setup(chat_id: str, provider: str):
    """All models selected — show summary."""
    import json as _json_p, urllib.request as _ur
    import yaml as _yaml
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    try:
        with _ur.urlopen(f"http://localhost:{_api_port}/provider", timeout=3) as r:
            data = _json_p.loads(r.read())
        routing = data.get("routing", {})
        lines = ["\u2601\ufe0f *Cloud setup complete!*\n"]
        for ct in _CLOUD_CALL_TYPES:
            info = routing.get(ct, {})
            icon = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                    "hf": "\U0001f917", "copilot": "\u26a1", "openrouter": "\U0001f310"}.get(info.get("provider", ""), "\U0001f4e1")
            model_suffix = f" `{info['model']}`" if info.get("model") else ""
            lines.append(f"{icon} `{ct}` \u2192 *{info['provider']}*{model_suffix}")
        lines.append("\nZero local VRAM used. Tap /start to switch modes.")
        _sm(chat_id, "\n".join(lines))
    except Exception as e:
        _sm(chat_id, f"\u2601\ufe0f Cloud configuration saved (provider: {provider}).")


