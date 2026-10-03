"""api/routers/models_voice.py — model management and voice routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Local model
scan/curation helpers (_local_model_scan/_local_model_size/_repo_downloaded/
_curated_local_models/_fetch_openrouter_models) and voice helpers
(_build_voice_audio_response) live here with the routes that use them.
Route fns resolve live api state through the _api() seam. Behavior-identical;
re-exported from api.__init__.
"""
import sys
import importlib
import asyncio
import os
import json
import re
import time
import uuid
import yaml

from ..helpers import _log_interaction, _resolve_config_path

from fastapi import APIRouter, Body as _Body, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from core.voice_activity import voice_activity

router = APIRouter()
# Dynamic model catalog (moved from api.__init__ — only used by these routes)
_MODEL_CACHE: dict = {}
_MODEL_CACHE_TTL = 3600  # 1 hour

# Voice server timeouts (moved from api.__init__)
_VOICE_CHAT_TRIAGE_TIMEOUT_S = float(os.environ.get("VOICE_CHAT_TRIAGE_TIMEOUT_S", "60"))
_VOICE_CHAT_TTS_TIMEOUT_S = float(os.environ.get("VOICE_CHAT_TTS_TIMEOUT_S", "180"))

# Hub pull job state (moved from api.__init__)
_pull_jobs: dict[str, dict] = {}  # job_id → job state dict



# Fallback model lists for providers without a dynamic model API
_FALLBACK_MODELS = {
    "openai": {
        "text": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "o3-mini", "o4-mini"],
        "vision": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-pro", "gpt-4.1", "gpt-4o"],
        "audio": ["gpt-audio", "gpt-audio-mini"],
    },
    "anthropic": {
        "text": ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5", "claude-sonnet-4", "claude-opus-4"],
        "vision": ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5", "claude-sonnet-4", "claude-opus-4"],
        "audio": [],
    },
    "hf": {
        "text": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-32B", "mistralai/Mixtral-8x22B"],
        "vision": ["Qwen/Qwen2.5-VL-72B-Instruct"],
        "audio": [],
    },
    "copilot": {
        "text": ["claude-sonnet-4.5", "gpt-5.4"],
        "vision": ["claude-sonnet-4.5", "gpt-5.4"],
        "audio": [],
    },
}



def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


def _fetch_openrouter_models() -> dict:
    """Fetch all models from OpenRouter, grouped by capability."""
    _m = _api()
    import urllib.request, json as _j
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers=headers, method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = _j.loads(resp.read())
        models = data.get("data", [])
        text_models, vision_models, audio_models = [], [], []
        for m in models:
            mid = m.get("id", "")
            inputs = m.get("architecture", {}).get("input_modalities", [])
            if not inputs:
                inputs = ["text"]
            has_text = "text" in inputs or not inputs
            has_image = "image" in inputs
            has_audio = "audio" in inputs
            if has_image:
                vision_models.append(mid)
            if has_audio:
                audio_models.append(mid)
            if has_text and not has_image and not has_audio:
                text_models.append(mid)
        return {
            "text": sorted(text_models),
            "vision": sorted(vision_models),
            "audio": sorted(audio_models),
        }
    except Exception as e:
        print(f"[api] OpenRouter model fetch failed: {e}", flush=True)
        return {}

@router.get("/provider/models")
def get_provider_models(provider: str = "", capability: str = "text"):
    """Return available models per provider and capability.
    _m = _api()
    Query params:
      provider (optional): filter to one provider
      capability (optional): "text" | "vision" | "audio" (default "text")
    """
    import time as _time
    provider = provider.strip()
    capability = capability.strip()

    # Refresh OpenRouter cache if stale
    global _MODEL_CACHE
    now = _time.time()
    if "openrouter" not in _MODEL_CACHE or now - _MODEL_CACHE.get("_ts", 0) > _MODEL_CACHE_TTL:
        or_models = _fetch_openrouter_models()
        if or_models:
            _MODEL_CACHE["openrouter"] = or_models
            _MODEL_CACHE["_ts"] = now

    # Build response: map all providers with their model lists for the requested capability
    result = {}
    providers_to_include = [provider] if provider else list(_FALLBACK_MODELS.keys()) + ["openrouter"]
    for prov in providers_to_include:
        if prov == "openrouter":
            mlist = _MODEL_CACHE.get("openrouter", {}).get(capability, [])
        else:
            mlist = _FALLBACK_MODELS.get(prov, {}).get(capability, [])
        result[prov] = mlist

    return {"capability": capability, "models": result}

def _curated_local_models() -> list[dict]:
    """Return a curated list of local models available to pull/assign."""
    _m = _api()
    curated = [
        {"repo_id": "google/gemma-4-E2B-it", "label": "Gemma 4 E2B (2.3B)", "slot": "audio", "multimodal": True},
        {"repo_id": "google/gemma-4-E4B-it", "label": "Gemma 4 E4B (4.5B)", "slot": None, "multimodal": True},
        {"repo_id": "nvidia/Nemotron-Labs-Diffusion-3B", "label": "Nemotron-Diffusion 3B", "slot": "primary", "multimodal": False},
        {"repo_id": "Qwen/Qwen2.5-Omni-3B", "label": "Qwen2.5-Omni 3B", "slot": "audio", "multimodal": True},
        {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "label": "Qwen3-VL 2B Instruct", "slot": None, "multimodal": True},
        {"repo_id": "deepseek-ai/Janus-Pro-7B", "label": "Janus-Pro 7B (vision)", "slot": None, "multimodal": True},
    ]
    return curated

def _local_model_scan() -> list[dict]:
    """Scan kernel-evolving's dedicated models folder + configured slots.
    _m = _api()

    Scans only MODELS_DIR (where /pull downloads) and the configured
    model_slots paths — NOT the whole shared HF cache. This keeps the local
    model list limited to agent-compatible models instead of every model the
    user has cached (e.g. FLUX image models, TTS voices).
    """
    try:
        from runtime_paths import MODELS_DIR
        models_dir = str(MODELS_DIR)
    except Exception:
        models_dir = None
    models: list[dict] = []
    seen: set = set()
    roots: list[str] = []
    if models_dir and os.path.isdir(models_dir):
        roots.append(models_dir)
        # snapshot_download(cache_dir=MODELS_DIR) stores snapshots under
        # MODELS_DIR/hub/models--org--repo, so also scan that subdir.
        hub_sub = os.path.join(models_dir, "hub")
        if os.path.isdir(hub_sub):
            roots.append(hub_sub)

    # First pass: scan the directory roots (dedicated MODELS_DIR + its hub
    # subdir) for models--org--repo dirs. This covers models pulled via /pull.
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            # Skip hidden/lock dirs (e.g. .locks) which are not models.
            if entry.startswith("."):
                continue
            if not os.path.isdir(full):
                continue
            # The 'hub' subdir itself is not a model — only its children are
            # (and they're scanned as their own root above).
            if entry == "hub":
                continue
            if entry.startswith("models--"):
                repo_id = entry[len("models--"):].replace("--", "/")
            else:
                repo_id = entry
            if repo_id in seen:
                continue
            seen.add(repo_id)
            models.append({
                "model": repo_id,
                "local_path": full,
                "size_bytes": None,
                "slot": None,
                "downloaded": True,
            })

    # Second pass: surface each configured model_slots entry. These are the
    # agent's compatible models (Gemma, Nemotron, Qwen, ...) which may live in
    # the shared HF cache (KERNEL_EVO_HF_HUB) rather than MODELS_DIR. We derive
    # the repo id from the slot's model_path (e.g. .../hub/models--org--repo/
    # snapshots/<hash>) so they show as downloaded/ready without pulling in the
    # incompatible shared-cache junk (FLUX, TTS voices, etc.).
    try:
        slots = (_m._cfg.get("model_slots", {}) or {}) if isinstance(_m._cfg, dict) else {}
    except Exception:
        slots = {}
    for name, slot in (slots or {}).items():
        sp = (slot or {}).get("model_path") or ""
        if not sp:
            continue
        expanded = os.path.expandvars(os.path.expanduser(sp))
        if not os.path.isdir(expanded):
            continue
        # Derive repo id from the hub-cache layout: .../models--org--repo/...
        m = re.search(r"models--([^/]+)--([^/]+)", expanded)
        if m:
            repo_id = f"{m.group(1)}/{m.group(2)}"
            local_path = expanded
        else:
            # Bare path — use the dir name as the model id.
            repo_id = os.path.basename(os.path.normpath(expanded))
            local_path = expanded
        if repo_id in seen:
            continue
        seen.add(repo_id)
        models.append({
            "model": repo_id,
            "local_path": local_path,
            "size_bytes": None,
            "slot": name,
            "downloaded": True,
        })
    return models

def _local_model_size(path: str) -> int:
    """Return a model directory's size in bytes via a portable, bounded walk.
    _m = _api()

    Uses os.scandir recursively (pure Python — works on Linux/macOS/Windows
    with no external `du` dependency). Bounded so a huge model dir cannot
    stall the request indefinitely.
    """
    total = 0
    try:
        stack = [path]
        visited = 0
        while stack and visited < 50000:  # safety cap on entries scanned
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        visited += 1
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(e.path)
                            else:
                                total += e.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
            except OSError:
                continue
    except Exception:
        return 0
    return total

def _repo_downloaded(repo_id: str, local_ids: set) -> bool:
    """Return True if a curated repo is already available locally.
    _m = _api()

    Checks (in order): the dedicated MODELS_DIR scan result, then the shared HF
    hub-cache layout (KERNEL_EVO_HF_HUB / HF_HOME). This lets curated compatible
    models (Gemma, Nemotron, Qwen) that were downloaded before the dedicated
    folder existed still show as "downloaded/ready", without pulling incompatible
    shared-cache models (FLUX, TTS voices) into the /models list itself.
    """
    rid = (repo_id or "").lower()
    if rid in local_ids:
        return True
    # Check the shared HF hub-cache: <root>/hub/models--org--repo
    try:
        from core.hf_cache import resolve_hf_hub_dir
        hub = resolve_hf_hub_dir()
        dir_name = "models--" + repo_id.replace("/", "--")
        return bool(hub and os.path.isdir(os.path.join(str(hub), dir_name)))
    except Exception:
        return False

@router.get("/models")
def list_models(with_size: str = "false"):
    """List locally downloaded models (dedicated folder + configured slots).
    _m = _api()

    Query param `with_size=true` computes each model's size (slower). By
    default sizes are None so the listing is fast and portable.
    """
    with_size_flag = str(with_size).lower() in ("1", "true", "yes", "y")
    local = _local_model_scan()
    if with_size_flag:
        for m in local:
            if m.get("local_path"):
                m["size_bytes"] = _local_model_size(m["local_path"])
    curated = _curated_local_models()
    # Mark which curated models are already downloaded (dedicated folder OR
    # shared HF hub-cache).
    local_ids = {m["model"].lower() for m in local}
    for c in curated:
        c["downloaded"] = _repo_downloaded(c["repo_id"], local_ids)
    # Expose which local slot the Think-at-Rest engine uses, and which model is
    # currently assigned to each slot, so callers can see/decide the thought model.
    thought_slot = "audio"
    slots = {}
    try:
        _slots_m._cfg = (_m._cfg.get("model_slots", {}) or {}) if isinstance(_m._cfg, dict) else {}
        slots = dict(_slots_m._cfg)
        thought_slot = ((_m._cfg.get("thinking", {}) or {}).get("thought_slot") or "audio")
    except Exception:
        pass
    return {
        "models": local,
        "curated": curated,
        "thought_slot": thought_slot,
        "slots": slots,
    }

@router.get("/models/curated")
def curated_models():
    """Return the curated local-model catalog (pull + assign candidates)."""
    _m = _api()
    local = _local_model_scan()
    local_ids = {m["model"].lower() for m in local}
    curated = _curated_local_models()
    for c in curated:
        c["downloaded"] = _repo_downloaded(c["repo_id"], local_ids)
    return {"curated": curated}

@router.get("/hub/search")
def hub_search(q: str = "", limit: int = 20):
    """Search HuggingFace Hub for models to pull. Uses HfApi.list_models."""
    _m = _api()
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        return JSONResponse({"error": f"huggingface_hub not available: {e}"}, status_code=500)
    try:
        api = HfApi()
        limit = max(1, min(int(limit), 50))
        if q.strip():
            models = api.list_models(search=q.strip(), limit=limit)
        else:
            models = api.list_models(limit=limit)
        results = [
            {
                "id": m.modelId,
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
            }
            for m in models
        ]
        return {"models": results}
    except Exception as e:
        return JSONResponse({"error": f"Hub search failed: {e}"}, status_code=502)

@router.post("/pull")
async def pull_model(body: dict):
    """Download a model from HuggingFace Hub in the background.
    _m = _api()

    Body: {model: "repo_id", init?: bool}. Returns {status, job_id}.
    Poll GET /jobs/{job_id} for progress. Mirrors ai-server-py /pull.
    """
    model_name = (body.get("model") or "").strip()
    if not model_name:
        return JSONResponse({"error": "Missing 'model'"}, status_code=400)
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        return JSONResponse({"error": f"huggingface_hub not available: {e}"}, status_code=500)

    job_id = str(uuid.uuid4())
    now = time.time()
    _pull_jobs[job_id] = {
        "job_id": job_id,
        "model": model_name,
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "local_path": None,
        "size_bytes": None,
    }

    async def _background_pull(jid: str, repo_id: str):
        job = _pull_jobs.get(jid)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = time.time()
        try:
            token = os.environ.get("HF_TOKEN", "")
            # Download into the shared HF hub root (KERNEL_EVO_HF_HUB, e.g.
            # /mnt/e/models/huggingface) so models land in its `hub/` subdir and
            # are shared across projects — NOT in kernel-evolving's workspace.
            # Falls back to MODELS_DIR only if the hub root env is unset.
            hub_root = os.environ.get("KERNEL_EVO_HF_HUB", "").strip()
            if hub_root:
                os.makedirs(hub_root, exist_ok=True)
                cache_dir = hub_root
            else:
                from runtime_paths import MODELS_DIR
                os.makedirs(MODELS_DIR, exist_ok=True)
                cache_dir = str(MODELS_DIR)
            path = await asyncio.to_thread(
                snapshot_download, repo_id=repo_id, cache_dir=cache_dir,
                token=token, ignore_patterns=["*.gguf"],
            )
            job["local_path"] = path
            job["size_bytes"] = _local_model_size(path)
            job["status"] = "succeeded"
            job["finished_at"] = time.time()
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = time.time()

    asyncio.create_task(_background_pull(job_id, model_name))
    return {"status": "accepted", "job_id": job_id}

@router.get("/jobs/{job_id}")
def get_pull_job(job_id: str):
    """Poll a background pull job's progress."""
    _m = _api()
    job = _pull_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job

@router.post("/models/assign")
def assign_model_to_slot(body: dict):
    """Assign a pulled local model to a named model_slots entry.
    _m = _api()

    Body: {slot: "audio", repo_id?: "google/gemma-4-E2B-it", local_path?: "/abs/path"}.
    Writes config.yaml model_slots.<slot>.model_path. The slot remains lazy-loaded
    on first use (no forced load).
    """
    slot = (body.get("slot") or "").strip()
    repo_id = (body.get("repo_id") or "").strip()
    local_path = (body.get("local_path") or "").strip()
    if not slot:
        return JSONResponse({"error": "Missing 'slot'"}, status_code=400)

    # Resolve the local path: explicit path, or find the pulled snapshot for repo_id.
    if not local_path and repo_id:
        for m in _local_model_scan():
            if m["model"].lower() == repo_id.lower():
                local_path = m["local_path"]
                break
    if not local_path:
        return JSONResponse(
            {"error": "Model not downloaded locally. Pull it first (POST /pull) or provide local_path."},
            status_code=404,
        )

    # Persist to config.yaml model_slots.<slot>.model_path.
    cfg_path = _resolve_config_path()
    try:
        with open(cfg_path) as f:
            on_disk = yaml.safe_load(f) or {}
    except Exception as e:
        return JSONResponse({"error": f"Could not read config.yaml: {e}"}, status_code=500)

    on_disk.setdefault("model_slots", {})
    on_disk["model_slots"].setdefault(slot, {})
    on_disk["model_slots"][slot]["model_path"] = local_path
    if "role" not in on_disk["model_slots"][slot]:
        on_disk["model_slots"][slot]["role"] = slot
    try:
        with open(cfg_path, "w") as f:
            yaml.dump(on_disk, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        return JSONResponse({"error": f"Could not write config.yaml: {e}"}, status_code=500)

    # Update in-memory config too.
    _m._cfg.setdefault("model_slots", {})
    _m._cfg["model_slots"].setdefault(slot, {})
    _m._cfg["model_slots"][slot]["model_path"] = local_path

    return {"ok": True, "slot": slot, "local_path": local_path, "repo_id": repo_id}

@router.get("/config/thought-slot")
def get_thought_slot():
    """Return which local model slot the Think-at-Rest engine uses for thoughts.
    _m = _api()

    The slot name (e.g. "audio" = Gemma 4 E2B-it, "primary" = Nemotron, or a
    custom slot) controls BOTH thought generation and evaluation, so only one
    local model loads. Also returns the available slots and their assigned models.
    """
    thought_slot = "audio"
    slots = {}
    thinking = (_m._cfg.get("thinking", {}) or {}) if isinstance(_m._cfg, dict) else {}
    thought_slot = thinking.get("thought_slot") or "audio"
    try:
        slots = dict((_m._cfg.get("model_slots", {}) or {}))
    except Exception:
        slots = {}
    return {"thought_slot": thought_slot, "slots": slots}

@router.post("/config/thought-slot")
def set_thought_slot(body: dict):
    """Set which local model slot the Think-at-Rest engine uses for thoughts.
    _m = _api()

    Body: {slot: "audio" | "primary" | "tool_calling" | <custom slot name>}.
    Persists `thinking.thought_slot` in config.yaml. Takes effect on the next
    think cycle (no restart required for the in-memory config; a restart is only
    needed if you want the change reflected in the running engine's constructor).
    """
    slot = (body.get("slot") or "").strip()
    if not slot:
        return JSONResponse({"error": "Missing 'slot'"}, status_code=400)

    # Validate the slot exists in model_slots (or is a known slot name).
    slots = {}
    try:
        slots = dict((_m._cfg.get("model_slots", {}) or {}))
    except Exception:
        slots = {}
    if slot not in slots:
        return JSONResponse(
            {"error": f"Unknown slot '{slot}'. Available slots: {sorted(slots.keys()) or ['audio','primary','tool_calling']}"},
            status_code=400,
        )

    # Persist to config.yaml thinking.thought_slot.
    cfg_path = _resolve_config_path()
    try:
        with open(cfg_path) as f:
            on_disk = yaml.safe_load(f) or {}
    except Exception as e:
        return JSONResponse({"error": f"Could not read config.yaml: {e}"}, status_code=500)

    on_disk.setdefault("thinking", {})
    on_disk["thinking"]["thought_slot"] = slot
    try:
        with open(cfg_path, "w") as f:
            yaml.dump(on_disk, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        return JSONResponse({"error": f"Could not write config.yaml: {e}"}, status_code=500)

    # Update in-memory config too (so the next think cycle uses it).
    _m._cfg.setdefault("thinking", {})
    _m._cfg["thinking"]["thought_slot"] = slot

    return {"ok": True, "thought_slot": slot}

async def _build_voice_audio_response(reply: str, session_id: str = ""):
    """Try voice clone TTS and return audio Response or JSON fallback payload."""
    _m = _api()
    from fastapi.responses import Response
    from fastapi.concurrency import run_in_threadpool
    import services.channels.telegram_bot as _tb

    try:
        wav_path = await asyncio.wait_for(
            run_in_threadpool(_tb._clone_voice_reply, reply),
            timeout=_VOICE_CHAT_TTS_TIMEOUT_S,
        )
        if wav_path and os.path.isfile(wav_path):
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            os.unlink(wav_path)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Assistant-Reply": reply[:2000],
                    "X-Voice-Session": session_id[:120],
                    "X-Voice-Clone": "ok",
                },
            )
        raise RuntimeError("clone returned no file")
    except asyncio.TimeoutError:
        return JSONResponse({
            "reply": reply,
            "tts_error": "timeout",
            "session_id": session_id,
        })
    except Exception as e:
        print(f"[voice/chat] TTS unavailable: {e}")
        return JSONResponse({"reply": reply, "tts_error": str(e), "session_id": session_id})

@router.get("/voice/status")
def voice_status():
    """Check whether the voice server (olly-voice-server) is configured and reachable."""
    _m = _api()
    import urllib.request
    cfg_voice = (_m._cfg.get("services", {}).get("voice_server", {}) if isinstance(_m._cfg, dict) else {})
    url = _m._VOICE_SERVER_URL
    install_hint = cfg_voice.get("install_hint", "")

    if not url:
        return {
            "available": False,
            "configured": False,
            "url": None,
            "message": "Voice server not configured. Set services.voice_server.url in config.yaml or VOICE_SERVER_URL env var.",
            "install_hint": install_hint,
        }

    try:
        req = urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3)
        reachable = req.status == 200
    except Exception:
        reachable = False

    return {
        "available": reachable,
        "configured": True,
        "url": url,
        "message": "Voice server reachable" if reachable else f"Voice server configured at {url} but not reachable. Start it first.",
        "install_hint": install_hint if not reachable else "",
    }

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """STT via Gemma 4 E2B -- saves upload to temp file, runs infer_with_audio."""
    _m = _api()
    import tempfile
    from fastapi.concurrency import run_in_threadpool
    print(f"[transcribe] Request: {file.filename} ({file.content_type})", flush=True)
    data = await file.read()
    print(f"[transcribe] Payload: {len(data)} bytes", flush=True)
    if not data or len(data) < 256:
        return JSONResponse({"error": "STT failed: empty or too-short audio payload"}, status_code=400)

    # Prefer extension from filename; fallback from content-type for browser recordings.
    filename = file.filename or ""
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    else:
        ctype = (file.content_type or "").lower()
        if "ogg" in ctype:
            suffix = ".ogg"
        elif "wav" in ctype:
            suffix = ".wav"
        elif "mp3" in ctype or "mpeg" in ctype:
            suffix = ".mp3"
        else:
            suffix = ".webm"

    if suffix not in {".webm", ".ogg", ".wav", ".mp3", ".m4a", ".aac", ".flac"}:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    print(f"[transcribe] Saved to: {tmp_path} ({suffix})", flush=True)
    try:
        with voice_activity("stt"):
            text = await run_in_threadpool(
                _m.mdl.infer_with_audio,
                tmp_path,
                "Transcribe exactly what is said in this audio. Output only the spoken words, nothing else.",
                256,
                None,  # history
                "stt",  # mode - explicitly force STT mode
            )
        print(f"[transcribe] Result: {repr(text[:120])}", flush=True)
        if isinstance(text, str) and text.startswith("[model_server error]"):
            return JSONResponse({"error": f"STT failed: {text}"}, status_code=500)
        if isinstance(text, str) and text.startswith("[model_client error]"):
            return JSONResponse({"error": f"STT failed: {text}"}, status_code=500)
        return {"text": text or ""}
    except Exception as e:
        print(f"[transcribe] Exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@router.post("/voice/chat")
async def voice_chat(body: dict):
    """text -> agent.triage() -> _clone_voice_reply() -> WAV response.
    _m = _api()
    Returns WAV bytes with X-Assistant-Reply header.
    Falls back to JSON {reply} if TTS unavailable.
    """
    from fastapi.concurrency import run_in_threadpool
    user_text = (body.get("text") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    print(f"[voice/chat] User: {repr(user_text[:100])}", flush=True)
    if not user_text:
        return JSONResponse({"error": "missing text"}, status_code=400)

    # Full triage -- history, context, tools, skills
    try:
        print(f"[voice/chat] Starting triage (timeout={_VOICE_CHAT_TRIAGE_TIMEOUT_S}s)...", flush=True)
        reply = await asyncio.wait_for(
            run_in_threadpool(_m.agent.triage, user_text, None, False, session_id or "voice-dashboard"),
            timeout=_VOICE_CHAT_TRIAGE_TIMEOUT_S,
        )
        print(f"[voice/chat] Triage complete: {repr(reply[:100])}", flush=True)
    except asyncio.TimeoutError:
        print(f"[voice/chat] Triage TIMEOUT after {_VOICE_CHAT_TRIAGE_TIMEOUT_S}s", flush=True)
        fallback = "I got your message, but processing timed out. Please try again."
        return JSONResponse({
            "reply": fallback,
            "timeout": "triage",
            "session_id": session_id,
        })
    except Exception as e:
        print(f"[voice/chat] Triage ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        fallback = f"Processing error: {e}"
        return JSONResponse({
            "reply": fallback,
            "error": str(e),
            "session_id": session_id,
        })

    if not reply or not str(reply).strip():
        print(f"[voice/chat] Empty reply from triage", flush=True)
        return JSONResponse({
            "reply": "I could not generate a response right now. Please try again.",
            "error": "empty_reply",
            "session_id": session_id,
        })

    reply = str(reply).strip()
    print(f"[voice/chat] Starting TTS clone...", flush=True)
    _log_interaction(user_text, reply, source="voice-dashboard")

    # TTS via _clone_voice_reply -- same function used by telegram_bot
    return await _build_voice_audio_response(reply, session_id=session_id)

@router.post("/voice/tts")
async def voice_tts(body: dict):
    """Generate cloned voice audio from text for dashboard retry/playback."""
    _m = _api()
    text = (body.get("text") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)
    with voice_activity("tts"):
        return await _build_voice_audio_response(text, session_id=session_id)

@router.delete("/voice/history/{session_id}")
async def delete_voice_history(session_id: str):
    """Clear voice session -- wipes JSON memory window (SQLite preserved)."""
    _m = _api()
    import core.memory.memory as _mem
    try:
        _mem.clear_chat(session_id)
    except Exception:
        pass
    return {"status": "cleared", "session_id": session_id}
