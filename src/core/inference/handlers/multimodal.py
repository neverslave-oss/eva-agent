"""handlers/multimodal.py — image/audio/local inference handlers (issue #2 split).

Extracted from src/core/inference/handlers/__init__.py: _handle_infer_with_image,
_handle_infer_with_audio, _handle_infer_local, _cloud_multimodal_infer. Imports
resolve via the core/inference package (..); re-exported from handlers/__init__.py.
"""
import os

from .. import state as server_state
from .. import vllm as _vllm
from ..handlers_common import (
    _get_vllm_processor,
    _load_config,
    _mark_activity_end,
    _mark_activity_start,
    _omni_generate_text,
)
from ..state import target_device as _target_device
from ..loaders import _janus_infer_image
from ..slots import _ensure_model, _ensure_multimodal_slot



def _handle_infer_with_image(params: dict) -> dict:
    """Multimodal image inference.
    Routes to the audio/vision slot (Gemma 4) when the main model is text-only
    (e.g. Nemotron diffusion), falling back to the main model if no vision slot
    is available.
    """
    image_path = params["image_path"]
    prompt = params["prompt"]
    max_new_tokens = params.get("max_new_tokens", 1024)
    # force_local: the native eyes (look/describe) always use the local Gemma
    # E2B vision slot, ignoring the cloud-vision config (which is for Telegram
    # images). When True, skip cloud routing entirely.
    force_local = bool(params.get("force_local", False))

    # ── Janus path (custom MultiModalityCausalLM — not standard generate) ───
    if server_state.is_janus:
        try:
            result = _janus_infer_image(image_path, prompt, max_new_tokens=max_new_tokens)
            return {"result": result}
        except Exception as e:
            print(f"[model_server] Janus image inference failed ({e})", flush=True)
            return {"error": f"Janus vision failed: {e}"}

    # ── Cloud vision routing (only when NOT forced local) ─────────────────────
    if not force_local:
        if server_state.config is None:
            _load_config(server_state.lazy_config_path)
        provider_cfg = (server_state.config or {}).get("providers", {})
        vision_provider = provider_cfg.get("vision", "local")
        if vision_provider != "local":
            vision_model = provider_cfg.get("model_overrides", {}).get("vision",
                              provider_cfg.get("models", {}).get(vision_provider, "google/gemma-4-26b-a4b-it"))
            print(f"[model_server] infer_with_image: routing to cloud ({vision_provider}/{vision_model})", flush=True)
            cloud_resp = _cloud_multimodal_infer("vision", vision_provider, vision_model,
                                                  image_path=image_path, prompt=prompt,
                                                  max_new_tokens=max_new_tokens)
            # Fallback: if the cloud provider failed (HTTP 402 out-of-credit,
            # missing key, network error), route natively to the local Gemma E2B
            # vision slot instead of erroring out.
            if "error" in cloud_resp:
                print(f"[model_server] infer_with_image: cloud vision failed ({cloud_resp['error']}); "
                      f"falling back to native vision slot", flush=True)
            else:
                return cloud_resp

    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    # Prefer the dedicated audio/vision slot (Gemma 4 E2B-it) for image inference
    # when the main model is text-only. Lazy-load via _ensure_multimodal_slot() if not yet warm
    # (same path as voice notes) — works whether PDF or voice came first.
    active_model = None
    active_processor = None
    use_hf_path = False

    if not server_state.audio_capable:
        # Main model is text-only — ensure Gemma 4 vision slot is loaded
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_image: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        if server_state.mm_model is not None and server_state.mm_processor is not None:
            print("[model_server] infer_with_image: routing to audio/vision slot (Gemma 4)", flush=True)
            active_model = server_state.mm_model
            active_processor = server_state.mm_processor
            use_hf_path = True
    elif server_state.mm_model is not None and server_state.mm_processor is not None:
        # Main model IS audio-capable but we still have a warm slot — prefer it
        active_model = server_state.mm_model
        active_processor = server_state.mm_processor
        use_hf_path = True

    if active_model is None:
        # Main model handles vision (or nothing else is available)
        _ensure_model()
        # If the main model is loaded, use it as the processor fallback (it may
        # have a processor even when the dedicated multimodal slot is absent).
        if active_processor is None and server_state.model is not None and server_state.processor is not None:
            active_model = server_state.model
            active_processor = server_state.processor
            use_hf_path = False

    # Guard: if the multimodal slot load left partial state behind, degrade
    # cleanly instead of crashing on active_model.parameters() or processor use.
    if active_model is None or active_processor is None:
        return {"error": "no vision model available (multimodal slot failed to load)"}

    # If we have a dedicated vision model (audio slot / Gemma 4), always use HF path
    if use_hf_path and active_model is not None:
        import torch
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = active_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        target_device = next(active_model.parameters()).device
        inputs = active_processor(text=text, images=[img], return_tensors="pt").to(target_device)
        # Align floating tensors to model dtype
        target_dtype = next(active_model.parameters()).dtype
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and hasattr(v, "to") and torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            elif hasattr(v, "to"):
                inputs[k] = v.to(device=target_device)
        input_len = inputs["input_ids"].shape[-1]
        _mark_activity_start()
        try:
            with torch.no_grad():
                out = active_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            _mark_activity_end()
        result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return {"result": result}

    if _vllm.is_enabled():
        try:
            proc = _get_vllm_processor()
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ]
            # Format text prompt; pass image as multi_modal_data
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            import torch
            inputs_proc = proc(text=text, images=[img], return_tensors="pt")
            # vLLM multimodal: pass pixel_values via multi_modal_data
            pixel_values = inputs_proc.get("pixel_values")
            mm_data = {"image": img} if pixel_values is not None else None

            from vllm import SamplingParams
            sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
            vllm_input = {"prompt": text}
            if mm_data:
                vllm_input["multi_modal_data"] = mm_data
            result = _vllm.run(_vllm.generate_multimodal_async(vllm_input, sp))
            return {"result": result.strip()}
        except Exception as e:
            print(f"[model_server] vLLM image inference failed ({e}) — falling back to HF", flush=True)
            if server_state.model is None:
                return {"error": f"vLLM failed and HF model not loaded: {e}"}

    # HF path
    import torch
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = server_state.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = server_state.processor(text=text, images=[img], return_tensors="pt").to(_target_device())
    input_len = inputs["input_ids"].shape[-1]
    if server_state.is_omni:
        # Omni's generate() is non-standard — use generation_mode="text" so it
        # returns text tokens fast instead of doing slow audio synthesis.
        out = _omni_generate_text(inputs, max_new_tokens=max_new_tokens)
    else:
        with torch.no_grad():
            _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if server_state.drafter is not None:
                _gen_kwargs["assistant_model"] = server_state.drafter
            out = server_state.model.generate(**inputs, **_gen_kwargs)
    result = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return {"result": result}

def _handle_infer_with_audio(params: dict) -> dict:
    """Multimodal audio inference.
    NOTE: Uses HF transformers for audio tensor inference (stable path).
    vLLM 0.21.0 supports Gemma4ForConditionalGeneration but multimodal audio
    input via vLLM API is deferred — HF path is simpler and well-tested.

    IMPORTANT: do NOT call _ensure_model() up front here. That loads the main
    text model (e.g. Nemotron), which is NOT audio-capable and fails with the
    VRAM guard when the Gemma multimodal slot is already resident. STT/audio
    routes to the Gemma multimodal slot (or a named audio slot) instead; the
    main model is only loaded in the branch where it is actually the
    audio-capable model.
    """
    # ── Cloud audio routing ──────────────────────────────────────────────────
    if server_state.config is None:
        _load_config(server_state.lazy_config_path)
    provider_cfg = (server_state.config or {}).get("providers", {})
    audio_provider = provider_cfg.get("stt", "local")
    if audio_provider != "local":
        audio_model = provider_cfg.get("model_overrides", {}).get("stt",
                          provider_cfg.get("models", {}).get(audio_provider, "google/gemini-2.5-pro"))
        print(f"[model_server] infer_with_audio: routing to cloud STT ({audio_provider}/{audio_model})", flush=True)
        prompt = params.get("prompt", "Transcribe this audio.")
        max_new_tokens = params.get("max_new_tokens", 1024)
        cloud_resp = _cloud_multimodal_infer("audio", audio_provider, audio_model,
                                              audio_path=params["audio_path"], prompt=prompt,
                                              max_new_tokens=max_new_tokens)
        # Fallback: if the cloud STT provider failed (HTTP 402 out-of-credit,
        # missing key, network error), route natively to the local Gemma E2B
        # multimodal slot (native STT) instead of erroring out.
        if "error" in cloud_resp:
            print(f"[model_server] infer_with_audio: cloud STT failed ({cloud_resp['error']}); "
                  f"falling back to native STT slot", flush=True)
        else:
            return cloud_resp

    import torch
    import soundfile as sf
    import numpy as np
    import subprocess
    import tempfile

    audio_path = params["audio_path"]
    prompt = params.get("prompt", "The user sent you a voice message. Listen and reply naturally.")
    max_new_tokens = params.get("max_new_tokens", 1024)
    history = params.get("history", [])
    mode = params.get("mode", "stt")

    # Pick which model/processor to use
    # Priority: named slot (if registry present and slot specified/loaded) → legacy path
    _requested_slot = params.get("slot")
    _use_slot_name = _requested_slot or ("audio" if server_state.slot_registry is not None else None)
    _slot_state = server_state.slot_registry.get(_use_slot_name) if (server_state.slot_registry is not None and _use_slot_name) else None

    if _slot_state is not None:
        active_model = _slot_state.model
        active_processor = _slot_state.processor
        print(f"[model_server] infer_with_audio: using named slot {_use_slot_name!r}", flush=True)
    elif server_state.audio_capable and not _vllm.is_enabled():
        # Legacy: main model is audio-capable on HF path. Only now do we load
        # the main model — it is the audio-capable model in this config.
        _ensure_model()
        active_model = server_state.model
        active_processor = server_state.processor
        print(f"[model_server] infer_with_audio: using main model (HF, audio_capable)", flush=True)
    else:
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_audio: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        active_model = server_state.mm_model
        active_processor = server_state.mm_processor
        print(f"[model_server] infer_with_audio: using multimodal slot (HF)", flush=True)

    # Guard: if the multimodal slot load left partial state behind, degrade
    # cleanly instead of crashing on active_model.parameters() or processor use.
    if active_model is None or active_processor is None:
        return {"error": "no audio model available (multimodal slot failed to load)"}

    # Normalise to 16kHz mono float32 WAV via ffmpeg
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        print(f"[infer_with_audio] Input: {audio_path} ({mode} mode)", flush=True)
        import os.path
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        print(f"[infer_with_audio] File size: {os.path.getsize(audio_path)} bytes", flush=True)
        
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", tmp_wav],
            capture_output=True, timeout=30, text=True
        )
        if result.returncode != 0:
            print(f"[infer_with_audio] ffmpeg error: {result.stderr}", flush=True)
            result.check_returncode()  # raise if failed
        
        audio_array, sample_rate = sf.read(tmp_wav, dtype="float32")
        print(f"[infer_with_audio] Decoded: {len(audio_array)} samples @ {sample_rate}Hz", flush=True)
    except Exception as e:
        print(f"[infer_with_audio] Decode via ffmpeg failed: {e}, trying raw read...", flush=True)
        try:
            audio_array, sample_rate = sf.read(audio_path, dtype="float32")
            print(f"[infer_with_audio] Raw read succeeded: {len(audio_array)} samples @ {sample_rate}Hz", flush=True)
        except Exception as e2:
            print(f"[infer_with_audio] Raw read also failed: {e2}", flush=True)
            return {"error": f"Audio decode failed: {e}"}
    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
        print(f"[infer_with_audio] Converted to mono", flush=True)
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
    print(f"[infer_with_audio] Amplitude range: [{audio_array.min():.4f}, {audio_array.max():.4f}]", flush=True)

    if mode == "stt":
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sample_rate}},
                {"type": "text", "text": prompt},
            ]}
        ]
    else:
        messages = [
            {"role": "system", "content": (
                "You are Evo, a conversational AI. "
                "When given audio, listen to what the user says and reply naturally. "
                "Do NOT just transcribe — understand and respond."
            )},
        ]
        for m in history:
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            content = str(m.get("content", ""))[:400]
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        messages.append({"role": "user", "content": [
            {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sample_rate}},
            {"type": "text", "text": prompt},
        ]})

    text = active_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_device = _target_device() if active_model is server_state.model else next(active_model.parameters()).device
    target_dtype = next(active_model.parameters()).dtype
    print(f"[infer_with_audio] Target: device={target_device}, dtype={target_dtype}", flush=True)

    print(f"[infer_with_audio] Processing audio through {active_model.__class__.__name__}...", flush=True)
    inputs = active_processor(
        text=text,
        audio=[audio_array],
        sampling_rate=16000,
        return_tensors="pt"
    )
    print(f"[infer_with_audio] Input tensors prepared", flush=True)

    # Keep token ids as integer tensors, but align floating tensors to model dtype
    # to avoid errors such as "expected scalar type BFloat16 but found Float".
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and hasattr(v, "to"):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            else:
                inputs[k] = v.to(device=target_device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[infer_with_audio] Input IDs length: {input_len}", flush=True)
    _mark_activity_start()
    try:
        if active_model is server_state.model and server_state.is_omni:
            # Omni any-to-any: text-only STT via generation_mode="text"
            print(f"[infer_with_audio] Omni STT generating (text mode)...", flush=True)
            out = _omni_generate_text(inputs, max_new_tokens=max_new_tokens)
        else:
            with torch.no_grad():
                _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
                if active_model is server_state.model and server_state.drafter is not None:
                    _gen_kwargs["assistant_model"] = server_state.drafter
                print(f"[infer_with_audio] Generating...", flush=True)
                out = active_model.generate(**inputs, **_gen_kwargs)
            print(f"[infer_with_audio] Generation complete, output shape: {out.shape}", flush=True)
    finally:
        _mark_activity_end()
    result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    print(f"[infer_with_audio] Result: {repr(result[:100])}", flush=True)
    return {"result": result}

def _handle_infer_local(params: dict) -> dict:
    """Local-slot text inference for Think-at-Rest.

    Runs plain-text chat inference against a named local model slot (default
    "audio" = Gemma 4 E2B-it), independent of cloud provider routing. This is
    what lets idle thoughts run on a resident/lazy-loadable local model even
    when `task_inference` is a cloud provider (e.g. `hf`).

    The slot is lazy-loaded on first use via _ensure_multimodal_slot() (same
    pattern as vision/STT). Do NOT call _ensure_model() here — that would load
    the cloud-only primary text model, which is unnecessary and can fail when
    the main model path is a cloud-only config.
    """
    slot = params.get("slot", "audio")
    messages = params.get("messages", [])
    max_new_tokens = params.get("max_new_tokens", 8192)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return {"error": f"'messages' must be a list, got {type(messages).__name__}"}

    # ── Resolve the slot model/processor ─────────────────────────────────────
    active_model = None
    active_processor = None

    # Prefer a registered named slot (e.g. "audio" wired into the SlotRegistry).
    if server_state.slot_registry is not None:
        try:
            _state = server_state.slot_registry.get(slot)
            if _state is not None:
                active_model = _state.model
                active_processor = _state.processor
                print(f"[model_server] infer_local: using named slot {slot!r}", flush=True)
        except Exception as _e:
            print(f"[model_server] infer_local: slot {slot!r} lookup failed ({_e})", flush=True)

    # Fallback: lazy-load the multimodal slot (Gemma 4 E2B-it).
    if active_model is None or active_processor is None:
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_local: multimodal slot load failed ({e})", flush=True)
        if server_state.mm_model is not None and server_state.mm_processor is not None:
            active_model = active_model or server_state.mm_model
            active_processor = active_processor or server_state.mm_processor
            print("[model_server] infer_local: using multimodal slot (Gemma 4)", flush=True)

    if active_model is None or active_processor is None:
        return {"error": "no local slot available (multimodal slot failed to load)"}

    # ── Plain-text chat inference against the slot ───────────────────────────
    import torch
    try:
        text = active_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception as e:
        return {"error": f"apply_chat_template failed: {e}"}

    target_device = next(active_model.parameters()).device
    target_dtype = next(active_model.parameters()).dtype
    inputs = active_processor(text=text, return_tensors="pt").to(target_device)
    # Align floating tensors to model dtype to avoid BFloat16/Float mismatch.
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and hasattr(v, "to"):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            else:
                inputs[k] = v.to(device=target_device)
    input_len = inputs["input_ids"].shape[-1]

    _mark_activity_start()
    try:
        with torch.no_grad():
            _gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=64,
            )
            out = active_model.generate(**inputs, **_gen_kwargs)
    finally:
        _mark_activity_end()
    result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return {"result": result}

def _cloud_multimodal_infer(modality: str, provider: str, model: str,
                             audio_path: str = "", image_path: str = "",
                             prompt: str = "", max_new_tokens: int = 1024) -> dict:
    """Route audio or vision inference through a cloud API (OpenRouter / OpenAI).
    Returns {"result": text} on success or {"error": msg} on failure.
    """
    import base64, json, urllib.request, urllib.error

    # Resolve API key and base URL
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("TMP_OPEN_AI_API_KEY", "")
        base_url = "https://api.openai.com/v1/chat/completions"
    else:
        return {"error": f"Cloud provider '{provider}' not supported for {modality}"}

    if not api_key:
        return {"error": f"No API key for {provider}"}

    # Build content parts based on modality
    content_parts = []

    if modality == "vision" and image_path:
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            import mimetypes
            mime, _ = mimetypes.guess_type(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime or 'image/jpeg'};base64,{img_b64}"}
            })
        except Exception as e:
            return {"error": f"Failed to read image: {e}"}

    elif modality == "audio" and audio_path:
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
            mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                        "webm": "audio/webm", "m4a": "audio/mp4", "flac": "audio/flac"}
            mime = mime_map.get(ext, "audio/wav")
            # OpenAI-compatible audio content format
            content_parts.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": ext if ext in ("wav", "mp3", "flac") else "wav"}
            })
        except Exception as e:
            return {"error": f"Failed to read audio: {e}"}

    else:
        return {"error": f"No content provided for {modality}"}

    content_parts.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": max_new_tokens,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8779"
        headers["X-Title"] = "kernel-evolving"

    try:
        req = urllib.request.Request(base_url, data=payload, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        result = data["choices"][0]["message"]["content"]
        print(f"[model_server] cloud_{modality}: {repr(result[:100])}", flush=True)
        return {"result": result}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        print(f"[model_server] cloud_{modality} HTTP {e.code}: {body}", flush=True)
        return {"error": f"Cloud {modality} failed: HTTP {e.code}"}
    except Exception as e:
        print(f"[model_server] cloud_{modality} error: {e}", flush=True)
        return {"error": f"Cloud {modality} failed: {e}"}

