"""
model_server.py — Long-lived model server process (thin entrypoint).

Loads Gemma 4 once, stays alive, owns GPU VRAM.
Exposes JSON-RPC over Unix socket /tmp/kernel_model.sock.
API and Telegram bot connect to it and restart freely.

Backend: vLLM AsyncLLMEngine (primary) with HF transformers fallback.
         vLLM 0.21.0+ supports Gemma4ForConditionalGeneration.
         Multimodal slot (Gemma 4 E2B-it) handles STT, vision, and combined inference via HF transformers
         (vLLM multimodal audio API differs from vision — kept on HF for stability).

Refactor: the request-handler/inference dispatch layer now lives in handlers.py;
shared inference helpers in handlers_common.py; config/activity/capabilities/state
in their leaf modules. This file owns the socket accept loop, the HANDLERS
dispatch table, and main(), and re-exports every moved symbol so existing
call-sites keep working unchanged.

Usage:
    python3 src/model_server.py [--config config.yaml]
"""
from . import state as server_state

import json
import os
import signal
import socket
import socketserver
import sys
import threading
import traceback
from pathlib import Path  # noqa: F401

# Allow running from src/ or repo root
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from runtime_paths import MODEL_ACTIVITY_FILE as _MODEL_ACTIVITY_PATH_IMPORT  # noqa: E402,F401
from core.hf_cache import ensure_hf_home_env  # noqa: E402

# MS4: pin HF_HOME early so huggingface_hub's own cache resolution (used when
# from_pretrained()/AsyncEngineArgs() are given a bare repo_id) can't silently
# disagree with what telegram_bot.py's /models download check reported.
ensure_hf_home_env()

# Activity tracking for Think-at-Rest — mirrors model_client.py counters
# so thought_engine doesn't fire during image/audio inference.
# Resolved from runtime_paths canonical location, overridable via env.
_ACTIVITY_PATH = str(_MODEL_ACTIVITY_PATH_IMPORT)  # noqa: F401

SOCKET_PATH = "/tmp/kernel_evolving_model.sock"


# ---------------------------------------------------------------------------
# Re-exports — every moved symbol is re-exported here so existing call-sites
# (model_server._handle_*, model_server._build_chat_prompt, etc.) keep working.
# ---------------------------------------------------------------------------

# Request handler/inference dispatch layer (moved to handlers.py).
from . import handlers as _handlers  # noqa: E402
from .handlers import *  # noqa: E402,F401,F403
from .handlers import (  # noqa: E402,F401
    _handle_infer,
    _handle_infer_plain,
    _run_two_stage_if_available,
    _nemotron_synthesize_answer,
    _handle_infer_with_tools,
    _handle_infer_with_image,
    _handle_infer_with_audio,
    _handle_infer_local,
    _cloud_multimodal_infer,
    _handle_vram_free_mb,
    _resolve_loaded_model_name,
    _handle_health,
    _handle_load_slot,
    _handle_unload_slot,
    _handle_slot_status,
    _handle_unload,
    _handle_swap_model,
    _ensure_drafter_only,
    _handle_infer_draft,
)

# Shared inference helpers (moved to handlers_common.py).
from .handlers_common import (  # noqa: E402,F401
    _build_chat_prompt,
    _get_vllm_processor,
    _omni_generate_text,
    _inference_cfg,
    _critique_cfg,
    _sync_globals_from_slot,
    _DEBUG_TWO_STAGE,
)

# Config helpers (leaf) — _load_config keeps the lazy-load side effect used by
# the old _critique_cfg wrapper and by main().
from .handlers_common import _load_config  # noqa: E402,F401

# Activity tracking + log preview (leaf).
from .activity import (  # noqa: E402,F401
    mark_activity_start as _mark_activity_start,
    mark_activity_end as _mark_activity_end,
    preview_payload_for_log as _preview_payload_for_log,
)

# Stateless parsing helpers (leaf) — re-export so existing
# `model_server._parse_function_eq_xml` etc. references stay resolvable.
from .parsing import (  # noqa: E402,F401
    _sanitize_text as _sanitize_text,
    _normalize_tool_args as _normalize_tool_args,
    _parse_function_eq_xml as _parse_function_eq_xml,
    _parse_tag_fallback as _parse_tag_fallback,
)

# Failure re-prompt helpers (leaf).
from .prompts import (  # noqa: E402,F401
    build_failure_reprompt as _build_failure_reprompt,
    build_schema_repair_hint as _build_schema_repair_hint,
)

# Stateless model-name/adapter predicates (leaf).
from .model_names import (  # noqa: E402,F401
    is_nemotron_model as _is_nemotron_model,
    is_janus_model as _is_janus_model,
    adapter_name as _adapter_name,
    is_peft_model as _is_peft_model,
    friendly_model_name as _friendly_model_name,
)

# Tool-loop validation (leaf).
from .validation import (  # noqa: E402,F401
    _VALIDATION_INFO_TOOLS as _VALIDATION_INFO_TOOLS,
    _VALIDATION_QUERY_MARKERS as _VALIDATION_QUERY_MARKERS,
    _VALIDATION_ERROR_TOKENS as _VALIDATION_ERROR_TOKENS,
    tool_result_answers_query as _tool_result_answers_query,
)

# vLLM backend subsystem (leaf).
from . import vllm as _vllm  # noqa: E402,F401

# Capability detection (leaf).
from .capabilities import detect as _detect_capability_flags  # noqa: E402,F401
from .handlers import _detect_capabilities  # noqa: E402,F401

# HF loaders, LoRA adapters, slot lifecycle (leaf modules).
from .loaders import (  # noqa: E402,F401
    _load_hf_model as _load_hf_model,
    _load_nemotron as _load_nemotron,
    _load_janus as _load_janus,
    _janus_infer_text as _janus_infer_text,
    _janus_infer_image as _janus_infer_image,
    _nemotron_infer as _nemotron_infer,
)
from .adapters import (  # noqa: E402,F401
    _load_adapter as _load_adapter,
    _use_adapter as _use_adapter,
)
from .slots import (  # noqa: E402,F401
    _make_slot_loader as _make_slot_loader,
    _load_model as _load_model,
    _ensure_model as _ensure_model,
    _ensure_multimodal_slot as _ensure_multimodal_slot,
    _ensure_tool_calling_slot as _ensure_tool_calling_slot,
)

# Slot registry availability / target-device helper (leaf).
_SLOTS_AVAILABLE = getattr(_handlers, "_SLOTS_AVAILABLE", False)
from .state import target_device as _target_device  # noqa: E402,F401


# ---------------------------------------------------------------------------
# RPC dispatch — method name -> handler. Each handler takes (params, ...) and
# returns a JSON-serialisable dict. Some handlers need the streaming `send_line`
# callback (two-stage / tool loop); wired here so the socket layer stays dumb.
# ---------------------------------------------------------------------------
HANDLERS = {
    "infer": lambda params, send_line: _handle_infer(params),
    "infer_draft": lambda params, send_line: _handle_infer_draft(params),
    "infer_local": lambda params, send_line: _handle_infer_local(params),
    "infer_with_tools": lambda params, send_line: _handle_infer_with_tools(params, send_line),
    "infer_with_image": lambda params, send_line: _handle_infer_with_image(params),
    "infer_with_audio": lambda params, send_line: _handle_infer_with_audio(params),
    "vram_free_mb": lambda params, send_line: _handle_vram_free_mb(params),
    "health": lambda params, send_line: _handle_health(params),
    "load_slot": lambda params, send_line: _handle_load_slot(params),
    "unload_slot": lambda params, send_line: _handle_unload_slot(params),
    "slot_status": lambda params, send_line: _handle_slot_status(params),
    "unload": lambda params, send_line: _handle_unload(params),
    "swap_model": lambda params, send_line: _handle_swap_model(params),
}


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8").strip())
            method = request.get("method", "")
            params = request.get("params", {})

            if isinstance(method, str) and method.startswith("infer"):
                print(
                    f"[model_server] request method={method} params=\n{_preview_payload_for_log(params)}",
                    flush=True,
                )

            def send_line(data: str):
                if isinstance(method, str) and method.startswith("infer"):
                    try:
                        payload = json.loads(data)
                    except Exception:
                        payload = data
                    print(
                        f"[model_server] response method={method} payload=\n{_preview_payload_for_log(payload)}",
                        flush=True,
                    )
                # Harden against a client that already disconnected (e.g. its socket
                # read timed out and closed). A broken pipe mid-loop used to crash
                # the whole handler; now we just stop writing to this dead peer.
                try:
                    self.wfile.write((data + "\n").encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    print(
                        "[model_server] client disconnected — aborting response stream",
                        flush=True,
                    )
                    raise

            handler = HANDLERS.get(method)
            if handler is not None:
                resp = handler(params, send_line)
                send_line(json.dumps(resp))
            else:
                send_line(json.dumps({"error": f"unknown method: {method}"}))

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[model_server] handler error: {exc}\n{tb}", flush=True)
            try:
                self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kernel Evolving model server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--socket", default=None, help="Override socket path")
    parser.add_argument("--lazy", action="store_true", help="Defer model load to first inference request")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter to load after base model")
    parser.add_argument("--model", default=None, help="Override base model path/name, bypassing config.yaml's model.path (for eval: load the adapter's own base model)")
    args = parser.parse_args()

    # Timestamped startup banner — marks each model-server (re)start attempt in the
    # log so we can tell which run produced which lines (log is shared across restarts).
    from datetime import datetime as _dt
    _adapter_banner = args.adapter if args.adapter else os.environ.get("MODEL_ADAPTER_PATH", "(config)")
    print(
        f"[model_server] ===== START {_dt.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"socket={args.socket or 'default'} adapter={_adapter_banner} =====",
        flush=True,
    )

    socket_path = args.socket or SOCKET_PATH

    cfg = _load_config(args.config)
    if cfg.get("model_server", {}).get("socket"):
        socket_path = cfg["model_server"]["socket"]
    if args.socket:
        socket_path = args.socket

    if os.path.exists(socket_path):
        try:
            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_sock.settimeout(2)
            test_sock.connect(socket_path)
            test_sock.close()
            print(f"[model_server] Socket {socket_path} is already live — another instance is running. Exiting.", flush=True)
            sys.exit(0)
        except (ConnectionRefusedError, OSError):
            print(f"[model_server] Removing stale socket {socket_path}", flush=True)
            os.unlink(socket_path)

    server_state.lazy_config_path = os.path.abspath(args.config)

    server_state.lazy_model_override = args.model

    if args.adapter:
        import os as _os
        _os.environ["MODEL_ADAPTER_PATH"] = args.adapter
        print(f"[model_server] Adapter mode: will load LoRA adapter from {args.adapter}", flush=True)

    if args.lazy:
        print(f"[model_server] Lazy mode: model will load on first inference request.", flush=True)
    else:
        _load_model(args.config, model_path_override=args.model)
        if args.adapter:
            _load_adapter(args.adapter)

    server = None

    def _shutdown(signum, frame):
        print(f"\n[model_server] Caught signal {signum}, shutting down...", flush=True)
        # Shut down vLLM event loop if running
        _vllm.shutdown()
        if server:
            threading.Thread(target=server.shutdown, daemon=True).start()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server = _ThreadingUnixServer(socket_path, _RequestHandler)
    os.chmod(socket_path, 0o600)

    print(f"[model_server] Ready on {socket_path}", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
