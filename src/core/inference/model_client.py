"""
                   er should check is_server_running() first).
"""
import json
import os
import socket
import time
from typing import Callable, List, Optional
from runtime_paths import MODEL_ACTIVITY_FILE
from core.tools import WORKSPACE as _DEFAULT_WORKSPACE

SOCKET_PATH = "/tmp/kernel_evolving_model.sock"
REQUEST_TIMEOUT = 300  # seconds — model inference can be slow
_ACTIVITY_PATH = str(MODEL_ACTIVITY_FILE)


def _read_activity() -> dict:
    try:
        with open(_ACTIVITY_PATH) as f:
            return json.load(f)
    except Exception:
        return {"in_flight": 0, "last_start_ts": 0.0, "last_end_ts": 0.0}


def _write_activity(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_ACTIVITY_PATH), exist_ok=True)
        with open(_ACTIVITY_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _mark_activity_start() -> None:
    data = _read_activity()
    data["in_flight"] = int(data.get("in_flight", 0)) + 1
    data["last_start_ts"] = time.time()
    _write_activity(data)


def _mark_activity_end() -> None:
    data = _read_activity()
    data["in_flight"] = max(0, int(data.get("in_flight", 0)) - 1)
    data["last_end_ts"] = time.time()
    _write_activity(data)


def _get_socket_path() -> str:
    """Return socket path: env MODEL_SERVER_SOCKET > config.yaml > default."""
    env_sock = os.environ.get("MODEL_SERVER_SOCKET")
    if env_sock:
        return env_sock
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model_server", {}).get("socket", SOCKET_PATH)
    except Exception:
        return SOCKET_PATH


def is_server_running() -> bool:
    """Return True if the model server socket exists and accepts connections."""
    sock_path = _get_socket_path()
    if not os.path.exists(sock_path):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(sock_path)
        s.close()
        return True
    except Exception:
        return False


def _call(request: dict, timeout: float = REQUEST_TIMEOUT) -> List[dict]:
    """
    Send a JSON-RPC request to the model server.
    Returns a list of parsed response lines (for streaming methods like infer_with_tools).
    Raises on connection failure or timeout.
    """
    sock_path = _get_socket_path()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    _mark_activity_start()
    try:
        s.connect(sock_path)

        payload = (json.dumps(request) + "\n").encode("utf-8")
        s.sendall(payload)

        # Read response lines until connection closes
        buf = b""
        lines = []
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line.decode("utf-8")))
        except socket.timeout:
            raise TimeoutError(f"model_server request timed out after {timeout}s")
        return lines
    finally:
        try:
            s.close()
        except Exception:
            pass
        _mark_activity_end()


# ---------------------------------------------------------------------------
# Public API — mirrors model.py
# ---------------------------------------------------------------------------

def infer_draft(prompt: str, max_new_tokens: int = 8192) -> str:
    """ADR-010: Fast draft inference using MTP drafter only (no main model pass).
    Falls back to full infer() if drafter not available on server."""
    try:
        resp_lines = _call({
            "method": "infer_draft",
            "params": {"prompt": prompt, "max_new_tokens": max_new_tokens},
        })
        if resp_lines:
            resp = resp_lines[0]
            if "error" in resp:
                return f"[model_server error] {resp['error']}"
            return resp.get("result", "")
        return ""
    except TimeoutError as e:
        return f"[model_server timeout] {e}"
    except Exception as e:
        return f"[model_client error] {e}"


def infer(messages: list, max_new_tokens: int = 8192, adapter_path: Optional[str] = None,
          skip_identity_guard: bool = False) -> str:
    """Run chat inference via model server. Returns response string.

    Routes to _handle_infer server-side (T1) — a plain, non-tool inference call.
    skip_identity_guard: set True for callers sending a fully self-contained
    utility prompt (e.g. a fact extractor with its own "You are X..." role) to
    avoid the server injecting a conflicting default Kernel-Evo identity system
    message. See .specs/fixes/IDENTITY-INJECTION-UTILITY-PROMPT-CONFLICT-2026-08-06.md
    """
    try:
        params = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "tools": [],  # explicitly empty so server doesn't short-circuit
        }
        if adapter_path:
            params["adapter_path"] = adapter_path
        if skip_identity_guard:
            params["skip_identity_guard"] = True
        resp_lines = _call({
            "method": "infer",
            "params": params,
        })
        if resp_lines:
            resp = resp_lines[0]
            if "error" in resp:
                return f"[model_server error] {resp['error']}"
            return resp.get("result", "")
        return ""
    except TimeoutError as e:
        return f"[model_server timeout] {e}"
    except Exception as e:
        return f"[model_client error] {e}"



def infer_with_tools(
    messages: list,
    tools: list,
    workspace: str = _DEFAULT_WORKSPACE,
    max_steps: int = 60,
    step_callback: Optional[Callable] = None,
    chunk_callback: Optional[Callable[[str], None]] = None,
    adapter_path: Optional[str] = None,
    chat_id: str = "",
    ) -> str:
    """
    Agentic tool-calling loop via model server.
    Server streams step lines then final result.
    step_callback(step_num, tool_name, tool_args, result) called per step.
    chat_id: propagated to the model_server process so exec_shell's Telegram
    authorization gate can fire there (R5/T4) — model_server runs in a
    separate process and can't see core.tools._current_chat_id.
    """
    try:
        params = {
            "messages": messages,
            "tools": tools,
            "workspace": workspace,
            "max_steps": max_steps,
            "chat_id": chat_id,
        }
        if adapter_path:
            params["adapter_path"] = adapter_path
        resp_lines = _call({
            "method": "infer_with_tools",
            "params": params,
        })
        final_result = ""
        for line in resp_lines:
            if line.get("type") == "step":
                result_text = line.get("result", "")
                if line.get("ok") is False:
                    status = line.get("status", "failed")
                    reason = line.get("failure_reason", "tool reported a failure")
                    result_text = f"{result_text}\n[status={status}] {reason}"
                if step_callback:
                    step_callback(
                        line.get("step", 0),
                        line.get("tool", ""),
                        line.get("args", {}),
                        result_text,
                    )
            elif line.get("type") == "chunk":
                if chunk_callback:
                    chunk_callback(str(line.get("text", "")))
            elif line.get("type") == "result":
                final_result = line.get("result", "")
                # Local model_server currently emits final text at once for the final answer.
                # Emit synthetic chunks so Telegram/UI still streams progressively.
                if chunk_callback and final_result:
                    _stream_text = str(final_result).replace("</think>", "").replace("<think>", "").strip()
                    for i in range(0, len(_stream_text), 40):
                        chunk_callback(_stream_text[i:i+40])
            elif "error" in line:
                return f"[model_server error] {line['error']}"
            else:
                # Fallback: plain result dict
                final_result = line.get("result", final_result)
        return final_result
    except TimeoutError as e:
        return f"[model_server timeout] {e}"
    except Exception as e:
        return f"[model_client error] {e}"


def infer_with_image(image_path: str, prompt: str, max_new_tokens: int = 8192,
                     force_local: bool = False) -> str:
    """Multimodal image inference via model server.

    `force_local`: when True, the model server skips cloud vision routing
    (providers.vision) and uses the local Gemma E2B vision slot. Used by the
    native `look`/describe flow so the eyes always describe with the onboard
    model regardless of the cloud-vision config (which is for Telegram images).
    """
    try:
        resp_lines = _call({
            "method": "infer_with_image",
            "params": {
                "image_path": image_path,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "force_local": force_local,
            },
        })
        if resp_lines:
            resp = resp_lines[0]
            if "error" in resp:
                return f"[model_server error] {resp['error']}"
            return resp.get("result", "")
        return ""
    except TimeoutError as e:
        return f"[model_server timeout] {e}"
    except Exception as e:
        return f"[model_client error] {e}"


def infer_with_audio(
    audio_path: str,
    prompt: str = "Transcribe this audio.",
    max_new_tokens: int = 8192,
    history: Optional[List[dict]] = None,
    mode: str = "stt",
    ) -> str:
    """Multimodal audio inference via model server."""
    try:
        resp_lines = _call({
            "method": "infer_with_audio",
            "params": {
                "audio_path": audio_path,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "history": history or [],
                "mode": mode,
            },
        })
        if resp_lines:
            resp = resp_lines[0]
            if "error" in resp:
                return f"[model_server error] {resp['error']}"
            return resp.get("result", "")
        return ""
    except TimeoutError as e:
        return f"[model_server timeout] {e}"
    except Exception as e:
        return f"[model_client error] {e}"


def vram_free_mb() -> int:
    """Return free VRAM in MB via model server."""
    try:
        resp_lines = _call({
            "method": "vram_free_mb",
            "params": {},
        })
        if resp_lines:
            resp = resp_lines[0]
            if "error" in resp:
                return 0
            return int(resp.get("vram_free_mb", 0))
        return 0
    except Exception:
        return 0


def health() -> dict:
    """Return health dict from model server."""
    try:
        resp_lines = _call({
            "method": "health",
            "params": {},
        })
        return resp_lines[0] if resp_lines else {}
    except Exception as e:
        return {"error": str(e)}


def swap_model(model_path: str, drafter_path: str = None, dtype: str = "bfloat16") -> dict:
    """Hot-swap the loaded model. drafter_path=None keeps current, drafter_path='' disables."""
    try:
        params = {"model_path": model_path, "dtype": dtype}
        if drafter_path is not None:
            params["drafter_path"] = drafter_path
        resp_lines = _call({"method": "swap_model", "params": params}, timeout=300)
        return resp_lines[0] if resp_lines else {"error": "no response"}
    except Exception as e:
        return {"error": str(e)}


def unload() -> dict:
    """Unload the current model from GPU VRAM without reloading.
    Call this before switching task_inference to a cloud provider to free VRAM.
    """
    try:
        resp_lines = _call({"method": "unload", "params": {}}, timeout=60)
        return resp_lines[0] if resp_lines else {"error": "no response"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Named slot management
# ---------------------------------------------------------------------------

def load_slot(name: str) -> dict:
    """Load a named model slot. Returns {"ok": True, "slot": name, "loaded_at": float}."""
    try:
        resp_lines = _call({"method": "load_slot", "params": {"slot": name}}, timeout=300)
        return resp_lines[0] if resp_lines else {"error": "no response"}
    except Exception as e:
        return {"error": str(e)}


def unload_slot(name: str) -> dict:
    """Unload a named model slot and free its VRAM."""
    try:
        resp_lines = _call({"method": "unload_slot", "params": {"slot": name}}, timeout=60)
        return resp_lines[0] if resp_lines else {"error": "no response"}
    except Exception as e:
        return {"error": str(e)}


def slot_status() -> list:
    """Return JSON-serialisable status of all registered model slots."""
    try:
        resp_lines = _call({"method": "slot_status", "params": {}})
        result = resp_lines[0] if resp_lines else {}
        return result.get("slots", [])
    except Exception as e:
        return [{"error": str(e)}]
