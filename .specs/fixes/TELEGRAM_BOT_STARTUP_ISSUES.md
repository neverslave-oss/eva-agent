# Telegram Bot & start.sh Startup Issues - Root Cause Analysis

## Overview

The Telegram bot breaks and services fail to start when `start.sh` is executed directly. This is due to **circular imports, initialization order problems, and model loading race conditions** introduced in commits `0b99da2` (FastAPI) and `cea36cc` (systemd).

---

## Issue #1: Circular Imports Block Module Load

### Problem Chain

**api.py** (line 9):
```
import yaml, agent, replica as rep, model as mdl
```

This imports `agent` at module load time, which triggers:

**agent.py** (lines 6-11) — immediate imports:
```
from model import infer, infer_with_tools, vram_free_mb
from skills import load_all as load_skills, find as find_skill, ...
from routines import load_all as load_routines, ...
from embedding_client import EmbeddingClient
```

These cascade:
- `model.py` tries to detect model_server, load config.yaml, check CUDA
- `skills.py` accesses filesystem (skills_dir may not exist)
- `embedding_client.py` initializes backend

### Impact

If ANY of these fail, the entire api.py module fails to import, and uvicorn crashes before startup() event even fires.

**Result**: When `./start.sh` runs, `python3 -m uvicorn api:app` fails immediately with an import error that the user never sees (buried in `/tmp/kernel_evolving_api.log`).

---

## Issue #2: Model Server vs. In-Process Load Race

### Problem

**api.py startup()** sequence:
```python
mdl.load(_config_path)           # Load in-process
agent.init(_config_path)         # Init agent
import telegram_bot
telegram_bot.start_bot_thread()  # Expects model ready
```

**Timing Issue**:
1. start.sh starts model_server.py and API in parallel
2. model_server takes 20-30s to load transformers/torch
3. But start.sh only waits 15s for socket (line ~40)
4. If socket isn't ready, API proceeds anyway
5. API tries mdl.load(), which checks if server is running
6. Server might be partially loaded or not ready
7. Falls back to in-process loading
8. Telegram bot thread tries to infer() but model state is inconsistent

**Result**: Bot crashes or hangs when trying to infer.

---

## Issue #3: Telegram Bot Import Fails or Crashes

### Problem

**telegram_bot.py** in `start_bot_thread()`:
```python
import model as _m
from model_client import is_server_running
# Later in polling loop:
from model import infer
```

If model_server socket isn't ready:
- model.py `_model` global is `None`
- Bot thread tries to call `infer()` with `_model=None`
- AttributeError or silent crash

If model_server is running but bot thread tries to use it from different context:
- CUDA device mismatch
- RuntimeError: CUDA out of memory (model locked on main thread)
- Bot hangs forever

### Impact

Bot is silently broken. No error in logs. Just stops responding to Telegram messages.

---

## Issue #4: Insufficient Wait Time in start.sh

**start.sh** lines 35-47:
```bash
# Wait for socket (up to 15s)
for i in $(seq 1 15); do
    [[ -S "$SOCKET" ]] && break
    sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
    echo "[kernel-evolving] WARNING: model server socket not ready — continuing anyway"
fi
```

**Problem**: 15 seconds is not enough for:
- PyTorch to import and initialize CUDA
- transformers to download model if first run
- Model to load to GPU

start.sh continues anyway and API falls back to in-process loading, causing conflicts.

---

## Issue #5: No Error Surface from API Load Failures

**start.sh** lines 58-62:
```bash
if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "[kernel-evolving] ✓ Ready on :$PORT"
else
    echo "[kernel-evolving] ✗O Health check failed — check $API_LOG"
    exit 1
fi
```

**Problem**: If API module fails to import, the curl fails silently. User only sees:
```
[kernel-evolving] ✗O Health check failed — check /tmp/kernel_evolving_api.log
```

But they have to manually inspect that log file. No streaming output of the actual error.

---

## The Failure Sequence

```
./start.sh
├─ Runs: python3 model_server.py (async, takes 20-30s)
├─ Waits 15s for /tmp/kernel_evolving_model.sock (TOO SHORT)
│  └─ Socket not ready yet
├─ Runs: python3 -m uvicorn api:app
│  ├─ Python loads api.py
│  ├─ api.py line 9: import agent
│  │  └─ agent.py line 6: from model import infer
│  │     └─ model.py tries is_server_running() — returns False (not ready yet)
│  │        └─ Attempts in-process load (conflicts with model_server still starting)
│  │           └─ May fail or succeed with wrong state
│  ├─ If import succeeds: uvicorn starts
│  ├─ Startup event fires: mdl.load(), agent.init(), telegram_bot.start_bot_thread()
│  │  └─ Telegram bot thread starts polling
│  │     └─ Calls model.infer() on first incoming message
│  │        └─ model.infer() checks is_server_running() — NOW returns True (finally ready)
│  │           └─ But sends request to server that's already handling API load
│  │              └─ Or server state is corrupted from prior in-process attempt
│  │                 └─ Bot crashes or returns garbage
│  │
│  └─ If import fails: uvicorn dies immediately
│     └─ Health check times out/fails
│        └─ start.sh reports generic error, user has to check log
└─ Result: Both API and Telegram bot non-functional
```

---

## Git History: What Broke It

| Commit | Change | Impact |
|--------|--------|--------|
| `0b99da2` | Added FastAPI api.py with eager agent import | Circular imports start happening |
| `cea36cc` | Added systemd service file | Didn't address startup ordering |
| `2cdcd74-...` | Added STT/voice models with complex imports | Made circular imports worse |
| Recent | Various model loading improvements | Didn't fix root cause |

**Root Cause**: Commit `0b99da2` refactored from simple script to FastAPI + bot integration, but didn't defer module imports or properly sequence startup.

---

## Why Docker Compose Works

Docker Compose orchestrates services:
1. Starts model_server container first
2. Uses `depends_on: model_server` and health checks
3. Ensures model_server socket is ready before starting API
4. API starts with model_server already operational
5. Circular imports happen but model is pre-loaded
6. Telegram bot inherits ready model state

---

## Solutions

### Solution 1: Defer agent import in api.py
Change line 9 from eager to lazy:
```python
# Instead of:
import yaml, agent, replica as rep, model as mdl

# Use:
import yaml, asyncio, os, json, time, uuid
# Import inside startup():
async def startup():
    global agent, rep, mdl
    import agent
    import replica as rep
    import model as mdl
    # ... rest of startup
```

### Solution 2: Increase model server wait time to 40-60s
```bash
for i in $(seq 1 60); do
    [[ -S "$SOCKET" ]] && break
    sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
    echo "[kernel-evolving] ERROR: Model server socket not ready after 60s"
    tail -20 "$MODEL_LOG"
    exit 1
fi
```

### Solution 3: Guard model imports in agent.py
```python
# Instead of importing at module level:
def _ensure_model_loaded():
    global _model_infer_func
    if not hasattr(_ensure_model_loaded, '_func'):
        from model import infer as _f
        _ensure_model_loaded._func = _f
    return _ensure_model_loaded._func
```

### Solution 4: Add error streaming to start.sh
```bash
# Capture API startup output in real-time
tail -f "$API_LOG" &
TAIL_PID=$!
sleep 4
if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    kill $TAIL_PID
    echo "[kernel-evolving] ✓ Ready on :$PORT"
else
    echo "[kernel-evolving] ERROR: API failed to start"
    kill $TAIL_PID
    echo "---- API Log ----"
    cat "$API_LOG"
    exit 1
fi
```

### Solution 5: Detect and fail early if model_server dies
```bash
MODEL_PID=$!
# Check if process is still running after socket check
if ! kill -0 $MODEL_PID 2>/dev/null; then
    echo "[kernel-evolving] ERROR: Model server process died"
    tail -30 "$MODEL_LOG"
    exit 1
fi
```

---

## Verification

**Current Broken Behavior**:
```bash
$ ./start.sh
[kernel-evolving] Stopping API on port 8779 (if running)...
[kernel-evolving] env loaded from .env
[kernel-evolving] Version: v1.14.0
[kernel-evolving] Starting model server (lazy)...
[kernel-evolving] Model server PID 12345 — log: /tmp/kernel_evolving_model_server.log
[kernel-evolving] ✗O Health check failed — check /tmp/kernel_evolving_api.log
```

**User has to do**:
```bash
$ tail /tmp/kernel_evolving_api.log
# See import error or timeout
```

**Expected Behavior** (after fixes):
```bash
$ ./start.sh
[kernel-evolving] Stopping API on port 8779 (if running)...
[kernel-evolving] env loaded from .env
[kernel-evolving] Version: v1.14.0
[kernel-evolving] Starting model server (lazy)...
[kernel-evolving] Model server PID 12345 — log: /tmp/kernel_evolving_model_server.log
[kernel-evolving] Waiting for model server socket (up to 60s)...
[kernel-evolving] Model server ready after 28s
[kernel-evolving] Starting API...
[api] Loading model...
[api] Agent ready with 5 skills, 3 routines
[bot] KERNEL_EVO_TELEGRAM_BOT_TOKEN set — Telegram bot enabled
[api] Kernel ready on :8779
[kernel-evolving] ✓ Ready on :8779
```

---

## Summary

**The core problem**: Commit `0b99da2` introduced FastAPI with eager module imports that block startup. This works in Docker (orchestrated) but breaks in `start.sh` (parallel unordered startup).

**The fix**: Defer all non-essential imports to api.py startup() event, increase model server wait timeout, and add error surface.