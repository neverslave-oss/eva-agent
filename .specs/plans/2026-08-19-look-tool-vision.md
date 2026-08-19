# Plan: Unified `look` Tool — Vision for EVA

**Created:** 2026-08-19
**Status:** Phase 1 + Phase 2 (describe) implemented
**Branch:** feat/look-tool

> **Update (2026-08-19):** Phase 1 (registry/router/eyes/look tool) and the Phase 2
> semantic `describe` path are both implemented and committed (`4855543`). See
> **Implementation Status** below.

## Overview

Give EVA (kernel-evolving) a unified **`look`** tool that abstracts the physical
cameras ("eyes") on the LAN, so the agent perceives the world through multiple
cameras via a single semantic interface. Scales to N eyes; plug-and-play (eyes
register/unregister as they come online). All endpoints/IPs are **dynamic config**
set via the companion desktop app — never hardcoded — so it's portable across
installs.

This plan covers **Phase 1** (the `look` tool wired to the YOLO eyes). The
semantic "describe" path (Phase 2) and gated `identify` (Phase 3) are scoped as
follow-ups.

---

## Background / Assets

### Eye 1 — XIAO ESP32S3 Sense (seedsennse)
- Camera streams MJPEG on **port 80** (`/stream`, `/annotated`).
- Brain = `yolo_server.py` (Flask, **port 5010**): YOLOv8n.
  - `POST /detect` → JSON detections
  - `POST /detect_text` → plain text
  - `POST /detect_annotated` → JPEG with boxes
  - `POST /detect_collect` → auto-collects person crops (face-recognition pipeline)
  - `/collect/status`, `/collect/toggle`
- Repo: `fabiopacificicom/seedsennse` (private). Local clone: `/tmp/seedsennse`.

### Eye 2 — Pi webcam (pi-pico-smart-monitor)
- Pi's webcam streams MJPEG on **port 5000** (`/video_feed`).
- Brain = `prototype_leaf_detection.py`: **YOLOv5 Nano**.
  - `POST /plant_health/capture_and_detect` → leaf crops
  - `GET /crops/<filename>` → retrieve crop
- Repo: `fabiopacificicom/pi-pico-smart-monitor` (public).

### Inference brain (Phase 2)
- private-ai-server (`localhost:8005`, vLLM, **Ollama drop-in**) — mirrors the
  Ollama API exactly. Used for the semantic "describe" path later.

---

## Architecture

### Unified `look` tool, intent-based routing
```
look(
  intent: "what's there" | "who is it" | "plant health" | "scan",
  target?: optional eye id hint
)
```
- Agent specifies **intent (semantic)**, not hardware.
- Router maps intent → eye via the **eye registry**.

### Eye registry (config-driven, NOT hardcoded)
- All endpoints/IPs are config values set via the companion desktop app.
- EVA reads config at startup; on another machine the hub IPs come from that
  user's installation/settings.
- Registry holds references to config keys, not literal IPs.

### Routing
- `"what's there" | "who is it"` → eye_left (YOLOv8n)
- `"plant health"` → eye_right (YOLOv5n)
- `"scan"` → all eyes, merge results

### Unified response envelope
```
{ eye, intent, observations, raw, timestamp, status }
```
Same envelope, different payload → uniform parsing as eyes multiply.

---

## Implementation

### Files
- `src/core/vision/__init__.py`
- `src/core/vision/registry.py` — load config, health-check, status tracking
- `src/core/vision/router.py` — intent → eye(s) mapping + scan merge
- `src/core/vision/eyes/object_face.py` — calls yolo_server endpoints
- `src/core/vision/eyes/plant_health.py` — calls plant server endpoints
- `src/core/vision/capture.py` — grab a single frame from an eye's stream
- `src/core/tools.py` — register the `look` tool in the `TOOLS` list + dispatch in `execute_tool`
- `config/eyes.yaml` — config-driven registry (populated by desktop app settings)
- `tests/test_look_tool.py` — unit + integration tests (isolated, dedicated test DB)

### Tool registration (matches existing `TOOLS` list style)
```python
{
    "type": "function",
    "function": {
        "name": "look",
        "description": "Perceive the world through connected cameras (eyes). Acts by intent and opens one eye or all.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["what's there", "who is it", "plant health", "scan"],
                    "description": "What the agent wants to perceive."
                },
                "target": {
                    "type": "string",
                    "description": "Optional eye id hint (e.g. 'left', 'right'). Router falls back to intent mapping if omitted."
                }
            },
            "required": ["intent"]
        }
    }
}
```

### Endpoint mapping
| intent | eye | call | returns |
|---|---|---|---|
| what's there | left | `POST /detect` | JSON detections |
| who is it | left | `POST /detect_collect` (gated, Phase 3) | person crops / recognition |
| plant health | right | `POST /plant_health/capture_and_detect` | `{num_crops, crops:[...]}` |
| scan | all | run each eye, merge | combined |

### Availability / plug-and-play
- `registry.py` health-checks each eye's `health` endpoint before routing.
- Eye offline → return `status:"offline"` for that eye (or auto-fall to another).
- Router never routes to a dead eye.

---

## Success Criteria
- [x] `look` tool registered in `TOOLS` and dispatched in `execute_tool`
- [x] `look(intent="describe")` grabs a frame from an eye's stream and describes it (local Gemma E2B → Ollama :8005 fallback)
- [ ] `look(intent="plant health")` returns leaf crops from Eye 2 (verified live)
- [ ] `look(intent="what's there")` returns JSON detections from Eye 1 (when online)
- [x] `look(intent="scan")` merges results from all online eyes
- [x] Eye offline → graceful `status:"offline"`, no crash
- [x] All endpoints/IPs come from config, none hardcoded
- [x] Tests green (isolated, dedicated test DB)

## Implementation Status (2026-08-19)

### Done
- `src/core/vision/registry.py` — config-driven eye registry, health-check, plug-and-play status
- `src/core/vision/router.py` — intent → eye(s) mapping, scan merge, offline degradation; **`describe` routes to first online eye**
- `src/core/vision/eyes/object_face.py` + `plant_health.py` — eye client calls
- `src/core/vision/capture.py` — `grab_frame()` extracts last complete JPEG from MJPEG stream (SOI/EOI delimiters)
- `src/core/vision/describe.py` — `describe_image()` semantic scene description: local Gemma E2B native first, Ollama-compatible `:8005` fallback, config-driven base/model
- `src/core/tools.py` — `look` registered in `TOOLS` + dispatched in `execute_tool`; `describe` added to intent enum
- `tests/test_look_tool.py` — registry/router/eye clients/dispatch + `TestDescribeRouting` (6 cases)
- `tests/test_describe.py` — 14 tests (describe_image backends, local Gemma, Ollama, grab_frame)

### Verified
- New describe/look tests: **42 passed**
- Full suite: **649 passed / 19 failed** — the 19 failures are **pre-existing** (identical on base with these changes stashed) and stem from provider routing returning `hf` instead of `local` for `task_inference` (intentional default; stale tests). Not caused by this work.

### Remaining (Phase 3 / live verification)
- [ ] Live verify `plant health` / `what's there` / `describe` against the wired eye
- [ ] Phase 3: `who is it` (gated) + self-evolving collection loops

---

## Phased Rollout
- **Phase 1 (this plan):** `look` tool + registry + router; `what's there` + `plant health` via YOLO eyes.
- **Phase 2:** `describe` path via private-ai-server; continuous-view loop.
- **Phase 3:** `who is it` (gated) + self-evolving collection loops.

---

## Open Questions
- [ ] Confirm XIAO's actual IP (currently offline; config value once known).
- [ ] Config source: how the companion desktop app writes `config/eyes.yaml` (key names to align).
- [ ] Health endpoint per eye (confirm `/health` on yolo_server, `/` on plant server).
