# Spec: `sensors` Tool — Environment Sensing for EVA

**Created:** 2026-08-19
**Status:** SPEC (design + implementation detail)
**Branch:** feat/sensors-tool (from dev)
**Author:** Fabio (pacificDev) + Copilot
**Scope:** Design + implementation spec for a unified `sensors` tool so EVA can read the Pi's sensor data (temperature, humidity, soil moisture) and, in a later hardware phase, control the water pump.

---

## 1. Goal

Give EVA (kernel-evolving) a unified **`sensors`** tool that abstracts the Pi's sensor endpoints behind a single semantic interface, so the agent can read environmental data (temperature, humidity, soil moisture) and, in a later phase, manually override the water pump. Modular and extensible to more sensors/actuators (touchscreen, Pi display, additional probes) with the same plug-and-play, config-driven pattern already used by the `look` vision tool.

## 2. Context / Existing Assets

### 2.1 Pi sensor endpoint (pi-pico-smart-monitor)
- `GET http://<pi>:5000/pico/sensors` → JSON:
  ```json
  {"humi":null,"moisture":0,"moisture_percent":0.0,"temp":null}
  ```
  - `temp` — temperature (°C) from DHT11 (null if Pico not connected)
  - `humi` — humidity (%) from DHT11 (null if Pico not connected)
  - `moisture` — digital soil moisture (1 = wet, 0 = dry)
  - `moisture_percent` — moisture percentage
- `POST http://<pi>:5000/pico/sensors` accepts JSON `{relay: bool, watering: bool}` to control the pump (actuation — deferred to hardware phase).
- Pi also has a small display (LCD) that can log messages (deferred extension point).

### 2.2 Two-Pico topology
- **Pico (USB)** — reads temp/humi/moisture, sends CSV over USB serial to the Pi → exposed at `/pico/sensors`.
- **Pico W (Wi-Fi)** — polls `/pico/sensors`, drives the water-pump relay (GP15) when dry. Firmware flash deferred to hardware session.

### 2.3 Reuse — `look` tool pattern (implemented, `src/core/vision/`)
The `sensors` tool mirrors the `look` tool's proven structure:
- `src/core/vision/registry.py` — config-driven registry, `_expand` env resolution
- `src/core/vision/router.py` — `route_look`, `_envelope` unified response
- `src/core/vision/eyes/plant_health.py` — plain `requests` client functions
- `src/core/tools.py` — `TOOLS` entry, `_run_look` wrapper, `execute_tool` dispatch
- `tests/test_look_tool.py` — test patterns (mock `requests`, injected config)

## 3. Architecture Decision

### 3.1 Unified `sensors` tool, action-based
```
sensors(
  action: "read" | "water on" | "water off",
  ...
)
```
- `sensors(action="read")` → GET `/pico/sensors` → returns temp/humi/moisture/moisture_percent.
- `sensors(action="water on"|"water off")` → POST relay override. **Reserved in the schema; actuation wired in the hardware phase** (Pico W firmware flash). For now returns a clear "deferred" message, or not exposed until wired.

### 3.2 Config-driven registry (never hardcoded)
```yaml
sensors:
  pi:
    base: "${KERNEL_EVO_SENSORS_PI_BASE:-}"   # e.g. http://<pi>:5000
    timeout_s: 5.0
```
- Env: `KERNEL_EVO_SENSORS_PI_BASE=http://192.168.1.120:5000` in `.env` / `.env.example`.
- Portable across installs; hub IP comes from each user's settings (same as the eyes).

### 3.3 Unified response envelope
```
sensors(action="read") →
  { sensor: "pi", action: "read", observations: {temp, humi, moisture, moisture_percent}, raw, error, timestamp, status }
```

### 3.4 Modular extension
- The `sensors` registry scales to N devices (each configured entry = a read source), like the eye registry.
- Future: `display` action (Pi LCD), touchscreen, additional probes — added as new entries/actions without restructuring.

## 4. Implementation Spec

### 4.1 Module structure (mirror `src/core/vision/`)
```
src/core/sensors/
  __init__.py
  registry.py   # load `sensors` config, resolve ${VAR}, health/status
  pi.py         # read_sensors(base) GET; (future) set_relay(base, relay, watering) POST
  router.py     # route_sensors(action, registry) + _envelope
```

### 4.2 Tool registration
- `src/core/tools.py` — add `sensors` to `TOOLS` (schema: `action` enum `["read"]`; reserve `water on`/`water off`), `_run_sensors(arguments)` wrapper, `elif name == "sensors"` dispatch.

### 4.3 Agent documentation (3 places)
- `src/core/memory/context.py` — tools table + native-tools list; bump count 12 → 13.
- `src/core/pipelines/micro_planner.py` — `NATIVE_TOOLS_BLOCK`.

### 4.4 Arg normalization
- `src/core/tool_arg_utils.py` — `sensors` block in `normalize_tool_args`.

### 4.5 Tests
- `tests/test_sensors_tool.py` (modeled on `test_look_tool.py`): client, dispatch, registration.

## 5. Files Affected
- `config.yaml`, `.env`, `.env.example`
- `src/core/sensors/` (new): `__init__.py`, `registry.py`, `pi.py`, `router.py`
- `src/core/tools.py`, `src/core/memory/context.py`, `src/core/pipelines/micro_planner.py`, `src/core/tool_arg_utils.py`
- `tests/test_sensors_tool.py` (new)

## 6. Verification
1. `python -m pytest tests/test_sensors_tool.py tests/test_look_tool.py tests/test_all_native_tools.py -q` → green.
2. `curl http://192.168.1.120:5000/pico/sensors` → sensor JSON.
3. Restart service; trigger `sensors(read)` via `/message` API → Eva returns temp/humi/moisture.

## 7. Scope / Non-goals (this phase)
- **Read-only**: `sensors(action="read")` only. Pump override (`water on/off`) and Pi display deferred to the dedicated hardware session (Pico W firmware flash).
- **Not** restoring the left eye / other sensors — separate hardware session.
- **Not** adding the touchscreen yet — modular extension point only.

## 8. Decisions Log
| # | Date | Decision |
|---|---|---|
| 1 | 2026-08-19 | Single `sensors` tool with `action` enum (mirrors `look`). |
| 2 | 2026-08-19 | Read-only now; pump override deferred to hardware phase. |
| 3 | 2026-08-19 | Config-driven endpoints (never hardcoded), portable. |
| 4 | 2026-08-19 | Pi display + touchscreen = future modular extension points. |
| 5 | 2026-08-19 | Left eye + all sensors back online = dedicated hardware session. |
