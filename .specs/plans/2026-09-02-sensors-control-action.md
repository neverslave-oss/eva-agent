# Spec: `sensors` tool — generic `control` action (LCD, motor, relay)

**Created:** 2026-09-02
**Status:** IMPLEMENTED (2026-09-02) — Pi proxy + kernel-evo `control` action wired and tested (26/26 sensors tests pass). Auto-discovery still deferred.
**Author:** Fabio (pacificDev) + Copilot
**Branch:** feature/sensors-control (from dev)
**Related:** ADR-024 (Unified Sensors Tool), ADR-022 (Unified Tool-First Pipeline)

---

## 1. Goal

Give EVA (kernel-evolving) a **generic actuator/control action** on the existing
`sensors` tool so it can, through the Pi hub, command the Pico-W to:

- write a custom message to the **LCD**,
- move the **motor** (servo),
- turn the **relay** on/off to start/stop the **pump**,
- and be **extensible** to future actuators (touchscreen, additional pumps,
  lights, etc.) without restructuring the tool.

The Pico-W already exposes a working REST API (`POST /message`, `POST /motor`,
`GET /motor`) at `http://192.168.1.136:8080`. The Pi hub is the single network
entry point kernel-evo already talks to, so the Pi will **proxy** control calls
to the Pico-W. This keeps kernel-evo's config unchanged (still targets the Pi)
and avoids direct device registration / auto-discovery for now.

> **Two separate circuits (important):**
> - **Motor (servo)** — moves the **head** (ultrasonic + webcam), driven by PWM
>   on GP0. It is NOT the pump.
> - **Relay (pump)** — a separate circuit on GP15 that turns the **water pump**
>   on/off. It is NOT the servo motor.
>
> These must have **two separate control endpoints**. The existing `POST /motor`
> endpoint currently toggles the relay — that is a prototype shortcut. This
> plan separates them: `/motor` moves the servo head, `/relay` controls the pump.

## 2. Context / Existing Assets

### 2.1 Pico-W REST API (experiment 02-1)
Current (working):
- `POST <pico>/message` `{"line1","line2"}` → show message on LCD for
  `LCD_CUSTOM_MSG_SECONDS` (default 5 s).
- `POST <pico>/motor` `{"state":"on"|"off"}` → **today this toggles the relay**
  (prototype). This will be re-purposed/renamed.
- `GET <pico>/motor` → `{"state","relay"}`.

Target (after this plan):
- `POST <pico>/message` `{"line1","line2"}` → LCD message.
- `POST <pico>/motor` `{"angle": 0..180}` → move the **servo head** to an angle.
  (Replaces the relay-toggle behaviour; the servo sweep continues in software.)
- `GET <pico>/motor` → `{"angle": <current>}`.
- `POST <pico>/relay` `{"state":"on"|"off"}` → turn the **pump relay** on/off.
- `GET <pico>/relay` → `{"state":"on"|"off", "relay": bool}`.

Pico-W IP: `192.168.1.136:8080` (may change via DHCP — see §5).

### 2.2 Pi hub (sensors_data_api.py)
- `GET/POST /pico/sensors` — telemetry sink; now also stores the new sensor
  fields (`servo_angle`, `distance_cm`, `uv_raw`, `uv_percent`).
- No actuator endpoints yet. The Pi runs Flask on `0.0.0.0:5000`.

### 2.3 Kernel-evo `sensors` tool (ADR-024, read-only phase)
- `src/core/sensors/registry.py` — config-driven registry, `_expand` env.
- `src/core/sensors/pi.py` — `read_sensors` GET; `set_relay` POST is implemented
  but **not exposed** (returns "deferred" today).
- `src/core/sensors/router.py` — `route_sensors`, `VALID_ACTIONS = ("read",)`;
  `water on` / `water off` reserved but un-wired.
- `src/core/tools.py` — `sensors` tool schema; `_run_sensors` → `route_sensors`.
- `src/core/tool_arg_utils.py` — arg normalization for `sensors`.
- `src/core/memory/context.py` — tool list in the system prompt.
- `tests/test_sensors_tool.py` — registry/router/client/dispatch tests.

## 3. Architecture Decision

### 3.1 Generic `control` action (extensible by design)
Instead of hardcoding `water on/off` + separate display actions, add a single
generic action:

```
sensors(action="control", target="pi",
        device="lcd" | "motor" | "relay",   # actuator id (extensible)
        command="...",                       # e.g. "on" | "off" | "message" | "move"
        params={...})                        # free-form key/value for the command
```

The router maps `(device, command)` to a Pi proxy endpoint. New actuators are
added by registering a new `(device, command)` → endpoint mapping, with no
restructuring.

### 3.2 Pi proxies control to the Pico-W
Add to the Pi's `sensors_data_api.py` a small control proxy. **Motor and relay
are separate devices mapped to separate Pico endpoints:**

```
POST /pico/control
  {"device": "relay", "command": "on"}          -> Pico POST /relay {"state":"on"}
  {"device": "relay", "command": "off"}         -> Pico POST /relay {"state":"off"}
  {"device": "motor", "command": "move", "params": {"angle": 90}}
                                                 -> Pico POST /motor {"angle":90}
  {"device": "motor", "command": "center"}      -> Pico POST /motor {"angle":90}
  {"device": "lcd",   "command": "message", "params": {"line1": "...", "line2": "..."}}
                                                 -> Pico POST /message {"line1","line2"}
```

The Pi reads the Pico-W base URL from an env var (`PICO_W_BASE`, default
`http://192.168.1.136:8080`) so it is configurable and not hardcoded in code.

### 3.3 Kernel-evo `pi.py` client
Add `control(base, device, command, params)` to `src/core/sensors/pi.py` that
POSTs to the Pi's `/pico/control`. Kernel-evo still only needs to know the Pi
base URL — no new config section required.

### 3.4 Router wiring
Extend `VALID_ACTIONS` with `"control"` and route it to the Pi proxy via the new
client. Keep `water on` / `water off` as **aliases** that map to
`control(device="relay", command="on|off")` for backward-compat with the schema
and the earlier spec language.

## 4. Implementation Spec

### 4.1 Pi hub — `pi/sensors_data_api.py`
Add `import os`, `import requests` (or `urllib`) and:

```python
PICO_W_BASE = os.environ.get("PICO_W_BASE", "http://192.168.1.136:8080")

@sensors_api.route('/pico/control', methods=['POST'])
def pico_control():
    data = request.get_json(force=True) or {}
    device = str(data.get('device', '')).lower()
    command = str(data.get('command', '')).lower()
    params = data.get('params') or {}
    # Relay (pump) is a SEPARATE circuit from the motor (servo head).
    if device == 'relay':
        state = 'on' if command in ('on', 'start') else 'off'
        return _proxy_post('/relay', {'state': state})
    if device == 'motor':
        # Motor = servo head (ultrasonic + webcam), not the pump.
        # Move to an angle; default to center (90).
        try:
            angle = int(params.get('angle', 90))
        except (TypeError, ValueError):
            angle = 90
        angle = max(0, min(180, angle))
        return _proxy_post('/motor', {'angle': angle})
    if device == 'lcd' and command == 'message':
        return _proxy_post('/message', {'line1': params.get('line1', ''), 'line2': params.get('line2', '')})
    return jsonify({'ok': False, 'error': f'unsupported device/command: {device}/{command}'}), 400
```

with a helper `_proxy_post(path, payload)` that forwards to `PICO_W_BASE + path`
and returns the Pico's JSON (or a clear error if the Pico is unreachable).

### 4.2 Kernel-evo — `src/core/sensors/pi.py`
Add:

```python
def control(base, device, command, params=None, timeout_s=5.0):
    url = f"{base.rstrip('/')}/pico/control"
    payload = {"device": device, "command": command, "params": params or {}}
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code >= 500:
            return {"error": f"control endpoint returned HTTP {r.status_code}"}
        try:
            return r.json()
        except Exception:
            return {"ok": True, "raw": r.text}
    except Exception as exc:
        logger.warning("[sensors] control failed from %s: %s", url, exc)
        return {"error": str(exc)}
```

### 4.3 Kernel-evo — `src/core/sensors/router.py`
- Extend `VALID_ACTIONS = ("read", "control")`.
- Add `_run_control(dev, device, command, params)` → `pi_client.control(...)`.
- Map `"water on"` → `control(device="relay", command="on")`,
  `"water off"` → `control(device="relay", command="off")` as aliases.
- Return the same unified envelope (`status`, `observations`, `error`, ...).

### 4.4 Kernel-evo — `src/core/tools.py`
- Update the `sensors` tool description to mention `control` and the supported
  devices/commands.
- Extend the `action` enum to `["read", "control", "water on", "water off"]`.
- Add `device` (enum `["lcd","motor","relay"]`) and `command` and `params`
  (object) properties to the schema.
- `_run_sensors` passes `device`, `command`, `params` through to `route_sensors`.

### 4.5 Kernel-evo — `src/core/tool_arg_utils.py`
- For `sensors`: alias `_move("device","target")` is already there; add nothing
  that conflicts. Optionally map `"do"` → `"command"`.

### 4.6 Kernel-evo — `src/core/memory/context.py`
- Update the tool-list lines to mention the `control` action and its devices.

### 4.7 Tests — `tests/test_sensors_tool.py`
- Pi client: `control` posts correct payload/url.
- Router: `control` with `device="relay", command="on"` calls `pi_client.control`.
- Router: `water on` alias maps to relay on.
- Router: unknown `device/command` returns error.
- Tool dispatch: `sensors` schema includes `control` in the enum and `device`/
  `command`/`params` properties.

## 5. Config / Environment

- **Pi**: `PICO_W_BASE=http://192.168.1.136:8080` (set in the Pi's environment or
  a `.env` next to `sensors_data_api.py`). Defaults to `192.168.1.136:8080`.
- **Kernel-evo**: no new config needed — it keeps talking to the Pi base URL
  (`KERNEL_EVO_SENSORS_PI_BASE`). If the Pico-W IP changes, only the Pi env var
  changes.
- Auto-discovery of the Pico-W is **out of scope** for this phase (see §7).

## 6. Verification / Acceptance Criteria

1. `curl -X POST <pi>:5000/pico/control -d '{"device":"relay","command":"on"}'`
   → Pico relay energizes; `{"ok":true,...}` returned.
2. `... '{"device":"relay","command":"off"}'` → relay releases.
3. `... '{"device":"lcd","command":"message","params":{"line1":"Hi","line2":"EVA"}}'`
   → LCD shows the message.
4. Kernel-evo `sensors(action="control", device="relay", command="on")` works
   end-to-end (via `execute_tool` with mocked/real Pi).
5. `sensors(action="water on")` alias still works.
6. `pytest tests/test_sensors_tool.py` passes.

## 7. Out of scope (future)

- Auto-discovery / device registration of the Pico-W (fixed env var for now).
- Dedicated servo `move` endpoint on the Pico (motor currently == pump relay).
- Touchscreen / additional actuators beyond lcd/motor/relay.
- Persisting control history / audit log.

## Status
[x] Implemented (2026-09-02)
- Pi `/pico/control` proxy endpoint added to `sensors_data_api.py`.
- Kernel-evo `pi.control()`, router `control` action + `water on/off` aliases.
- `sensors` tool schema extended with `device`/`command`/`params`.
- `context.py` + `tool_arg_utils.py` updated.
- Tests added; `pytest tests/test_sensors_tool.py` → 26 passed.
- Pico-W `POST /motor` (servo head) + `POST /relay` (pump) verified live.
