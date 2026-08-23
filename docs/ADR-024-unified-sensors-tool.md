# ADR-024: Unified Sensors Tool (Read Pi Sensor Data)

**Status:** Implemented (read-only phase)
**Date:** 2026-08-20
**Author:** Fabio + Olly
**Related:** ADR-023 (Unified Look Tool), ADR-022 (Unified Tool-First Pipeline)

---

## Problem

EVA had no way to read environmental sensor data (temperature, humidity, soil
moisture) from the Raspberry Pi. Sensor access was unimplemented, and adding it ad
hoc would duplicate the same config-driven pattern already used for vision.

## Decision

Add a **unified `sensors` tool** mirroring the `look` vision tool (ADR-023), so EVA
can read environmental data from the Pi's `/pico/sensors` endpoint:

- **Registry** (`src/core/sensors/registry.py`): declares sensors + env-expands the
  Pi base URL (`KERNEL_EVO_SENSORS_PI_BASE`).
- **Pi client** (`src/core/sensors/pi.py`): `read_sensors` GET; `set_relay` POST
  (pump override) deferred to the hardware phase.
- **Router** (`src/core/sensors/router.py`): `route_sensors` envelope.
- **Dispatch** (`src/core/tools.py`): the `sensors` tool entry point.

Config lives under a `sensors:` section in `config.yaml`. The tool is registered in
the system prompt (`context.py`), micro-planner, and arg normalization
(`tool_arg_utils.py`).

## Scope / Deferred

- **Read-only phase now:** temp / humidity / moisture / moisture_percent.
- **Deferred to hardware phase:** pump override (water on/off) + Pi display.

## Consequences

- EVA can natively query environmental sensors via a first-class tool.
- Sensor sources are config-driven and extensible, consistent with the `look` tool.
- Tool count bumped to 14 with this addition.

**Key files:** `src/core/sensors/`, `src/core/tools.py`, `config.yaml`, `tests/test_sensors_tool.py`
