"""Configuration access helpers for the model server.

Extracted from the original model_server.py monolith (refactor). These read a
(already loaded) config dict and return the relevant sub-config with defaults.
The module-level `_config` global itself stays owned by model_server.py — these
functions take the config as an argument so they stay stateless leaves.
"""


def inference_cfg(config: dict) -> dict:
    """Return inference sub-config with defaults."""
    return (config or {}).get("inference", {})


def critique_cfg(config: dict) -> dict:
    """Return the tool-critique/repair sub-config for this run's tool loop.

    Defaults: schema_validate + repair_critique ON (they are pure, additive,
    and fail open — a validator bug can never block execution). revisor and
    refusal-reprompt are also ON by default since they only re-inject a nudge
    and are fail-open. Read live each turn so config edits take effect without
    a restart.
    """
    blk = (config or {}).get("critique") or {}
    return {
        "enabled": bool(blk.get("enabled", True)),
        "schema_validate": bool(blk.get("schema_validate", True)),
        "repair_critique": bool(blk.get("repair_critique", True)),
        "refusal_reprompt": bool(blk.get("refusal_reprompt", True)),
        "max_repair_retries": int(blk.get("max_repair_retries", 2)),
        "revisor": {
            "enabled": bool((blk.get("revisor") or {}).get("enabled", False)),
        },
    }
