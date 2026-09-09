"""vision_brain.py — LLM vision brain for computer-use.

Turns a screenshot + goal + action history into ONE structured next Action,
using the configured `computer_use` inference call_type (default: MiniMax-M3
via HF Router / novita). This is the "look at the screen, decide the next
step" primitive that replaces the deterministic planner's blind rules.

The brain is a thin adapter over the existing InferenceProvider routing — it
sends the screenshot as a multimodal message and parses the LLM's structured
JSON reply into a schema-valid Action. Guardrails live in the caller
(LLMPlanner): whitelist validation, policy engine, step cap, confirm gate.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path

from computer_use.schema import Action, ActionKind

logger = logging.getLogger(__name__)

# Allowed action kinds the LLM may emit (the whitelist). Anything else is rejected.
_ALLOWED_KINDS = set(ActionKind.__args__) - {"observe"}

# System prompt instructing the model to return ONLY a JSON action object.
_SYSTEM = (
    "You are a computer-use agent controlling a real desktop/browser. You see a "
    "screenshot of the current screen and a goal. Decide the SINGLE next action to "
    "take toward the goal, then return ONLY a JSON object, no prose, no markdown.\n\n"
    "Valid action kinds: click, double_click, type, hotkey, navigate, scroll, wait, "
    "submit, launch, done, abort.\n"
    "- click: selector can be 'auto' (click current cursor position) or 'x,y' screen "
    "coordinates.\n"
    "- type: text is what to type.\n"
    "- hotkey: text is the key combo, e.g. 'ctrl+l'.\n"
    "- navigate: url is the target URL (browser only).\n"
    "- launch: text is the app name to open (desktop).\n"
    "- wait: timeout_ms is how long to wait.\n"
    "- done: the goal is achieved; include a short text summary.\n"
    "- abort: the goal cannot be achieved; include a short reason.\n\n"
    "Return format (exact):\n"
    '{"kind": "<one of the above>", "selector": null, "text": null, "url": null, '
    '"timeout_ms": 5000, "metadata": {}}'
)


def _build_messages(goal: str, screenshot: str | None, history: list[dict]) -> list[dict]:
    """Build the multimodal message list for the vision model."""
    user_parts: list[dict] = []
    b64 = None
    if screenshot:
        if screenshot.startswith("data:image"):
            # Inline base64 data URI (e.g. from pyautogui/playwright screenshot()).
            try:
                b64 = screenshot.split(",", 1)[1]
            except Exception:
                b64 = None
        elif Path(screenshot).exists():
            # Local file path.
            try:
                with open(screenshot, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                b64 = None
    if b64:
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    user_parts.append({"type": "text", "text": f"Goal: {goal}"})

    if history:
        # Compact recent history so the model knows what it already did.
        lines = []
        for h in history[-8:]:
            act = h.get("action", {})
            res = h.get("result", {})
            lines.append(
                f"- {act.get('kind')} {act.get('text') or act.get('selector') or act.get('url') or ''} "
                f"=> {res.get('status', '?')}"
            )
        user_parts.append({
            "type": "text",
            "text": "Recent actions:\n" + "\n".join(lines),
        })

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_parts},
    ]


def _parse_action(raw: str) -> Action | None:
    """Parse the LLM's JSON reply into a schema-valid Action, or None if invalid."""
    text = (raw or "").strip()
    # Strip markdown fences if the model wrapped the JSON.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Find the first {...} block as a fallback.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        logger.warning("[vision_brain] could not parse LLM JSON: %r", raw[:200])
        return None
    kind = data.get("kind")
    if kind not in _ALLOWED_KINDS:
        logger.warning("[vision_brain] LLM emitted disallowed kind: %r", kind)
        return None
    try:
        return Action(
            kind=kind,
            selector=data.get("selector"),
            text=data.get("text"),
            url=data.get("url"),
            timeout_ms=data.get("timeout_ms", 5000),
            metadata=data.get("metadata", {}) or {},
        )
    except Exception as e:
        logger.warning("[vision_brain] invalid action from LLM: %s", e)
        return None


class VisionBrain:
    """Wraps InferenceProvider to get the next computer-use action from a screenshot."""

    def __init__(self, provider=None, call_type: str = "computer_use"):
        self._provider = provider  # InferenceProvider instance (injected for tests)
        self.call_type = call_type
        # Last failure reason from next_action(), so callers can surface WHY a
        # vision decision failed instead of a generic "unavailable".
        self.last_error: str | None = None

    def _get_provider(self):
        if self._provider is not None:
            return self._provider
        # Lazy import to avoid hard dependency at module load.
        from core.inference.provider import InferenceProvider
        import yaml
        cfg_path = os.environ.get(
            "KERNEL_EVO_CONFIG",
            str(Path(__file__).resolve().parents[3] / "config.yaml"),
        )
        try:
            cfg = yaml.safe_load(open(cfg_path)) or {}
        except Exception:
            cfg = {}
        return InferenceProvider(cfg)

    def next_action(self, goal: str, screenshot: str | None,
                    history: list[dict] | None = None,
                    max_new_tokens: int = 512) -> Action | None:
        """Return the next Action for the current screenshot, or None on failure.

        On failure, sets self.last_error to the specific reason (inference
        error, empty/unavailable response, or parse failure) so callers can
        log/surface WHY the vision brain failed rather than masking it.
        """
        self.last_error = None
        messages = _build_messages(goal, screenshot, history or [])
        provider = self._get_provider()
        try:
            raw = provider.infer(
                messages, max_new_tokens=max_new_tokens, call_type=self.call_type
            )
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        if not raw:
            self.last_error = "inference returned empty response"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        if raw == "(inference unavailable)":
            self.last_error = "inference unavailable (provider returned the unavailable sentinel)"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        action = _parse_action(raw)
        if action is None:
            self.last_error = f"could not parse valid action from model output: {raw[:200]!r}"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        return action
