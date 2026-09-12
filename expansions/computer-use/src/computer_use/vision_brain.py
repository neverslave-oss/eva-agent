"""vision_brain.py — LLM vision brain for computer-use.

Turns a screenshot + goal + action history into ONE structured next Action,
using the configured `computer_use` inference call_type (DeepSeek Vision via
HF Router / deepinfra). This is the "look at the screen, decide the next
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
    "## Available actions\n"
    "- click / double_click: selector MUST be real 'x,y' screen coordinates of the "
    "target (e.g. \"1191,73\"). NEVER use \"auto\" — an auto click at the current "
    "cursor position is a blind click and will not hit the target.\n"
    "- type: text is what to type into the focused field.\n"
    "- hotkey: text is the key combo, e.g. 'ctrl+l'.\n"
    "- navigate: url is the target URL (browser only).\n"
    "- launch: text is the app name to open (desktop only).\n"
    "- scroll: scroll the page or window.\n"
    "- wait: timeout_ms is how long to wait.\n"
    "- submit: submit a form.\n"
    "- done: the goal is achieved; include a short text summary.\n"
    "- abort: the goal cannot be achieved; include a short reason.\n\n"
    "## Driver selection\n"
    "- \"desktop\" (PyAutoGUI): clicks pixels on the real screen; used for launch, "
    "window management, or anything outside a browser.\n"
    "- \"browser\" (Playwright): drives an already-open page by DOM selector; used "
    "for navigate, fill forms, click links/buttons in a page, read page text.\n"
    "- IMPORTANT: launch is ALWAYS a desktop action — it opens the app on the real "
    "screen; never use \"browser\" for launch (Playwright cannot open an app).\n"
    "- For a browser goal (firefox/chrome/edge or a website): launch it via desktop "
    "first, then switch to browser to control the page.\n\n"
    "## History (what you already did)\n"
    "The 'Recent actions' list below shows your previous steps in order, oldest to "
    "newest, each with its result (ok / error / dry_run / done). Use it to avoid "
    "repeating yourself:\n"
    "- NEVER repeat an action that already appears in Recent actions.\n"
    "- If an action failed (error), it did NOT take effect — do not retry it the same "
    "way; either fix the cause, try a different approach, or return done/abort.\n"
    "- If the screen has not changed and your previous action already achieved the "
    "goal, return done.\n"
    "- Prefer returning done as soon as the goal is met; do not keep acting.\n\n"
    "## Hard rules\n"
    "- NEVER emit a click/double_click without real 'x,y' screen coordinates in "
    "selector. If you cannot identify exact coordinates, return wait or done instead.\n"
    "- NEVER repeat an action already in Recent actions.\n"
    "- Return ONLY the JSON object below — no prose, no markdown, no code fences.\n\n"
    "## Return format (exact)\n"
    '{"kind": "<one of the above>", "selector": null, "text": null, "url": null, '
    '"driver": "desktop", "timeout_ms": 5000, "metadata": {}}'
)


# System prompt for PLAN mode: return a JSON ARRAY of ordered actions, not one.
_SYSTEM_PLAN = (
    "You are a computer-use agent controlling a real desktop/browser. You see a "
    "screenshot of the current screen and a goal. Produce a concise, ordered PLAN "
    "of actions that will achieve the goal, then return ONLY a JSON array of action "
    "objects, no prose, no markdown.\n\n"
    "## Available actions\n"
    "- click / double_click: selector MUST be real 'x,y' screen coordinates of the "
    "target (e.g. \"1191,73\"). NEVER use \"auto\" — an auto click at the current "
    "cursor position is a blind click and will not hit the target.\n"
    "- type: text is what to type into the focused field.\n"
    "- hotkey: text is the key combo, e.g. 'ctrl+l'.\n"
    "- navigate: url is the target URL (browser only).\n"
    "- launch: text is the app name to open (desktop only).\n"
    "- scroll: scroll the page or window.\n"
    "- wait: timeout_ms is how long to wait.\n"
    "- submit: submit a form.\n"
    "- done: the goal is achieved; include a short text summary.\n"
    "- abort: the goal cannot be achieved; include a short reason.\n\n"
    "## Driver selection\n"
    "- \"desktop\" (PyAutoGUI): clicks pixels on the real screen; used for launch, "
    "window management, or anything outside a browser.\n"
    "- \"browser\" (Playwright): drives an already-open page by DOM selector; used "
    "for navigate, fill forms, click links/buttons in a page, read page text.\n"
    "- IMPORTANT: launch is ALWAYS a desktop action — it opens the app on the real "
    "screen; never use \"browser\" for launch (Playwright cannot open an app).\n"
    "- For a browser goal (firefox/chrome/edge or a website): launch it via desktop "
    "first, then switch to browser to control the page.\n\n"
    "## History (what you already did)\n"
    "The 'Recent actions' list below shows your previous steps in order, oldest to "
    "newest, each with its result (ok / error / dry_run / done). Use it to avoid "
    "repeating yourself:\n"
    "- NEVER repeat an action that already appears in Recent actions.\n"
    "- If an action failed (error), it did NOT take effect — do not retry it the same "
    "way; either fix the cause, try a different approach, or return done/abort.\n"
    "- Do not repeat the same action twice in the plan.\n"
    "- Keep the plan short (3-8 actions). Only end with done when the goal is fully "
    "achieved; only end with abort when it is impossible.\n\n"
    "## Hard rules\n"
    "- NEVER emit a click/double_click without real 'x,y' screen coordinates in "
    "selector. If you cannot identify exact coordinates, use wait or done instead.\n"
    "- NEVER repeat an action already in Recent actions.\n"
    "- Return ONLY the JSON array below — no prose, no markdown, no code fences.\n\n"
    "## Return format (exact): a JSON array, e.g.\n"
    '[{"kind": "click", "selector": "1191,73", "text": null, "url": null, '
    '"driver": "browser", "timeout_ms": 5000, "metadata": {}}, {"kind": "type", '
    '"selector": null, "text": "hello", "url": null, "driver": "browser", '
    '"timeout_ms": 5000, "metadata": {}}]'
)


def _build_messages(goal: str, screenshot: str | None, history: list[dict],
                    hint: str | None = None, system: str | None = None) -> list[dict]:
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

    if hint:
        user_parts.append({"type": "text", "text": f"Controller feedback: {hint}"})

    if history:
        # Rich recent history so the model knows exactly what it already did
        # and what happened. Numbered steps, oldest to newest, with full detail.
        lines = []
        start = max(0, len(history) - 8)
        for i, h in enumerate(history[start:], start=start + 1):
            act = h.get("action", {})
            res = h.get("result", {})
            detail = (
                act.get("text")
                or act.get("selector")
                or act.get("url")
                or act.get("driver")
                or ""
            )
            status = res.get("status", "?")
            err = res.get("error", "")
            line = f"  {i}. {act.get('kind')} {detail} => {status}"
            if err:
                line += f" (error: {err})"
            lines.append(line)
        user_parts.append({
            "type": "text",
            "text": "Recent actions (oldest to newest, do NOT repeat any of these):\n"
            + "\n".join(lines),
        })

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_parts},
    ]


def _extract_first_json(text: str) -> str | None:
    """Extract the first complete, balanced JSON object from a string.

    The model sometimes wraps its answer in <actions>...</actions> tags or
    emits several JSON objects, and occasionally truncates mid-object. A
    plain regex can't handle nested braces (e.g. metadata objects), so we
    scan for the first '{' and walk to its matching '}' accounting for
    nesting and string literals.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_action(raw: str) -> Action | None:
    """Parse the LLM's JSON reply into a schema-valid Action, or None if invalid."""
    text = (raw or "").strip()
    # Strip markdown fences if the model wrapped the JSON.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Try the whole thing first; if that fails, extract the first complete
    # balanced JSON object (handles <actions> wrappers, multiple objects, and
    # trailing truncated content).
    data = None
    try:
        data = json.loads(text)
    except Exception:
        obj = _extract_first_json(text)
        if obj:
            try:
                data = json.loads(obj)
            except Exception:
                data = None
    if data is None:
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
            driver=data.get("driver"),
            metadata=data.get("metadata", {}) or {},
        )
    except Exception as e:
        logger.warning("[vision_brain] invalid action from LLM: %s", e)
        return None


def _extract_first_array(text: str) -> str | None:
    """Extract the first complete, balanced JSON array from a string.

    Mirrors _extract_first_json but for arrays (the plan mode returns a JSON
    array of actions). Handles nesting and string literals.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_plan(raw: str) -> list[Action] | None:
    """Parse the LLM's JSON-array reply into a list of schema-valid Actions.

    Returns None if the output is not a valid non-empty list of actions.
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        arr = _extract_first_array(text)
        if arr:
            try:
                data = json.loads(arr)
            except Exception:
                data = None
    # Tolerate a single action object (the model sometimes ignores the "return
    # an array" instruction and emits one object). Wrap it into a one-item plan.
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        logger.warning("[vision_brain] could not parse LLM plan array: %r", raw[:200])
        return None
    actions: list[Action] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind not in _ALLOWED_KINDS:
            logger.warning("[vision_brain] plan contains disallowed kind: %r", kind)
            return None
        try:
            actions.append(Action(
                kind=kind,
                selector=item.get("selector"),
                text=item.get("text"),
                url=item.get("url"),
                timeout_ms=item.get("timeout_ms", 5000),
                driver=item.get("driver"),
                metadata=item.get("metadata", {}) or {},
            ))
        except Exception as e:
            logger.warning("[vision_brain] invalid plan action: %s", e)
            return None
    if not actions:
        return None
    return actions


class VisionBrain:
    """Wraps InferenceProvider to get the next computer-use action from a screenshot."""

    def __init__(self, provider=None, call_type: str = "computer_use"):
        self._provider = provider  # InferenceProvider instance (injected for tests)
        self.call_type = call_type
        # Last failure reason from next_action(), so callers can surface WHY a
        # vision decision failed instead of a generic "unavailable".
        self.last_error: str | None = None
        # Exact raw output the model returned for the most recent next_action(),
        # so callers/audit can see precisely what the LLM emitted (before
        # parsing). Set even on empty/unavailable responses.
        self.last_raw: str | None = None

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

    def resolved_model(self) -> str | None:
        """Return the actual model id this brain routes to for its call_type.

        Reads the configured model_overrides for the call_type (e.g.
        ``providers.model_overrides.computer_use``) so trajectory records and
        logs report the real model (DeepSeek Vision) instead of a stale
        hardcoded default.
        """
        try:
            provider = self._get_provider()
            p = provider.get_provider(self.call_type)
            return provider.get_model(p, self.call_type)
        except Exception:
            return None

    def next_action(self, goal: str, screenshot: str | None,
                    history: list[dict] | None = None,
                    hint: str | None = None,
                    max_new_tokens: int = 512) -> Action | None:
        """Return the next Action for the current screenshot, or None on failure.

        On failure, sets self.last_error to the specific reason (inference
        error, empty/unavailable response, or parse failure) so callers can
        log/surface WHY the vision brain failed rather than masking it.
        """
        self.last_error = None
        messages = _build_messages(goal, screenshot, history or [], hint=hint)
        provider = self._get_provider()
        # --- MODEL IO LOG (grep-able) ---
        try:
            logger.info("\n---input---\n%s\n---end--",
                        json.dumps(messages, ensure_ascii=False)[:8000])
        except Exception:
            pass
        try:
            raw = provider.infer(
                messages, max_new_tokens=max_new_tokens, call_type=self.call_type
            )
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        try:
            logger.info("\n---output---\n%s\n---end--", str(raw)[:8000])
        except Exception:
            pass
        self.last_raw = raw
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

    def plan_actions(self, goal: str, screenshot: str | None,
                     history: list[dict] | None = None,
                     hint: str | None = None,
                     max_new_tokens: int = 1024) -> list[Action] | None:
        """Return an ordered PLAN (list of Actions) for the current screenshot.

        The model sees the goal + current screen once and returns a coherent
        sequence of actions to achieve it, instead of one action at a time.
        On failure sets self.last_error and returns None.
        """
        self.last_error = None
        messages = _build_messages(goal, screenshot, history or [], hint=hint,
                                   system=_SYSTEM_PLAN)
        provider = self._get_provider()
        # --- MODEL IO LOG (grep-able) ---
        try:
            logger.info("\n---input---\n%s\n---end--",
                        json.dumps(messages, ensure_ascii=False)[:8000])
        except Exception:
            pass
        try:
            raw = provider.infer(
                messages, max_new_tokens=max_new_tokens, call_type=self.call_type
            )
        except Exception as e:
            self.last_error = f"inference failed: {e}"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        try:
            logger.info("\n---output---\n%s\n---end--", str(raw)[:8000])
        except Exception:
            pass
        self.last_raw = raw
        if not raw:
            self.last_error = "inference returned empty response"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        if raw == "(inference unavailable)":
            self.last_error = "inference unavailable (provider returned the unavailable sentinel)"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        plan = _parse_plan(raw)
        if plan is None:
            self.last_error = f"could not parse valid plan from model output: {raw[:200]!r}"
            logger.error("[vision_brain] %s", self.last_error)
            return None
        return plan
