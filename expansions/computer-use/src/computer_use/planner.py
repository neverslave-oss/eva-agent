from __future__ import annotations

import re

from .schema import Action, ActionBatch, Observation

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class Planner:
    """Deterministic goal-to-actions planner.

    This baseline planner is intentionally rule-based so the expansion works
    without requiring an LLM dependency at runtime. It emits schema-valid,
    atomic actions only.
    """

    def plan(self, goal: str, observation: Observation) -> ActionBatch:
        text = (goal or "").strip()
        if not text:
            return ActionBatch(actions=[Action(kind="abort", text="empty goal")])

        actions: list[Action] = []

        # URL-aware navigation
        m = _URL_RE.search(text)
        if m:
            actions.append(Action(kind="navigate", url=m.group(0)))

        lowered = text.lower()
        if "scroll" in lowered:
            actions.append(Action(kind="scroll"))
        if "click" in lowered:
            actions.append(Action(kind="click", selector="auto"))

        # Minimal typing intent extraction: type "..."
        quoted = re.search(r'type\s+["“](.+?)["”]', text, re.IGNORECASE)
        if quoted:
            actions.append(Action(kind="type", text=quoted.group(1)))

        if "wait" in lowered:
            actions.append(Action(kind="wait", timeout_ms=2000))

        actions.append(Action(kind="done", text="planned and executed"))
        return ActionBatch(actions=actions)
