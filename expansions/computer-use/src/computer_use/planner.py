from __future__ import annotations

import re

from .schema import Action, ActionBatch, Observation

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SELECTOR_HINTS = [
    (re.compile(r"submit", re.I), "button[type=submit], input[type=submit], button:has-text('submit')"),
    (re.compile(r"search", re.I), "input[type=search], input[name*=search], input[placeholder*=search i]"),
    (re.compile(r"login|sign\s?in", re.I), "input[name*=user i], input[name*=email i], input[name*=login i]"),
    (re.compile(r"password", re.I), "input[type=password]"),
]


class Planner:
    """Deterministic goal-to-actions planner.

    Turns a natural-language goal into schema-valid, atomic actions. This
    baseline is rule-based so the expansion works without an LLM dependency,
    but it produces real, driver-executable actions (navigate/click/type/...)
    rather than stubs.
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

        # Click intent: "click <target>" — pick a selector from hints or fall back to auto.
        click_m = re.search(r"click(?:\s+(?:on|the))?\s+([\w\s-]+)", text, re.I)
        if click_m:
            target = click_m.group(1).strip().lower()
            sel = "auto"
            for pat, hint in _SELECTOR_HINTS:
                if pat.search(target):
                    sel = hint
                    break
            actions.append(Action(kind="click", selector=sel, metadata={"target": target}))
        elif "click" in lowered:
            # Bare "click" with no explicit target — still emit a real click action.
            actions.append(Action(kind="click", selector="auto"))

        if "scroll" in lowered:
            actions.append(Action(kind="scroll"))

        # Typing intent: type "..." or type into <field> "..."
        quoted = re.search(r'type\s+(?:into\s+)?["“](.+?)["”]', text, re.I)
        if quoted:
            actions.append(Action(kind="type", selector="input", text=quoted.group(1)))

        # Explicit wait
        if "wait" in lowered:
            actions.append(Action(kind="wait", timeout_ms=2000))

        # Assertions
        assert_m = re.search(r'(?:assert|check|verify)\s+(?:text\s+)?["“](.+?)["”]', text, re.I)
        if assert_m:
            actions.append(Action(kind="assert_text", text=assert_m.group(1)))

        actions.append(Action(kind="done", text="planned and executed"))
        return ActionBatch(actions=actions)
