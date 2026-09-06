from __future__ import annotations

from computer_use.schema import Observation
from .base import BaseDriver


class PyAutoGUIDriver(BaseDriver):
    def observe(self, target: dict) -> Observation:
        return Observation(source="desktop", text="")

    def execute(self, action, target: dict) -> dict:
        return {"status": "ok", "driver": "pyautogui", "action": action.kind}
