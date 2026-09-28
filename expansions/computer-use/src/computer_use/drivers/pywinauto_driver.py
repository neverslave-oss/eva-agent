from __future__ import annotations

from computer_use.schema import Observation
from .base import BaseDriver


class PyWinAutoDriver(BaseDriver):
    def observe(self, target: dict) -> Observation:
        return Observation(source="desktop", text="", state_hash="windows")

    def execute(self, action, target: dict) -> dict:
        return {"status": "ok", "driver": "pywinauto", "action": action.kind}
