from __future__ import annotations

from computer_use.schema import Observation
from .base import BaseDriver


class OpenClawDriver(BaseDriver):
    def observe(self, target: dict) -> Observation:
        return Observation(source="mock", text="openclaw-bridge")

    def execute(self, action, target: dict) -> dict:
        return {"status": "ok", "driver": "openclaw", "action": action.kind}
