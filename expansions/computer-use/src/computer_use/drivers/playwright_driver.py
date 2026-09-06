from __future__ import annotations

from computer_use.schema import Observation
from .base import BaseDriver


class PlaywrightDriver(BaseDriver):
    def observe(self, target: dict) -> Observation:
        return Observation(source="browser", url=target.get("url"), text="")

    def execute(self, action, target: dict) -> dict:
        return {"status": "ok", "driver": "playwright", "action": action.kind}
