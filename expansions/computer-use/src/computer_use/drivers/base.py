from __future__ import annotations

from abc import ABC, abstractmethod
from computer_use.schema import Observation


class BaseDriver(ABC):
    @abstractmethod
    def observe(self, target: dict) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def execute(self, action, target: dict) -> dict:
        raise NotImplementedError
