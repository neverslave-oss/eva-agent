from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


ActionKind = Literal[
    "observe",
    "click",
    "double_click",
    "type",
    "hotkey",
    "navigate",
    "scroll",
    "wait",
    "assert_text",
    "assert_url",
    "upload",
    "submit",
    "done",
    "abort",
]


class Action(BaseModel):
    kind: ActionKind
    selector: str | None = None
    text: str | None = None
    url: str | None = None
    timeout_ms: int = Field(default=5000, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionBatch(BaseModel):
    actions: list[Action] = Field(default_factory=list)


class Observation(BaseModel):
    source: Literal["browser", "desktop", "mock"]
    url: str | None = None
    text: str | None = None
    state_hash: str | None = None


class Expectation(BaseModel):
    text_contains: str | None = None
    url_contains: str | None = None


class ExecutionResult(BaseModel):
    status: Literal["ok", "blocked", "error", "done"]
    message: str = ""
    completed: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
