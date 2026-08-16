"""
prompt_logger.py — thin shim delegating to database.agent.prompt_log.

The original module was refactored into database/agent/prompt_log.py.
This shim preserves backward compatibility for callers that still use
`import prompt_logger`.
"""
from __future__ import annotations

from database.agent.prompt_log import PromptLogRepository as _Repo
from runtime_paths import PROMPT_LOG_DB

_repo = _Repo(PROMPT_LOG_DB)


def log_prompt(chat_id: str, prompt: str, response: str, model: str = "") -> None:
    _repo.log_prompt(chat_id=chat_id, prompt=prompt, response=response, model=model)


def query_recent(limit: int = 20, chat_id: str | None = None):
    return _repo.query_recent(limit=limit, chat_id=chat_id)


def get_entry(entry_id: int):
    return _repo.get_entry(entry_id)


def clear_chat_logs(chat_id: str) -> int:
    return _repo.clear_chat_logs(chat_id=chat_id)
