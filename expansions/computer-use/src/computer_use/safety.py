from __future__ import annotations

from urllib.parse import urlparse
from .schema import Action


class PolicyViolation(Exception):
    pass


class PolicyEngine:
    def __init__(self, policy: dict | None = None):
        self.policy = policy or {}
        self.allow_actions = set(self.policy.get("allow_actions", []))
        self.deny_actions = set(self.policy.get("deny_actions", []))
        self.browser_allowlist = set(self.policy.get("browser_domain_allowlist", []))

    def validate_action(self, action: Action) -> None:
        if self.allow_actions and action.kind not in self.allow_actions:
            raise PolicyViolation(f"action not allowed: {action.kind}")
        if action.kind in self.deny_actions:
            raise PolicyViolation(f"action explicitly denied: {action.kind}")
        if action.kind == "navigate" and self.browser_allowlist and action.url:
            host = (urlparse(action.url).hostname or "").lower()
            if host not in self.browser_allowlist:
                raise PolicyViolation(f"domain not allowlisted: {host}")

    def mask_text(self, value: str) -> str:
        lowered = value.lower()
        if any(k in lowered for k in ["password", "token", "secret", "api_key"]):
            return "[MASKED]"
        return value
