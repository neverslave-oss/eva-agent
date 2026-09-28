import pytest

from computer_use.schema import Action
from computer_use.safety import PolicyEngine, PolicyViolation


def test_allowlisted_navigation_happy_path():
    p = PolicyEngine({
        "allow_actions": ["navigate"],
        "browser_domain_allowlist": ["docs.openclaw.ai"],
    })
    p.validate_action(Action(kind="navigate", url="https://docs.openclaw.ai/guide"))


def test_non_allowlisted_domain_blocked_edge_case():
    p = PolicyEngine({
        "allow_actions": ["navigate"],
        "browser_domain_allowlist": ["docs.openclaw.ai"],
    })
    with pytest.raises(PolicyViolation):
        p.validate_action(Action(kind="navigate", url="https://evil.example.com"))


def test_explicitly_denied_action_edge_case():
    p = PolicyEngine({
        "allow_actions": ["submit", "click"],
        "deny_actions": ["submit"],
    })
    with pytest.raises(PolicyViolation):
        p.validate_action(Action(kind="submit"))
