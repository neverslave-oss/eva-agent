"""
refusal_patterns.py — Shared detection for capability-refusal / empty / dead-end
assistant answers that should not be treated as valid final responses or
persisted to conversation history.

Used by:
  - model_server.py (_looks_like_capability_refusal — re-prompt instead of
    returning the refusal as a final answer)
  - agent.py (triage — guard against saving refusal/empty/dead-end text to
    memory, since it poisons subsequent turns as history — R8)
"""

CAPABILITY_REFUSAL_MARKERS = (
    "cannot access the internet",
    "can't access the internet",
    "cannot browse",
    "can't browse",
    "no internet access",
    "i cannot perform web searches",
    "i can't perform web searches",
    "no access to external databases",
    "i don't have access to files",
    "i do not have access to files",
    "can't access files",
    "cannot access files",
    "i cannot access your filesystem",
    "cannot access the local filesystem",
    "i cannot execute commands",
    "i can't execute commands",
    "i don't have tools",
    "i do not have tools",
    "no tools available",
    "these tools are not real",
    "i cannot access real tools",
    "i'm just a language model",
    "i am just a language model",
    "i'm a text model",
    "i am a text model",
)

# Dead-end / non-answers that should never be persisted as a real assistant turn.
DEAD_END_MARKERS = (
    "(max steps reached)",
    "i could not complete that request.",
)


def looks_like_capability_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in CAPABILITY_REFUSAL_MARKERS)


def looks_like_dead_end(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return True
    return any(m in low for m in DEAD_END_MARKERS)


def should_skip_persisting(text: str) -> bool:
    """True if this assistant answer should NOT be saved to conversation history."""
    return looks_like_dead_end(text) or looks_like_capability_refusal(text)
