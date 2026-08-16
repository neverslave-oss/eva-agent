"""
ADR-006: Capability verification via local model inference.
Asks the local model whether a skill can handle a task before acquisition.
Cloud is NOT used here — verification is a local-only lightweight check.
"""
import logging

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """Does the skill "{name}" ({description}) have the capability to handle this task:
"{task}"
Answer with exactly YES or NO, then one sentence of reasoning."""


def verify(skill: dict, task: str, infer_fn) -> tuple[bool, str]:
    """
    Ask the local model whether skill can handle task.
    Returns (can_handle: bool, reasoning: str).
    Fails open on inference error (returns True, logs warning).
    """
    name = skill.get("name", "")
    description = skill.get("description", "")
    prompt = VERIFY_PROMPT.format(name=name, description=description, task=task)
    try:
        response = infer_fn(prompt, max_new_tokens=60)
        text = response.strip().upper()
        can_handle = text.startswith("YES")
        logger.info(f"[ADR-006] verify '{name}' for '{task[:50]}' → {can_handle}")
        return can_handle, response.strip()
    except Exception as e:
        logger.warning(f"[ADR-006] verification inference failed: {e} — failing open")
        return True, f"skipped (error: {e})"
