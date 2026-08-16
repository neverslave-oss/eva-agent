"""
critique_layer.py — ADR-019 Phase 3: CritiqueLayer

After maybe_evolve() installs a skill, CritiqueLayer runs the skill on the
original task and assesses whether the gap was genuinely addressed.
Failed/partial attempts are retried up to max_iterations times with critique
notes injected into synthesis. RESOLVED verdicts close the goal_discovery signal
and write a routing hint to the skill's SKILL.md.
"""

import concurrent.futures
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from core.evolution.evolution_hook import EvolutionResult

logger = logging.getLogger(__name__)

ASSESS_PROMPT = """You are assessing whether a skill execution satisfied a user's original request.

Original user request: {original_request}

Skill output:
{output}

Assess whether the output would satisfy the user's original request.
Answer with exactly one of: RESOLVED, PARTIAL, FAILED
Then one sentence of reasoning.

RESOLVED = output directly satisfies the request with useful, complete content
PARTIAL  = output is related but incomplete or only partially satisfies the request
FAILED   = output is empty, an error, or completely unrelated to the request
"""


@dataclass
class CritiqueVerdict:
    verdict: str          # "RESOLVED" | "PARTIAL" | "FAILED"
    notes: str            # short explanation for retry loop
    should_retry: bool    # True if PARTIAL or FAILED and attempts < max
    attempt: int          # current attempt number (1-based)


class CritiqueLayer:
    """
    Assesses whether a newly installed skill genuinely addressed a capability gap.

    Config keys under thinking.critique_* (or nested dict `critique`):
        enabled: bool (default True)
        max_iterations: int (default 3)
        run_timeout_s: int (default 30)
    """

    def __init__(self, config: dict):
        # Support both flat `critique_*` keys and nested `critique:` dict
        _crit = config.get("critique", {})
        self._enabled = _crit.get("enabled", config.get("critique_enabled", True))
        self._max_iterations = int(_crit.get("max_iterations", config.get("critique_max_iterations", 3)))
        self._timeout_s = int(_crit.get("run_timeout_s", config.get("critique_run_timeout_s", 30)))

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        thought: dict,
        evolution_result: "EvolutionResult",
        infer_fn: Optional[Callable],
        attempt: int = 1,
    ) -> CritiqueVerdict:
        """
        1. Run the newly installed skill on the original user request (ADR-020) or thought text.
        2. Ask infer_fn to assess whether the output satisfies the original request.
        3. If RESOLVED: send Telegram confirmation buttons instead of marking resolved immediately.
        4. Return a CritiqueVerdict.
        """
        if not self._enabled:
            return CritiqueVerdict(verdict="RESOLVED", notes="critique disabled", should_retry=False, attempt=attempt)

        # ADR-020: prefer original user message over abstract thought text
        original_request = thought.get("_source_user_message") or thought.get("thought", "")
        thought_text = thought.get("thought", "")
        gap = getattr(evolution_result, "gap", thought_text) or thought_text

        # Sandboxed skill run against the original user request
        output = self._run_skill_sandboxed(original_request, infer_fn)

        # If skill didn't produce output, that's a FAILED
        if not output or not output.strip():
            return self._make_verdict("FAILED", "Skill produced no output (timeout or error)", attempt)

        # If no infer_fn, we can't assess properly
        if infer_fn is None:
            return self._make_verdict(
                "PARTIAL",
                "infer_fn not available — cannot assess output quality",
                attempt,
            )

        # Ask model to assess against original request
        prompt = ASSESS_PROMPT.format(
            original_request=original_request[:300],
            output=output[:500]
        )
        try:
            raw = infer_fn(prompt, max_new_tokens=80)
        except Exception as e:
            logger.warning(f"[CritiqueLayer] infer_fn error during assessment: {e}")
            return self._make_verdict("FAILED", f"Assessment model error: {e}", attempt)

        verdict_str, notes = self._parse_assessment(raw or "")

        # ADR-020: on RESOLVED — send confirmation buttons to user instead of auto-resolving
        if verdict_str == "RESOLVED":
            request_id = thought.get("_source_request_id")
            installed = getattr(evolution_result, "installed", None)
            skill_name = (installed[0] if isinstance(installed, list) and installed
                         else installed or "unknown")
            self._send_confirmation_request(
                thought=thought,
                original_request=original_request,
                skill_name=skill_name,
                skill_output=output,
                request_id=request_id,
            )
            # Return PENDING instead of RESOLVED — resolution happens via callback
            return CritiqueVerdict(
                verdict="PENDING_CONFIRM",
                notes=f"Sent confirmation to user. Skill: {skill_name}. {notes}",
                should_retry=False,
                attempt=attempt,
            )

        return self._make_verdict(verdict_str, notes, attempt)

    def _send_confirmation_request(
        self,
        thought: dict,
        original_request: str,
        skill_name: str,
        skill_output: str,
        request_id: Optional[int],
    ) -> None:
        """Send Telegram message with skill output + ✅/❌ confirmation buttons."""
        try:
            import services.channels.telegram_bot as _tb
            import database.agent.failed_requests as _fr

            chat_id = _tb.ALLOWED_CHAT_ID
            if not chat_id:
                logger.warning("[CritiqueLayer] no ALLOWED_CHAT_ID — cannot send confirmation")
                return

            if request_id:
                _fr.mark_pending_confirm(request_id, skill_name, skill_output)

            confirm_id = request_id or 0
            original_truncated = original_request[:120]
            output_truncated = skill_output[:800]

            msg = (
                f"🧠 *Evolution result* (attempt confirmed by critique)\n"
                f"You asked: _{original_truncated}_\n\n"
                f"I synthesised `{skill_name}` and here's what it produces:\n\n"
                f"{output_truncated}"
            )

            buttons = [[
                {"text": "✅ Yes, resolved", "callback_data": f"evo_confirm_resolved:{confirm_id}"},
                {"text": "❌ No, still broken", "callback_data": f"evo_confirm_failed:{confirm_id}"},
            ]]

            _tb.send_buttons(str(chat_id), msg, buttons)
            logger.info(f"[CritiqueLayer] confirmation request sent for request_id={confirm_id}")
        except Exception as e:
            logger.warning(f"[CritiqueLayer] _send_confirmation_request error: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_verdict(self, verdict: str, notes: str, attempt: int) -> CritiqueVerdict:
        if verdict == "RESOLVED":
            should_retry = False
        else:
            should_retry = attempt < self._max_iterations
        return CritiqueVerdict(verdict=verdict, notes=notes, should_retry=should_retry, attempt=attempt)

    def _parse_assessment(self, raw: str) -> tuple[str, str]:
        """Parse first word as verdict, rest as notes. Defaults to FAILED on parse error."""
        text = raw.strip()
        for verdict in ("RESOLVED", "PARTIAL", "FAILED"):
            if text.upper().startswith(verdict):
                notes = text[len(verdict):].strip().lstrip("-—:").strip()
                return verdict, notes or f"Verdict: {verdict}"
        logger.warning(f"[CritiqueLayer] could not parse assessment response: {text[:80]!r}")
        return "FAILED", f"Parse error — raw response: {text[:80]}"

    def _run_skill_sandboxed(self, task_text: str, infer_fn) -> Optional[str]:
        """
        Run agent.triage() in a sandboxed thread with timeout.
        Returns output string or None on error/timeout.
        """
        try:
            import core.agent as agent
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(agent.triage, task_text, None, True, "__critique__", None)
                return future.result(timeout=self._timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning("[CritiqueLayer] skill run timed out")
            return None
        except Exception as e:
            logger.warning(f"[CritiqueLayer] skill run error: {e}")
            return None

    def _write_routing_hint(
        self,
        skill_name: str,
        thought: dict,
        evidence_summary: str,
        resolved_at: str,
    ) -> None:
        """
        Append a ## Trigger Context section to the skill's SKILL.md.
        Only writes if the section doesn't already exist (idempotent).
        """
        skills_base = os.path.expanduser("~/.kernel-evolving/ecosystem")
        for root, dirs, files in os.walk(skills_base):
            if "SKILL.md" in files and Path(root).name == skill_name:
                skill_md = Path(root) / "SKILL.md"
                try:
                    content = skill_md.read_text(encoding="utf-8", errors="ignore")
                    if "## Trigger Context" in content:
                        logger.debug(f"[CritiqueLayer] routing hint already exists in {skill_md}")
                        return  # idempotent — already written
                    hint = (
                        f"\n\n## Trigger Context\n"
                        f"<!-- auto-generated by CritiqueLayer on {resolved_at} -->\n"
                        f"Invoke this skill when: {thought.get('thought', '')[:200]}\n"
                        f"Evidence that prompted creation: {evidence_summary}\n"
                        f"Confirmed effective on: {resolved_at}\n"
                    )
                    skill_md.write_text(content + hint, encoding="utf-8")
                    logger.info(f"[CritiqueLayer] routing hint written to {skill_md}")
                except Exception as e:
                    logger.warning(f"[CritiqueLayer] could not write routing hint to {skill_md}: {e}")
                return  # found the skill dir — stop walking
