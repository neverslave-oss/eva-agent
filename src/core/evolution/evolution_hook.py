"""
evolution_hook.py — ADR-004 entry point for the self-evolving agent.
NOT wired into agent.py yet — see the TODO comment below for the call site.

Usage (when ready to wire):
    In agent.py triage(), after the fallthrough to infer_with_tools:

    # TODO: ADR-004 evolution hook — wire when kernel-evolving is validated
    # result = maybe_evolve(text, _config, skills_dir)
    # if result and result.retry: return triage(text)  # retry with new skill
"""
import os
import threading
from .evolver import Evolver, EvolutionResult
from .evolution_log import EvolutionLog
from . import evolution_state as _evo_state
from . import pending_synthesis as _ps

EVOLUTION_ENABLED = os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true"


def maybe_evolve(task: str, config: dict, skills_dir: str, infer_fn=None, chat_id: str | None = None) -> EvolutionResult | None:
    """
    Called when agent.triage() finds no skill/routine match.
    Respects the evolution state machine: only runs when state=RUNNING
    and iteration cap has not been reached.
    `infer_fn` is an optional callable(prompt: str, max_new_tokens: int) -> str
    used for ADR-006 capability verification.
    `chat_id` is an optional Telegram chat id. When provided it is threaded into
    the ADR-021 Tier 2 approval gate so the user is asked for approval instead of
    the gate being silently skipped (fix 2026-08-26).
    """
    if not EVOLUTION_ENABLED:
        return None

    if not _evo_state.should_evolve():
        return None  # paused, stopped, or cap reached

    evolver = Evolver(config=config, skills_dir=skills_dir, infer_fn=infer_fn)
    result = evolver.run(task)

    # Persist to evolution log (with ADR-006 fields)
    try:
        log = EvolutionLog()
        log.record(task, result)
    except Exception:
        pass

    # If Tier 1 escalated and Tier 2 synthesis is available, attempt it
    if result.escalated:
        result = _try_tier2(task, result.gap, config, chat_id=chat_id)

    # Count against cap only when a new skill was actually installed
    if result.found and result.installed:
        _evo_state.increment()
        # Notify via Telegram
        try:
            import services.channels.telegram_bot as _tb
            chat_id = _tb.ALLOWED_CHAT_ID
            if chat_id:
                tier = "Tier 2 (synthesised)" if result.escalated else "Tier 1 (acquired)"
                provider = f" via `{result.provider_used}`" if result.provider_used else ""
                _tb.send_message(chat_id,
                    f"🧬 *Evolution complete*\n"
                    f"Task: _{task[:80]}_\n"
                    f"Installed: `{'`, `'.join(result.installed)}`\n"
                    f"{tier}{provider}\n"
                    f"Confidence: `{result.confidence}`"
                )
        except Exception:
            pass

    return result


_SYNTH_PROMPT = """\
You are a skill synthesiser for a local AI agent named Kernel Evo.
Generate a complete SKILL.md file for the following capability gap.

Follow the skill-creator pattern strictly:
- YAML frontmatter must contain ONLY: name, description
- Put all operating guidance in the Markdown body after the closing ---
- Use imperative instructions another agent can follow
- Keep it concise and practical
- Do not include README, changelog, or extra docs

Required structure:
---
name: <kebab-case-name>
description: <one clear sentence describing what the skill does and when to use it>
---

## Instructions
<step-by-step instructions for the agent>

Capability gap: {task}

Output ONLY the SKILL.md content, nothing else.
"""

_CRITIC_PROMPT = """\
You are a quality critic for AI agent skills. Evaluate the following synthesised SKILL.md.

Check:
1. Does the skill name (kebab-case) semantically match the capability gap?
2. Is the description a clear, searchable one-sentence summary with useful trigger guidance?
3. Does the Markdown body contain actionable instructions another agent can follow?
4. Does the skill avoid stuffing operating instructions into YAML frontmatter?

Capability gap: {task}

SKILL.md to review:
{skill_content}

Return ONLY valid JSON (no markdown, no explanation):
{{"verdict": "PASS", "score": 0.0, "issues": [], "suggested_name": "skill-name"}}
"""


def _parse_critic_verdict(text: str) -> dict:
    """Parse critic JSON response with fallback."""
    import json, re
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # Keyword fallback
    upper = text.upper()
    if "PASS" in upper:
        return {"verdict": "PASS", "score": 0.7, "issues": [], "suggested_name": ""}
    return {"verdict": "FAIL", "score": 0.0, "issues": ["parse error"], "suggested_name": ""}


# ── ADR-021: Tier 2 human-approval gate ──────────────────────────────────────

_SKIP_LOCAL_APPROVAL = os.environ.get("TIER2_APPROVAL_SKIP_LOCAL", "false").lower() == "true"


def _infer_skill_name_from_gap(gap: str) -> str:
    """Best-effort kebab-case skill name derived from the gap description."""
    import re
    words = re.sub(r'[^a-z0-9 ]', '', gap.lower()).split()
    return '-'.join(words[:5]) or 'new-skill'


def _send_tier2_approval_request(
    task: str, gap: str, provider: str, chat_id: str, synthesis_id: str
) -> None:
    """Send Telegram inline-button message asking user to approve/reject Tier 2 synthesis."""
    try:
        import services.channels.telegram_bot as _tb
        proposed_name = _infer_skill_name_from_gap(gap)
        cost_note = (
            "_(uses local model — no API credits)_"
            if provider == "local"
            else "_⚠️ This will use API credits._"
        )
        text = (
            f"🧬 *Tier 2 Evolution — Approval Required*\n\n"
            f"Kernel Evo wants to synthesise a new skill:\n"
            f"*Proposed name:* `{proposed_name}`\n"
            f"*Gap:* _{gap[:200]}_\n"
            f"*Provider:* `{provider}`\n"
            f"{cost_note}\n\n"
            f"Approve skill creation?"
        )
        buttons = [[
            {"text": "✅ Approve", "callback_data": f"tier2_approve:{synthesis_id}"},
            {"text": "❌ Reject",  "callback_data": f"tier2_reject:{synthesis_id}"},
        ]]
        _tb.send_buttons(str(chat_id), text, buttons)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[ADR-021] approval request send failed: {e}")


def _gate_tier2(
    task: str, gap: str, provider: str | None, config: dict, chat_id: str | None
) -> EvolutionResult | None:
    """
    ADR-021 gate: store the request, notify the user, return EvolutionResult(pending_approval=True).
    Returns None when the gate is disabled (local provider + skip flag) so the caller
    proceeds immediately.  Otherwise returns a pending result — the actual synthesis
    will be triggered by the approve callback.
    """
    if not provider:
        # No provider available — gate not needed; will fail anyway
        return None

    if provider == "local" and _SKIP_LOCAL_APPROVAL:
        return None

    resolved_chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not resolved_chat_id:
        # No channel to ask — skip gate, let synthesis proceed unguarded
        import logging
        logging.getLogger(__name__).warning(
            "[ADR-021] no chat_id available — skipping approval gate"
        )
        return None

    synthesis_id = _ps.enqueue(
        task=task, gap=gap, provider=provider, chat_id=str(resolved_chat_id)
    )
    _send_tier2_approval_request(task, gap, provider, str(resolved_chat_id), synthesis_id)
    return EvolutionResult(
        found=False,
        confidence="LOW",
        escalated=True,
        gap=gap,
        provider_used=provider,
        pending_approval=True,
        synthesis_id=synthesis_id,
    )


def _run_approved_synthesis(synthesis_id: str) -> None:
    """Called from telegram_bot callback when user approves Tier 2 synthesis."""
    record = _ps.remove(synthesis_id)
    if not record:
        return
    import threading as _t
    def _worker():
        approved_provider = record.get("provider", "")
        result = _try_tier2_pipeline(record["task"], record["gap"], {}, _approved=True)
        chat_id = record.get("chat_id", "")
        try:
            import services.channels.telegram_bot as _tb
            if result.found and result.installed:
                installed = ', '.join(f"`{s}`" for s in result.installed)
                used_provider = result.provider_used or approved_provider
                fallback_note = (
                    f"\n_(fell back from `{approved_provider}` — quota/error)_"
                    if used_provider and used_provider != approved_provider
                    else ""
                )
                _tb.send_message(chat_id,
                    f"🧬 *Skill synthesised and installed!*\n"
                    f"Installed: {installed}\n"
                    f"Provider: `{used_provider}`{fallback_note}\n"
                    f"Confidence: `{result.confidence}`"
                )
            else:
                _tb.send_message(chat_id,
                    f"⚠️ *Skill synthesis failed* (all providers tried)\n"
                    f"Reason: {result.gap[:200] or 'unknown'}"
                )
        except Exception:
            pass
    _t.Thread(target=_worker, daemon=True).start()


def _reject_synthesis(synthesis_id: str) -> None:
    """Called from telegram_bot callback when user rejects Tier 2 synthesis."""
    record = _ps.remove(synthesis_id)
    if not record:
        return
    # Write negative reward trajectory
    try:
        from core.evolution.evolution_log import EvolutionLog
        EvolutionLog().record_synthesis_rejection(
            task=record["task"], gap=record["gap"], provider=record["provider"]
        )
    except Exception:
        pass


# ── end ADR-021 ───────────────────────────────────────────────────────────────


def _try_tier2_pipeline(task: str, gap: str, config: dict, *, _approved: bool = False, chat_id: str | None = None) -> EvolutionResult:
    """ADR-010: Tier 2 via critic replica pipeline — synthesise then critic-gate before install."""
    try:
        from pathlib import Path
        import time as _time
        import yaml as _yaml

        # Load config for provider — needed before gate so we can show the user which provider
        _cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.yaml')
        with open(_cfg_path) as _f:
            _cfg = _yaml.safe_load(_f)
        from core.inference.provider import get_provider as _get_provider
        _prov = _get_provider(_cfg)

        # ── ADR-021: human-approval gate ─────────────────────────────────────────
        if not _approved:
            # Determine which provider would be used (best-effort)
            _provider_name = getattr(_prov, 'name', None) or type(_prov).__name__.lower()
            gate_result = _gate_tier2(task, gap, _provider_name, config, chat_id)
            if gate_result is not None:
                return gate_result
        # ── end ADR-021 gate ─────────────────────────────────────────────────────

        tmp_dir = Path("~/.kernel-evolving/workspace/tmp").expanduser()
        tmp_dir.mkdir(parents=True, exist_ok=True)

        def _synthesise(extra_context: str = "") -> str:
            synth_task = gap
            if extra_context:
                synth_task += f"\n\nPrevious attempt issues:\n{extra_context}"
            prompt = _SYNTH_PROMPT.format(task=synth_task)
            return _prov.infer([{"role": "user", "content": prompt}], max_new_tokens=8192, call_type="synthesis")

        def _critique(skill_content: str) -> dict:
            prompt = _CRITIC_PROMPT.format(task=task, skill_content=skill_content[:3000])
            raw = _prov.infer([
                {"role": "user", "content": prompt}
            ], max_new_tokens=256, call_type="critic")
            return _parse_critic_verdict(raw)

        # Stage 1: drafter synthesises
        skill_md_text = _synthesise()
        if not skill_md_text or len(skill_md_text) < 50:
            return _try_tier2_legacy(task, gap, config, _approved=True)

        # Stage 2: critic evaluates
        verdict = _critique(skill_md_text)

        # One retry if critic rejects
        if verdict["verdict"] == "FAIL" and verdict["score"] < 0.6:
            issues_str = "; ".join(verdict.get("issues", []))
            skill_md_text = _synthesise(extra_context=issues_str)
            if skill_md_text and len(skill_md_text) > 50:
                verdict = _critique(skill_md_text)

        if verdict["verdict"] == "FAIL":
            # Both attempts failed — log and return gap
            try:
                from core.evolution.evolution_hook import _create_gap_todo_standalone
                _create_gap_todo_standalone(task)
            except Exception:
                pass
            return EvolutionResult(
                found=False, confidence="LOW", escalated=True,
                gap=f"Critic rejected synthesis (score={verdict['score']:.2f}): {'; '.join(verdict.get('issues', []))}",
            )

        # Passed — use suggested name if critic provided a better one
        import re, yaml
        skill_name = verdict.get("suggested_name", "").strip()
        if not skill_name:
            m = re.match(r'^---\n(.*?)\n---', skill_md_text, re.DOTALL)
            if m:
                try:
                    fm = yaml.safe_load(m.group(1)) or {}
                    skill_name = fm.get("name", "")
                except Exception:
                    pass
        if not skill_name:
            skill_name = task.lower().replace(" ", "-")[:40]

        # Write to ecosystem private skills
        from core.evolution.code_synthesizer import CodeSynthesizer
        synth = CodeSynthesizer(config=config)
        skill_dir = Path(synth.private_skills) / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md_text)

        # Trajectory logging
        try:
            from core.evolution.evolution_log import EvolutionLog
            EvolutionLog().log_synthesis_trajectory(
                task=task, gap=gap,
                prompt=_SYNTH_PROMPT.format(task=gap),
                output_skill=skill_md_text,
                output_script="",
                validation=f"PASS (critic score={verdict['score']:.2f})",
                provider="drafter+critic",
                skill_name=skill_name,
                critic_score=verdict.get('score'),
                critic_issues=verdict.get('issues', []),
                suggested_name=verdict.get('suggested_name', ''),
            )
        except Exception:
            pass

        return EvolutionResult(
            found=True, installed=[skill_name],
            confidence="MEDIUM", retry=True, escalated=True,
            provider_used="drafter+critic",
        )

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[ADR-010] pipeline failed, falling back to legacy: {e}")
        return _try_tier2_legacy(task, gap, config, _approved=True)


def _create_gap_todo_standalone(task: str):
    """Create tracker todo for unresolved gap (standalone, no evolver instance)."""
    try:
        import subprocess
        subprocess.run(
            ["python3", "-m", "workspace_tracker", "todo", "add",
             f"Evolution gap: {task[:80]}", "--priority", "medium"],
            capture_output=True, timeout=5
        )
    except Exception:
        pass


def _try_tier2(task: str, gap: str, config: dict, *, chat_id: str | None = None) -> EvolutionResult:
    """ADR-010: Attempt Tier 2 — try critic pipeline first, fallback to legacy."""
    return _try_tier2_pipeline(task, gap, config, chat_id=chat_id)


# Public alias used by tests and ADR docs
_run_evolution_pipeline = _try_tier2_pipeline


def _try_tier2_legacy(task: str, gap: str, config: dict, *, _approved: bool = False, chat_id: str | None = None) -> EvolutionResult:
    """Legacy Tier 2 code synthesis (pre-ADR-010 fallback)."""
    """Attempt Tier 2 code synthesis when Tier 1 found no match."""
    try:
        from core.evolution.code_synthesizer import CodeSynthesizer
        synth = CodeSynthesizer(config=config)
        provider = synth.resolve_provider()

        # ── ADR-021: human-approval gate ─────────────────────────────────────────
        if not _approved:
            gate_result = _gate_tier2(task, gap, provider or 'unknown', config, chat_id)
            if gate_result is not None:
                return gate_result
        # ── end ADR-021 gate ─────────────────────────────────────────────────────

        if not provider:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=None,
                gap=gap,
            )

        tmpdir, actual_provider = synth.synthesize_with_fallback(task, gap, provider)
        if not tmpdir:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=provider,
                gap=f"All providers exhausted (primary: {provider})",
            )
        # Use the provider that actually responded
        provider = actual_provider or provider

        passed, reason = synth.validate_synthesis(tmpdir)
        if not passed:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=provider,
                gap=f"Synthesis validation failed: {reason}",
            )

        # Derive skill name from SKILL.md if possible
        import re
        import yaml
        skill_name = task.lower().replace(" ", "-")[:40]
        skill_md = tmpdir / "SKILL.md"
        if skill_md.exists():
            text = skill_md.read_text(errors="replace")
            m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if m:
                try:
                    fm = yaml.safe_load(m.group(1)) or {}
                    skill_name = fm.get("name", skill_name)
                except Exception:
                    pass

        installed = synth.install(tmpdir, skill_name)
        if not installed:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=provider,
                gap=f"Failed to install synthesized skill: {skill_name}",
            )

        # ── Trajectory logging for fine-tune dataset ──────────────────────────────
        try:
            from core.evolution.evolution_log import EvolutionLog
            from pathlib import Path as _Path
            _skill_md_path = _Path(synth.private_skills) / skill_name / "SKILL.md"
            _skill_md_txt  = _skill_md_path.read_text(errors="replace") if _skill_md_path.exists() else ""
            # Grab first script content
            _scripts_dir = _Path(synth.private_skills) / skill_name / "scripts"
            _script_txt  = ""
            if _scripts_dir.exists():
                _scripts = list(_scripts_dir.iterdir())
                if _scripts:
                    _script_txt = _scripts[0].read_text(errors="replace")[:8000]
            EvolutionLog().log_synthesis_trajectory(
                task=task, gap=gap,
                prompt=f"Write a Kernel skill for: {gap}",  # reconstructed
                output_skill=_skill_md_txt,
                output_script=_script_txt,
                validation="PASS",
                provider=provider,
                skill_name=skill_name,
            )
        except Exception:
            pass  # trajectory logging never breaks the agent

        return EvolutionResult(
            found=True,
            installed=[skill_name],
            confidence="MEDIUM",
            retry=True,
            escalated=True,
            provider_used=provider,
        )

    except Exception as e:
        return EvolutionResult(
            found=False,
            confidence="LOW",
            escalated=True,
            gap=f"Tier 2 synthesis error: {e}",
        )
