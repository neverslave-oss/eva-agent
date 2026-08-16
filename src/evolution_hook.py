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
from evolver import Evolver, EvolutionResult
from evolution_log import EvolutionLog
import evolution_state as _evo_state

EVOLUTION_ENABLED = os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true"


def maybe_evolve(task: str, config: dict, skills_dir: str) -> EvolutionResult | None:
    """
    Called when agent.triage() finds no skill/routine match.
    Respects the evolution state machine: only runs when state=RUNNING
    and iteration cap has not been reached.
    """
    if not EVOLUTION_ENABLED:
        return None

    if not _evo_state.should_evolve():
        return None  # paused, stopped, or cap reached

    evolver = Evolver(config=config, skills_dir=skills_dir)
    result = evolver.run(task)

    # Persist to evolution log
    try:
        log = EvolutionLog()
        log.record(task, result)
    except Exception:
        pass

    # If Tier 1 escalated and Tier 2 synthesis is available, attempt it
    if result.escalated:
        result = _try_tier2(task, result.gap, config)

    # Count against cap only when a new skill was actually installed
    if result.found and result.installed:
        _evo_state.increment()

    return result


def _try_tier2(task: str, gap: str, config: dict) -> EvolutionResult:
    """Attempt Tier 2 code synthesis when Tier 1 found no match."""
    try:
        from code_synthesizer import CodeSynthesizer
        synth = CodeSynthesizer(config=config)
        provider = synth.resolve_provider()

        if not provider:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=None,
                gap=gap,
            )

        tmpdir = synth.synthesize(task, gap, provider)
        if not tmpdir:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                provider_used=provider,
                gap=gap,
            )

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
            from evolution_log import EvolutionLog
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
