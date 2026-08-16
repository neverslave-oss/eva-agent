"""Tests for ADR-006 capability_verifier and related evolver changes."""
import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_verify_returns_true_on_yes_response():
    from core.evolution.capability_verifier import verify
    skill = {"name": "github", "description": "GitHub CLI skill"}
    result, reason = verify(skill, "monitor GitHub issues", lambda p, **kw: "YES it can handle this")
    assert result is True


def test_verify_returns_false_on_no_response():
    from core.evolution.capability_verifier import verify
    skill = {"name": "kernel-doc-retrieval", "description": "document retrieval"}
    result, reason = verify(skill, "transcribe audio recording", lambda p, **kw: "NO this skill handles documents not audio")
    assert result is False


def test_verify_fails_open_on_inference_error():
    from core.evolution.capability_verifier import verify
    def bad_infer(p, **kw): raise RuntimeError("model unavailable")
    skill = {"name": "test", "description": "test skill"}
    result, reason = verify(skill, "any task", bad_infer)
    assert result is True  # fail-open


def test_verify_prompt_contains_skill_name_and_task():
    from core.evolution.capability_verifier import verify, VERIFY_PROMPT
    captured = []
    def capture_infer(p, **kw):
        captured.append(p)
        return "YES"
    skill = {"name": "my-skill", "description": "does stuff"}
    verify(skill, "do the thing", capture_infer)
    assert "my-skill" in captured[0]
    assert "do the thing" in captured[0]


def _make_skill_dir(tmpdir, name, description="a skill", repo="https://github.com/fabiopacifici-bot/x"):
    skill_dir = Path(tmpdir) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nrepo: {repo}\n---\n"
    )
    return skill_dir


def test_evolver_skips_verification_when_infer_fn_is_none():
    """When infer_fn is None, verification is skipped and first match is used."""
    tmpdir = tempfile.mkdtemp()
    try:
        _make_skill_dir(tmpdir, "github", description="GitHub skill",
                        repo="https://github.com/fabiopacifici-bot/github")
        from core.evolution.evolver import Evolver
        ev = Evolver(config={"private_skills_dir": tmpdir + "/private"}, skills_dir=tmpdir, infer_fn=None)
        with patch.object(ev, "acquire", return_value=True), \
             patch.object(ev, "validate", return_value="HIGH"), \
             patch.object(ev, "_fetch_remote_skills", return_value=[]):
            result = ev.run("manage github issues")
        assert result.found is True
        assert result.verification_result is None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_evolver_escalates_when_all_candidates_fail():
    """When infer_fn rejects all candidates, result should escalate."""
    tmpdir = tempfile.mkdtemp()
    try:
        for name in ["skill-a", "skill-b", "skill-c"]:
            _make_skill_dir(tmpdir, name, description=f"{name} description",
                            repo=f"https://github.com/fabiopacifici-bot/{name}")
        from core.evolution.evolver import Evolver
        def always_no(prompt, **kw):
            return "NO this skill cannot handle the task"
        ev = Evolver(config={"private_skills_dir": tmpdir + "/private"}, skills_dir=tmpdir, infer_fn=always_no)
        with patch.object(ev, "_fetch_remote_skills", return_value=[]):
            # ensure all 3 local skills get a decent score above min_candidate_score
            with patch.object(ev, "_combined_score", return_value=0.9):
                result = ev.run("do something unusual")
        assert result.escalated is True
        assert result.found is False
        assert result.verification_result == "NO"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
