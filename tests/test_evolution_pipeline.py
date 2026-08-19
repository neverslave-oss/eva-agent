"""
test_evolution_pipeline.py — ADR-010 tests

Verifies:
- PASS path: critic passes → skill installed
- FAIL → retry path: first fail triggers retry, second PASS installs
- Double FAIL path: both attempts fail → skill NOT installed, gap logged
- Critic verdict parser handles valid JSON, partial JSON, and PASS/FAIL keywords
- replica.pipeline() programmatic API returns results dict
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_parse_critic_verdict_valid_json():
    from core.evolution.evolution_hook import _parse_critic_verdict
    raw = '{"verdict": "PASS", "score": 0.9, "issues": [], "suggested_name": "my-skill"}'
    v = _parse_critic_verdict(raw)
    assert v["verdict"] == "PASS"
    assert v["score"] == 0.9
    assert v["suggested_name"] == "my-skill"


def test_parse_critic_verdict_embedded_json():
    from core.evolution.evolution_hook import _parse_critic_verdict
    raw = 'Here is my evaluation:\n{"verdict": "FAIL", "score": 0.3, "issues": ["bad name"], "suggested_name": ""}\nDone.'
    v = _parse_critic_verdict(raw)
    assert v["verdict"] == "FAIL"
    assert v["score"] == 0.3


def test_parse_critic_verdict_keyword_pass():
    from core.evolution.evolution_hook import _parse_critic_verdict
    raw = "The skill looks good. PASS overall."
    v = _parse_critic_verdict(raw)
    assert v["verdict"] == "PASS"


def test_parse_critic_verdict_keyword_fail():
    from core.evolution.evolution_hook import _parse_critic_verdict
    raw = "This skill fails to address the gap."
    v = _parse_critic_verdict(raw)
    assert v["verdict"] == "FAIL"


def test_evolution_pipeline_pass_path(tmp_path, monkeypatch):
    """Mock model calls return PASS → skill installed."""
    import core.evolution.evolution_hook as eh
    import core.inference.provider as _provider_mod

    # Patch InferenceProvider.infer — pipeline uses _prov.infer(), not model_client directly
    def fake_prov_infer(self, messages, max_new_tokens=1024, call_type="task_inference", **kwargs):
        return "---\nname: test-skill\ndescription: A test skill\ncommands:\n  - /test\ninstructions: |\n  Do the thing.\n---" if call_type == "synthesis" \
            else '{"verdict": "PASS", "score": 0.85, "issues": [], "suggested_name": "test-skill"}'
    monkeypatch.setattr(_provider_mod.InferenceProvider, "infer", fake_prov_infer)

    # Mock CodeSynthesizer
    class FakeSynth:
        private_skills = str(tmp_path / "private" / "skills")
        def __init__(self, config): pass

    import core.evolution.code_synthesizer as _cs_mod
    monkeypatch.setattr(_cs_mod, "CodeSynthesizer", FakeSynth)

    result = eh._run_evolution_pipeline("test task", "test gap", {})
    assert result.found is True
    assert result.retry is True
    # Skill dir should exist
    skill_dir = tmp_path / "private" / "skills" / "test-skill"
    assert skill_dir.exists()


def test_evolution_pipeline_fail_retry(tmp_path, monkeypatch):
    """First critic FAIL triggers retry; second PASS installs."""
    import core.evolution.evolution_hook as eh
    import core.inference.provider as _provider_mod

    call_count = [0]

    def fake_prov_infer(self, messages, max_new_tokens=1024, call_type="task_inference", **kwargs):
        if call_type == "synthesis":
            return "---\nname: retry-skill\ndescription: Retry skill\ncommands:\n  - /retry\ninstructions: |\n  Do retry.\n---"
        call_count[0] += 1
        if call_count[0] == 1:
            return '{"verdict": "FAIL", "score": 0.4, "issues": ["missing instructions"], "suggested_name": ""}'
        return '{"verdict": "PASS", "score": 0.8, "issues": [], "suggested_name": "retry-skill"}'

    monkeypatch.setattr(_provider_mod.InferenceProvider, "infer", fake_prov_infer)

    class FakeSynth:
        private_skills = str(tmp_path / "private" / "skills")
        def __init__(self, config): pass

    import core.evolution.code_synthesizer as _cs_mod
    monkeypatch.setattr(_cs_mod, "CodeSynthesizer", FakeSynth)

    result = eh._run_evolution_pipeline("retry task", "retry gap", {})
    assert result.found is True
    assert call_count[0] == 2  # exactly 2 infer (critique) calls (initial + retry)


def test_evolution_pipeline_double_fail(tmp_path, monkeypatch):
    """Both critic attempts fail → skill NOT installed."""
    import core.evolution.evolution_hook as eh
    import core.inference.provider as _provider_mod

    def fake_prov_infer(self, messages, max_new_tokens=1024, call_type="task_inference", **kwargs):
        if call_type == "synthesis":
            return "---\nname: bad-skill\ndescription: Bad skill.\ncommands:\n  - /bad\ninstructions: |\n  Bad.\n---"
        return '{"verdict": "FAIL", "score": 0.2, "issues": ["incomplete"], "suggested_name": ""}'

    monkeypatch.setattr(_provider_mod.InferenceProvider, "infer", fake_prov_infer)

    class FakeSynth:
        private_skills = str(tmp_path / "private" / "skills")
        def __init__(self, config): pass

    import core.evolution.code_synthesizer as _cs_mod
    monkeypatch.setattr(_cs_mod, "CodeSynthesizer", FakeSynth)

    result = eh._run_evolution_pipeline("bad task", "bad gap", {})
    assert result.found is False
    # No skill directory should have been created
    assert not (tmp_path / "private" / "skills" / "bad-skill").exists()


def test_pipeline_programmatic_api(monkeypatch):
    """core.replica.replica.pipeline() returns results dict without HTTP."""
    import core.replica.replica as replica

    def fake_infer(messages, max_new_tokens=1024, adapter_path=None, **kwargs):
        return "mocked reply"

    # pipeline() routes through the provider (cloud-safe) via _provider_infer,
    # so patch that helper on the replica module to avoid any real inference.
    monkeypatch.setattr(replica, "_provider_infer", fake_infer)

    stages = [
        {"name": "writer", "role": "custom", "brief": "Write stuff", "task": "Do task A"},
        {"name": "reader", "role": "custom", "brief": "Read stuff", "task": "Review the above", "input_from": "writer"},
    ]
    results = replica.pipeline(stages)
    assert "writer" in results
    assert "reader" in results
    assert results["writer"] == "mocked reply"
