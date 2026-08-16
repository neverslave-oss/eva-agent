"""
test_evolver.py — ADR-004 comprehensive tests for evolver.py, code_synthesizer.py,
evolution_log.py, and evolution_hook.py.

All external calls (git, model inference, HTTP, tracker.py) are mocked.
Run with: python3 -m pytest tests/test_evolver.py -v
"""
import os
import sys
import json
import shutil
import tempfile
import textwrap
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Ensure src/ is on path
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

# Mock torch before any import of agent/model (torch not installed in test container)
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.cuda.mem_get_info.return_value = (4096 * 1024 * 1024, 8192 * 1024 * 1024)
_mock_torch.bfloat16 = "bfloat16"
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("accelerate", MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_skill_md(tmpdir: Path, name: str, description: str = "A skill",
                  commands=None, repo: str = "", exec_field: str = None,
                  bad_yaml: bool = False, no_frontmatter: bool = False) -> Path:
    """Write a SKILL.md into a subdirectory of tmpdir."""
    skill_dir = tmpdir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"

    if no_frontmatter:
        skill_path.write_text("No frontmatter here at all.\n")
        return skill_path

    if bad_yaml:
        skill_path.write_text("---\n: bad: yaml: {{{{ \n---\nContent\n")
        return skill_path

    cmds = commands or [f"/{name}"]
    fm_lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"commands:",
    ]
    for c in cmds:
        fm_lines.append(f"  - {c}")
    if repo:
        fm_lines.append(f"repo: {repo}")
    if exec_field:
        fm_lines.append(f"exec: {exec_field}")
    fm_lines.append("---")
    fm_lines.append(f"\n# {name}\n\nInstructions go here.\n")
    skill_path.write_text("\n".join(fm_lines))
    return skill_path


def make_config(skills_dir: str) -> dict:
    return {
        "skills_dir": skills_dir,
        "private_skills_dir": str(Path(skills_dir) / "private" / "skills"),
        # Use no-op backend in unit tests to avoid loading the real model
        "embedding_backend": "none",
        # Lower threshold so keyword-only scoring (0.0 semantic fallback) passes
        "evolution": {"min_candidate_score": 0.0},
    }


# ===========================================================================
# TestEcosystemSearch
# ===========================================================================

class TestEcosystemSearch:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _evolver(self):
        from core.evolution.evolver import Evolver
        return Evolver(config=make_config(str(self.tmpdir)), skills_dir=str(self.tmpdir))

    def test_search_finds_skill_by_name_match(self):
        make_skill_md(self.tmpdir, "weather", description="Shows weather forecasts",
                      repo="https://github.com/fabiopacifici-bot/weather")
        ev = self._evolver()
        results = ev.search_ecosystem("check weather forecast")
        assert len(results) >= 1
        assert results[0]["name"] == "weather"

    def test_search_finds_skill_by_description_keyword(self):
        make_skill_md(self.tmpdir, "translator", description="translate text between languages",
                      repo="https://github.com/fabiopacifici-bot/translator")
        ev = self._evolver()
        results = ev.search_ecosystem("translate this document")
        assert len(results) >= 1
        assert any(r["name"] == "translator" for r in results)

    def test_search_finds_skill_by_command_trigger(self):
        make_skill_md(self.tmpdir, "search-web", description="Browse the internet",
                      commands=["/search", "/browse"],
                      repo="https://github.com/fabiopacifici-bot/search-web")
        ev = self._evolver()
        results = ev.search_ecosystem("search for Python tutorials")
        assert len(results) >= 1
        assert any(r["name"] == "search-web" for r in results)

    def test_search_returns_empty_when_no_match(self):
        make_skill_md(self.tmpdir, "calendar", description="Manage calendar events")
        ev = self._evolver()
        # Set threshold impossibly high so no keyword-only match passes
        ev._min_candidate_score = 1.0
        results = ev.search_ecosystem("zzz_totally_unrelated_xyzabc")
        assert results == []

    def test_search_handles_malformed_skill_md_gracefully(self):
        # One bad skill, one good skill
        make_skill_md(self.tmpdir, "good-skill", description="useful tool for testing",
                      repo="https://github.com/fabiopacifici-bot/good-skill")
        make_skill_md(self.tmpdir, "bad-skill", bad_yaml=True)
        ev = self._evolver()
        # Should not raise; bad skill is silently skipped
        results = ev.search_ecosystem("testing tool")
        assert all(r["name"] != "bad-skill" for r in results)

    def test_search_handles_no_frontmatter_gracefully(self):
        make_skill_md(self.tmpdir, "no-fm", no_frontmatter=True)
        make_skill_md(self.tmpdir, "real-skill", description="a real skill that works")
        ev = self._evolver()
        results = ev.search_ecosystem("real skill")
        assert all(r["name"] != "no-fm" for r in results)


# ===========================================================================
# TestAcquire
# ===========================================================================

class TestAcquire:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _evolver(self):
        from core.evolution.evolver import Evolver
        return Evolver(config=make_config(str(self.tmpdir)), skills_dir=str(self.tmpdir))

    @patch("core.evolution.evolver.subprocess.run")
    def test_acquire_clones_allowlisted_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        ev = self._evolver()
        result = ev.acquire({
            "name": "weather",
            "repo": "https://github.com/fabiopacifici-bot/weather",
        })
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "git" in args
        assert "clone" in args

    @patch("core.evolution.evolver.subprocess.run")
    def test_acquire_rejects_non_allowlisted_repo(self, mock_run):
        ev = self._evolver()
        result = ev.acquire({
            "name": "evil-skill",
            "repo": "https://github.com/hacker/evil-skill",
        })
        assert result is False
        mock_run.assert_not_called()

    @patch("core.evolution.evolver.subprocess.run")
    def test_acquire_is_idempotent_no_second_clone(self, mock_run):
        ev = self._evolver()
        dest = Path(ev._private_skills) / "weather"
        dest.mkdir(parents=True)
        result = ev.acquire({
            "name": "weather",
            "repo": "https://github.com/fabiopacifici-bot/weather",
        })
        assert result is True
        mock_run.assert_not_called()  # No clone since dir exists

    @patch("core.evolution.evolver.subprocess.run")
    def test_acquire_returns_false_on_git_clone_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        ev = self._evolver()
        result = ev.acquire({
            "name": "weather",
            "repo": "https://github.com/fabiopacifici-bot/weather",
        })
        assert result is False

    @patch("core.evolution.evolver.subprocess.run")
    def test_acquire_handles_missing_repo_gracefully(self, mock_run):
        mock_run.side_effect = Exception("network error")
        ev = self._evolver()
        result = ev.acquire({
            "name": "weather",
            "repo": "https://github.com/fabiopacifici-bot/weather",
        })
        assert result is False


# ===========================================================================
# TestValidate
# ===========================================================================

class TestValidate:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _evolver(self):
        from core.evolution.evolver import Evolver
        return Evolver(config=make_config(str(self.tmpdir)), skills_dir=str(self.tmpdir))

    def test_valid_skill_with_exec_returns_high(self):
        ev = self._evolver()
        private = Path(ev._private_skills) / "my-skill"
        private.mkdir(parents=True)
        # Write SKILL.md with exec field pointing to a real file
        script = private / "scripts" / "main.py"
        script.parent.mkdir(parents=True)
        script.write_text("print('hello')\n")
        (private / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: test\nexec: scripts/main.py\n---\nContent\n"
        )
        assert ev.validate("my-skill") == "HIGH"

    def test_valid_skill_without_exec_returns_medium(self):
        ev = self._evolver()
        private = Path(ev._private_skills) / "no-exec-skill"
        private.mkdir(parents=True)
        (private / "SKILL.md").write_text(
            "---\nname: no-exec-skill\ndescription: test\n---\nContent\n"
        )
        assert ev.validate("no-exec-skill") == "MEDIUM"

    def test_skill_with_missing_scripts_returns_low_if_exec_declared(self):
        ev = self._evolver()
        private = Path(ev._private_skills) / "broken-skill"
        private.mkdir(parents=True)
        (private / "SKILL.md").write_text(
            "---\nname: broken-skill\ndescription: test\nexec: scripts/missing.py\n---\nContent\n"
        )
        # exec declared but file doesn't exist
        assert ev.validate("broken-skill") == "LOW"

    def test_skill_with_unparseable_skill_md_returns_low(self):
        ev = self._evolver()
        private = Path(ev._private_skills) / "bad-fm"
        private.mkdir(parents=True)
        (private / "SKILL.md").write_text("no frontmatter at all\n")
        assert ev.validate("bad-fm") == "LOW"


# ===========================================================================
# TestEvolutionResult (run() integration)
# ===========================================================================

class TestEvolutionResult:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _evolver(self):
        from core.evolution.evolver import Evolver
        return Evolver(config=make_config(str(self.tmpdir)), skills_dir=str(self.tmpdir))

    @patch("core.evolution.evolver.subprocess.run")
    def test_run_returns_found_true_when_skill_acquired_successfully(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        make_skill_md(
            self.tmpdir, "weather",
            description="Shows weather forecasts",
            repo="https://github.com/fabiopacifici-bot/weather",
        )
        # After clone, plant a valid SKILL.md in private/skills/weather
        ev = self._evolver()

        def fake_clone(*args, **kwargs):
            dest = Path(ev._private_skills) / "weather"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "SKILL.md").write_text(
                "---\nname: weather\ndescription: test\n---\nContent\n"
            )
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_clone
        result = ev.run("check the weather today")
        assert result.found is True
        assert "weather" in result.installed
        assert result.retry is True

    @patch("core.evolution.evolver.subprocess.run")
    def test_run_returns_escalated_true_when_no_ecosystem_match(self, mock_run):
        ev = self._evolver()
        result = ev.run("do something completely unknown xyz123")
        assert result.found is False
        assert result.escalated is True
        mock_run.assert_not_called()

    @patch("core.evolution.evolver.subprocess.run")
    def test_run_creates_tracker_todo_when_gap_unresolved(self, mock_run):
        """When clone succeeds but validation returns LOW, todo should be created."""
        make_skill_md(
            self.tmpdir, "bad-install-skill",
            description="something partially installed",
            repo="https://github.com/fabiopacifici-bot/bad-install-skill",
        )
        ev = self._evolver()

        def fake_clone_bad(*args, **kwargs):
            # Clone succeeds (returncode=0) but writes no SKILL.md → validate → LOW
            dest = Path(ev._private_skills) / "bad-install-skill"
            dest.mkdir(parents=True, exist_ok=True)
            # Intentionally write an unparseable SKILL.md
            (dest / "SKILL.md").write_text("no frontmatter\n")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_clone_bad
        with patch.object(ev, "_create_gap_todo") as mock_todo:
            result = ev.run("bad-install-skill task")
            # validate() should return LOW → retry=False → todo called
            assert result.found is True
            assert result.retry is False
            mock_todo.assert_called_once()


# ===========================================================================
# TestCodeSynthesizer
# ===========================================================================

class TestCodeSynthesizer:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _synth(self):
        from core.evolution.code_synthesizer import CodeSynthesizer
        return CodeSynthesizer(config={
            "private_skills_dir": str(self.tmpdir / "private" / "skills"),
        })

    def test_resolve_provider_returns_olly_when_openclaw_url_healthy(self):
        synth = self._synth()
        mock_resp = MagicMock()
        mock_resp.status = 200
        with patch.dict(os.environ, {"OPENCLAW_URL": "http://localhost:9999"}, clear=False):
            with patch("core.evolution.code_synthesizer.urllib.request.urlopen", return_value=mock_resp):
                result = synth.resolve_provider()
        assert result == "olly"

    def test_resolve_provider_returns_openai_when_key_set_no_olly(self):
        synth = self._synth()
        env = {"OPENAI_API_KEY": "sk-test", "OPENCLAW_URL": ""}
        with patch.dict(os.environ, env, clear=False):
            # OPENCLAW_URL is empty so olly check is skipped
            result = synth.resolve_provider()
        assert result == "openai"

    def test_resolve_provider_returns_none_when_nothing_available(self):
        synth = self._synth()
        env = {
            "OPENCLAW_URL": "",
            "OPENAI_API_KEY": "",
            "TMP_OPEN_AI_API_KEY": "",  # also clear the alias
            "ANTHROPIC_API_KEY": "",
            "HF_HOME": str(self.tmpdir / "empty_hf"),
        }
        with patch.dict(os.environ, env, clear=False):
            result = synth.resolve_provider()
        assert result is None

    def test_validate_synthesis_passes_on_valid_skill_md(self):
        synth = self._synth()
        skill_dir = self.tmpdir / "test-skill"
        skill_dir.mkdir()
        script = skill_dir / "scripts" / "main.py"
        script.parent.mkdir()
        script.write_text("print('hello')\n")
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: a test\nexec: scripts/main.py\n---\nContent\n"
        )
        passed, reason = synth.validate_synthesis(skill_dir)
        assert passed is True
        assert reason == "OK"

    def test_validate_synthesis_fails_on_missing_frontmatter(self):
        synth = self._synth()
        skill_dir = self.tmpdir / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("no frontmatter here\n")
        passed, reason = synth.validate_synthesis(skill_dir)
        assert passed is False
        assert "frontmatter" in reason.lower()

    def test_validate_synthesis_fails_on_missing_skill_md(self):
        synth = self._synth()
        skill_dir = self.tmpdir / "empty-skill"
        skill_dir.mkdir()
        passed, reason = synth.validate_synthesis(skill_dir)
        assert passed is False
        assert "SKILL.md" in reason


# ===========================================================================
# TestEvolutionLog
# ===========================================================================

class TestEvolutionLog:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = str(self.tmpdir / "evolution.db")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _log(self):
        from core.evolution.evolution_log import EvolutionLog
        return EvolutionLog(db_path=self.db_path)

    def _make_result(self, found=True, gap="", escalated=False, retry=True):
        from core.evolution.evolver import EvolutionResult
        return EvolutionResult(
            found=found,
            installed=["skill-a"] if found else [],
            confidence="HIGH" if found else "LOW",
            retry=retry,
            escalated=escalated,
            provider_used=None,
            gap=gap,
        )

    def test_record_saves_to_sqlite(self):
        log = self._log()
        result = self._make_result(found=True)
        row_id = log.record("test task", result)
        assert isinstance(row_id, int)
        assert row_id > 0
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM evolution_events").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_get_gaps_returns_only_unresolved_entries(self):
        log = self._log()
        log.record("resolved task", self._make_result(found=True, retry=True))
        log.record("unresolved task", self._make_result(found=False, gap="can't do it", retry=False))
        gaps = log.get_gaps()
        assert len(gaps) == 1
        assert "unresolved task" in gaps[0]["task"]

    def test_get_history_returns_correct_limit(self):
        log = self._log()
        for i in range(10):
            log.record(f"task {i}", self._make_result(found=True))
        history = log.get_history(limit=5)
        assert len(history) == 5

    @patch("core.evolution.evolution_log.subprocess.run")
    def test_create_tracker_todos_for_gaps_calls_tracker_for_each_gap(self, mock_run):
        # Create a fake tracker script
        tracker_path = self.tmpdir / "tracker.py"
        tracker_path.write_text("# fake tracker\n")
        mock_run.return_value = MagicMock(returncode=0)

        log = self._log()
        log.record("gap 1", self._make_result(found=False, gap="gap one", retry=False))
        log.record("gap 2", self._make_result(found=False, gap="gap two", retry=False))
        log.record("resolved", self._make_result(found=True, retry=True))

        with patch("core.evolution.evolution_log.os.path.exists", return_value=True):
            with patch("core.evolution.evolution_log.os.path.expanduser", return_value=str(tracker_path)):
                log.create_tracker_todos_for_gaps()

        assert mock_run.call_count == 2


# ===========================================================================
# TestEvolutionHook
# ===========================================================================

class TestEvolutionHook:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_maybe_evolve_returns_none_when_disabled(self):
        import importlib
        import core.evolution.evolution_hook as evolution_hook
        with patch.object(evolution_hook, "EVOLUTION_ENABLED", False):
            result = evolution_hook.maybe_evolve(
                "do something",
                config=make_config(str(self.tmpdir)),
                skills_dir=str(self.tmpdir),
            )
        assert result is None

    def test_maybe_evolve_calls_evolver_when_enabled(self):
        import core.evolution.evolution_hook as evolution_hook
        import core.evolution.evolution_state as _evo_state
        from core.evolution.evolver import EvolutionResult
        mock_result = EvolutionResult(found=True, installed=["skill-x"],
                                      confidence="HIGH", retry=True)
        with patch.object(evolution_hook, "EVOLUTION_ENABLED", True):
            with patch.object(_evo_state, "should_evolve", return_value=True):
                with patch("core.evolution.evolution_hook.Evolver") as MockEvolver:
                    instance = MockEvolver.return_value
                    instance.run.return_value = mock_result
                    with patch("core.evolution.evolution_hook.EvolutionLog") as MockLog:
                        MockLog.return_value.record.return_value = 1
                        result = evolution_hook.maybe_evolve(
                            "do something",
                            config=make_config(str(self.tmpdir)),
                            skills_dir=str(self.tmpdir),
                        )
        assert result is not None
        assert result.found is True
        instance.run.assert_called_once_with("do something")


# ===========================================================================
# TestGoalDiscovery
# ===========================================================================

class TestGoalDiscovery:
    def setup_method(self):
        import core.pipelines.goal_discovery as gd
        # Reset module-level state before each test
        with gd._log_lock:
            gd._interaction_log.clear()

    def test_record_unhandled_adds_to_log(self):
        import core.pipelines.goal_discovery as gd
        gd.record_unhandled("how do I check the weather")
        with gd._log_lock:
            assert len(gd._interaction_log) == 1
            assert "weather" in gd._interaction_log[0]

    def test_extract_intent_keywords_strips_stopwords(self):
        import core.pipelines.goal_discovery as gd
        result = gd._extract_intent_keywords("can you please help me translate this document")
        assert "translate" in result
        assert "can" not in result
        assert "please" not in result
        assert "help" not in result

    def test_discover_patterns_returns_recurring(self):
        import core.pipelines.goal_discovery as gd
        # Add same intent 3+ times
        for _ in range(3):
            gd.record_unhandled("translate this document now")
        patterns = gd._discover_patterns()
        assert len(patterns) >= 1
        assert any("translate" in p for p in patterns)

    def test_discover_patterns_ignores_below_threshold(self):
        import core.pipelines.goal_discovery as gd
        # Only 2 hits — below default threshold of 3
        for _ in range(2):
            gd.record_unhandled("scan the network ports")
        patterns = gd._discover_patterns()
        assert not any("scan" in p for p in patterns)

    def test_remote_fetch_returns_empty_on_network_error(self):
        import tempfile, shutil
        from core.evolution.evolver import Evolver
        tmpdir = tempfile.mkdtemp()
        try:
            ev = Evolver(config=make_config(tmpdir), skills_dir=tmpdir)
            with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
                result = ev._fetch_remote_skills()
            assert result == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# TestRemoteEcosystemFetch
# ===========================================================================

class TestRemoteEcosystemFetch:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _evolver(self):
        from core.evolution.evolver import Evolver
        return Evolver(config=make_config(str(self.tmpdir)), skills_dir=str(self.tmpdir))

    def test_fetch_remote_skills_returns_empty_list_on_network_error(self):
        import core.evolution.evolver as ev_mod
        ev = self._evolver()
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = ev._fetch_remote_skills()
        assert result == []

    def test_search_ecosystem_merges_remote_and_local(self):
        """Local skills are still returned even when remote fetch fails."""
        make_skill_md(self.tmpdir, "local-skill", description="local useful tool",
                      repo="https://github.com/fabiopacifici-bot/local-skill")
        ev = self._evolver()
        with patch.object(ev, "_fetch_remote_skills", return_value=[]):
            results = ev.search_ecosystem("local tool")
        assert any(r["name"] == "local-skill" for r in results)
