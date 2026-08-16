"""
tests/test_tier2_approval_gate.py — ADR-021: Tier 2 human-approval gate unit tests.
"""
import sys
import os
import json
import importlib
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPendingSynthesisStore(unittest.TestCase):
    """Tests for pending_synthesis.py store operations."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import core.evolution.pending_synthesis as ps
        self._orig_path = ps._STORE_PATH
        from pathlib import Path
        ps._STORE_PATH = Path(self.tmp) / "core.evolution.pending_synthesis.json"
        self.ps = ps

    def tearDown(self):
        self.ps._STORE_PATH = self._orig_path

    def test_enqueue_returns_id(self):
        sid = self.ps.enqueue("task", "gap", "openai", "12345")
        self.assertIsInstance(sid, str)
        self.assertTrue(len(sid) > 0)

    def test_get_returns_record(self):
        sid = self.ps.enqueue("task", "gap", "openai", "12345")
        record = self.ps.get(sid)
        self.assertIsNotNone(record)
        self.assertEqual(record["task"], "task")
        self.assertEqual(record["provider"], "openai")

    def test_remove_returns_and_deletes(self):
        sid = self.ps.enqueue("task", "gap", "anthropic", "12345")
        record = self.ps.remove(sid)
        self.assertIsNotNone(record)
        self.assertIsNone(self.ps.get(sid))

    def test_list_pending(self):
        self.ps.enqueue("t1", "g1", "openai", "1")
        self.ps.enqueue("t2", "g2", "local", "2")
        items = self.ps.list_pending()
        self.assertEqual(len(items), 2)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.ps.get("nonexistent-id"))


class TestGateTier2(unittest.TestCase):
    """Tests for _gate_tier2 in evolution_hook."""

    def setUp(self):
        # Patch pending_synthesis and telegram_bot to avoid side effects
        self.mock_ps = MagicMock()
        self.mock_ps.enqueue.return_value = "test-synth-id"
        self.mock_tb = MagicMock()

        sys.modules['core.evolution.pending_synthesis'] = self.mock_ps
        sys.modules['pending_synthesis'] = self.mock_ps
        _fake_ch = types.ModuleType('services.channels')
        _fake_ch.telegram_bot = self.mock_tb
        sys.modules['services.channels'] = _fake_ch
        sys.modules['services.channels.telegram_bot'] = self.mock_tb
        sys.modules['telegram_bot'] = self.mock_tb
        # Patch the `services` package itself so that cached real-module attr lookups
        # (services.channels.telegram_bot) also resolve to our mock
        import services as _svc_pkg
        self._orig_svc_channels = getattr(_svc_pkg, 'channels', None)
        _svc_pkg.channels = _fake_ch

        # Reload evolution_hook with mocked deps
        for mod in ['core.evolution.evolution_hook', 'evolution_hook']:
            sys.modules.pop(mod, None)
        # Force core.evolution package to use our mocks
        import core.evolution as _evo_pkg
        _evo_pkg.__dict__.pop('evolution_hook', None)
        _evo_pkg.__dict__['pending_synthesis'] = self.mock_ps
        _evo_pkg.__dict__['telegram_bot'] = self.mock_tb

        # Provide the REAL evolver module so EvolutionResult is a proper dataclass
        import importlib.util, pathlib
        _src = pathlib.Path(__file__).parent.parent / 'src' / 'core' / 'evolution' / 'evolver.py'
        _spec = importlib.util.spec_from_file_location('core.evolution.evolver', str(_src))
        _evolver_real = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_evolver_real)
            sys.modules['core.evolution.evolver'] = _evolver_real
            sys.modules['evolver'] = _evolver_real
        except Exception:
            # Fallback: leave mock in place
            for mod in ['core.evolution.evolver', 'evolver', 'core.evolution.evolution_log', 'evolution_log', 'core.evolution.evolution_state', 'evolution_state']:
                if mod not in sys.modules:
                    sys.modules[mod] = MagicMock()

        for mod in ['core.evolution.evolution_log', 'evolution_log', 'core.evolution.evolution_state', 'evolution_state']:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        import core.evolution.evolution_hook as evolution_hook
        self.eh = evolution_hook

    def tearDown(self):
        for mod in ['core.evolution.pending_synthesis', 'pending_synthesis',
                    'services.channels.telegram_bot', 'telegram_bot',
                    'core.evolution.evolution_hook', 'evolution_hook']:
            sys.modules.pop(mod, None)

    def test_gate_with_chat_id_returns_pending(self):
        """Gate should store the request and return a pending EvolutionResult."""
        result = self.eh._gate_tier2("do X", "X not found", "openai", {}, "12345")
        self.assertIsNotNone(result)
        self.assertTrue(result.pending_approval)
        self.assertEqual(result.synthesis_id, "test-synth-id")
        self.mock_ps.enqueue.assert_called_once()
        self.mock_tb.send_buttons.assert_called_once()

    def test_gate_no_chat_id_no_env_returns_none(self):
        """Without chat_id and env var, gate skips (returns None = proceed)."""
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        result = self.eh._gate_tier2("do X", "gap", "openai", {}, None)
        self.assertIsNone(result)

    def test_gate_local_skip_flag_returns_none(self):
        """When TIER2_APPROVAL_SKIP_LOCAL=true and provider=local, gate skips."""
        with patch.dict(os.environ, {"TIER2_APPROVAL_SKIP_LOCAL": "true"}):
            # Need to reload to pick up env var
            if 'evolution_hook' in sys.modules:
                del sys.modules['evolution_hook']
            import core.evolution.evolution_hook as eh2
            result = eh2._gate_tier2("do X", "gap", "local", {}, None)
            self.assertIsNone(result)

    def test_gate_no_provider_returns_none(self):
        """With no provider available, gate returns None (will fail downstream anyway)."""
        result = self.eh._gate_tier2("do X", "gap", None, {}, "12345")
        self.assertIsNone(result)


class TestApproveRejectCallbacks(unittest.TestCase):
    """Tests for _run_approved_synthesis and _reject_synthesis."""

    def setUp(self):
        self.mock_ps = MagicMock()
        self.mock_tb = MagicMock()
        sys.modules['core.evolution.pending_synthesis'] = self.mock_ps
        sys.modules['pending_synthesis'] = self.mock_ps
        _fake_ch2 = types.ModuleType('services.channels')
        _fake_ch2.telegram_bot = self.mock_tb
        sys.modules['services.channels'] = _fake_ch2
        sys.modules['services.channels.telegram_bot'] = self.mock_tb
        sys.modules['telegram_bot'] = self.mock_tb
        import services as _svc_pkg2
        self._orig_svc_channels2 = getattr(_svc_pkg2, 'channels', None)
        _svc_pkg2.channels = _fake_ch2
        for mod in ['core.evolution.evolver', 'evolver', 'core.evolution.evolution_log', 'evolution_log', 'core.evolution.evolution_state', 'evolution_state']:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()
        for mod in ['core.evolution.evolution_hook', 'evolution_hook']:
            sys.modules.pop(mod, None)
        import core.evolution as _evo_pkg2
        _evo_pkg2.__dict__.pop('evolution_hook', None)
        _evo_pkg2.__dict__['pending_synthesis'] = self.mock_ps
        _evo_pkg2.__dict__['telegram_bot'] = self.mock_tb
        import core.evolution.evolution_hook as evolution_hook
        self.eh = evolution_hook

    def tearDown(self):
        for mod in ['core.evolution.pending_synthesis', 'pending_synthesis',
                    'services.channels.telegram_bot', 'services.channels',
                    'telegram_bot',
                    'core.evolution.evolution_hook', 'evolution_hook']:
            sys.modules.pop(mod, None)
        # Restore services.channels attr on the services package
        import services as _svc_pkg_r
        orig = getattr(self, '_orig_svc_channels2', getattr(self, '_orig_svc_channels', None))
        if orig is not None:
            _svc_pkg_r.channels = orig
        elif hasattr(_svc_pkg_r, 'channels'):
            del _svc_pkg_r.channels

    def test_reject_removes_record_and_logs(self):
        self.mock_ps.remove.return_value = {
            "id": "abc", "task": "t", "gap": "g", "provider": "openai", "chat_id": "1"
        }
        mock_elog = MagicMock()
        with patch('core.evolution.evolution_log.EvolutionLog', return_value=mock_elog):
            self.eh._reject_synthesis("abc")
        self.mock_ps.remove.assert_called_once_with("abc")

    def test_reject_nonexistent_is_noop(self):
        self.mock_ps.remove.return_value = None
        self.eh._reject_synthesis("nonexistent")
        # Should not raise

    def test_approve_nonexistent_is_noop(self):
        self.mock_ps.remove.return_value = None
        self.eh._run_approved_synthesis("nonexistent")
        # Should not raise


if __name__ == "__main__":
    unittest.main()
