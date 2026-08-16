"""
tests/test_provider_fallback.py — ADR-021 addendum: Tier 2 provider fallback chain.
"""
import sys
import os
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.evolution.code_synthesizer import CodeSynthesizer, SynthesisProviderError, _PROVIDER_CHAIN

_VALID_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit tests.
---

## Instructions
Do the thing.
"""


def _make_synth(tmp_dir: str) -> CodeSynthesizer:
    return CodeSynthesizer(config={"private_skills_dir": tmp_dir})


class TestSynthesisProviderError(unittest.TestCase):
    def test_is_exception(self):
        e = SynthesisProviderError("openai", 429, "quota exceeded")
        self.assertIsInstance(e, Exception)
        self.assertEqual(e.provider, "openai")
        self.assertEqual(e.status_code, 429)


class TestProviderAvailable(unittest.TestCase):
    def setUp(self):
        self.synth = _make_synth(tempfile.mkdtemp())

    def test_openai_available_with_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            self.assertTrue(self.synth._provider_available("openai"))

    def test_openai_unavailable_without_key(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "TMP_OPEN_AI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(self.synth._provider_available("openai"))

    def test_anthropic_available_with_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"}):
            self.assertTrue(self.synth._provider_available("anthropic"))

    def test_local_always_available(self):
        # local is always available as last resort
        self.assertTrue(self.synth._provider_available("local"))

    def test_olly_available_with_url(self):
        with patch.dict(os.environ, {"OPENCLAW_URL": "http://localhost:18789"}):
            self.assertTrue(self.synth._provider_available("olly"))

    def test_olly_unavailable_without_url(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENCLAW_URL"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(self.synth._provider_available("olly"))


class TestSynthesizeWithFallback(unittest.TestCase):
    """Tests for the provider fallback chain in synthesize_with_fallback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.synth = _make_synth(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mock_call_provider(self, success_on: str, fail_codes: dict | None = None):
        """Return a side_effect for _call_provider.
        success_on: provider name that returns valid skill text.
        fail_codes: {provider: status_code} for providers that should raise SynthesisProviderError.
        """
        fail_codes = fail_codes or {}
        def side_effect(provider, prompt):
            if provider in fail_codes:
                raise SynthesisProviderError(provider, fail_codes[provider], "error")
            if provider == success_on:
                return _VALID_SKILL_MD
            return None
        return side_effect

    def test_primary_provider_succeeds(self):
        with patch.object(self.synth, '_call_provider', side_effect=self._mock_call_provider("openai")), \
             patch.object(self.synth, '_provider_available', return_value=True):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "openai")
            self.assertIsNotNone(tmpdir)
            self.assertEqual(provider, "openai")
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_falls_back_on_429(self):
        """When primary raises 429, should fall back to anthropic and succeed."""
        with patch.object(self.synth, '_call_provider',
                          side_effect=self._mock_call_provider("anthropic", {"openai": 429})), \
             patch.object(self.synth, '_provider_available', return_value=True):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "openai")
            self.assertIsNotNone(tmpdir)
            self.assertEqual(provider, "anthropic")
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_falls_back_on_401(self):
        """Invalid key (401) should trigger fallback."""
        with patch.object(self.synth, '_call_provider',
                          side_effect=self._mock_call_provider("local", {"openai": 401, "anthropic": 401})), \
             patch.object(self.synth, '_provider_available', return_value=True):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "openai")
            self.assertIsNotNone(tmpdir)
            self.assertEqual(provider, "local")
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_all_providers_fail_returns_none(self):
        """When every provider fails, returns (None, '')."""
        def all_fail(provider, prompt):
            raise SynthesisProviderError(provider, 429, "quota")
        with patch.object(self.synth, '_call_provider', side_effect=all_fail), \
             patch.object(self.synth, '_provider_available', return_value=True):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "openai")
            self.assertIsNone(tmpdir)
            self.assertEqual(provider, "")

    def test_unavailable_providers_skipped(self):
        """Providers without credentials are skipped, not called."""
        call_log = []
        def side_effect(provider, prompt):
            call_log.append(provider)
            if provider == "local":
                return _VALID_SKILL_MD
            raise SynthesisProviderError(provider, 429, "quota")

        def available(provider):
            return provider in ("openai", "local")

        with patch.object(self.synth, '_call_provider', side_effect=side_effect), \
             patch.object(self.synth, '_provider_available', side_effect=available):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "openai")
            self.assertEqual(provider, "local")
            self.assertNotIn("anthropic", call_log)
            self.assertNotIn("olly", call_log)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_primary_is_first_in_chain(self):
        """Non-default primary provider is tried first even if not first in _PROVIDER_CHAIN."""
        call_order = []
        def side_effect(provider, prompt):
            call_order.append(provider)
            if provider == "anthropic":
                return _VALID_SKILL_MD
            raise SynthesisProviderError(provider, 429, "quota")

        with patch.object(self.synth, '_call_provider', side_effect=side_effect), \
             patch.object(self.synth, '_provider_available', return_value=True):
            tmpdir, provider = self.synth.synthesize_with_fallback("task", "gap", "anthropic")
            self.assertEqual(call_order[0], "anthropic")
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
