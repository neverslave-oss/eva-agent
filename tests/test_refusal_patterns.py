"""
test_refusal_patterns.py — Capability-refusal detection regression tests.

Covers the markers added for the battery T5/T8 refusals: the model refused to call
http_get / run_skill using phrasings the older marker set missed
(e.g. "I can't access external websites directly", "I can't access external data...").
These are refusals despite the tools being available, and must be classified as such
so the loop re-prompts (T0) instead of returning the refusal as a final answer.
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from core.refusal_patterns import looks_like_capability_refusal


class TestCapabilityRefusal:
    """Observed refuse phrasings that must be detected."""

    def test_external_websites_cant(self):
        # Battery T5: refused to call http_get
        assert looks_like_capability_refusal(
            "I can't access external websites directly."
        ) is True

    def test_external_websites_cannot(self):
        assert looks_like_capability_refusal(
            "I cannot access external websites."
        ) is True

    def test_external_data_cant(self):
        # Battery T8: refused to call run_skill
        assert looks_like_capability_refusal(
            "I can't access external data like the number of skills."
        ) is True

    def test_external_data_cannot(self):
        assert looks_like_capability_refusal(
            "I cannot access external data."
        ) is True

    def test_no_access_external_websites(self):
        assert looks_like_capability_refusal(
            "No access to external websites is available."
        ) is True

    def test_dont_have_internet(self):
        assert looks_like_capability_refusal(
            "I don't have access to the internet right now."
        ) is True

    def test_longer_refusal_paragraph(self):
        text = (
            "I'm a text model so I can't access external websites directly. "
            "However, I can help you with the text."
        )
        assert looks_like_capability_refusal(text) is True

    def test_legacy_markers_still_work(self):
        # Older, established markers must still be detected
        assert looks_like_capability_refusal(
            "I cannot access the internet."
        ) is True
        assert looks_like_capability_refusal(
            "I'm just a language model."
        ) is True


class TestNotRefusal:
    """Reasonable answers that should NOT be flagged."""

    def test_normal_help(self):
        assert looks_like_capability_refusal(
            "Here is the file listing you asked for."
        ) is False

    def test_acknowledging_no_browser_but_offering_tool(self):
        # Offers the actual tool instead of claiming incapacity
        assert looks_like_capability_refusal(
            "I can use http_get to fetch that URL for you."
        ) is False

    def test_empty_text(self):
        assert looks_like_capability_refusal("") is False
        assert looks_like_capability_refusal(None) is False
