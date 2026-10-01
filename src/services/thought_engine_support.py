"""
thought_engine_support.py — support classes + prompts for the Think-at-Rest subsystem.

Extracted from thought_engine.py (ADR-005). Contains the self-contained System 1/2
primitives IdleDetector, ThoughtGenerator, ThoughtEvaluator and their prompt templates.
Kept behavior-identical; thought_engine.py re-exports these names, so all existing
import sites (from services.thought_engine import ...) keep working unchanged.
"""
import json
import logging
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_THOUGHT_SYSTEM_PROMPT = (
    "You are Kernel Evo, a local self-evolving AI agent. You are currently idle.\n"
    "Use idle time productively: think about your user, the world, and your own growth — in that order of priority.\n"
)

_GENERATOR_TEMPLATE = """\
You are Kernel Evo, a local self-evolving AI agent. You are idle.

Your context:
- Skills you haven't used recently: {unused_skills}
- Evolution history (last 5): {evolution_context}

Generate exactly ONE exploratory curiosity thought.
This is NOT about what capabilities you lack — it's about exploring the outside world.

Good examples:
- "I wonder what changed in my favourite open-source repo this week — let me check git log"
- "I should benchmark my speculative decoding speed against last week's baseline"
- "What's the GPU temperature trend over the last 24 hours?"
- "I want to check if there are new releases of my dependencies"

Bad examples (do NOT write):
- "I lack a robust X pipeline" or any gap/inability framing
- "The thought focuses on the gap between X and Y"
- "I should prioritize integrating Y" (that's a gap, not curiosity)

Format: one line, actionable, outward-facing.
"""

_EVALUATOR_TEMPLATE = """\
Review these thought seeds from your idle reflection.
Score each 0.0-1.0 for depth/insight. Classify each as: retrospective | gap_reflection | \
self_improvement | curiosity. Discard if score < {min_score}.

Thoughts:
{seeds}

Respond as JSON array: [{{"thought": "...", "score": 0.x, "category": "...", "promote": true/false}}]
Only return the JSON array, nothing else.
"""


class IdleDetector:
    """Tracks last request timestamp and fires callback when idle > threshold_s."""

    def __init__(self, threshold_s: float, callback: Callable, poll_interval_s: float = 10.0,
                 startup_grace_s: float = 600.0):
        self._threshold_s = threshold_s
        self._callback = callback
        self._poll_interval_s = poll_interval_s
        # startup_grace_s: extra time after boot before idle detection begins.
        # Prevents ThinkAtRest firing immediately after every restart.
        self._startup_grace_s = startup_grace_s
        self._boot_time = time.monotonic()
        self._last_request = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def ping(self):
        """Call this on every incoming request to reset the idle clock."""
        self._last_request = time.monotonic()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="idle-detector")
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        poll = min(self._poll_interval_s, max(0.1, self._threshold_s / 10))
        while not self._stop_event.wait(poll):
            # Respect startup grace period — don't fire during boot window
            if (time.monotonic() - self._boot_time) < self._startup_grace_s:
                continue
            elapsed = time.monotonic() - self._last_request
            if elapsed >= self._threshold_s:
                try:
                    self._callback()
                except Exception as e:
                    logger.error(f"[IdleDetector] callback error: {e}")


class ThoughtGenerator:
    """System 1: prefer the configured local thought slot (default Gemma 'audio'),
    fall back to drafter-only generation when the local slot is unavailable."""

    def __init__(self, unused_skills_fn: Optional[Callable] = None,
                 recent_gaps_fn: Optional[Callable] = None,
                 last_thought_fn: Optional[Callable] = None,
                 recent_interactions_fn: Optional[Callable] = None,
                 thought_slot: str = "audio"):
        self._unused_skills_fn = unused_skills_fn
        self._recent_gaps_fn = recent_gaps_fn
        self._last_thought_fn = last_thought_fn
        self._recent_interactions_fn = recent_interactions_fn
        self._thought_slot = thought_slot
        self._recent_thoughts: list = []

    def set_recent_thoughts(self, thoughts: list) -> None:
        """Store recent thought summaries for anti-seed injection in generate()."""
        self._recent_thoughts = thoughts or []

    def generate(self, exploration_summary: str = "") -> list:
        """Return list of raw thought strings (1-3). Local model only.

        When exploration_summary is provided, generates a curiosity thought
        based on real outward data. When empty, returns nothing — no synthetic
        generation from vacuum.
        """
        import core.inference.model_client as model_client
        if not model_client.is_server_running():
            logger.debug("[ThoughtGenerator] model server not running — skipping")
            return []

        if not exploration_summary:
            logger.debug("[ThoughtGenerator] no exploration data — skipping synthetic generation")
            return []

        unused_skills = "none"
        try:
            if self._unused_skills_fn:
                unused_skills = self._unused_skills_fn() or "none"
        except Exception:
            pass

        # Get evolution context for enrichment
        evolution_context = "none"
        try:
            if self._recent_gaps_fn:
                evolution_context = self._recent_gaps_fn() or "none"
        except Exception:
            pass

        prompt = _GENERATOR_TEMPLATE.format(
            unused_skills=unused_skills,
            evolution_context=evolution_context,
        )
        prompt += f"\n\nRecent exploration data:\n{exploration_summary[:1200]}"

        # Anti-seed to prevent repetition
        if self._recent_thoughts:
            anti_seed_block = "\n".join(f"- {s}" for s in self._recent_thoughts[:5])
            prompt += (
                "\n\nDo not repeat or closely paraphrase any of these recent thoughts:\n"
                + anti_seed_block
            )

        messages = [
            {"role": "system", "content": _THOUGHT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Priority #0 + thought_slot config (2026-08-26): run thoughts on the
        # configured local slot (default Gemma 'audio') so idle reflection works
        # even in cloud mode and only ONE local model loads (no Nemotron drafter).
        raw = ""
        try:
            raw = model_client.infer_local(messages, max_new_tokens=8192, slot=self._thought_slot)
        except Exception as e:
            logger.error(f"[ThoughtGenerator] local slot infer error: {e}")

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtGenerator] local slot '{self._thought_slot}' unavailable, falling back to drafter: {raw!r}")
            try:
                raw = model_client.infer_draft(prompt, max_new_tokens=8192)
            except Exception as e:
                logger.error(f"[ThoughtGenerator] infer_draft error: {e}")
                return []

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtGenerator] bad response: {raw!r}")
            return []

        # Parse one thought per line, skip empty lines
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # Remove numbered prefixes like "1. " or "- "
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\d]+\.\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if line:
                cleaned.append(line)
        return cleaned[:5]


class ThoughtEvaluator:
    """System 2: uses the configured local thought slot to score + classify seeds."""

    def __init__(self, min_score: float = 0.65, thought_slot: str = "audio"):
        self._min_score = min_score
        self._thought_slot = thought_slot

    def evaluate(self, seeds: list) -> list:
        """
        Score and classify thought seeds.
        Returns list of dicts: {thought, score, category, promote}
        Discards score < min_score.
        """
        if not seeds:
            return []

        import core.inference.model_client as model_client
        if not model_client.is_server_running():
            logger.debug("[ThoughtEvaluator] model server not running — skipping")
            return []

        seeds_text = "\n".join(f"- {s}" for s in seeds)
        prompt = _EVALUATOR_TEMPLATE.format(
            seeds=seeds_text,
            min_score=self._min_score,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            raw = model_client.infer_local(messages, max_new_tokens=8192, slot=self._thought_slot)
        except Exception as e:
            logger.error(f"[ThoughtEvaluator] local slot infer error: {e}")
            return []

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtEvaluator] bad response: {raw!r}")
            return []

        # Extract JSON array from response
        try:
            # Try to find JSON array in response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                results = json.loads(match.group())
            else:
                results = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[ThoughtEvaluator] JSON parse error: {e} — raw: {raw[:200]}")
            return []

        if not isinstance(results, list):
            return []

        accepted = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            if score < self._min_score:
                continue
            # Normalise category
            category = item.get("category", "curiosity")
            valid_categories = {"retrospective", "gap_reflection", "self_improvement", "curiosity"}
            if category not in valid_categories:
                category = "curiosity"
            accepted.append({
                "thought": item.get("thought", ""),
                "score": score,
                "category": category,
                "promote": bool(item.get("promote", score >= 0.7)),
            })
        return accepted

