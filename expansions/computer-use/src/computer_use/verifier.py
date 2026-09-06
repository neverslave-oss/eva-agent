from __future__ import annotations

from .schema import Expectation, Observation


class Verifier:
    def verify(self, expectation: Expectation | None, observation: Observation) -> tuple[bool, str]:
        if expectation is None:
            return True, "no expectation"
        if expectation.text_contains and (expectation.text_contains not in (observation.text or "")):
            return False, "expected text missing"
        if expectation.url_contains and (expectation.url_contains not in (observation.url or "")):
            return False, "expected url fragment missing"
        return True, "ok"
