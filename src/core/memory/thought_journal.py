"""
thought_journal.py — ADR-005 Thought Journal writer/reader.

Writes accepted thoughts to ~/.kernel-evolving/workspace/thoughts/YYYY-MM-DD.md

ADR-019 audit extension:
  - Observer verdict, classification, evidence, and evolution path are persisted
    in the journal .md alongside each thought entry.
  - A machine-readable JSONL audit log is written in parallel to
    ~/.kernel-evolving/workspace/thoughts/observer_audit.jsonl
    for grep/analysis without parsing markdown.
  - Critique verdicts (when evolution is enabled) are appended to the same
    JSONL record via append_critique_audit().
"""
import json
import os
from datetime import datetime, date
from pathlib import Path
from runtime_paths import THOUGHTS_DIR

_DEFAULT_JOURNAL_DIR = str(THOUGHTS_DIR)
_AUDIT_LOG_NAME = "observer_audit.jsonl"


class ThoughtJournal:
    def __init__(self, journal_dir: str | None = None):
        self.journal_dir = os.path.expanduser(journal_dir or _DEFAULT_JOURNAL_DIR)
        os.makedirs(self.journal_dir, exist_ok=True)
        self._audit_path = os.path.join(self.journal_dir, _AUDIT_LOG_NAME)

    def _path_for_date(self, d: date) -> str:
        return os.path.join(self.journal_dir, d.strftime("%Y-%m-%d") + ".md")

    def write(self, thought_dict: dict) -> None:
        """Append a thought entry to today's journal file.

        Persists observer verdict fields (_observer_verdict, _observer_path,
        _observer_evidence) when present, and writes a JSONL audit record.
        """
        now = datetime.now()
        path = self._path_for_date(now.date())
        thought = thought_dict.get("thought", "")
        category = thought_dict.get("category", "unknown")
        score = thought_dict.get("score", 0.0)
        promoted = thought_dict.get("promote", False)
        promoted_str = "yes" if promoted else "no"

        # ADR-019: observer metadata
        obs_verdict  = thought_dict.get("_observer_verdict", "")
        obs_path     = thought_dict.get("_observer_path", "")
        obs_evidence = thought_dict.get("_observer_evidence", "")

        obs_line = ""
        if obs_verdict:
            obs_line = f"Observer: {obs_verdict} / {obs_path}"
            if obs_evidence:
                obs_line += f" | evidence: {obs_evidence}"
            obs_line = obs_line + "\n"

        entry = (
            f"\n## {now.strftime('%H:%M')} — {category}\n"
            f"{thought}\n"
            f"Score: {score:.2f} | Promoted: {promoted_str}\n"
            f"{obs_line}"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)

        # ADR-019: JSONL audit record (always written, even if no observer fields)
        audit_record = {
            "ts": now.isoformat(),
            "category": category,
            "score": score,
            "promoted": promoted,
            "thought": thought[:200],
            "observer_verdict": obs_verdict or None,
            "observer_path": obs_path or None,
            "observer_evidence": obs_evidence or None,
            "critique_verdict": None,
            "critique_attempt": None,
            "critique_notes": None,
        }
        self._write_audit(audit_record)

    def append_critique_audit(self, thought_dict: dict) -> None:
        """Update the most recent JSONL audit record with critique verdict fields.

        Called by ThinkAtRest after _run_evolution_with_critique completes so
        that the critique result is associated with the original thought entry.
        """
        verdict  = thought_dict.get("_critique_verdict")
        attempt  = thought_dict.get("_critique_attempt")
        notes    = thought_dict.get("_critique_notes", "")
        if verdict is None:
            return
        # Rewrite last line of audit log with updated critique fields
        try:
            audit_path = Path(self._audit_path)
            if not audit_path.exists():
                return
            lines = audit_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return
            last = json.loads(lines[-1])
            last["critique_verdict"] = verdict
            last["critique_attempt"] = attempt
            last["critique_notes"] = notes
            lines[-1] = json.dumps(last, ensure_ascii=False)
            audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass  # audit write failure must never crash the main pipeline

    def _write_audit(self, record: dict) -> None:
        """Append one JSON record to the audit JSONL file."""
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # audit write failure must never crash the main pipeline

    # ── Read helpers ──────────────────────────────────────────────────────────

    def read_today(self) -> list:
        """Return list of thought dicts from today's journal."""
        return self.read_date(date.today())

    def read_date(self, d: date) -> list:
        """Return list of thought dicts parsed from the given date's journal."""
        path = self._path_for_date(d)
        if not os.path.exists(path):
            return []
        return self._parse_journal(path)

    def _parse_journal(self, path: str) -> list:
        """Parse a journal markdown file into a list of thought dicts."""
        thoughts = []
        with open(path, encoding="utf-8") as f:
            content = f.read()

        import re
        pattern = re.compile(
            r"^## (\d{2}:\d{2}) — (\w+)\n(.+?)\nScore: ([\d.]+) \| Promoted: (\w+)",
            re.MULTILINE | re.DOTALL,
        )
        for m in pattern.finditer(content):
            time_str, category, thought, score, promoted = m.groups()
            thoughts.append({
                "time": time_str,
                "category": category,
                "thought": thought.strip(),
                "score": float(score),
                "promote": promoted == "yes",
            })
        return thoughts

    def read_recent(self, days: int = 7) -> list:
        """Return thoughts from the last N days."""
        from datetime import timedelta
        result = []
        today = date.today()
        for i in range(days):
            d = today - timedelta(days=i)
            result.extend(self.read_date(d))
        return result

    def read_audit(self, date_str: str | None = None) -> list:
        """Return parsed JSONL audit records, optionally filtered by date prefix (YYYY-MM-DD)."""
        try:
            lines = Path(self._audit_path).read_text(encoding="utf-8").splitlines()
            records = [json.loads(l) for l in lines if l.strip()]
            if date_str:
                records = [r for r in records if r.get("ts", "").startswith(date_str)]
            return records
        except Exception:
            return []
