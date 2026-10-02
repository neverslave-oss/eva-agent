"""telegram_memory.py — collective-memory search/write helpers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains
_search_collective_memory and _write_collective_memory. Kept behavior-identical;
telegram_bot.py re-imports these names.
"""
import os
import subprocess

# @TODO: change hardcoded path reference to review.
def _search_collective_memory(query: str) -> str:
    """Run collective memory search via installed context_provider skills.
    Falls back to the direct script path if no skill is found.
    """
    # Prefer skill-based discovery (same path agent.py uses)
    try:
        import core.agent as _ag
        _ctx_skills = [s for s in (_ag._skills or []) if s.get("context_provider") and s.get("context_search_cmd")]
        if _ctx_skills:
            import shlex as _sl, subprocess as _sp
            results = []
            for _cs in _ctx_skills:
                _cmd = _cs["context_search_cmd"].replace("{query}", _sl.quote(query[:200]))
                _res = _sp.run(_cmd, shell=True, capture_output=True, text=True, timeout=6)
                _out = (_res.stdout or "").strip()
                if _out and len(_out) > 20:
                    results.append(_out[:600])
            return "\n\n".join(results)
    except Exception:
        pass
    # Fallback: direct script
    script = os.path.expanduser("~/.openclaw/workspace/collective-memory/scripts/search.py")
    try:
        result = subprocess.run(
            ["python3", script, query, "--top", "2"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""

# @TODO: change hardcoded path reference to review.
def _write_collective_memory(user_message: str, reply: str) -> None:
    """Write a new collective memory entry based on the conversation turn.

    Quality gate: only write entries that are genuinely informative — skip
    greetings, short exchanges, error messages, or anything that adds no signal.
    """
    entries_dir = Path(os.path.expanduser("~/.openclaw/workspace/collective-memory/entries"))
    entries_dir.mkdir(parents=True, exist_ok=True)

    # ── Quality gate ───────────────────────────────────────────────
    # Skip trivial / low-signal turns that pollute the collective memory.
    _skip_patterns = (
        # Greetings
        r'^(hi|hey|hello|ciao|good morning|good evening|yo)[ !.,]*$',
        # One-word or very short messages
        r'^.{0,15}$',
        # Status checks
        r'^(ping|status|ok|yes|no|sure|thanks|thank you|great|nice|cool)[ !.,]*$',
    )
    _msg_lower = user_message.strip().lower()
    for _pat in _skip_patterns:
        if re.match(_pat, _msg_lower, re.I):
            return

    # Skip error replies or short model outputs
    if len(reply) < 300:
        return
    if any(marker in reply for marker in ("[model_server error]", "[model_client error]", "VRAM guard")):
        return

    # Skip replies that are purely conversational without factual content
    # (heuristic: meaningful entries tend to have URLs, file paths, numbers, or code)
    _has_substance = bool(re.search(
        r'(https?://|/home/|/mnt/|```|\$\s|v\d+\.\d+|€|\d{4}-\d{2}-\d{2}|ADR-\d|\bPR\b|\bcommit\b)',
        reply, re.I
    ))
    if not _has_substance:
        return
    # ────────────────────────────────────────────────────────

    today = date.today().isoformat()
    # Infer topic from first 6 words of user message
    words = re.sub(r"[^a-zA-Z0-9\s]", "", user_message).split()[:6]
    topic = " ".join(words) if words else "general"
    slug = re.sub(r"\s+", "-", topic.lower())[:60]
    filename = entries_dir / f"{today}-{slug}.md"

    # Don't overwrite an existing entry with the same slug
    if filename.exists():
        return

    # Simple confidence heuristic: if reply contains hedging words → inferred
    hedging = re.search(
        r"\b(maybe|perhaps|might|could|unsure|unclear|I think|probably)\b", reply, re.I
    )
    confidence = "inferred" if hedging else "confirmed"

    # Bullet-point the reply lines (first 10 non-empty lines)
    lines = [l.strip() for l in reply.splitlines() if l.strip()][:10]
    bullets = "\n".join(f"- {l}" for l in lines)

    content = (
        f"---\n"
        f"date: {today}\n"
        f"agent: kernel\n"
        f"topic: {topic}\n"
        f"tags: [telegram, auto]\n"
        f"confidence: {confidence}\n"
        f"---\n"
        f"# {topic.title()}\n"
        f"{bullets}\n"
    )
    try:
        filename.write_text(content)
        print(f"[bot] collective memory written: {filename.name}", flush=True)
    except Exception as e:
        print(f"[bot] collective memory write error: {e}")

