"""telegram_formatting.py — text/tool-step formatting helpers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains the
pure formatting primitives: _esc, _html, tool emoji/verb tables,
_tool_args_label, _format_tool_step. Kept behavior-identical; telegram_bot.py
re-imports these names so all call sites keep working unchanged.
"""


def _esc(text: str) -> str:
    """Escape underscores in dynamic text before wrapping in Telegram Markdown italic _..._"""
    return str(text).replace("_", "\\_")


def _html(text: str) -> str:
    """HTML-escape text for Telegram parse_mode="HTML".

    Telegram's legacy Markdown parser rejects messages containing unescaped
    markdown specials (`_`, `*`, backticks, `[`) with a 400 "can't parse
    entities" error. HTML parse mode only requires escaping `<`, `>`, and `&`,
    so it's far more robust for arbitrary model-generated text (which is full
    of code, links, and punctuation).
    """
    import html as _html_lib
    return _html_lib.escape(str(text), quote=False)


# ── Tool step display (Telegram) ─────────────────────────────────────
# Per-tool emoji + verb so tool-call progress messages are informative
# ("what tool, to do what") instead of a bare "🔧 tool1 → tool2".
_TOOL_EMOJI = {
    "read_file": "📄",
    "write_file": "✍️",
    "exec_shell": "⚙️",
    "http_get": "🌐",
    "web_search": "🔍",
    "browser_use": "🧭",
    "run_skill": "🧩",
    "run_routine": "🔄",
    "send_file": "📤",
    "search_skills": "🔎",
    "list_routines": "📋",
    "recall_memory": "🧠",
    "computer": "🖥️",
}
_TOOL_VERB = {
    "read_file": "read",
    "write_file": "write",
    "exec_shell": "run",
    "http_get": "fetch",
    "web_search": "search",
    "browser_use": "browse",
    "run_skill": "run skill",
    "run_routine": "run routine",
    "send_file": "send file",
    "search_skills": "find skill",
    "list_routines": "list routines",
    "recall_memory": "recall memory",
    "computer": "drive",
}
# Arg fields to surface first when summarising a tool call.
_TOOL_ARG_PRIORITY = (
    "path", "file_path", "query", "url", "command", "content",
    "skill_name", "routine_name", "input", "caption",
)


def _tool_args_label(name: str, args) -> str:
    """Return a short human-readable label for a tool call's args."""
    if args is None:
        return ""
    if isinstance(args, dict):
        for k in _TOOL_ARG_PRIORITY:
            v = args.get(k)
            if v:
                s = str(v).strip()
                if s:
                    return s[:80]
        return ""
    s = str(args).strip()
    return s[:80]


def _format_tool_step(n, tool_name, args=None, result=None) -> str:
    """Format one tool-call step as a readable, emoji-tagged Telegram line.

    Uses HTML formatting (Telegram HTML parse mode) — the tool name is in
    <code>, the step number is <b>bold</b>, and dynamic values are HTML-escaped
    so arbitrary model output never breaks the entity parser.
    """
    name = str(tool_name or "?")
    emoji = _TOOL_EMOJI.get(name, "🔧")
    verb = _TOOL_VERB.get(name, name.replace("_", " "))
    # Show the tool name in <code> (precise) with a friendly verb prefix.
    line = f"<b>Step {n}</b> {emoji} <code>{_html(verb)}</code> (<code>{_html(name)}</code>)"
    label = _tool_args_label(name, args)
    if label:
        line += f" — <code>{_html(label)}</code>"
    if result is not None:
        r = str(result).strip()
        if r and not r.lower().startswith("(error") and "error:" not in r.lower()[:60]:
            line += "\n  ✅ ok"
        else:
            line += f"\n  ⚠️ {_html(r[:120])}"
    return line


