"""telegram_handlers.py — core message-reply handler for the Telegram bot.

Extracted from telegram_bot.py handle_message (issue #3 — decompose monolith,
see .specs/plans/2026-10-02-telegram-handle_message-split.md). Slice C:
the /thoughts command, unknown-command triage, and the core streaming
reply/triage loop. Seam names resolve through the registered bot module at
call time so test patches apply. Behavior-identical; telegram_bot.py
re-imports these names.
"""
from telegram_config import CONFIG_PATH


def _bot_module():
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _handle_thoughts(chat_id: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    _esc = bm._esc
    try:
        import requests as _req
        r = _req.get("http://localhost:8779/thoughts", timeout=5)
        thoughts = r.json() if r.ok else []
    except Exception:
        thoughts = []
    if not thoughts:
        send_message(chat_id, "🤔 No thoughts recorded today yet.")
    else:
        recent = thoughts[-10:]
        lines = ["🤔 *Kernel's thoughts today:*\n"]
        for t in recent:
            score = t.get("score", 0.0)
            category = t.get("category", "thought")
            thought_text = t.get("thought", "")
            time_str = t.get("time", "")
            lines.append(f"*{time_str}* [{category}] _{_esc(thought_text)}_ (score: {score:.2f})")
        send_message(chat_id, "\n".join(lines))
    return

# For unrecognised slash commands, try agent.triage() first
# so skills with exec dispatch (e.g. /markdown, /anonymize) run deterministically.
# Known built-in commands are already handled above — only forward unknowns.
_BUILTIN_COMMANDS = {
    "/start", "/help", "/skills", "/routines", "/run", "/skill",
    "/status", "/verbose", "/packages", "/search", "/install",
    "/clone", "/private_repo", "/update", "/restart", "/stop", "/rollback",
    "/replica", "/workspaces", "/system", "/version", "/models",
    "/evolve", "/thoughts", "/new", "/voices", "/provider", "/init",
}


def _handle_unknown_command(chat_id: str, text: str) -> bool:
    """Triage unrecognised slash commands via agent.triage; True if handled."""
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    _ensure_agent = bm._ensure_agent
    TypingKeepAlive = bm.TypingKeepAlive
    _BUILTIN_COMMANDS = {
        "/start", "/help", "/skills", "/routines", "/run", "/skill",
        "/status", "/verbose", "/packages", "/search", "/install",
        "/clone", "/private_repo", "/update", "/restart", "/stop", "/rollback",
        "/replica", "/workspaces", "/system", "/version", "/models",
        "/evolve", "/thoughts", "/new", "/voices", "/provider", "/init",
    }
    if not (text.startswith("/") or text.startswith("set_voice_") or text.startswith("/skill_") or text.startswith("/run_")):
        return False
    cmd_word = text.split()[0].lower()
    if cmd_word not in _BUILTIN_COMMANDS and not text.startswith("set_voice_"):
        _ensure_agent()
        # Immediate ack + keep typing indicator alive during long exec (e.g. /markdown)
        working_id = send_message(chat_id, f"🐬 Running `{cmd_word}`\u2026")
        import core.agent as _a
        with TypingKeepAlive(chat_id):
            result = _a.triage(text, chat_id=str(chat_id))
        if result:
            reply_text = f"🐬 {result[:3800]}"
            if working_id:
                edit_message(chat_id, working_id, f"🐬 `{cmd_word}` complete ✅")
            send_message(chat_id, reply_text, parse_mode="")
        return True
    return False


def _handle_core_reply(chat_id: str, text: str, sender_name: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    _ensure_agent = bm._ensure_agent
    _html = bm._html
    _format_tool_step = bm._format_tool_step
    _write_collective_memory = bm._write_collective_memory
    TypingKeepAlive = bm.TypingKeepAlive
    _ATTACHMENT_RECENCY_SECONDS = bm._ATTACHMENT_RECENCY_SECONDS
    if not bm._agent_ready:
        send_message(chat_id, "⏳ Loading model (~60s)...")

    _ensure_agent()
    from core.inference.model import infer
    import core.memory.memory as _memory_mod

    # Build rich system prompt with live context
    import core.agent as _agent_mod_ctx
    import yaml as _yaml_ctx
    import os as _os_ctx
    from core.memory.context import build_system_prompt as _build_prompt
    from core.skills import load_all as _load_skills
    from core.routines import load_all as _load_routines
    from core.inference.model import vram_free_mb as _vram_free
    _cfg_path = CONFIG_PATH
    _cfg = _yaml_ctx.safe_load(open(_cfg_path))
    _skills = _load_skills(_os_ctx.path.expanduser(_cfg.get("skills_dir", "./skills")))
    _routines = _load_routines(_os_ctx.path.expanduser(_cfg.get("routines_dir", "./routines")))
    system_prompt = _build_prompt(
        _cfg, _skills, _routines,
        vram_free_fn=_vram_free,
        channel="telegram",
        sender_name=sender_name,
        agent_ready=bm._agent_ready,
    )

    # Prepend relevant collective memory results if found.
    # NOTE: triage() also injects collective memory via agent.py's PD1 path
    # (for all callers, not just Telegram). The Telegram path builds its own
    # system_prompt here and then passes the user message into triage() — both
    # prompts coexist. The legacy subprocess-based search below is now skipped
    # since the HTTP client in core.collective_memory_client handles this more
    # reliably for all callers. Kept as a comment to explain the removed call.

    # No placeholder — streaming commences immediately on first chunk/step
    working_id = [None]  # mutable so closures can assign

    try:
        with TypingKeepAlive(chat_id):
            import core.agent as _agent_mod
            log_lines = []
            step_num = [0]
            # Streaming state — accumulate chunks and edit message live
            _stream_buf = [""]
            _stream_char_count = [0]
            _STREAM_EDIT_EVERY = 40  # edit message every N chars to avoid flood

            def _chunk_cb(chunk: str):
                """Called with each streamed text chunk from the model."""
                _stream_buf[0] += chunk
                _stream_char_count[0] += len(chunk)
                if _stream_char_count[0] >= _STREAM_EDIT_EVERY:
                    _stream_char_count[0] = 0
                    preview = _stream_buf[0]
                    snippet = ("🔍 " + "\n\n".join(log_lines[-3:]) + f"\n\n✍️ {_html(preview)}"
                               if log_lines else f"✍️ {_html(preview)}")
                    if working_id[0]:
                        edit_message(chat_id, working_id[0], snippet[:4000], parse_mode="HTML")
                    else:
                        # First output — create the message now, no prior placeholder
                        working_id[0] = send_message(chat_id, snippet[:4000], parse_mode="HTML")

            def _step_cb(n, tool_name, args=None, result=None):
                step_num[0] = n
                # Clear stream buffer — step output takes over display
                _stream_buf[0] = ""
                _stream_char_count[0] = 0
                # Show an informative, emoji-tagged step line (what tool, to do what).
                log_lines.append(_format_tool_step(n, tool_name, args, result))
                # Keep the last few steps so the message stays readable.
                snippet = "🔍 " + "\n\n".join(log_lines[-4:])
                if working_id[0]:
                    edit_message(chat_id, working_id[0], snippet[:4000], parse_mode="HTML")
                else:
                    working_id[0] = send_message(chat_id, snippet[:4000], parse_mode="HTML")

            # Inject recent attachment context only when message references a file
            _triage_text = text
            _att_ctx = None
            _text_lower = text.lower()
            _ATT_REF_KEYWORDS = (
                "pdf", "document", "attachment", "uploaded", "sent you",
                "the file", "this file", "that file", "the doc", "the pdf",
                "image", "photo", "picture", "voice", "audio", "recording",
                "the one i sent", "what i sent", "that i sent",
            )
            if any(kw in _text_lower for kw in _ATT_REF_KEYWORDS):
                _att_ctx = _memory_mod.attachment_context_block(chat_id=str(chat_id), limit=3, max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
            if _att_ctx:
                _triage_text = f"{_att_ctx}\n\n{text}"

            reply = _agent_mod.triage(_triage_text, step_callback=_step_cb, chat_id=str(chat_id), chunk_callback=_chunk_cb)

            # --- Completion gates ---
            # 1. Attachment guard: if user referenced a file, did reply use it?
            if _att_ctx:
                _guard = _memory_mod.attachment_guard(text, reply, chat_id=str(chat_id), max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                if not _guard["ok"]:
                    print(f"[bot] chat guard FAIL (retrying): {_guard['reason']}", flush=True)
                    _retry_text = (
                        f"{_att_ctx}\n\n"
                        f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                        f"{text}"
                    )
                    reply = _agent_mod.triage(_retry_text, step_callback=_step_cb, chat_id=str(chat_id), chunk_callback=_chunk_cb)
                    _guard2 = _memory_mod.attachment_guard(text, reply, chat_id=str(chat_id), max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                    if not _guard2["ok"]:
                        reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

            # 2. Artifact check: if user asked to create/save something, was a file produced?
            _art = _memory_mod.artifact_check(text, reply)
            if not _art["ok"]:
                print(f"[bot] artifact_check FAIL: {_art['reason']}", flush=True)
                reply = reply + "\n\n⚠️ (Note: I was asked to produce a file but couldn't verify it was saved — please check your workspace or re-ask.)"

            # 3. ADR-020: Failure detection — log unresolved requests as evolution anchors
            try:
                import database.agent.failed_requests as _fr
                _failure_type = _fr.detect_failure(reply, text)
                if _failure_type:
                    _fr.record(str(chat_id), text, reply, _failure_type)
                    print(f"[bot] ADR-020 failure logged: type={_failure_type}", flush=True)
            except Exception as _fre:
                print(f"[bot] failed_requests record error: {_fre}", flush=True)

            # Memory is already persisted inside agent.triage().
            # Do NOT save here again, otherwise each plain-chat turn is duplicated
            # and history inflates/noises future responses.
            if step_num[0] == 0 and len(reply) > 200:
                _write_collective_memory(text, reply)

        # Final reply — never send an empty body ("🐬 " only) if model returns blank.
        _final_reply = (reply or "").strip()
        if not _final_reply:
            _streamed = (_stream_buf[0] or "").strip()
            if _streamed:
                _final_reply = _streamed
            elif log_lines:
                _last_steps = "\n\n".join(log_lines[-2:])
                _final_reply = (
                    "I couldn't produce a final answer.\n\n"
                    "Last tool activity:\n"
                    f"{_last_steps}"
                )
            else:
                _final_reply = (
                    "I couldn't produce a final answer and no tool step was emitted. "
                    "Check model/tool logs for this request and retry."
                )

        # Build final message: preserve all step logs then append final answer.
        # Escape dynamic content for Telegram HTML parse mode — legacy Markdown
        # rejects unescaped `_`/`*`/backticks with a 400 "can't parse entities",
        # which silently drops the final answer. HTML only needs < > & escaped.
        if log_lines:
            _steps_section = "🔍 " + "\n\n".join(_html(l) for l in log_lines)
            reply_text = f"{_steps_section}\n\n🐬 {_html(_final_reply)}"
        else:
            reply_text = f"🐬 {_html(_final_reply)}"
        # Telegram hard limit is 4096 chars; truncate from beginning to keep final answer
        if len(reply_text) > 4000:
            reply_text = "…" + reply_text[-3998:]
        if working_id[0] and not edit_message(chat_id, working_id[0], reply_text, parse_mode="HTML"):
            send_message(chat_id, reply_text, parse_mode="HTML")
        elif not working_id[0]:
            send_message(chat_id, reply_text, parse_mode="HTML")
    except Exception as e:
        print(f"[bot] ERROR in infer: {e}", flush=True)
        err_text = f"🐬 Error: {str(e)[:200]}"
        if working_id[0] and not edit_message(chat_id, working_id[0], err_text):
            send_message(chat_id, err_text)
