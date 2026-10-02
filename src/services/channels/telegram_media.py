"""telegram_media.py — media (photo/voice/document) handlers for the Telegram bot.

Extracted from telegram_bot.py handle_message (issue #3 — decompose monolith,
see .specs/plans/2026-10-02-telegram-handle_message-split.md). Contains the
photo/voice/document attachment handlers. Seam names resolve through the
registered bot module at call time so test patches apply. Behavior-identical;
telegram_bot.py re-imports these names.
"""
import os

from runtime_paths import DOCUMENTS_DIR
from core.voice_activity import voice_activity


def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot.send_message / bot.edit_message / bot._ensure_agent / etc. apply.
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _handle_photo_message(chat_id: str, text: str, photo_file_id: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    send_voice = bm.send_voice
    send_typing = bm.send_typing
    download_file = bm.download_file
    _esc = bm._esc
    _format_tool_step = bm._format_tool_step
    _clone_voice_reply = bm._clone_voice_reply
    _list_voice_samples = bm._list_voice_samples
    TypingKeepAlive = bm.TypingKeepAlive
    _MultimodalActivity = bm._MultimodalActivity
    _ensure_agent = bm._ensure_agent
    _ATTACHMENT_RECENCY_SECONDS = bm._ATTACHMENT_RECENCY_SECONDS
    if photo_file_id:
        _ensure_agent()
        local_path = download_file(photo_file_id)
        if local_path:
            try:
                working_id = send_message(chat_id, "🔍 _Analysing image\u2026_")
                with _MultimodalActivity(), TypingKeepAlive(chat_id, action="upload_photo"):
                    from core.inference.model import infer_with_image
                    # Force the LOCAL Gemma E2B slot (no cloud credits available)
                    # and cap max_new_tokens to match the proven-working `look`/
                    # describe path. This skips the cloud attempt entirely (which
                    # 402s without credits) and avoids the native-slot hang seen
                    # with large token budgets, so the description returns promptly.
                    reply = infer_with_image(local_path, text or "Describe this image.",
                                             max_new_tokens=1024, force_local=True)
                # Raise on error strings so except block handles them cleanly
                if isinstance(reply, str) and reply.startswith(("[model_server", "[model_client", "[model_server timeout]")):
                    raise RuntimeError(reply)
                if not reply or not reply.strip():
                    raise RuntimeError("Vision returned empty response")
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
                import core.memory.memory as _memory_mod
                _memory_mod.record_attachment(
                    kind="photo", local_path="(temp-deleted)",
                    original_name="photo.jpg", mime_type="image/jpeg",
                    caption=text or "Describe this image.", chat_id=str(chat_id)
                )
                history = _memory_mod.load(chat_id=str(chat_id))
                # Store the user turn with the caption + a brief image summary so the
                # model has full context when the user asks follow-up questions.
                _img_user_content = f"[Image sent]{': ' + text if text else ''}"
                _img_summary = reply[:300] if len(reply) > 300 else reply
                history.append({"role": "user", "content": _img_user_content})
                history.append({"role": "assistant", "content": _img_summary})
                _memory_mod.save(history, chat_id=str(chat_id))
                print(f"[bot] photo: reply len={len(reply)} working_id={working_id}", flush=True)
                if working_id and not edit_message(chat_id, working_id, f"🐬 {reply}", parse_mode=None):
                    print(f"[bot] photo: edit_message failed — falling back to send_message", flush=True)
                    _sent = send_message(chat_id, f"🐬 {reply}", parse_mode=None)
                    print(f"[bot] photo: send_message result={_sent}", flush=True)
                else:
                    print(f"[bot] photo: edit_message ok (or working_id None)", flush=True)
            except Exception as e:
                err = f"🐬 Vision error: {str(e)[:200]}"
                print(f"[bot] photo EXCEPTION: {type(e).__name__}: {e}", flush=True)
                if working_id and not edit_message(chat_id, working_id, err):
                    send_message(chat_id, err)
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
        else:
            send_message(chat_id, "🐬 Could not download image.")
        return


def _handle_voice_message(chat_id: str, text: str, voice_file_id: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    send_voice = bm.send_voice
    send_typing = bm.send_typing
    download_file = bm.download_file
    _esc = bm._esc
    _format_tool_step = bm._format_tool_step
    _clone_voice_reply = bm._clone_voice_reply
    _list_voice_samples = bm._list_voice_samples
    TypingKeepAlive = bm.TypingKeepAlive
    _MultimodalActivity = bm._MultimodalActivity
    _ensure_agent = bm._ensure_agent
    _ATTACHMENT_RECENCY_SECONDS = bm._ATTACHMENT_RECENCY_SECONDS
    if voice_file_id:
        _ensure_agent()
        working_id = send_message(chat_id, "🎙️ _Listening…_")
        local_path = download_file(voice_file_id)
        if local_path:
            try:
                import core.memory.memory as _memory_mod
                import core.agent as _agent_voice
                from core.inference.model import infer_with_audio

                # Stage 1: STT
                if working_id:
                    edit_message(chat_id, working_id, "🎙️ _Transcribing audio…_")
                with _MultimodalActivity(), TypingKeepAlive(chat_id), voice_activity("stt"):
                    transcript = infer_with_audio(
                        local_path,
                        prompt="Transcribe exactly what is said in this audio. Output only the spoken words, nothing else.",
                        max_new_tokens=8192,
                        mode="stt",
                    )
                try:
                    os.unlink(local_path)
                except Exception:
                    pass

                if not transcript or not transcript.strip():
                    raise ValueError("STT returned empty transcript")

                # Prepend any typed text (rare but possible)
                user_text = f"{text} {transcript}".strip() if text else transcript
                print(f"[voice] transcript: {user_text[:120]!r}", flush=True)

                # ── First-contact voice sample collection ────────────────────────
                # If no voice samples are available, save this voice as the first sample
                # so voice cloning works going forward.
                if not _list_voice_samples():
                    _samples_dir = os.path.expanduser(
                        os.environ.get("KERNEL_VOICE_SAMPLES_DIR",
                                        os.path.expanduser("~/.openclaw/media/voice-samples"))
                    )
                    os.makedirs(_samples_dir, exist_ok=True)
                    _sample_path = os.path.join(_samples_dir, "first-contact-voice.ogg")
                    if os.path.isfile(local_path):
                        import shutil
                        shutil.copy2(local_path, _sample_path)
                        print(f"[voice] First-contact sample saved: {_sample_path}", flush=True)
                        # Also set as active voice sample (convert to wav for clone)
                        bm._active_voice_sample = _sample_path
                    else:
                        print(f"[voice] Cannot save first-contact sample: {local_path} not found", flush=True)
                # ──────────────────────────────────────────────────────────────────

                # Stage 2: Show transcript, start thinking
                if working_id:
                    edit_message(chat_id, working_id, f"🎙️ _{_esc(transcript)}_\n\n🤔 _Thinking…_")

                log_lines = []
                step_num = [0]

                def _voice_step_cb(n, tool_name, args=None, result=None):
                    step_num[0] = n
                    log_lines.append(_format_tool_step(n, tool_name, args, result))
                    snippet = f"🎙️ _{_esc(transcript[:60])}_\n\n🔍 " + "\n\n".join(log_lines[-4:])
                    if working_id:
                        edit_message(chat_id, working_id, snippet)

                # Stage 3: Agent inference
                with TypingKeepAlive(chat_id):
                    reply = _agent_voice.triage(user_text, step_callback=_voice_step_cb, chat_id=str(chat_id))

                if not reply or not reply.strip():
                    raise ValueError("triage returned empty response")

                # Persist voice note + reply into conversation history.
                # Always run after triage so the user turn carries the [Voice note] label,
                # even when infer_with_tools already saved a plain-text user entry.
                _voice_history = _memory_mod.load(chat_id=str(chat_id))
                if (_voice_history
                        and _voice_history[-1].get("role") == "assistant"
                        and len(_voice_history) >= 2
                        and _voice_history[-2].get("role") == "user"):
                    # triage already saved the pair — just relabel the user turn
                    _voice_history[-2]["content"] = f"[Voice note] {user_text}"
                else:
                    _voice_history.append({"role": "user", "content": f"[Voice note] {user_text}"})
                    _voice_history.append({"role": "assistant", "content": reply})
                _memory_mod.save(_voice_history, chat_id=str(chat_id))

                _memory_mod.record_attachment(
                    kind="voice", local_path="(temp-deleted)",
                    original_name="voice.ogg", mime_type="audio/ogg",
                    caption=f"[Voice note] {user_text}", chat_id=str(chat_id)
                )

                # Stage 4: Deliver text reply — keep transcript visible above reply
                # so the user can see what EVA heard before the answer.
                reply_text = f"🎙️ _{_esc(transcript)}_\n\n🐬 {reply}"
                if working_id and not edit_message(chat_id, working_id, reply_text):
                    send_message(chat_id, reply_text)

                # Stage 5: Clone voice async — send status msg first, audio arrives after
                import threading as _threading
                _clone_status_id = send_message(chat_id, "🔊 _Cloning voice reply…_")

                def _send_voice_async(text: str, cid: str, status_id):
                    try:
                        with TypingKeepAlive(cid, action="record_voice"), voice_activity("clone"):
                            _wav = _clone_voice_reply(text)
                        if _wav:
                            send_voice(cid, _wav)
                            os.unlink(_wav)
                            if status_id:
                                edit_message(cid, status_id, "🔊 _Voice reply sent_")
                        else:
                            if status_id:
                                edit_message(cid, status_id, "🔇 _Voice clone unavailable_")
                    except Exception as _ve:
                        print(f"[voice] clone async error: {_ve}")
                        if status_id:
                            edit_message(cid, status_id, f"🔇 _Voice error: {_esc(str(_ve)[:80])}_")

                _threading.Thread(
                    target=_send_voice_async,
                    args=(reply, chat_id, _clone_status_id),
                    daemon=True
                ).start()

            except Exception as e:
                err = f"🐬 Audio error: {str(e)[:200]}"
                if working_id and not edit_message(chat_id, working_id, err):
                    send_message(chat_id, err)
        else:
            err = "🐬 Could not download voice note."
            if working_id and not edit_message(chat_id, working_id, err):
                send_message(chat_id, err)
        return


def _handle_document_message(chat_id: str, text: str, document_file_id: str, document_name: str, document_mime: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    send_voice = bm.send_voice
    send_typing = bm.send_typing
    download_file = bm.download_file
    _esc = bm._esc
    _format_tool_step = bm._format_tool_step
    _clone_voice_reply = bm._clone_voice_reply
    _list_voice_samples = bm._list_voice_samples
    TypingKeepAlive = bm.TypingKeepAlive
    _MultimodalActivity = bm._MultimodalActivity
    _ensure_agent = bm._ensure_agent
    _ATTACHMENT_RECENCY_SECONDS = bm._ATTACHMENT_RECENCY_SECONDS
    if document_file_id:
        _ensure_agent()
        local_path = download_file(document_file_id)
        if local_path:
            try:
                import time as _time, shutil as _shutil
                fname = document_name or os.path.basename(local_path)
                ws_docs = str(DOCUMENTS_DIR)
                os.makedirs(ws_docs, exist_ok=True)
                ts = _time.strftime("%Y%m%d-%H%M%S")
                saved_name = f"{ts}_{fname}"
                saved_path = os.path.join(ws_docs, saved_name)
                _shutil.copy2(local_path, saved_path)
                os.unlink(local_path)
                print(f"[bot] Document saved: {saved_path}", flush=True)

                # Hand path + caption to agent — skill selection happens in triage/tool loop
                user_caption = text.strip() if text.strip() else "Process this document."
                full_prompt = (
                    f"{user_caption}\n\n"
                    f"[File saved to: {saved_path} | name: {fname} | type: {document_mime or 'unknown'}]"
                )

                import core.memory.memory as _memory_mod
                # Record attachment durably before triage
                _memory_mod.record_attachment(
                    kind="document",
                    local_path=saved_path,
                    original_name=fname,
                    mime_type=document_mime or "",
                    caption=user_caption,
                    chat_id=str(chat_id),
                )
                # Inject recent-attachment context into the prompt
                att_ctx = _memory_mod.attachment_context_block(chat_id=str(chat_id), limit=3, max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                if att_ctx:
                    full_prompt = f"{att_ctx}\n\n{full_prompt}"

                # For PDFs: invoke extract_document native tool directly —
                # no skill lookup, no ecosystem dependency, always available.
                is_pdf = fname.lower().endswith(".pdf") or (document_mime or "").lower() == "application/pdf"
                if is_pdf:
                    try:
                        import os as _os
                        from core.skills import load_all as _load_skills, find as _find_skill, run as _run_skill
                        import core.agent as _agent_mod
                        _skills_dir = _agent_mod._config.get("skills_dir", "./skills") if _agent_mod._config else "./skills"
                        _skills_dir = _os.path.expanduser(_os.environ.get("SKILLS_DIR") or _skills_dir)
                        _core = _agent_mod._config.get("core_skills", []) if _agent_mod._config else []
                        _all_skills = _load_skills(_skills_dir, core_skills=_core)
                        _skill = _find_skill("kernel-doc-retrieval", _all_skills)
                        working_id = send_message(chat_id, "🐬 Processing PDF…")
                        with _MultimodalActivity():
                            if _skill:
                                skill_result = _run_skill(_skill, f"/markdown {saved_path}", lambda *a: "")
                                reply = skill_result if skill_result and skill_result.strip() else (
                                    f"✅ PDF processed: `{fname}` — markdown sent as Telegram attachment."
                                )
                            else:
                                reply = f"⚠️ kernel-doc-retrieval skill not found — PDF saved to: `{saved_path}`"
                        if working_id and not edit_message(chat_id, working_id, f"🐬 {reply}"):
                            send_message(chat_id, f"🐬 {reply}")
                    except Exception as _pdf_err:
                        send_message(chat_id, f"🐬 PDF error: {str(_pdf_err)[:200]}")
                    history = _memory_mod.load(chat_id=str(chat_id))
                    history.append({"role": "user", "content": f"[Document: {fname}] {text}"})
                    history.append({"role": "assistant", "content": reply})
                    _memory_mod.save(history, chat_id=str(chat_id))
                    return

                from core.agent import triage
                reply = triage(full_prompt, chat_id=str(chat_id))

                # Completion guard: retry once if reply ignored the attached file
                guard = _memory_mod.attachment_guard(user_caption, reply, chat_id=str(chat_id))
                if not guard["ok"]:
                    print(f"[bot] attachment_guard FAIL (retrying): {guard['reason']}", flush=True)
                    retry_prompt = (
                        f"{att_ctx}\n\n"
                        f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                        f"{full_prompt}"
                    )
                    reply = triage(retry_prompt, chat_id=str(chat_id))
                    guard2 = _memory_mod.attachment_guard(user_caption, reply, chat_id=str(chat_id))
                    if not guard2["ok"]:
                        reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

                history = _memory_mod.load(chat_id=str(chat_id))

                history.append({"role": "user", "content": f"[Document: {fname}] {text}"})
                history.append({"role": "assistant", "content": reply})
                _memory_mod.save(history, chat_id=str(chat_id))

                send_message(chat_id, f"\U0001f42c {reply}")
            except Exception as e:
                send_message(chat_id, f"\U0001f42c Document error: {str(e)[:200]}")
        else:
            send_message(chat_id, "\U0001f42c Could not download document.")
        return
