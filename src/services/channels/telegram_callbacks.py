"""telegram_callbacks.py — inline-button callback dispatcher for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains
handle_callback (the inline-button dispatcher) and the _MultimodalActivity
context manager. Seam names (send_message/edit_message/handle_message/
provider pickers/_ensure_agent/_esc/_CLOUD_CALL_TYPES) resolve through the
registered bot module at call time so test patches apply. Behavior-identical;
telegram_bot.py re-imports these names.
"""
import os
import re
from pathlib import Path

from telegram_config import CONFIG_PATH


def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot.send_message / bot._apply_modal_model / etc. apply.
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def handle_callback(chat_id: str, data: str, message_id: int):
    """Handle inline button callbacks."""
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    handle_message = bm.handle_message
    _esc = bm._esc
    _ensure_agent = bm._ensure_agent
    _CLOUD_CALL_TYPES = bm._CLOUD_CALL_TYPES
    _apply_cloud_model = bm._apply_cloud_model
    _apply_modal_model = bm._apply_modal_model
    _show_cloud_model_picker = bm._show_cloud_model_picker
    _show_modal_model_picker = bm._show_modal_model_picker
    try:
        print(f"[bot] callback: chat={chat_id} data={data}", flush=True)
        # ── Shell command authorization ──────────────────────────────────
        if data.startswith("auth_allow_all_"):
            from core.auth_gate import resolve_auth
            request_id = data.split("_", 3)[-1]
            resolve_auth(request_id, True, allow_all=True)
            if message_id:
                edit_message(chat_id, message_id, "✅ Shell command approved — *all* commands for this task will run automatically")
            return
        if data.startswith("auth_allow_") or data.startswith("auth_deny_"):
            from core.auth_gate import resolve_auth
            approved = data.startswith("auth_allow_")
            request_id = data.split("_", 2)[-1]
            resolve_auth(request_id, approved)
            status = "✅ Approved" if approved else "❌ Denied"
            if message_id:
                edit_message(chat_id, message_id, f"*Shell command* — {status}")
            return
        # ── Computer-use risky action confirmation ────────────────────────
        if data.startswith("cu_allow_") or data.startswith("cu_deny_"):
            from core.computer_confirm_gate import resolve_confirm
            approved = data.startswith("cu_allow_")
            request_id = data.split("_", 2)[-1]
            resolve_confirm(request_id, approved)
            status = "✅ Allowed" if approved else "❌ Denied"
            if message_id:
                edit_message(chat_id, message_id, f"*Risky computer-use action* — {status}")
            return
        # ── Ask-questions user prompt ──────────────────────────────────────
        # Format: aq_<request_id>_<option_idx>
        if data.startswith("aq_"):
            parts = data.split("_")
            # parts: ['aq', '<rid>', '<idx>']
            if len(parts) >= 3:
                request_id = parts[1]
                try:
                    option_idx = int(parts[2])
                except ValueError:
                    option_idx = 0
                from core.ask_questions_gate import resolve_question
                resolve_question(request_id, option_idx)
                if message_id:
                    edit_message(chat_id, message_id, "✅ Got it — thanks!")
            return
        if data.startswith("install_"):
            parts = data.split("_", 2)
            item_type = parts[1] if len(parts) > 1 else None
            name = parts[2] if len(parts) > 2 else ""
            from bootstrap import install as eco_install
            result = eco_install(name, item_type)
            send_message(chat_id, result["message"])
            return
        elif data.startswith("cloud_provider_"):
            provider = data[len("cloud_provider_"):]
            _show_cloud_model_picker(chat_id, provider, 0)
            return
        elif data.startswith("cmprov_"):
            # cmprov_{call_type}|{provider}
            rest = data[len("cmprov_"):]
            parts = rest.split("|", 1)
            if len(parts) >= 2:
                ct = parts[0]
                modal_provider = parts[1]
                try:
                    idx = _CLOUD_CALL_TYPES.index(ct)
                except ValueError:
                    idx = 0
                if modal_provider == "local":
                    # Keep local — set provider=local, advance
                    _apply_modal_model(chat_id, "local", ct, None, idx)
                else:
                    # Show model picker for this provider
                    _show_modal_model_picker(chat_id, modal_provider, idx)
            return
        elif data.startswith("cmm_"):
            # cmm_{modal_provider}|{call_type}|{model}
            rest = data[len("cmm_"):]
            parts = rest.split("|", 2)
            if len(parts) >= 2:
                modal_provider = parts[0]
                ct = parts[1]
                model = parts[2].replace("~", "/") if len(parts) > 2 and parts[2] != "default" else None
                if ct in _CLOUD_CALL_TYPES:
                    try:
                        idx = _CLOUD_CALL_TYPES.index(ct)
                    except ValueError:
                        idx = 0
                    _apply_modal_model(chat_id, modal_provider, ct, model, idx)
            return
        elif data.startswith("cm_"):
            # cm_{provider}|{call_type}|{model}
            rest = data[len("cm_"):]
            parts = rest.split("|", 2)
            if len(parts) >= 2:
                provider = parts[0]
                ct = parts[1]
                model = parts[2].replace("~", "/") if len(parts) > 2 and parts[2] != "default" else None
                if ct in _CLOUD_CALL_TYPES:
                    _apply_cloud_model(chat_id, provider, ct, model)
                else:
                    send_message(chat_id, f"\u274c Unknown call type: {ct}")
            return
        elif data.startswith("routine_run_"):
            name = data[12:]
            handle_message(chat_id, f"/run {name}")
            return
        elif data.startswith("routine_info_"):
            name = data[13:]
            _ensure_agent()
            import yaml

            with open(CONFIG_PATH) as _f:
                _cfg = yaml.safe_load(_f)
            from core.routines import load_all, find

            routines = load_all(
                str(os.path.expanduser(_cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
            )
            r = find(name, routines)
            if r:
                trigger = r.get("trigger", {})
                t = (
                    trigger.get("also", trigger.get("cron", "manual"))
                    if isinstance(trigger, dict)
                    else "manual"
                )
                send_message(
                    chat_id,
                    f"*{r['name']}*\n_{_esc(r['description'])}_\nTrigger: `{t}`\n\nTap ▶ Run to execute.",
                )
            return
        elif data.startswith("skill_info_"):
            name = data[11:]
            _ensure_agent()
            import yaml

            with open(CONFIG_PATH) as _f:
                _cfg = yaml.safe_load(_f)
            from core.skills import load_all, find

            skills = load_all(
                str(os.path.expanduser(_cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
            )
            s = find(name, skills)
            if s:
                send_message(chat_id, f"*{s['name']}*\n_{_esc(s['description'])}_")
            return
        elif data.startswith("set_voice_"):
            handle_message(chat_id, data)
            return
        elif data == "thought_skip":
            send_message(chat_id, "❌ Skipped.")
            return
        elif data.startswith("thought_do_"):
            # User approved an actionable thought — triage it as a task
            send_message(chat_id, "✅ On it…")
            _ensure_agent()
            import core.agent as _agent_thought
            import core.memory.memory as _mem_thought
            # Find the thought text from the last proactive message in memory
            history = _mem_thought.load(chat_id=str(chat_id))
            thought_text = ""
            for m in reversed(history[-20:]):
                c = str(m.get("content", ""))
                if "Kernel is thinking" in c:
                    # Extract italic text between _ _ markers
                    match = re.search(r'_(.+?)_', c, re.DOTALL)
                    if match:
                        thought_text = match.group(1).strip()
                    break
            if thought_text:
                result = _agent_thought.triage(f"Act on this: {thought_text}", chat_id=str(chat_id))
                send_message(chat_id, f"🐬 {result[:1500]}")
            else:
                send_message(chat_id, "🐬 Could not find the thought to act on — let me know what you want done.")
            return
        elif data.startswith("evo_confirm_resolved:"):
            # ADR-020: user confirmed evolution resolved their request
            try:
                request_id = int(data.split(":", 1)[1])
                import database.agent.failed_requests as _fr
                _fr.mark_resolved(request_id, reward=1)
                # Write positive trajectory for fine-tuning
                try:
                    from core.evolution.evolution_log import EvolutionLog
                    _elog = EvolutionLog()
                    _elog.record_resolution(request_id, reward=1)
                except Exception:
                    pass
                send_message(chat_id, "✅ *Resolved!* I've locked in what I learned. Thanks for confirming — this becomes part of my training.")
                print(f"[bot] ADR-020 evo_confirm_resolved: id={request_id} reward=1", flush=True)
            except Exception as _ce:
                print(f"[bot] evo_confirm_resolved error: {_ce}", flush=True)
                send_message(chat_id, "✅ Marked as resolved.")
            return
        elif data.startswith("evo_confirm_failed:"):
            # ADR-020: user rejected — retry evolution or escalate
            try:
                request_id = int(data.split(":", 1)[1])
                import database.agent.failed_requests as _fr
                retry_count = _fr.increment_retry(request_id)
                # Write negative trajectory
                try:
                    from core.evolution.evolution_log import EvolutionLog
                    _elog = EvolutionLog()
                    _elog.record_resolution(request_id, reward=0)
                except Exception:
                    pass
                if retry_count < 3:
                    send_message(
                        chat_id,
                        f"❌ Got it — still not there. I'll try again during the next idle cycle "
                        f"(attempt {retry_count + 1}/3)."
                    )
                else:
                    _fr.mark_resolved(request_id, reward=0)  # close as unresolvable
                    send_message(
                        chat_id,
                        f"❌ Three attempts, still unresolved. I've flagged this for manual review — "
                        f"you may need to install a specific skill or give me more context."
                    )
                print(f"[bot] ADR-020 evo_confirm_failed: id={request_id} retry={retry_count}", flush=True)
            except Exception as _ce:
                print(f"[bot] evo_confirm_failed error: {_ce}", flush=True)
                send_message(chat_id, "❌ Noted — I'll keep trying.")
            return
        elif data.startswith("tier2_approve:"):
            # ADR-021: user approved Tier 2 skill synthesis
            synthesis_id = data.split(":", 1)[1]
            send_message(chat_id, f"✅ *Approved!* Starting skill synthesis `{synthesis_id}` — I'll let you know when it's done.")
            try:
                import core.evolution.evolution_hook as _evo_hook
                _evo_hook._run_approved_synthesis(synthesis_id)
            except Exception as _ae:
                send_message(chat_id, f"⚠️ Synthesis failed to start: {str(_ae)[:200]}")
            return
        elif data.startswith("tier2_reject:"):
            # ADR-021: user rejected Tier 2 skill synthesis
            synthesis_id = data.split(":", 1)[1]
            send_message(chat_id, "❌ *Skill synthesis cancelled.* No tokens spent, nothing installed.")
            try:
                import core.evolution.evolution_hook as _evo_hook
                _evo_hook._reject_synthesis(synthesis_id)
            except Exception:
                pass
            return

        # Fallback: route unknown callbacks (e.g. /models, /provider) through the message handler
        handle_message(chat_id, data)
        return
    except Exception as e:
        print(f"[bot] callback error for {data}: {e}", flush=True)
        try:
            send_message(chat_id, f"❌ Button error: {str(e)[:200]}")
        except Exception:
            pass   
             
class _MultimodalActivity:
    """Context manager: marks model_activity in_flight during multimodal processing
    so ThinkAtRest idle evolution doesn't fire while vision/audio is running."""
    def __enter__(self):
        try:
            from core.inference.model_client import _mark_activity_start
            _mark_activity_start()
        except Exception:
            pass
        return self
    def __exit__(self, *_):
        try:
            from core.inference.model_client import _mark_activity_end
            _mark_activity_end()
        except Exception:
            pass

