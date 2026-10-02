"""telegram_commands.py — slash-command handlers for the Telegram bot.

Extracted from telegram_bot.py handle_message (issue #3 — decompose monolith,
see .specs/plans/2026-10-02-telegram-handle_message-split.md). Slice B:
command handlers, one _handle_* function per command. Seam names resolve
through the registered bot module at call time so test patches apply.
Behavior-identical; telegram_bot.py re-imports these names.
"""
import json
import os
import subprocess
from pathlib import Path

from telegram_config import CONFIG_PATH


def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot.send_message / bot.send_buttons / bot._ensure_agent / etc. apply.
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _handle_skill_command(chat_id, text) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    _skill_parts = text[7:].split(None, 1)
    skill_slug = _skill_parts[0].strip()
    skill_user_input = _skill_parts[1].strip() if len(_skill_parts) > 1 else ""
    try:
        import core.agent as _ag_sk
        import re as _re_sk
        def _slug_match(name):
            return _re_sk.sub(r"[^a-z0-9_]", "_", name.lower())[:26]  # 32-6 for "skill_"
        matched = next((s for s in (_ag_sk._skills or []) if _slug_match(s["name"]) == skill_slug), None)
        if matched:
            print(f"[bot] /skill_ dispatch: {matched['name']} input={skill_user_input!r}", flush=True)
            _ensure_agent()
            # Route through the same /run skill path so step-logging and streaming work correctly
            _run_text = f"/run {matched['name']}"
            if skill_user_input:
                _run_text = f"/run {matched['name']} {skill_user_input}"
            with TypingKeepAlive(chat_id):
                result = _ag_sk.triage(_run_text, chat_id=str(chat_id))
            send_message(chat_id, f"🐬 {result}" if result else "🐬 Done.")
        else:
            send_message(chat_id, f"🐬 No skill matching `{skill_slug}`. Try /skills.")
    except Exception as _e:
        send_message(chat_id, f"🐬 Skill error: {str(_e)[:200]}")
    return


def _handle_run_command(chat_id, text) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    routine_slug = text[5:].split()[0].strip()
    try:
        import core.agent as _ag_rn
        import re as _re_rn
        def _slug_match_r(name):
            return _re_rn.sub(r"[^a-z0-9_]", "_", name.lower())[:28]  # 32-4 for "run_"
        matched = next((r for r in (_ag_rn._routines or []) if _slug_match_r(r["name"]) == routine_slug), None)
        if matched:
            print(f"[bot] /run_ dispatch: {matched['name']}", flush=True)
            _ensure_agent()
            with TypingKeepAlive(chat_id):
                result = _ag_rn.triage(f"/run {matched['name']}", chat_id=str(chat_id))
            send_message(chat_id, f"🐬 {result}" if result else "🐬 Done.")
        else:
            send_message(chat_id, f"🐬 No routine matching `{routine_slug}`. Try /routines.")
    except Exception as _e:
        send_message(chat_id, f"🐬 Routine error: {str(_e)[:200]}")
    return


def _handle_new(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    import core.memory.memory as _mem_new
    import prompt_logger as _pl_new
    _mem_new.clear_chat(str(chat_id))
    _pl_new.clear_chat_logs(str(chat_id))
    send_message(chat_id, "🐬 Fresh start — conversation cleared. What's on your mind?")
    return


def _handle_fresh(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    import core.memory.memory as _mem_fresh
    import prompt_logger as _pl_fresh
    result = _mem_fresh.fresh_chat(str(chat_id))
    _pl_fresh.clear_chat_logs(str(chat_id))
    msgs = result.get("messages_deleted", 0)
    atts = result.get("attachments_deleted", 0)
    send_message(chat_id, f"🐬 Factory reset complete — {msgs} messages and {atts} attachments cleared. Clean slate!")
    return


def _handle_session(chat_id, text) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    import core.memory.memory as _mem_sess
    sub_sess = text[len("/session"):].strip().lower()
    if sub_sess in ("new", "split"):
        result = _mem_sess.rotate_session(str(chat_id))
        if "error" in result:
            send_message(chat_id, f"❌ Session rotation failed: {result['error']}")
        else:
            prev = result.get("prev_session_id", "?")
            num = result.get("session_number", "?")
            send_message(chat_id,
                f"🐬 New session #{num} started.\n"
                f"Previous session: `{prev}`\n"
                f"Use `recall_memory('{prev}')` to pull context from it."
            )
    else:
        send_message(chat_id,
            "📋 *Session commands*\n\n"
            "`/session new` — start a fresh session (keeps old history accessible via recall_memory)\n"
            "`/fresh` — factory reset (deletes all history for this chat)"
        )
    return


def _handle_voice_clone(chat_id, text) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    samples = _list_voice_samples()
    active_label = os.path.basename(bm._active_voice_sample).replace(".wav","").replace(".mp3","").replace(".m4a","")
    msg = (
        "\U0001f3a4 *Voice Clone*\n"
        "Active: `" + active_label + "`\n\n"
        "Usage:\n"
        "`/voice-clone <text>` \u2014 active sample\n"
        "`/voice-clone ricky <text>` \u2014 ricky sample\n\n"
        "Or tap a sample below to switch:"
    )
    if samples:
        buttons = [[{"text": ("✅ " if s["name"] == active_name else "") + s["name"], "callback_data": f"set_voice_{i}"}] for i, s in enumerate(samples)]
        send_buttons(chat_id, msg, buttons)
    else:
        send_message(chat_id, msg)
    return


def _handle_voices(chat_id, text) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    if text.startswith("set_voice_"):
        # callback from inline button: set_voice_<index>
        try:
            idx = int(text.split("_")[-1])
            samples = _list_voice_samples()
            if 0 <= idx < len(samples):
                bm._active_voice_sample = samples[idx]["path"]
                send_message(chat_id, f"🎤 Active voice: *{samples[idx]['name']}*")
            else:
                send_message(chat_id, "❌ Invalid voice index.")
        except Exception as e:
            send_message(chat_id, f"❌ Error: {e}")
        return
    samples = _list_voice_samples()
    if not samples:
        send_message(chat_id, "🐬 No voice samples found in ~/.openclaw/media/voice-samples/")
        return
    active_name = os.path.basename(bm._active_voice_sample).replace(".wav", "").replace(".mp3", "").replace(".m4a", "")
    msg = f"🎤 *Available voices*\nActive: `{active_name}`\n\nTap to switch:"
    buttons = [[{"text": ("\u2705 " if s["name"] == active_label else "") + s["name"], "callback_data": f"set_voice_{i}"}] for i, s in enumerate(samples)]
    send_buttons(chat_id, msg, buttons)
    return


def _handle_start_help(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _list_voice_samples = bm._list_voice_samples
    _get_current_version = bm._get_current_version
    TypingKeepAlive = bm.TypingKeepAlive
    __version__ = _get_current_version()
    intro = (
        f"🐬 *Kernel Evo v{__version__}*\n"
        f"Self-evolving AI agent — pick your mode:\n\n"
        f"🏠 **/local** — load Nemotron-3B locally for inference. "
        f"STT/vision/voice all work. Skill synthesis and planning go to cloud.\n\n"
        f"☁️ **/cloud** — everything runs through cloud providers. "
        f"Zero local VRAM used. No STT/voice/vision (no local model loaded).\n\n"
        f"🎛️ **/models** — pick a provider and its model (local pulls on demand).\n"
        f"🗺️ **/provider** — routing table for each call type."
    )
    buttons = [
        # ─ Mode selection
        [{"text": "🏠 /local — load Nemotron", "callback_data": "/local"},
         {"text": "☁️ /cloud — all cloud", "callback_data": "/cloud"}],
        # ─ Models & Provider
        [{"text": "🎛️ Models", "callback_data": "/models"},
         {"text": "🗺️ Provider", "callback_data": "/provider"}],
        # ─ Chat & Inference
        [{"text": "🧠 Status", "callback_data": "/status"}, {"text": "💭 Thoughts", "callback_data": "/thoughts"}],
        # ─ Skills & Routines
        [{"text": "🔧 Skills", "callback_data": "/skills"}, {"text": "⚙️ Routines", "callback_data": "/routines"}],
        # ─ Replicas
        [{"text": "🤖 List replicas", "callback_data": "/replica list"}, {"text": "➕ Spawn", "callback_data": "/replica spawn"}, {"text": "👥 Clone", "callback_data": "/replica clone"}],
        [{"text": "⏹ Stop replica", "callback_data": "/replica stop"}, {"text": "🗂️ Workspaces", "callback_data": "/workspaces"}],
        # ─ Evolution
        [{"text": "🧬 Evolve", "callback_data": "/evolve"}, {"text": "📊 Evolve status", "callback_data": "/evolve status"}, {"text": "📈 Evolve dash", "callback_data": "/evolve dash"}],
        # ─ System
        [{"text": "📊 Status", "callback_data": "/status"}, {"text": "🔢 Version", "callback_data": "/version"}, {"text": "🖥️ System", "callback_data": "/system"}],
        [{"text": "🚀 Init workspace", "callback_data": "/init"}],
        # ─ Maintenance
        [{"text": "🔄 Update", "callback_data": "/update"}, {"text": "🔁 Restart", "callback_data": "/restart"}, {"text": "⏪ Rollback", "callback_data": "/rollback"}],
        [{"text": "🎤 Voices", "callback_data": "/voices"}, {"text": "🆕 New conversation", "callback_data": "/new"}],
        # ─ Repo access
        [{"text": "🔒 Private repos", "callback_data": "/private_repo show"}, {"text": "🧯 Clone repo", "callback_data": "/clone"}],
    ]
    send_buttons(chat_id, intro, buttons)
    return


def _handle_skills(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _ensure_agent()
    import yaml

    with open(CONFIG_PATH) as _f:
        _cfg = yaml.safe_load(_f)
    from core.skills import load_all

    skills = load_all(
        str(os.path.expanduser(_cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
    )
    if not skills:
        send_message(chat_id, "🔧 No skills loaded.")
    else:
        # Send as inline buttons (2 per row)
        buttons = []
        row = []
        for s in skills:
            row.append(
                {"text": s["name"], "callback_data": f"skill_info_{s['name']}"}
            )
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        send_buttons(
            chat_id, f"🔧 *Skills ({len(skills)}) — tap to learn more:*", buttons
        )
    return


def _handle_routines(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _ensure_agent = bm._ensure_agent
    _ensure_agent()
    import yaml

    with open(CONFIG_PATH) as _f:
        _cfg = yaml.safe_load(_f)
    from core.routines import load_all

    routines = load_all(
        str(os.path.expanduser(_cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
    )
    if not routines:
        send_message(chat_id, "⚙️ No routines loaded.")
    else:
        # Send as inline buttons — one per row with Run button
        buttons = []
        for r in routines:
            buttons.append(
                [
                    {
                        "text": f"⚙️ {r['name']}",
                        "callback_data": f"routine_info_{r['name']}",
                    },
                    {"text": "▶ Run", "callback_data": f"routine_run_{r['name']}"},
                ]
            )
        send_buttons(chat_id, f"⚙️ *Routines ({len(routines)}):*", buttons)
    return


def _handle_packages(chat_id) -> None:
    bm = _bot_module()
    send_buttons = bm.send_buttons
    _esc = bm._esc
    from bootstrap import ECOSYSTEM_ROOT
    import yaml
    skills_list = []
    routines_list = []
    for tier_dir in sorted(ECOSYSTEM_ROOT.iterdir()):
        manifest = tier_dir / "manifest.yaml"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            data = yaml.safe_load(f) or {}
        tier = tier_dir.name
        for s in data.get("skills", []):
            skills_list.append((s["name"], s.get("description", ""), tier))
        for r in data.get("routines", []):
            routines_list.append((r["name"], r.get("description", ""), tier))

    lines_out = [f"📦 *Packages — {len(skills_list)} skills · {len(routines_list)} routines*\n"]
    lines_out.append("\n*Skills:*")
    for name, desc, tier in skills_list:
        lines_out.append(f"  • `{name}` — {_esc(desc)} _[{_esc(tier)}]_")
    lines_out.append("\n*Routines:*")
    for name, desc, tier in routines_list:
        lines_out.append(f"  • `{name}` — {_esc(desc)} _[{_esc(tier)}]_")

    buttons = []
    row = []
    for name, desc, tier in skills_list:
        row.append({"text": f"📥 {name}", "callback_data": f"install_skill_{name}"})
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    row = []
    for name, desc, tier in routines_list:
        row.append({"text": f"📥 {name}", "callback_data": f"install_routine_{name}"})
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    send_buttons(chat_id, "\n".join(lines_out), buttons)
    return


def _handle_status(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    _get_current_version = bm._get_current_version
    _agent_ready = bm._agent_ready
    __version__ = _get_current_version()

    # ── Live model-server health (model, active adapter, VRAM, slots) ──
    _mc_unreachable = False
    try:
        from core.inference.model_client import health as _mc_health
        _mh = _mc_health()
        _mc_unreachable = not _mh or "error" in _mh
    except Exception:
        _mh = {}
        _mc_unreachable = True

    _model_label = _mh.get("model", "unknown")
    if _mh.get("nemotron"):
        _nmode = _mh.get("nemotron_mode", "ar")
        _model_label = f"{_model_label} [Nemotron/{_nmode}]"
    if _mc_unreachable:
        _model_label = "unknown (model server unreachable)"

    _adapter_label = _mh.get("adapter") or "(none)"
    _main_loaded = _mh.get("main_model_loaded")
    _drafter = "✅" if _mh.get("drafter_loaded") else "—"
    _audio = "✅" if _mh.get("audio_capable") else "—"
    _vram_free = _mh.get("vram_free_mb")
    if _vram_free is None:
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _free, _total = _torch.cuda.mem_get_info()
                _vram_free = _free // (1024 * 1024)
        except Exception:
            pass

    # ── Effective serving route: local vs cloud (provider + model) ────
    # get_provider() applies thermal fallback live, so this reflects what is
    # ACTUALLY serving task_inference right now, not just the configured value.
    _route_kind = "local"
    _route_provider = "local"
    _route_model = None
    try:
        from core.inference.provider import get_provider as _gp_s
        _prov = _gp_s()
        _route_provider = _prov.get_provider("task_inference")
        _route_model = _prov.get_model(_route_provider, "task_inference")
        _route_kind = "local" if _route_provider == "local" else "cloud"
    except Exception:
        _route_provider = "unknown"
    _route_emoji = "💻" if _route_kind == "local" else "☁️"
    if _route_kind == "local":
        _route_line = f"Serving: {_route_emoji} local"
    else:
        _route_line = (f"Serving: {_route_emoji} cloud ({_route_provider})")
        if _route_model:
            _route_line += f"\nCloud model: {_route_model}"

    # ── Evolution / finetune-gate state ──────────────────────────────
    _gate_state = {}
    try:
        _gsf = Path.home() / ".kernel-evolving/workspace/data/finetune_gate_state.json"
        if _gsf.exists():
            _gate_state = json.loads(_gsf.read_text())
    except Exception:
        _gate_state = {}
    _last_ft = (_gate_state.get("last_finetune_ts") or "never")[:16]

    _traj_total = _traj_clean = None
    try:
        import sqlite3 as _sq
        _db = Path.home() / ".kernel-evolving/workspace/data/evolution.db"
        if _db.exists():
            _con = _sq.connect(str(_db))
            _traj_total = _con.execute("SELECT COUNT(*) FROM task_trajectories").fetchone()[0]
            _traj_clean = _con.execute(
                "SELECT COUNT(*) FROM task_trajectories WHERE critic_score >= 0.7"
            ).fetchone()[0]
            _con.close()
    except Exception:
        pass

    _gate_running = "✅"
    try:
        if subprocess.run(["systemctl", "--user", "is-active", "auto-finetune-gate.service"],
                          capture_output=True, text=True).stdout.strip() != "active":
            _gate_running = "⛔"
    except Exception:
        _gate_running = "?"

    # ── Slots / replicas ─────────────────────────────────────────────
    _slots = _mh.get("slots") or []
    _slot_line = ", ".join(s.get("name", s) if isinstance(s, dict) else str(s) for s in _slots) or "—"

    update_note = f"\n🆕 Update available: {bm._latest_version}" if bm._latest_version and bm._latest_version != __version__ else ""
    _loaded_txt = (f"\nModel loaded: {'✅' if _main_loaded else '⏳ lazy (first message loads it)'}")
    _traj_line = (f"\nTrajectories: {_traj_total} total · {_traj_clean} clean (≥0.7)"
                  if _traj_total is not None else "\nTrajectories: n/a")

    send_message(
        chat_id,
        (
            f"🐬 *Kernel Evo Status*\n"
            f"Version: v{__version__}{update_note}\n"
            f"🧠 Model: {_model_label}{_loaded_txt}\n"
            f"Adapter: {_adapter_label}\n"
            f"{_route_line}\n"
            f"VRAM free: {_vram_free}MB\n"
            f"Drafter: {_drafter} · Audio: {_audio}\n"
            f"Slots: {_slot_line}\n"
            f"Gate: {_gate_running} · last fine-tune: {_last_ft}\n"
            f"{_traj_line}\n"
            f"Ready: {'✅' if _agent_ready else '⏳ loading on first message'}"
        ),
    )
    return



def _handle_local(chat_id) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    _ensure_agent = bm._ensure_agent
    _ensure_agent()
    import yaml as _yaml, json as _json_p
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    try:
        import urllib.request as _ur
        body = {
            "task_inference": "local",
            "synthesis": "openai",
            "critic": "openrouter",
            "planning": "openrouter",
            "trajectory_teacher": "openai",
            "persist": True,
        }
        payload = _json_p.dumps(body).encode()
        req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                          data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=5) as r:
            result = _json_p.loads(r.read())
        send_message(chat_id, f"🏠 *Local mode activated* — task inference routed to Nemotron (will load on first request).")
    except Exception as e:
        send_message(chat_id, f"\u274c Mode switch error: {e}")
    return


def _handle_models(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    edit_message = bm.edit_message
    _register_repo_token = bm._register_repo_token
    _resolve_repo_token = bm._resolve_repo_token
    _curated_slot_for_repo = bm._curated_slot_for_repo
    parts = text.split(None, 2)
    sub = parts[1].lower() if len(parts) > 1 else ""

    from pathlib import Path as _Path
    from core.hf_cache import resolve_hf_hub_dir, ensure_hf_home_env

    # MS4: shared with model_server.py so the bot's "is this downloaded?"
    # check can't silently disagree with what the server can actually load.
    ensure_hf_home_env()
    _hub_dir = resolve_hf_hub_dir()

    def _cache_prefix(repo_id: str) -> str:
        return f"models--{repo_id.replace('/', '--')}"

    # Build from config.yaml model_catalog; fall back to a hardcoded minimal set
    # so the menu still works if the config key is missing.
    try:
        import yaml as _yaml_mc
        _cfg_models = _yaml_mc.safe_load(open(CONFIG_PATH)) or {}
        _raw_catalog = _cfg_models.get("model_catalog", [])
        KNOWN_MODELS = {
            entry["key"]: {
                "label": entry["label"],
                "repo_id": entry["repo_id"],
                "cache_prefix": _cache_prefix(entry["repo_id"]),
                "drafter": entry.get("drafter", ""),
                **({"note": entry["note"]} if entry.get("note") else {}),
            }
            for entry in _raw_catalog if entry.get("key") and entry.get("repo_id")
        }
    except Exception:
        _cfg_models = {}
        KNOWN_MODELS = {}

    def _collect_path_values(obj):
        values = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    values.extend(_collect_path_values(v))
                elif isinstance(v, str) and ("path" in str(k).lower() or v.startswith("/") or v.startswith("~")):
                    values.append(v)
        elif isinstance(obj, list):
            for item in obj:
                values.extend(_collect_path_values(item))
        return values

    _configured_paths = []
    for _p in _collect_path_values(_cfg_models):
        try:
            _configured_paths.append(_Path(_p).expanduser())
        except Exception:
            pass

    def _is_downloaded(info: dict) -> bool:
        if any(_hub_dir.glob(f"{info['cache_prefix']}*")):
            return True

        target = info["cache_prefix"]
        for p in _configured_paths:
            try:
                if p.exists() and target in str(p):
                    return True
            except Exception:
                continue
        return False

    if sub == "" or sub == "menu":
        # Unified model manager: show active provider/model + provider selector.
        try:
            import core.inference.model_client as _mc
            h = _mc.health()
            current = h.get("model", "unknown")
            vram = h.get("vram_free_mb", 0)
        except Exception:
            current, vram = "unknown", 0

        _prov_data = {}
        try:
            import urllib.request as _ur2
            with _ur2.urlopen(f"http://localhost:{8779}/provider", timeout=3) as _pr:
                _prov_data = json.loads(_pr.read())
                _task_provider = _prov_data.get("routing", {}).get("task_inference", {}).get("provider", "?")
                _task_model = _prov_data.get("routing", {}).get("task_inference", {}).get("model", "")
        except Exception:
            _task_provider, _task_model = "?", ""

        _provider_icons = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                           "hf": "\U0001f917", "copilot": "\u26a1", "openrouter": "\U0001f310"}
        _prov_icon = _provider_icons.get(_task_provider, "\U0001f4e1")

        lines = [
            f"\U0001f916 *Model Manager*\n",
            f"{_prov_icon} Task inference: `{_task_provider}`" + (f" / `{_task_model}`" if _task_model else ""),
            f"\U0001f4be Local model loaded: `{current}`",
            f"\U0001f5a5 VRAM free: `{vram} MB`\n",
            "*Select a provider to pick its model:*",
        ]
        buttons = [
            # Provider selector — each opens that provider's model list
            [{"text": "\U0001f3e0 Local", "callback_data": "/models provider local"},
             {"text": "\U0001f916 OpenAI", "callback_data": "/models provider openai"}],
            [{"text": "\U0001f9e0 Anthropic", "callback_data": "/models provider anthropic"},
             {"text": "\U0001f917 HF", "callback_data": "/models provider hf"}],
            [{"text": "\u26a1 Copilot", "callback_data": "/models provider copilot"},
             {"text": "\U0001f310 OpenRouter", "callback_data": "/models provider openrouter"}],
            # Local management
            [{"text": "\U0001f504 Reload current", "callback_data": "/models reload"},
             {"text": "\U0001f50d Search Hub", "callback_data": "/models search "}],
        ]
        # If Nemotron is active, add mode-switch buttons
        if h.get("nemotron"):
            cur_mode = h.get("nemotron_mode", "?")
            lines.append(f"\nNemotron mode: `{cur_mode}` (block={h.get('nemotron_block_length',32)})")
            buttons.append([
                {"text": "AR",          "callback_data": "/models mode ar"},
                {"text": "Diffusion",   "callback_data": "/models mode diffusion"},
                {"text": "\u26a1 linear_spec", "callback_data": "/models mode linear_spec"},
            ])
        send_buttons(chat_id, "\n".join(lines), buttons)
        return

    elif sub == "provider" and len(parts) >= 3:
        # /models provider <name> — show that provider's selectable models.
        provider = parts[2].lower()
        if provider == "local":
            # Local picker: actual pulled models (selectable) + curated unpulled (pull).
            try:
                import urllib.request as _ur_l
                # Pulled local models from /models
                local_models = []
                try:
                    with _ur_l.urlopen(f"http://localhost:{8779}/models", timeout=5) as _rm:
                        local_models = (json.loads(_rm.read()) or {}).get("models", [])
                except Exception:
                    pass
                # Curated catalog from /models/curated (marks which are downloaded)
                curated = []
                try:
                    with _ur_l.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
                        curated = (json.loads(_rc.read()) or {}).get("curated", [])
                except Exception:
                    pass
                downloaded_ids = {m.get("model", "").lower() for m in local_models}

                lines = ["\U0001f3e0 *Local Models*\n"]
                buttons = []

                # 1) Downloaded models — always surfaced, never empty.
                pulled = [m for m in local_models if m.get("model")]
                if pulled:
                    lines.append("*✅ Downloaded (tap to use):*")
                    for m in pulled:
                        repo = m.get("model", "")
                        slot = m.get("slot") or ""
                        lines.append(f"\u2705 `{repo}`" + (f" \u2192 slot `{slot}`" if slot else ""))
                        tok = _register_repo_token(repo)
                        buttons.append([{"text": f"\U0001f504 Use {repo}", "callback_data": f"/models use {tok}"}])
                else:
                    lines.append("*\u2139\ufe0f No models downloaded yet.*\n_Pull one below or search the Hub._")

                # 2) Curated models not yet pulled → pull.
                unpulled = [c for c in curated if c.get("repo_id", "").lower() not in downloaded_ids]
                if unpulled:
                    lines.append("\n*📥 Not downloaded (pull to use):*")
                    for cm in unpulled:
                        repo = cm.get("repo_id", "")
                        label = cm.get("label") or repo
                        lines.append(f"\u23ec `{repo}`")
                        tok = _register_repo_token(repo)
                        buttons.append([{"text": f"\u23ec Pull {label}", "callback_data": f"/models pull {tok}"}])

                # 3) Search Hub to find + download more locally.
                buttons.append([
                    {"text": "\U0001f50d Search Hub", "callback_data": "/models search "},
                    {"text": "\U0001f504 Refresh", "callback_data": "/models provider local"},
                    {"text": "\u2190 Back", "callback_data": "/models"},
                ])
                send_buttons(chat_id, "\n".join(lines), buttons)
            except Exception as e:
                send_message(chat_id, f"\u274c Local models error: {e}")
            return

        # Cloud provider: show that provider's available models from /provider/models.
        try:
            import urllib.request as _ur_c
            with _ur_c.urlopen(f"http://localhost:{8779}/provider/models?provider={provider}&capability=text", timeout=8) as _rc:
                data = json.loads(_rc.read())
            models = (data.get("models", {}) or {}).get(provider, [])
            if not models:
                send_message(chat_id, f"\u274c No models listed for provider `{provider}` (or it's unavailable).")
                return
            _provider_icons = {"openai": "\U0001f916", "anthropic": "\U0001f9e0", "hf": "\U0001f917",
                               "copilot": "\u26a1", "openrouter": "\U0001f310"}
            icon = _provider_icons.get(provider, "\U0001f4e1")
            lines = [f"{icon} *{provider.title()} models*\n", "_Tap a model to route task_inference to it:_"]
            buttons = []
            for m in models[:12]:
                lines.append(f"\U0001f4e6 `{m}`")
                buttons.append([{"text": f"\U0001f504 Use {m}", "callback_data": f"/provider set task_inference {provider} persist|model|{m}"}])
            buttons.append([{"text": "\u2190 Back", "callback_data": "/models"}])
            send_buttons(chat_id, "\n".join(lines), buttons)
        except Exception as e:
            send_message(chat_id, f"\u274c Cloud models error: {e}")
        return

    elif sub == "use" and len(parts) >= 3:
        # /models use <token> — set task_inference to local + load the pulled model.
        repo_id = _resolve_repo_token(parts[2].strip())
        slot = _curated_slot_for_repo(repo_id)
        try:
            import urllib.request as _ur_u
            # Assign to slot if we know one, then route task_inference to local.
            if slot:
                apayload = json.dumps({"repo_id": repo_id, "slot": slot}).encode()
                areq = _ur_u.Request(f"http://localhost:{8779}/models/assign",
                                     data=apayload, headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with _ur_u.urlopen(areq, timeout=5):
                        pass
                except Exception:
                    pass
            body = {"task_inference": "local", "persist": True}
            payload = json.dumps(body).encode()
            req = _ur_u.Request(f"http://localhost:{8779}/provider/set",
                                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur_u.urlopen(req, timeout=5) as r:
                result = json.loads(r.read())
            send_message(chat_id, f"\u2705 Task inference \u2192 local (`{repo_id}`)" + (f" \u2192 slot `{slot}`" if slot else ""))
        except Exception as e:
            send_message(chat_id, f"\u274c Use failed: {e}")
        return

    elif sub == "load" and len(parts) >= 3:
        key = parts[2].lower()
        if key not in KNOWN_MODELS:
            send_message(chat_id, f"\u274c Unknown model key `{key}`. Valid: {', '.join(KNOWN_MODELS)}")
            return
        info = KNOWN_MODELS[key]
        if not _is_downloaded(info):
            send_message(chat_id, f"\u274c Model not downloaded yet in `{_hub_dir}` or configured model paths for `{info['repo_id']}`\nUse `/models download {key}` first.")
            return
        working_id = send_message(chat_id, f"\u23f3 Loading {info['label']}\u2026\n_Step 1/4: starting model server_")
        try:
            import core.inference.model_client as _mc
            import urllib.request as _ur_m
            import subprocess as _sp
            import time as _time

            # Step 1 — ensure model server is running (start it if not)
            if not _mc.is_server_running():
                edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n_Step 1/4: model server not running — starting..._")
                _sp.Popen(
                    [sys.executable,
                     str((Path(__file__).resolve().parent / 'model_server.py')),
                     '--config', CONFIG_PATH, '--lazy'],
                    stdout=open('/tmp/kernel_evolving_model_server.log', 'a'),
                    stderr=_sp.STDOUT,
                    start_new_session=True,
                )
                # Wait up to 15s for socket
                for _ in range(15):
                    _time.sleep(1)
                    if _mc.is_server_running():
                        break
                if not _mc.is_server_running():
                    edit_message(chat_id, working_id, "\u274c Model server failed to start. Check /tmp/kernel_evolving_model_server.log")
                    return
                edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 1/4: model server started_\n_Step 2/4: unloading current model from VRAM_")
            else:
                edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 1/4: model server already running_\n_Step 2/4: unloading current model from VRAM_")

            # Step 2 — unload current model to free VRAM
            if _mc.is_server_running():
                unload_result = _mc.unload()
                freed = unload_result.get("freed_mb", 0)
                edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 2/4: freed ~{freed}MB VRAM_\n_Step 3/4: loading {info['label']} (~30-60s)_")

            # Step 3 — load the new model via swap_model
            result = _mc.swap_model(info["repo_id"], drafter_path=info["drafter"] or "")
            if "error" in result:
                edit_message(chat_id, working_id, f"\u274c Swap failed: {result['error']}")
                return

            # Step 4 — switch task_inference to local
            try:
                req_body = json.dumps({"task_inference": "local"}).encode()
                req = _ur_m.Request(f"http://localhost:{8779}/provider/set",
                                    data=req_body, headers={"Content-Type": "application/json"}, method="POST")
                _ur_m.urlopen(req, timeout=5)
                edit_message(chat_id, working_id,
                    f"\u2705 *{info['label']}* loaded\n"
                    f"Model: `{result.get('model','?')}` | Backend: `{result.get('backend','?')}`\n"
                    f"Task inference: `local` (auto-switched)")
            except Exception as _pe:
                edit_message(chat_id, working_id,
                    f"\u2705 *{info['label']}* loaded\nModel: `{result.get('model','?')}`\n"
                    f"\u26a0\ufe0f Provider switch failed: {_pe}")
        except Exception as e:
            edit_message(chat_id, working_id, f"\u274c Exception: {e}")
        return

    elif sub == "reload":
        working_id = send_message(chat_id, "\u23f3 Reloading current model\u2026")
        try:
            import yaml as _yaml
            from core.inference import model_client as _mc
            cfg = _yaml.safe_load(open(CONFIG_PATH))
            path = cfg["model"].get("path") or cfg["model"].get("name", "")
            drafter = cfg["model"].get("drafter_path") or ""
            result = _mc.swap_model(path, drafter_path=drafter)
            if "error" in result:
                edit_message(chat_id, working_id, f"\u274c Reload failed: {result['error']}")
            else:
                edit_message(chat_id, working_id, f"\u2705 Reloaded: `{result.get('model','?')}`")
        except Exception as e:
            edit_message(chat_id, working_id, f"\u274c Exception: {e}")
        return

    elif sub == "download":
        key = parts[2].lower() if len(parts) >= 3 else ""
        if key not in KNOWN_MODELS:
            send_message(chat_id, f"\u274c Unknown model key. Valid: {', '.join(KNOWN_MODELS)}")
            return
        info = KNOWN_MODELS[key]
        send_message(chat_id, f"\u23ec Downloading {info['label']}\u2026 This runs in background, I'll ping you when done.")
        import threading as _thr
        def _do_download(chat_id=chat_id, info=info, key=key):
            try:
                from huggingface_hub import snapshot_download
                from core.hf_cache import resolve_hf_cache_root
                import os as _os
                token = _os.environ.get("HF_TOKEN", "")
                repo_id = info["repo_id"]
                _hf_home = resolve_hf_cache_root()
                snapshot_download(repo_id=repo_id, cache_dir=_hf_home, token=token, ignore_patterns=["*.gguf"])
                send_message(chat_id, f"\u2705 Downloaded {info['label']}\nCache: `{_hf_home}`\nUse `/models load {key}` to switch.")
            except Exception as e:
                send_message(chat_id, f"\u274c Download failed: {e}")
        _thr.Thread(target=_do_download, daemon=True).start()
        return

    elif sub == "mode" and len(parts) >= 3:
        # /models mode ar|diffusion|linear_spec  — switch Nemotron generation mode at runtime
        new_mode = parts[2].lower()
        valid_modes = ("ar", "diffusion", "linear_spec")
        if new_mode not in valid_modes:
            send_message(chat_id, f"\u274c Invalid mode. Valid: {', '.join(valid_modes)}")
            return
        try:
            import core.inference.model_client as _mc
            h = _mc.health()
            if not h.get("nemotron"):
                send_message(chat_id, "\u274c Nemotron is not the active model. Load it first with `/models load nemotron3b`.")
                return
            # Update config live via /provider API trick — write to config.yaml
            import yaml as _yaml
            with open(CONFIG_PATH) as _f:
                _cfg_live = _yaml.safe_load(_f)
            _cfg_live.setdefault("model", {})["generation_mode"] = new_mode
            with open(CONFIG_PATH, "w") as _f:
                _yaml.dump(_cfg_live, _f, default_flow_style=False, allow_unicode=True)
            # Reload the model server so it picks up new mode
            working_id = send_message(chat_id, f"\u23f3 Switching Nemotron mode \u2192 `{new_mode}`\u2026 (reloading model ~30s)")
            path = _cfg_live["model"].get("path") or _cfg_live["model"].get("name")
            result = _mc.swap_model(path)
            if "error" in result:
                edit_message(chat_id, working_id, f"\u274c Mode switch failed: {result['error']}")
            else:
                edit_message(chat_id, working_id,
                    f"\u2705 Nemotron mode: `{new_mode}`\n"
                    f"block_length={_cfg_live['model'].get('block_length', 32)} "
                    f"threshold={_cfg_live['model'].get('threshold', 0.9)}")
        except Exception as e:
            send_message(chat_id, f"\u274c {e}")
        return

    elif sub == "search":
        # /models search <query> — search HuggingFace Hub via /hub/search
        query = " ".join(parts[2:]).strip()
        try:
            import urllib.request as _ur_s, urllib.parse as _up
            url = f"http://localhost:{8779}/hub/search?q={_up.quote(query)}&limit=10"
            with _ur_s.urlopen(url, timeout=15) as _r:
                data = json.loads(_r.read())
            results = data.get("models", [])
            if not results:
                send_message(chat_id, f"\U0001f50d No Hub results for `{query}`")
                return
            lines = [f"\U0001f50d *Hub Search: {query}*\n"]
            buttons = []
            for m in results[:10]:
                mid = m.get("id", "")
                tag = m.get("pipeline_tag") or "unknown"
                dl = m.get("downloads") or 0
                lines.append(f"\U0001f4e6 `{mid}`\n   _{tag} \u00b7 {dl} downloads_")
                # Use a short token to stay under Telegram's 64-byte callback limit.
                tok = _register_repo_token(mid)
                buttons.append([{"text": f"\u23ec Pull {mid}", "callback_data": f"/models pull {tok}"}])
            buttons.append([{"text": "\U0001f9e0 Back to models", "callback_data": "/models"}])
            send_buttons(chat_id, "\n".join(lines), buttons)
        except Exception as e:
            send_message(chat_id, f"\u274c Hub search failed: {e}")
        return

    elif sub == "pull":
        # /models pull <repo_id|token> — pull an arbitrary model via /pull (background)
        raw = " ".join(parts[2:]).strip()
        if not raw:
            send_message(chat_id, "Usage: `/models pull <repo_id>` e.g. `/models pull Qwen/Qwen2.5-Omni-3B`")
            return
        # Resolve a short callback token back to the full repo id.
        repo_id = _resolve_repo_token(raw)
        try:
            import urllib.request as _ur_p
            payload = json.dumps({"model": repo_id}).encode()
            req = _ur_p.Request(f"http://localhost:{8779}/pull",
                                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur_p.urlopen(req, timeout=5) as _r:
                data = json.loads(_r.read())
            job_id = data.get("job_id", "")
            send_message(chat_id,
                f"\u23ec Pulling `{repo_id}` in background (job `{job_id[:8]}\u2026`)\n"
                f"I'll ping you when it finishes. Then assign to a slot with `/models assign {repo_id} <slot>`.")
            # Poll in a background thread and notify on completion.
            import threading as _thr_p
            def _watch(jid=job_id, rid=repo_id):
                import time as _tp
                for _ in range(600):
                    _tp.sleep(3)
                    try:
                        with _ur_p.urlopen(f"http://localhost:{8779}/jobs/{jid}", timeout=5) as _rj:
                            j = json.loads(_rj.read())
                        st = j.get("status", "running")
                        if st == "succeeded":
                            send_message(chat_id, f"\u2705 Pulled `{rid}`\nUse `/models assign {rid} <slot>` to assign.")
                            return
                        if st == "failed":
                            send_message(chat_id, f"\u274c Pull of `{rid}` failed: {j.get('error')}")
                            return
                    except Exception:
                        continue
                send_message(chat_id, f"\u23f3 Pull of `{rid}` still running \u2014 check later.")
            _thr_p.Thread(target=_watch, daemon=True).start()
        except Exception as e:
            send_message(chat_id, f"\u274c Pull request failed: {e}")
        return

    elif sub == "assign":
        # /models assign <repo_id> <slot> — assign a pulled model to a named slot
        parts_a = text.split()
        if len(parts_a) < 4:
            send_message(chat_id, "Usage: `/models assign <repo_id> <slot>` e.g. `/models assign Qwen/Qwen2.5-Omni-3B audio`")
            return
        repo_id = parts_a[2].strip()
        slot = parts_a[3].strip()
        try:
            import urllib.request as _ur_a
            payload = json.dumps({"repo_id": repo_id, "slot": slot}).encode()
            req = _ur_a.Request(f"http://localhost:{8779}/models/assign",
                                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur_a.urlopen(req, timeout=5) as _r:
                data = json.loads(_r.read())
            if data.get("error"):
                send_message(chat_id, f"\u274c Assign failed: {data['error']}")
                return
            send_message(chat_id, f"\u2705 Assigned `{repo_id}` \u2192 slot `{slot}`\nIt will lazy-load on first use.")
        except Exception as e:
            send_message(chat_id, f"\u274c Assign request failed: {e}")
        return

    else:
        send_message(chat_id, "Usage:\n"
            "`/models` — show menu\n"
            "`/models load e2b|e4b|nemotron3b` — hot-swap model (unloads current first)\n"
            "`/models mode ar|diffusion|linear_spec` — switch Nemotron generation mode\n"
            "`/models reload` — reload current\n"
            "`/models download nemotron3b` — download from HuggingFace\n"
            "`/models search <query>` — search HuggingFace Hub\n"
            "`/models pull <repo_id>` — pull any model from Hub (background)\n"
            "`/models assign <repo_id> <slot>` — assign pulled model to a model_slots entry")
        return


def _handle_private_repo(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    arg = text[13:].strip()
    import yaml
    cfg = yaml.safe_load(open(CONFIG_PATH))
    if not arg or arg == "show":
        current = cfg.get("ecosystem", {}).get("private", "(not set)")
        send_message(chat_id, f"🔒 Private ecosystem repo: `{current}`\n\nTo change: `/private_repo owner/repo`")
        return
    # Validate format
    if "/" not in arg or len(arg.split("/")) != 2:
        send_message(chat_id, "❌ Invalid format. Use: `/private_repo owner/repo`\nExample: `/private_repo myorg/my-kernel-skills`")
        return
    # Save to config
    if "ecosystem" not in cfg:
        cfg["ecosystem"] = {}
    cfg["ecosystem"]["private"] = arg
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    send_message(chat_id, f"✅ Private repo set to `{arg}`\nRe-bootstrapping ecosystem...")
    # Trigger re-bootstrap in background
    import threading
    def _rebootstrap():
        try:
            from bootstrap import bootstrap
            result = bootstrap(cfg)
            send_message(chat_id, f"✅ Bootstrap complete. Skills/routines updated.")
        except Exception as e:
            send_message(chat_id, f"⚠️ Bootstrap error: {str(e)[:200]}")
    threading.Thread(target=_rebootstrap, daemon=True).start()
    return


def _handle_clone(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    parts = text[6:].strip().split(None, 1)
    if not parts:
        send_message(chat_id, "Usage: /clone <path> [name]\nExample: /clone ~/.openclaw/workspace/skills/mental-map")
        return
    src_path = parts[0]
    name = parts[1] if len(parts) > 1 else None
    from bootstrap import clone_from_agent
    result = clone_from_agent(src_path, name)
    send_message(chat_id, result["message"])
    return


def _handle_search(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    query = text[7:].strip()
    if not query:
        send_message(chat_id, "Usage: /search <query>\nExample: /search tracker")
        return
    from bootstrap import search as eco_search
    results = eco_search(query)
    if not results:
        send_message(chat_id, f"🔍 No results for '{query}'.\nTry /search with a different term.")
        return
    lines = [f"🔍 *Results for '{query}':*\n"]
    for r in results[:10]:
        icon = "🔧" if r["type"] == "skill" else "⚙️"
        lines.append(f"{icon} *{r['name']}* ({r['source']})\n   {r['description'] or 'No description'}")
    buttons = [[{"text": f"📥 Install {r['name']}", "callback_data": f"install_{r['type']}_{r['name']}"}] for r in results[:5]]
    send_buttons(chat_id, "\n".join(lines), buttons)
    return


def _handle_install(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    parts = text[8:].strip().split()
    if not parts:
        send_message(chat_id, "Usage: /install <name>\nExample: /install open-workspace-tracker")
        return
    name = parts[0]
    item_type = parts[1] if len(parts) > 1 else None
    from bootstrap import install as eco_install
    result = eco_install(name, item_type)
    send_message(chat_id, result["message"])
    return


def _handle_verbose(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    bm._verbose_mode = not bm._verbose_mode
    if bm._verbose_mode:
        send_message(chat_id, "🔍 Verbose mode ON — I'll show my reasoning.")
    else:
        send_message(chat_id, "🔇 Verbose mode OFF.")
    return


def _handle_replica(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _call_api = bm._call_api
    parts = text.split(None, 2)
    sub = parts[1] if len(parts) > 1 else "list"

    if sub == "clone":
        if len(parts) > 2:
            # /replica clone <agent_name> — spawn from agents dir
            agent_name = parts[2].strip().lower().replace('.md', '')
            agents_dir = Path.home() / '.openclaw' / 'workspace-client' / 'agents'
            brief_path = agents_dir / f"{agent_name}.md"
            if not brief_path.exists():
                available = [f.stem for f in agents_dir.glob('*.md')] if agents_dir.exists() else []
                send_message(chat_id, f"❌ Agent `{agent_name}` not found.\nAvailable: {', '.join(available) or 'none'}")
            else:
                result = _call_api("POST", "/replica/named", {
                    "name": agent_name,
                    "role": "custom",
                    "brief_path": str(brief_path)
                })
                if result and result.get("status") == "spawned":
                    send_message(chat_id, f"✅ Replica `{agent_name}` spawned with brief.\nChat: `/replica msg {agent_name} <message>`")
                else:
                    reason = result.get("reason", "unknown") if result else "unreachable"
                    send_message(chat_id, f"❌ Could not spawn: {reason}")
        else:
            # Show dynamic list of available agents as buttons
            agents_dir = Path.home() / '.openclaw' / 'workspace-client' / 'agents'
            agents = sorted([f.stem for f in agents_dir.glob('*.md')]) if agents_dir.exists() else []
            if not agents:
                send_message(chat_id, "No agents found in workspace-client/agents/")
            else:
                buttons = [[{"text": f"🤖 {a}", "callback_data": f"/replica clone {a}"}] for a in agents]
                send_buttons(chat_id, "*Clone an agent replica:*\nSelect an agent to spawn with their brief loaded:", buttons)

    elif sub == "list":
        active = _call_api("GET", "/replica/active") or []
        if not active:
            send_message(chat_id, "No active replicas.")
        else:
            lines = ["*Active replicas:*"]
            for r in active:
                status = "💬 persistent" if r.get("persistent") else ("✅ done" if r.get("done") else "⚙️ running")
                lines.append(f"• `{r['name']}` ({r['role']}) — {status}")
            send_message(chat_id, "\n".join(lines))

    elif sub == "spawn":
        if len(parts) > 2:
            # /replica spawn <name> [prompt]
            spawn_parts = parts[2].split(None, 1)
            name = spawn_parts[0]
            prompt = spawn_parts[1] if len(spawn_parts) > 1 else None
            body = {"name": name}
            if prompt:
                body["custom_prompt"] = prompt
            result = _call_api("POST", "/replica/named", body)
            if result and result.get("status") == "spawned":
                send_message(chat_id, f"✅ Replica `{name}` spawned and ready.\nSend messages to it with:\n`/replica msg {name} <your message>`")
            else:
                reason = result.get("reason", "unknown error") if result else "unreachable"
                send_message(chat_id, f"❌ Could not spawn replica: {reason}")
        else:
            send_message(chat_id, "Usage:\n`/replica spawn <name>` — spawn with default prompt\n`/replica spawn <name> <custom prompt>` — spawn with custom role\n\nExample:\n`/replica spawn analyst You are a data analyst.`")

    elif sub == "msg" and len(parts) > 2:
        msg_parts = parts[2].split(None, 1)
        name = msg_parts[0]
        user_msg = msg_parts[1] if len(msg_parts) > 1 else ""
        if not user_msg:
            send_message(chat_id, "Usage: `/replica msg <name> <message>`")
        else:
            result = _call_api("POST", f"/replica/{name}/message", {"message": user_msg})
            if result and "reply" in result:
                send_message(chat_id, f"🤖 *{name}:* {result['reply']}")
            else:
                send_message(chat_id, f"❌ Replica `{name}` not found or error.")

    elif sub == "stop" and len(parts) > 2:
        name = parts[2].strip()
        result = _call_api("DELETE", f"/replica/{name}")
        if result and result.get("status") == "stopped":
            send_message(chat_id, f"✅ Replica `{name}` stopped.")
        else:
            send_message(chat_id, f"❌ Could not stop `{name}`.")

    else:
        send_message(chat_id, "Usage:\n`/replica list` — show active replicas\n`/replica clone` — spawn from available agents (dynamic list)\n`/replica clone <name>` — spawn a specific agent\n`/replica spawn <name> [prompt]` — spawn with custom prompt\n`/replica msg <name> <message>` — chat with a replica\n`/replica stop <name>` — stop a named replica")
    return


def _handle_run(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    edit_message = bm.edit_message
    _ensure_agent = bm._ensure_agent
    run_arg = text[5:].strip()
    # Split into name + optional input (e.g. "/run olly-recovery check status")
    parts = run_arg.split(" ", 1)
    run_name = parts[0]
    run_input = parts[1] if len(parts) > 1 else ""

    _ensure_agent()
    import yaml
    import os as _os

    with open(CONFIG_PATH) as _f:
        _cfg = yaml.safe_load(_f)
    from core.routines import load_all as load_routines, find as find_routine, run as run_routine
    from core.skills import load_all as load_skills, find as find_skill, run as run_skill
    from core.inference.model import infer, infer_with_tools, _model
    from core.tools import TOOLS

    # Try routine first
    routines_dir = str(os.path.expanduser(_os.environ.get("ROUTINES_DIR") or _cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
    routines = load_routines(routines_dir)
    r = find_routine(run_name, routines)
    if r:
        working_id = send_message(chat_id, f"⚙️ *{r['name']}* — starting\u2026")
        log_lines = [f"⚙️ *{r['name']}*"]

        def _routine_step(n, tool_name, args, result):
            args_str = str(args)[:120]
            result_str = str(result)[:1200]
            log_lines.append(f"  *Step {n}* `{tool_name}`\n  ▸ `{args_str}`\n  ↳ {result_str}")
            snippet = "\n".join(log_lines[-6:])  # last 6 entries to stay within Telegram limit
            edit_message(chat_id, working_id, snippet)

        from evo_routine_executor import execute_routine as _execute_evo_routine
        workspace = _cfg.get("workspace", "~/.openclaw/workspace")
        with TypingKeepAlive(chat_id):
            result = _execute_evo_routine(
                r,
                workspace=workspace,
                step_callback=_routine_step,
            )
        log_lines.append(f"\n✅ *Done*")
        final = "\n".join(log_lines[-8:])
        if not edit_message(chat_id, working_id, final):
            send_message(chat_id, final)
        send_message(chat_id, f"📋 *Result:*\n{result[:3800]}")
        return

    # Try skill
    skills_dir = str(os.path.expanduser(_os.environ.get("SKILLS_DIR") or _cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
    skills = load_skills(skills_dir)
    s = find_skill(run_name, skills)
    if s:
        working_id = send_message(chat_id, f"🔧 *{run_name}* — starting\u2026")
        log_lines = [f"🔧 *{run_name}*"]

        def _skill_step(n, tool_name, args, result):
            args_str = str(args)[:120]
            result_str = str(result)[:1200]
            log_lines.append(f"  *Step {n}* `{tool_name}`\n  ▸ `{args_str}`\n  ↳ {result_str}")
            snippet = "\n".join(log_lines[-6:])
            edit_message(chat_id, working_id, snippet)

        user_input = run_input or f"Execute the {run_name} skill."
        with TypingKeepAlive(chat_id):
            result = run_skill(s, user_input, infer)
        log_lines.append(f"\n✅ *Done*")
        final = "\n".join(log_lines[-8:])
        if not edit_message(chat_id, working_id, final):
            send_message(chat_id, final)
        send_message(chat_id, f"📋 *Result:*\n{result[:3800]}")
        return

    send_message(chat_id, f"❌ No routine or skill named `{run_name}` found.\nTry /routines or /skills to see available options.")
    return


def _handle_provider(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    _vram_fit_mark = bm._vram_fit_mark
    _estimate_model_gb = bm._estimate_model_gb
    _curated_slot_for_repo = bm._curated_slot_for_repo
    _register_repo_token = bm._register_repo_token
    _resolve_repo_token = bm._resolve_repo_token
    import yaml as _yaml, json as _json_p
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    parts = text.split()
    sub = parts[1].strip().lower() if len(parts) > 1 else ""

    if sub == "" or sub == "menu":
        # Show routing table + inline swap buttons
        try:
            import urllib.request as _ur
            with _ur.urlopen(f"http://localhost:{8779}/provider", timeout=3) as r:
                data = json.loads(r.read())
            routing = data.get("routing", {})
            collect = data.get("collect_trajectories", False)
            lines = ["\U0001f500 *Provider Router*\n"]
            for ct, info in routing.items():
                icon = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                        "hf": "\U0001f917", "copilot": "\u26a1"}.get(info["provider"], "\U0001f4e1")
                stream = " \U0001f4e1" if info.get("streaming") else ""
                model = f" `{info['model']}`" if info.get("model") else ""
                lines.append(f"{icon} `{ct}` \u2192 *{info['provider']}*{model}{stream}")
            collect_icon = "\U0001f3af ON" if collect else "\u23f8 OFF"
            lines.append(f"\nTrajectory collection: {collect_icon}")
            msg = "\n".join(lines)
            buttons = [
                # Quick set all
                [{"text": "\U0001f310 All \u2192 OpenRouter", "callback_data": "/provider set all openrouter persist"},
                 {"text": "\U0001f916 All \u2192 OpenAI",    "callback_data": "/provider set all openai persist"},
                 {"text": "\U0001f3e0 All \u2192 Local",     "callback_data": "/provider set all local persist"}],
                # Swap per call type row 1
                [{"text": "\U0001f504 task \u2192 swap",      "callback_data": "/provider swap task_inference"},
                 {"text": "\U0001f9ec synth \u2192 swap",     "callback_data": "/provider swap synthesis"},
                 {"text": "\U0001f4cb critic \u2192 swap",    "callback_data": "/provider swap critic"}],
                # Swap per call type row 2
                [{"text": "\U0001f5fa plan \u2192 swap",      "callback_data": "/provider swap planning"},
                 {"text": "\U0001f393 teacher \u2192 swap",   "callback_data": "/provider swap trajectory_teacher"}],
                # Misc
                [{"text": "\u21ba Reset defaults",           "callback_data": "/provider reset"},
                 {"text": "\U0001f4ca Availability",         "callback_data": "/provider status"}],
                [{"text": f"\U0001f3af Collect: {'ON \u2192 off' if collect else 'OFF \u2192 on'}",
                  "callback_data": f"/provider collect {'false' if collect else 'true'}"}],
                [{"text": "\U0001f9e0 Models / Load Nemotron", "callback_data": "/models"}],
            ]
            send_buttons(chat_id, msg, buttons)
        except Exception as e:
            send_message(chat_id, f"\u274c Provider status error: {e}")

    elif sub == "status":
        try:
            import urllib.request as _ur
            with _ur.urlopen(f"http://localhost:{8779}/provider/available", timeout=3) as r:
                avail = json.loads(r.read())
            lines = ["\U0001f4ca *Provider Availability*\n"]
            for prov, info in avail.items():
                icon = "\u2705" if info["ready"] else "\u274c"
                lines.append(f"{icon} `{prov}` \u2014 {info['reason']}")
            send_message(chat_id, "\n".join(lines))
        except Exception as e:
            send_message(chat_id, f"\u274c {e}")

    elif sub == "reset":
        try:
            import urllib.request as _ur
            for ct in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
                payload = json.dumps({ct: _cfg.get("providers", {}).get(ct, "local"), "persist": True}).encode()
                req = _ur.Request(f"http://localhost:{8779}/provider/set",
                                  data=payload, headers={"Content-Type": "application/json"}, method="POST")
                _ur.urlopen(req, timeout=3)
            send_message(chat_id, "\u21ba Provider routing reset to config.yaml defaults")
        except Exception as e:
            send_message(chat_id, f"\u274c {e}")

    elif sub == "set" and len(parts) >= 4:
        # /provider set <calltype> <provider>  OR  /provider set all <provider>
        # Optionally with a trailing token from the model pickers:
        #   "persist|assign|<token>" — local: also assign the chosen model to its slot.
        #   "persist|model|<model>"  — cloud: also set a model_override for the call type.
        call_type_or_all = parts[2].lower()
        provider_name    = parts[3].lower()
        persist_change = any(p.lower() in ("persist", "--persist") for p in parts[4:])
        assign_repo = ""
        model_override = ""
        for p in parts[4:]:
            if "|assign|" in p:
                _tok = p.split("|assign|", 1)[1].strip()
                assign_repo = _resolve_repo_token(_tok)  # token → repo_id
            elif "|model|" in p:
                model_override = p.split("|model|", 1)[1].strip()
        try:
            import urllib.request as _ur
            valid_providers = {"local", "openai", "anthropic", "hf", "copilot", "openrouter"}
            if provider_name not in valid_providers:
                send_message(chat_id, f"\u274c Unknown provider `{provider_name}`. Valid: {', '.join(sorted(valid_providers))}")
                return
            if call_type_or_all == "all":
                body = {ct: provider_name for ct in ("task_inference","synthesis","critic","planning","trajectory_teacher")}
            else:
                body = {call_type_or_all: provider_name}
            # Apply a cloud model override if provided (e.g. from /models provider <cloud>).
            if model_override:
                body["model_override"] = {call_type_or_all: model_override}
            body["persist"] = persist_change
            payload = json.dumps(body).encode()
            req = _ur.Request(f"http://localhost:{8779}/provider/set",
                              data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(req, timeout=3) as r:
                result = json.loads(r.read())
            persisted_label = " (persisted)" if result.get("persisted") else " (runtime only)"
            reply = f"\u2705 Provider updated{persisted_label}: `{result['changed']}`"
            if model_override:
                reply += f"\n\U0001f4e6 Model override: `{model_override}`"

            # XP7: if a local model was chosen from the picker, assign it to its slot.
            if assign_repo and provider_name == "local":
                slot = _curated_slot_for_repo(assign_repo)
                if slot:
                    try:
                        apayload = json.dumps({"repo_id": assign_repo, "slot": slot}).encode()
                        areq = _ur.Request(f"http://localhost:{8779}/models/assign",
                                           data=apayload, headers={"Content-Type": "application/json"}, method="POST")
                        with _ur.urlopen(areq, timeout=5) as ar:
                            adata = json.loads(ar.read())
                        if adata.get("error"):
                            reply += f"\n\u26a0\ufe0f Assign to slot `{slot}` failed: {adata['error']}"
                        else:
                            reply += f"\n\u2705 Assigned `{assign_repo}` \u2192 slot `{slot}`"
                    except Exception as _ae:
                        reply += f"\n\u26a0\ufe0f Assign request failed: {_ae}"
                else:
                    reply += f"\n\u26a0\ufe0f No slot mapping for `{assign_repo}` \u2014 use `/models assign {assign_repo} <slot>`"
            send_message(chat_id, reply)
        except Exception as e:
            send_message(chat_id, f"\u274c {e}")

    elif sub == "swap" and len(parts) >= 3:
        # Inline button: show per-provider choice for one call type
        call_type = parts[2]
        buttons = [
            [
                {"text": "\U0001f3e0 local",       "callback_data": f"/models provider local"},
                {"text": "\U0001f916 openai",      "callback_data": f"/models provider openai"},
                {"text": "\U0001f9e0 anthropic",   "callback_data": f"/models provider anthropic"},
            ],[
                {"text": "\U0001f917 hf",          "callback_data": f"/models provider hf"},
                {"text": "\u26a1 copilot",         "callback_data": f"/models provider copilot"},
                {"text": "\U0001f310 openrouter",  "callback_data": f"/models provider openrouter"},
            ],[
                {"text": "\u2190 Back",            "callback_data": "/provider"},
            ]
        ]
        send_buttons(chat_id, f"Pick a provider for `{call_type}` \u2014 then choose its model:", buttons)

    elif sub == "local" and len(parts) >= 3:
        # XP7: local model picker — curated models shown as inline buttons,
        # with a VRAM-fit indicator, plus Hub search for RAM-compatible models.
        call_type = parts[2]
        try:
            import urllib.request as _ur_l, os as _os_l
            # Curated local models from the new /models/curated endpoint.
            curated = []
            try:
                with _ur_l.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
                    curated = (json.loads(_rc.read()) or {}).get("curated", [])
            except Exception:
                pass
            # Available VRAM (MB) from /health.
            vram_free = 0
            try:
                with _ur_l.urlopen(f"http://localhost:{8779}/health", timeout=5) as _rh:
                    vram_free = int((json.loads(_rh.read()) or {}).get("vram_free_mb", 0) or 0)
            except Exception:
                pass

            lines = [f"\U0001f3e0 *Local Model Picker* \u2014 `{call_type}`",
                     f"VRAM free: `{vram_free} MB`\n"]
            buttons = []
            for cm in curated:
                repo = cm.get("repo_id", "")
                label = cm.get("label") or repo
                dl = cm.get("downloaded", False)
                # Rough VRAM fit estimate: assume ~2 bytes/param + overhead;
                # flag models likely too big for current free VRAM.
                est_gb = _estimate_model_gb(repo)
                fit = _vram_fit_mark(est_gb, vram_free)
                icon = "\u2705" if dl else "\u23ec"
                lines.append(f"{icon} {label} `{repo}` {fit}")
                # Use a short token to stay under Telegram's 64-byte callback limit.
                tok = _register_repo_token(repo)
                buttons.append([{
                    "text": f"{icon} {label}",
                    "callback_data": f"/provider set {call_type} local persist|assign|{tok}",
                }])
            buttons.append([
                {"text": "\U0001f50d Search Hub for more", "callback_data": f"/models search "},
                {"text": "\u2190 Back", "callback_data": "/provider"},
            ])
            send_buttons(chat_id, "\n".join(lines), buttons)
        except Exception as e:
            send_message(chat_id, f"\u274c Local picker error: {e}")
        return

    elif sub == "collect" and len(parts) >= 3:
        val = parts[2].lower() == "true"
        persist_change = any(p.lower() in ("persist", "--persist") for p in parts[3:])
        try:
            import urllib.request as _ur
            payload = json.dumps({"collect_trajectories": val, "persist": persist_change}).encode()
            req = _ur.Request(f"http://localhost:{8779}/provider/set",
                              data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(req, timeout=3) as r:
                result = json.loads(r.read())
            persisted_label = " (persisted)" if result.get("persisted") else " (runtime only)"
            send_message(chat_id, f"\U0001f3af Trajectory collection{persisted_label}: {'\U0001f7e2 ON' if val else '\u26ab OFF'}")
        except Exception as e:
            send_message(chat_id, f"\u274c {e}")
    else:
        send_message(chat_id,
            "*Provider commands:*\n"
            "`/provider` or `/providers` \u2014 show routing + swap menu\n"
            "`/provider set <calltype> <provider>` \u2014 e.g. `/provider set task_inference openai`\n"
            "`/provider set <calltype> <provider> persist` \u2014 save to config.yaml\n"
            "`/provider set all openai` \u2014 flip everything\n"
            "`/provider status` \u2014 check which keys are set\n"
            "`/provider reset` \u2014 revert to config.yaml defaults\n"
            "`/provider collect true|false [persist]` \u2014 toggle trajectory collection"
        )
    return


def _handle_evolve(chat_id: str, text: str) -> None:
    bm = _bot_module()
    send_message = bm.send_message
    send_buttons = bm.send_buttons
    edit_message = bm.edit_message
    send_file = bm.send_file
    _call_api = bm._call_api
    parts = text.split(maxsplit=1)
    sub = parts[1].strip() if len(parts) > 1 else ""

    # /evolve — show evolution menu with state overview
    if sub == "" or sub == "menu":
        state = _call_api("GET", "/evolution/state") or {}
        status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(state.get("state", ""), "❓")
        msg = (
            f"🧬 *Kernel Evolution* {status_icon}\n"
            f"State: `{state.get('state', 'unknown')}`  "
            f"Iterations: `{state.get('iterations', 0)}/{state.get('cap', '?')}`\n"
            "Use the buttons below to control the evolution sandbox."
        )
        buttons = [
            [{"text": "📊 Status", "callback_data": "/evolve status"}, {"text": "📈 Dashboard", "callback_data": "/evolve dash"}],
            [{"text": "▶ Start", "callback_data": "/evolve control start"}, {"text": "⏸ Pause", "callback_data": "/evolve control pause"}],
            [{"text": "▶▶ Resume", "callback_data": "/evolve control resume"}, {"text": "⏹ Stop", "callback_data": "/evolve control stop"}],
            [{"text": "🔄 Reset", "callback_data": "/evolve control reset"}],
            [{"text": "🎯 Send Task", "callback_data": "/evolve task "}],
        ]
        send_buttons(chat_id, msg, buttons)
        return

    # /evolve status — full evolution state + history summary
    if sub == "status":
        state = _call_api("GET", "/evolution/state") or {}
        history = _call_api("GET", "/evolution") or {}
        status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(state.get("state", ""), "❓")
        gaps = history.get("gaps", [])
        installed = history.get("installed_skills", [])
        recent = history.get("cycles", [])[-5:] if history.get("cycles") else []
        lines = [
            f"🧬 *Evolution Status* {status_icon}",
            f"State: `{state.get('state', 'unknown')}` · Iter: `{state.get('iterations', 0)}/{state.get('cap', '?')}` · Remaining: `{state.get('remaining', '?')}`",
            f"Skills installed: `{len(installed)}`",
            f"Open gaps: `{len(gaps)}`",
        ]
        if installed:
            lines.append("\n📦 *Last installed:* " + ", ".join(f"`{s}`" for s in installed[-5:]))
        if gaps:
            gap_texts = []
            for g in gaps[-3:]:
                if isinstance(g, dict):
                    gap_texts.append(g.get('gap', str(g)))
                else:
                    gap_texts.append(str(g))
            lines.append("\n❓ *Recent gaps:* " + "; ".join(gap_texts))
        if recent:
            lines.append("\n🔁 *Recent cycles:*")
            for c in recent:
                lines.append(f"  • {c.get('task', '?')} → {c.get('outcome', '?')}")
        send_message(chat_id, "\n".join(lines))
        return

    # /evolve dash — generate + send dashboard HTML as file
    if sub == "dash":
        working_id = send_message(chat_id, "📊 Generating evolution dashboard…")
        dash_url = "http://localhost:8779/evolution/dashboard"
        try:
            import urllib.request
            import tempfile
            resp = urllib.request.urlopen(dash_url, timeout=10)
            html_bytes = resp.read()
            tmp_path = os.path.join(tempfile.gettempdir(), "kernel_evolution_dashboard.html")
            with open(tmp_path, "wb") as _f:
                _f.write(html_bytes)
            if working_id:
                edit_message(chat_id, working_id, "📊 Dashboard ready — sending file…")
            send_file(chat_id, tmp_path, caption="🧬 Kernel Evolution Dashboard")
            os.unlink(tmp_path)
        except Exception as e:
            err_msg = f"❌ Dashboard error: {str(e)[:200]}"
            if working_id:
                edit_message(chat_id, working_id, err_msg)
            else:
                send_message(chat_id, err_msg)
        return

    # /evolve control <action> [cap=N]
    if sub.startswith("control"):
        ctrl_parts = sub.split()
        action = ctrl_parts[1] if len(ctrl_parts) > 1 else ""
        cap_val = None
        for p in ctrl_parts[2:]:
            if p.startswith("cap="):
                try:
                    cap_val = int(p.split("=", 1)[1])
                except ValueError:
                    pass
        valid_actions = {"start", "pause", "resume", "stop", "reset"}
        if action not in valid_actions:
            send_message(chat_id, f"Usage: `/evolve control <action> [cap=N]`\nActions: start, pause, resume, stop, reset")
            return
        body = {"action": action}
        if cap_val is not None:
            body["cap"] = cap_val
        result = _call_api("POST", "/evolution/control", body)
        if result:
            status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(result.get("state", ""), "❓")
            send_message(
                chat_id,
                f"✅ Evolution `{action}` applied\n"
                f"State: {status_icon} `{result.get('state')}` · "
                f"Iter: `{result.get('iterations', 0)}/{result.get('cap', '?')}`"
            )
        else:
            send_message(chat_id, f"❌ Evolution control failed for action `{action}`")
        return

    # /evolve task <description> — manually trigger an evolution cycle
    if sub.startswith("task ") or sub.startswith("task	"):
        task_desc = sub[5:].strip()
        if not task_desc:
            send_message(chat_id, "Usage: `/evolve task <description>`\nExample: `/evolve task analyse sentiment of customer feedback`")
            return
        working_id = send_message(chat_id, f"🧬 Triggering evolution for: _{task_desc}_…")
        result = _call_api("POST", "/evolution/trigger", {"task": task_desc})
        if result:
            outcome = result.get("outcome", "unknown")
            skill = result.get("skill_installed")
            confidence = result.get("confidence")
            lines = [f"🧬 *Evolution cycle complete*", f"Task: `{task_desc}`", f"Outcome: `{outcome}`"]
            if skill:
                lines.append(f"📦 Skill installed: `{skill}`")
            if confidence is not None:
                lines.append(f"🎯 Confidence: `{confidence}`")
            if working_id:
                edit_message(chat_id, working_id, "🧬 Evolution cycle done")
            send_message(chat_id, "\n".join(lines))
        else:
            err = "❌ Evolution trigger failed — is the sandbox running and EVOLUTION_ENABLED=true?"
            if working_id:
                edit_message(chat_id, working_id, err)
            else:
                send_message(chat_id, err)
        return

    # /evolve backup [description] — create a timestamped backup
    if sub.startswith("backup"):
        desc = sub[6:].strip() if len(sub) > 6 else ""
        working_id = send_message(chat_id, "💾 Creating backup…")
        body = {"description": desc, "full": True}
        result = _call_api("POST", "/evolve/backup", body)
        if result and result.get("status") == "created":
            backup = result.get("backup", {})
            size = backup.get("size_mb", 0)
            sha = backup.get("sha256", "")[:8]
            msg = (
                f"✅ *Backup created*\n"
                f"Path: `{backup.get('backup_path', '')}`\n"
                f"Size: {size} MB • SHA256: `{sha}`\n"
                f"Timestamp: {backup.get('timestamp', '')}"
            )
            if working_id:
                edit_message(chat_id, working_id, msg)
            else:
                send_message(chat_id, msg)
        else:
            err = result.get("error", "Backup failed")
            if working_id:
                edit_message(chat_id, working_id, f"❌ Backup failed: {err}")
            else:
                send_message(chat_id, f"❌ Backup failed: {err}")
        return

    # /evolve init [--force] — initialize workspace and databases
    if sub.startswith("init"):
        force = "--force" in sub
        desc = sub[4:].strip() if len(sub) > 4 else ""
        desc = desc.replace("--force", "").strip()
        working_id = send_message(chat_id, "⚙️ Initializing kernel-evolving…")
        body = {"force": force, "description": desc}
        result = _call_api("POST", "/evolve/init", body)
        if result and result.get("status") == "initialized":
            msg = f"✅ Initialization complete\n{json.dumps(result, indent=2)}"
        else:
            msg = f"⚠️ Init not yet implemented\n{json.dumps(result, indent=2)}"
        if working_id:
            edit_message(chat_id, working_id, msg)
        else:
            send_message(chat_id, msg)
        return

    # /evolve fresh [description] — reset to fresh state (requires confirmation)
    if sub.startswith("fresh"):
        desc = sub[5:].strip() if len(sub) > 5 else ""
        # Step 1: create backup and show confirmation button
        working_id = send_message(chat_id, "💾 Creating backup before fresh reset…")
        body = {"description": "Pre-fresh backup: " + desc, "full": True}
        result = _call_api("POST", "/evolve/backup", body)
        if not result or result.get("status") != "created":
            err = result.get("error", "Backup failed")
            if working_id:
                edit_message(chat_id, working_id, f"❌ Backup failed: {err}")
            else:
                send_message(chat_id, f"❌ Backup failed: {err}")
            return
        backup = result.get("backup", {})
        backup_path = backup.get("backup_path", "")
        # Dry run to list actual targets
        dry_result = _call_api("POST", "/evolve/fresh", {"dry_run": True, "description": "Dry run before confirmation"})
        dry_summary = ""
        if dry_result and dry_result.get("status") == "dry_run":
            summary = dry_result.get("summary", {})
            db_count = summary.get("databases", 0)
            file_count = summary.get("files", 0)
            dir_count = summary.get("directories_to_clear", 0)
            dry_summary = (
                f"\n**Dry‑run summary:**\n"
                f"• Databases: {db_count} (`memory/chat_history_evolving.db`, `data/evolution.db`, `data/promoted_signals.db`)\n"
                f"• Files: {file_count} (`user.json`, `todos.md`, etc.)\n"
                f"• Directories to clear: {dir_count} (`thoughts/`, `notes/`, `memory/`, `data/`, `artifacts/`, `logs/`, `tmp/`)\n"
            )
        else:
            dry_summary = "\n⚠️ Dry run failed — proceeding with generic list.\n"
        # Show confirmation with inline button
        msg = (
            f"🔄 *Fresh Reset Confirmation*\n"
            f"A backup has been created at:\n`{backup_path}`\n"
            f"\n**This will:**\n"
            "• Delete workspace databases from `memory/` and `data/`\n"
            "• Keep ecosystem skills/routines (no change)\n"
            "• Reinitialize workspace with empty databases\n"
            f"{dry_summary}"
            "• Restart kernel-evolving API (recommended)\n"
            f"\nDescription: {desc}"
        )
        buttons = [[
            {"text": "✅ Confirm Fresh Reset", "callback_data": f"/evolve fresh confirm {backup_path}"},
            {"text": "❌ Cancel", "callback_data": "/evolve fresh cancel"}
        ]]
        if working_id:
            edit_message(chat_id, working_id, msg)
            send_buttons(chat_id, "Please confirm:", buttons)
        else:
            send_buttons(chat_id, msg, buttons)
        return

    # /evolve fresh confirm <backup_path> — execute fresh reset after confirmation
    if sub.startswith("fresh confirm"):
        parts = sub.split(maxsplit=2)
        if len(parts) < 3:
            send_message(chat_id, "❌ Missing backup path.")
            return
        backup_path = parts[2]
        working_id = send_message(chat_id, f"🔄 Performing fresh reset using backup {backup_path}…")
        # Call fresh API endpoint (dry_run=False)
        result = _call_api("POST", "/evolve/fresh", {"dry_run": False, "description": "Fresh reset after confirmation"})
        if result and result.get("status") in ["fresh_executed", "dry_run"]:
            deleted = result.get("summary", {}).get("deleted_count", 0)
            cleared = result.get("summary", {}).get("cleared_count", 0)
            msg = (
                f"✅ *Fresh reset completed*\n"
                f"Backup: `{backup_path}`\n"
                f"Deleted {deleted} files, cleared {cleared} directories.\n"
                f"Databases reinitialized.\n"
            )
            errors = result.get("errors", [])
            if errors:
                msg += f"\n⚠️ Errors: {len(errors)}"
            # Suggest restart of API
            msg += "\n⚠️ *Restart kernel-evolving API* with `bash start.sh` for clean state."
            if working_id:
                edit_message(chat_id, working_id, msg)
            else:
                send_message(chat_id, msg)
        else:
            err = result.get("error", "Fresh failed")
            if working_id:
                edit_message(chat_id, working_id, f"❌ Fresh failed: {err}")
            else:
                send_message(chat_id, f"❌ Fresh failed: {err}")
        return

    # /evolve fresh cancel — cancel fresh reset
    if sub.startswith("fresh cancel"):
        send_message(chat_id, "❌ Fresh reset cancelled.")
        return

    # Unknown /evolve subcommand
    send_message(
        chat_id,
        "🧬 *Evolution commands:*\n"
        "`/evolve` — menu + current state\n"
        "`/evolve status` — full status + history\n"
        "`/evolve dash` — send live dashboard HTML\n"
        "`/evolve control start|pause|resume|stop|reset [cap=N]` — state machine control\n"
        "`/evolve task <description>` — manually trigger an evolution cycle\n"
        "`/evolve backup [description]` — create timestamped backup\n"
        "`/evolve init [--force]` — initialize workspace and databases\n"
        "`/evolve fresh [description]` — reset to fresh state (with confirmation)"
    )
    return

