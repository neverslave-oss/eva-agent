"""
agent.py — Core Kernel Evo agent loop.
Triage: handle locally or escalate.
"""
import requests
import yaml
import os
import json
from core.skills import load_all as load_skills, find as find_skill, run as run_skill
from core.memory.embedding_client import EmbeddingClient
from core.routines import load_all as load_routines, find as find_routine, run as run_routine
from core.replica.replica import spawn, active as active_replicas, can_spawn
from core.tools import WORKSPACE as _DEFAULT_WORKSPACE
from runtime_paths import NOTES_DIR, ARTIFACTS_DIR, load_config as _load_config
from core.memory.context import build_system_prompt
from database.agent.prompt_log import PromptLogRepository


def _log_prompt(prompt, chat_id="", history=None, user_message="", provider="", model=""):
    _get_log_repo().log_prompt(prompt, chat_id=chat_id, history=history, user_message=user_message, provider=provider, model=model)

def _get_log_repo():
    from runtime_paths import SYSTEM_DEBUG_LOG_DB
    return PromptLogRepository(SYSTEM_DEBUG_LOG_DB)

_config = None
_skills = []
_routines = []
_embedding_client: EmbeddingClient | None = None
_model_infer_func = None
_model_infer_with_tools_func = None
_model_vram_free_func = None


def _ensure_model_funcs():
    """Lazily import core.inference.model functions to avoid startup-time circular import failures."""
    global _model_infer_func, _model_infer_with_tools_func, _model_vram_free_func
    if _model_infer_func is None or _model_infer_with_tools_func is None or _model_vram_free_func is None:
        from core.inference.model import infer as _infer_impl, infer_with_tools as _infer_tools_impl, vram_free_mb as _vram_impl
        _model_infer_func = _infer_impl
        _model_infer_with_tools_func = _infer_tools_impl
        _model_vram_free_func = _vram_impl


def infer(messages, max_new_tokens=8192):
    _ensure_model_funcs()
    try:
        return _model_infer_func(messages, max_new_tokens=max_new_tokens)
    except Exception as e:
        return f"[model error] {e}"


def infer_with_tools(messages, tools, workspace=_DEFAULT_WORKSPACE, max_steps=60, step_callback=None, chat_id=""):
    _ensure_model_funcs()
    return _model_infer_with_tools_func(
        messages,
        tools,
        workspace=workspace,
        max_steps=max_steps,
        step_callback=step_callback,
        chat_id=chat_id,
    )


def vram_free_mb():
    _ensure_model_funcs()
    return _model_vram_free_func()


def _provider_infer_fn():
    """Return an infer callable routed through the configured provider.

    Routines/skills use this for their LLM summary so cloud-mode setups route
    through the cloud provider instead of the local model (which crashes in
    cloud mode — no model server → None processor). Falls back to the local
    infer wrapper if the provider is unavailable.
    """
    try:
        from core.inference.provider import get_provider as _gp
        _prov = _gp()
        def _fn(_msgs, _max_new_tokens=8192):
            return _prov.infer(_msgs, max_new_tokens=_max_new_tokens, call_type="task_inference")
        return _fn
    except Exception:
        return infer


def init(config_path="config.yaml"):
    global _config, _skills, _routines, _embedding_client
    # Use the env-expanding loader so ${VAR} references in config.yaml (e.g.
    # collective_memory.url: ${KERNEL_EVO_COLLECTIVE_MEMORY_URL}) resolve from
    # the environment instead of leaking the literal placeholder.
    _config = _load_config(config_path)
    # Initialise provider singleton with config (ADR-013)
    from core.inference.provider import get_provider as _gp
    _gp(_config)
    # Env var overrides for Docker / bare-metal
    skills_dir = os.path.expanduser(os.environ.get("SKILLS_DIR") or _config.get("skills_dir", "./skills"))
    routines_dir = os.path.expanduser(os.environ.get("ROUTINES_DIR") or _config.get("routines_dir", "./routines"))
    _core_skills = _config.get("core_skills", [])
    _skills   = load_skills(skills_dir, core_skills=_core_skills)
    _routines = load_routines(routines_dir)
    embedding_url = _config.get("embedding_server_url", "http://localhost:8770/embeddings")
    _embedding_client = EmbeddingClient(
        backend=_config.get("embedding_backend", "native"),
        embedding_url=embedding_url,
        model_path=_config.get("embedding_model_path", ""),
    )
    print(f"[agent] {len(_skills)} skills, {len(_routines)} routines loaded")
    # Reconcile open evolution gaps against installed skills — marks gaps resolved
    # when a matching skill is already in the ecosystem. Prevents think-at-rest from
    # surfacing the same gap repeatedly after it has been addressed.
    try:
        from core.evolution.evolution_log import EvolutionLog
        resolved = EvolutionLog().reconcile_gaps_with_skills(_skills)
        if resolved:
            print(f"[agent] reconciled {resolved} stale gaps against installed skills", flush=True)
    except Exception as _rec_err:
        print(f"[agent] gap reconcile failed (non-fatal): {_rec_err}", flush=True)


def triage(text: str, step_callback=None, _evo_retry: bool = False, chat_id: str = "", chunk_callback=None) -> str:
    """
    Classify and handle a user request.
    Returns the response string.
    chat_id: optional session key for namespaced conversation history.
    """
    global _skills  # needed for evolution reload

    # ── Set chat_id for exec_shell authorization gate ─────────────────
    import core.tools as _tools_mod
    _tools_mod._current_chat_id = chat_id or ""

    lower = text.lower().strip()

    # --- Slash commands (highest priority) ---
    if lower.startswith("/skills"):
        lines = [f"• {s['name']} — {s.get('description', '')}" for s in _skills]
        return "\n".join(lines) if lines else "No skills loaded."

    if lower.startswith("/routines"):
        lines = [f"• {r['name']} — {r.get('description', '')}" for r in _routines]
        return "\n".join(lines) if lines else "No routines loaded."

    if lower.startswith("/run "):
        name = text[5:].strip().lower()
        r = find_routine(name, _routines)
        if r:
            print(f"[agent] /run routine: {r['name']} (matched '{name}')")
            return run_routine(r, _provider_infer_fn(), workspace=_DEFAULT_WORKSPACE, step_callback=step_callback)
        s = find_skill(name, _skills)
        if s:
            print(f"[agent] /run skill: {s['name']} (matched '{name}')")
            return run_skill(s, text, _provider_infer_fn())
        return f"No skill or routine named '{name}' found. Try /routines or /skills."

    if lower.startswith("/skill "):
        # Explicit skill invocation with fuzzy name matching
        query = text[7:].strip().lower()
        matched = None
        for s in _skills:
            sname = s["name"].lower()
            sdesc = s.get("description", "").lower()
            scmds = [c.lower().lstrip("/") for c in s.get("commands", [])]
            if query == sname or query in sdesc or query in scmds or sname.startswith(query):
                matched = s
                break
        if matched:
            print(f"[agent] /skill explicit: {matched['name']}")
            return run_skill(matched, text, _provider_infer_fn())
        return f"No skill matching '{query}' found. Try /skills to list all."

    if lower.startswith("/status"):
        replicas = active_replicas()
        return (
            f"Kernel status:\n"
            f"  VRAM free: {vram_free_mb()} MB\n"
            f"  Active replicas: {len(replicas)}/{3}\n"
            f"  Can spawn: {can_spawn()}\n"
            f"  Skills: {len(_skills)}\n"
            f"  Routines: {len(_routines)}"
        )

    # --- Routine trigger ---
    import re as _re

    # ADR-019: check incoming message against active observer probes
    try:
        from services.thought_engine import EvolvingThinkAtRest as _EvolvingThinkAtRest
        _te = getattr(_EvolvingThinkAtRest, '_instance', None)
        if _te is None:
            # Try api module where EvolvingThinkAtRest is instantiated
            import sys as _sys
            _api_mod = _sys.modules.get('api')
            if _api_mod:
                _te = getattr(_api_mod, '_think', None)
        if _te and hasattr(_te, '_observer') and _te._observer is not None:
            _te._observer.check_interaction_for_probes(text)
    except Exception:
        pass

    for r in _routines:
        rname = r["name"].lower()
        # Explicit request signals — always match: "/<name>", "run <name>",
        # "routine <name>", "run the <name> routine".
        explicit = (
            lower.startswith("/" + rname)
            or f"run {rname}" in lower
            or f"routine {rname}" in lower
            or f"run the {rname} routine" in lower
            or f"run the {rname}" in lower
        )
        # Bare word-boundary match is ONLY treated as a request when the message
        # is short and command-like (e.g. "deploy"), NOT when the word merely
        # appears mid-sentence ("i'll deploy it there") — that caused false
        # routine triggers on conversational messages.
        _word_count = len(lower.split())
        bare = bool(_re.search(r'\b' + _re.escape(rname) + r'\b', lower)) and _word_count <= 4
        matched = explicit or bare
        if not matched:
            # Check trigger.also aliases (e.g. "/morning or /briefing")
            also_raw = r.get("trigger", {}).get("also", "") or ""
            aliases = [a.strip().lstrip("/").lower() for a in _re.split(r"\bor\b|,", also_raw) if a.strip()]
            for alias in aliases:
                if alias and (lower.startswith("/" + alias) or f"run {alias}" in lower or f"routine {alias}" in lower):
                    matched = True
                    break
        if matched:
            print(f"[agent] Running routine: {r['name']}")
            # Route routine LLM summary through the configured provider (cloud in
            # cloud mode) instead of the local model.
            return run_routine(r, _provider_infer_fn(), workspace=_DEFAULT_WORKSPACE, step_callback=step_callback)

    # --- Status / system queries ---
    _status_phrases = ["kernel status", "vram free", "vram is free", "how many replicas", "system status", "/status"]
    if lower.strip() in ("status", "vram", "health") or any(p in lower for p in _status_phrases):
        replicas = active_replicas()
        return (
            f"Kernel status:\n"
            f"  VRAM free: {vram_free_mb()} MB\n"
            f"  Active replicas: {len(replicas)}/{3}\n"
            f"  Can spawn: {can_spawn()}\n"
            f"  Skills: {len(_skills)}\n"
            f"  Routines: {len(_routines)}"
        )

    # --- EVERYTHING ELSE → tool-first pipeline (ADR-022) ---
    # Skill matching, semantic search, micro-planner, evolution hook, and goal
    # discovery have been removed. All free-text requests reach infer_with_tools
    # directly so tool calls are never suppressed by routing interception.
    from core.tools import TOOLS
    print(f"[agent] Routing to infer_with_tools: {text[:60]!r}")

    # ── Determine effective context window for active model ─────────────────
    # Cloud providers have large, model-specific context windows. We do NOT cap
    # cloud context — the cloud model handles its own window. Only local
    # inference (Nemotron) needs a cap to fit in VRAM / its own window.
    def _get_context_tokens() -> tuple[int, bool]:
        try:
            from core.inference.provider import get_provider as _gp
            _task_provider = _gp().get_provider("task_inference")
        except Exception:
            _task_provider = "local"
        if _task_provider != "local":
            # Cloud: unbounded history (no trimming). Return a sentinel ctx for
            # logging only; the trimming loop is skipped via the flag below.
            return 0, True

        # Local inference: use the loaded model's context length from the map.
        try:
            from core.inference.model_client import health as _mh
            h = _mh()
            loaded_model = h.get("model", "").lower()
        except Exception:
            loaded_model = (_config or {}).get("model", {}).get("name", "").lower()
        ctx_map = (_config or {}).get("model_context_lengths", {})
        for key, tokens in ctx_map.items():
            if key.lower() in loaded_model:
                return int(tokens), False
        return int((_config or {}).get("model", {}).get("max_context_length", 32768)), False

    _ctx_tokens, _unbounded_context = _get_context_tokens()

    # ── Build system prompt first (PD2: measure real cost before history budget)
    # Must happen before _max_history_chars so we can subtract the actual prompt
    # size instead of assuming a fixed 4096-token reserve (AGENTS.md alone is
    # ~5k tokens, exhausting the old fixed reserve on small-context models).
    system_prompt = build_system_prompt(_config, _skills, _routines, vram_free_fn=vram_free_mb, chat_id=chat_id)

    # ── FR-C: inject previous session hint ─────────────────────────────────
    # If this chat has a prior session (after a rotation), tell the agent about
    # it so it can use recall_memory() when context from the old session matters.
    if chat_id:
        try:
            from core.memory.memory import get_previous_session_hint as _get_hint
            _hint = _get_hint(chat_id)
            if _hint.get("previous_session_id"):
                _prev_id = _hint["previous_session_id"]
                _sess_num = _hint.get("session_number", "?")
                system_prompt += (
                    f"\n\n**Previous conversation (session #{_sess_num - 1}):** "
                    f"`{_prev_id}` — use `recall_memory('{_prev_id}')` to retrieve context from it."
                )
        except Exception:
            pass

    # ── PD1: collective memory search ──────────────────────────────────────
    # Runs for ALL callers of triage() (API, desktop app, mobile) — not just
    # the Telegram bot path.  TTL-cached per query key, capped at 600 chars,
    # with a 3s timeout so the service can never block the response.
    _cm_url = (_config or {}).get("collective_memory", {}).get("url", "")
    if _cm_url:
        try:
            from core.collective_memory_client import search as _cm_search
            _cm_results = _cm_search(text, _cm_url)
            if _cm_results:
                system_prompt = f"## Collective memory\n{_cm_results}\n\n" + system_prompt
                print(f"[agent] collective memory: {len(_cm_results)} chars injected", flush=True)
        except Exception as _cm_err:
            print(f"[agent] collective memory skipped: {_cm_err}", flush=True)

    # ── PD2: compute history budget from actual system prompt size ──────────
    # ~4 chars per token is a conservative English approximation.
    # Reserve 512 tokens for: current user message + tool definitions +
    # _retrieved_context that will be appended below.
    _sys_prompt_tokens = len(system_prompt) // 4
    if _unbounded_context:
        # Cloud: no history cap — send the full conversation to the cloud model.
        _history_token_budget = 0
        _max_history_chars = 0
        print(f"[agent] cloud context: UNBOUNDED (no history cap) sys_prompt={_sys_prompt_tokens}tok", flush=True)
    else:
        _history_token_budget = max(2000, _ctx_tokens - _sys_prompt_tokens - 512)
        _max_history_chars = int(_history_token_budget / 0.4)
        print(f"[agent] ctx={_ctx_tokens} sys_prompt={_sys_prompt_tokens}tok history_budget={_history_token_budget}tok max_history_chars={_max_history_chars}", flush=True)

    # ── Embedding-based memory retrieval ───────────────────────────────
    _retrieved_context = ""
    if _embedding_client is not None:
        try:
            import core.memory.memory as _mem_mod
            from core.memory.embedding_client import cosine_similarity
            _all_history = _mem_mod.load(chat_id=chat_id)
            if len(_all_history) > 20:
                # Embed current message + all candidate turns in one batched
                # call (get_batch caches per-text) instead of one HTTP/model
                # call per turn (R4 perf follow-up).
                _query_emb = _embedding_client.get(text)
                _candidates = [
                    (_m["role"], str(_m.get("content", ""))[:500])
                    for _m in _all_history
                    if _m.get("role") in ("user", "assistant") and str(_m.get("content", "")).strip()
                ]
                if _query_emb is not None and _candidates:
                    _emb_map = _embedding_client.get_batch([c for _, c in _candidates])
                    _scored = []
                    for _role, _c in _candidates:
                        _emb = _emb_map.get(_c)
                        if _emb is None:
                            continue
                        _scored.append((cosine_similarity(_query_emb, _emb), _role, _c))
                    _scored.sort(key=lambda x: x[0], reverse=True)
                    _top = [f"[{r}] {c[:200]}" for s, r, c in _scored[:5] if s > 0.55]
                    if _top:
                        _retrieved_context = "## Relevant past context (semantic retrieval)\n" + "\n".join(f"  • {t}" for t in _top)
                        print(f"[agent] semantic retrieval: {len(_top)} relevant turns (score>{0.55:.2f})", flush=True)
        except Exception as _emb_err:
            print(f"[agent] embedding retrieval skipped: {_emb_err}", flush=True)

    # Append retrieved context to the system prompt already built above (PD1/PD2).
    if _retrieved_context:
        system_prompt = system_prompt + "\n\n" + _retrieved_context

    _prompt_log_snapshot = system_prompt

    # ── Token-aware history window ───────────────────────────────────
    # Load full history then trim to fit the context budget (char-based proxy).
    # Keeps RECENT turns first — prunes from the oldest end.
    import core.memory.memory as _mem
    _history = _mem.load(chat_id=chat_id)

    # Soft cap: trim oldest turns until total chars fit in budget.
    # Skipped entirely for cloud providers (unbounded context).
    if not _unbounded_context:
        _total_chars = sum(len(str(m.get("content", ""))) for m in _history)
        while _history and _total_chars > _max_history_chars:
            removed = _history.pop(0)
            _total_chars -= len(str(removed.get("content", "")))

    _hist_summary = [(m["role"], str(m.get("content", ""))[:80]) for m in _history]
    _ctx_disp = "unbounded" if _unbounded_context else _ctx_tokens
    print(f"[agent] chat_id={chat_id!r} history={len(_history)} turns ctx_tokens={_ctx_disp} last={_hist_summary[-2:] if _hist_summary else []}", flush=True)

    messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": m["role"], "content": str(m.get("content", ""))}
            for m in _history
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")  # skip malformed entries
        ],
        {"role": "user", "content": text},
    ]
    from core.inference.provider import get_provider as _get_provider
    _prov = _get_provider()  # use existing singleton initialised in init()

    # Snapshot the actual provider/model BEFORE the call so the trajectory
    # tag reflects what actually executed the inference, not what get_provider()
    # returns later (which may have changed due to thermal override).
    _actual_provider = _prov.get_provider("task_inference")
    _actual_model = _prov.get_model(_actual_provider, "task_inference")
    # For local provider, get_model returns null — resolve from model_server health instead
    if not _actual_model and _actual_provider == "local":
        try:
            import core.inference.model_client as _mc
            _actual_model = _mc.health().get("model", "local")
        except Exception:
            _actual_model = "local"
    # Log the exact conversation turns passed to the model (messages[1:]).
    _model_history_for_log = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in _history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
    ] + [{"role": "user", "content": text}]

    # Log the complete context: final system prompt + model input conversation + user message.
    _log_prompt(
        _prompt_log_snapshot,
        chat_id=chat_id,
        history=_model_history_for_log,
        user_message=text,
        provider=_actual_provider,
        model=_actual_model,
    )

    # ── Pre-inference persistence ──────────────────────────────────
    # Write the user message to the JSON hot-window BEFORE inference,
    # so it survives model crashes.  If infer_with_tools crashes (e.g.
    # the triple-quote bug in _two_stage_helpers.py), the user message
    # is in the JSON window — the next boot loads it and the
    # conversation continues instead of losing the turn entirely.
    #
    # Only the JSON window is touched; SQLite is written by the
    # post-inference save() below for a complete turn pair.
    try:
        _pre_history = list(_history) + [
            {"role": "user", "content": text},
        ]
        mem_file = _mem._chat_memory_file(chat_id)
        mem_file.write_text(json.dumps({'messages': _pre_history[-(_mem.MAX_TURNS * 2):]}, indent=2))
        print(f"[agent] pre-save ok: chat_id={chat_id!r} user_msg={text[:60]!r}", flush=True)
    except Exception as _presave_err:
        print(f"[agent] pre-save failed (non-fatal): {_presave_err}", flush=True)

    # Collect tool call steps for trajectory recording
    _recorded_steps: list = []
    def _recording_step_cb(n, tool_name, args, result_str):
        _recorded_steps.append({"tool": tool_name, "args": args, "result": result_str})
        if step_callback:
            step_callback(n, tool_name, args, result_str)

    # Some providers stream chunks but occasionally return an empty final string.
    # Capture streamed text so memory/prompt logs persist what the user actually saw.
    _chunk_parts: list[str] = []

    def _recording_chunk_cb(chunk: str):
        if chunk:
            _chunk_parts.append(str(chunk))
            if chunk_callback:
                chunk_callback(chunk)

    result = _prov.infer_with_tools(messages, TOOLS, workspace=_DEFAULT_WORKSPACE,
                                     step_callback=_recording_step_cb, call_type="task_inference",
                                     chunk_callback=_recording_chunk_cb, chat_id=chat_id)

    if (not result) and _chunk_parts:
        result = "".join(_chunk_parts)

    # Persist this turn to memory (namespaced by chat_id).
    # Skip saving dead-end/refusal answers (empty, "(max steps reached)",
    # capability-refusal text) — these poison subsequent turns when replayed
    # as history (R8).
    from core.refusal_patterns import should_skip_persisting
    if should_skip_persisting(result):
        print(f"[agent] skipping memory save — dead-end/refusal answer: {result[:80]!r}", flush=True)
    else:
        try:
            _updated_history = list(_history) + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": result or ""},
            ]
            _mem.save(_updated_history, chat_id=chat_id)
            print(f"[agent] memory saved: chat_id={chat_id!r} total_turns={len(_updated_history)}", flush=True)
        except Exception as _save_err:
            print(f"[agent] memory save failed: {_save_err}", flush=True)

    # Trajectory collection (ADR-013) — best-effort, non-blocking
    try:
        if _config and _config.get("providers", {}).get("collect_trajectories", False):
            import threading, time as _time
            # Snapshot steps now — closure captures by reference otherwise
            _steps_snapshot = list(_recorded_steps)
            _result_snapshot = result
            def _collect_trajectory():
                try:
                    from core.evolution.trajectory_collector import get_collector as _get_collector
                    _col = _get_collector(_config)
                    # Use the provider/model captured before the call — reflects what actually ran
                    _provider_name = _actual_provider
                    _model_name = _actual_model

                    # Reject error responses immediately — no point recording OOM/error runs
                    _error_markers = ("[model_server error]", "[model_client error]", "[model_server timeout]", "CUDA error", "out of memory")
                    if any(m in (_result_snapshot or "") for m in _error_markers):
                        return

                    # Reject if no tool calls fired — plain text reply, not a trajectory
                    if not _steps_snapshot:
                        return

                    # Quick critic pass
                    _critic_score = None
                    _critic_verdict = None
                    try:
                        _crit_prompt = f"Rate this task result 0.0-1.0. Task: {text[:200]}\nTools used: {[s['tool'] for s in _steps_snapshot]}\nResult: {_result_snapshot[:400]}\nRespond with score only (e.g. 0.8):"
                        _crit_raw = _prov.infer([{"role": "user", "content": _crit_prompt}], max_new_tokens=16, call_type="critic")
                        import re as _re
                        _m = _re.search(r"\b(0\.\d+|1\.0)\b", _crit_raw)
                        if _m:
                            _critic_score = float(_m.group(1))
                            _critic_verdict = "PASS" if _critic_score >= 0.7 else "FAIL"
                    except Exception:
                        pass

                    # Detect artifacts — only files written by tool calls (not system files)
                    # Look in home dir for evo_* files and workspace notes/output dirs
                    import glob
                    _now = _time.time()
                    _watch_dirs = [
                        os.path.expanduser("~/evo_*"),
                        os.path.expanduser("~/kernel-evo-notes/**"),
                        str(NOTES_DIR / "**"),
                        str(ARTIFACTS_DIR / "**"),
                    ]
                    _artifacts = []
                    for _pat in _watch_dirs:
                        _artifacts += [
                            f for f in glob.glob(_pat, recursive=True)
                            if os.path.isfile(f) and (_now - os.path.getmtime(f)) < 90
                        ]
                    # Also include any file written by write_file tool call directly
                    for _step in _steps_snapshot:
                        if _step.get("tool") == "write_file":
                            _p = _step.get("args", {}).get("path", "")
                            if _p:
                                _p = os.path.expanduser(_p)
                                if os.path.isfile(_p):
                                    _artifacts.append(_p)
                    _artifacts = list(set(_artifacts))

                    if _col.should_record(_critic_score, _artifacts or None, tool_calls=_steps_snapshot):
                        _col.record(
                            task=text, provider=_provider_name, model_name=_model_name,
                            call_type="task_inference", tool_calls=_steps_snapshot, final_reply=_result_snapshot,
                            artifacts=_artifacts, critic_score=_critic_score, critic_verdict=_critic_verdict,
                        )
                except Exception:
                    pass
            _t = threading.Thread(target=_collect_trajectory, daemon=True)
            _t.start()
    except Exception:
        pass

    return result


def escalate(text: str) -> str:
    """Delegate to OpenClaw main session."""
    endpoint = _config.get("api", {}).get("openclaw_endpoint", "http://localhost:18789")
    try:
        r = requests.post(f"{endpoint}/api/message", json={"message": text}, timeout=10)
        return r.json().get("reply", "Escalated to Olly.")
    except Exception as e:
        return f"Escalation failed: {e}"
