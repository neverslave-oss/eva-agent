#!/usr/bin/env python3
"""
Sim 12 — Repeatable thesis proof

Goal:
1. Confirm a capability is missing in a fresh isolated ecosystem
2. Synthesize a new skill via Tier 2
3. Confirm the skill is installed and reusable
4. Confirm agent.triage() can handle the same flow during a conversation

This sim does NOT depend on the live installed skill set. Each run creates a
fresh temporary ecosystem under ~/kernel-evo-notes/sim12/<run-id>/ecosystem.
That makes the proof repeatable.
"""

import importlib
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skills import load_all as load_skills, find as find_skill, run as run_skill  # noqa: E402
from model import infer as model_infer  # noqa: E402

OUT_BASE = Path.home() / "kernel-evo-notes" / "sim12"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SOCKET = "/tmp/kernel_evolving_model.sock"

MODEL_CFG = {
    "label": "Gemma 4 E2B (2.3B) + drafter",
    "provider": "local",
    "model_path": "~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
    "drafter_path": "~/models/huggingface/hub/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587",
}


def _safe_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _normalize(text: str) -> str:
    return text.replace("```json", "").replace("```toml", "").replace("```", "").strip()


def _swap_model(cfg: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(300)
    s.connect(MODEL_SOCKET)
    req = json.dumps(
        {
            "method": "swap_model",
            "params": {
                "model_path": cfg["model_path"],
                "drafter_path": cfg["drafter_path"] or "",
                "dtype": "bfloat16",
            },
        }
    ) + "\n"
    s.sendall(req.encode())
    resp = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
        if b"\n" in resp:
            break
    s.close()
    return json.loads(resp.strip())


def _build_isolated_config(run_dir: Path) -> tuple[dict, Path, Path, Path]:
    base_cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    ecosystem_root = run_dir / "ecosystem"
    private_skills = ecosystem_root / "private" / "skills"
    routines_dir = Path(os.path.expanduser(base_cfg.get("routines_dir", "~/.kernel-evolving/ecosystem")))

    private_skills.mkdir(parents=True, exist_ok=True)
    run_workspace = run_dir / "workspace"
    run_workspace.mkdir(parents=True, exist_ok=True)

    base_cfg["skills_dir"] = str(ecosystem_root)
    base_cfg["private_skills_dir"] = str(private_skills)
    base_cfg["workspace"] = str(run_workspace)
    base_cfg["kernel_workspace"] = str(run_workspace)
    base_cfg["olly_workspace"] = ""
    base_cfg.setdefault("planning", {})["enabled"] = False

    cfg_path = run_dir / "config.sim12.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")
    return base_cfg, cfg_path, ecosystem_root, private_skills


def _prepare_modules(cfg_path: Path, skills_dir: Path, routines_dir: Path):
    os.environ["SKILLS_DIR"] = str(skills_dir)
    os.environ["ROUTINES_DIR"] = str(routines_dir)
    os.environ["EVOLUTION_ENABLED"] = "true"
    os.environ.setdefault("EVOLUTION_MAX_ITERATIONS", "10")

    import evolution_state
    import evolution_hook
    import agent

    importlib.reload(evolution_state)
    importlib.reload(evolution_hook)
    importlib.reload(agent)

    evolution_hook.EVOLUTION_ENABLED = True
    evolution_state.reset(cap=1)
    evolution_state.start(cap=1)
    agent.init(str(cfg_path))
    return agent, evolution_hook, evolution_state


def _run_skill_isolated(skills_dir: Path, skill_name: str, user_input: str) -> str:
    all_skills = load_skills(str(skills_dir))
    skill = find_skill(skill_name, all_skills)
    if not skill:
        available = ", ".join(sorted(s["name"] for s in all_skills))
        return f"(error: skill '{skill_name}' not found. Available: {available})"
    return run_skill(skill, user_input, model_infer)


def _phase_a(run_dir: Path, run_id: str, cfg: dict, skills_dir: Path, private_skills: Path) -> dict:
    phase_dir = run_dir / "phase_a_explicit"
    phase_dir.mkdir(parents=True, exist_ok=True)

    skill_name = f"sim12-tsv-to-toml-{run_id}"
    task = f"Create a skill named {skill_name} that converts TSV content into a TOML array of tables"
    tsv_input = "name\trole\nFabio\tfounder\nOlly\tagent\n"

    before = _run_skill_isolated(skills_dir, skill_name, tsv_input)
    _safe_write(phase_dir / "01_before.txt", before)
    missing_confirmed = "not found" in before.lower()

    import evolution_state
    evolution_state.reset(cap=1)
    evolution_state.start(cap=1)

    import evolution_hook
    primary_result = evolution_hook.maybe_evolve(task, cfg, skills_dir=str(skills_dir))
    payload = {
        "primary": {
            "task": task,
            "found": getattr(primary_result, "found", None),
            "installed": getattr(primary_result, "installed", []),
            "confidence": getattr(primary_result, "confidence", None),
            "retry": getattr(primary_result, "retry", None),
            "escalated": getattr(primary_result, "escalated", None),
            "provider_used": getattr(primary_result, "provider_used", None),
            "gap": getattr(primary_result, "gap", None),
        }
    }

    installed_names = payload["primary"].get("installed", [])
    used_fallback = False
    if not installed_names:
        fallback_task = "convert TSV content into a TOML array of tables"
        evolution_state.reset(cap=1)
        evolution_state.start(cap=1)
        fallback_result = evolution_hook.maybe_evolve(fallback_task, cfg, skills_dir=str(skills_dir))
        payload["fallback"] = {
            "task": fallback_task,
            "found": getattr(fallback_result, "found", None),
            "installed": getattr(fallback_result, "installed", []),
            "confidence": getattr(fallback_result, "confidence", None),
            "retry": getattr(fallback_result, "retry", None),
            "escalated": getattr(fallback_result, "escalated", None),
            "provider_used": getattr(fallback_result, "provider_used", None),
            "gap": getattr(fallback_result, "gap", None),
        }
        installed_names = payload["fallback"].get("installed", [])
        used_fallback = bool(installed_names)

    _safe_write(phase_dir / "02_maybe_evolve.json", json.dumps(payload, indent=2))

    installed_name = installed_names[0] if installed_names else skill_name
    skill_path = private_skills / installed_name / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    _safe_write(phase_dir / "03_skill.md", skill_text)

    after = _run_skill_isolated(skills_dir, installed_name, tsv_input)
    _safe_write(phase_dir / "04_after.txt", after)
    after_norm = _normalize(after)
    after_ok = (
        "[[" in after_norm
        and 'name = "Fabio"' in after_norm
        and 'role = "founder"' in after_norm
        and 'name = "Olly"' in after_norm
    )

    skill_format_ok = (
        skill_text.startswith("---\nname:")
        and "description:" in skill_text
        and "## Instructions" in skill_text
        and "commands:" not in skill_text.split("---", 2)[1]
    )

    return {
        "requested_skill_name": skill_name,
        "installed_skill_name": installed_name,
        "used_fallback": used_fallback,
        "missing_confirmed": missing_confirmed,
        "result": payload,
        "skill_exists": skill_path.exists(),
        "skill_format_ok": skill_format_ok,
        "after_ok": after_ok,
        "after_head": after_norm[:400],
    }


def _phase_b(run_dir: Path, cfg_path: Path, cfg: dict, skills_dir: Path, private_skills: Path, routines_dir: Path, run_id: str) -> dict:
    phase_dir = run_dir / "phase_b_conversation"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Fresh isolated environment for the conversation proof
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)
    private_skills.mkdir(parents=True, exist_ok=True)

    agent, evolution_hook, evolution_state = _prepare_modules(cfg_path, skills_dir, routines_dir)
    evolution_state.reset(cap=1)
    evolution_state.start(cap=1)

    before_skill_count = len(agent._skills)
    convo_text = (
        "Convert this TSV content into a TOML array of tables. "
        f"Conversation proof marker: sim12-convo-{run_id}.\n\n"
        "name\trole\nFabio\tfounder\nOlly\tagent\n"
    )

    reply = agent.triage(convo_text)
    _safe_write(phase_dir / "01_reply.txt", reply)

    # agent.triage reloads _skills itself after evolution retry on success
    after_skill_count = len(agent._skills)
    new_skills = [s["name"] for s in agent._skills]
    _safe_write(phase_dir / "02_loaded_skills.json", json.dumps(new_skills, indent=2))
    skill_paths = [str(private_skills / name / "SKILL.md") for name in new_skills if (private_skills / name / "SKILL.md").exists()]
    _safe_write(phase_dir / "03_skill_paths.json", json.dumps(skill_paths, indent=2))

    reply_norm = _normalize(reply)
    reply_ok = (
        "[[" in reply_norm
        and 'name = "Fabio"' in reply_norm
        and 'role = "founder"' in reply_norm
        and 'name = "Olly"' in reply_norm
        and "encountered errors" not in reply_norm.lower()
        and "if you need me to perform this conversion now" not in reply_norm.lower()
    )
    evolved_in_conversation = after_skill_count > before_skill_count

    return {
        "before_skill_count": before_skill_count,
        "after_skill_count": after_skill_count,
        "new_skills": new_skills,
        "skill_paths": skill_paths,
        "skill_exists": bool(skill_paths),
        "evolved_in_conversation": evolved_in_conversation,
        "reply_ok": reply_ok,
        "reply_head": reply_norm[:400],
    }


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_BASE / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_path, skills_dir, private_skills = _build_isolated_config(run_dir)
    routines_dir = Path(os.path.expanduser(cfg.get("routines_dir", "~/.kernel-evolving/ecosystem")))

    swap = _swap_model(MODEL_CFG)
    if "error" in swap:
        raise RuntimeError(f"swap_model failed: {swap['error']}")

    agent, evolution_hook, evolution_state = _prepare_modules(cfg_path, skills_dir, routines_dir)

    print(f"[sim12] run_id={run_id}")
    print(f"[model] {swap.get('model', '?')}")
    print(f"[skills_dir] {skills_dir}")
    print(f"[private_skills_dir] {private_skills}")

    phase_a = _phase_a(run_dir, run_id, cfg, skills_dir, private_skills)
    print(f"[phase_a] missing={phase_a['missing_confirmed']} installed={phase_a['result'].get('installed')} after_ok={phase_a['after_ok']}")

    phase_b = _phase_b(run_dir, cfg_path, cfg, skills_dir, private_skills, routines_dir, run_id)
    print(f"[phase_b] evolved_in_conversation={phase_b['evolved_in_conversation']} reply_ok={phase_b['reply_ok']}")

    summary = {
        "sim": "sim12",
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_CFG["label"],
        "phase_a": phase_a,
        "phase_b": phase_b,
        "thesis_proved": (
            phase_b["evolved_in_conversation"]
            and phase_b["skill_exists"]
            and phase_b["reply_ok"]
        ),
    }

    out_file = RESULTS_DIR / f"sim12_{run_id}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[results] {out_file}")
    print(f"[thesis] {'PASS' if summary['thesis_proved'] else 'FAIL'}")


if __name__ == "__main__":
    main()
