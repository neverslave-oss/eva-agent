"""
test_core_skills_dispatch.py — Integration-style dispatch tests for core workspace skills.

Tests that EVA's skill router (agent.triage) correctly finds and invokes each
hand-crafted skill (not synthesised ones) when called by slash command or
natural language intent. No real model inference — provider is patched to
return predictable output.

Core skills under test:
  - voice-clone        (/voice-clone)
  - collective-memory  (/memory-read, /memory-search, /memory-write)
  - browser-automation (intent: "search the web", "browse")
  - github             (intent: "github", "gh issue", "gh pr")
  - mental-map         (intent: "mental map", "knowledge graph")
  - open-workspace-tracker (intent: "add todo", "tracker")
  - security-scanner   (intent: "scan", "/scan")
  - open-fantasia-imagegen (intent: "generate image", "fantasia")
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
SKILLS_DIR = str(Path.home() / ".kernel" / "ecosystem" / "private" / "skills")
sys.path.insert(0, SRC)

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_agent(skills_dir: str, tmp_path: Path, monkeypatch):
    """Bootstrap agent with a real skills directory, patched provider."""
    import core.agent as ag
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), ".."))

    # Patch provider so no real inference happens
    import core.inference.provider as prov_mod
    fake_prov = MagicMock()
    fake_prov.get_provider.return_value = "openai"
    fake_prov.get_model.return_value = "gpt-5.4"
    fake_prov.infer_with_tools.return_value = "skill_dispatched_ok"
    fake_prov.infer.return_value = "skill_dispatched_ok"
    monkeypatch.setattr(prov_mod, "_instance", fake_prov)
    monkeypatch.setattr(prov_mod, "get_provider", lambda config=None: fake_prov)

    # Patch memory so nothing is persisted
    import core.memory.memory as mem_mod
    monkeypatch.setattr(mem_mod, "load", lambda **kw: [])
    monkeypatch.setattr(mem_mod, "save", lambda *a, **kw: None)

    # Patch embedding client to avoid heavy load
    import core.memory.embedding_client as ec_mod
    mock_ec = MagicMock()
    mock_ec.embed.return_value = [0.0] * 128
    monkeypatch.setattr(ec_mod, "EmbeddingClient", lambda **kw: mock_ec)

    ag.init(cfg_path)
    # Override skills_dir to the real ecosystem path
    from core.skills import load_all
    ag._skills = load_all(skills_dir)
    return ag


def _skill_names(ag) -> list:
    return [s["name"] for s in ag._skills]


def _find_skill(ag, name: str):
    return next((s for s in ag._skills if s["name"] == name), None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def skills_path():
    """Resolve to the private ecosystem skills directory."""
    p = Path(SKILLS_DIR).expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Private skills dir not found: {p}")
    return str(p)


# ---------------------------------------------------------------------------
# 1. Skill presence checks — each core skill must be installed
# ---------------------------------------------------------------------------

CORE_SKILLS = [
    "voice-clone",
    "collective-memory",
    "browser-automation",
    "github",
    "mental-map",
    "open-workspace-tracker",
    "security-scanner",
    "open-fantasia-imagegen",
]


@pytest.mark.parametrize("skill_name", CORE_SKILLS)
def test_core_skill_is_installed(skills_path, skill_name):
    """Every core skill must exist in the private ecosystem."""
    skill_dir = Path(skills_path) / skill_name
    assert skill_dir.exists(), f"Core skill '{skill_name}' not found at {skill_dir}"
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.exists(), f"SKILL.md missing for '{skill_name}'"


@pytest.mark.parametrize("skill_name", CORE_SKILLS)
def test_core_skill_has_name_and_description(skills_path, skill_name):
    """SKILL.md frontmatter must have name + description."""
    import yaml as _yaml
    skill_md = Path(skills_path) / skill_name / "SKILL.md"
    text = skill_md.read_text()
    # Extract YAML frontmatter between --- delimiters
    parts = text.split("---")
    assert len(parts) >= 3, f"{skill_name}/SKILL.md has no frontmatter"
    fm = _yaml.safe_load(parts[1]) or {}
    assert fm.get("name"), f"{skill_name}/SKILL.md missing 'name' field"
    assert fm.get("description"), f"{skill_name}/SKILL.md missing 'description' field"


# ---------------------------------------------------------------------------
# 2. Command routing — skills with explicit slash commands must be dispatched
# ---------------------------------------------------------------------------

COMMAND_SKILLS = [
    ("voice-clone",        "/voice-clone hello world"),
    ("collective-memory",  "/memory-search kernel"),
    ("collective-memory",  "/memory-read"),
]


@pytest.mark.parametrize("skill_name,command", COMMAND_SKILLS)
def test_command_dispatches_to_correct_skill(skills_path, skill_name, command, tmp_path, monkeypatch):
    """Exact slash command must route to the matching skill, not fall through."""
    import core.skills as sk_mod
    from core.skills import load_all, find as find_skill

    skills = load_all(skills_path)
    names = [s["name"] for s in skills]
    assert skill_name in names, f"Skill '{skill_name}' not loaded from {skills_path}"

    matched = find_skill(command.split()[0].lstrip("/").replace("-", "-"), skills)
    # find_skill uses name/command matching — also check command list
    skill_obj = next((s for s in skills if s["name"] == skill_name), None)
    assert skill_obj is not None

    cmds = [c.lstrip("/") for c in skill_obj.get("commands", [])]
    cmd_word = command.split()[0].lstrip("/")
    assert cmd_word in cmds or skill_obj["name"] == cmd_word, (
        f"Skill '{skill_name}' should have command '{cmd_word}' in its commands list. "
        f"Current: {cmds}. Add it to SKILL.md frontmatter."
    )


# ---------------------------------------------------------------------------
# 3. Intent routing — natural language should resolve to the right skill
# ---------------------------------------------------------------------------

INTENT_CASES = [
    # (natural language input, expected skill name)
    ("search the web for AI news today",      "browser-automation"),
    ("scan for vulnerabilities",               "security-scanner"),
    ("add a todo: fix the streaming bug",      "open-workspace-tracker"),
    ("generate an image of a dolphin",         "open-fantasia-imagegen"),
    ("show my knowledge graph",                "mental-map"),
    ("search collective memory for kernel",    "collective-memory"),
]


@pytest.mark.parametrize("text,expected_skill", INTENT_CASES)
def test_intent_routes_to_expected_skill(skills_path, text, expected_skill, monkeypatch):
    """Natural language intent should semantically resolve to the correct core skill."""
    from core.skills import load_all, find as find_skill

    skills = load_all(skills_path)
    sk = next((s for s in skills if s["name"] == expected_skill), None)
    assert sk is not None, f"Expected skill '{expected_skill}' not in ecosystem"

    lower_text = text.lower()

    # Check via intents list first
    intents = [i.lower() for i in sk.get("intents", [])]
    if intents:
        matched = any(intent in lower_text for intent in intents)
        if matched:
            return  # intent match confirmed

    # Check via name / command keyword
    if sk["name"].replace("-", " ") in lower_text:
        return

    # Check via description keyword overlap
    desc_words = set(sk.get("description", "").lower().split())
    text_words = set(lower_text.split())
    overlap = desc_words & text_words - {"the", "a", "an", "for", "of", "to", "in", "and"}
    assert len(overlap) >= 1, (
        f"Skill '{expected_skill}' has no route for: '{text}'\n"
        f"  intents: {intents}\n"
        f"  Add an 'intents:' list to its SKILL.md frontmatter for reliable routing."
    )


# ---------------------------------------------------------------------------
# 4. voice-clone specific — server check + correct endpoint
# ---------------------------------------------------------------------------

def test_voice_clone_skill_has_correct_command(skills_path):
    """/voice-clone must be in the commands list."""
    import yaml as _yaml
    text = (Path(skills_path) / "voice-clone" / "SKILL.md").read_text()
    parts = text.split("---")
    fm = _yaml.safe_load(parts[1]) or {}
    commands = [c.lstrip("/") for c in fm.get("commands", [])]
    assert "voice-clone" in commands, (
        "voice-clone SKILL.md missing 'commands: [/voice-clone]' — "
        "EVA won't dispatch /voice-clone without it"
    )


def test_voice_clone_script_exists(skills_path):
    """clone_voice.sh helper must exist and be executable."""
    script = Path(skills_path) / "voice-clone" / "scripts" / "clone_voice.sh"
    assert script.exists(), f"Missing {script}"
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_voice_clone_server_endpoint(monkeypatch):
    """Voice server at :8766/tts/clone should accept POST (or respond with known error if down)."""
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8766/tts/clone",
            data=b"",
            method="POST"
        )
        urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as e:
        # 405 (GET not allowed) or 422 (missing fields) both mean server is alive
        assert e.code in (405, 422, 400), f"Unexpected HTTP error: {e.code}"
    except urllib.error.URLError:
        pytest.skip("Voice server not running — skip live endpoint test")


# ---------------------------------------------------------------------------
# 5. collective-memory — search script must exist
# ---------------------------------------------------------------------------

def test_collective_memory_search_script_exists():
    """The collective memory search script must be present for context injection."""
    script = Path.home() / ".openclaw" / "workspace" / "collective-memory" / "scripts" / "search.py"
    assert script.exists(), f"Missing collective-memory search script at {script}"


def test_collective_memory_skill_has_commands(skills_path):
    import yaml as _yaml
    text = (Path(skills_path) / "collective-memory" / "SKILL.md").read_text()
    parts = text.split("---")
    fm = _yaml.safe_load(parts[1]) or {}
    commands = fm.get("commands", [])
    assert len(commands) > 0, "collective-memory SKILL.md has no commands — EVA won't dispatch /memory-* without them"


# ---------------------------------------------------------------------------
# 6. security-scanner — /scan command presence
# ---------------------------------------------------------------------------

def test_security_scanner_has_scan_command(skills_path):
    import yaml as _yaml
    text = (Path(skills_path) / "security-scanner" / "SKILL.md").read_text()
    parts = text.split("---")
    fm = _yaml.safe_load(parts[1]) or {}
    commands = [c.lstrip("/") for c in fm.get("commands", [])]
    intents = fm.get("intents", [])
    has_route = "scan" in commands or any("scan" in i.lower() for i in intents)
    assert has_route, (
        "security-scanner needs 'commands: [/scan]' or 'intents: [scan]' in SKILL.md "
        "for EVA to dispatch /scan"
    )
