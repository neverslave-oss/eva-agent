"""
evolver.py — ADR-004 Tier 1: Gemma 4 ecosystem search and skill acquisition.
Searches local SKILL.md files for matching skills and acquires them via git clone.
NOT imported by agent.py yet — see evolution_hook.py for the wiring point.
"""
import os
import re
import json
import sqlite3
import subprocess
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from core.memory.embedding_client import EmbeddingClient
from . import capability_verifier
from runtime_paths import EVOLUTION_DB
from database.evolution import RecommendationStore


_RECS_DB_PATH = str(EVOLUTION_DB)


# RecommendationStore is now provided by database.evolution
# Keeping class name here for backward compat — it's the same class imported above
# (RecommendationStore from database.evolution is already imported at top of file)


# Repos allowed for auto-clone (allowlist only — no arbitrary URLs ever)
_ALLOWLIST_PREFIX = os.environ.get("KERNEL_GH_ALLOWLIST_PREFIX", "https://github.com/")
_ALLOWLIST_ORG = os.environ.get("KERNEL_GH_ALLOWLIST_ORG", "")


@dataclass
class EvolutionResult:
    found: bool
    installed: list = field(default_factory=list)
    confidence: str = "LOW"      # HIGH / MEDIUM / LOW
    retry: bool = False
    escalated: bool = False
    provider_used: str | None = None
    gap: str = ""                # description if nothing resolved
    verification_result: str | None = None      # ADR-006
    verification_reasoning: str | None = None    # ADR-006
    recommendations: list = field(default_factory=list)  # ADR-007
    pending_approval: bool = False          # ADR-021: waiting for user gate decision
    synthesis_id: str | None = None         # ADR-021: key in pending_synthesis.json


def _parse_skill_md_text(text: str) -> dict | None:
    """Parse a SKILL.md string and return metadata dict or None on failure."""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return None
    return {
        "name": fm.get("name", ""),
        "description": fm.get("description", ""),
        "commands": fm.get("commands", fm.get("metadata", {}).get("commands", [])),
        "repo": fm.get("repo", ""),
        "exec": fm.get("exec", None),
    }


def _parse_skill_md(path: Path) -> dict | None:
    """Parse a SKILL.md and return metadata dict or None on failure."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return None
    return {
        "name": fm.get("name", path.parent.name),
        "description": fm.get("description", ""),
        "commands": fm.get("commands", fm.get("metadata", {}).get("commands", [])),
        "repo": fm.get("repo", ""),
        "exec": fm.get("exec", None),
        "path": str(path),
    }


def _score_match(skill: dict, task: str) -> int:
    """Return a relevance score for a skill vs task string. Higher = more relevant."""
    task_lower = task.lower()
    score = 0
    name = skill.get("name", "").lower()
    desc = skill.get("description", "").lower()
    commands = [str(c).lower() for c in skill.get("commands", [])]

    if name and name in task_lower:
        score += 10
    if name and any(word in task_lower for word in name.split("-")):
        score += 4
    for word in desc.split():
        if len(word) > 3 and word in task_lower:
            score += 2
    for cmd in commands:
        cmd_clean = cmd.lstrip("/").strip()
        if cmd_clean and cmd_clean in task_lower:
            score += 5
    return score


def _score_match_semantic(skill: dict, task: str, client: EmbeddingClient) -> float:
    """Semantic similarity score using embeddings. Returns 0.0 if embedding unavailable."""
    name = skill.get("name", "")
    desc = skill.get("description", "")
    commands = " ".join(str(c) for c in skill.get("commands", []))
    skill_text = f"{name} {desc} {commands}".strip()
    if not skill_text:
        return 0.0
    sim = client.similarity(task, skill_text)
    return sim if sim is not None else 0.0


class Evolver:
    def __init__(self, config: dict, skills_dir: str, infer_fn=None):
        self.config = config
        self.skills_dir = os.path.expanduser(skills_dir)
        self._infer_fn = infer_fn
        # Private skills install path
        self._private_skills = os.path.expanduser(
            config.get("private_skills_dir",
                        os.path.join(skills_dir, "private", "skills"))
        )
        embedding_url = config.get("embedding_server_url", "http://localhost:8770/embeddings")
        self._embedding_client = EmbeddingClient(
            backend=config.get("embedding_backend", "native"),
            embedding_url=embedding_url,
            model_path=config.get("embedding_model_path", ""),
        )
        self._min_candidate_score = float(
            config.get("evolution", {}).get("min_candidate_score", 0.35)
        )
        # FIX #4: recommendation boost — configurable additive score for prior near-misses
        self._recommendation_boost = float(
            config.get("evolution", {}).get("recommendation_boost", 0.05)
        )
        self._rec_store = RecommendationStore(_RECS_DB_PATH)

    def _fetch_remote_skills(self) -> list[dict]:
        """
        Fetch SKILL.md files from the configured GitHub ecosystem org
        via GitHub API (public repo, no auth needed).
        Configurable via KERNEL_GH_ALLOWLIST_ORG env var or config evolution.ecosystem_org.
        """
        import urllib.request, urllib.error, base64, json

        _fetcher_org = self.config.get("evolution", {}).get(
            "ecosystem_org", os.environ.get("KERNEL_GH_ALLOWLIST_ORG", "fabiopacifici-bot")
        )
        base_api = f"https://api.github.com/repos/{_fetcher_org}/microclaw_community_ecosystem/contents/skills"
        results = []
        try:
            req = urllib.request.Request(base_api, headers={"User-Agent": "kernel-evolver/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            dirs = json.loads(resp.read())
            for entry in dirs:
                if entry.get("type") != "dir":
                    continue
                skill_name = entry["name"]
                skill_api = f"{base_api}/{skill_name}/SKILL.md"
                try:
                    req2 = urllib.request.Request(skill_api, headers={"User-Agent": "kernel-evolver/1.0"})
                    resp2 = urllib.request.urlopen(req2, timeout=10)
                    data = json.loads(resp2.read())
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    skill = _parse_skill_md_text(content)
                    if skill:
                        skill["repo"] = f"https://github.com/{_fetcher_org}/microclaw_community_ecosystem"
                        skill["ecosystem_path"] = f"skills/{skill_name}"
                        results.append(skill)
                except Exception:
                    continue
        except Exception:
            pass  # Network unavailable — fall back to local only
        return results

    def _combined_score(self, skill: dict, task: str) -> float:
        """Semantic score with keyword fallback, normalised to 0-1 range."""
        semantic = _score_match_semantic(skill, task, self._embedding_client)
        if semantic > 0.0:
            return semantic
        # fallback: normalise keyword score to 0-1 range (keyword max is ~20)
        kw = _score_match(skill, task)
        # TODO(nemotron-swap): wire trajectory scores into evolver after model swap
        # When Nemotron-Labs-Diffusion-3B is integrated, feed task_trajectories
        # similarity/success scores as an additive component here.
        return min(kw / 20.0, 1.0)

    def search_ecosystem(self, task: str) -> list[dict]:
        """
        Fetch remote community ecosystem skills, then walk local skills_dir.
        Match name/description/commands against task.
        Return list of {name, repo, path, score} sorted by relevance.
        """
        results = []
        # FIX #4: load prior recommendation hits to apply score boost
        prior_hits = self._rec_store.get_hits()

        def _boost(name: str, base_score: float) -> float:
            """Apply recommendation_boost once per prior hit (capped at 1.0)."""
            if name in prior_hits:
                return min(base_score + self._recommendation_boost, 1.0)
            return base_score

        # 1. Fetch and score remote skills
        for skill in self._fetch_remote_skills():
            score = _boost(skill["name"], self._combined_score(skill, task))
            if score >= self._min_candidate_score:
                results.append({
                    "name": skill["name"],
                    "description": skill.get("description", ""),
                    "repo": skill.get("repo", ""),
                    "ecosystem_path": skill.get("ecosystem_path", ""),
                    "path": "",   # remote — no local path yet
                    "score": score,
                })

        # 2. Walk local skills and score
        base = Path(self.skills_dir)
        if base.exists():
            for root, dirs, files in os.walk(str(base), followlinks=True):
                if "SKILL.md" in files:
                    skill_path = Path(root) / "SKILL.md"
                    skill = _parse_skill_md(skill_path)
                    if not skill:
                        continue
                    score = _boost(skill["name"], self._combined_score(skill, task))
                    if score >= self._min_candidate_score:
                        results.append({
                            "name": skill["name"],
                            "description": skill.get("description", ""),
                            "repo": skill.get("repo", ""),
                            "path": str(skill_path),
                            "score": score,
                        })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _is_allowlisted(self, repo_url: str) -> bool:
        """Check if the repo URL is on the allowlist (configurable via KERNEL_GH_ALLOWLIST_ORG)."""
        if not repo_url:
            return False
        # Resolve the org at call time (may have been set after module load)
        org = self.config.get("evolution", {}).get(
            "ecosystem_org", os.environ.get("KERNEL_GH_ALLOWLIST_ORG", "fabiopacifici-bot")
        ) or _ALLOWLIST_ORG
        if not org:
            return False
        # Accept https://github.com/<org>/* or git@github.com:<org>/*
        https_ok = repo_url.startswith(f"https://github.com/{org}/")
        ssh_ok = repo_url.startswith(f"git@github.com:{org}/")
        return https_ok or ssh_ok

    def acquire(self, skill_match: dict) -> bool:
        """
        git clone skill into private/skills/. Idempotent (check exists first).
        Only clones from allowlisted repos. Returns True on success.
        """
        repo_url = skill_match.get("repo", "")
        if not self._is_allowlisted(repo_url):
            return False

        skill_name = skill_match.get("name", "")
        if not skill_name:
            return False

        dest = os.path.join(self._private_skills, skill_name)

        # Idempotent: already installed
        if os.path.isdir(dest):
            return True

        os.makedirs(self._private_skills, exist_ok=True)

        # Handle community ecosystem skills: clone whole repo then move subdirectory
        ecosystem_path = skill_match.get("ecosystem_path", "")
        if ecosystem_path:
            return self._acquire_ecosystem_subdir(repo_url, ecosystem_path, skill_name, dest)

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, dest],
                capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _acquire_ecosystem_subdir(self, repo_url: str, ecosystem_path: str, skill_name: str, dest: str) -> bool:
        """Clone community ecosystem repo and extract the skill subdirectory."""
        import tempfile, shutil
        try:
            with tempfile.TemporaryDirectory() as tmpclone:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, tmpclone],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    return False
                src = os.path.join(tmpclone, ecosystem_path)
                if not os.path.isdir(src):
                    return False
                shutil.copytree(src, dest)
                return True
        except Exception:
            return False

    def validate(self, skill_name: str) -> str:
        """
        Check SKILL.md is parseable and exec field (if present) points to an
        existing file. Returns HIGH / MEDIUM / LOW confidence.
        """
        skill_path = Path(self._private_skills) / skill_name / "SKILL.md"
        if not skill_path.exists():
            return "LOW"

        skill = _parse_skill_md(skill_path)
        if not skill:
            return "LOW"

        exec_field = skill.get("exec")
        scripts_dir = skill_path.parent / "scripts"

        if exec_field:
            exec_path = skill_path.parent / exec_field
            if exec_path.exists():
                return "HIGH"
            return "LOW"

        if scripts_dir.exists() and any(scripts_dir.iterdir()):
            return "MEDIUM"

        return "MEDIUM"

    def run(self, task: str) -> EvolutionResult:
        """Full Tier 1 cycle: search → verify → acquire → validate → report."""
        matches = self.search_ecosystem(task)

        if not matches:
            # No ecosystem match — escalate to Tier 2
            return EvolutionResult(
                found=False,
                confidence="LOW",
                retry=False,
                escalated=True,
                gap=f"No ecosystem skill found for task: {task}",
            )

        # ADR-006: iterate top-3 candidates, verify each before acquiring
        best = None
        verification_result = None
        verification_reasoning = None

        for candidate in matches[:3]:
            if self._infer_fn is not None:
                can_handle, reasoning = capability_verifier.verify(
                    candidate, task, self._infer_fn
                )
                verification_result = "YES" if can_handle else "NO"
                verification_reasoning = reasoning
                if not can_handle:
                    import logging
                    logging.getLogger(__name__).info(
                        f"[ADR-006] '{candidate['name']}' rejected — trying next candidate"
                    )
                    continue
            best = candidate
            break

        if best is None:
            # ADR-007: surface partial matches before Tier 2
            from core.evolution.recommender import recommend as _recommend
            try:
                recs = _recommend(
                    task=task,
                    all_matches=matches,
                    rejected_names={c.get("name") for c in matches[:3]},
                    embedding_client=self._embedding_client,
                )
            except Exception:
                recs = []
            rec_names = [r.skill_name for r in recs]
            if recs:
                import logging as _logging
                rec_summaries = [f"{r.skill_name}({r.coverage},{r.similarity:.2f})" for r in recs]
                _logging.getLogger(__name__).info(
                    f"[ADR-007] recommendations for '{task[:50]}': {rec_summaries}"
                )
                # FIX #4: persist recommendation hits so future search_ecosystem boosts them
                self._rec_store.record(rec_names)
            # All candidates failed verification → escalate to Tier 2
            return EvolutionResult(
                found=False,
                confidence="LOW",
                retry=False,
                escalated=True,
                gap=f"No verified skill found for task: {task}",
                verification_result=verification_result,
                verification_reasoning=verification_reasoning,
                recommendations=rec_names,
            )

        repo_url = best.get("repo", "")

        # If no repo URL, we found the skill locally but can't acquire it remotely
        if not repo_url or not self._is_allowlisted(repo_url):
            # Skill exists locally — check if it's already usable
            local_skill_md = _parse_skill_md(Path(best["path"]))
            if local_skill_md:
                return EvolutionResult(
                    found=True,
                    installed=[best["name"]],
                    confidence="MEDIUM",
                    retry=True,
                    escalated=False,
                    verification_result=verification_result,
                    verification_reasoning=verification_reasoning,
                )
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                gap=f"Skill found but no allowlisted repo to acquire: {task}",
                verification_result=verification_result,
                verification_reasoning=verification_reasoning,
            )

        acquired = self.acquire(best)
        if not acquired:
            return EvolutionResult(
                found=False,
                confidence="LOW",
                escalated=True,
                gap=f"Failed to acquire skill '{best['name']}' for task: {task}",
                verification_result=verification_result,
                verification_reasoning=verification_reasoning,
            )

        confidence = self.validate(best["name"])
        retry = confidence in ("HIGH", "MEDIUM")

        # Create tracker todo for unresolved gap
        if not retry:
            self._create_gap_todo(task)

        return EvolutionResult(
            found=True,
            installed=[best["name"]],
            confidence=confidence,
            retry=retry,
            escalated=False,
            verification_result=verification_result,
            verification_reasoning=verification_reasoning,
        )

    def _create_gap_todo(self, task: str):
        """Create a tracker todo for an unresolved gap (best-effort)."""
        try:
            tracker = os.path.expanduser(
                "~/.openclaw/workspace/repositories/open-workspace-tracker/scripts/tracker.py"
            )
            if os.path.exists(tracker):
                subprocess.run(
                    ["python3", tracker, "todo", "add",
                     f"[ADR-004] Unresolved evolution gap: {task[:120]}",
                     "--project", "kernel", "--category", "research"],
                    capture_output=True, timeout=10,
                )
        except Exception:
            pass
