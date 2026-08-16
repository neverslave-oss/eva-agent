"""
code_synthesizer.py — ADR-004 Tier 2: capable model writes a new skill.
Called by evolver when Tier 1 finds no ecosystem match (escalated=True).
"""
import logging
import os
import re
import json
import shutil
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# Provider fallback order for synthesis calls.
# Cloud providers are tried first; local is the last-resort fallback.
_PROVIDER_CHAIN = ["olly", "openai", "anthropic", "local"]

# HTTP status codes that are retriable by trying the next provider.
_FALLBACK_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}


class SynthesisProviderError(Exception):
    """Raised by _call_* when a provider fails with a known client/server error."""
    def __init__(self, provider: str, status_code: int | None, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")



_PRIVATE_SKILLS_DEFAULT = os.path.expanduser("~/.kernel-evolving/ecosystem/private/skills")


class CodeSynthesizer:
    def __init__(self, config: dict):
        self.config = config
        self.private_skills = os.path.expanduser(
            config.get("private_skills_dir", _PRIVATE_SKILLS_DEFAULT)
        )

    def resolve_provider(self) -> str | None:
        """
        Returns: 'olly' | 'openai' | 'anthropic' | 'local' | None
        Priority:
          1. OPENCLAW_URL set + /health ok → 'olly'
          2. OPENAI_API_KEY or TMP_OPEN_AI_API_KEY set → 'openai'
          3. ANTHROPIC_API_KEY set → 'anthropic'
          # 4. GitHub Copilot — future release (GITHUB_TOKEN + Copilot API)
          4. HF_HOME has capable coding model → 'local'
          5. None
        """
        openclaw_url = os.environ.get("OPENCLAW_URL", "").rstrip("/")
        if openclaw_url:
            try:
                req = urllib.request.urlopen(f"{openclaw_url}/health", timeout=3)
                if req.status == 200:
                    return "olly"
            except Exception:
                pass

        # Accept both canonical and temporary env var names for OpenAI
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY"):
            return "openai"

        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"

        # TODO (future release): GitHub Copilot provider
        # if os.environ.get("GITHUB_TOKEN"):
        #     return "copilot"

        # Check HF_HOME for capable local coding models
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        capable_models = [
            "Qwen3-Coder", "qwen3-coder",          # Qwen3 Coder series
            "DeepSeek-Coder-V3", "deepseek-coder", # DeepSeek Coder V3+
            "Codestral",                             # Mistral Codestral
        ]
        hub_path = Path(hf_home) / "hub"
        if hub_path.exists():
            for item in hub_path.iterdir():
                for model_name in capable_models:
                    if model_name.lower() in item.name.lower():
                        return "local"

        return None

    def synthesize(self, task: str, gap_description: str, provider: str) -> Path | None:
        """
        Generate a new skill in a temp dir. Returns path to temp dir or None.
        Prompt asks for SKILL.md + scripts/ directory.
        """
        prompt = (
            f"Write a Kernel skill for the following task: {gap_description}\n\n"
            "Use the skill-creator format as your reference.\n"
            "Output the files using EXACTLY this format with === FILE === separators:\n\n"
            "=== SKILL.md ===\n"
            "---\n"
            "name: <skill-name>\n"
            "description: <one-line description explaining what the skill does and when to use it>\n"
            "---\n\n"
            "## Instructions\n"
            "<skill instructions for the agent>\n\n"
            "Optional only if deterministic execution is truly needed:\n"
            "=== scripts/<script_filename> ===\n"
            "<script content — Python or bash, self-contained, no external dependencies beyond stdlib>\n\n"
            "Rules:\n"
            "- SKILL.md MUST start with --- frontmatter delimiters, not code fences\n"
            "- YAML frontmatter must contain ONLY name and description\n"
            "- Put all operating guidance in the Markdown body after the closing ---\n"
            "- Include a script file only when deterministic execution is truly needed\n"
            "- No placeholder code — write a real working implementation\n"
        )

        tmpdir = Path(tempfile.mkdtemp(prefix="kernel-synth-"))
        try:
            raw = self._call_provider(provider, prompt)
            if not raw:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None
            self._parse_and_write(raw, tmpdir)
            return tmpdir
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None

    def _call_provider(self, provider: str, prompt: str) -> str | None:
        """Route to the appropriate provider and return raw text response.
        Raises SynthesisProviderError on known client/server errors.
        """
        if provider == "olly":
            return self._call_olly(prompt)
        elif provider == "openai":
            return self._call_openai(prompt)
        elif provider == "anthropic":
            return self._call_anthropic(prompt)
        elif provider == "local":
            return self._call_local(prompt)
        return None

    def synthesize_with_fallback(
        self, task: str, gap_description: str, primary_provider: str
    ) -> tuple[Path | None, str]:
        """
        Try synthesis with primary_provider, then fall through the fallback chain
        on quota / auth / server errors.  Returns (tmpdir_path, provider_used).
        `provider_used` is the provider that actually succeeded, or empty string on
        total failure.
        """
        # Build ordered chain: primary first, then the rest in default order
        chain = [primary_provider] + [
            p for p in _PROVIDER_CHAIN if p != primary_provider
        ]
        last_error: str = ""
        for provider in chain:
            # Skip providers we have no credentials for
            if not self._provider_available(provider):
                logger.debug(f"[synthesizer] {provider} unavailable — skipping")
                continue
            tmpdir = Path(tempfile.mkdtemp(prefix="kernel-synth-"))
            try:
                raw = self._call_provider(provider, task)
                if not raw:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    logger.warning(f"[synthesizer] {provider} returned empty response")
                    continue
                self._parse_and_write(raw, tmpdir)
                logger.info(f"[synthesizer] synthesis succeeded via {provider}")
                return tmpdir, provider
            except SynthesisProviderError as e:
                shutil.rmtree(tmpdir, ignore_errors=True)
                last_error = str(e)
                logger.warning(
                    f"[synthesizer] {provider} failed ({e.status_code}) — "
                    f"trying next in chain"
                )
                continue
            except Exception as e:
                shutil.rmtree(tmpdir, ignore_errors=True)
                last_error = str(e)
                logger.warning(f"[synthesizer] {provider} unexpected error: {e}")
                continue
        logger.error(f"[synthesizer] all providers exhausted. Last error: {last_error}")
        return None, ""

    def _provider_available(self, provider: str) -> bool:
        """Quick check: does this provider have credentials or local model available?"""
        if provider == "olly":
            return bool(os.environ.get("OPENCLAW_URL", "").strip())
        elif provider == "openai":
            return bool(
                os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")
            )
        elif provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        elif provider == "local":
            # Re-use resolve_provider logic for local check
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            hub_path = Path(hf_home) / "hub"
            capable_models = [
                "Qwen3-Coder", "qwen3-coder",
                "DeepSeek-Coder-V3", "deepseek-coder",
                "Codestral",
            ]
            if hub_path.exists():
                for item in hub_path.iterdir():
                    for m in capable_models:
                        if m.lower() in item.name.lower():
                            return True
            # Always allow local as last resort — model_client may still work
            return True
        return False

    def _call_olly(self, prompt: str) -> str | None:
        """POST synthesis request to Olly (OpenClaw) session API."""
        openclaw_url = os.environ.get("OPENCLAW_URL", "").rstrip("/")
        if not openclaw_url:
            return None
        try:
            payload = json.dumps({"message": prompt}).encode()
            req = urllib.request.Request(
                f"{openclaw_url}/message",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            return data.get("response") or data.get("message") or str(data)
        except urllib.error.HTTPError as e:
            if e.code in _FALLBACK_STATUS_CODES:
                raise SynthesisProviderError("olly", e.code, e.reason or str(e.code))
            return None
        except Exception:
            return None

    def _call_openai(self, prompt: str) -> str | None:
        """Call OpenAI chat completions API."""
        # Accept both canonical and temporary env var name
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY", "")
        if not api_key:
            return None
        try:
            payload = json.dumps({
                "model": "gpt-5.4",  # Latest capable OpenAI model
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 4096,  # gpt-5.x uses max_completion_tokens
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read()[:200].decode(errors="replace")
            except Exception:
                pass
            logger.warning(f"[synthesizer] OpenAI HTTP {e.code}: {body}")
            if e.code in _FALLBACK_STATUS_CODES:
                raise SynthesisProviderError("openai", e.code, body)
            return None
        except Exception as e:
            logger.warning(f"[synthesizer] OpenAI error: {e}")
            return None

    def _call_anthropic(self, prompt: str) -> str | None:
        """Call Anthropic Messages API."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        try:
            payload = json.dumps({
                "model": "claude-sonnet-4-5",  # Latest Claude Sonnet
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read()[:200].decode(errors="replace")
            except Exception:
                pass
            logger.warning(f"[synthesizer] Anthropic HTTP {e.code}: {body}")
            if e.code in _FALLBACK_STATUS_CODES:
                raise SynthesisProviderError("anthropic", e.code, body)
            return None
        except Exception as e:
            logger.warning(f"[synthesizer] Anthropic error: {e}")
            return None

    def _call_local(self, prompt: str) -> str | None:
        """Use local model_client for inference."""
        try:
            import sys
            src_dir = os.path.join(os.path.dirname(__file__))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from core.inference.model_client import infer as local_infer
            return local_infer(prompt)
        except Exception:
            return None

    def _parse_and_write(self, raw: str, tmpdir: Path):
        """Parse the structured response and write files to tmpdir."""
        # Split on === FILE === markers
        parts = re.split(r"=== (.+?) ===", raw)
        # parts[0] is preamble, then pairs of (filename, content)
        i = 1
        while i + 1 < len(parts):
            filename = parts[i].strip()
            content = parts[i + 1].strip()
            # Strip markdown code fences (```yaml, ```python, ``` etc.)
            content = re.sub(r'^```[a-zA-Z]*\n', '', content)
            content = re.sub(r'\n```$', '', content)
            content = content.strip()
            out_path = tmpdir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content + "\n")
            i += 2

    def validate_synthesis(self, skill_dir: Path) -> tuple[bool, str]:
        """
        Check generated skill has valid SKILL.md frontmatter + parseable exec script.
        Returns (passed, reason).
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return False, "Missing SKILL.md"

        text = skill_md.read_text(errors="replace")
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            return False, "SKILL.md missing YAML frontmatter"

        try:
            import yaml
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception as e:
            return False, f"SKILL.md frontmatter parse error: {e}"

        required = ["name", "description"]
        for field in required:
            if not fm.get(field):
                return False, f"SKILL.md missing required field: {field}"

        exec_field = fm.get("exec")
        if exec_field:
            exec_path = skill_dir / exec_field
            if not exec_path.exists():
                return False, f"exec field points to missing file: {exec_field}"

        return True, "OK"

    def install(self, skill_dir: Path, skill_name: str) -> bool:
        """Move validated skill to private/skills/<name> and persist as git repo."""
        dest = Path(self.private_skills) / skill_name
        if dest.exists():
            return True  # Already installed — idempotent
        try:
            os.makedirs(self.private_skills, exist_ok=True)
            shutil.move(str(skill_dir), str(dest))
            self._persist_to_git(dest, skill_name)
            return True
        except Exception:
            return False

    def _persist_to_git(self, skill_dir: Path, skill_name: str):
        """Init a git repo in the synthesised skill dir and push to remote."""
        import subprocess
        try:
            subprocess.run(["git", "init"], cwd=skill_dir, check=True,
                           capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=skill_dir, check=True,
                           capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"feat: synthesised skill {skill_name} (ADR-004 Tier 2)"],
                cwd=skill_dir, check=True, capture_output=True
            )
            # Push to dedicated synthesis repo if GH_SYNTHESIS_REPO env var is set
            synthesis_repo = os.environ.get(
                "GH_SYNTHESIS_REPO",
                f"fabiopacifici-bot/kernel-synthesized-skills"
            )
            remote_url = f"https://github.com/{synthesis_repo}.git"
            # Use gh CLI to create remote branch (handles auth via gh token)
            result = subprocess.run(
                ["gh", "repo", "view", synthesis_repo],
                capture_output=True
            )
            if result.returncode != 0:
                # Repo doesn't exist yet — create it
                subprocess.run(
                    ["gh", "repo", "create", synthesis_repo,
                     "--private", "--description",
                     "Auto-synthesised Kernel skills (ADR-004 Tier 2)"],
                    capture_output=True
                )
            subprocess.run(
                ["git", "remote", "add", "origin", remote_url],
                cwd=skill_dir, capture_output=True
            )
            subprocess.run(
                ["git", "push", "-u", "origin",
                 f"HEAD:refs/heads/{skill_name}"],
                cwd=skill_dir, capture_output=True
            )
            logger.info(f"[synthesizer] {skill_name} pushed to {synthesis_repo}/{skill_name}")
        except Exception as e:
            # Non-fatal — skill is installed locally even if git persist fails
            logger.warning(f"[synthesizer] git persist failed for {skill_name}: {e}")
