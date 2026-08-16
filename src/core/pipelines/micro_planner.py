"""
micro_planner.py — ADR-011: Detect multi-step requests and decompose into sequential sub-tasks.

MicroPlanner uses the drafter (System 1) to generate a step plan cheaply,
then main model (System 2) validates it. PlanExecutor runs each step through
agent.triage() recursively, injecting previous step outputs as context.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    order: int
    task: str               # natural language sub-task
    skill_hint: str = ""    # suggested skill name (from planner, not enforced)
    depends_on: list = field(default_factory=list)  # step indices whose output this step needs


@dataclass
class Plan:
    original: str
    steps: list  # list[PlanStep]
    rationale: str = ""


def _parse_steps(text: str) -> list[PlanStep]:
    """Parse a numbered step list from text into PlanStep objects."""
    steps = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match numbered list: "1. task" or "1) task"
        m = re.match(r'^(\d+)[.)]\s*(.+)$', line)
        if m:
            order = int(m.group(1))
            task = m.group(2).strip()
            steps.append(PlanStep(order=order, task=task))
    return steps


class MicroPlanner:
    COMPLEXITY_SIGNALS = [
        "then", "after", "next", "first", "second", "finally",
        "and then", "also", "additionally", "step", "extract",
        "fetch", "download", "convert", "save", "summarize",
        "summarise", "analyze", "analyse", "generate", "create",
        "send", "email", "upload",
    ]
    MAX_STEPS = 5

    def __init__(self, config: dict = None, provider_config: dict = None):
        self._config = config or {}
        self._provider = None
        if provider_config:
            try:
                from core.inference.provider import get_provider
                self._provider = get_provider(provider_config)
            except Exception:
                self._provider = None

    def should_plan(self, text: str) -> bool:
        """Quick heuristic: is this request multi-step?"""
        lower = text.lower()
        signal_count = sum(1 for s in self.COMPLEXITY_SIGNALS if s in lower)
        # Count sentences (rough: split on '. ', '; ', or newlines)
        sentences = re.split(r'[.;]\s+|\n', text.strip())
        sentence_count = len([s for s in sentences if s.strip()])
        threshold = self._config.get("complexity_threshold", 2)
        return signal_count >= threshold or sentence_count >= 3

    def plan(self, text: str, available_skills: list, available_routines: list) -> Optional[Plan]:
        """
        Stage 1 (drafter — fast): generate a raw step list
        Stage 2 (main model): validate steps against available skills, rewrite if needed
        Returns None if request is actually single-step after validation.
        """
        use_draft = self._config.get("draft_plan", True)
        max_steps = self._config.get("max_steps", self.MAX_STEPS)

        # Stage 1: drafter generates the raw step list
        NATIVE_TOOLS_BLOCK = (
            "NATIVE TOOLS — call these directly by their exact name. They are built-in functions, separate from skills:\n"
            "  exec_shell(command)       → run shell commands\n"
            "  read_file(path)           → read a local file\n"
            "  write_file(path, content) → write a local file (paths must be inside ~/.kernel-evolving/)\n"
            "  http_get(url)             → HTTP GET request\n"
            "  web_search(query)         → search the web\n"
            "  send_file(path)           → send a file to the user via Telegram\n"
            "  run_skill(name, input)    → execute a named skill\n"
            "  run_routine(name)         → execute a named routine\n"
            "  search_skills(query)      → find a skill by keyword\n"
            "  list_routines()           → list available routines\n"
            "  recall_memory(query)      → search long-term memory\n"
        )

        planning_prompt = (
            f"You are a task planner. Decompose the following request into EXECUTABLE steps only.\n"
            f"Rules:\n"
            f"- Each step must call one of the native tools listed below using its exact name.\n"
            f"- Steps must be concrete actions only: read a file, write a file, run a command, fetch a URL.\n"
            f"- Every step that produces a file must end with: write_file(<full_path_under_~/.kernel-evolving/>, content).\n"
            f"- Use only real file paths or real URLs that actually exist.\n"
            f"- Maximum {max_steps} steps. If single-action, output: 1. {{original request}}\n\n"
            f"{NATIVE_TOOLS_BLOCK}\n"
            f"Request: {text}\n\n"
            f"Output ONLY a numbered list of executable actions using the exact tool names above, nothing else."
        )

        draft_text = ""
        if use_draft:
            try:
                from core.inference.model_client import infer_draft
                draft_text = infer_draft(planning_prompt, max_new_tokens=8192)
                if draft_text.startswith("[model_server") or draft_text.startswith("[model_client"):
                    draft_text = ""
            except Exception as e:
                logger.debug(f"[micro_planner] drafter unavailable: {e}")

        if not draft_text:
            # Fall back to routed planning provider for stage 1
            try:
                if self._provider:
                    draft_text = self._provider.infer(
                        [{"role": "user", "content": planning_prompt}],
                        max_new_tokens=8192,
                        call_type="planning",
                    )
                else:
                    from core.inference.model_client import infer
                    draft_text = infer(
                        [{"role": "user", "content": planning_prompt}],
                        max_new_tokens=8192,
                    )
            except Exception as e:
                logger.warning(f"[micro_planner] plan generation failed: {e}")
                return None

        # Parse the draft output into PlanStep objects
        steps = _parse_steps(draft_text)
        if not steps or len(steps) <= 1:
            return None  # caller falls through to infer_with_tools

        # Stage 2: main model validates and corrects steps
        skill_names = [s.get("name", "") for s in available_skills]
        routine_names = [r.get("name", "") for r in available_routines]
        available_names = ", ".join(skill_names + routine_names)

        steps_text = "\n".join(f"{s.order}. {s.task}" for s in steps)
        validation_prompt = (
            f"Review these planned steps for the request: \"{text}\"\n"
            f"Available skills/routines: {available_names}\n\n"
            f"{NATIVE_TOOLS_BLOCK}\n"
            f"Steps:\n{steps_text}\n\n"
            f"Rules:\n"
            f"- Every step must call a native tool by its exact name (listed above) OR a skill/routine from the available list.\n"
            f"- Replace any vague step like 'read the file' with the exact call: read_file(~/.kernel-evolving/workspace/USER.md).\n"
            f"- Replace any invented or placeholder URL with the real local file path.\n"
            f"- Output ONLY the corrected numbered list."
        )

        try:
            if self._provider:
                validated_text = self._provider.infer(
                    [{"role": "user", "content": validation_prompt}],
                    max_new_tokens=8192,
                    call_type="planning",
                )
            else:
                from core.inference.model_client import infer
                validated_text = infer(
                    [{"role": "user", "content": validation_prompt}],
                    max_new_tokens=8192,
                )
            validated_steps = _parse_steps(validated_text)
            if validated_steps and len(validated_steps) > 0:
                steps = validated_steps
        except Exception as e:
            logger.debug(f"[micro_planner] validation step failed, using draft: {e}")

        if len(steps) <= 1:
            return None

        return Plan(
            original=text,
            steps=steps[:max_steps],
            rationale=f"Decomposed into {len(steps)} sequential steps",
        )


class PlanExecutor:
    """
    Executes a Plan step by step.

    Key improvements (ADR-011 v2):
    - Step persistence: results journalled to ~/.kernel-evolving/workspace/plans/<plan_id>.json
    - Critic verification: after each step, a lightweight critic checks the artifact was
      actually produced. On failure, retries the step once with the critic feedback injected.
    - Resumes incomplete plans on restart by loading the journal.
    """

    _CRITIC_PROMPT = """You are a step verifier for a task execution pipeline.
A step was supposed to produce a concrete artifact or result.

Step task: {task}
Step output: {output}

Did the step actually complete its goal? Check:
1. If the task asked to save/write a file — confirm the output says "Written to" or similar success message.
2. If the task asked to fetch/retrieve data — confirm the output contains actual data, not an error or placeholder.
3. If the task asked to produce code/text — confirm the output contains the actual code/text, not just a description of it.

Reply with JSON only: {{"pass": true/false, "reason": "one sentence", "retry_hint": "what to do differently if retrying"}}
"""

    def __init__(self, provider_config: dict = None):
        self._provider = None
        if provider_config:
            try:
                from core.inference.provider import get_provider
                self._provider = get_provider(provider_config)
            except Exception:
                self._provider = None

    def run(self, plan: Plan, triage_fn: Callable, step_callback: Optional[Callable] = None,
            chunk_callback: Optional[Callable] = None) -> str:
        """Execute plan steps with persistence and critic verification."""
        import hashlib, json as _json, time as _time
        from pathlib import Path as _Path

        # Stable plan ID from original task
        plan_id = hashlib.md5(plan.original.encode()).hexdigest()[:12]
        journal_dir = _Path(os.path.expanduser("~/.kernel-evolving/workspace/plans"))
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_path = journal_dir / f"{plan_id}.json"

        # Load persisted results if resuming
        results = {}
        if journal_path.exists():
            try:
                saved = _json.loads(journal_path.read_text())
                if saved.get("original") == plan.original:
                    results = {int(k): v for k, v in saved.get("results", {}).items()}
                    logger.info(f"[PlanExecutor] Resuming plan {plan_id} — {len(results)} steps already done")
            except Exception:
                pass

        def _save():
            journal_path.write_text(_json.dumps({
                "plan_id": plan_id, "original": plan.original,
                "results": {str(k): v for k, v in results.items()},
                "updated": _time.time(),
            }))

        def _critic_verify(task: str, output: str) -> tuple[bool, str]:
            """Critic: verifies artifact exists on disk for file-producing steps, otherwise uses LLM."""
            import re as _re
            task_lower = task.lower()

            # Hard check: if task mentions writing/saving a file, verify it actually exists
            file_patterns = [
                _re.search(r'save.*?to\s+(~?/[\w/.~-]+)', task_lower),
                _re.search(r'write.*?to\s+(~?/[\w/.~-]+)', task_lower),
                _re.search(r'save.*?([\w-]+\.(?:py|md|txt|json|yaml|sh))', task_lower),
            ]
            for fp in file_patterns:
                if fp:
                    raw_path = fp.group(1).strip()
                    check_path = os.path.expanduser(raw_path) if raw_path.startswith("~") else raw_path
                    # Only check if it looks like an actual path (not a word fragment)
                    if "/" in raw_path or "." in raw_path:
                        exists = os.path.isfile(check_path)
                        if not exists:
                            return False, (
                                f"The file {raw_path} was NOT found on disk. "
                                "You must call write_file with the full path and complete content. "
                                "Do not describe writing the file — actually call write_file."
                            )
                        return True, ""

            # Soft check: if output contains "error" or looks like a plan description, fail
            output_lower = output.lower()
            fail_signals = ["(error:", "failed to", "could not", "does not exist",
                            "i will now write", "i will create", "the next step", "step 1:"]
            for sig in fail_signals:
                if sig in output_lower:
                    return False, f"Output looks like a plan/error rather than a completed action. Actually execute the action."

            # LLM soft check for non-file steps
            try:
                prompt = PlanExecutor._CRITIC_PROMPT.format(task=task[:300], output=output[:400])
                if self._provider:
                    raw = self._provider.infer(
                        [{"role": "user", "content": prompt}],
                        max_new_tokens=512,
                        call_type="critic",
                    )
                else:
                    from core.inference.model_client import infer
                    raw = infer([{"role": "user", "content": prompt}], max_new_tokens=512)
                m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if m:
                    verdict = _json.loads(m.group(0))
                    return bool(verdict.get("pass")), verdict.get("retry_hint", "")
            except Exception:
                pass
            return True, ""

        for step in plan.steps:
            if step.order in results:
                logger.info(f"[PlanExecutor] Step {step.order} already done — skipping")
                continue

            task = step.task
            for dep in (step.depends_on or []):
                if dep in results:
                    task += f"\n\n[Output from step {dep}]:\n{results[dep]}"
            prev = step.order - 1
            if prev >= 1 and prev in results and not step.depends_on:
                task += f"\n\n[Previous step output]:\n{results[prev][:800]}"

            logger.info(f"[PlanExecutor] Step {step.order}/{len(plan.steps)}: {task[:70]!r}")

            try:
                result = triage_fn(task, step_callback=step_callback, chunk_callback=chunk_callback)
            except TypeError:
                # triage_fn may not accept chunk_callback (e.g. in tests)
                result = triage_fn(task, step_callback=step_callback)

            # Critic verification — did the step actually complete?
            passed, retry_hint = _critic_verify(task, result)
            if not passed:
                logger.info(f"[PlanExecutor] Step {step.order} critic FAIL — retrying with hint: {retry_hint}")
                retry_task = task
                if retry_hint:
                    retry_task += f"\n\n[Previous attempt failed. Critic feedback: {retry_hint}. Please try again and ensure the artifact is produced.]"
                try:
                    result = triage_fn(retry_task, step_callback=step_callback, chunk_callback=chunk_callback)
                except TypeError:
                    result = triage_fn(retry_task, step_callback=step_callback)
                except Exception as e:
                    result = f"[step {step.order} retry error: {e}]"

            results[step.order] = result
            _save()
            logger.info(f"[PlanExecutor] Step {step.order} done — journalled to {journal_path}")

        # Clean up journal on success
        try:
            journal_path.unlink()
        except Exception:
            pass

        return self._synthesise(plan.original, results, chunk_callback=chunk_callback)

    def _synthesise(self, original: str, results: dict, chunk_callback: Optional[Callable] = None) -> str:
        """Synthesise all step results into a final reply."""
        if not results:
            return "No results to synthesise."

        # Build structured input for synthesis
        steps_summary = "\n\n".join(
            f"Step {k}: {v}" for k, v in sorted(results.items())
        )
        synthesis_prompt = (
            f"The user asked: \"{original}\"\n\n"
            f"Here are the results of each step:\n\n{steps_summary}\n\n"
            f"Synthesise a clear, complete answer to the user's original request "
            f"using all the step results above."
        )

        use_draft = True  # drafter sufficient for synthesis from structured inputs
        if not chunk_callback:
            try:
                from core.inference.model_client import infer_draft
                result = infer_draft(synthesis_prompt, max_new_tokens=8192)
                if result and not result.startswith("[model_server") and not result.startswith("[model_client"):
                    return result
            except Exception:
                pass

        # Fallback to configured synthesis provider (or legacy local path)
        try:
            if self._provider:
                return self._provider.infer(
                    [{"role": "user", "content": synthesis_prompt}],
                    max_new_tokens=8192,
                    call_type="synthesis",
                )
            from core.inference.model_client import infer_with_tools
            return infer_with_tools(
                [{"role": "user", "content": synthesis_prompt}],
                tools=[],
                max_new_tokens=8192,
                chunk_callback=chunk_callback,
            )
        except Exception:
            pass
        try:
            from core.inference.model_client import infer
            return infer(
                [{"role": "user", "content": synthesis_prompt}],
                max_new_tokens=8192,
            )
        except Exception as e:
            # Last resort: concatenate step results
            return f"Multi-step results:\n{steps_summary}"
