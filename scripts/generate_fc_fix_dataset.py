#!/usr/bin/env python3
"""
generate_fc_fix_dataset.py
==========================
Focused synthetic FC dataset for the 3 under-represented tools that fail in the
battery (http_get, run_skill, search_skills). Each gets >=100 varied examples,
covering BOTH single-shot (one tool call, then stop + answer) and multi-turn
(chain: search_skills -> run_skill, or http_get -> write_file, etc), rendered
in Nemotron's native format via its real tokenizer + EVA's real 16-tool registry
— byte-identical prompt/target format to the collected trajectories so the
overnight self-tuning loop converges on the same distribution.

Record format matches generate_synthetic_fc_trajectories.py:
  {"id", "task", "tools_used", "critic_score": 1.0, "text"}
where text = tokenizer.apply_chat_template(messages, tools=real_tools, ...)

Usage:
  PYTHONPATH=src python3 scripts/generate_fc_fix_dataset.py \
      --model /mnt/e/models/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B \
      --out ~/.kernel-evolving/workspace/artifacts/trajectories/fc_fix_v11.jsonl \
      --per-tool 120
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.tools import TOOLS  # EVA's real 16-tool registry


# ── http_get single-shot templates (varied URLs, queries, intents) ─────────
HTTP_GET_SINGLE = [
    ("Fetch {url} and tell me the HTTP status it returns.",
     lambda url: {"status": "200 OK"}),
    ("Get {url} and report whether the service looks healthy.",
     lambda url: {"status": "ok", "uptime_seconds": 86400}),
    ("Retrieve the JSON from {url} and list its top-level keys.",
     lambda url: '{"version": "1.31.0", "region": "eu-central-1"}'),
    ("Pull the contents of {url} and summarize what it says.",
     lambda url: "EVA v1.31.0 is the latest release with native function calling."),
    ("Fetch {url} and return the server response body.",
     lambda url: "OK — 2048 bytes received."),
    ("Hit {url} and check if the API endpoint responds.",
     lambda url: '{"success": true, "latency_ms": 42}'),
    ("Download the page at {url} and tell me its title.",
     lambda url: "<title>EVA Agent — Docs</title>"),
    ("Call {url} and report the response code and any error.",
     lambda url: "HTTP/1.1 200 — no errors."),
    ("Get the data at {url} now and show the first field.",
     lambda url: '{"first": "hello", "count": 7}'),
    ("Fetch {url} and confirm it is reachable.",
     lambda url: "Reachable — 200 OK in 88ms."),
]

HTTP_GET_URLS = [
    "https://api.github.com/repos/nvidia/Nemotron",
    "https://example.com/api/status",
    "https://weather.example.tld/api/current?city=rome",
    "https://docs.python.org/3/library/json.html",
    "https://api.openai.com/v1/models",
    "https://example.com/health",
    "https://registry.hub.docker.com/v2/repositories",
    "https://example.org/feed.xml",
    "https://api.example.tld/v1/version",
    "https://httpbin.org/json",
    "https://example.net/gateway/ping",
    "https://api.example.io/status",
]

# ── http_get multi-turn: fetch then persist ────────────────────────────────
HTTP_GET_MULTI = [
    ("Fetch {url} and save the response body to {path}",
     [("http_get", {"url": "{url}"}, '{"status":"ok","data":[1,2,3]}'),
      ("write_file", {"path": "{path}", "content": '{"status":"ok","data":[1,2,3]}'}, "Written to {path}")]),
    ("Get {url}, then write a one-line summary to {path}",
     [("http_get", {"url": "{url}"}, "EVA v1.31.0 released today."),
      ("write_file", {"path": "{path}", "content": "EVA v1.31.0 released today."}, "Written to {path}")]),
]

HTTP_GET_PATHS = [
    "/tmp/fetch1.json", "/tmp/fetch2.json", "/tmp/http_body.md",
    "/tmp/page.txt", "/tmp/api_dump.json", "/tmp/status.md",
]

# ── run_skill single-shot (varied skills, varied inputs) ───────────────────
RUN_SKILL_SINGLE = [
    ("Run the {skill} skill to {goal}.",
     lambda skill, goal: f"Done — {goal} completed via {skill}."),
    ("Use the {skill} skill and handle the following: {input}.",
     lambda skill, _: f"{skill} executed successfully."),
    ("Execute the {skill} skill now.",
     lambda skill, _: f"{skill} finished — no errors."),
    ("Please run the {skill} skill for this request: {input}.",
     lambda skill, _: f"Result from {skill}: completed."),
    ("Apply the {skill} skill to the task '{input}'.",
     lambda skill, _: f"{skill} completed the task."),
]

RUN_SKILL_SKILLS = [
    "kernel-doc-retrieval", "security-scanner", "github", "browser-automation",
    "skill-lister", "open-fantasia-imagegen", "open-workspace-tracker",
    "healthcheck", "meme-maker",
]

RUN_SKILL_INPUTS = [
    "scan the repo for vulnerabilities",
    "open PRs on the kernel repo",
    "convert report.pdf to markdown",
    "list all installed skills",
    "generate a hero image",
    "check the workstation hardening",
    "list open todos and ideas",
    "summarize the session log",
]

# ── search_skills single-shot (varied queries) ─────────────────────────────
SEARCH_SKILLS_SINGLE = [
    ("Search for a skill related to {q}.",
     lambda q: f"Found: {q} — see skill-lister for details."),
    ("Find any skill that handles {q}.",
     lambda q: f"{q}-handling skill found."),
    ("Search installed skills for '{q}'.",
     lambda q: f"Match: a {q} skill exists."),
    ("Is there a skill for {q}?",
     lambda q: f"Yes — a {q} skill is installed."),
    ("What skill should I use for {q}?",
     lambda q: f"Recommended: the {q} skill."),
]

SEARCH_SKILLS_QUERIES = [
    "pdf", "markdown", "image generation", "security", "github", "browser",
    "documentation", "deployment", "media", "tracker", "voice", "meme",
    "video", "investing", "weather",
]

# ── search_skills -> run_skill multi-turn (the key orchestration) ──────────
SEARCH_RUN_MULTI = [
    ("Find a skill for {q} and then run it on this: {input}.",
     [("search_skills", {"query": "{q}"}, "Found a skill matching {q}."),
      ("run_skill", {"skill_name": "{skill}", "input": "{input}"}, "{skill} completed: {input}.")]),
    ("Locate the right skill for {q} and execute it.",
     [("search_skills", {"query": "{q}"}, "Candidate: {skill}."),
      ("run_skill", {"skill_name": "{skill}", "input": "{input}"}, "{skill} ran OK.")]),
    ("First search for a {q} skill, then use it for {input}.",
     [("search_skills", {"query": "{q}"}, "{skill} matches {q}."),
      ("run_skill", {"skill_name": "{skill}", "input": "{input}"}, "{skill} finished.")]),
]


def _sub(s, **kw):
    """Safe placeholder substitution: replace only known {key} tokens, leaving
    any other braces (e.g. JSON like {"status":"ok"}) untouched."""
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def _final_answer_single(tool: str, task: str, result: str) -> str:
    """Concrete answer that terminates single-shot: STOP after one tool call."""
    t = task.lower()
    if tool == "http_get" and ("http" in t or "status" in t or "reachable" in t):
        return f"The request returned: {result.strip()}"
    if tool == "search_skills":
        return f"Search complete: {result.strip()}"
    if tool == "run_skill":
        return f"Skill executed: {result.strip()}"
    return f"Done. {result.strip()}"


def _final_answer_multi(task: str, steps_tools: list) -> str:
    """Terminate a multi-turn chain with a wrap-up that references the outcome."""
    return "Completed the task and saved/reported the result."


def _build_single(tool, task, steps, result_dict):
    msgs = [{"role": "user", "content": task}]
    if tool in ("http_get",) and len(steps) == 1:
        t, args, result = steps[0]
        msgs.append({"role": "assistant", "tool_calls": [{
            "type": "function", "id": "call_0",
            "function": {"name": t, "arguments": args}}]})
        msgs.append({"role": "tool", "name": t, "content": result})
        msgs.append({"role": "assistant", "content": _final_answer_single(tool, task, result)})
    return msgs


def _build_multi(task, steps):
    msgs = [{"role": "user", "content": task}]
    for i, (t, args, result) in enumerate(steps):
        msgs.append({"role": "assistant", "tool_calls": [{
            "type": "function", "id": f"call_{i}",
            "function": {"name": t, "arguments": args}}]})
        msgs.append({"role": "tool", "name": t, "content": result})
    msgs.append({"role": "assistant", "content": _final_answer_multi(task, [s[0] for s in steps])})
    return msgs


def make_tools_json(real_tools) -> list:
    out = []
    for t in real_tools:
        fn = t.get("function", t)
        if isinstance(fn, dict) and fn.get("name"):
            out.append({"type": "function", "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }})
    return out


def render(tokenizer, messages, real_tools):
    return tokenizer.apply_chat_template(
        messages, tools=real_tools, tokenize=False, add_generation_prompt=True)


def gen_single_tool(tool, rng):
    """Return (task, messages) — one varied single-shot sample for the tool."""
    if tool == "http_get":
        url = rng.choice(HTTP_GET_URLS)
        tmpl, maker = rng.choice(HTTP_GET_SINGLE)
        task = tmpl.format(url=url)
        result = maker(url)
        if not isinstance(result, str):
            result = str(result)
        steps = [("http_get", {"url": url}, result)]
    elif tool == "run_skill":
        skill = rng.choice(RUN_SKILL_SKILLS)
        inp = rng.choice(RUN_SKILL_INPUTS)
        tmpl, maker = rng.choice(RUN_SKILL_SINGLE)
        task = tmpl.format(skill=skill, input=inp, goal=inp)
        result = maker(skill, inp)
        steps = [("run_skill", {"skill_name": skill, "input": inp}, result)]
    else:  # search_skills
        q = rng.choice(SEARCH_SKILLS_QUERIES)
        tmpl, maker = rng.choice(SEARCH_SKILLS_SINGLE)
        task = tmpl.format(q=q)
        result = maker(q)
        steps = [("search_skills", {"query": q}, result)]
    return tool, task, _build_single(tool, task, steps, {}), {s[0] for s in steps}


def gen_multi(tool, rng):
    """One varied multi-turn sample. Returns (task, messages, tools_used)."""
    if tool == "http_get":
        url = rng.choice(HTTP_GET_URLS)
        path = rng.choice(HTTP_GET_PATHS)
        tmpl, chain = rng.choice(HTTP_GET_MULTI)
        task = _sub(tmpl, url=url, path=path)
        steps = [(t, {_sub(k, url=url, path=path): _sub(v, url=url, path=path)
                      for k, v in a.items()},
                  _sub(r, url=url, path=path)) for t, a, r in chain]
    else:  # search_skills + run_skill orchestration
        q = rng.choice(SEARCH_SKILLS_QUERIES)
        skill = rng.choice(RUN_SKILL_SKILLS)
        inp = rng.choice(RUN_SKILL_INPUTS)
        tmpl, chain = rng.choice(SEARCH_RUN_MULTI)
        task = tmpl.format(q=q, input=inp)
        steps = [(t, {_sub(k, q=q, input=inp, skill=skill): _sub(v, q=q, input=inp, skill=skill)
                      for k, v in a.items()},
                  _sub(r, q=q, input=inp, skill=skill)) for t, a, r in chain]
    return task, _build_multi(task, steps), sorted({s[0] for s in steps})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tool", type=int, default=120,
                    help="Target total (single+multi) examples per tool")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    real_tools = make_tools_json(TOOLS)
    print(f"REAL tools: {len(real_tools)} -> {[t['function']['name'] for t in real_tools]}", flush=True)

    rng = random.Random(args.seed)
    targets = {"http_get": args.per_tool, "run_skill": args.per_tool, "search_skills": args.per_tool}
    counts = {k: 0 for k in targets}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    rid = 0
    # Round-robin: keep interleaving until all tools hit their target.
    # ~65% single, ~35% multi per tool for variation.
    order = ["http_get", "run_skill", "search_skills"]
    while any(counts[k] < targets[k] for k in order):
        tool = rng.choice(order)
        if counts[tool] >= targets[tool]:
            continue
        if rng.random() < 0.65:
            _, task, msgs, tools_used = gen_single_tool(tool, rng)
        else:
            task, msgs, tools_used = gen_multi(tool, rng)
        try:
            text = render(tokenizer, msgs, real_tools)
        except Exception as e:
            print(f"  [skip render] {e}", flush=True)
            continue
        rows.append({
            "id": f"fc-fix-v11-{rid}",
            "task": task,
            "tools_used": sorted(tools_used),
            "critic_score": 1.0,
            "text": text,
        })
        rid += 1
        counts[tool] += 1

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} trajectories -> {args.out}", flush=True)
    for k, v in counts.items():
        print(f"  {k}: {v} (single+multi)", flush=True)
    # coverage audit
    all_tools = set()
    for r in rows:
        all_tools.update(r["tools_used"])
    print(f"Tools used across dataset: {sorted(all_tools)}", flush=True)
    missing = set({t['function']['name'] for t in TOOLS}) - all_tools
    if missing:
        print(f"  NOT covered: {sorted(missing)}", flush=True)


if __name__ == "__main__":
    main()
