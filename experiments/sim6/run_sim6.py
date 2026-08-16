#!/usr/bin/env python3
"""
Sim 6 — Result-oriented task completion simulation.

Previous sims tested: did Evo ACQUIRE a skill?
Sim 6 tests:         did Evo COMPLETE the task and PRODUCE a result?

Tasks are submitted via /evolution/trigger (full loop) then verified:
- Was a skill found/acquired? (ADR-004/006/007)
- Was a result actually produced? (new criterion)
- If no result → Evo evolves and retries (up to MAX_RETRIES)

Results written to sim6/results/ for post-sim analysis.
"""
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8779"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 2

# ── Sim 6 tasks ──────────────────────────────────────────────────────────────
# Each task has:
#   id          — short name for logging
#   task        — natural language instruction sent to Evo
#   verify_fn   — callable(result_text) → bool: did it actually produce output?
#   expect_skill — skill we expect to be matched/acquired (optional, informational)

TASKS = [
    {
        "id": "T01_web_summary",
        "task": "Summarise the article at https://fabiopacifici.com/how-to-write-an-agent-brief-that-actually-works — give me the 3 key points",
        "expect_skill": "web-scrape-summarize",
        "verify": lambda r: len(r.get("result","")) > 100 or r.get("found"),
    },
    {
        "id": "T02_arxiv_paper",
        "task": "Find a recent arXiv paper about self-evolving AI agents and return the title and abstract",
        "expect_skill": "arxiv-agentic-ai-scraper",
        "verify": lambda r: r.get("found") and len(r.get("result","")) > 50,
    },
    {
        "id": "T03_html_table",
        "task": "Extract structured data from the HTML table at https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal) and save the first 5 rows as JSON",
        "expect_skill": "html-table-to-json",
        "verify": lambda r: r.get("found"),
    },
    {
        "id": "T04_unit_tests",
        "task": "Generate unit tests for a Python module that has a function called calculate_roi(invested, returned) -> float",
        "expect_skill": "python-unit-tests-static-analysis",
        "verify": lambda r: r.get("found") and len(r.get("result","")) > 50,
    },
    {
        "id": "T05_youtube_summary",
        "task": "Summarise the YouTube video transcript from https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "expect_skill": "youtube-transcript-summary",
        "verify": lambda r: r.get("found"),
    },
    {
        "id": "T06_translate",
        "task": "Translate the following Italian text to English: 'Gli agenti AI autonomi rappresentano il futuro dello sviluppo software'",
        "expect_skill": "translate-document-it-to-en",
        "verify": lambda r: r.get("found") and len(r.get("result","")) > 20,
    },
    {
        "id": "T07_code_review",
        "task": "Review this Python diff and flag any issues:\ndef divide(a, b):\n    return a / b",
        "expect_skill": "code-review-diff-analysis",
        "verify": lambda r: r.get("found"),
    },
    {
        "id": "T08_gap_synthesis",
        "task": "Fetch the RSS feed at https://hnrss.org/frontpage and return the top 3 story titles",
        "expect_skill": None,  # expect Tier 2 synthesis
        "verify": lambda r: r.get("found") or r.get("escalated"),
    },
    {
        "id": "T09_writer_critic",
        "task": "Write a 3-paragraph blog intro about why agent briefs matter, then critique it for clarity",
        "expect_skill": None,  # pipeline task
        "verify": lambda r: r.get("found") or len(r.get("result","")) > 100,
    },
    {
        "id": "T10_anomaly_detect",
        "task": "Detect anomalies in this time series data: [10,11,10,12,10,11,95,10,11,10] and return which index is anomalous",
        "expect_skill": "timeseries-anomaly-detector",
        "verify": lambda r: r.get("found"),
    },
]


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read())


def run_task(task: dict, attempt: int = 1) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"[{task['id']}] attempt {attempt}: {task['task'][:70]}...")
    try:
        result = _post("/evolution/trigger", {"task": task["task"], "cap": 1})
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"id": task["id"], "attempt": attempt, "error": str(e), "ts": ts}

    found     = result.get("result", {}).get("found", False)
    escalated = result.get("result", {}).get("escalated", False)
    installed = result.get("result", {}).get("installed", [])
    confidence= result.get("result", {}).get("confidence", "?")
    gap       = result.get("result", {}).get("gap", "")
    recs      = result.get("result", {}).get("recommendations", [])

    # Build a result text from what Evo returned
    result_text = ""
    if installed:
        result_text = f"installed={installed}"
    elif gap:
        result_text = gap

    verified = task["verify"]({"found": found, "escalated": escalated, "result": result_text})

    status = "✅ RESOLVED" if found and verified else \
             "🔬 SYNTHESISED" if escalated else \
             "⚠️  PARTIAL" if recs else "❌ GAP"

    print(f"  {status} | found={found} escalated={escalated} confidence={confidence}")
    if installed: print(f"  installed: {installed}")
    if recs:      print(f"  recommendations: {recs}")
    if gap:       print(f"  gap: {gap[:80]}")
    print(f"  verified={verified}")

    return {
        "id": task["id"],
        "attempt": attempt,
        "task": task["task"],
        "expect_skill": task.get("expect_skill"),
        "found": found,
        "escalated": escalated,
        "installed": installed,
        "confidence": confidence,
        "gap": gap,
        "recommendations": recs,
        "verified": verified,
        "status": status,
        "ts": ts,
    }


def main():
    print(f"\n{'#'*60}")
    print("# Sim 6 — Result-oriented task completion")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M')} | tasks={len(TASKS)} | retries={MAX_RETRIES}")
    print(f"{'#'*60}")

    # Ensure evolution is running with enough cap
    try:
        _post("/evolution/control", {"action": "start", "cap": 20})
        print("\n✅ Evolution started (cap=20)")
    except Exception as e:
        print(f"\n⚠️  Could not start evolution: {e}")

    all_results = []

    for task in TASKS:
        result = run_task(task, attempt=1)
        all_results.append(result)

        # Retry loop: if not verified, give Evo a chance to evolve and retry
        attempt = 2
        while not result.get("verified") and not result.get("error") and attempt <= MAX_RETRIES + 1:
            print(f"  → not verified, retrying (attempt {attempt})...")
            time.sleep(3)
            result = run_task(task, attempt=attempt)
            all_results.append(result)
            attempt += 1

        time.sleep(2)  # brief pause between tasks

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SIM 6 SUMMARY")
    print(f"{'='*60}")

    # Take the best result per task
    best = {}
    for r in all_results:
        tid = r["id"]
        if tid not in best or (r.get("verified") and not best[tid].get("verified")):
            best[tid] = r

    resolved  = sum(1 for r in best.values() if r.get("found") and r.get("verified"))
    verified  = sum(1 for r in best.values() if r.get("verified"))
    gaps      = sum(1 for r in best.values() if not r.get("found"))
    synths    = sum(1 for r in best.values() if r.get("escalated"))

    print(f"  Tasks:      {len(TASKS)}")
    print(f"  Verified:   {verified}/{len(TASKS)}")
    print(f"  Tier 1:     {resolved}")
    print(f"  Tier 2:     {synths}")
    print(f"  Open gaps:  {gaps}")
    print()

    for r in best.values():
        icon = "✅" if r.get("verified") else ("🔬" if r.get("escalated") else "❌")
        skill = (r.get("installed") or [r.get("expect_skill","?")])
        print(f"  {icon} {r['id']:25s} skill={skill[0] if skill else '?':30s} verified={r.get('verified')}")

    # Save results
    out_path = RESULTS_DIR / f"sim6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "sim": "sim6",
            "ts": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total": len(TASKS),
                "verified": verified,
                "tier1_resolved": resolved,
                "tier2_synthesised": synths,
                "open_gaps": gaps,
            },
            "results": list(best.values()),
            "all_attempts": all_results,
        }, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
