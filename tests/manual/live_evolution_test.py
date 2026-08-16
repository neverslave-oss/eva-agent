#!/usr/bin/env python3
"""
live_evolution_test.py — Manual end-to-end test of ADR-004 evolution loop.
Runs Tier 1 (ecosystem search) + Tier 2 (GPT-5.4 synthesis) for a real task.
Uses TMP_OPEN_AI_API_KEY from environment.
"""
import os, sys, json, tempfile, shutil
from pathlib import Path

# Point at the kernel-evolving src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evolver import Evolver, EvolutionResult
from code_synthesizer import CodeSynthesizer
from evolution_log import EvolutionLog

# Use a temp ecosystem so we don't pollute real ~/.kernel
FAKE_ECOSYSTEM = tempfile.mkdtemp(prefix="kernel-evolve-test-")

config = {
    "private_skills_dir": FAKE_ECOSYSTEM,
    "skills_dir": FAKE_ECOSYSTEM,
}

print(f"\n{'='*60}")
print("ADR-004 Live Evolution Test")
print(f"Ecosystem: {FAKE_ECOSYSTEM}")
print(f"{'='*60}\n")

# ── Tier 1: search empty ecosystem ──────────────────────────────────────────
task = "convert markdown files to PDF documents"
print(f"Task: \"{task}\"\n")

evolver = Evolver(config, FAKE_ECOSYSTEM)
result = evolver.run(task)
print(f"Tier 1 result: found={result.found}, escalated={result.escalated}, gap={result.gap}\n")

# ── Tier 2: synthesize with GPT-5.4 ─────────────────────────────────────────
synth = CodeSynthesizer(config)
provider = synth.resolve_provider()
print(f"Provider resolved: {provider}")

if not provider:
    print("❌ No provider available — set TMP_OPEN_AI_API_KEY")
    sys.exit(1)

print(f"Calling {provider} to synthesize skill...\n")
tmpdir = synth.synthesize(task, result.gap or task, provider)

if not tmpdir:
    print("❌ Synthesis returned None")
    sys.exit(1)

print(f"Synthesis output in: {tmpdir}")
for f in tmpdir.rglob("*"):
    if f.is_file():
        print(f"  {f.relative_to(tmpdir)}")

# ── Validate ─────────────────────────────────────────────────────────────────
passed, reason = synth.validate_synthesis(tmpdir)
print(f"\nValidation: {'✅ PASS' if passed else '❌ FAIL'} — {reason}")

if passed:
    # Read and show SKILL.md
    skill_md = (tmpdir / "SKILL.md").read_text()
    print(f"\nGenerated SKILL.md:\n{'-'*40}\n{skill_md[:800]}\n{'-'*40}")

    installed = synth.install(tmpdir, "md-to-pdf")
    print(f"\nInstalled: {installed}")
    if installed:
        print(f"Skill at: {FAKE_ECOSYSTEM}/md-to-pdf/")
else:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ── Cleanup ──────────────────────────────────────────────────────────────────
shutil.rmtree(FAKE_ECOSYSTEM, ignore_errors=True)
print(f"\n{'='*60}")
print("✅ Evolution test complete" if passed else "⚠️  Evolution test finished (synthesis did not pass validation)")
print(f"{'='*60}\n")
