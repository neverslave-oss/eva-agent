#!/usr/bin/env python3
"""
generate_v5_trajectories.py
============================
v5 trajectory set — three tracks:

  Gap A — read_file + reason + write (T04 fix, 20 tasks, double of original plan)
    Re-runs the full Gap A from generate_targeted_gap_trajectories.py.

  Gap B — http_get + write (T05 fix, 20 tasks, double of original plan)
    Re-runs the full Gap B from generate_targeted_gap_trajectories.py.

  Gap F — vision / kernel-doc-retrieval skill (20 tasks)
    Teaches Gemma 4 to use its vision capability via the kernel-doc-retrieval
    skill: /markdown on PDFs, /doc queries, image-based reasoning.
    Tool path: run_skill("kernel-doc-retrieval", ...)

  Gap G — voice-clone skill (20 tasks)
    Teaches Gemma 4 to generate cloned audio via the voice-clone skill.
    Tool path: run_skill("voice-clone", ...)

  Gap H — multimodal chaining: PDF → voice (10 tasks)
    Extract text from a document then narrate it with voice clone.
    Chains run_skill("kernel-doc-retrieval") + run_skill("voice-clone").

Total: ~90 targeted tasks for v5 adapter.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from trajectory_collector import TrajectoryCollector
from model import infer_with_tools
from tools import TOOLS
from context import build_system_prompt
import yaml

# Import Gap A and Gap B from existing script
from generate_targeted_gap_trajectories import (
    GAP_A_TASKS,
    GAP_B_TASKS,
    run_teacher_task,
    run_multi_turn,
)

WORKSPACE = Path.home() / ".kernel-evolving/workspace"
VOICE_SAMPLE = str(Path.home() / ".openclaw/media/voice-samples/fabio-ita-phonetic.wav")

# ── Gap F: vision / kernel-doc-retrieval ─────────────────────────────────────
# Gemma 4 has native vision. kernel-doc-retrieval skill exposes it via run_skill.
# Tasks teach the model to route document/image tasks through the skill.

GAP_F_TASKS = [
    # PDF → markdown extraction
    ("Use the kernel-doc-retrieval skill to convert the PDF at ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf to markdown and save the result.",
     ["run_skill"], "gap-F:pdf-to-markdown"),
    ("Extract text from the PDF ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf using the kernel-doc-retrieval skill. Report how many pages were processed.",
     ["run_skill"], "gap-F:pdf-page-count"),
    ("Use /markdown to convert ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf and then write a one-paragraph summary of the extracted content to ~/evo_pdf_summary.md",
     ["run_skill", "write_file"], "gap-F:pdf-extract-summarise"),
    ("The user has a PDF invoice at /tmp/invoice.pdf — they want it converted to markdown. Route this to the kernel-doc-retrieval skill.",
     ["run_skill"], "gap-F:pdf-invoice-route"),
    ("Anonymize the document at ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf using the kernel-doc-retrieval skill /anonymize command.",
     ["run_skill"], "gap-F:pdf-anonymize"),
    ("Use the kernel-doc-retrieval skill to redact PII from /tmp/contract.pdf. Report what was done.",
     ["run_skill"], "gap-F:pdf-redact-pii"),
    ("The user says: 'can you read this PDF for me?' and attaches ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf. Use the doc-retrieval skill to extract and summarise it.",
     ["run_skill"], "gap-F:pdf-read-user-request"),
    ("Use the /doc command via kernel-doc-retrieval to query: 'what is the main topic?' against ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf",
     ["run_skill"], "gap-F:pdf-doc-query"),
    ("Convert ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf to markdown, then count the number of headings in the output and write 'Heading count: N' to ~/evo_pdf_headings.txt",
     ["run_skill", "write_file"], "gap-F:pdf-extract-count-headings"),
    ("Use kernel-doc-retrieval to extract the first page of ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf and write the text to ~/evo_first_page.txt",
     ["run_skill", "write_file"], "gap-F:pdf-first-page"),
    # Vision / image reasoning via Gemma 4
    ("Describe what you see in the image at ~/.openclaw/workspace/repositories/kernel-evolving/docs/architecture.png using the vision capability via kernel-doc-retrieval. Write the description to ~/evo_vision_desc.md",
     ["run_skill", "write_file"], "gap-F:vision-image-describe"),
    ("Use Gemma 4 vision via the kernel-doc-retrieval skill to read any text visible in ~/.openclaw/media/screenshots/latest.png. Write the extracted text to ~/evo_screenshot_text.txt",
     ["run_skill", "write_file"], "gap-F:vision-screenshot-ocr"),
    ("The user uploads a diagram PNG at /tmp/diagram.png. Use kernel-doc-retrieval to extract any text or labels from the image.",
     ["run_skill"], "gap-F:vision-diagram-labels"),
    ("Run /markdown on ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf then write the title (first heading found) to ~/evo_pdf_title.txt",
     ["run_skill", "write_file"], "gap-F:pdf-extract-title"),
    ("Use kernel-doc-retrieval to convert /tmp/report.pdf to markdown. If the skill succeeds, write 'DONE' to ~/evo_pdf_status.txt; if it fails, write 'ERROR: <reason>'.",
     ["run_skill", "write_file"], "gap-F:pdf-status-check"),
    ("The user asks: 'what does page 2 of this PDF say?' pointing to ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf. Use kernel-doc-retrieval to extract and answer.",
     ["run_skill"], "gap-F:pdf-page-specific-query"),
    ("Use the kernel-doc-retrieval skill to extract all tables from ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf and write them as markdown to ~/evo_pdf_tables.md",
     ["run_skill", "write_file"], "gap-F:pdf-extract-tables"),
    ("Convert the legal document at ~/.openclaw/workspace-legal/documents/neverslave/IP-Agreement-EN.pdf to markdown using kernel-doc-retrieval and save to ~/evo_legal_doc.md",
     ["run_skill", "write_file"], "gap-F:pdf-legal-doc"),
    ("Use kernel-doc-retrieval /anonymize on ~/.openclaw/workspace-legal/documents/neverslave/IP-Agreement-EN.pdf and write a summary of what PII was found to ~/evo_pii_found.txt",
     ["run_skill", "write_file"], "gap-F:pdf-pii-summary"),
    ("The user says: 'Kernel, read this file and tell me the author' pointing to a PDF at /tmp/paper.pdf. Route to kernel-doc-retrieval, extract, find the author field, reply with it.",
     ["run_skill"], "gap-F:pdf-author-extract"),
]

# ── Gap G: voice-clone skill ──────────────────────────────────────────────────
# Teaches Gemma 4 to use the voice-clone skill via run_skill.
# Voice sample: fabio-ita-phonetic.wav (exists locally).

GAP_G_TASKS = [
    # Basic TTS clone requests
    (f"Use the voice-clone skill to synthesise 'Ciao, come stai?' using the voice sample at {VOICE_SAMPLE}. Save the output WAV to ~/evo_voice_test.wav",
     ["run_skill"], "gap-G:voice-basic-ita"),
    (f"Generate a cloned voice saying 'Welcome to Kernel Evolving' using the sample at {VOICE_SAMPLE} and save to ~/evo_welcome.wav",
     ["run_skill"], "gap-G:voice-basic-eng"),
    (f"The user says: 'Say this in my voice: Buongiorno a tutti'. Use voice-clone with sample {VOICE_SAMPLE} to generate the audio.",
     ["run_skill"], "gap-G:voice-user-request-ita"),
    (f"Generate a voice clone of the phrase 'Training complete, adapter v5 is ready' using {VOICE_SAMPLE}. Save to ~/evo_training_done.wav",
     ["run_skill"], "gap-G:voice-training-announcement"),
    (f"Use voice-clone with model 0.6 to synthesise 'Il sistema è online' from sample {VOICE_SAMPLE}. Write the output path to ~/evo_voice_path.txt after generation.",
     ["run_skill", "write_file"], "gap-G:voice-model-06-ita"),
    (f"Use voice-clone with model 1.7 to synthesise 'This is a test of the higher quality model' from sample {VOICE_SAMPLE} and save to ~/evo_voice_hq.wav",
     ["run_skill"], "gap-G:voice-model-17-eng"),
    # Voice + write chaining
    (f"Generate a cloned voice reading 'Kernel Evolving — session complete' with {VOICE_SAMPLE}, save to ~/evo_session_done.wav, then write the absolute path of the generated file to ~/evo_voice_output.txt",
     ["run_skill", "write_file"], "gap-G:voice-chain-write-path"),
    (f"The user asks for a voice memo: 'remind me: Boolean presentation Thursday at 9am'. Use voice-clone with sample {VOICE_SAMPLE} to generate and save to ~/evo_reminder.wav",
     ["run_skill"], "gap-G:voice-reminder-memo"),
    # Multi-step: read text then narrate
    (f"Read the first line of ~/evo_soul_reflection.md (or write 'You are Kernel Evolving' if it does not exist), then use voice-clone with {VOICE_SAMPLE} to narrate that line and save to ~/evo_soul_audio.wav",
     ["read_file", "run_skill"], "gap-G:voice-read-then-narrate"),
    (f"Read the content of ~/evo_config_summary.md (or 'Configuration summary' if missing), take only the first sentence, and generate a voice clone of it using {VOICE_SAMPLE}. Save to ~/evo_config_audio.wav",
     ["read_file", "run_skill"], "gap-G:voice-config-narrate"),
    # Error handling
    (f"Try to use the voice-clone skill to synthesise 'test' using a non-existent sample at /tmp/missing_sample.wav. Report the error gracefully.",
     ["run_skill"], "gap-G:voice-missing-sample-error"),
    ("The user says 'generate audio for me' without specifying a sample file. Ask for the sample path before proceeding with voice-clone.",
     [], "gap-G:voice-missing-sample-clarify"),
    # Routing: when to use voice-clone vs plain text
    (f"The user says: 'I want to hear this, not read it: Kernel is a local AI agent running on your machine.' Route to voice-clone with {VOICE_SAMPLE} and save to ~/evo_hear_this.wav",
     ["run_skill"], "gap-G:voice-routing-hear"),
    (f"Generate audio for the phrase 'La settimana prossima presentiamo il corso multistack' using voice-clone with sample {VOICE_SAMPLE}. Save to ~/evo_multistack_audio.wav",
     ["run_skill"], "gap-G:voice-course-promo-ita"),
    (f"Use voice-clone to narrate the phrase 'Alert: new item in the AI feed — check your digest' using {VOICE_SAMPLE}. Save to ~/evo_alert_audio.wav",
     ["run_skill"], "gap-G:voice-alert-narrate"),
    # Short voice memos
    (f"Generate a voice clone saying 'You have 3 unread notifications' using {VOICE_SAMPLE}. Save to ~/evo_notif.wav",
     ["run_skill"], "gap-G:voice-notification"),
    (f"Use voice-clone to say 'GPU training complete. v5 adapter saved.' with {VOICE_SAMPLE}. Save to ~/evo_gpu_done.wav",
     ["run_skill"], "gap-G:voice-gpu-complete"),
    (f"The user asks: 'read my todos aloud'. Read ~/evo_todos.md (or use 'Fix T04, Fix T05, Run sim17' if missing), then use voice-clone with {VOICE_SAMPLE} to narrate the list. Save to ~/evo_todos_audio.wav",
     ["read_file", "run_skill"], "gap-G:voice-read-todos-aloud"),
    (f"Generate a short voice introduction: 'Hi, I am Kernel Evolving — your local AI agent.' with {VOICE_SAMPLE}. Save to ~/evo_intro_audio.wav",
     ["run_skill"], "gap-G:voice-self-intro"),
    (f"Use voice-clone to synthesise 'Benvenuto. Il tuo agente locale è pronto.' with {VOICE_SAMPLE}. Save to ~/evo_benvenuto.wav",
     ["run_skill"], "gap-G:voice-welcome-ita"),
]

# ── Gap H: multimodal chaining (PDF → voice) ─────────────────────────────────
# Teaches Gemma 4 to chain: read a PDF via vision, extract text, narrate it.

GAP_H_TASKS = [
    (f"Extract the first paragraph from ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf using kernel-doc-retrieval, then use voice-clone with {VOICE_SAMPLE} to narrate that paragraph. Save audio to ~/evo_pdf_narrated.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-extract-then-narrate"),
    (f"Convert ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf to markdown, take the document title, and generate a voice memo saying 'Document ready: <title>' using voice-clone with {VOICE_SAMPLE}. Save to ~/evo_doc_ready.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-title-voice-memo"),
    (f"Read ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf using kernel-doc-retrieval, write the extracted text to ~/evo_extracted.md, then narrate the first sentence with voice-clone using {VOICE_SAMPLE}.",
     ["run_skill", "write_file", "run_skill"], "gap-H:pdf-extract-write-narrate"),
    (f"Use kernel-doc-retrieval to anonymize ~/.openclaw/workspace-legal/documents/neverslave/IP-Agreement-EN.pdf, then generate a voice memo saying 'Anonymization complete' with {VOICE_SAMPLE}. Save to ~/evo_anon_done.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-anonymize-then-announce"),
    (f"Extract the abstract or first section from ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf, summarise it in one sentence, and narrate that sentence using voice-clone with {VOICE_SAMPLE}. Save audio to ~/evo_abstract_audio.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-summarise-narrate"),
    (f"Convert the PDF at ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf to text, count the words, write 'Word count: N' to ~/evo_word_count.txt, then use voice-clone with {VOICE_SAMPLE} to say 'Document has N words'. Save audio to ~/evo_word_count.wav",
     ["run_skill", "write_file", "run_skill"], "gap-H:pdf-count-words-announce"),
    (f"Read the legal agreement at ~/.openclaw/workspace-legal/documents/neverslave/IP-Agreement-EN.pdf using kernel-doc-retrieval, extract the first clause, and narrate it with voice-clone using {VOICE_SAMPLE}. Save to ~/evo_clause1_audio.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-legal-clause-narrate"),
    (f"The user says: 'read this PDF to me' pointing to ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf. Use kernel-doc-retrieval to extract text, then voice-clone with {VOICE_SAMPLE} to read the first 2 sentences aloud. Save to ~/evo_read_aloud.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-read-aloud-user-req"),
    (f"Extract text from ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf, write a 2-sentence summary to ~/evo_summary.md, then narrate the summary using voice-clone with {VOICE_SAMPLE}. Save audio to ~/evo_summary_audio.wav",
     ["run_skill", "write_file", "run_skill"], "gap-H:pdf-summarise-write-narrate"),
    (f"Use kernel-doc-retrieval to extract all section headings from ~/.openclaw/workspace/repositories/kernel-doc-retrieval/tests/sample.pdf, then generate a voice index: 'This document contains: <headings list>' using voice-clone with {VOICE_SAMPLE}. Save to ~/evo_toc_audio.wav",
     ["run_skill", "run_skill"], "gap-H:pdf-toc-voice-index"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate v5 trajectories: A+B (T04/T05 fix) + F (vision) + G (voice) + H (multimodal)")
    parser.add_argument("--batches", default="all",
                        help="Comma-separated: all, A, B, F, G, H (default: all)")
    parser.add_argument("--count", type=int, default=0, help="Max tasks per batch (0=all)")
    parser.add_argument("--export", action="store_true", help="Export JSONL after generation")
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    batches = set(args.batches.upper().split(",")) if args.batches != "all" else {"A", "B", "F", "G", "H"}

    cfg_path = _ROOT / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    col = TrajectoryCollector(config)

    total_ok = 0
    total_run = 0

    if "A" in batches:
        print(f"\n{'='*55}")
        print("Gap A — read_file + reason + write (T04 fix, 20 tasks)")
        print('='*55)
        tasks = GAP_A_TASKS[:args.count] if args.count else GAP_A_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "B" in batches:
        print(f"\n{'='*55}")
        print("Gap B — http_get + write (T05 fix, 20 tasks)")
        print('='*55)
        tasks = GAP_B_TASKS[:args.count] if args.count else GAP_B_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "F" in batches:
        print(f"\n{'='*55}")
        print("Gap F — vision / kernel-doc-retrieval (20 tasks)")
        print('='*55)
        tasks = GAP_F_TASKS[:args.count] if args.count else GAP_F_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "G" in batches:
        print(f"\n{'='*55}")
        print("Gap G — voice-clone skill (20 tasks)")
        print('='*55)
        tasks = GAP_G_TASKS[:args.count] if args.count else GAP_G_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "H" in batches:
        print(f"\n{'='*55}")
        print("Gap H — multimodal chaining: PDF → voice (10 tasks)")
        print('='*55)
        tasks = GAP_H_TASKS[:args.count] if args.count else GAP_H_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    print(f"\n{'='*55}")
    print(f"Done: {total_ok}/{total_run} passed (>= {args.min_score})")

    if args.export:
        path, count = col.export_jsonl(min_score=args.min_score)
        print(f"Exported {count} total records → {path}")


if __name__ == "__main__":
    main()
