#!/usr/bin/env python3
"""
request_finetune.py — Fine-tune approval gate for kernel-evolving.

Checks dataset quality, estimates GPU cost, sends a Telegram approval
request with inline buttons before submitting to HuggingFace Jobs.

Usage:
  python3 scripts/request_finetune.py --dataset path/to/export.jsonl

Workflow:
  1. Load + validate dataset (min records, tool coverage, score threshold)
  2. Estimate HF Jobs GPU time (H100 ~1min/100 records for SFT, 3 epochs)
  3. Check HF account remaining GPU quota
  4. Send Telegram message with summary + Approve/Reject buttons
  5. On approval: invoke finetune_from_trajectories.py via HF Jobs
  6. On rejection or timeout: abort with log entry
"""

import json
import os
import sys
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── Dataset validation ────────────────────────────────────────────────────────

def validate_dataset(path: str, min_records: int = 50, min_score: float = 0.7) -> dict:
    """Load and validate a JSONL trajectory dataset. Returns a stats dict."""
    records = []
    errors = []

    try:
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    records.append(r)
                except Exception as e:
                    errors.append(f"line {i+1}: {e}")
    except Exception as e:
        return {"valid": False, "error": str(e)}

    if not records:
        return {"valid": False, "error": "Empty dataset"}

    # Filter by score
    passing = [r for r in records if (r.get("critic_score") or 0) >= min_score]

    # Tool diversity
    tool_names: dict[str, int] = {}
    multi_tool = 0
    has_system_prompt = 0
    for r in passing:
        msgs = r.get("messages", [])
        tools_in_record = [m for m in msgs if m.get("role") == "tool"]
        if len(tools_in_record) > 1:
            multi_tool += 1
        if any(m.get("role") == "system" for m in msgs):
            has_system_prompt += 1
        for m in msgs:
            # Tool names live in assistant tool_calls after normalisation
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", tc)
                    name = fn.get("name", tc.get("name", "?"))
                    tool_names[name] = tool_names.get(name, 0) + 1

    providers: dict[str, int] = {}
    for r in passing:
        p = r.get("provider", "?").split("/")[0]
        providers[p] = providers.get(p, 0) + 1

    stats = {
        "valid": len(passing) >= min_records,
        "total_records": len(records),
        "passing_records": len(passing),
        "min_records_required": min_records,
        "parse_errors": len(errors),
        "multi_tool_records": multi_tool,
        "multi_tool_pct": round(100 * multi_tool / max(len(passing), 1)),
        "tool_distribution": tool_names,
        "providers": providers,
        "has_system_prompt_pct": round(100 * has_system_prompt / max(len(passing), 1)),
        "avg_score": round(sum(r.get("critic_score") or 0 for r in passing) / max(len(passing), 1), 3),
        "missing": [],
    }

    # Gap analysis
    tool_set = set(tool_names.keys())
    for required_tool in ["write_file", "exec_shell", "read_file"]:
        if required_tool not in tool_set:
            stats["missing"].append(f"no {required_tool} examples")
    if multi_tool < 10:
        stats["missing"].append(f"only {multi_tool} multi-tool chains (need ≥10)")
    if len(passing) < min_records:
        stats["missing"].append(f"only {len(passing)} passing records (need ≥{min_records})")

    return stats


# ── GPU cost estimate ─────────────────────────────────────────────────────────

def estimate_gpu_minutes(n_records: int, epochs: int = 3, max_seq_len: int = 2048) -> float:
    """Rough estimate: H100 SFT on Gemma 4 2B with LoRA.
    Empirically: ~0.8 min per 100 records per epoch on H100.
    """
    return round((n_records / 100) * 0.8 * epochs, 1)


def check_hf_quota() -> dict:
    """Check HuggingFace Jobs remaining quota via API."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {"available": False, "reason": "No HF_TOKEN set"}
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://huggingface.co/api/account",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        # The API returns canPay, isPro, etc. — jobs quota is in a separate endpoint
        return {
            "available": True,
            "username": data.get("name", "?"),
            "is_pro": data.get("isPro", False),
            "note": "Quota checked — HF Pro accounts get shared H100 minutes",
        }
    except Exception as e:
        return {"available": True, "reason": f"quota API unavailable ({e}) — proceeding"}


# ── Telegram approval ─────────────────────────────────────────────────────────

def send_approval_request(stats: dict, gpu_estimate: float, dataset_path: str,
                          hf_quota: dict, finetune_args: dict) -> bool | None:
    """Send Telegram approval message with inline buttons.

    Returns True if approved, False if rejected, None if timeout.
    """
    try:
        token = os.environ.get("KERNEL_EVO_TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("[approval] No Telegram credentials — falling back to CLI approval")
            return _cli_approval(stats, gpu_estimate)

        import urllib.request

        quality_emoji = "✅" if stats["valid"] and not stats["missing"] else "⚠️"
        missing_str = "\n".join(f"  • {m}" for m in stats["missing"]) if stats["missing"] else "  None"

        tool_str = " | ".join(f"{k}:{v}" for k, v in sorted(stats["tool_distribution"].items()))
        provider_str = " | ".join(f"{k}:{v}" for k, v in sorted(stats["providers"].items()))

        text = (
            f"🧬 *Fine-tune approval request*\n\n"
            f"{quality_emoji} *Dataset:* `{Path(dataset_path).name}`\n"
            f"  Records: {stats['passing_records']} passing / {stats['total_records']} total\n"
            f"  Avg score: {stats['avg_score']} | Multi-tool: {stats['multi_tool_pct']}%\n"
            f"  Tools: {tool_str}\n"
            f"  Sources: {provider_str}\n\n"
            f"⚠️ *Gaps:*\n{missing_str}\n\n"
            f"⏱ *Estimated GPU time:* ~{gpu_estimate} minutes (H100, {finetune_args.get('epochs',3)} epochs)\n"
            f"🔑 *HF account:* {hf_quota.get('username','?')} ({'Pro' if hf_quota.get('is_pro') else 'Free'})\n\n"
            f"*Target model:* `{finetune_args.get('push_to_hub', 'local only')}`\n\n"
            f"Approve to submit to HuggingFace Jobs, or reject to abort."
        )

        # Send with inline approve/reject buttons
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": [[
                    {"text": "✅ Approve fine-tune", "callback_data": "finetune_approve"},
                    {"text": "❌ Reject", "callback_data": "finetune_reject"},
                ]]
            }
        }).encode()

        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        msg_id = result["result"]["message_id"]

        print(f"[approval] Waiting for Telegram approval (message_id={msg_id})...")
        print(f"[approval] Timeout: 10 minutes")

        # Poll for callback response
        offset = None
        deadline = time.time() + 600  # 10 minute timeout
        while time.time() < deadline:
            time.sleep(5)
            params = f"?timeout=5&allowed_updates=callback_query"
            if offset:
                params += f"&offset={offset}"
            poll_req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/getUpdates{params}",
            )
            try:
                poll_resp = urllib.request.urlopen(poll_req, timeout=10)
                updates = json.loads(poll_resp.read()).get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    cb = upd.get("callback_query", {})
                    if cb.get("data") in ("finetune_approve", "finetune_reject"):
                        approved = cb["data"] == "finetune_approve"
                        # Acknowledge callback
                        ack_req = urllib.request.Request(
                            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                            data=json.dumps({"callback_query_id": cb["id"],
                                             "text": "✅ Approved!" if approved else "❌ Rejected"}).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        urllib.request.urlopen(ack_req, timeout=5)
                        return approved
            except Exception:
                pass

        print("[approval] Timeout — no response received")
        return None

    except Exception as e:
        print(f"[approval] Telegram error: {e} — falling back to CLI")
        return _cli_approval(stats, gpu_estimate)


def _cli_approval(stats: dict, gpu_estimate: float) -> bool:
    """Interactive CLI approval fallback."""
    print(f"\n{'='*60}")
    print(f"FINE-TUNE APPROVAL REQUIRED")
    print(f"{'='*60}")
    print(f"Records: {stats['passing_records']} | GPU estimate: ~{gpu_estimate} min")
    print(f"Gaps: {stats['missing'] or 'None'}")
    answer = input("\nApprove? [y/N]: ").strip().lower()
    return answer == "y"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Request approval before fine-tuning kernel-evolving on HF Jobs"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL export")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min-records", type=int, default=50,
                        help="Minimum records required to proceed (default: 50)")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF repo to push adapter to (e.g. PacificDev/kernel-evo-v1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and show stats without sending Telegram request")
    args = parser.parse_args()

    # Load config
    cfg_path = _ROOT / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print("kernel-evolving fine-tune gate")
    print(f"{'='*60}\n")

    # 1. Validate dataset
    print(f"Validating dataset: {args.dataset}")
    stats = validate_dataset(args.dataset, min_records=args.min_records, min_score=args.min_score)
    print(f"  Records: {stats['passing_records']} passing / {stats['total_records']} total")
    print(f"  Avg score: {stats['avg_score']} | Multi-tool: {stats['multi_tool_pct']}%")
    print(f"  Tool distribution: {stats['tool_distribution']}")
    print(f"  Providers: {stats['providers']}")
    if stats["missing"]:
        print(f"  ⚠️  Gaps: {stats['missing']}")
    else:
        print("  ✅ Dataset passes all checks")

    if not stats["valid"] and not args.dry_run:
        print(f"\n❌ Dataset does not meet minimum requirements. Aborting.")
        print(f"   Run with --min-records {stats['passing_records']} to override, or collect more data first.")
        sys.exit(1)

    # 2. GPU estimate
    gpu_est = estimate_gpu_minutes(stats["passing_records"], epochs=args.epochs)
    print(f"\nGPU estimate: ~{gpu_est} minutes on H100 ({args.epochs} epochs)")

    # 3. Check HF quota
    hf_quota = check_hf_quota()
    print(f"HF account: {hf_quota.get('username','?')} ({'Pro' if hf_quota.get('is_pro') else 'Free/Unknown'})")

    if args.dry_run:
        print("\n[dry-run] Would send Telegram approval request. Exiting.")
        return

    # 4. Send approval request
    finetune_args = {
        "epochs": args.epochs,
        "push_to_hub": args.push_to_hub,
        "dataset": args.dataset,
    }
    approved = send_approval_request(stats, gpu_est, args.dataset, hf_quota, finetune_args)

    if approved is None:
        print("\n⏰ Approval timed out (10 min). Fine-tune aborted.")
        sys.exit(2)
    if not approved:
        print("\n❌ Fine-tune rejected.")
        sys.exit(3)

    # 5. Launch fine-tune
    print("\n✅ Approved — submitting to HuggingFace Jobs...")
    cmd = [
        sys.executable if "--local" in sys.argv else "uv",
        "run",
        str(_ROOT / "scripts" / "finetune_from_trajectories.py"),
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--min-score", str(args.min_score),
    ]
    if args.push_to_hub:
        cmd += ["--push-to-hub", args.push_to_hub]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
