#!/usr/bin/env python3
"""
auto_finetune_gate.py — Autonomous fine-tune gate watcher.

Runs as a background daemon. Every 30 minutes:
  1. Counts new clean trajectories (critic_score >= 0.7) since last fine-tune
  2. If count >= threshold: triggers fine-tune via train.sh
  3. If train succeeds: runs eval_adapter.sh
  4. If finetuned pass rate >= baseline + min_improvement: promotes adapter
  5. Sends Telegram notifications at each stage

Config (config.yaml, under evolution.finetune_gate):
  enabled: false
  trajectory_threshold: 50
  min_improvement: 0
    adapter_output_dir: ~/.kernel-evolving/workspace/artifacts/finetune
    adapter_active_dir: ~/.kernel-evolving/workspace/artifacts/adapter_active
    state_file: ~/.kernel-evolving/workspace/data/finetune_gate_state.json
  shadow_port: 8780
  shadow_socket: /tmp/kernel_evo_shadow.sock

Usage:
  python3 scripts/auto_finetune_gate.py [--config config.yaml] [--once]
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [finetune_gate] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune_gate")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Autonomous fine-tune gate watcher")
parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
parser.add_argument("--once", action="store_true", help="Run a single check then exit (for testing)")
args = parser.parse_args()

CONFIG_PATH = Path(args.config).resolve()
REPO_DIR = CONFIG_PATH.parent


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def gate_cfg(cfg: dict) -> dict:
    return cfg.get("evolution", {}).get("finetune_gate", {})


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------
def _telegram_creds() -> tuple:
    """Read bot token + chat_id from env or openclaw.json."""
    token = os.environ.get("KERNEL_EVO_TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID")

    if not token:
        try:
            oc_path = Path.home() / ".openclaw/openclaw.json"
            with open(oc_path) as f:
                oc = json.load(f)
            accounts = oc.get("channels", {}).get("telegram", {}).get("accounts", [])
            if isinstance(accounts, list) and accounts:
                acct = accounts[0]
                if isinstance(acct, dict):
                    token = acct.get("token") or acct.get("botToken")
                    chat_id = chat_id or str(acct.get("defaultChatId", ""))
        except Exception as e:
            logger.warning(f"Could not read openclaw.json for Telegram creds: {e}")

    if not token:
        try:
            env_path = REPO_DIR / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("KERNEL_EVO_TELEGRAM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                    if line.startswith("KERNEL_EVO_TELEGRAM_CHAT_ID="):
                        chat_id = line.split("=", 1)[1].strip()
        except Exception:
            pass

    return token, chat_id


def send_telegram(text: str):
    token, chat_id = _telegram_creds()
    if not token or not chat_id:
        logger.warning("Telegram creds not available — notification skipped")
        return
    try:
        payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
            if not resp.get("ok"):
                logger.warning(f"Telegram send failed: {resp}")
    except Exception as e:
        logger.warning(f"Telegram notification error: {e}")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return {"last_finetune_ts": None, "last_run_ts": None}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Trajectory counting
# ---------------------------------------------------------------------------
def count_new_trajectories(since_ts: str | None, min_score: float = 0.7) -> int:
    db_path = Path.home() / ".kernel-evolving/workspace/data/evolution.db"
    if not db_path.exists():
        logger.warning(f"Evolution DB not found: {db_path}")
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        if since_ts:
            cur = conn.execute(
                "SELECT COUNT(*) FROM task_trajectories WHERE critic_score >= ? AND ts > ?",
                (min_score, since_ts),
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(*) FROM task_trajectories WHERE critic_score >= ?",
                (min_score,),
            )
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"DB query error: {e}")
        return 0


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------
def run_train() -> int:
    train_sh = REPO_DIR / "scripts/train.sh"
    logger.info(f"Running fine-tune: {train_sh}")
    result = subprocess.run(["bash", str(train_sh)], cwd=str(REPO_DIR))
    return result.returncode


# ---------------------------------------------------------------------------
# Eval adapter + parse results
# ---------------------------------------------------------------------------
def run_eval_adapter(adapter_dir: str) -> dict:
    """Run eval_adapter.sh, parse finetuned/baseline pass rates from output."""
    eval_sh = REPO_DIR / "scripts/eval_adapter.sh"
    logger.info(f"Running eval_adapter.sh with adapter: {adapter_dir}")
    result = subprocess.run(
        ["bash", str(eval_sh), adapter_dir],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    logger.info(f"eval_adapter.sh output:\n{output[-2000:]}")

    # Parse exported KEY=VALUE lines at end of output
    parsed = {}
    for line in output.splitlines():
        m = re.match(r"^(FINETUNED_PASS|FINETUNED_TOTAL|BASELINE_PASS|BASELINE_TOTAL)=(.+)$", line.strip())
        if m:
            parsed[m.group(1)] = m.group(2)

    finetuned_pass = int(parsed.get("FINETUNED_PASS", 0)) if parsed.get("FINETUNED_PASS", "N/A").isdigit() else 0
    finetuned_total = int(parsed.get("FINETUNED_TOTAL", 8))
    baseline_pass_raw = parsed.get("BASELINE_PASS", "N/A")
    baseline_pass = int(baseline_pass_raw) if baseline_pass_raw.isdigit() else None
    baseline_total = int(parsed.get("BASELINE_TOTAL", 8))

    return {
        "exit_code": result.returncode,
        "finetuned_pass": finetuned_pass,
        "finetuned_total": finetuned_total,
        "baseline_pass": baseline_pass,
        "baseline_total": baseline_total,
        "output": output[-2000:],
    }


# ---------------------------------------------------------------------------
# Promote adapter
# ---------------------------------------------------------------------------
def promote_adapter(adapter_output_dir: Path, adapter_active_dir: Path):
    """Copy adapter from output dir to active dir."""
    logger.info(f"Promoting adapter: {adapter_output_dir} → {adapter_active_dir}")
    if adapter_active_dir.exists():
        shutil.rmtree(str(adapter_active_dir))
    shutil.copytree(str(adapter_output_dir), str(adapter_active_dir))
    logger.info("Adapter promoted.")


def update_config_adapter_path(new_path: str):
    """Update model.adapter_path in config.yaml."""
    try:
        cfg_text = CONFIG_PATH.read_text()
        # Replace or insert model.adapter_path
        if re.search(r"^\s*adapter_path\s*:", cfg_text, re.MULTILINE):
            cfg_text = re.sub(
                r"^(\s*adapter_path\s*:)\s*.*$",
                rf"\1 {new_path}",
                cfg_text,
                flags=re.MULTILINE,
            )
        else:
            # Insert after model: block
            cfg_text = re.sub(
                r"^(model\s*:\s*\n)",
                rf"\1  adapter_path: {new_path}\n",
                cfg_text,
                flags=re.MULTILINE,
            )
        CONFIG_PATH.write_text(cfg_text)
        logger.info(f"config.yaml adapter_path set to: {new_path}")
    except Exception as e:
        logger.error(f"Failed to update config.yaml: {e}")


def restart_kernel_evolving():
    """Restart kernel-evolving to load new adapter."""
    start_sh = REPO_DIR / "start.sh"
    if not start_sh.exists():
        logger.warning("start.sh not found — cannot restart kernel-evolving")
        return
    # Kill existing api + model_server
    for pat in ["uvicorn api:app.*8779", "repositories/kernel-evolving/src/model_server.py"]:
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)
    subprocess.Popen(
        ["bash", str(start_sh)],
        cwd=str(REPO_DIR),
        stdout=open("/tmp/kernel_evolving_start.log", "a"),
        stderr=subprocess.STDOUT,
    )
    logger.info("kernel-evolving restart triggered (check /tmp/kernel_evolving_start.log)")


# ---------------------------------------------------------------------------
# Single check cycle
# ---------------------------------------------------------------------------
def run_check():
    cfg = load_config()
    gc = gate_cfg(cfg)

    if not gc.get("enabled", False):
        logger.info("finetune_gate.enabled=false — skipping check")
        return

    threshold = int(gc.get("trajectory_threshold", 50))
    min_improvement = int(gc.get("min_improvement", 0))
    adapter_output_dir = expand(gc.get("adapter_output_dir", "~/.kernel-evolving/workspace/artifacts/finetune"))
    adapter_active_dir = expand(gc.get("adapter_active_dir", "~/.kernel-evolving/workspace/artifacts/adapter_active"))
    state_file = expand(gc.get("state_file", "~/.kernel-evolving/workspace/data/finetune_gate_state.json"))
    min_score = float(cfg.get("providers", {}).get("trajectory_min_critic_score", 0.7))

    state = load_state(state_file)
    last_ts = state.get("last_finetune_ts")

    # 1. Count new clean trajectories
    new_count = count_new_trajectories(last_ts, min_score)
    logger.info(f"New clean trajectories since last fine-tune: {new_count} (threshold: {threshold})")

    if new_count < threshold:
        logger.info(f"Not enough trajectories ({new_count}/{threshold}) — waiting")
        return

    # 2. Threshold reached
    logger.info(f"Threshold reached ({new_count} >= {threshold}) — starting fine-tune")
    send_telegram(f"🧠 Trajectory threshold reached ({new_count} new). Starting fine-tune...")

    train_exit = run_train()

    now_ts = datetime.now(timezone.utc).isoformat()
    if train_exit != 0:
        logger.error(f"train.sh exited with code {train_exit}")
        send_telegram(f"❌ Fine-tune failed (exit {train_exit}). Check /tmp/finetune_train.log")
        state["last_finetune_ts"] = now_ts
        save_state(state_file, state)
        return

    logger.info("Fine-tune completed. Running eval...")

    # 3. Eval adapter
    eval_result = run_eval_adapter(str(adapter_output_dir))
    finetuned_pass = eval_result["finetuned_pass"]
    finetuned_total = eval_result["finetuned_total"]
    baseline_pass = eval_result["baseline_pass"]
    baseline_total = eval_result["baseline_total"]

    logger.info(f"Eval result: finetuned={finetuned_pass}/{finetuned_total}, baseline={baseline_pass}/{baseline_total}")

    # 4. Check improvement threshold
    if baseline_pass is not None:
        improves = finetuned_pass >= (baseline_pass + min_improvement)
    else:
        # Baseline not available — promote if finetuned passes all
        improves = finetuned_pass == finetuned_total

    if improves:
        logger.info(f"Fine-tune improved! Promoting adapter.")
        promote_adapter(adapter_output_dir, adapter_active_dir)
        update_config_adapter_path(str(adapter_active_dir))
        send_telegram(
            f"✅ Fine-tune promoted. Pass rate: {baseline_pass}/{baseline_total} → {finetuned_pass}/{finetuned_total}"
        )
        restart_kernel_evolving()
    else:
        logger.info(f"Fine-tune did not improve (finetuned={finetuned_pass}, baseline={baseline_pass}). Keeping baseline.")
        send_telegram(
            f"⚠️ Fine-tune did not improve. "
            f"Finetuned: {finetuned_pass}/{finetuned_total}, Baseline: {baseline_pass}/{baseline_total}. Keeping baseline."
        )

    state["last_finetune_ts"] = now_ts
    save_state(state_file, state)


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info(f"Caught signal {signum} — shutting down")
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

INTERVAL_S = 30 * 60  # 30 minutes

if __name__ == "__main__":
    logger.info(f"auto_finetune_gate starting (config: {CONFIG_PATH})")
    if args.once:
        logger.info("--once mode: running single check")
        run_check()
        sys.exit(0)

    while _running:
        try:
            run_check()
        except Exception as e:
            logger.error(f"Unexpected error in run_check: {e}", exc_info=True)
        if not _running:
            break
        logger.info(f"Sleeping {INTERVAL_S}s until next check...")
        # Sleep in 30s increments to allow clean shutdown
        for _ in range(INTERVAL_S // 30):
            if not _running:
                break
            time.sleep(30)

    logger.info("auto_finetune_gate stopped.")
