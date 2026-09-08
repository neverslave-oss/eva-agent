from pathlib import Path
import sys

KERNEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(KERNEL_ROOT / "src"))

from core.expansions import computer_use_bridge as bridge  # noqa: E402


def _reset_bridge_cache():
    bridge._sidecar = None


def test_bridge_available_happy_path():
    _reset_bridge_cache()
    assert bridge.available() is True


def test_inject_context_empty_text_edge_case():
    _reset_bridge_cache()
    assert bridge.inject_computer_use_context(chat_id="c1", text="") == ""


def test_run_computer_task_happy_path():
    _reset_bridge_cache()
    out = bridge.run_computer_task(chat_id="chat-1", goal="open https://docs.openclaw.ai", target={"kind": "browser"})
    assert out["ok"] is True
    assert out["status"] in {"ok", "done"}
    assert out["dry_run"] is True
    assert out["run_id"]


def test_run_computer_task_explicit_live_mode_edge_case():
    _reset_bridge_cache()
    out = bridge.run_computer_task(
        chat_id="chat-1", goal="click and wait", target={"kind": "desktop"}, dry_run=False
    )
    assert out["ok"] is True
    assert out["dry_run"] is False


def test_debug_snapshot_shape_happy_path():
    _reset_bridge_cache()
    snap = bridge.debug_snapshot(chat_id="chat-2", query="test")
    assert snap["available"] is True
    assert "chat_id" in snap
    assert "registry" in snap
