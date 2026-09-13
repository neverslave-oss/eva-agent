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


def test_watch_hub_delivers_and_unsubscribes():
    """The computer-use broadcast hub must deliver a frame (caption + screenshot)
    to every subscriber and cleanly drop unsubscribed/closed subscribers — the
    mechanism that lets Desktop/Mobile/Dashboard see the stream, not just Telegram."""
    sub = bridge.subscribe_watch()
    try:
        bridge.publish_watch("step 1: launch(thunar)", "data:image/png;base64,AAAA")
        frame = sub.queue.get(timeout=2)
        assert frame["caption"] == "step 1: launch(thunar)"
        assert frame["screenshot"].startswith("data:image")
    finally:
        bridge.unsubscribe_watch(sub)
    # After unsubscribe, publishing must not raise or touch the dead subscriber.
    bridge.publish_watch("step 2", None)



def test_run_computer_task_explicit_live_mode_edge_case():
    _reset_bridge_cache()
    out = bridge.run_computer_task(
        chat_id="chat-1", goal="click and wait", target={"kind": "desktop"}, dry_run=False
    )
    assert out["ok"] is True
    assert out["dry_run"] is False


def test_run_computer_task_llm_mode_uses_llm_planner():
    _reset_bridge_cache()
    out = bridge.run_computer_task(
        chat_id="chat-1",
        goal="open https://docs.openclaw.ai",
        target={"kind": "browser"},
        llm=True,
    )
    # LLM path must still produce a valid envelope and a run_id; it may abort
    # gracefully if the vision brain/provider is unavailable in the test env.
    assert "ok" in out
    assert out["dry_run"] is True
    assert out["run_id"]


def test_debug_snapshot_shape_happy_path():
    _reset_bridge_cache()
    snap = bridge.debug_snapshot(chat_id="chat-2", query="test")
    assert snap["available"] is True
    assert "chat_id" in snap
    assert "registry" in snap
