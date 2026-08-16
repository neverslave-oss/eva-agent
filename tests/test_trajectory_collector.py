"""tests/test_trajectory_collector.py — ADR-013 trajectory collector tests."""
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_collector(tmp_path, min_score=0.7, require_artifact=True, enabled=True):
    """Create a TrajectoryCollector with a temp DB."""
    import core.evolution.trajectory_collector as tc
    # Patch DB path to temp dir
    tc._DB_PATH = str(tmp_path / "evolution.db")
    tc._collector = None  # reset singleton
    cfg = {
        "providers": {
            "collect_trajectories": enabled,
            "trajectory_min_critic_score": min_score,
            "trajectory_require_artifact": require_artifact,
        }
    }
    return tc.TrajectoryCollector(cfg)


def test_trajectory_record(tmp_path):
    col = _make_collector(tmp_path)
    col.record(
        task="write a hello world script",
        provider="openai",
        model_name="gpt-5.4",
        call_type="task_inference",
        tool_calls=[{"tool": "write_file", "args": {"path": "/tmp/hello.py"}, "result": "ok"}],
        final_reply="Done.",
        artifacts=["/tmp/hello.py"],
        critic_score=0.9,
        critic_verdict="PASS",
        elapsed_s=1.5,
    )
    with col._conn() as conn:
        rows = conn.execute("SELECT * FROM task_trajectories").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["provider"] == "openai"
    assert row["critic_score"] == 0.9
    assert row["critic_verdict"] == "PASS"
    tool_calls = json.loads(row["tool_calls"])
    assert tool_calls[0]["tool"] == "write_file"


def test_trajectory_should_record_above_threshold(tmp_path):
    col = _make_collector(tmp_path, min_score=0.7, require_artifact=False)
    # tool_calls must be non-empty for collection to trigger
    assert col.should_record(0.8, None, tool_calls=[{"tool": "write_file"}]) is True
    assert col.should_record(0.7, None, tool_calls=[{"tool": "exec_shell"}]) is True


def test_trajectory_should_not_record_below_threshold(tmp_path):
    col = _make_collector(tmp_path, min_score=0.7, require_artifact=False)
    assert col.should_record(0.5, None) is False
    assert col.should_record(0.69, None) is False


def test_trajectory_export_jsonl(tmp_path):
    col = _make_collector(tmp_path, require_artifact=False)
    col.record(
        task="fetch and summarise",
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        call_type="task_inference",
        tool_calls=[],
        final_reply="Summary here.",
        artifacts=[],
        critic_score=0.85,
        critic_verdict="PASS",
    )
    col.record(
        task="low score task",
        provider="local",
        model_name=None,
        call_type="task_inference",
        tool_calls=[],
        final_reply="Failed.",
        artifacts=[],
        critic_score=0.4,
        critic_verdict="FAIL",
    )
    export_path = str(tmp_path / "export.jsonl")
    path, count = col.export_jsonl(path=export_path, min_score=0.7)
    assert count == 1
    assert os.path.exists(path)
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1
    assert lines[0]["provider"] == "anthropic"
    assert lines[0]["critic_score"] == 0.85
    assert isinstance(lines[0]["messages"], list)
