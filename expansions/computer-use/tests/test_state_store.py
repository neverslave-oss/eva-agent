from computer_use.state_store import StateStore


def test_state_store_per_chat_run_roundtrip_happy_path(tmp_path):
    db = tmp_path / "state.json"
    s = StateStore(db)
    s.update_run("telegram:123", "run-1", {"step": 2, "status": "ok"})

    s2 = StateStore(db)
    got = s2.get_run("telegram:123", "run-1")
    assert got["step"] == 2
    assert got["status"] == "ok"

    chat = s2.get_chat("telegram:123")
    assert chat["last_run_id"] == "run-1"
    assert chat["cursor"]["step"] == 2


def test_state_store_chat_isolation_edge_case(tmp_path):
    s = StateStore(tmp_path / "state.json")
    s.update_run("chat-a", "run-1", {"step": 1})
    s.update_run("chat-b", "run-1", {"step": 9})

    assert s.get_run("chat-a", "run-1")["step"] == 1
    assert s.get_run("chat-b", "run-1")["step"] == 9


def test_state_store_missing_chat_or_run_edge_case(tmp_path):
    s = StateStore(tmp_path / "state.json")
    assert s.get_chat("missing")["runs"] == {}
    assert s.get_run("missing", "run-x") == {}
