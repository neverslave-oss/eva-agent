from computer_use.debug import snapshot


def test_debug_snapshot_happy_path():
    snap = snapshot()
    assert snap["available"] is True
    assert "registry" in snap
    assert "policy_files" in snap
