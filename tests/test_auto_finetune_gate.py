"""
Unit tests for auto_finetune_gate.in_train_window (night-window deferral).

Pure-function tests: no DB, no HTTP, no filesystem. Import the gate module
directly via a sys.path shim so the argparse/daemon side effects in __main__
never execute.
"""
import os
import sys
from datetime import datetime

import pytest

# The gate module parses argv at import time (module-level argparse + daemon
# setup). Give it a bare argv so import succeeds without side effects when the
# test harness passes its own arguments.
sys.argv = ["auto_finetune_gate.py"]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import auto_finetune_gate as gate  # noqa: E402


def _dt(hh, mm):
    return datetime(2026, 9, 20, hh, mm)


def _gc(start, end):
    return {"train_window_start": start, "train_window_end": end}


def test_inside_window_day():
    gc = _gc("09:00", "17:00")
    assert gate.in_train_window(gc, _dt(10, 0)) is True
    assert gate.in_train_window(gc, _dt(9, 0)) is True
    assert gate.in_train_window(gc, _dt(16, 59)) is True


def test_outside_window_day():
    gc = _gc("09:00", "17:00")
    assert gate.in_train_window(gc, _dt(8, 59)) is False
    assert gate.in_train_window(gc, _dt(17, 0)) is False
    assert gate.in_train_window(gc, _dt(23, 30)) is False


def test_wraparound_midnight_window():
    # 23:00 -> 05:00 crosses midnight
    gc = _gc("23:00", "05:00")
    assert gate.in_train_window(gc, _dt(23, 30)) is True
    assert gate.in_train_window(gc, _dt(0, 30)) is True
    assert gate.in_train_window(gc, _dt(4, 59)) is True
    assert gate.in_train_window(gc, _dt(5, 0)) is False
    assert gate.in_train_window(gc, _dt(12, 0)) is False


def test_missing_settings_default_to_allowed():
    # Absent/malformed window -> always allowed (gate must never silently stop)
    assert gate.in_train_window({}, _dt(12, 0)) is True
    assert gate.in_train_window({"train_window_start": "not-a-time", "train_window_end": "05:00"}, _dt(12, 0)) is True
    assert gate.in_train_window({"train_window_start": "23:00", "train_window_end": "junk"}, _dt(12, 0)) is True


def test_equal_boundaries_allowed():
    # start == end -> treat as open (default 00:00-00:00)
    gc = _gc("00:00", "00:00")
    assert gate.in_train_window(gc, _dt(3, 0)) is True
