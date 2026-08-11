#!/usr/bin/env python3
"""Tests for scripts/cron_update.py — G9 status vocabulary + UTC timestamps.

The make chain stamps Makefile STATUS values (ok/oom/timeout/error) with
naive-local timestamps into cron_status.json; the tasker mirror writes
tasker vocabulary (success/blocked) with UTC-aware ISO shortly after. This
test pins the alignment at the cron_update write boundary (vocabulary map
+ timezone-aware now) so the file no longer flip-flops vocab/tz between
the two writers (G9, 2026-08-11).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cron_update.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cron_update", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("make_status", "expected"),
    [
        ("ok", "success"),
        ("oom", "error"),
        ("timeout", "timeout"),
        ("error", "error"),
    ],
)
def test_status_vocabulary_mapping(tmp_path, monkeypatch, make_status, expected):
    mod = _load_module()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["cron_update.py", "portfolio-lab-daily-pnl", make_status, "5", "tasker"],
    )
    mod.main()
    data = json.loads((tmp_path / "data" / "cron_status.json").read_text())
    row = next(j for j in data["jobs"] if j["name"] == "portfolio-lab-daily-pnl")
    assert row["status"] == expected
    assert row["last_run"].endswith("+00:00"), row["last_run"]


def test_mark_to_market_row_still_appended(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["cron_update.py", "portfolio-lab-mark-to-market", "ok", "1", "tasker"],
    )
    mod.main()
    data = json.loads((tmp_path / "data" / "cron_status.json").read_text())
    names = [j["name"] for j in data["jobs"]]
    assert "portfolio-lab-mark-to-market" in names  # roster-append unchanged (A4)
    row = next(j for j in data["jobs"] if j["name"] == "portfolio-lab-mark-to-market")
    assert row["status"] == "success"
