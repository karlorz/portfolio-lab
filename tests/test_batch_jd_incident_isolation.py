"""Batch JG TI1 — refuse live incident/kill SSOT writes under pytest.

Session A plan (JG/JE/JF): H16 isolates PUBLIC only; private DATA_DIR remains
live unless IncidentManager is hermetic. Under PYTEST_CURRENT_TEST, default
live paths must not open/resolve/arm kill.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitor.incident_manager import IncidentManager
from src.paths import DATA_DIR


def test_ti1_a_default_live_paths_do_not_mutate_operator_ssot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case A: default manager under pytest must not increase live open_count."""
    live_summary = Path(DATA_DIR) / "incidents.json"
    live_log = Path(DATA_DIR) / "incidents.jsonl"
    live_kill = Path(DATA_DIR) / "kill_switch.json"

    before_open = 0
    before_bytes = b""
    before_log_size = 0
    kill_existed = live_kill.exists()
    if live_summary.is_file():
        before_bytes = live_summary.read_bytes()
        try:
            before_open = int(json.loads(before_bytes.decode("utf-8")).get("open_count") or 0)
        except (json.JSONDecodeError, UnicodeDecodeError):
            before_open = 0
    if live_log.is_file():
        before_log_size = live_log.stat().st_size

    # Ensure we are under pytest (pytest sets this automatically; reaffirm)
    assert os_environ_pytest_active()

    # Default constructor → live DATA_DIR paths
    mgr = IncidentManager()
    assert mgr.summary_path.resolve() == live_summary.resolve()

    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="TI1 pollution probe sector_rotation",
        details={"actionable_unavailable": ["sector_rotation"]},
    )
    # May return an in-memory Incident, but disk must stay immutable
    assert opened is None or opened.channel == "signal_staleness"

    if live_summary.is_file():
        after = live_summary.read_bytes()
        after_open = int(json.loads(after.decode("utf-8")).get("open_count") or 0)
        assert after_open == before_open
        assert after == before_bytes, "live incidents.json body must not change under pytest"
    else:
        assert not live_summary.exists(), "must not create live incidents.json under pytest"

    if live_log.is_file():
        assert live_log.stat().st_size == before_log_size
    else:
        assert not live_log.exists()

    if not kill_existed:
        assert not live_kill.exists(), "must not arm live kill under pytest"


def test_ti1_b_hermetic_tmp_lifecycle_still_works(tmp_path: Path) -> None:
    """Case B: tmp summary/log/kill still allow full open→resolve."""
    mgr = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=True,
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="hermetic open",
        details={"stale_signals": ["ensemble_voting"]},
    )
    assert opened is not None
    summary = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
    assert summary.get("open_count") == 1

    mgr.record_alert(
        channel="signal_staleness",
        level="pass",
        message="hermetic clear",
    )
    summary = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
    assert summary.get("open_count") == 0


@pytest.mark.allow_live_incidents
def test_ti1_c_allow_flag_permits_live_path_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case C: PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS=1 permits write (to tmp live-like)."""
    # Simulate live DATA_DIR at tmp so we never touch operator SSOT even with allow.
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS", "1")
    monkeypatch.setattr(
        "src.monitor.incident_manager.DATA_DIR",
        fake_data,
        raising=False,
    )
    # Rebuild defaults after DATA_DIR patch by constructing with explicit paths
    # that resolve equal to "live" under the patched DATA_DIR.
    summary = fake_data / "incidents.json"
    log = fake_data / "incidents.jsonl"
    kill = fake_data / "kill_switch.json"
    mgr = IncidentManager(
        log_path=log,
        summary_path=summary,
        kill_switch_path=kill,
        escalation_enabled=False,
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="allow-live probe",
    )
    assert opened is not None
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body.get("open_count") == 1


def os_environ_pytest_active() -> bool:
    import os

    return bool(os.environ.get("PYTEST_CURRENT_TEST"))
