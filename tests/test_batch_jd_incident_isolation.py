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


def test_ti1_d_incidents_isolate_is_session_scoped(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Autouse isolation must not mktemp-per-test (watchdog inode storm).

    One basetemp child named incidents-isolate* is enough for the whole session.
    """
    conf = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "def _incidents_isolate_root" in conf
    assert 'scope="session"' in conf or "scope='session'" in conf
    assert "_clear_incidents_isolate_root" in conf
    # Executable call must live only on the session fixture (docstrings may mention
    # the historical mktemp-per-test bug).
    assert "return tmp_path_factory.mktemp(\"incidents-isolate\")" in conf
    assert "root = tmp_path_factory.mktemp(\"incidents-isolate\")" not in conf

    basetemp = tmp_path_factory.getbasetemp()
    # pytest also creates an ``incidents-isolatecurrent`` symlink next to the real dir.
    isolate_dirs = [
        p
        for p in basetemp.iterdir()
        if p.is_dir()
        and not p.is_symlink()
        and p.name.startswith("incidents-isolate")
        and not p.name.endswith("current")
    ]
    assert len(isolate_dirs) == 1, (
        f"expected one session incidents-isolate dir, found {len(isolate_dirs)} under {basetemp}"
    )


def test_ti1_e_session_root_clears_between_tests(
    _incidents_isolate_root: Path,
) -> None:
    """Reused session root must not leak open_count across tests."""
    import src.monitor.alerting as alerting

    mgr = alerting.get_incident_manager()
    # Autouse fixture already cleared root and installed hermetic manager.
    assert mgr.summary_path.parent == _incidents_isolate_root
    assert not (mgr.summary_path.exists() and mgr.summary_path.stat().st_size > 2) or (
        json.loads(mgr.summary_path.read_text(encoding="utf-8")).get("open_count", 0) == 0
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="session-root probe",
    )
    assert opened is not None
    summary = json.loads(mgr.summary_path.read_text(encoding="utf-8"))
    assert summary.get("open_count") == 1


def test_ti1_f_session_root_was_cleared_after_prior_probe(
    _incidents_isolate_root: Path,
) -> None:
    """Runs after ti1_e in same process — open_count must start at 0 again."""
    import src.monitor.alerting as alerting

    mgr = alerting.get_incident_manager()
    assert mgr.summary_path.parent == _incidents_isolate_root
    if mgr.summary_path.is_file():
        body = json.loads(mgr.summary_path.read_text(encoding="utf-8"))
        assert body.get("open_count", 0) == 0
    else:
        assert not mgr.summary_path.exists()


def test_ti1_g_pytest_tmp_retention_is_tight() -> None:
    """pyproject must not keep multi-run basetemp trees on lab hosts."""
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'tmp_path_retention_policy = "failed"' in text
    assert "tmp_path_retention_count = 1" in text
