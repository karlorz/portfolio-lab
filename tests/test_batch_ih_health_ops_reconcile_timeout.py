"""Batch IH/II: health_ops kill/open disk SSOT, dashboard timeout, alt freshness.

Session A residuals (II):
- DE4: kill_switch.json can arm after health stamp; reconcile/publish must show
  checks.kill_switch.enabled=true without waiting for next full health rebuild
  from a stale clear report.
- DF3: dashboard wall ~116s under load → configured timeout ≥180 (triple-sync).
- DG3: AlternativeDataComposite.data_freshness_hours must not be hardcoded 12.0.

Authority: never touches target_allocations / order_router.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def test_reconcile_arms_kill_when_disk_enabled_after_clear_stamp(
    tmp_path, monkeypatch
) -> None:
    """DE4: sticky clear monitor health must pick up mid-cycle kill arm."""
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    data.mkdir()
    # Disk SSOT: kill just armed after last health stamp
    (data / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "warning",
                "reason": "unresolved_incident:signal_staleness",
                "source": "incident_lifecycle",
                "message": "stale alternative_data",
                "timestamp": "2026-07-23T12:42:00+00:00",
                "incident_id": "inc-de4",
                "mode": "paper",
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": "inc-de4",
                        "channel": "signal_staleness",
                        "severity": "p2",
                        "state": "firing",
                        "message": "stale alternative_data",
                        "kill_switch_level": "warning",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Sticky clear stamp from prior health cycle
    (data / "health.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "timestamp": "2026-07-23T12:30:00+00:00",
                "scope": "operational_readiness",
                "service": "portfolio-lab",
                "checks": {
                    "circuit_breaker": {"status": "ok", "state": "closed"},
                    "kill_switch": {
                        "status": "ok",
                        "enabled": False,
                        "level": None,
                    },
                    "open_incidents": {
                        "status": "ok",
                        "open_count": 0,
                        "incidents": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(hc, "DATA_DIR", data, raising=False)

    wrote = hc.reconcile_monitor_health_with_disk_ssot(data_dir=data)
    assert wrote is True
    on_disk = json.loads((data / "health.json").read_text(encoding="utf-8"))
    kill = on_disk["checks"]["kill_switch"]
    open_inc = on_disk["checks"]["open_incidents"]
    assert kill.get("enabled") is True
    assert kill.get("level") == "warning"
    assert open_inc.get("open_count") == 1
    assert on_disk.get("ssot_reconcile_source") == "disk_incidents_kill"
    assert on_disk["status"] in {"warning", "critical", "degraded"}


def test_run_health_check_rereads_disk_kill_before_persist(
    tmp_path, monkeypatch
) -> None:
    """DE4 end-of-run: mid-check kill arm appears on persisted report."""
    from src.monitor import health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    # Minimal freshness files so health job does not hard-fail
    (public / "prices.json").write_text("{}", encoding="utf-8")
    (public / "signals.json").write_text(
        json.dumps(
            {
                "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(hc, "DATA_DIR", data, raising=False)
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json", raising=False)

    # First read sees clear; before persist, arm kill on disk
    real_check_kill = hc._check_kill_switch
    calls = {"n": 0}

    def _flip_then_read(data_dir=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_check_kill(data_dir=data_dir)
        # After first snapshot, arm kill as if incident lifecycle wrote mid-run
        (data / "kill_switch.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "level": "warning",
                    "reason": "mid_run_arm",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        (data / "incidents.json").write_text(
            json.dumps(
                {
                    "open_count": 1,
                    "incidents": [
                        {
                            "incident_id": "mid",
                            "state": "firing",
                            "channel": "signal_staleness",
                            "kill_switch_level": "warning",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return real_check_kill(data_dir=data_dir)

    monkeypatch.setattr(hc, "_check_kill_switch", _flip_then_read)
    # Avoid heavy dual-write side effects
    monkeypatch.setattr(hc, "publish_ops_health_surfaces", lambda report: None)
    monkeypatch.setattr(hc, "publish_health_alerts_json", lambda report: None)
    monkeypatch.setattr(
        hc, "update_graduation_circuit_breaker_state", lambda **k: None
    )
    monkeypatch.setattr(
        hc,
        "_check_data_freshness",
        lambda: {
            "prices": {"status": "ok", "age_hours": 0.1},
            "signals": {"status": "ok", "age_hours": 0.1},
            "cron": {"status": "ok", "total_jobs": 0, "failed_jobs": 0},
        },
    )
    monkeypatch.setattr(
        hc,
        "_check_circuit_breaker",
        lambda: {"status": "ok", "state": "closed", "fail_count": 0},
    )

    report = hc.run_health_check()
    assert report["checks"]["kill_switch"]["enabled"] is True
    on_disk = json.loads((data / "health.json").read_text(encoding="utf-8"))
    assert on_disk["checks"]["kill_switch"]["enabled"] is True


def test_dashboard_timeout_budget_at_least_180() -> None:
    """DF3 + IU DT2: Makefile + tasker + cron_compat + shell guard ≥180s."""
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    # dashboard target: timeout N ... generator
    m = re.search(
        r"timeout\s+(\d+)\s+\$\(PYTHON_RUNTIME\)\s+-m\s+src\.dashboard\.generator",
        makefile,
    )
    assert m is not None, "dashboard timeout line missing from Makefile"
    assert int(m.group(1)) >= 180, f"Makefile dashboard timeout {m.group(1)} < 180"

    tasker = (root / "config" / "tasker.yaml").read_text(encoding="utf-8")
    # portfolio-lab-dashboard block timeout_seconds
    block = re.search(
        r"id:\s*portfolio-lab-dashboard.*?timeout_seconds:\s*(\d+)",
        tasker,
        re.S,
    )
    assert block is not None
    assert int(block.group(1)) >= 180, f"tasker timeout {block.group(1)} < 180"

    cron_compat = (root / "src" / "cron_compat.py").read_text(encoding="utf-8")
    m2 = re.search(
        r'"portfolio-lab-dashboard"\s*:\s*(\d+)',
        cron_compat,
    )
    assert m2 is not None
    assert int(m2.group(1)) >= 180, f"cron_compat timeout {m2.group(1)} < 180"

    # Batch IU DT2: shell cron path must match (was stuck at 120 after DF3)
    shell = (root / "scripts" / "cron" / "portfolio-lab-dashboard.sh").read_text(
        encoding="utf-8"
    )
    m3 = re.search(r'cron_guard_start\s+"pf-dashboard"\s+(\d+)', shell)
    assert m3 is not None, "cron_guard_start pf-dashboard missing from shell"
    assert int(m3.group(1)) >= 180, f"shell dashboard guard {m3.group(1)} < 180"


def test_alt_data_freshness_hours_not_hardcoded_12(
    tmp_path, monkeypatch
) -> None:
    """DG3: data_freshness_hours tracks producer/input age, not constant 12.0."""
    from src.signals.alternative_data_signal import (
        AlternativeDataSignalGenerator,
        ComponentSignal,
    )

    gen = AlternativeDataSignalGenerator()
    components = [
        ComponentSignal("treasury_curve", 0.1, 0.5, {}),
        ComponentSignal("sector_rotation", 0.0, 0.5, {}),
    ]

    now = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)
    # Age ~2h
    older = now - timedelta(hours=2)
    monkeypatch.setattr(
        "src.signals.alternative_data_signal._utc_now_iso",
        lambda: now.isoformat(),
    )
    # Inject prices mtime via helper override
    monkeypatch.setattr(
        gen,
        "_input_data_freshness_hours",
        lambda components=None, now_ts=None: 2.0,
    )
    c1 = gen.calculate_composite(components)
    assert c1.data_freshness_hours == pytest.approx(2.0)
    assert c1.data_freshness_hours != 12.0

    monkeypatch.setattr(
        gen,
        "_input_data_freshness_hours",
        lambda components=None, now_ts=None: 0.5,
    )
    c2 = gen.calculate_composite(components)
    assert c2.data_freshness_hours == pytest.approx(0.5)
    assert c1.data_freshness_hours != c2.data_freshness_hours


def test_alt_data_freshness_from_prices_mtime(tmp_path, monkeypatch) -> None:
    """DG3: default freshness uses prices.json mtime age in hours."""
    import os
    import time

    from src.signals import alternative_data_signal as ads

    prices = tmp_path / "prices.json"
    prices.write_text("{}", encoding="utf-8")
    # Set mtime ~3 hours ago
    age_sec = 3 * 3600
    past = time.time() - age_sec
    os.utime(prices, (past, past))
    monkeypatch.setattr(ads, "PRICES_PATH", prices)

    gen = ads.AlternativeDataSignalGenerator()
    hours = gen._input_data_freshness_hours()
    assert 2.5 <= hours <= 3.5
    assert hours != 12.0
