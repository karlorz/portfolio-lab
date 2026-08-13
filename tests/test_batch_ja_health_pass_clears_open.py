"""Batch JA / IZ — DO2: health cadence re-evaluates signal_staleness lifecycle.

Session A plan (JA = surgical IZ):
- run_health_check must call check_staleness_and_alert so PASS clears false
  opens without waiting for full dashboard generate.
- Does not touch signals.json.target_allocations / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

def _intentional_only_staleness() -> dict:
    return {
        "stale_signals": [],
        "unavailable_signals": [
            "two_stage_regime",
            "regime_transition",
            "fred_macro",
        ],
        "unavailable_ownership": [
            {
                "signal": "two_stage_regime",
                "intentional_when_fred_unconfigured": True,
                "intentional_lab_gap": True,
            },
            {
                "signal": "regime_transition",
                "intentional_when_fred_unconfigured": True,
                "intentional_lab_gap": True,
            },
            {
                "signal": "fred_macro",
                "intentional_when_fred_unconfigured": True,
                "intentional_lab_gap": True,
            },
        ],
        "healthy_count": 20,
        "total_count": 23,
    }

def _stub_health_heavy(hc, monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal stubs so run_health_check does not depend on live SSOT I/O."""
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
    monkeypatch.setattr(
        hc,
        "_check_fred_md_cache",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(hc, "publish_ops_health_surfaces", lambda report: None)
    monkeypatch.setattr(hc, "publish_health_alerts_json", lambda report: None)
    monkeypatch.setattr(
        hc,
        "update_graduation_circuit_breaker_state",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        hc,
        "attach_shared_freshness_slis_to_ops_report",
        lambda report, data_dir=None: report,
    )
    monkeypatch.setattr(hc, "_stamp_health_self_job_running_success", lambda freshness: None)

def test_ja_do2_health_clears_open_when_staleness_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case A: open firing + intentional-only staleness → open_count=0 after health."""
    import src.monitor.alerting as alerting
    import src.monitor.health_check as hc
    from src.monitor.incident_manager import IncidentManager

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    mgr = IncidentManager(
        log_path=data / "incidents.jsonl",
        summary_path=data / "incidents.json",
        kill_switch_path=data / "kill_switch.json",
        escalation_cycles=99,  # do not escalate WARN→kill in this unit
        escalation_enabled=False,
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="2/23 signals unavailable: sector_rotation, risk_decomposition",
        details={
            "actionable_unavailable": ["sector_rotation", "risk_decomposition"],
        },
    )
    assert opened is not None
    assert json.loads((data / "incidents.json").read_text(encoding="utf-8")).get(
        "open_count"
    ) == 1

    (data / "signals.json").write_text(
        json.dumps(
            {
                "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                "staleness": _intentional_only_staleness(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(alerting, "_incident_manager", mgr)
    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.DATA_DIR", data, raising=False)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    _stub_health_heavy(hc, monkeypatch)

    # Real open/kill probes so post-PASS re-read reflects disk SSOT
    report = hc.run_health_check()
    assert report["status"] in {"ok", "warning", "degraded", "critical"}

    incidents = json.loads((data / "incidents.json").read_text(encoding="utf-8"))
    assert incidents.get("open_count") == 0, (
        f"DO2 health must PASS-clear false open; got {incidents.get('open_count')}"
    )
    open_check = report["checks"]["open_incidents"]
    assert int(open_check.get("open_count") or 0) == 0

def test_ja_do2_health_invokes_check_staleness_and_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case B: run_health_check calls check_staleness_and_alert at least once."""
    import src.monitor.health_check as hc

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "signals.json").write_text(
        json.dumps(
            {
                "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                "staleness": _intentional_only_staleness(),
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(hc, "DATA_DIR", data)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.DATA_DIR", data, raising=False)
    monkeypatch.setattr(hc, "HEALTH_PATH", data / "health.json")
    monkeypatch.setattr(hc, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_kill_surfaces.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_dashboard_apply.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.health_rollup.PUBLIC_DATA_DIR", public, raising=False)
    _stub_health_heavy(hc, monkeypatch)

    calls: list[dict] = []

    def _spy(st: dict) -> None:
        calls.append(dict(st) if isinstance(st, dict) else {"_raw": st})

    monkeypatch.setattr(
        "src.monitor.alerting.check_staleness_and_alert",
        _spy,
    )

    hc.run_health_check()
    assert len(calls) >= 1, "DO2: health must invoke check_staleness_and_alert"
    assert calls[0].get("unavailable_signals") == [
        "two_stage_regime",
        "regime_transition",
        "fred_macro",
    ]
