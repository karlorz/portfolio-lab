"""Batch HO: dashboard health lag SLI + signals.health restamp honesty.

Residuals after HN:
- Live probe lag can be 1 while private health stamp is 0 (under-report) and
  signals.health compact can stick at lagging=6 after deploy/index churn.
- Public dashboard health.json carries ops_health_status but no
  repo_public_mirror_lag* keys (split-brain vs health_ops / signals.health).
- resolve_mirror_lag_for_consumer (max live,stamp) existed but was not applied
  on dashboard merge or signals nested restamp.

Authority: never touches target_allocations / order_router.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def test_apply_ops_monitor_projects_mirror_lag_onto_dashboard_health(
    tmp_path, monkeypatch
) -> None:
    """Case DD: ops merge stamps repo_public_mirror_lag* on dashboard health."""
    from src.monitor.health_check import apply_ops_monitor_to_dashboard_health

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    # Batch IF: mock live probe. Without it, conftest plab-pytest PUBLIC_DATA_DIR
    # falls back to live WWW vs empty tmp dest → lagging≈36 and clobbers stamp=3.
    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 3,
            "total": 33,
            "lagging_paths": ["decision_registry.json"],
            "source": "/var/www/x",
            "dest": str(public),
            "ok": True,
        },
    )

    ops_report = {
        "status": "warning",
        "timestamp": "2026-07-23T04:00:00+00:00",
        "repo_public_mirror_lagging_count": 3,
        "repo_public_mirror_total": 33,
        "repo_public_mirror_lagging_paths": ["decision_registry.json"],
        "repo_public_mirror_lag_status": "lagging",
        "repo_public_mirror_lag_badge": "lagging=3/33",
        "repo_public_mirror_source": "/var/www/x",
        "repo_public_mirror_dest": str(public),
        "repo_public_mirror_lag": {
            "lagging_count": 3,
            "total": 33,
            "status": "lagging",
            "badge": "lagging=3/33",
            "paths": ["decision_registry.json"],
        },
    }
    dash = {
        "system_status": "healthy",
        "generated_at": "2026-07-23T03:00:00+00:00",
        "cron_jobs": [],
    }
    out = apply_ops_monitor_to_dashboard_health(
        dash, ops_report, data_dir=data, public_dir=public
    )
    assert out["repo_public_mirror_lagging_count"] == 3
    assert out["repo_public_mirror_lag_status"] == "lagging"
    assert out["repo_public_mirror_lag_badge"] == "lagging=3/33"
    assert out["repo_public_mirror_lag"]["lagging_count"] == 3
    # Soft-elevate dashboard system_status when lagging (ops hygiene, not halt)
    assert out["system_status"] == "warning"
    # Honesty meta must agree with projected lag (not sticky pre-merge stamp)
    assert out["mirror_lag_source_of_truth"] in ("live", "stamp")
    assert out["mirror_lag_live_lagging_count"] == 3
    assert out["mirror_lag_stamp_lagging_count"] == 3


def test_apply_ops_monitor_uses_max_live_stamp_when_probe_provided(
    tmp_path, monkeypatch
) -> None:
    """Case DE: under-report defense — stamp 0 / live 4 → consumer max 4."""
    from src.monitor.health_check import apply_ops_monitor_to_dashboard_health

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 4,
            "total": 33,
            "lagging_paths": ["a.json", "b.json"],
            "source": "s",
            "dest": "d",
            "ok": True,
        },
    )

    ops_report = {
        "status": "ok",
        "timestamp": "2026-07-23T04:00:00+00:00",
        # Stale heal stamp on ops report (under-report vs live)
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 33,
        "repo_public_mirror_lag_status": "ok",
        "repo_public_mirror_lag_badge": "lagging=0/33",
    }
    dash = {"system_status": "healthy", "cron_jobs": []}
    out = apply_ops_monitor_to_dashboard_health(
        dash, ops_report, data_dir=data, public_dir=public
    )
    assert out["repo_public_mirror_lagging_count"] == 4
    assert out["repo_public_mirror_lag_status"] == "lagging"
    assert out["mirror_lag_source_of_truth"] == "live"
    assert out["mirror_lag_live_lagging_count"] == 4
    assert out["mirror_lag_stamp_lagging_count"] == 0


def test_restamp_signals_json_nested_health_lag(tmp_path) -> None:
    """Case DF: restamp rewrites sticky signals.health lag from live probe."""
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    signals_path = tmp_path / "signals.json"
    sticky = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {
            "status": "warning",
            "repo_public_mirror_lagging_count": 6,
            "repo_public_mirror_total": 33,
            "repo_public_mirror_lagging_paths": [
                "index.json",
                "source_manifest.json",
            ],
            "repo_public_mirror_lag_status": "lagging",
            "repo_public_mirror_lag_badge": "lagging=6/33",
        },
    }
    signals_path.write_text(json.dumps(sticky, indent=2) + "\n", encoding="utf-8")

    live = {
        "lagging_count": 1,
        "total": 33,
        "lagging_paths": ["decision_registry.json"],
        "source": "/var/www/x",
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(
        paths=[signals_path],
        lag_summary=live,
    )
    assert any("signals" in r for r in result["restamped"])
    out = json.loads(signals_path.read_text(encoding="utf-8"))
    # Authority preserved
    assert out["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    h = out["health"]
    assert h["repo_public_mirror_lagging_count"] == 1
    assert h["repo_public_mirror_lag_status"] == "lagging"
    assert h["repo_public_mirror_lagging_paths"] == ["decision_registry.json"]
    assert "mirror_lag_restamped_at" in h
    # Restamp must not re-darken Caddy (0600); atomic write leaves 0o644
    mode = signals_path.stat().st_mode & 0o777
    assert mode == 0o644 or (mode & 0o004), oct(mode)


def test_restamp_monitor_health_advances_embedded_timestamp(
    tmp_path, monkeypatch
) -> None:
    """NG4 (2026-08-11 session B): mirror-lag restamp rewrites of the
    monitor-schema report must advance the embedded timestamp.

    Live artifact between :00/:30 health runs: the soft-mirror restamp
    rewrote data/health.json (fresh mtime + ssot_reconciled_at) while the
    embedded timestamp stayed at report generation time — mtime-based
    freshness overstated content by up to 30 min. Any SSOT re-projection
    write must advance timestamp; the fix lives at the shared patch seam
    (_patch_monitor_report_kill_open).

    DN3 disk-SSOT projection contract (Batch IN): before write, the restamp
    re-projects kill/open from the disk SSOT via
    _disk_kill_and_open_incidents() (repo_public_mirror_lag.py), so a lag
    restamp cannot freeze sticky kill.enabled while kill_switch.json is
    clear (or the reverse arm lag). That reader defaults to the module-level
    DATA_DIR binding in health_kill_surfaces — NOT monkeypatched by default
    — so this test pins it to an ARMED fixture dir: the projection must
    follow the patched disk SSOT (enabled stays True), never the live
    repo's clear switch.
    """
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    # ARMED disk SSOT fixture: the DN3 re-projection reads kill/open here.
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "warning",
                "reason": "unresolved_incident:ic_decay",
                "source": "incident_lifecycle",
                "message": "fixture arm",
                "timestamp": "2026-08-11T07:00:00+00:00",
                "incident_id": "8115a9c1-0000-0000-0000-000000000000",
                "mode": "paper",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "incidents.json").write_text(
        json.dumps(
            {
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": "8115a9c1-0000-0000-0000-000000000000",
                        "channel": "ic_decay",
                        "severity": "p0",
                        "state": "firing",
                        "message": "fixture open",
                        "kill_switch_level": "warning",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    import src.monitor.health_kill_surfaces as hks

    monkeypatch.setattr(hks, "DATA_DIR", tmp_path, raising=False)

    health_path = tmp_path / "health.json"
    old_ts = "2026-08-11T07:00:05+00:00"
    health_path.write_text(
        json.dumps(
            {
                "status": "critical",
                "timestamp": old_ts,
                # SLI keys present as in production monitor health.json
                # (stamped by earlier mirror-lag restamps).
                "repo_public_mirror_lagging_count": 0,
                "repo_public_mirror_total": 33,
                "repo_public_mirror_lag_status": "ok",
                "checks": {
                    "kill_switch": {
                        "status": "critical",
                        "enabled": True,
                        "level": "halt",
                        "reason": "unresolved_incident:ic_decay",
                    },
                    "open_incidents": {
                        "status": "critical",
                        "open_count": 1,
                        "incident_id": "8115a9c1-0000-0000-0000-000000000000",
                    },
                    "data_freshness": {"signals": {"status": "fresh"}},
                },
            }
        ),
        encoding="utf-8",
    )

    live = {
        "lagging_count": 0,
        "total": 33,
        "lagging_paths": [],
        "source": "/var/www/x",
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(
        paths=[health_path],
        lag_summary=live,
    )
    assert any("health" in r for r in result["restamped"])
    out = json.loads(health_path.read_text(encoding="utf-8"))
    new_ts = out.get("timestamp")
    assert new_ts and new_ts != old_ts
    new_dt = datetime.fromisoformat(new_ts)
    assert abs((datetime.now(timezone.utc) - new_dt).total_seconds()) < 300
    # Kill/open projection + SSOT disclosure stamps still present.
    assert out["checks"]["kill_switch"]["enabled"] is True
    assert out.get("ssot_reconciled_at")
