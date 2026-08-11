"""Batch EK: health_ops shares mirror lag + timeline SLIs with compact health.

Deep-research: dual surfaces (ops report vs compact dashboard) must read the
same freshness metrics — no split-brain.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.health_check import attach_shared_freshness_slis_to_ops_report


def test_ops_report_gets_mirror_and_timeline_from_disk(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    (data / "rebalance_health.json").write_text(
        json.dumps(
            {
                "canonical_execution_days": 4,
                "total_executions": 4,
                "raw_history_entries": 96,
                "snapshot_rewrite_files": 55,
                "execution_timeline_policy": (
                    "canonical_event_day; raw rewrites forensic only"
                ),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.monitor.health_check.DATA_DIR", data, raising=False
    )
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    def _fake_lag(**kwargs):
        return {
            "lagging_count": 11,
            "total": 33,
            "lagging_paths": ["signals.json", "health.json"],
            "source": "/var/www/x",
            "dest": str(public),
            "ok": True,
        }

    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(mlag, "summarize_repo_public_mirror_lag", _fake_lag)

    report = attach_shared_freshness_slis_to_ops_report(
        {"status": "ok", "service": "portfolio-lab"},
        data_dir=data,
    )

    assert report["repo_public_mirror_lagging_count"] == 11
    assert report["repo_public_mirror_lag_status"] == "critical"
    assert report["repo_public_mirror_lag"]["lagging_count"] == 11
    assert report["rebalance_execution_timeline_status"] == "rewrite_inflated"
    assert report["rebalance_unique_execution_days"] == 4
    assert report["rebalance_raw_history_entries"] == 96
    assert report["rebalance_execution_timeline"]["source"] == "disk"
    assert "unique=4" in (report.get("rebalance_execution_timeline_badge") or "")


def test_ops_report_timeline_unknown_without_panel(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 0,
            "total": 10,
            "lagging_paths": [],
            "source": "s",
            "dest": "d",
            "ok": True,
        },
    )

    report = attach_shared_freshness_slis_to_ops_report(
        {"status": "ok"}, data_dir=data
    )
    assert report["repo_public_mirror_lag_status"] == "ok"
    assert report["rebalance_execution_timeline_status"] == "unknown"
    assert report["rebalance_execution_timeline"]["source"] == "missing"


def test_ops_report_timeline_ok_when_rewrites_within_retention(
    tmp_path, monkeypatch
) -> None:
    """G6 re-policy: bounded forensic retention must not flag rewrite_inflated.

    Live pre-fix state: raw=116 rewrites=73 vs 5 canonical days flagged the
    intended daily-snapshot retention forever. With the producer cap (14
    days) the SLI flags only when rewrite files exceed 2× the window.
    """
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    (data / "rebalance_health.json").write_text(
        json.dumps(
            {
                "canonical_execution_days": 5,
                "total_executions": 5,
                "raw_history_entries": 51,
                "snapshot_rewrite_files": 12,
                "execution_timeline_policy": (
                    "canonical_event_day; raw rewrites forensic only; "
                    "daily snapshot retention 14 days"
                ),
                "generated": "2026-08-11T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.monitor.health_check.DATA_DIR", data, raising=False
    )
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        lambda **kwargs: {
            "lagging_count": 0,
            "total": 33,
            "lagging_paths": [],
            "ok": True,
        },
    )

    report = attach_shared_freshness_slis_to_ops_report(
        {"status": "ok", "service": "portfolio-lab"},
        data_dir=data,
    )

    assert report["rebalance_execution_timeline_status"] == "ok"
    assert report["rebalance_snapshot_rewrite_files"] == 12
    assert report["rebalance_raw_history_entries"] == 51
    assert "unique=5" in (report.get("rebalance_execution_timeline_badge") or "")
