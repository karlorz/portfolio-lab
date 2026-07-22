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
