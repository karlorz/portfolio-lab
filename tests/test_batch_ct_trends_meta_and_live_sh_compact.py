"""Batch CT: trends _meta ignored by signal; signals compact rebuilds live SH."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.dashboard.generator import _compact_health_summary
from src.signals.google_trends_signal import GoogleTrendsSignal


def _recent_series(days: int = 30) -> dict[str, int]:
    today = datetime.now().date()
    out: dict[str, int] = {}
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        out[d.isoformat()] = 40 + (i % 5)
    return out


def test_google_trends_signal_skips_meta_keys(tmp_path: Path) -> None:
    series = _recent_series(30)
    payload = {
        "_meta": {
            "schema": "google-trends-cache/v1",
            "fetched_at": datetime.now().isoformat(),
            "latest_observation": max(series),
        },
        "recession": series,
        "inflation": series,
        "stock market crash": series,
        "interest rates": series,
    }
    path = tmp_path / "google_trends.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    snap = GoogleTrendsSignal(data_path=str(path)).get_signal_snapshot()
    assert snap.is_active is True
    assert snap.source == "google_trends"
    assert (snap.metadata or {}).get("data_age_days", 99) <= 14


def test_compact_prefers_live_quality_disclosure_over_stale_summary() -> None:
    """Stale summary freeze@46d corrected by quality_disclosure block."""
    report = {
        "system_status": "degraded",
        "signal_health": {
            "status": "degraded",
            "overall_health": "degraded",
            "summary": {
                "healthy": 3,
                "degraded": 5,
                "unhealthy": 1,
                "total_tracked": 9,
                "quality_badge": "3/9 healthy sources",
                # sticky pre-CQ fields
                "zero_healthy_sources": False,
                "ensemble_weight_freeze_active": True,
                "ensemble_weights_age_days": 46.23,
                "ensemble_weights_file_stale": True,
            },
            "quality_disclosure": {
                "badge": "3/9 healthy sources",
                "ensemble_weight_freeze": {
                    "weight_freeze_active": False,
                    "weight_file_stale": False,
                    "ensemble_weights_age_days": 0.01,
                },
            },
        },
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
    }
    compact = _compact_health_summary(report)
    assert compact.get("ensemble_weight_freeze_active") is False
    assert compact.get("ensemble_weights_file_stale") is False
    assert compact.get("ensemble_weights_age_days") == 0.01


def test_generate_signals_rebuilds_signal_health_for_compact() -> None:
    """Source contract: generate_signals_json rebuilds SH before compact."""
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "Batch CT: canonical WWW health.json may embed" in src
    assert 'health_report["signal_health"] = build_signal_health_section' in src
