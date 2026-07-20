"""Tests for signal_health_section builders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.dashboard.signal_health_section import (
    build_fred_readiness_section,
    build_signal_health_section,
    fred_readiness_unavailable_payload,
    signal_health_unavailable_payload,
)


def test_signal_health_unavailable_payload_shape() -> None:
    payload = signal_health_unavailable_payload(RuntimeError("boom"))
    assert payload["status"] == "unavailable"
    assert "boom" in payload["error"]


def test_fred_readiness_unavailable_payload_shape() -> None:
    payload = fred_readiness_unavailable_payload(ImportError("fredapi"))
    assert payload["ready"] is True
    assert payload["blocking"] is False
    assert "fredapi" in payload["message"]


def test_build_signal_health_section_success() -> None:
    mock_report = {
        "timestamp": "2026-07-01T12:00:00Z",
        "summary": {"ok": 3},
        "scores": {"msm": 0.55},
        "alerts": [],
        "overall_health": "degraded",
        "status": "degraded",
        "label_horizon": "SPY actual direction resolved by prediction date",
    }
    with patch("src.signals.health_tracker.SignalHealthTracker") as tracker_cls:
        tracker = tracker_cls.return_value
        tracker.get_health_report.return_value = mock_report
        tracker.resolve_pending_labels.return_value = {
            "dates_considered": 2,
            "dates_resolved": 1,
            "predictions_updated": 5,
            "skipped_no_spy_return": [],
            "max_days": 30,
        }
        out = build_signal_health_section()
    # Newest-first + oldest-first catch-up batches
    assert tracker.resolve_pending_labels.call_count == 2
    assert out["overall_health"] == "degraded"
    assert out["status"] == "degraded"
    assert out["label_horizon"] == "SPY actual direction resolved by prediction date"
    assert out["scores"]["msm"] == 0.55
    # Sum of both resolve batches (mock returns 5 each)
    assert out["label_resolve"]["predictions_updated"] == 10
    assert "newest_first" in out["label_resolve"]
    assert "oldest_first" in out["label_resolve"]


def test_build_signal_health_section_can_skip_resolve() -> None:
    mock_report = {
        "timestamp": "2026-07-01T12:00:00Z",
        "summary": {},
        "scores": {},
        "alerts": [],
        "overall_health": "unknown",
        "status": "unknown",
        "label_horizon": "x",
    }
    with patch("src.signals.health_tracker.SignalHealthTracker") as tracker_cls:
        tracker = tracker_cls.return_value
        tracker.get_health_report.return_value = mock_report
        out = build_signal_health_section(resolve_labels=False)
    tracker.resolve_pending_labels.assert_not_called()
    assert "label_resolve" not in out


def test_build_signal_health_section_failure() -> None:
    with patch("src.signals.health_tracker.SignalHealthTracker", side_effect=ImportError("no module")):
        out = build_signal_health_section()
    assert out["status"] == "unavailable"


def test_build_fred_readiness_section_delegates() -> None:
    readiness = {"status": "ok", "ready": True, "blocking": False}
    with patch("src.data.fred_data.get_fred_md_cache_health", return_value={"status": "ok"}):
        with patch("src.monitor.fred_readiness.assess_fred_readiness", return_value=readiness) as assess:
            out = build_fred_readiness_section()
    assess.assert_called_once()
    assert out["status"] == "ok"
