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
        tracker_cls.return_value.get_health_report.return_value = mock_report
        out = build_signal_health_section()
    assert out["overall_health"] == "degraded"
    assert out["status"] == "degraded"
    assert out["label_horizon"] == "SPY actual direction resolved by prediction date"
    assert out["scores"]["msm"] == 0.55


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
