"""Batch CY: hybrid health gate + inactive_signal disclosure."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.dashboard.generator import DashboardGenerator
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_voter import EnsembleVoter, SignalReading


def test_unhealthy_nonneg_ic_soft_floors_live_vix_shape() -> None:
    """Batch DU: live-shape VIX IC 0.059 is weak (<0.08) → hard sleep; strong IC soft-floors."""
    voter = EnsembleVoter.__new__(EnsembleVoter)
    scores = {
        "cross_asset_rv": SimpleNamespace(
            health_score=0.57, status="healthy", ic=0.15
        ),
        # Live residual IC ~0.059 is below ENSEMBLE_UNHEALTHY_MIN_IC → hard sleep
        "vix_term_structure": SimpleNamespace(
            health_score=0.4648, status="unhealthy", ic=0.059
        ),
        # Strong IC unhealthy still soft-floors (DU)
        "google_trends": SimpleNamespace(
            health_score=0.40, status="unhealthy", ic=0.15
        ),
        "alternative_data": SimpleNamespace(
            health_score=0.51, status="degraded", ic=-0.07
        ),
    }
    mock = MagicMock()
    mock.calculate_all_health_scores.return_value = scores
    base = {
        SignalSource.CROSS_ASSET_RV: 0.4,
        SignalSource.VIX_TERM_STRUCTURE: 0.2,
        SignalSource.GOOGLE_TRENDS: 0.2,
        SignalSource.ALTERNATIVE_DATA: 0.2,
    }
    with patch(
        "src.signals.health_tracker.SignalHealthTracker", return_value=mock
    ):
        out = voter._apply_health_weights(base)
    assert out[SignalSource.VIX_TERM_STRUCTURE] == 0.0
    assert "vix_term_structure" in voter._health_gate_slept
    assert out[SignalSource.GOOGLE_TRENDS] > 0.0
    assert "google_trends" not in voter._health_gate_slept
    assert out[SignalSource.ALTERNATIVE_DATA] == 0.0
    assert "alternative_data" in voter._health_gate_slept


def test_inactive_signal_status_from_breakdown() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {
                "source": "international_momentum",
                "weight": 0.0,
                "is_active": False,
                "inactive_explanation": "Intl Momentum: neutral, conf=low",
            },
            {"source": "cross_asset_rv", "weight": 0.5, "is_active": True},
        ],
    )
    by = {r["source"]: r for r in statuses}
    intl = by["international_momentum"]
    assert intl["status"] == "inactive_signal"
    assert "inactive" in intl["reason"].lower() or "not actionable" in intl["reason"].lower()
    assert intl["status"] != "zero_weight"


def test_breakdown_includes_is_active() -> None:
    votes = [
        SignalReading(
            source=SignalSource.INTERNATIONAL_MOMENTUM,
            timestamp="t",
            value=0.0,
            confidence=0.0,
            weight=0.0,
            regime_fit="all",
            explanation="Intl Momentum: neutral",
            is_active=False,
        )
    ]
    rows = DashboardGenerator._build_ensemble_source_breakdown(votes)
    assert rows[0]["is_active"] is False
    assert "neutral" in rows[0].get("inactive_explanation", "")
