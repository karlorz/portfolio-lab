"""Batch CV: inactive snapshots stored for disclosure; vote weight forced 0."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.signals.signal_snapshot import SignalSnapshot
from src.strategy.ensemble_voter import SignalReading, SignalSource
from src.strategy.signal_aggregator import SignalAggregator


def test_store_inactive_snapshot_still_in_readings() -> None:
    readings: dict = {}
    snap = SignalSnapshot(
        source="international_momentum",
        timestamp="2026-07-22T00:00:00",
        value=0.0,
        confidence=0.0,
        is_active=False,
        explanation="Intl Momentum: neutral",
    )
    SignalAggregator._store_active_snapshot(
        readings, SignalSource.INTERNATIONAL_MOMENTUM, snap
    )
    assert SignalSource.INTERNATIONAL_MOMENTUM in readings
    reading = readings[SignalSource.INTERNATIONAL_MOMENTUM]
    assert isinstance(reading, SignalReading)
    assert reading.is_active is False
    assert reading.value == 0.0


def test_apply_weights_forces_zero_for_inactive() -> None:
    from src.strategy.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter.__new__(EnsembleVoter)
    readings = {
        SignalSource.INTERNATIONAL_MOMENTUM: SignalReading(
            source=SignalSource.INTERNATIONAL_MOMENTUM,
            timestamp="t",
            value=0.0,
            confidence=0.0,
            weight=0.0,
            regime_fit="all",
            is_active=False,
        ),
        SignalSource.CROSS_ASSET_RV: SignalReading(
            source=SignalSource.CROSS_ASSET_RV,
            timestamp="t",
            value=0.5,
            confidence=0.7,
            weight=0.0,
            regime_fit="all",
            is_active=True,
        ),
    }
    weights = {
        SignalSource.INTERNATIONAL_MOMENTUM: 0.21,
        SignalSource.CROSS_ASSET_RV: 0.11,
    }
    out = EnsembleVoter._apply_weights_to_readings(voter, readings, weights)
    by = {r.source: r for r in out}
    assert by[SignalSource.INTERNATIONAL_MOMENTUM].weight == 0.0
    assert by[SignalSource.CROSS_ASSET_RV].weight == 0.11


def test_intl_explanation_uses_pp_not_percent_format() -> None:
    from src.signals.international_momentum import InternationalMomentumSignal

    sig = InternationalMomentumSignal(
        timestamp="t",
        signal_type="neutral",
        confidence=0.0,
        confidence_level="low",
        efa_momentum_6m=6.7,
        eem_momentum_6m=0.0,
        spy_momentum_6m=9.7,
        efa_vs_spy=-3.01,
        eem_vs_spy=-9.75,
        spy_shift=0.0,
        efa_shift=0.0,
        eem_shift=0.0,
        max_allocation_efa=0.05,
        max_allocation_eem=0.03,
        holding_period_days=30,
        data_fresh=True,
        vix_filter_active=False,
        correlation_override=False,
        risk_controls_status="evaluated_passed",
        risk_controls_available=True,
        risk_controls_reason="ok",
        vix_level=17.0,
        correlation_efa_spy=0.8,
    )
    snap = sig.to_signal_snapshot()
    assert "pp" in snap.explanation
    assert "301" not in snap.explanation
    assert snap.is_active is False


def test_configured_status_zero_weight_when_collected_inactive() -> None:
    from src.dashboard.generator import DashboardGenerator

    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "international_momentum", "weight": 0.0},
            {"source": "cross_asset_rv", "weight": 0.5},
        ],
    )
    by = {r["source"]: r for r in statuses}
    intl = by["international_momentum"]
    assert intl["collected"] is True
    assert intl["status"] == "zero_weight"
    assert intl["status"] != "missing"
