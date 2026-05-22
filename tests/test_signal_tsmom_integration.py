"""
Tests for src/signals/tsmom_integration.py — TSMOMSignalAdapter, confidence, deltas.
Mocks TSMOMOverlay.compute_signal to isolate adapter logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.signals.tsmom_integration import (
    TSMOMSignalAdapter,
    get_tsmom_integrator_result,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_tsmom_signal(
    signal=1,
    lookback_return=0.15,
    realized_vol=0.12,
    vol_scaled_position=0.8,
    base_weight=0.46,
    adjustment=0.05,
    target_weight=0.51,
    timestamp="2026-01-01T00:00:00",
):
    """Create a mock TSMOMSignal."""
    mock = MagicMock()
    mock.signal = signal
    mock.lookback_return = lookback_return
    mock.realized_vol = realized_vol
    mock.vol_scaled_position = vol_scaled_position
    mock.base_weight = base_weight
    mock.adjustment = adjustment
    mock.target_weight = target_weight
    mock.timestamp = timestamp
    return mock


def _patch_compute_signal(signal_or_none):
    """Patch TSMOMOverlay.compute_signal to return the given signal."""
    return patch(
        "src.signals.tsmom_integration.TSMOMOverlay.compute_signal",
        return_value=signal_or_none,
    )


# ── TSMOMSignalAdapter.__init__ ──────────────────────────────────────


class TestAdapterInit:
    def test_default_params(self):
        adapter = TSMOMSignalAdapter()
        assert adapter.base_allocation is not None
        assert "SPY" in adapter.base_allocation

    def test_custom_base_allocation(self):
        custom = {"SPY": 0.5, "GLD": 0.5}
        adapter = TSMOMSignalAdapter(base_allocation=custom)
        assert adapter.base_allocation == custom


# ── get_signal() ─────────────────────────────────────────────────────


class TestGetSignal:
    def test_returns_signal_source_result(self):
        sig = _make_tsmom_signal(signal=1)
        with _patch_compute_signal(sig):
            adapter = TSMOMSignalAdapter()
            result = adapter.get_signal("SPY")
        assert result is not None
        assert result.source_type == "tsmom"
        assert result.source_name == "aqrs_tsmom"

    def test_signal_value_passed(self):
        sig = _make_tsmom_signal(signal=-1)
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("GLD")
        assert result.signal == -1

    def test_none_when_no_signal(self):
        with _patch_compute_signal(None):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert result is None

    def test_raw_score_is_lookback_return(self):
        sig = _make_tsmom_signal(lookback_return=0.22)
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert result.raw_score == 0.22

    def test_metadata_fields(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert "lookback_return" in result.metadata
        assert "realized_vol" in result.metadata
        assert "vol_scaled_position" in result.metadata
        assert "base_weight" in result.metadata
        assert "adjustment" in result.metadata
        assert "target_weight" in result.metadata

    def test_timestamp_passed(self):
        sig = _make_tsmom_signal(timestamp="2026-03-15T12:00:00")
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert result.timestamp == "2026-03-15T12:00:00"

    def test_historical_accuracy_constant(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert result.historical_accuracy == 0.68

    def test_raw_unit(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_signal("SPY")
        assert result.raw_unit == "formation_return_12m"

    def test_generate_signal_alias(self):
        """generate_signal should be an alias for get_signal."""
        adapter = TSMOMSignalAdapter()
        assert adapter.generate_signal == adapter.get_signal


# ── get_portfolio_signals() ──────────────────────────────────────────


class TestGetPortfolioSignals:
    def test_multiple_tickers(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_portfolio_signals(["SPY", "GLD", "TLT"])
        assert len(result) == 3
        assert "SPY" in result
        assert "GLD" in result
        assert "TLT" in result

    def test_none_signals_excluded(self):
        """Tickers with None signals should not appear in results."""
        call_count = 0

        def side_effect(ticker):
            nonlocal call_count
            call_count += 1
            if ticker == "GLD":
                return None
            return _make_tsmom_signal()

        with patch("src.signals.tsmom_integration.TSMOMOverlay.compute_signal", side_effect=side_effect):
            result = TSMOMSignalAdapter().get_portfolio_signals(["SPY", "GLD", "TLT"])
        assert "GLD" not in result
        assert "SPY" in result
        assert "TLT" in result

    def test_empty_tickers(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_portfolio_signals([])
        assert result == {}


# ── get_allocation_deltas() ──────────────────────────────────────────


class TestGetAllocationDeltas:
    def test_default_tickers(self):
        sig = _make_tsmom_signal(adjustment=0.05)
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_allocation_deltas()
        assert "SPY" in result
        assert "GLD" in result
        assert "TLT" in result

    def test_delta_is_adjustment(self):
        sig = _make_tsmom_signal(adjustment=0.08)
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_allocation_deltas(["SPY"])
        assert result["SPY"] == 0.08

    def test_none_signal_gives_zero_delta(self):
        with _patch_compute_signal(None):
            result = TSMOMSignalAdapter().get_allocation_deltas(["SPY"])
        assert result["SPY"] == 0.0

    def test_custom_tickers(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = TSMOMSignalAdapter().get_allocation_deltas(["SPY", "GLD"])
        assert len(result) == 2


# ── _compute_confidence() ────────────────────────────────────────────


class TestComputeConfidence:
    def test_base_confidence_is_050(self):
        """Zero return and high vol gives base 0.50."""
        sig = _make_tsmom_signal(
            signal=0,
            lookback_return=0.0,
            realized_vol=0.30,
        )
        adapter = TSMOMSignalAdapter()
        conf = adapter._compute_confidence(sig)
        # trend_strength=0, vol_stability=0, clarity=0 → 0.50
        assert conf == pytest.approx(0.50, abs=0.01)

    def test_strong_trend_boosts_confidence(self):
        """Strong lookback return should boost confidence."""
        sig = _make_tsmom_signal(lookback_return=0.30, realized_vol=0.10)
        adapter = TSMOMSignalAdapter()
        conf = adapter._compute_confidence(sig)
        assert conf > 0.50

    def test_low_vol_boosts_confidence(self):
        """Low volatility should boost confidence."""
        low_vol = _make_tsmom_signal(realized_vol=0.05, lookback_return=0.15)
        high_vol = _make_tsmom_signal(realized_vol=0.25, lookback_return=0.15)
        adapter = TSMOMSignalAdapter()
        conf_low = adapter._compute_confidence(low_vol)
        conf_high = adapter._compute_confidence(high_vol)
        assert conf_low > conf_high

    def test_signal_nonzero_adds_clarity(self):
        """Non-zero signal adds clarity bonus."""
        zero_signal = _make_tsmom_signal(signal=0, lookback_return=0.0)
        nonzero_signal = _make_tsmom_signal(signal=1, lookback_return=0.15)
        adapter = TSMOMSignalAdapter()
        conf_zero = adapter._compute_confidence(zero_signal)
        conf_nonzero = adapter._compute_confidence(nonzero_signal)
        assert conf_nonzero > conf_zero

    def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        sig = _make_tsmom_signal(
            lookback_return=0.50,
            realized_vol=0.01,
            signal=1,
        )
        adapter = TSMOMSignalAdapter()
        conf = adapter._compute_confidence(sig)
        assert conf <= 1.0

    def test_trend_strength_capped(self):
        """Trend strength contribution maxes out at 0.25."""
        sig = _make_tsmom_signal(lookback_return=1.0, realized_vol=0.30)
        adapter = TSMOMSignalAdapter()
        conf = adapter._compute_confidence(sig)
        # Even with extreme return, confidence is capped
        assert conf <= 1.0


# ── get_tsmom_integrator_result() ────────────────────────────────────


class TestGetTsmomIntegratorResult:
    def test_default_tickers(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = get_tsmom_integrator_result()
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_custom_tickers(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = get_tsmom_integrator_result(tickers=["SPY", "GLD"])
        assert "SPY" in result
        assert "GLD" in result

    def test_custom_allocation(self):
        sig = _make_tsmom_signal()
        with _patch_compute_signal(sig):
            result = get_tsmom_integrator_result(
                tickers=["SPY"],
                base_allocation={"SPY": 0.5, "GLD": 0.5},
            )
        assert "SPY" in result
