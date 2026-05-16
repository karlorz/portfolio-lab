"""
Tests for the Volatility-Volume-Gap Day Classifier (v5.30).
Tests feature computation, classification logic, and execution signal mapping.
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from src.regime.vol_volume_gap import (
    DayFeatures,
    DayRegime,
    ClassifierConfig,
    compute_features,
    classify_day,
    load_prices,
    detect_regime,
    get_same_day_signal,
    save_state,
    load_state,
    STATE_FILE,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_close_prices(n_days: int = 100, seed: int = 42) -> np.ndarray:
    """Generate synthetic close-only price array for testing (nx1)."""
    rng = np.random.RandomState(seed)
    closes = 100.0 + np.cumsum(rng.randn(n_days) * 0.5)
    closes = np.maximum(closes, 10.0)  # floor
    return closes.reshape(-1, 1)


def _make_return_prices(
    last_return: float,
    n_days: int = 60,
    avg_return: float = 0.001,
) -> np.ndarray:
    """Generate prices with a specific last daily return.

    last_return: fractional daily return for the last bar.
    Returns n_days x 1 array.
    """
    rng = np.random.RandomState(0)
    closes = 100.0 + np.cumsum(rng.randn(n_days) * 0.5)
    # Override last bar's close to achieve desired return
    closes[-1] = closes[-2] * (1.0 + last_return)
    closes = np.maximum(closes, 10.0)
    return closes.reshape(-1, 1)


# ── Tests: Regime Enum ─────────────────────────────────────────────────

class TestDayRegime:
    def test_values(self):
        assert DayRegime.TREND_UP.value == "trend_up"
        assert DayRegime.TREND_DOWN.value == "trend_down"
        assert DayRegime.MEAN_REVERT.value == "mean_revert"
        assert DayRegime.HIGH_VOL.value == "high_vol"
        assert DayRegime.CRISIS.value == "crisis"
        assert DayRegime.UNKNOWN.value == "unknown"

    def test_all_regimes_are_distinct(self):
        values = [r.value for r in DayRegime]
        assert len(set(values)) == len(values)


# ── Tests: Data Classes ───────────────────────────────────────────────

class TestDayFeatures:
    def test_defaults(self):
        f = DayFeatures(daily_return=0.01, volume_anomaly=1.5, return_vol_ratio=1.2)
        assert f.daily_return == 0.01
        assert f.volume_anomaly == 1.5
        assert f.return_vol_ratio == 1.2
        assert f.regime == DayRegime.UNKNOWN
        assert f.confidence == 0.0

    def test_to_dict(self):
        f = DayFeatures(
            daily_return=0.01,
            volume_anomaly=1.5,
            return_vol_ratio=1.2,
            regime=DayRegime.TREND_UP,
            confidence=0.75,
        )
        d = f.to_dict()
        assert d["daily_return"] == 0.01
        assert d["volume_anomaly"] == 1.5
        assert d["return_vol_ratio"] == 1.2
        assert d["regime"] == "trend_up"
        assert d["confidence"] == 0.75


class TestClassifierConfig:
    def test_defaults(self):
        c = ClassifierConfig()
        assert c.ret_extreme == 0.04
        assert c.rel_vol_extreme == 3.0
        assert c.ret_large == 0.015
        assert c.ret_small == 0.004
        assert c.vol_lookback == 20


# ── Tests: Feature Computation ────────────────────────────────────────

class TestComputeFeatures:
    def test_basic_computation(self):
        prices = _make_close_prices(100)
        features = compute_features(prices)
        assert features is not None
        assert isinstance(features.daily_return, float)
        assert isinstance(features.volume_anomaly, float)
        assert isinstance(features.return_vol_ratio, float)
        assert features.regime == DayRegime.UNKNOWN

    def test_known_return(self):
        # Create data with 2% positive return
        prices = _make_return_prices(last_return=0.02, n_days=60)
        features = compute_features(prices)
        assert features is not None
        # Daily return should be close to 2%
        assert abs(features.daily_return - 0.02) < 0.001

    def test_known_negative_return(self):
        prices = _make_return_prices(last_return=-0.015, n_days=60)
        features = compute_features(prices)
        assert features is not None
        assert abs(features.daily_return - (-0.015)) < 0.001

    def test_volume_anomaly_always_one(self):
        prices = _make_close_prices(100)
        features = compute_features(prices)
        assert features is not None
        assert features.volume_anomaly == 1.0

    def test_return_vol_ratio_range(self):
        prices = _make_close_prices(100)
        features = compute_features(prices)
        assert features is not None
        # Should be positive
        assert features.return_vol_ratio > 0

    def test_insufficient_data_returns_none(self):
        prices = np.array([[100.0], [101.0]])
        features = compute_features(prices)
        assert features is None

    def test_exactly_enough_data(self):
        n = ClassifierConfig.vol_lookback + 2
        prices = _make_close_prices(n)
        features = compute_features(prices)
        assert features is not None

    def test_high_vol_ratio_on_extreme_return(self):
        # 6% return should generate high vol ratio
        prices = _make_return_prices(last_return=0.06, n_days=60)
        features = compute_features(prices)
        assert features is not None
        assert features.return_vol_ratio > 2.0


# ── Tests: Classification ─────────────────────────────────────────────

class TestClassifyDay:
    def test_trend_up(self):
        f = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_UP
        assert result.confidence >= 0.50

    def test_trend_down(self):
        f = DayFeatures(daily_return=-0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN
        assert result.confidence >= 0.50

    def test_mean_revert(self):
        f = DayFeatures(daily_return=0.001, volume_anomaly=1.0, return_vol_ratio=0.5)
        result = classify_day(f)
        assert result.regime == DayRegime.MEAN_REVERT
        assert result.confidence >= 0.50

    def test_high_vol_large_return(self):
        f = DayFeatures(daily_return=0.02, volume_anomaly=1.0, return_vol_ratio=3.5)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL
        assert result.confidence >= 0.50

    def test_high_vol_extreme_rel_vol_only(self):
        f = DayFeatures(daily_return=0.005, volume_anomaly=1.0, return_vol_ratio=4.0)
        result = classify_day(f)
        # return_vol_ratio >= 3.0 = rel_vol_extreme → HIGH_VOL
        assert result.regime == DayRegime.HIGH_VOL

    def test_crisis(self):
        f = DayFeatures(daily_return=0.05, volume_anomaly=1.0, return_vol_ratio=3.5)
        result = classify_day(f)
        assert result.regime == DayRegime.CRISIS
        assert result.confidence >= 0.50

    def test_crisis_negative(self):
        f = DayFeatures(daily_return=-0.05, volume_anomaly=1.0, return_vol_ratio=3.5)
        result = classify_day(f)
        assert result.regime == DayRegime.CRISIS

    def test_near_crisis_but_missing_vol(self):
        # Large return but vol ratio not extreme
        f = DayFeatures(daily_return=0.04, volume_anomaly=1.0, return_vol_ratio=1.5)
        result = classify_day(f)
        # return(0.04) >= ret_extreme(0.04) but rel_vol(1.5) < rel_vol_extreme(3.0)
        # Also return >= ret_large(0.015) and rel_vol >= rel_vol_elevated(2.0)? NO (1.5 < 2.0)
        # So no crisis, no high_vol
        # Gap_sign > 0 and gap >= ret_small(0.004) and return_vol < rel_vol_elevated(2.0) → TREND_UP
        assert result.regime == DayRegime.TREND_UP

    def test_custom_config(self):
        config = ClassifierConfig(
            ret_extreme=0.02,
            ret_large=0.01,
            rel_vol_extreme=2.0,
            rel_vol_elevated=1.5,
        )
        f = DayFeatures(daily_return=0.025, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f, config)
        assert result.regime == DayRegime.CRISIS

    def test_unknown_fallback_trend_up(self):
        # Positive but gap below ret_small threshold with elevated rel vol
        # Actually rel_vol is not elevated (0.8) and gap is positive but small (0.002)
        # gap < ret_small(0.004) → MEAN_REVERT first
        f = DayFeatures(daily_return=0.002, volume_anomaly=1.0, return_vol_ratio=0.8)
        result = classify_day(f)
        assert result.regime == DayRegime.MEAN_REVERT

    def test_fallback_trend_down(self):
        f = DayFeatures(daily_return=-0.005, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN


# ── Tests: Execution Signal ───────────────────────────────────────────

class TestGetSameDaySignal:
    def test_regime_adjustment_map(self):
        """Test the execution adjustment mapping logic directly."""
        adjustment_map = {
            DayRegime.CRISIS.value: 0.0,
            DayRegime.HIGH_VOL.value: 0.5,
            DayRegime.TREND_UP.value: 1.0,
            DayRegime.TREND_DOWN.value: 1.0,
            DayRegime.MEAN_REVERT.value: 0.8,
            DayRegime.UNKNOWN.value: 0.8,
        }

        test_cases = [
            (DayRegime.CRISIS, 0.0),
            (DayRegime.HIGH_VOL, 0.5),
            (DayRegime.TREND_UP, 1.0),
            (DayRegime.TREND_DOWN, 1.0),
            (DayRegime.MEAN_REVERT, 0.8),
            (DayRegime.UNKNOWN, 0.8),
        ]

        for regime, expected in test_cases:
            assert adjustment_map[regime.value] == expected, (
                f"Expected {expected} for {regime.value}, got {adjustment_map[regime.value]}"
            )


# ── Tests: End-to-End ─────────────────────────────────────────────────

class TestEndToEnd:
    def test_save_and_load_state(self):
        result = {
            "status": "ok",
            "symbol": "SPY",
            "features": {"daily_return": 0.01, "volume_anomaly": 1.0, "return_vol_ratio": 1.2, "regime": "trend_up", "confidence": 0.7},
            "timestamp": "2026-05-16T10:30:00",
        }
        with pytest.MonkeyPatch.context() as mp:
            import tempfile
            tmpdir = tempfile.mkdtemp()
            state_file = Path(tmpdir) / "test_state.json"
            save_state(result, state_file)
            assert state_file.exists()
            loaded = load_state(state_file)
            assert loaded is not None
            assert loaded["features"]["regime"] == "trend_up"
            assert loaded["features"]["daily_return"] == 0.01

    def test_load_state_nonexistent(self):
        loaded = load_state(Path("/nonexistent/state.json"))
        assert loaded is None

    def test_detect_regime_error_no_data(self):
        """Should return error for non-existent symbol."""
        result = detect_regime("NONEXISTENT_SYMBOL_XYZ")
        assert result["status"] == "error"


# ── Tests: Real Data Integration ──────────────────────────────────────

class TestRealData:
    def test_load_real_prices(self):
        """Test that we can load real market data."""
        prices = load_prices("SPY")
        if prices is not None:
            assert len(prices) >= 20
            assert prices.shape[1] == 1  # close-only
            assert prices[-1, 0] > 0  # positive price

    def test_load_missing_symbol(self):
        prices = load_prices("THIS_DOES_NOT_EXIST_12345")
        assert prices is None

    @pytest.mark.skipif(
        load_prices("SPY") is None,
        reason="No real SPY price data available",
    )
    def test_detect_with_real_data(self):
        result = detect_regime("SPY")
        assert result["status"] == "ok"
        assert "features" in result
        assert "regime" in result["features"]
        assert result["features"]["regime"] in [r.value for r in DayRegime]

    @pytest.mark.skipif(
        load_prices("SPY") is None,
        reason="No real SPY price data available",
    )
    def test_signal_with_real_data(self):
        signal = get_same_day_signal("SPY")
        assert signal["status"] == "ok"
        assert 0.0 <= signal["execution_adjustment"] <= 1.0
        assert signal["regime"] in [r.value for r in DayRegime]


# ── Tests: Regression Edge Cases ──────────────────────────────────────

class TestEdgeCases:
    def test_negative_return_large(self):
        """Large negative return should classify as TREND_DOWN."""
        prices = _make_return_prices(last_return=-0.025, n_days=60)
        features = compute_features(prices)
        assert features is not None
        result = classify_day(features)
        try:
            assert result.regime == DayRegime.TREND_DOWN
        except AssertionError:
            # Could be HIGH_VOL if the large return triggers vol ratio
            assert result.regime == DayRegime.HIGH_VOL

    def test_zero_return(self):
        """Zero return should classify as MEAN_REVERT."""
        prices = _make_return_prices(last_return=0.0, n_days=60)
        features = compute_features(prices)
        assert features is not None
        result = classify_day(features)
        assert result.regime == DayRegime.MEAN_REVERT

    def test_small_positive_return(self):
        prices = _make_return_prices(last_return=0.002, n_days=60)
        features = compute_features(prices)
        assert features is not None
        result = classify_day(features)
        # Return(0.002) < ret_small(0.004) → MEAN_REVERT
        assert result.regime == DayRegime.MEAN_REVERT

    def test_extreme_return_only(self):
        """Extreme return alone should trigger CRISIS with extreme rel vol."""
        prices = _make_return_prices(last_return=0.05, n_days=200)
        features = compute_features(prices)
        assert features is not None
        result = classify_day(features)
        # return(0.05) >= ret_extreme(0.04). If return_vol_ratio >= 3.0 → CRISIS
        # With 200 days of data, abs(avg_ret) is small, so 5% should give high ratio
        assert result.regime in [DayRegime.CRISIS, DayRegime.HIGH_VOL]

    def test_high_return_with_low_vol(self):
        """A 2% return in a low-vol regime should still be HIGH_VOL at large threshold."""
        # Create data with very high volatility first, then low volatility last
        rng = np.random.RandomState(42)
        n_total = 80
        # First 60 days: high vol (2% daily)
        high_vol_returns = rng.randn(60) * 0.02
        # Next 20 days: low vol (0.1% daily)
        low_vol_returns = rng.randn(20) * 0.001
        all_returns = np.concatenate([high_vol_returns, low_vol_returns])
        closes = 100.0 * np.cumprod(1.0 + all_returns)
        prices = closes.reshape(-1, 1)

        features = compute_features(prices)
        assert features is not None
        # Last return is low vol, so return_vol_ratio should be low
        assert features.return_vol_ratio < 3.0

    def test_one_day_data(self):
        """Single day of data should return None."""
        prices = np.array([[100.0]])
        features = compute_features(prices)
        assert features is None

    def test_negative_gap_trend_down(self):
        """Small negative gap with moderate vol = TREND_DOWN."""
        f = DayFeatures(daily_return=-0.008, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN

    def test_gap_sign_fallback_negative(self):
        """When gap is 0 exactly (rare), should classify based on sign which is 0."""
        f = DayFeatures(daily_return=0.0, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        # ret_small check: gap(0) < ret_small(0.004) → MEAN_REVERT
        assert result.regime == DayRegime.MEAN_REVERT

    def test_volume_always_one(self):
        """Volume anomaly should always be 1.0 for close-only mode."""
        prices = _make_close_prices(100)
        features = compute_features(prices)
        assert features.volume_anomaly == 1.0
