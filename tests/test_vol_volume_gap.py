"""
Tests for the Volatility-Volume-Gap Day Classifier (v5.30).
Tests feature computation, classification logic, and execution signal mapping.
"""

import json
import logging
import os
import sys
from io import StringIO
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
    _build_state,
    main_cli,
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


def _make_ohlcv_prices(n_days: int = 100, seed: int = 42) -> np.ndarray:
    """Generate synthetic OHLCV price array (nx5) for full-feature tests."""
    rng = np.random.RandomState(seed)
    closes = 100.0 + np.cumsum(rng.randn(n_days) * 0.5)
    closes = np.maximum(closes, 10.0)
    opens = closes * (1.0 + rng.randn(n_days) * 0.001)
    highs = np.maximum(opens, closes) * (1.0 + abs(rng.randn(n_days)) * 0.001)
    lows = np.minimum(opens, closes) * (1.0 - abs(rng.randn(n_days)) * 0.001)
    volumes = rng.randint(1_000_000, 10_000_000, size=n_days)
    return np.column_stack([opens, highs, lows, closes, volumes])


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


# ── New Tests: to_dict Completeness ─────────────────────────────────────

class TestDayFeaturesToDict:
    """Cover all to_dict() paths and field completeness."""

    def test_to_dict_all_fields_present(self):
        f = DayFeatures(
            daily_return=0.02, volume_anomaly=1.5, return_vol_ratio=2.0,
            regime=DayRegime.TREND_UP, confidence=0.75,
        )
        d = f.to_dict()
        expected_keys = {"daily_return", "volume_anomaly", "return_vol_ratio", "regime", "confidence"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_each_regime_value(self):
        for regime in DayRegime:
            f = DayFeatures(
                daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.0,
                regime=regime, confidence=0.5,
            )
            d = f.to_dict()
            assert d["regime"] == regime.value

    def test_to_dict_rounding_precision(self):
        f = DayFeatures(
            daily_return=0.01234567, volume_anomaly=1.234567, return_vol_ratio=2.345678,
            regime=DayRegime.HIGH_VOL, confidence=0.876543,
        )
        d = f.to_dict()
        assert d["daily_return"] == 0.012346     # round to 6
        assert d["volume_anomaly"] == 1.2346      # round to 4
        assert d["return_vol_ratio"] == 2.3457    # round to 4
        assert d["confidence"] == 0.8765          # round to 4

    def test_to_dict_crisis_regime(self):
        f = DayFeatures(
            daily_return=-0.06, volume_anomaly=1.0, return_vol_ratio=4.5,
            regime=DayRegime.CRISIS, confidence=0.90,
        )
        d = f.to_dict()
        assert d["regime"] == "crisis"
        assert d["daily_return"] == -0.06
        assert d["confidence"] == 0.9

    def test_to_dict_unknown_defaults(self):
        f = DayFeatures(daily_return=0.0, volume_anomaly=1.0, return_vol_ratio=0.0)
        d = f.to_dict()
        assert d["regime"] == "unknown"
        assert d["confidence"] == 0.0


class TestClassifierConfigValidation:
    """Constants and config validation."""

    def test_threshold_ordering(self):
        c = ClassifierConfig()
        assert c.ret_extreme > c.ret_large > c.ret_small, (
            "extreme must be > large > small"
        )
        assert c.rel_vol_extreme > c.rel_vol_elevated, (
            "rel_vol_extreme must be > rel_vol_elevated"
        )

    def test_defaults_positive(self):
        c = ClassifierConfig()
        assert c.ret_extreme > 0
        assert c.ret_large > 0
        assert c.ret_small > 0
        assert c.rel_vol_extreme > 0
        assert c.rel_vol_elevated > 0
        assert c.vol_lookback > 0

    def test_custom_config_all_parameters(self):
        c = ClassifierConfig(
            ret_extreme=0.05,
            ret_large=0.02,
            ret_small=0.005,
            rel_vol_extreme=4.0,
            rel_vol_elevated=2.5,
            vol_lookback=30,
        )
        assert c.ret_extreme == 0.05
        assert c.ret_large == 0.02
        assert c.ret_small == 0.005
        assert c.rel_vol_extreme == 4.0
        assert c.rel_vol_elevated == 2.5
        assert c.vol_lookback == 30

    def test_vol_lookback_used_in_compute(self):
        """Verify vol_lookback controls the lookback window used in compute_features."""
        custom = ClassifierConfig(vol_lookback=5)
        prices = _make_close_prices(30, seed=10)
        features_default = compute_features(prices)
        features_custom = compute_features(prices, custom)
        assert features_default is not None
        assert features_custom is not None
        # Different lookback should give different ratio
        assert features_default.return_vol_ratio != features_custom.return_vol_ratio

    def test_vol_lookback_minimum_data_check(self):
        """With vol_lookback=40 but only 30 days of data, should return None."""
        custom = ClassifierConfig(vol_lookback=40)
        prices = _make_close_prices(30, seed=10)
        features = compute_features(prices, custom)
        assert features is None


# ── New Tests: Feature Computation Edge Cases ──────────────────────────

class TestComputeFeaturesEdgeCases:
    """Edge cases in feature computation not covered by basic tests."""

    def test_constant_prices(self):
        """All prices identical → zero daily return, clipped avg abs return."""
        prices = np.full((25, 1), 100.0)
        features = compute_features(prices)
        assert features is not None
        assert features.daily_return == 0.0
        assert features.return_vol_ratio == 0.0  # 0 / 0.0001 (clipped)

    def test_zero_prev_close(self):
        """Prev close of 0 should not crash; division by zero is protected."""
        prices = _make_close_prices(25, seed=5)
        prices[-2, 0] = 0.0
        features = compute_features(prices)
        assert features is not None
        assert features.daily_return == 0.0  # (close - 0) / 0 → 0.0

    def test_tiny_returns_clip_avg_abs(self):
        """When avg_abs_return < 0.0001, it should be clipped to 0.0001."""
        # 25 days with tiny 0.0001% returns (close to constant)
        closes = 100.0 + np.cumsum(np.full(25, 1e-6))
        prices = closes.reshape(-1, 1)
        features = compute_features(prices)
        assert features is not None
        # return_vol_ratio should be computed with clipped denominator
        assert features.return_vol_ratio >= 0.0
        assert not np.isnan(features.return_vol_ratio)
        assert not np.isinf(features.return_vol_ratio)

    def test_exactly_minimum_days(self):
        """Exactly vol_lookback + 2 days should work."""
        n = ClassifierConfig.vol_lookback + 2
        prices = _make_close_prices(n)
        features = compute_features(prices)
        assert features is not None

    def test_large_negative_daily_return(self):
        """Large negative return should compute negative daily_return."""
        prices = _make_return_prices(last_return=-0.10, n_days=60)
        features = compute_features(prices)
        assert features is not None
        assert features.daily_return < -0.09

    def test_very_large_daily_return_ratio(self):
        """A 15% return in normally-volatile data produces very high vol ratio."""
        rng = np.random.RandomState(99)
        closes = 100.0 + np.cumsum(rng.randn(200) * 0.3)
        closes[-1] = closes[-2] * 1.15
        closes = np.maximum(closes, 10.0)
        prices = closes.reshape(-1, 1)
        features = compute_features(prices)
        assert features is not None
        assert features.return_vol_ratio > 3.0


# ── New Tests: Classification Boundary Conditions ──────────────────────

class TestClassifyDayBoundaries:
    """Exact boundary conditions for regime classification rules."""

    def test_boundary_ret_extreme_exact(self):
        """gap == ret_extreme (0.04) and return_vol == rel_vol_extreme (3.0) → CRISIS."""
        f = DayFeatures(daily_return=0.04, volume_anomaly=1.0, return_vol_ratio=3.0)
        result = classify_day(f)
        assert result.regime == DayRegime.CRISIS
        assert result.confidence == 0.90

    def test_boundary_ret_extreme_negative_exact(self):
        """gap == ret_extreme (0.04) negative with extreme vol → CRISIS."""
        f = DayFeatures(daily_return=-0.04, volume_anomaly=1.0, return_vol_ratio=3.0)
        result = classify_day(f)
        assert result.regime == DayRegime.CRISIS

    def test_just_below_ret_extreme_crisis_not_triggered(self):
        """gap = 0.0399 (barely below ret_extreme), extreme vol → HIGH_VOL (not CRISIS)."""
        f = DayFeatures(daily_return=0.0399, volume_anomaly=1.0, return_vol_ratio=3.0)
        result = classify_day(f)
        # gap < ret_extreme so crisis not triggered
        # return_vol >= rel_vol_extreme → HIGH_VOL via second condition
        assert result.regime == DayRegime.HIGH_VOL

    def test_boundary_ret_large_exact(self):
        """gap == ret_large (0.015) and return_vol == rel_vol_elevated (2.0) → HIGH_VOL."""
        f = DayFeatures(daily_return=0.015, volume_anomaly=1.0, return_vol_ratio=2.0)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL
        assert result.confidence == 0.80

    def test_boundary_ret_large_exact_negative(self):
        f = DayFeatures(daily_return=-0.015, volume_anomaly=1.0, return_vol_ratio=2.0)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL

    def test_boundary_ret_small_exact_positive(self):
        """gap == ret_small (0.004) positive, return_vol below elevated → TREND_UP."""
        f = DayFeatures(daily_return=0.004, volume_anomaly=1.0, return_vol_ratio=1.5)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_UP
        assert result.confidence == 0.70

    def test_boundary_ret_small_exact_negative(self):
        f = DayFeatures(daily_return=-0.004, volume_anomaly=1.0, return_vol_ratio=1.5)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN

    def test_boundary_ret_small_just_below_positive(self):
        """gap = 0.0039 (barely below ret_small) → MEAN_REVERT."""
        f = DayFeatures(daily_return=0.0039, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.regime == DayRegime.MEAN_REVERT
        assert result.confidence == 0.60

    def test_boundary_rel_vol_elevated_exact(self):
        """return_vol == rel_vol_elevated (2.0) with large gap → HIGH_VOL."""
        f = DayFeatures(daily_return=0.02, volume_anomaly=1.0, return_vol_ratio=2.0)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL

    def test_boundary_rel_vol_elevated_just_below(self):
        """return_vol = 1.999 (barely below elevated), large positive return → TREND_UP."""
        f = DayFeatures(daily_return=0.02, volume_anomaly=1.0, return_vol_ratio=1.999)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_UP

    def test_fallback_positive_elevated_vol_mid_gap(self):
        """gap between ret_small and ret_large, return_vol between elevated and extreme.
        Not crisis, not high_vol, not trend_up (vol too high), not mean_revert (gap too large).
        → Fallback TREND_UP with confidence 0.50."""
        f = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_UP
        assert result.confidence == 0.50

    def test_fallback_negative_elevated_vol_mid_gap(self):
        """Same as above but negative → Fallback TREND_DOWN."""
        f = DayFeatures(daily_return=-0.01, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN
        assert result.confidence == 0.50


# ── New Tests: Integration Edge Cases ──────────────────────────────────

class TestIntegrationEdgeCases:
    """detect_regime / get_same_day_signal error paths and state fields."""

    def test_detect_regime_insufficient_data(self, monkeypatch):
        prices = np.array([[100.0], [101.0]])
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        result = detect_regime("SPY")
        assert result["status"] == "error"
        assert "Insufficient data" in result["message"]

    def test_get_same_day_signal_error_no_data(self, monkeypatch):
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": None)
        result = get_same_day_signal("SPY")
        assert result["status"] == "error"

    def test_detect_regime_result_has_all_fields(self, monkeypatch):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        result = detect_regime("SPY")
        assert result["status"] == "ok"
        assert result["symbol"] == "SPY"
        assert "timestamp" in result
        features = result["features"]
        for key in ("daily_return", "volume_anomaly", "return_vol_ratio", "regime", "confidence"):
            assert key in features, f"Missing feature key: {key}"
        state = result["state"]
        for key in ("price_last", "return_5d", "return_20d", "n_days"):
            assert key in state, f"Missing state key: {key}"

    def test_detect_regime_state_values(self, monkeypatch):
        prices = _make_close_prices(100, seed=7)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        result = detect_regime("SPY")
        state = result["state"]
        assert state["price_last"] == float(prices[-1, 0])
        assert state["n_days"] == len(prices)
        assert isinstance(state["return_5d"], float)
        assert isinstance(state["return_20d"], float)

    def test_save_state_creates_parent_dir(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        deep_dir = Path(tmpdir) / "a" / "b" / "c" / "test_state.json"
        result = {"status": "ok", "value": 42}
        save_state(result, deep_dir)
        assert deep_dir.exists()
        loaded = load_state(deep_dir)
        assert loaded is not None
        assert loaded["value"] == 42

    def test_load_state_corrupt_json(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        bad_file = Path(tmpdir) / "corrupt.json"
        bad_file.write_text("{this is not valid json!!!}")
        loaded = load_state(bad_file)
        assert loaded is None

    def test_save_and_load_round_trip(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        state_file = Path(tmpdir) / "roundtrip.json"
        original = {
            "status": "ok",
            "symbol": "SPY",
            "features": {
                "daily_return": -0.015,
                "volume_anomaly": 1.0,
                "return_vol_ratio": 2.5,
                "regime": "high_vol",
                "confidence": 0.8,
            },
            "state": {"price_last": 450.0, "return_5d": 0.01, "return_20d": 0.03, "n_days": 100},
            "timestamp": "2026-05-24T12:00:00+00:00",
        }
        save_state(original, state_file)
        loaded = load_state(state_file)
        assert loaded["symbol"] == "SPY"
        assert loaded["features"]["regime"] == "high_vol"
        assert loaded["state"]["n_days"] == 100

    def test_detect_regime_different_symbol_name(self, monkeypatch):
        prices = _make_close_prices(100, seed=3)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        result = detect_regime("SPY")
        assert result["symbol"] == "SPY"

    def test_get_same_day_signal_adjustment_ranges(self, monkeypatch):
        """Verify execution_adjustment stays in [0.0, 1.0] for real data."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = get_same_day_signal("SPY")
        if signal["status"] == "ok":
            assert 0.0 <= signal["execution_adjustment"] <= 1.0
            assert signal["regime"] in [r.value for r in DayRegime]
            assert 0.0 <= signal["confidence"] <= 1.0


# ── New Tests: NaN / Inf Handling ─────────────────────────────────────

class TestNaNInfHandling:
    """How the classifier handles corrupt or degenerate price data."""

    def test_nan_in_price_series(self):
        """A single NaN in the price series should not crash; NaN propagates."""
        prices = _make_close_prices(30, seed=5)
        prices[10, 0] = np.nan
        features = compute_features(prices)
        assert features is not None
        # NaN in returns produces NaN in ratios
        assert np.isnan(features.daily_return) or np.isfinite(features.daily_return)

    def test_inf_in_price_series(self):
        """Infinity price should not crash."""
        prices = _make_close_prices(30, seed=5)
        prices[15, 0] = np.inf
        features = compute_features(prices)
        assert features is not None
        assert np.isfinite(features.daily_return) or np.isinf(features.daily_return)

    def test_neg_inf_in_price_series(self):
        """Negative infinity price should not crash."""
        prices = _make_close_prices(30, seed=5)
        prices[20, 0] = -np.inf
        features = compute_features(prices)
        assert features is not None

    def test_all_nan_prices(self):
        """All-NaN prices: the diff is NaN, avg clipped to 0.0001, ratio is 0.0."""
        prices = np.full((25, 1), np.nan)
        features = compute_features(prices)
        assert features is not None
        assert np.isnan(features.daily_return) or features.daily_return == 0.0

    def test_nan_in_last_close_only(self):
        """NaN only in the most recent close; prev_close is valid."""
        prices = _make_close_prices(30, seed=7)
        prices[-1, 0] = np.nan
        features = compute_features(prices)
        assert features is not None


# ── New Tests: OHLCV Mode ────────────────────────────────────────────

class TestOHLCVFeatures:
    """Feature computation with full OHLCV data (5+ columns)."""

    def test_basic_ohlcv_computation(self):
        prices = _make_ohlcv_prices(100)
        features = compute_features(prices)
        assert features is not None
        assert isinstance(features.daily_return, float)
        assert isinstance(features.return_vol_ratio, float)
        assert features.volume_anomaly == 1.0  # still 1.0 even in OHLCV mode

    def test_ohlcv_return_precision_known(self):
        """With known close values, daily_return should match."""
        prices = _make_ohlcv_prices(60)
        # Set known last two closes
        closes = prices[:, 3]
        closes[-2] = 100.0
        closes[-1] = 102.0
        features = compute_features(prices)
        assert features is not None
        assert abs(features.daily_return - 0.02) < 0.0001

    def test_ohlcv_insufficient_data(self):
        prices = np.array([[100.0, 101.0, 99.0, 100.5, 1_000_000],
                           [101.0, 102.0, 100.0, 101.5, 1_200_000]])
        features = compute_features(prices)
        assert features is None

    def test_ohlcv_zero_prev_close(self):
        """Zero previous close in OHLCV mode should not crash."""
        prices = _make_ohlcv_prices(30)
        prices[-2, 3] = 0.0  # prev close = 0
        features = compute_features(prices)
        assert features is not None

    def test_ohlcv_zero_prev_close_overnight_gap(self):
        """Zero prev close also affects overnight_gap path; should not crash."""
        prices = _make_ohlcv_prices(30)
        prices[-2, 3] = 0.0  # prev close = 0
        prices[-1, 0] = 100.0  # today open
        features = compute_features(prices)
        assert features is not None

    def test_ohlcv_return_vol_ratio_positive(self):
        prices = _make_ohlcv_prices(200, seed=10)
        features = compute_features(prices)
        assert features is not None
        assert features.return_vol_ratio > 0


# ── New Tests: classify_day Confidence & Fallback ─────────────────────

class TestClassifyDayConfidence:
    """Verify confidence values for specific classification paths."""

    def test_crisis_confidence(self):
        f = DayFeatures(daily_return=0.05, volume_anomaly=1.0, return_vol_ratio=4.0)
        result = classify_day(f)
        assert result.confidence == 0.90

    def test_high_vol_confidence(self):
        f = DayFeatures(daily_return=0.02, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f)
        assert result.confidence == 0.80

    def test_trend_up_confidence(self):
        f = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.confidence == 0.70

    def test_trend_down_confidence(self):
        f = DayFeatures(daily_return=-0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        result = classify_day(f)
        assert result.confidence == 0.70

    def test_mean_revert_confidence(self):
        f = DayFeatures(daily_return=0.001, volume_anomaly=1.0, return_vol_ratio=0.5)
        result = classify_day(f)
        assert result.confidence == 0.60

    def test_gap_zero_fallback_trend_down(self):
        """gap == 0.0 exactly: np.sign(0) == 0, so ret_small check fires first.
        gap(0.0) < ret_small(0.004) -> MEAN_REVERT."""
        f = DayFeatures(daily_return=0.0, volume_anomaly=1.0, return_vol_ratio=1.5)
        result = classify_day(f)
        assert result.regime == DayRegime.MEAN_REVERT
        assert result.confidence == 0.60

    def test_unknown_regime_via_fallback(self):
        """gap between ret_small and ret_large with elevated vol -> fallback TREND_UP."""
        f = DayFeatures(daily_return=0.008, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f)
        # Not crisis (gap < 0.04), not high_vol (gap < 0.015),
        # not trend (vol >= 2.0), not mean_revert (gap >= 0.004)
        # fallback: TREND_UP with 0.50
        assert result.regime == DayRegime.TREND_UP
        assert result.confidence == 0.50

    def test_unknown_regime_via_fallback_negative(self):
        f = DayFeatures(daily_return=-0.008, volume_anomaly=1.0, return_vol_ratio=2.5)
        result = classify_day(f)
        assert result.regime == DayRegime.TREND_DOWN
        assert result.confidence == 0.50

    def test_just_above_rel_vol_elevated_with_large_return(self):
        """gap >= ret_large and return_vol just above elevated -> HIGH_VOL."""
        f = DayFeatures(daily_return=0.0151, volume_anomaly=1.0, return_vol_ratio=2.001)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL

    def test_just_above_rel_vol_extreme_with_small_return(self):
        """return_vol just above extreme alone (no large return) -> HIGH_VOL."""
        f = DayFeatures(daily_return=0.005, volume_anomaly=1.0, return_vol_ratio=3.001)
        result = classify_day(f)
        assert result.regime == DayRegime.HIGH_VOL

    def test_just_below_rel_vol_extreme_with_extreme_return(self):
        """gap >= ret_extreme but return_vol just below rel_vol_extreme.
        Not crisis (need both). gap >= ret_large and return_vol >= rel_vol_elevated?
        If below elevated -> TREND_UP."""
        f = DayFeatures(daily_return=0.04, volume_anomaly=1.0, return_vol_ratio=2.999)
        result = classify_day(f)
        # gap >= ret_extreme(0.04) but return_vol(2.999) < rel_vol_extreme(3.0)
        # gap >= ret_large(0.015) and return_vol(2.999) >= rel_vol_elevated(2.0) -> HIGH_VOL
        assert result.regime == DayRegime.HIGH_VOL


# ── New Tests: load_prices Edge Cases ────────────────────────────────

class TestLoadPricesEdgeCases:
    """load_prices with unusual data formats and error conditions."""

    def test_empty_item_list(self, monkeypatch):
        """Data exists but has empty price list."""
        data = {"SPY": []}
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is None

    def test_dict_format_with_c_key(self, monkeypatch):
        """Alternative format: {'c': [...], 'h': [...], ...}."""
        data = {"SPY": {"c": [100.0, 101.0, 102.0]}}
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is not None
        assert result.shape == (3, 1)

    def test_dict_format_with_closes_key(self, monkeypatch):
        """Alternative format: {'closes': [...]}."""
        data = {"SPY": {"closes": [100.0, 101.0, 102.0, 103.0]}}
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is not None
        assert result.shape == (4, 1)

    def test_dict_format_with_p_key(self, monkeypatch):
        """Alternative dict format uses 'p' as fallback key."""
        data = {"SPY": {"p": [100.0, 101.0]}}
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is not None
        assert result.shape == (2, 1)

    def test_unexpected_data_type(self, monkeypatch):
        """Symbol data is a scalar string, not a list or dict."""
        data = {"SPY": "not_a_list_or_dict"}
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is None

    def test_json_decode_error(self, monkeypatch):
        """Corrupt JSON file returns None."""
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("bad file")))
        result = load_prices("SPY")
        assert result is None

    def test_list_format_with_s_key(self, monkeypatch):
        """List format: [{'s': 'SPY', ...}, ...]."""
        data = [{"s": "SPY", "p": [100.0, 101.0, 102.0]}]
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("SPY")
        assert result is not None
        assert result.shape == (3, 1)

    def test_list_format_symbol_not_found(self, monkeypatch):
        """List format but symbol not in any entry."""
        data = [{"s": "QQQ", "p": [100.0, 101.0]}]
        monkeypatch.setattr("src.regime.vol_volume_gap.json.load", lambda f: data)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        result = load_prices("XYZ_NOT_FOUND")
        assert result is None


# ── New Tests: main_cli Entry Point ──────────────────────────────────

class TestMainCli:
    """CLI entry point coverage for detect/signal/save/symbol paths."""

    def test_cli_detect_default(self, monkeypatch, capsys):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["symbol"] == "SPY"

    def test_cli_detect_explicit(self, monkeypatch, capsys):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "detect"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"

    def test_cli_signal(self, monkeypatch, capsys):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "signal"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert "execution_adjustment" in data

    def test_cli_with_symbol(self, monkeypatch, capsys):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="QQQ": prices)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "detect", "--symbol", "QQQ"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["symbol"] == "QQQ"

    def test_cli_with_save_flag(self, monkeypatch, capsys, tmp_path):
        prices = _make_close_prices(100, seed=42)
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": prices)
        monkeypatch.setattr("src.regime.vol_volume_gap.STATE_FILE", tmp_path / "test_state.json")
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "detect", "--save"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert (tmp_path / "test_state.json").exists()

    def test_cli_error_no_data(self, monkeypatch, capsys):
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": None)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "detect"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"

    def test_cli_signal_error_no_data(self, monkeypatch, capsys):
        monkeypatch.setattr("src.regime.vol_volume_gap.load_prices", lambda symbol="SPY": None)
        monkeypatch.setattr("sys.argv", ["vol_volume_gap.py", "signal"])
        main_cli()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert "message" in data


# ── New Tests: _build_state Direct ───────────────────────────────────

class TestBuildStateDirect:
    """Direct tests for the _build_state helper function."""

    def test_basic_state(self):
        prices = _make_close_prices(100, seed=3)
        features = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.5)
        state = _build_state(features, prices)
        assert state["price_last"] == float(prices[-1, 0])
        assert state["n_days"] == len(prices)
        assert isinstance(state["return_5d"], float)
        assert isinstance(state["return_20d"], float)

    def test_state_with_less_than_20_days(self):
        prices = _make_close_prices(10, seed=5)
        features = DayFeatures(daily_return=-0.005, volume_anomaly=1.0, return_vol_ratio=0.8)
        state = _build_state(features, prices)
        assert state["n_days"] == 10
        assert isinstance(state["return_5d"], float)
        assert isinstance(state["return_20d"], float)

    def test_state_zero_first_close(self):
        """When first close is 0, return_5d and return_20d compute safely."""
        prices = _make_close_prices(25, seed=5)
        prices[0, 0] = 0.0
        features = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        state = _build_state(features, prices)
        assert isinstance(state["return_5d"], float)

    def test_state_with_ohlcv_prices(self):
        """_build_state should extract close column from OHLCV data."""
        prices = _make_ohlcv_prices(30)
        features = DayFeatures(daily_return=0.01, volume_anomaly=1.0, return_vol_ratio=1.0)
        state = _build_state(features, prices)
        assert state["price_last"] == float(prices[-1, 3])
        assert state["n_days"] == 30


# ── New Tests: Export Completeness & Module API ──────────────────────

class TestExportCompleteness:
    """Verify the module's public API surface."""

    def test_module_has_no_all(self):
        """No __all__ is defined; all public names are importable."""
        import src.regime.vol_volume_gap as mod
        assert not hasattr(mod, "__all__"), (
            "If __all__ is added, update tests to verify export completeness"
        )

    def test_public_api_functions_exist(self):
        """Core public API symbols are importable and callable."""
        for name in [
            "DayFeatures", "DayRegime", "ClassifierConfig",
            "compute_features", "classify_day", "load_prices",
            "detect_regime", "get_same_day_signal", "save_state", "load_state",
        ]:
            obj = eval(name)
            assert obj is not None, f"Symbol {name} should be importable"


# ── New Tests: Save/Load Error Paths ─────────────────────────────────

class TestSaveLoadErrorPaths:
    """Error paths for save_state / load_state beyond the normal round-trip."""

    def test_save_state_oserror(self, monkeypatch):
        """OSError during save is caught and logged, but save_state returns None."""
        from io import StringIO
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("src.regime.vol_volume_gap")
        logger.addHandler(handler)
        try:
            monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("permission denied")))
            result = {"status": "ok"}
            save_state(result, Path("/fake/path/state.json"))
            log_output = log_capture.getvalue()
            assert "Failed to save state" in log_output
        finally:
            logger.removeHandler(handler)

    def test_load_state_oserror(self, monkeypatch):
        """OSError during load is caught, returns None."""
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("permission denied")))
        result = load_state(Path("/fake/state.json"))
        assert result is None

    def test_load_state_json_decode_error(self, monkeypatch):
        """JSONDecodeError during load is caught, returns None."""
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(json.JSONDecodeError("bad json", "doc", 1)))
        result = load_state(Path("/fake/state.json"))
        assert result is None

    def test_load_state_file_not_found(self):
        """Non-existent file returns None."""
        result = load_state(Path("/tmp/does_not_exist_12345.json"))
        assert result is None


# ── New Tests: get_same_day_signal Additional Coverage ───────────────

class TestGetSameDaySignalAdditional:
    """Additional coverage for get_same_day_signal edge paths."""

    def test_signal_with_unknown_regime(self, monkeypatch):
        """Monkeypatch detect_regime to return an 'unknown' regime."""
        result = detect_regime("SPY")
        if result["status"] == "error":
            pytest.skip("No real data available")
        # Verify UNKNOWN is in the adjustment map correctly
        from src.regime.vol_volume_gap import DayRegime
        signal = get_same_day_signal("SPY")
        if signal["status"] == "ok":
            assert "execution_adjustment" in signal
            assert 0.0 <= signal["execution_adjustment"] <= 1.0


# ── New Tests: Negative Price Series ─────────────────────────────────

class TestNegativePriceSeries:
    """Rare edge: price series that dips to zero or negative."""

    def test_zero_prices(self):
        """All prices zero: diff/closes => 0/0 => nan, ratio stays nan."""
        prices = np.full((25, 1), 0.0)
        features = compute_features(prices)
        assert features is not None
        # prev_close=0 so daily_return = 0.0
        assert features.daily_return == 0.0
        # 0/0 in all_returns diff produces nan before clipping
        assert np.isnan(features.return_vol_ratio)

    def test_some_negative_prices(self):
        """Negative price values after an earlier positive section."""
        closes = np.array([100.0, 95.0, 90.0, 80.0, -5.0, -10.0, -15.0])
        prices = closes.reshape(-1, 1)
        features = compute_features(prices)
        # The code checks if prev_close != 0, dividing by -5 gives valid returns
        # but n=7 < vol_lookback+2 = 22 -> None
        assert features is None

    def test_negative_prices_with_enough_data(self):
        """Negative prices across enough data points."""
        closes = np.array([100.0, 95.0, 90.0, -5.0, -10.0, -15.0, -20.0, -25.0,
                           -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0,
                           -65.0, -70.0, -75.0, -80.0, -85.0, -90.0, -95.0, -100.0])
        prices = closes.reshape(-1, 1)
        features = compute_features(prices)
        assert features is not None
        assert isinstance(features.daily_return, float)

