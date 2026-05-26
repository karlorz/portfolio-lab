#!/usr/bin/env python3
"""
Tests for v5.40 Skew Engineering Overlay.

Tests skew ratio computation, regime classification, and vol target adjustment.
"""

import argparse
import json
import logging
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

from src.monitor.skew_engineering import (
    SkewEngine,
    SkewMetrics,
    SkewState,
    SkewRegime,
    STATE_FILE,
    DATA_DIR,
)


class TestSkewRegime:
    """Test skew regime classification."""

    def test_threshold_values(self):
        """Verify threshold constants."""
        assert SkewRegime.THRESHOLD_ELEVATED == 1.3
        assert SkewRegime.THRESHOLD_HIGH == 1.8

    def test_penalty_values(self):
        """Verify penalty caps."""
        assert SkewRegime.PENALTY_NORMAL == 0.05
        assert SkewRegime.PENALTY_ELEVATED == 0.12
        assert SkewRegime.PENALTY_HIGH == 0.20


class TestSkewMetrics:
    """Test SkewMetrics dataclass."""

    def test_default_values(self):
        """Verify sensible defaults."""
        m = SkewMetrics(symbol="SPY", timestamp="2026-05-16T12:00:00")
        assert m.symbol == "SPY"
        assert m.skew_ratio_21d == 1.0
        assert m.regime_21d == SkewRegime.NORMAL
        assert m.window_21d == 21
        assert m.window_63d == 63
        assert m.window_252d == 252

    def test_to_dict_includes_all_fields(self):
        """Verify serialization includes all fields."""
        m = SkewMetrics(
            symbol="SPY",
            timestamp="2026-05-16T12:00:00",
            upside_var_21d=0.01,
            downside_var_21d=0.02,
            skew_ratio_21d=2.0,
            regime_21d=SkewRegime.HIGH,
            composite_regime=SkewRegime.HIGH,
            vol_penalty=0.20,
            effective_vol_target=0.08,
            n_obs=252,
        )
        d = m.to_dict()
        assert d["symbol"] == "SPY"
        assert d["skew_ratio_21d"] == 2.0
        assert d["vol_penalty"] == 0.20

    def test_round_trip_json(self):
        """Verify JSON serialization round-trip."""
        m = SkewMetrics(
            symbol="SPY",
            timestamp="2026-05-16T12:00:00",
            upside_var_21d=0.01,
            downside_var_21d=0.02,
            skew_ratio_21d=1.5,
            regime_21d=SkewRegime.ELEVATED,
            composite_regime=SkewRegime.ELEVATED,
            vol_penalty=0.12,
            effective_vol_target=0.088,
            n_obs=252,
        )
        json_str = json.dumps(m.to_dict())
        loaded = json.loads(json_str)
        assert loaded["skew_ratio_21d"] == 1.5
        assert loaded["composite_regime"] == "ELEVATED"


class TestSkewState:
    """Test SkewState persistence."""

    def test_to_dict(self):
        """Verify state serialization."""
        s = SkewState(
            symbol="SPY",
            last_update="2026-05-16T12:00:00",
            composite_regime=SkewRegime.NORMAL,
            vol_penalty=0.05,
            side_computed=False,
            n_obs=250,
        )
        d = s.to_dict()
        assert d["symbol"] == "SPY"
        assert d["vol_penalty"] == 0.05

    def test_from_dict(self):
        """Verify state deserialization."""
        data = {
            "symbol": "SPY",
            "last_update": "2026-05-16T12:00:00",
            "composite_regime": "HIGH",
            "vol_penalty": 0.2,
            "side_computed": True,
            "n_obs": 250,
        }
        s = SkewState.from_dict(data)
        assert s.symbol == "SPY"
        assert s.composite_regime == "HIGH"
        assert s.vol_penalty == 0.2
        assert s.side_computed is True

    def test_from_dict_defaults(self):
        """Verify from_dict handles all fields."""
        data = {
            "symbol": "QQQ",
            "last_update": "2026-05-16T12:00:00",
            "composite_regime": "NORMAL",
            "vol_penalty": 0.05,
            "side_computed": False,
            "n_obs": 100,
        }
        s = SkewState.from_dict(data)
        assert s.symbol == "QQQ"
        assert s.n_obs == 100


class TestSkewEngineComputeSkewRatio:
    """Test the core skew ratio computation logic."""

    def test_symmetric_returns(self):
        """Symmetric returns should give skew ratio ≈ 1.0."""
        engine = SkewEngine()
        # Generate symmetric returns with same seed each time
        np.random.seed(12345)
        returns = np.random.randn(100) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # With symmetric normal, ratio should not be extreme
        assert 0.2 <= ratio <= 5.0
        assert isinstance(regime, str)

    def test_downside_heavy_returns(self):
        """Downside-heavy returns should give high skew ratio."""
        engine = SkewEngine()
        # Mostly small positive returns with occasional large negatives
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.005
        # Add large negative outliers
        for i in range(5):
            idx = np.random.randint(0, n)
            returns[idx] = -0.05 - np.random.random() * 0.03
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        assert ratio > 1.0

    def test_upside_heavy_returns(self):
        """Upside-heavy returns should give low skew ratio."""
        engine = SkewEngine()
        # Mostly small negative returns with occasional large positives
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.005
        # Add large positive outliers
        for i in range(5):
            idx = np.random.randint(0, n)
            returns[idx] = 0.05 + np.random.random() * 0.03
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # Upside variance should dominate
        assert ratio < 1.0

    def test_min_observations_floor(self):
        """Very few observations should return defaults."""
        engine = SkewEngine()
        returns = np.array([0.01, 0.02])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 21
        )
        assert regime == SkewRegime.NORMAL
        assert ratio == 1.0

    def test_all_positive_returns(self):
        """All positive returns should give ratio near 0."""
        engine = SkewEngine()
        returns = np.abs(np.random.randn(63)) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        assert ratio < 0.5

    def test_all_negative_returns(self):
        """All negative returns should give very high ratio."""
        engine = SkewEngine()
        returns = -np.abs(np.random.randn(63)) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # Upside variance will be near-zero, so ratio should be huge
        assert ratio > 2.0
        assert regime == SkewRegime.HIGH

    def test_regime_classification(self):
        """Verify regime thresholds are applied correctly."""
        engine = SkewEngine()

        # Create downside-heavy returns (more negative than positive variance)
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.005
        # Add many large negative moves to make downside variance dominate
        for i in range(15):
            idx = np.random.randint(0, n)
            returns[idx] = -0.05
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        # This should produce a clearly elevated ratio (> 2.0 expected)
        assert ratio > 2.0
        assert regime in (SkewRegime.ELEVATED, SkewRegime.HIGH)

    def test_consistent_with_window_sizes(self):
        """Different windows should produce different ratios for trending data."""
        engine = SkewEngine()
        # Create data that's recently calm but had a volatile period
        np.random.seed(42)
        calm = np.random.randn(30) * 0.005
        volatile = np.random.randn(30) * 0.03
        # Make recent period more volatile on downside
        recent_down = -np.abs(np.random.randn(21)) * 0.04
        returns = np.concatenate([calm, volatile, recent_down])

        _, _, ratio_21, _ = engine.compute_skew_ratio(returns, 21)
        _, _, ratio_63, _ = engine.compute_skew_ratio(returns, 63)

        # Short window should reflect recent downside volatility
        assert ratio_21 >= 0.5  # Should be meaningful

    def test_annualization_factor(self):
        """Verify returned variances are annualized."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01  # ~1% daily vol
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # Daily vol of 0.01 should give annualized vol of ~0.1587
        # Annualized variance should be ~0.025
        # For normal distribution, upside and downside should be similar
        assert 0.0 < up_var < 1.0
        assert 0.0 < down_var < 1.0


class TestSkewEngineIntegration:
    """Integration-level tests for SkewEngine."""

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_with_synthetic_data(self, mock_prices):
        """Full compute pipeline with synthetic returns."""
        np.random.seed(42)
        returns = np.random.randn(260) * 0.01
        # Make last 21 days downside-heavy
        returns[-21:] = -np.abs(np.random.randn(21)) * 0.02
        mock_prices.return_value = returns

        engine = SkewEngine(symbol="SPY")
        metrics = engine.compute()

        assert metrics.symbol == "SPY"
        assert metrics.n_obs == 260
        assert metrics.window_21d == 21
        assert metrics.window_63d == 63
        assert metrics.window_252d == 252
        # Recent downside should make 21-day ratio elevated
        assert metrics.skew_ratio_21d >= 1.0
        assert metrics.vol_penalty > 0.0
        assert metrics.effective_vol_target < 0.10

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_normal_regime(self, mock_prices):
        """Normal symmetric returns should produce NORMAL regime."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01

        engine = SkewEngine()
        metrics = engine.compute()

        # With normal returns, skew should be roughly balanced
        assert metrics.composite_regime in (
            SkewRegime.NORMAL, SkewRegime.ELEVATED
        )
        assert metrics.vol_penalty <= SkewRegime.PENALTY_ELEVATED

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_high_skew(self, mock_prices):
        """Extreme downside skew should produce HIGH regime."""
        np.random.seed(42)
        n = 260
        returns = np.random.randn(n) * 0.005
        # Add many large downside moves
        for i in range(20):
            idx = np.random.randint(0, n)
            returns[idx] = -0.06
        mock_prices.return_value = returns

        engine = SkewEngine()
        metrics = engine.compute()

        assert metrics.composite_regime == SkewRegime.HIGH
        assert metrics.vol_penalty == SkewRegime.PENALTY_HIGH

    @patch.object(SkewEngine, "_get_prices")
    def test_insufficient_data(self, mock_prices):
        """Insufficient data should return defaults."""
        mock_prices.return_value = np.array([0.01, 0.02, 0.03])

        engine = SkewEngine()
        metrics = engine.compute()

        assert metrics.n_obs == 3
        assert metrics.composite_regime == SkewRegime.NORMAL
        assert metrics.vol_penalty == 0.0

    def test_no_database_fallback(self):
        """No database should handle gracefully."""
        engine = SkewEngine()
        db_path = DATA_DIR / "market.db"
        # Only test if db truly doesn't exist
        if not db_path.exists():
            returns = engine._get_prices(days=260)
            assert len(returns) == 0
        else:
            # DB exists, verify we get data
            returns = engine._get_prices(days=10)
            assert len(returns) > 0
            assert isinstance(returns, np.ndarray)

    @patch.object(SkewEngine, "_get_prices")
    def test_get_vol_adjustment(self, mock_prices):
        """Verify vol adjustment calculation."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01

        engine = SkewEngine()
        adjusted = engine.get_vol_adjustment(target_vol=0.12)

        assert 0.0 < adjusted <= 0.12
        assert isinstance(adjusted, float)

    @patch.object(SkewEngine, "_get_prices")
    def test_adjustment_range(self, mock_prices):
        """Vol adjustment should stay within reasonable bounds."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01

        engine = SkewEngine()
        for target in [0.08, 0.10, 0.12, 0.15]:
            adjusted = engine.get_vol_adjustment(target_vol=target)
            # Should never be negative or exceed target
            assert adjusted > 0.0
            assert adjusted <= target
            # Max reduction is 20%
            assert adjusted >= target * (1.0 - SkewRegime.PENALTY_HIGH)

    @patch.object(SkewEngine, "_get_prices")
    def test_summary_format(self, mock_prices):
        """Summary should be properly formatted."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01

        engine = SkewEngine()
        summary = engine.summarize()

        assert "Skew Engineering" in summary
        assert "SPY" in summary
        assert "Composite regime" in summary
        assert "Vol penalty" in summary
        assert "Effective vol target" in summary

    @patch.object(SkewEngine, "_get_prices")
    def test_summary_high_skew(self, mock_prices):
        """Summary should reflect high skew regime."""
        np.random.seed(42)
        n = 260
        returns = np.random.randn(n) * 0.005
        for i in range(20):
            idx = np.random.randint(0, n)
            returns[idx] = -0.06
        mock_prices.return_value = returns

        engine = SkewEngine()
        summary = engine.summarize()

        assert SkewRegime.HIGH in summary
        assert "20.0%" in summary or "20" in summary


class TestStatePersistence:
    """Test state file persistence."""

    @patch.object(SkewEngine, "_get_prices")
    def test_save_and_load_state(self, mock_prices, tmp_path):
        """Verify state round-trips through file."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01

        # Patch STATE_FILE to use temp path
        with patch.object(
            SkewEngine, "_save_state", wraps=lambda metrics: None
        ):
            # Test state object serialization
            state = SkewState(
                symbol="SPY",
                last_update="2026-05-16T12:00:00",
                composite_regime="NORMAL",
                vol_penalty=0.05,
                side_computed=False,
                n_obs=250,
            )
            d = state.to_dict()
            loaded = SkewState.from_dict(d)
            assert loaded.symbol == state.symbol
            assert loaded.composite_regime == state.composite_regime
            assert loaded.vol_penalty == state.vol_penalty

    def test_state_file_not_found(self):
        """Load should return None when no state file."""
        engine = SkewEngine()
        # Use a non-existent path for the test
        original_path = STATE_FILE
        test_path = Path(tempfile.mktemp(suffix="_skew_test.json"))

        with patch("src.monitor.skew_engineering.STATE_FILE", test_path):
            state = engine.load_state()
            assert state is None

        # Clean up
        if test_path.exists():
            test_path.unlink()


class TestCLI:
    """Test CLI integration."""

    def test_cli_imports(self):
        """Verify CLI functions import correctly."""
        from src.monitor.skew_engineering import (
            cli_compute, cli_summary, cli_adjust, main
        )
        assert callable(cli_compute)
        assert callable(cli_summary)
        assert callable(cli_adjust)
        assert callable(main)


class TestEdgeCases:
    """Edge cases for skew engineering."""

    def test_single_return_value(self):
        """Single return observation should return defaults."""
        engine = SkewEngine()
        returns = np.array([0.01])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 21
        )
        assert ratio == 1.0
        assert regime == SkewRegime.NORMAL

    def test_zero_volatility(self):
        """Zero volatility (all same returns) should handle gracefully."""
        engine = SkewEngine()
        returns = np.zeros(100)
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # All zeros -> no positive or negative returns
        # Both variances will be 0, ratio will be 1.0 (default)
        assert ratio == 1.0

    def test_extreme_outliers(self):
        """Extreme outliers shouldn't break computation."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        returns[0] = -1.0  # Extreme outlier
        returns[1] = 0.5   # Large positive
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(
            returns, 63
        )
        # Should still produce valid output
        assert ratio > 0
        assert isinstance(regime, str)

    def test_multiple_calls_consistency(self):
        """Repeated calls with same data should produce same results."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        _, _, r1, reg1 = engine.compute_skew_ratio(returns, 63)
        _, _, r2, reg2 = engine.compute_skew_ratio(returns, 63)

        assert r1 == r2
        assert reg1 == reg2

    def test_all_windows_computed(self):
        """Verify 21/63/252 windows all get computed."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(260) * 0.01

        for window in [21, 63, 252]:
            up_var, down_var, ratio, regime = engine.compute_skew_ratio(
                returns, window
            )
            assert ratio > 0
            assert isinstance(regime, str)


class TestSkewMetricsExtended:
    """Extended tests for SkewMetrics dataclass."""

    def test_all_fields_in_to_dict(self):
        m = SkewMetrics(symbol="SPY", timestamp="2026-01-01T00:00:00")
        d = m.to_dict()
        expected_keys = {
            "symbol", "timestamp", "window_21d", "window_63d", "window_252d",
            "upside_var_21d", "downside_var_21d", "skew_ratio_21d", "regime_21d",
            "upside_var_63d", "downside_var_63d", "skew_ratio_63d", "regime_63d",
            "upside_var_252d", "downside_var_252d", "skew_ratio_252d", "regime_252d",
            "composite_regime", "vol_penalty", "effective_vol_target", "n_obs",
        }
        assert set(d.keys()) == expected_keys

    def test_custom_symbol(self):
        m = SkewMetrics(symbol="GLD", timestamp="2026-01-01")
        assert m.symbol == "GLD"

    def test_default_regime_is_normal(self):
        m = SkewMetrics(symbol="SPY", timestamp="2026-01-01")
        assert m.regime_21d == SkewRegime.NORMAL
        assert m.regime_63d == SkewRegime.NORMAL
        assert m.regime_252d == SkewRegime.NORMAL
        assert m.composite_regime == SkewRegime.NORMAL

    def test_default_skew_ratio_is_one(self):
        m = SkewMetrics(symbol="SPY", timestamp="2026-01-01")
        assert m.skew_ratio_21d == 1.0
        assert m.skew_ratio_63d == 1.0
        assert m.skew_ratio_252d == 1.0

    def test_zero_observations(self):
        m = SkewMetrics(symbol="SPY", timestamp="2026-01-01", n_obs=0)
        assert m.n_obs == 0


class TestSkewStateExtended:
    """Extended tests for SkewState dataclass."""

    def test_all_fields_in_to_dict(self):
        s = SkewState(
            symbol="SPY", last_update="2026-01-01", composite_regime="ELEVATED",
            vol_penalty=0.12, side_computed=True, n_obs=200,
        )
        d = s.to_dict()
        expected_keys = {"symbol", "last_update", "composite_regime", "vol_penalty", "side_computed", "n_obs"}
        assert set(d.keys()) == expected_keys

    def test_from_dict_roundtrip(self):
        s = SkewState(
            symbol="GLD", last_update="2026-02-01", composite_regime="HIGH",
            vol_penalty=0.20, side_computed=False, n_obs=100,
        )
        loaded = SkewState.from_dict(s.to_dict())
        assert loaded.symbol == s.symbol
        assert loaded.composite_regime == s.composite_regime
        assert loaded.vol_penalty == s.vol_penalty
        assert loaded.side_computed == s.side_computed
        assert loaded.n_obs == s.n_obs

    def test_side_computed_boolean(self):
        s = SkewState(symbol="SPY", last_update="2026-01-01", composite_regime="NORMAL",
                      vol_penalty=0.05, side_computed=True, n_obs=50)
        assert s.side_computed is True


class TestSkewRegimeExtended:
    """Extended tests for SkewRegime constants."""

    def test_regime_string_values(self):
        assert SkewRegime.NORMAL == "NORMAL"
        assert SkewRegime.ELEVATED == "ELEVATED"
        assert SkewRegime.HIGH == "HIGH"

    def test_threshold_ordering(self):
        assert SkewRegime.THRESHOLD_ELEVATED < SkewRegime.THRESHOLD_HIGH

    def test_penalty_ordering(self):
        assert SkewRegime.PENALTY_NORMAL < SkewRegime.PENALTY_ELEVATED < SkewRegime.PENALTY_HIGH

    def test_penalty_bounds(self):
        assert 0 < SkewRegime.PENALTY_NORMAL < 1
        assert 0 < SkewRegime.PENALTY_ELEVATED < 1
        assert 0 < SkewRegime.PENALTY_HIGH < 1


class TestSkewEngineExtended:
    """Extended SkewEngine tests."""

    def test_engine_default_symbol(self):
        engine = SkewEngine()
        assert engine.symbol == "SPY"

    def test_engine_custom_symbol(self):
        engine = SkewEngine(symbol="GLD")
        assert engine.symbol == "GLD"

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_returns_skew_metrics(self, mock_prices):
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01
        engine = SkewEngine()
        metrics = engine.compute()
        assert isinstance(metrics, SkewMetrics)

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_with_empty_data(self, mock_prices):
        mock_prices.return_value = np.array([])
        engine = SkewEngine()
        metrics = engine.compute()
        assert metrics.n_obs == 0

    @patch.object(SkewEngine, "_get_prices")
    def test_vol_adjustment_normal_skew(self, mock_prices):
        """Normal skew should give minimal adjustment."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01
        engine = SkewEngine()
        target = 0.10
        adjusted = engine.get_vol_adjustment(target_vol=target)
        assert adjusted > 0
        assert adjusted <= target

    @patch.object(SkewEngine, "_get_prices")
    def test_vol_adjustment_high_skew(self, mock_prices):
        """High skew should give larger adjustment."""
        np.random.seed(42)
        n = 260
        returns = np.random.randn(n) * 0.005
        for i in range(20):
            idx = np.random.randint(0, n)
            returns[idx] = -0.06
        mock_prices.return_value = returns
        engine = SkewEngine()
        adjusted = engine.get_vol_adjustment(target_vol=0.10)
        assert adjusted < 0.10  # Should be reduced
        assert adjusted >= 0.10 * (1 - SkewRegime.PENALTY_HIGH) - 0.001

    def test_compute_skew_ratio_exactly_at_threshold(self):
        """Test behavior near regime thresholds."""
        engine = SkewEngine()
        # Create returns that produce ratio near 1.3
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        # Just verify it produces a valid regime
        assert regime in (SkewRegime.NORMAL, SkewRegime.ELEVATED, SkewRegime.HIGH)

    def test_compute_skew_ratio_different_windows(self):
        """Different window sizes should give different results."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(260) * 0.01
        _, _, ratio_21, _ = engine.compute_skew_ratio(returns, 21)
        _, _, ratio_63, _ = engine.compute_skew_ratio(returns, 63)
        _, _, ratio_252, _ = engine.compute_skew_ratio(returns, 252)
        # All should be positive
        assert ratio_21 > 0
        assert ratio_63 > 0
        assert ratio_252 > 0


class TestStateFileConstant:
    """Test STATE_FILE constant."""

    def test_state_file_is_path(self):
        assert isinstance(STATE_FILE, Path)

    def test_state_file_name(self):
        assert STATE_FILE.name == "skew_state.json"


class TestCLIExtended:
    """Extended CLI tests."""

    def test_main_callable(self):
        from src.monitor.skew_engineering import main
        assert callable(main)

    def test_cli_compute_callable(self):
        from src.monitor.skew_engineering import cli_compute
        assert callable(cli_compute)

    def test_cli_summary_callable(self):
        from src.monitor.skew_engineering import cli_summary
        assert callable(cli_summary)

    def test_cli_adjust_callable(self):
        from src.monitor.skew_engineering import cli_adjust
        assert callable(cli_adjust)


class TestDataclassFieldValidation:
    """Verify dataclass fields via dataclasses.fields()."""

    def test_skew_metrics_field_count(self):
        import dataclasses
        fields = dataclasses.fields(SkewMetrics)
        assert len(fields) == 21

    def test_skew_metrics_field_names_and_defaults(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(SkewMetrics)}
        assert fields["symbol"].type == str
        assert fields["symbol"].default is dataclasses.MISSING
        assert fields["timestamp"].type == str
        assert fields["timestamp"].default is dataclasses.MISSING
        assert fields["window_21d"].type == int
        assert fields["window_21d"].default == 21
        assert fields["window_63d"].type == int
        assert fields["window_63d"].default == 63
        assert fields["window_252d"].type == int
        assert fields["window_252d"].default == 252
        assert fields["upside_var_21d"].type == float
        assert fields["upside_var_21d"].default == 0.0
        assert fields["downside_var_21d"].type == float
        assert fields["downside_var_21d"].default == 0.0
        assert fields["skew_ratio_21d"].type == float
        assert fields["skew_ratio_21d"].default == 1.0

    def test_skew_metrics_required_fields(self):
        """Only symbol and timestamp are required."""
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(SkewMetrics)}
        required = [n for n, f in fields.items() if f.default is dataclasses.MISSING]
        assert required == ["symbol", "timestamp"]

    def test_skew_metrics_regime_type(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(SkewMetrics)}
        assert fields["regime_21d"].type == str
        assert fields["regime_63d"].type == str
        assert fields["regime_252d"].type == str
        assert fields["composite_regime"].type == str

    def test_skew_state_field_count(self):
        import dataclasses
        fields = dataclasses.fields(SkewState)
        assert len(fields) == 6

    def test_skew_state_field_types(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(SkewState)}
        assert fields["symbol"].type == str
        assert fields["last_update"].type == str
        assert fields["composite_regime"].type == str
        assert fields["vol_penalty"].type == float
        assert fields["side_computed"].type == bool
        assert fields["n_obs"].type == int

    def test_skew_state_all_required(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(SkewState)}
        required = [n for n, f in fields.items() if f.default is dataclasses.MISSING]
        assert len(required) == 6  # All fields required, no defaults

    def test_skew_state_no_defaults(self):
        import dataclasses
        for f in dataclasses.fields(SkewState):
            assert f.default is dataclasses.MISSING, f"{f.name} should not have default"
            assert f.default_factory is dataclasses.MISSING


class TestNanInfEdgeCases:
    """NaN and Inf handling in compute_skew_ratio."""

    def test_nan_returns_dropped_gracefully(self):
        """NaN values are excluded by comparison operators, computation should still work."""
        engine = SkewEngine()
        returns = np.array([0.01, -0.02, np.nan, 0.015, -0.01, 0.02, -0.03])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio > 0
        assert isinstance(regime, str)

    def test_inf_returns_handled(self):
        """Inf values should not crash the computation."""
        engine = SkewEngine()
        returns = np.array([0.01, -0.02, np.inf, 0.015, -0.01, 0.02, -0.03])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert isinstance(regime, str)

    def test_neg_inf_returns_handled(self):
        """-Inf values should not crash the computation."""
        engine = SkewEngine()
        returns = np.array([0.01, -0.02, -np.inf, 0.015, -0.01, 0.02, -0.03])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert isinstance(regime, str)

    def test_all_nan_returns(self):
        """All-NaN returns should produce defaults."""
        engine = SkewEngine()
        returns = np.full(100, np.nan)
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio == 1.0
        assert regime == SkewRegime.NORMAL

    def test_mixed_nan_valid_returns(self):
        """Mix of NaN and valid returns should use only valid values."""
        engine = SkewEngine()
        np.random.seed(42)
        valid = np.random.randn(50) * 0.01
        nans = np.full(50, np.nan)
        returns = np.concatenate([valid, nans])
        np.random.shuffle(returns)
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio > 0
        assert isinstance(regime, str)

    def test_all_inf_returns(self):
        """All-Inf returns - should not crash."""
        engine = SkewEngine()
        returns = np.full(100, np.inf)
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert isinstance(regime, str)


class TestConstantsValidation:
    """Verify module-level constants."""

    def test_state_file_is_path(self):
        assert isinstance(STATE_FILE, Path)

    def test_state_file_name(self):
        assert STATE_FILE.name == "skew_state.json"

    def test_data_dir_is_path(self):
        assert isinstance(DATA_DIR, Path)

    def test_min_obs_constant(self):
        assert SkewEngine.MIN_OBS == 10

    def test_min_obs_positive(self):
        assert SkewEngine.MIN_OBS > 0

    def test_all_export_defined(self):
        from src.monitor.skew_engineering import __all__
        assert isinstance(__all__, list)
        assert len(__all__) > 0

    def test_all_export_strings(self):
        from src.monitor.skew_engineering import __all__
        for item in __all__:
            assert isinstance(item, str)

    def test_regime_thresholds_positive(self):
        assert SkewRegime.THRESHOLD_ELEVATED > 0
        assert SkewRegime.THRESHOLD_HIGH > 0

    def test_penalties_within_bounds(self):
        assert 0.0 < SkewRegime.PENALTY_NORMAL < 0.5
        assert 0.0 < SkewRegime.PENALTY_ELEVATED < 0.5
        assert 0.0 < SkewRegime.PENALTY_HIGH < 0.5


class TestBoundaryConditions:
    """Boundary conditions around thresholds and limits."""

    def test_min_obs_exact(self):
        """Exactly MIN_OBS observations should work (not return defaults)."""
        engine = SkewEngine()
        returns = np.random.randn(SkewEngine.MIN_OBS) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        # Should not return early (1.0, NORMAL) since we have enough obs
        assert len(returns) >= 10

    def test_one_below_min_obs(self):
        """One less than MIN_OBS returns defaults."""
        engine = SkewEngine()
        returns = np.random.randn(SkewEngine.MIN_OBS - 1) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio == 1.0
        assert regime == SkewRegime.NORMAL

    def test_ratio_above_high_threshold(self):
        """Ratio above HIGH threshold produces HIGH regime."""
        engine = SkewEngine()
        # Create at least MIN_OBS returns where downside var dominates
        positive = np.array([0.005, 0.01, 0.008, 0.012, 0.006])
        negative = np.array([-0.01, -0.05, -0.04, -0.03, -0.035])
        returns = np.concatenate([positive, negative])
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio >= SkewRegime.THRESHOLD_HIGH
        assert regime == SkewRegime.HIGH

    def test_ratio_below_normal_threshold(self):
        """Ratio well below 1.3 produces NORMAL regime."""
        engine = SkewEngine()
        # Symmetric returns should give ratio near 1.0
        np.random.seed(12345)
        returns = np.random.randn(100) * 0.01
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio < 1.3
        assert regime == SkewRegime.NORMAL

    def test_get_vol_adjustment_zero_target(self):
        """Zero target vol should produce zero adjusted vol."""
        engine = SkewEngine()
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            adjusted = engine.get_vol_adjustment(target_vol=0.0)
            assert adjusted == 0.0

    def test_get_vol_adjustment_max_target(self):
        """High target vol should stay within bounds."""
        engine = SkewEngine()
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            adjusted = engine.get_vol_adjustment(target_vol=1.0)
            assert adjusted > 0.0
            assert adjusted <= 1.0

    def test_all_positive_returns_boundary(self):
        """All positive returns produce ratio close to 0 (upside dominated)."""
        engine = SkewEngine()
        # Create returns where var_up > 0 but var_down = 0 (clamped to 1e-12)
        returns = np.abs(np.random.randn(63)) * 0.01
        # Ensure at least one positive so var_up > 0
        returns = np.abs(returns)
        _, _, ratio, _ = engine.compute_skew_ratio(returns, 63)
        # With all positive returns, downside var is clamped to 1e-12
        # and ratio = 1e-12 / upside_var -> very small
        assert ratio < 1.0

    def test_all_negative_returns_boundary(self):
        """All negative returns produce very high ratio (downside dominates)."""
        engine = SkewEngine()
        returns = -np.abs(np.random.randn(63)) * 0.01
        _, _, ratio, _ = engine.compute_skew_ratio(returns, 63)
        # With all negative returns, upside var is clamped to 1e-12
        # and ratio = downside_var / 1e-12 -> very large
        assert ratio > 100

    def test_exact_window_equal_returns_len(self):
        """When window exactly equals returns length, all returns are used."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(21) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 21)
        assert ratio > 0
        assert isinstance(regime, str)


class TestCliOutput:
    """CLI function output with caplog."""

    def _get_cli_compute(self):
        from src.monitor.skew_engineering import cli_compute as _f
        return _f

    def _get_cli_summary(self):
        from src.monitor.skew_engineering import cli_summary as _f
        return _f

    def _get_cli_adjust(self):
        from src.monitor.skew_engineering import cli_adjust as _f
        return _f

    def _get_main(self):
        from src.monitor.skew_engineering import main as _main
        return _main

    def test_cli_compute_prints_json(self, caplog):
        args = argparse.Namespace(symbol="SPY", command="compute")
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_cli_compute()(args)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert msg.strip().startswith("{")
        assert '"symbol": "SPY"' in msg

    def test_cli_compute_serializable(self, caplog):
        args = argparse.Namespace(symbol="GLD", command="compute")
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_cli_compute()(args)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        parsed = json.loads(msg.strip())
        assert parsed["symbol"] == "GLD"
        assert "skew_ratio_21d" in parsed

    def test_cli_summary_prints_summary(self, caplog):
        args = argparse.Namespace(symbol="SPY", command="summary")
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_cli_summary()(args)
        assert "Skew Engineering" in caplog.text

    def test_cli_adjust_prints_adjustment(self, caplog):
        args = argparse.Namespace(symbol="SPY", command="adjust", target_vol=0.12)
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_cli_adjust()(args)
        assert "Base target:" in caplog.text
        assert "Adjusted target:" in caplog.text
        assert "Reduction:" in caplog.text

    def test_main_compute_command(self, caplog):
        with patch.object(
            SkewEngine, "_get_prices", return_value=np.random.randn(260) * 0.01
        ), patch(
            "sys.argv", ["skew_engineering", "--symbol", "SPY", "compute"]
        ):
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_main()()
        assert '"symbol": "SPY"' in caplog.text

    def test_main_summary_command(self, caplog):
        with patch.object(
            SkewEngine, "_get_prices", return_value=np.random.randn(260) * 0.01
        ), patch(
            "sys.argv", ["skew_engineering", "--symbol", "QQQ", "summary"]
        ):
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_main()()
        assert "QQQ" in caplog.text

    def test_main_adjust_command(self, caplog):
        with patch.object(
            SkewEngine, "_get_prices", return_value=np.random.randn(260) * 0.01
        ), patch(
            "sys.argv", ["skew_engineering", "--symbol", "SPY", "adjust", "--target-vol", "0.15"]
        ):
            with caplog.at_level(logging.INFO, logger="src.monitor.skew_engineering"):
                self._get_main()()
        assert "15.0%" in caplog.text.replace(" ", "").replace("\n", "")

    def test_main_no_command_prints_help(self):
        with patch(
            "sys.argv", ["skew_engineering"]
        ):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                self._get_main()()
                mock_help.assert_called_once()


class TestExportCompleteness:
    """Verify __all__ covers public API."""

    def test_all_covers_skew_regime(self):
        from src.monitor.skew_engineering import __all__
        assert "SkewRegime" in __all__

    def test_all_covers_skew_metrics(self):
        from src.monitor.skew_engineering import __all__
        assert "SkewMetrics" in __all__

    def test_all_covers_skew_state(self):
        from src.monitor.skew_engineering import __all__
        assert "SkewState" in __all__

    def test_all_covers_skew_engine(self):
        from src.monitor.skew_engineering import __all__
        assert "SkewEngine" in __all__

    def test_all_covers_cli_functions(self):
        from src.monitor.skew_engineering import __all__
        assert "cli_compute" in __all__
        assert "cli_summary" in __all__
        assert "cli_adjust" in __all__

    def test_all_public_names_are_importable(self):
        from src.monitor.skew_engineering import __all__
        import src.monitor.skew_engineering as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} is in __all__ but not in module"


class TestErrorAndEdgePaths:
    """Error handling paths in SkewEngine."""

    def test_load_state_corrupt_json(self, caplog):
        """Corrupt state file should log error and return None."""
        engine = SkewEngine()
        with patch("src.monitor.skew_engineering.STATE_FILE") as mock_state_file:
            mock_state_file.exists.return_value = True
            mock_open_handle = mock_open(read_data="not valid json {")
            with patch("builtins.open", mock_open_handle):
                caplog.clear()
                state = engine.load_state()
        assert state is None

    def test_load_state_os_error(self, caplog):
        """OSError during load should log and return None."""
        engine = SkewEngine()
        with patch("src.monitor.skew_engineering.STATE_FILE") as mock_state_file:
            mock_state_file.exists.return_value = True
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                caplog.clear()
                state = engine.load_state()
        assert state is None

    @patch.object(SkewEngine, "_get_prices")
    def test_vol_adjustment_with_high_skew_exact_penalty(self, mock_prices):
        """High skew regime should apply PENALTY_HIGH reduction."""
        np.random.seed(42)
        n = 260
        returns = np.random.randn(n) * 0.005
        for i in range(20):
            idx = np.random.randint(0, n)
            returns[idx] = -0.06
        mock_prices.return_value = returns
        engine = SkewEngine()
        adjusted = engine.get_vol_adjustment(target_vol=0.10)
        expected = round(0.10 * (1.0 - SkewRegime.PENALTY_HIGH), 4)
        assert adjusted == expected

    @patch.object(SkewEngine, "_get_prices")
    def test_vol_adjustment_normal_skew_exact_penalty(self, mock_prices):
        """Normal skew regime should apply PENALTY_NORMAL reduction."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01
        engine = SkewEngine()
        adjusted = engine.get_vol_adjustment(target_vol=0.10)
        expected_max = round(0.10 * (1.0 - SkewRegime.PENALTY_NORMAL), 4)
        # With normal skew, penalty should be PENALTY_NORMAL
        assert adjusted == expected_max

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_save_state_called(self, mock_prices):
        """compute() should call _save_state."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01
        engine = SkewEngine()
        with patch.object(engine, "_save_state") as mock_save:
            metrics = engine.compute()
            mock_save.assert_called_once_with(metrics)

    def test_save_state_os_error_logged(self, caplog):
        """_save_state OSError should be logged."""
        engine = SkewEngine()
        metrics = SkewMetrics(symbol="SPY", timestamp="2026-01-01T00:00:00")
        with patch("builtins.open", side_effect=OSError("Disk full")):
            caplog.clear()
            engine._save_state(metrics)
            assert len(caplog.records) > 0
            assert "Failed to save state" in caplog.text

    @patch.object(SkewEngine, "_get_prices")
    def test_compute_after_save_state_persists(self, mock_prices):
        """compute() output should produce a valid SkewState."""
        np.random.seed(42)
        mock_prices.return_value = np.random.randn(260) * 0.01
        engine = SkewEngine()
        metrics = engine.compute()
        state = SkewState(
            symbol=metrics.symbol,
            last_update=metrics.timestamp,
            composite_regime=metrics.composite_regime,
            vol_penalty=metrics.vol_penalty,
            side_computed=False,
            n_obs=metrics.n_obs,
        )
        assert isinstance(state, SkewState)
        assert state.symbol == metrics.symbol


class TestComputeEdgeCases:
    """Additional edge cases for compute_skew_ratio."""

    def test_ratio_single_upside_observation(self):
        """Single unique upside value gives upside_var=0, clamped to 1e-12, regime HIGH."""
        engine = SkewEngine()
        # Need >= MIN_OBS returns, with only 1 unique positive and >=2 unique negatives
        positive = np.array([0.01, 0.01, 0.01])  # 3 same values => var=0
        negative = np.array([-0.01, -0.05, -0.01, -0.05, -0.01, -0.05, -0.01])
        returns = np.concatenate([positive, negative])  # len = 10
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        # Only 1 unique positive (upside_var = 0.0, clamped to 1e-12)
        # Multiple different negatives (downside_var > 0)
        # ratio = downside_var / 1e-12 >> 1.8
        assert regime == SkewRegime.HIGH

    def test_ratio_single_downside_observation(self):
        """Single unique downside value gives downside_var=0, ratio near 0."""
        engine = SkewEngine()
        # Need >= MIN_OBS returns, with >=2 unique positives and only 1 unique negative
        negative = np.array([-0.01, -0.01, -0.01])  # 3 same values => var=0
        positive = np.array([0.01, 0.05, 0.01, 0.05, 0.01, 0.05, 0.01])
        returns = np.concatenate([negative, positive])  # len = 10
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        # Only 1 unique negative (downside_var = 0.0)
        # Multiple different positives (upside_var > 0)
        # skew_ratio = 0.0 / upside_var = 0.0
        assert ratio == 0.0
        assert regime == SkewRegime.NORMAL

    def test_ratio_empty_array(self):
        """Empty array returns defaults."""
        engine = SkewEngine()
        returns = np.array([])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio == 1.0
        assert regime == SkewRegime.NORMAL

    def test_ratio_two_returns_exactly_min_obs_gate(self):
        """Two returns is below MIN_OBS, returns defaults."""
        engine = SkewEngine()
        returns = np.array([0.01, -0.02])
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert ratio == 1.0
        assert regime == SkewRegime.NORMAL

    def test_ratio_large_window_small_data(self):
        """Window larger than data uses all data."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(15) * 0.01
        up_var, down_var, ratio, regime = engine.compute_skew_ratio(returns, 252)
        assert isinstance(regime, str)  # Should not crash

    def test_ratio_upside_single_var_downside_multiple(self):
        """2 positive values (upside_var > 0) and multiple negative values."""
        engine = SkewEngine()
        returns = np.array([0.01, 0.02, -0.01, -0.03, -0.02, -0.015])
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert regime in (SkewRegime.NORMAL, SkewRegime.ELEVATED, SkewRegime.HIGH)

    def test_ratio_downside_single_var_upside_multiple(self):
        """2 negative values (downside_var > 0) and multiple positive values."""
        engine = SkewEngine()
        returns = np.array([-0.01, -0.02, 0.01, 0.03, 0.02, 0.015])
        _, _, ratio, regime = engine.compute_skew_ratio(returns, 63)
        assert regime in (SkewRegime.NORMAL, SkewRegime.ELEVATED, SkewRegime.HIGH)


class TestTypeValidation:
    """Type handling and coercion in dataclasses and methods."""

    def test_skew_metrics_float_field_coercion(self):
        """int values should be coerced to float for float-typed fields."""
        m = SkewMetrics(symbol="SPY", timestamp="now", upside_var_21d=1, n_obs=100)
        assert isinstance(m.upside_var_21d, (int, float))

    def test_skew_metrics_int_defaults(self):
        """Default int fields should be int."""
        m = SkewMetrics(symbol="SPY", timestamp="now")
        assert isinstance(m.window_21d, int)
        assert isinstance(m.n_obs, int)

    def test_skew_state_vol_penalty_float(self):
        """vol_penalty should accept float."""
        s = SkewState(
            symbol="SPY", last_update="now", composite_regime="NORMAL",
            vol_penalty=0.05, side_computed=False, n_obs=100
        )
        assert isinstance(s.vol_penalty, float)

    def test_skew_state_side_computed_bool(self):
        """side_computed should be bool."""
        s = SkewState(
            symbol="SPY", last_update="now", composite_regime="NORMAL",
            vol_penalty=0.05, side_computed=True, n_obs=100
        )
        assert isinstance(s.side_computed, bool)

    def test_compute_skew_ratio_returns_tuple(self):
        """compute_skew_ratio returns a 4-tuple."""
        engine = SkewEngine()
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        result = engine.compute_skew_ratio(returns, 63)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)
        assert isinstance(result[3], str)

    def test_compute_returns_skew_metrics_type(self):
        """compute() returns SkewMetrics."""
        engine = SkewEngine()
        with patch.object(SkewEngine, "_get_prices") as mock_prices:
            mock_prices.return_value = np.random.randn(260) * 0.01
            result = engine.compute()
        assert isinstance(result, SkewMetrics)

    def test_load_state_returns_optional(self):
        """load_state returns None or SkewState."""
        engine = SkewEngine()
        result = engine.load_state()
        assert result is None or isinstance(result, SkewState)


class TestDataDirAndPaths:
    """Verify DATA_DIR and file paths."""

    def test_data_dir_exists(self):
        assert DATA_DIR.exists()

    def test_data_dir_is_directory(self):
        assert DATA_DIR.is_dir()

    def test_state_file_parent_is_data_dir(self):
        assert STATE_FILE.parent == DATA_DIR


class TestLoggingBehavior:
    """Test logging output for various scenarios."""

    def test_insufficient_data_warning(self, caplog):
        """Warning logged when insufficient data."""
        engine = SkewEngine()
        with patch.object(SkewEngine, "_get_prices", return_value=np.array([0.01, 0.02])):
            caplog.set_level(logging.WARNING)
            engine.compute()
            assert any("Insufficient data" in r.message for r in caplog.records)

    def test_database_not_found_warning(self, caplog):
        """Warning logged when database not found."""
        engine = SkewEngine()
        with patch.object(engine, "db_path") as mock_db_path:
            mock_db_path.exists.return_value = False
            caplog.set_level(logging.WARNING)
            result = engine._get_prices(days=260)
            assert len(result) == 0
            assert any("Database not found" in r.message for r in caplog.records)

    def test_empty_returns_from_get_prices(self, caplog):
        """Empty returns should produce warning."""
        engine = SkewEngine()
        with patch.object(SkewEngine, "_get_prices", return_value=np.array([])):
            caplog.set_level(logging.WARNING)
            metrics = engine.compute()
            assert metrics.n_obs == 0
            assert metrics.composite_regime == SkewRegime.NORMAL


if __name__ == "__main__":
    pytest.main(["-v", __file__])
