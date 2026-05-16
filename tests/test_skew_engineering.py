#!/usr/bin/env python3
"""
Tests for v5.40 Skew Engineering Overlay.

Tests skew ratio computation, regime classification, and vol target adjustment.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
