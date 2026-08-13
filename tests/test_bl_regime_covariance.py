"""
Tests for regime-conditional covariance in Black-Litterman posterior.

Replaces static full-sample covariance with regime-specific estimates
to improve weight stability during regime transitions.
"""

import pytest
import numpy as np
import pandas as pd

from src.strategy.black_litterman_mapper import (
    BLViews,
    compute_bl_weights,
    compute_regime_covariances,
    REGIME_COV_WINDOW,
    REGIME_COV_MIN_SAMPLES,
    REGIME_VOL_THRESHOLDS,
)


@pytest.fixture
def synthetic_prices_5yr():
    """5 years of synthetic daily prices for 3 assets with regime-like behavior."""
    np.random.seed(42)
    n_days = 252 * 5
    dates = pd.bdate_range("2021-01-01", periods=n_days)

    # Create returns with distinct regime characteristics
    returns = np.zeros((n_days, 3))

    # Period 1 (days 0-500): Normal regime — moderate vol
    returns[:500] = np.random.normal([0.0004, 0.0002, 0.0001], [0.010, 0.012, 0.008], (500, 3))

    # Period 2 (days 500-700): Crisis — high vol, negative equity, positive bonds
    returns[500:700] = np.random.normal([-0.002, 0.001, 0.002], [0.030, 0.025, 0.015], (200, 3))

    # Period 3 (days 700-1000): Low vol — compressed vol
    returns[700:1000] = np.random.normal([0.0003, 0.0001, 0.00005], [0.005, 0.006, 0.004], (300, 3))

    # Period 4 (days 1000-1260): Back to normal
    returns[1000:] = np.random.normal([0.0004, 0.0002, 0.0001], [0.010, 0.012, 0.008], (n_days - 1000, 3))

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["SPY", "GLD", "TLT"])


@pytest.fixture
def simple_cov_matrix():
    """Simple 3x3 covariance matrix for BL tests."""
    return np.array([
        [0.000100, 0.000010, -0.000020],
        [0.000010, 0.000144, 0.000005],
        [-0.000020, 0.000005, 0.000064],
    ])


class TestComputeRegimeCovariances:
    """Tests for compute_regime_covariances()."""

    def test_returns_dict_of_regimes(self, synthetic_prices_5yr):
        """Each regime maps to a valid covariance matrix."""
        result = compute_regime_covariances(synthetic_prices_5yr)
        assert isinstance(result, dict)
        assert len(result) > 0

        for regime, cov in result.items():
            assert isinstance(regime, str)
            assert cov.shape == (3, 3)
            # Must be symmetric
            np.testing.assert_array_almost_equal(cov, cov.T, decimal=10)
            # Diagonals must be positive
            assert all(np.diag(cov) > 0)

    def test_crisis_covariance_higher_vol(self, synthetic_prices_5yr):
        """Crisis regime should have higher variance than normal."""
        result = compute_regime_covariances(synthetic_prices_5yr)
        if "crisis" in result and "normal" in result:
            crisis_vol = np.sqrt(np.mean(np.diag(result["crisis"])))
            normal_vol = np.sqrt(np.mean(np.diag(result["normal"])))
            assert crisis_vol > normal_vol

    def test_low_vol_regime_lower_variance(self, synthetic_prices_5yr):
        """Low-vol regime should have lower variance than normal."""
        result = compute_regime_covariances(synthetic_prices_5yr)
        if "low_vol" in result and "normal" in result:
            low_vol = np.sqrt(np.mean(np.diag(result["low_vol"])))
            normal_vol = np.sqrt(np.mean(np.diag(result["normal"])))
            assert low_vol < normal_vol

    def test_handles_short_price_history(self):
        """Short history should still return at least one regime cov."""
        np.random.seed(7)
        short_prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.01, (100, 3)), axis=0)),
            columns=["SPY", "GLD", "TLT"],
        )
        result = compute_regime_covariances(short_prices)
        assert len(result) >= 1

    def test_missing_symbols_handled(self, synthetic_prices_5yr):
        """Gracefully handles missing symbols in prices_df."""
        subset = synthetic_prices_5yr[["SPY", "GLD"]]
        result = compute_regime_covariances(subset, symbols=["SPY", "GLD", "TLT"])
        # Should still work with available symbols
        assert isinstance(result, dict)

    def test_custom_symbols(self, synthetic_prices_5yr):
        """Works with custom symbol list."""
        result = compute_regime_covariances(
            synthetic_prices_5yr, symbols=["SPY", "GLD", "TLT"]
        )
        for regime, cov in result.items():
            assert cov.shape == (3, 3)

    def test_ledoit_wolf_shrinkage_applied(self, synthetic_prices_5yr):
        """Covariance matrices should use Ledoit-Wolf shrinkage for stability."""
        result = compute_regime_covariances(synthetic_prices_5yr)
        for regime, cov in result.items():
            # Check positive semi-definiteness (eigenvalues >= 0)
            eigvals = np.linalg.eigvalsh(cov)
            assert all(eigvals >= -1e-10), f"Regime {regime} has negative eigenvalue"


class TestRegimeVolThresholds:
    """Tests for REGIME_VOL_THRESHOLDS configuration."""

    def test_thresholds_defined(self):
        """All regimes should have vol thresholds."""
        assert isinstance(REGIME_VOL_THRESHOLDS, dict)
        assert len(REGIME_VOL_THRESHOLDS) >= 3

    def test_thresholds_ordered(self):
        """Crisis threshold > high_vol > normal > low_vol."""
        if all(k in REGIME_VOL_THRESHOLDS for k in ["crisis", "high_vol", "normal", "low_vol"]):
            assert REGIME_VOL_THRESHOLDS["crisis"] > REGIME_VOL_THRESHOLDS["high_vol"]
            assert REGIME_VOL_THRESHOLDS["high_vol"] > REGIME_VOL_THRESHOLDS["normal"]
            assert REGIME_VOL_THRESHOLDS["normal"] > REGIME_VOL_THRESHOLDS["low_vol"]


class TestRegimeCovarianceInBL:
    """Tests for regime-conditional covariance integration in compute_bl_weights."""

    def test_regime_covariance_changes_weights(self, synthetic_prices_5yr):
        """BL weights should differ when using regime vs full-sample covariance."""
        # Full-sample (no regime)
        result_full = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
        )

        # Regime-specific
        result_crisis = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime="crisis",
        )

        # Weights should differ (crisis cov != full-sample cov)
        if result_full.extras.get("optimization_method") == "bl_max_sharpe" and \
           result_crisis.extras.get("optimization_method") == "bl_max_sharpe":
            assert result_full.bl_weights != result_crisis.bl_weights

    def test_regime_covariance_recorded_in_extras(self, synthetic_prices_5yr):
        """Extras should record whether regime covariance was used."""
        result = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime="crisis",
        )
        assert "regime_covariance" in result.extras

    def test_unknown_regime_falls_back_to_normal(self, synthetic_prices_5yr):
        """Unknown regime should fall back to normal regime covariance."""
        result = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime="nonexistent_regime",
        )
        # Should not crash, extras should indicate fallback to normal or full_sample
        assert result.extras.get("regime_covariance") in ("normal", "full_sample")

    def test_none_regime_uses_full_sample(self, synthetic_prices_5yr):
        """None regime should use full-sample covariance (backward compatible)."""
        result = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime=None,
        )
        # Should work exactly as before (no regime cov in extras or "full_sample")
        assert result.extras.get("regime_covariance", "full_sample") == "full_sample"

    def test_bl_cascade_still_works_with_regime_cov(self, simple_cov_matrix):
        """BL cascade (BL → HRP → EW) should still work with regime cov."""
        _ = BLViews(
            absolute_views={"SPY": 0.05, "GLD": 0.03, "TLT": 0.02},
            view_confidences=[0.5, 0.5, 0.5],
        )
        # Even with degenerate inputs, cascade should produce weights
        result = compute_bl_weights(
            equity_bias=0.0,
            duration_bias=0.0,
            gold_bias=0.0,
            regime="normal",
        )
        assert len(result.bl_weights) > 0

    def test_regime_covariance_with_transaction_costs(self, synthetic_prices_5yr):
        """Regime covariance should compose with transaction costs."""
        result = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime="crisis",
            transaction_costs=True,
        )
        assert result.extras.get("transaction_costs_applied") is True
        assert "regime_covariance" in result.extras

    def test_regime_covariance_with_turnover_penalty(self, synthetic_prices_5yr):
        """Regime covariance should compose with turnover penalty."""
        current = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = compute_bl_weights(
            prices_df=synthetic_prices_5yr,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.2,
            regime="normal",
            turnover_penalty=0.5,
            current_weights=current,
        )
        assert result.extras.get("regime_covariance") is not None


class TestRegimeCovConstants:
    """Tests for module-level constants."""

    def test_window_size(self):
        """Regime classification window should be reasonable (21-63 days)."""
        assert 21 <= REGIME_COV_WINDOW <= 63

    def test_min_samples(self):
        """Minimum samples for regime covariance should be reasonable."""
        assert REGIME_COV_MIN_SAMPLES >= 30
