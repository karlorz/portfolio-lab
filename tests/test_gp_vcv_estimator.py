"""Tests for src/monitor/gp_vcv_estimator.py — GP-VCV covariance estimation.

All tests use synthetic data (no real prices needed). Run only with:
    PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/test_gp_vcv_estimator.py --include-heavy -v
"""

import os
import sys
import numpy as np
import pytest

# Ensure ML is enabled for these tests
if os.environ.get("PORTFOLIO_LAB_ENABLE_ML") != "1":
    pytest.skip(
        "ML disabled — set PORTFOLIO_LAB_ENABLE_ML=1 to run GP-VCV tests",
        allow_module_level=True,
    )


@pytest.fixture
def multi_asset_returns():
    """Generate synthetic log-returns for 5 assets over 500 days."""
    rng = np.random.default_rng(42)
    n_assets, n_days = 5, 500
    # Realistic: correlated returns with regime structure
    base = rng.normal(0.0005, 0.01, n_days)  # market factor
    returns = np.zeros((n_days, n_assets))
    # Asset 0: SPY-like (high beta)
    returns[:, 0] = base * 1.2 + rng.normal(0, 0.008, n_days)
    # Asset 1: TLT-like (low/negative beta)
    returns[:, 1] = base * -0.3 + rng.normal(0, 0.006, n_days)
    # Asset 2: GLD-like (near zero beta)
    returns[:, 2] = base * 0.1 + rng.normal(0, 0.009, n_days)
    # Asset 3: IEF-like (low beta)
    returns[:, 3] = base * 0.2 + rng.normal(0, 0.004, n_days)
    # Asset 4: QQQ-like (high beta, high vol)
    returns[:, 4] = base * 1.5 + rng.normal(0, 0.012, n_days)
    return returns


@pytest.fixture
def gp_estimator(multi_asset_returns):
    """Create a GP-VCV estimator with default config."""
    from src.monitor.gp_vcv_estimator import GaussianProcessVCV
    return GaussianProcessVCV(lookback=252)


def test_module_imports_conditionally():
    """Verify sklearn import guard works. Test runs with ML enabled."""
    from src.monitor.gp_vcv_estimator import _HAS_SKLEARN, GaussianProcessVCV
    assert _HAS_SKLEARN is True
    assert GaussianProcessVCV is not None


def test_estimate_returns_valid_covariance(gp_estimator, multi_asset_returns):
    """GP-VCV should return a valid, PSD covariance matrix."""
    result = gp_estimator.estimate(multi_asset_returns)

    assert result.cov_matrix.shape == (5, 5)
    assert result.vol_estimates.shape == (5,)
    assert result.corr_matrix.shape == (5, 5)
    assert result.is_psd, "Matrix must be PSD, min eig was negative"
    assert result.condition_number > 0
    assert result.condition_number < 1000, \
        f"Condition number {result.condition_number} too high"


def test_vol_estimates_are_positive(gp_estimator, multi_asset_returns):
    """All volatility estimates must be positive."""
    result = gp_estimator.estimate(multi_asset_returns)
    assert np.all(result.vol_estimates > 0), \
        f"Got negative vol: {result.vol_estimates}"


def test_correlation_diagonal_is_one(gp_estimator, multi_asset_returns):
    """Correlation matrix diagonal must be exactly 1.0."""
    result = gp_estimator.estimate(multi_asset_returns)
    diag = np.diag(result.corr_matrix)
    np.testing.assert_allclose(diag, 1.0, atol=1e-10)


def test_vol_ordering_matches_input(gp_estimator, multi_asset_returns):
    """Asset 4 (QQQ-like) should have higher vol than Asset 3 (IEF-like)."""
    result = gp_estimator.estimate(multi_asset_returns)
    assert result.vol_estimates[4] > result.vol_estimates[3], \
        f"QQQ vol={result.vol_estimates[4]:.4f}, IEF vol={result.vol_estimates[3]:.4f}"


def test_ewma_comparison_runs(gp_estimator, multi_asset_returns):
    """GP vs EWMA comparison should return metrics for all assets."""
    result = gp_estimator.compare(multi_asset_returns)
    assert "gp_vols" in result
    assert "ewma_vols" in result
    assert "frobenius_diff" in result
    assert len(result["gp_vols"]) == 5
    assert result["frobenius_diff"] > 0  # they should differ


def test_state_save_load_roundtrip(gp_estimator, multi_asset_returns, tmp_path):
    """State should survive a save→load roundtrip."""
    gp_estimator.data_dir = tmp_path
    result = gp_estimator.estimate(multi_asset_returns)
    gp_estimator.save_state()
    state = gp_estimator.load_state()
    assert state is not None
    assert "cov_matrix" in state
    assert "vol_estimates" in state
    assert "condition_number" in state


def test_lookback_limits_data(gp_estimator, multi_asset_returns):
    """Estimator should respect lookback window."""
    gp_estimator.lookback = 126  # half a year
    result = gp_estimator.estimate(multi_asset_returns)
    assert result.lookback_days <= 126


def test_custom_asset_labels(gp_estimator, multi_asset_returns):
    """Asset labels should propagate to result."""
    labels = ["SPY", "GLD", "TLT", "IEF", "QQQ"]
    result = gp_estimator.estimate(multi_asset_returns, asset_labels=labels)
    assert result.asset_labels == labels


def test_single_asset_returns(gp_estimator):
    """Single asset should still work (edge case)."""
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0003, 0.01, (300, 1))
    result = gp_estimator.estimate(returns, asset_labels=["SPY"])
    assert result.cov_matrix.shape == (1, 1)
    assert result.vol_estimates[0] > 0
    assert result.corr_matrix[0, 0] == 1.0


def test_short_history_falls_back(gp_estimator):
    """Very short history should not crash."""
    rng = np.random.default_rng(456)
    returns = rng.normal(0.0003, 0.01, (30, 3))
    result = gp_estimator.estimate(returns)
    assert result.cov_matrix.shape == (3, 3)
    assert result.is_psd


def test_all_zero_returns(gp_estimator):
    """Flat zero returns should produce near-zero covariance (edge case)."""
    returns = np.zeros((200, 3))
    result = gp_estimator.estimate(returns)
    assert result.cov_matrix.shape == (3, 3)
    assert np.all(result.vol_estimates >= 0)


def test_covariance_matches_outer_product(gp_estimator, multi_asset_returns):
    """Cov[i,j] ≈ corr[i,j] * vol[i] * vol[j] (by construction)."""
    result = gp_estimator.estimate(multi_asset_returns)
    expected = result.corr_matrix * np.outer(
        result.vol_estimates, result.vol_estimates
    )
    np.testing.assert_allclose(result.cov_matrix, expected, rtol=1e-10)


def test_high_vol_asset_gets_higher_vol_estimate(gp_estimator):
    """Asset with intentionally amplified returns should get higher vol."""
    rng = np.random.default_rng(789)
    base = rng.normal(0.0003, 0.005, (252, 2))
    base[:, 1] = base[:, 1] * 3.0
    result = gp_estimator.estimate(base, asset_labels=["LowVol", "HighVol"])
    assert result.vol_estimates[1] > result.vol_estimates[0] * 1.5, \
        f"HighVol={result.vol_estimates[1]:.4f}, LowVol={result.vol_estimates[0]:.4f}"


def test_condition_number_reasonable(gp_estimator, multi_asset_returns):
    """Condition number should be under 500 for well-behaved data."""
    result = gp_estimator.estimate(multi_asset_returns)
    assert result.condition_number < 500, \
        f"Condition number {result.condition_number} is excessive"


def test_ewma_produces_valid_covariance():
    """EWMA baseline should return valid covariance."""
    from src.monitor.gp_vcv_estimator import GaussianProcessVCV

    rng = np.random.default_rng(999)
    returns = rng.normal(0.0005, 0.01, (504, 4))
    ewma_cov = GaussianProcessVCV.estimate_ewma(returns, halflife=42)
    assert ewma_cov.shape == (4, 4)
    eigvals = np.linalg.eigvalsh(ewma_cov)
    assert np.all(eigvals >= -1e-10)
    assert np.all(np.diag(ewma_cov) > 0)


def test_compare_returns_both_estimates(gp_estimator, multi_asset_returns):
    """Compare method returns both GP and EWMA covariance matrices."""
    result = gp_estimator.compare(multi_asset_returns)
    gp_cov = np.array(result["gp_cov"])
    ewma_cov = np.array(result["ewma_cov"])
    assert gp_cov.shape == ewma_cov.shape == (5, 5)
    assert result["gp_condition"] > 0
    assert result["ewma_condition"] > 0


def test_timestamp_is_iso_format(gp_estimator, multi_asset_returns):
    """Timestamp field should be valid ISO 8601."""
    result = gp_estimator.estimate(multi_asset_returns)
    from datetime import datetime
    try:
        datetime.fromisoformat(result.timestamp)
    except ValueError:
        pytest.fail(f"Invalid ISO timestamp: {result.timestamp}")


def test_nan_returns_handling(gp_estimator):
    """Returns with NaN values should either handle or raise clean error."""
    returns = np.random.default_rng(42).normal(0.0003, 0.01, (252, 3))
    returns[100:105, 1] = np.nan
    try:
        result = gp_estimator.estimate(returns)
        assert result.cov_matrix.shape == (3, 3)
    except (ValueError, RuntimeError) as e:
        msg = str(e).lower()
        assert any(kw in msg for kw in ("nan", "input", "contains"))
