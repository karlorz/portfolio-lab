"""Tests for Maximum Diversification Portfolio (MDP).

TDD red phase — defines behavior before implementation.
MDP maximizes diversification ratio: weighted avg vol / portfolio vol.
"""

import pytest
import numpy as np
from src.strategy.max_diversification import compute_mdp_weights


# Test covariance matrix (3 assets)
COV_3 = np.array([
    [0.04, 0.005, 0.008],
    [0.005, 0.03, 0.006],
    [0.008, 0.006, 0.025],
])

SYMBOLS_3 = ["SPY", "GLD", "TLT"]

# Correlated assets — diversification should matter more
COV_CORR = np.array([
    [0.04, 0.02, 0.01],
    [0.02, 0.04, 0.01],
    [0.01, 0.01, 0.01],
])


class TestMaxDiversification:
    """Test suite for Maximum Diversification Portfolio."""

    def test_basic_computation(self):
        """compute_mdp_weights should return valid weights."""
        result = compute_mdp_weights(COV_3, SYMBOLS_3)
        assert "weights" in result
        assert "diversification_ratio" in result
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=0.01)

    def test_weights_non_negative(self):
        """All weights should be non-negative (long-only)."""
        result = compute_mdp_weights(COV_3, SYMBOLS_3)
        for w in result["weights"].values():
            assert w >= -0.001  # Small numerical tolerance

    def test_diversification_ratio_above_one(self):
        """DR should be >= 1 (equal to 1 only for perfectly correlated assets)."""
        result = compute_mdp_weights(COV_3, SYMBOLS_3)
        assert result["diversification_ratio"] >= 1.0

    def test_mdp_gives_lower_vol_asset_more_weight(self):
        """MDP should favor lower-volatility assets (they diversify better)."""
        result = compute_mdp_weights(COV_3, SYMBOLS_3)
        weights = result["weights"]
        # TLT has lowest vol (sqrt(0.025)=0.158), SPY highest (sqrt(0.04)=0.2)
        assert weights["TLT"] > weights["SPY"]

    def test_equal_cov_gives_equal_weights(self):
        """With equal vol and zero correlation, MDP should give equal weights."""
        n = 3
        cov_identity = np.eye(n) * 0.04  # All same vol
        symbols = ["A", "B", "C"]
        result = compute_mdp_weights(cov_identity, symbols)
        for w in result["weights"].values():
            assert w == pytest.approx(1.0 / n, abs=0.05)

    def test_higher_correlation_reduces_dr(self):
        """More correlated assets → lower diversification ratio."""
        # Uncorrelated
        cov_uncorr = np.diag([0.04, 0.04, 0.01])
        result_uncorr = compute_mdp_weights(cov_uncorr, SYMBOLS_3)
        # Highly correlated
        result_corr = compute_mdp_weights(COV_CORR, SYMBOLS_3)
        assert result_uncorr["diversification_ratio"] > result_corr["diversification_ratio"]

    def test_two_assets(self):
        """MDP should work with minimum 2 assets."""
        cov = np.array([[0.04, 0.01], [0.01, 0.01]])
        result = compute_mdp_weights(cov, ["SPY", "TLT"])
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=0.01)
        assert result["diversification_ratio"] >= 1.0

    def test_result_has_method_field(self):
        """Result should indicate optimization method."""
        result = compute_mdp_weights(COV_3, SYMBOLS_3)
        assert "method" in result
        assert result["method"] == "max_diversification"

    def test_shape_mismatch_raises(self):
        """Mismatched cov matrix and symbols should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            compute_mdp_weights(COV_3, ["SPY", "GLD"])  # 3x3 but 2 symbols

    def test_single_asset_raises(self):
        """Single asset portfolio should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_mdp_weights(np.array([[0.04]]), ["SPY"])
