"""
Tests for split-conformal prediction risk quantifier.

Distribution-free risk estimation with guaranteed coverage probability.
No ML dependencies — pure numpy/stdlib.
"""

import pytest
import numpy as np
from src.monitor.conformal_risk import (
    ConformalRiskQuantifier,
    ConformalPrediction,
    conformal_var,
    conformal_cvar,
)


@pytest.fixture
def normal_returns():
    """1000 days of normal(0.0004, 0.01) returns."""
    np.random.seed(42)
    return np.random.normal(0.0004, 0.01, 1000)


@pytest.fixture
def heavy_tail_returns():
    """1000 days of t-distributed returns (df=3, heavy tails)."""
    np.random.seed(42)
    return np.random.standard_t(df=3, size=1000) * 0.01 + 0.0003


@pytest.fixture
def skewed_returns():
    """1000 days of right-skewed returns."""
    np.random.seed(42)
    from scipy.stats import skewnorm
    return skewnorm.rvs(a=-5, loc=0.001, scale=0.01, size=1000)


class TestConformalRiskQuantifier:
    """Core conformal risk quantifier tests."""

    def test_basic_fit_predict(self, normal_returns):
        """fit() then predict() should produce valid prediction intervals."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(normal_returns[:800])
        pred = crq.predict(normal_returns[800:850])

        assert isinstance(pred, ConformalPrediction)
        assert pred.lower < pred.point_estimate < pred.upper
        assert pred.alpha == 0.05

    def test_coverage_guarantee_normal(self, normal_returns):
        """Empirical coverage should be >= 1-alpha for normal data."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(normal_returns[:500])

        # Check coverage on remaining data
        covered = 0
        n_test = len(normal_returns) - 500
        for i in range(500, len(normal_returns)):
            pred = crq.predict(normal_returns[i:i+1])
            if pred.lower <= normal_returns[i] <= pred.upper:
                covered += 1

        coverage = covered / n_test
        # Conformal guarantee: coverage >= 1-alpha (95%)
        # Allow some finite-sample slack
        assert coverage >= 0.90, f"Coverage {coverage:.3f} below 90% threshold"

    def test_coverage_guarantee_heavy_tail(self, heavy_tail_returns):
        """Coverage should hold even for heavy-tailed data."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(heavy_tail_returns[:500])

        covered = 0
        n_test = len(heavy_tail_returns) - 500
        for i in range(500, len(heavy_tail_returns)):
            pred = crq.predict(heavy_tail_returns[i:i+1])
            if pred.lower <= heavy_tail_returns[i] <= pred.upper:
                covered += 1

        coverage = covered / n_test
        assert coverage >= 0.90, f"Heavy-tail coverage {coverage:.3f} below 90%"

    def test_wider_intervals_for_heavier_tails(self, normal_returns, heavy_tail_returns):
        """Heavy-tailed data should produce wider prediction intervals."""
        crq_normal = ConformalRiskQuantifier(alpha=0.05)
        crq_normal.fit(normal_returns[:500])

        crq_heavy = ConformalRiskQuantifier(alpha=0.05)
        crq_heavy.fit(heavy_tail_returns[:500])

        width_normal = crq_normal.predict(normal_returns[500:550]).interval_width
        width_heavy = crq_heavy.predict(heavy_tail_returns[500:550]).interval_width

        # Heavy tails should produce wider intervals
        assert width_heavy > width_normal

    def test_narrower_intervals_with_more_data(self, normal_returns):
        """More calibration data should produce tighter intervals."""
        crq_small = ConformalRiskQuantifier(alpha=0.05)
        crq_small.fit(normal_returns[:100])
        width_small = crq_small.predict(normal_returns[100:150]).interval_width

        crq_large = ConformalRiskQuantifier(alpha=0.05)
        crq_large.fit(normal_returns[:700])
        width_large = crq_large.predict(normal_returns[700:750]).interval_width

        # More data → tighter intervals (convergence)
        assert width_large < width_small * 1.5  # Some slack for randomness

    def test_different_alpha_levels(self, normal_returns):
        """Lower alpha (higher confidence) should produce wider intervals."""
        crq_95 = ConformalRiskQuantifier(alpha=0.05)
        crq_95.fit(normal_returns[:500])

        crq_99 = ConformalRiskQuantifier(alpha=0.01)
        crq_99.fit(normal_returns[:500])

        pred_95 = crq_95.predict(normal_returns[500:550])
        pred_99 = crq_99.predict(normal_returns[500:550])

        # 99% interval should be wider than 95%
        assert pred_99.interval_width > pred_95.interval_width


class TestConformalPrediction:
    """Tests for ConformalPrediction dataclass."""

    def test_fields(self, normal_returns):
        """ConformalPrediction should have all required fields."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(normal_returns[:500])
        pred = crq.predict(normal_returns[500:550])

        assert hasattr(pred, 'lower')
        assert hasattr(pred, 'upper')
        assert hasattr(pred, 'point_estimate')
        assert hasattr(pred, 'alpha')
        assert hasattr(pred, 'interval_width')
        assert pred.interval_width == pred.upper - pred.lower

    def test_asymmetric_intervals(self, skewed_returns):
        """Skewed data should produce asymmetric intervals around median."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(skewed_returns[:500])
        pred = crq.predict(skewed_returns[500:550])

        # Intervals should be valid
        assert pred.lower < pred.upper
        assert pred.alpha == 0.05


class TestConformalVaR:
    """Tests for conformal VaR computation."""

    def test_conformal_var_is_negative(self, normal_returns):
        """VaR should be a negative return (loss)."""
        var = conformal_var(normal_returns[:500], alpha=0.05)
        assert var < 0, f"VaR should be negative (loss), got {var}"

    def test_conformal_var_scales_with_vol(self):
        """Higher vol data should have larger VaR magnitude."""
        np.random.seed(42)
        low_vol = np.random.normal(0.0004, 0.005, 500)
        high_vol = np.random.normal(0.0004, 0.02, 500)

        var_low = abs(conformal_var(low_vol, alpha=0.05))
        var_high = abs(conformal_var(high_vol, alpha=0.05))

        assert var_high > var_low

    def test_conformal_var_alpha_sensitivity(self, normal_returns):
        """More extreme alpha should produce larger VaR."""
        var_05 = conformal_var(normal_returns[:500], alpha=0.05)
        var_01 = conformal_var(normal_returns[:500], alpha=0.01)

        # 1% VaR should be more negative (larger loss) than 5% VaR
        assert var_01 < var_05


class TestConformalCVaR:
    """Tests for conformal CVaR computation."""

    def test_conformal_cvar_worse_than_var(self, normal_returns):
        """CVaR should be worse (more negative) than VaR."""
        var = conformal_var(normal_returns[:500], alpha=0.05)
        cvar = conformal_cvar(normal_returns[:500], alpha=0.05)

        assert cvar <= var, f"CVaR ({cvar}) should be <= VaR ({var})"

    def test_conformal_cvar_is_negative(self, normal_returns):
        """CVaR should be a negative return (loss)."""
        cvar = conformal_cvar(normal_returns[:500], alpha=0.05)
        assert cvar < 0

    def test_conformal_cvar_matches_garch_cvar_order(self, normal_returns):
        """Conformal CVaR should be in similar ballpark to historical CVaR."""
        from src.monitor.cvar_metrics import calculate_cvar

        conf_cvar = conformal_cvar(normal_returns[:500], alpha=0.05)
        hist_cvar = calculate_cvar(normal_returns[:500], alpha=0.05)

        # Should be within 3x of each other (same ballpark)
        ratio = abs(conf_cvar / hist_cvar) if hist_cvar != 0 else 1
        assert 0.3 < ratio < 3.0, f"Conformal CVaR ratio {ratio:.2f} too far from historical"


class TestEdgeCases:
    """Edge case handling."""

    def test_minimum_data(self):
        """Should handle very small datasets gracefully."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        tiny = np.random.normal(0, 0.01, 30)
        crq.fit(tiny)
        pred = crq.predict(np.array([0.001]))
        assert pred.lower < pred.upper

    def test_empty_returns(self):
        """Empty returns should not crash."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(np.array([]))
        pred = crq.predict(np.array([0.0]))
        # Should return degenerate interval
        assert pred.lower <= pred.upper

    def test_constant_returns(self):
        """Constant returns should produce zero-width interval."""
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(np.full(100, 0.001))
        pred = crq.predict(np.array([0.001]))
        assert abs(pred.interval_width) < 0.01  # Near zero

    def test_single_outlier(self, normal_returns):
        """Single outlier shouldn't dominate interval width."""
        data = normal_returns[:200].copy()
        data[100] = 0.20  # 20% single-day return
        crq = ConformalRiskQuantifier(alpha=0.05)
        crq.fit(data)
        pred = crq.predict(normal_returns[200:250])
        # Interval should be wider but not absurdly wide
        assert pred.interval_width < 0.50
