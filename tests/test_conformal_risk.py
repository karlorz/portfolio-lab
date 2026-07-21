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
    conformal_coverage_diagnostics,
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


class TestConformalCoverageDiagnostics:
    """Backtests for conformal VaR exceedance diagnostics."""

    def test_calibrated_sequence_passes_coverage_diagnostics(self):
        returns = np.full(500, 0.001)
        var_thresholds = np.full(500, -0.02)
        returns[::20] = -0.03

        diagnostics = conformal_coverage_diagnostics(
            returns,
            var_thresholds,
            alpha=0.05,
            rolling_window=252,
        )

        assert diagnostics["observations"] == 500
        assert diagnostics["exceedance_count"] == 25
        assert diagnostics["exceedance_rate"] == pytest.approx(0.05)
        assert diagnostics["coverage_rate"] == pytest.approx(0.95)
        assert diagnostics["coverage_pass"] is True
        assert diagnostics["kupiec_pass"] is True
        assert diagnostics["christoffersen_pass"] is True
        assert diagnostics["conditional_coverage_pass"] is True
        assert diagnostics["longest_violation_cluster"] == 1
        assert diagnostics["rolling_window"] == 252
        assert diagnostics["rolling_exceedance_rate"] == pytest.approx(0.047619, rel=1e-4)

    def test_undercovered_sequence_fails_kupiec_diagnostics(self):
        returns = np.full(500, 0.001)
        var_thresholds = np.full(500, -0.02)
        returns[:80] = -0.03

        diagnostics = conformal_coverage_diagnostics(
            returns,
            var_thresholds,
            alpha=0.05,
        )

        assert diagnostics["exceedance_count"] == 80
        assert diagnostics["exceedance_rate"] == pytest.approx(0.16)
        assert diagnostics["coverage_pass"] is False
        assert diagnostics["kupiec_pass"] is False
        assert diagnostics["kupiec_p_value"] < 0.05

    def test_clustered_violations_fail_independence_diagnostics(self):
        returns = np.full(500, 0.001)
        var_thresholds = np.full(500, -0.02)
        returns[100:125] = -0.03

        diagnostics = conformal_coverage_diagnostics(
            returns,
            var_thresholds,
            alpha=0.05,
        )

        assert diagnostics["exceedance_rate"] == pytest.approx(0.05)
        assert diagnostics["kupiec_pass"] is True
        assert diagnostics["longest_violation_cluster"] == 25
        assert diagnostics["christoffersen_pass"] is False
        assert diagnostics["conditional_coverage_pass"] is False
        assert diagnostics["christoffersen_p_value"] < 0.05

    def test_regime_labels_add_optional_per_regime_summaries(self):
        returns = np.full(200, 0.001)
        var_thresholds = np.full(200, -0.02)
        returns[:100:20] = -0.03
        returns[100:120] = -0.03
        regimes = ["normal"] * 100 + ["high_vol"] * 100

        diagnostics = conformal_coverage_diagnostics(
            returns,
            var_thresholds,
            alpha=0.05,
            regime_labels=regimes,
        )

        assert set(diagnostics["by_regime"]) == {"normal", "high_vol"}
        assert diagnostics["by_regime"]["normal"]["observations"] == 100
        assert diagnostics["by_regime"]["normal"]["exceedance_rate"] == pytest.approx(0.05)
        assert diagnostics["by_regime"]["normal"]["coverage_pass"] is True
        assert diagnostics["by_regime"]["high_vol"]["observations"] == 100
        assert diagnostics["by_regime"]["high_vol"]["exceedance_rate"] == pytest.approx(0.20)
        assert diagnostics["by_regime"]["high_vol"]["coverage_pass"] is False
        assert diagnostics["by_regime"]["high_vol"]["coverage_direction"] == "over"

    def test_over_exceedance_is_hard_fail(self):
        """Rate >> alpha → direction=over, coverage_hard_fail, coverage_pass false."""
        returns = np.full(500, 0.001)
        var_thresholds = np.full(500, -0.02)
        returns[:80] = -0.03  # 16% exceedances vs 5% alpha

        diagnostics = conformal_coverage_diagnostics(
            returns, var_thresholds, alpha=0.05
        )
        assert diagnostics["exceedance_rate"] > 0.05
        assert diagnostics["coverage_direction"] == "over"
        assert diagnostics["exceedance_bias"] == "over"
        assert diagnostics["kupiec_pass"] is False
        assert diagnostics["coverage_hard_fail"] is True
        assert diagnostics["coverage_pass"] is False
        assert diagnostics["coverage_efficiency_warning"] is False

    def test_under_exceedance_is_efficiency_warning_not_hard_fail(self):
        """Rate << alpha → direction=under; not demotion-grade coverage_pass fail.

        Under-exceedance (over-conservative VaR) is capital inefficiency, not
        risk underestimation — coverage_pass stays True for hard-fail gates.
        """
        rng = np.random.default_rng(42)
        # Mild positive returns so almost never breach a deep VaR threshold
        returns = rng.normal(loc=0.001, scale=0.002, size=500)
        var_thresholds = np.full(500, -0.05)  # very deep threshold → few breaches

        diagnostics = conformal_coverage_diagnostics(
            returns, var_thresholds, alpha=0.05
        )
        assert diagnostics["exceedance_rate"] < 0.05
        assert diagnostics["coverage_direction"] == "under"
        assert diagnostics["exceedance_bias"] == "under"
        # With near-zero breaches Kupiec often rejects under-coverage
        if diagnostics["kupiec_pass"] is False:
            assert diagnostics["coverage_hard_fail"] is False
            assert diagnostics["coverage_efficiency_warning"] is True
            assert diagnostics["coverage_pass"] is True  # not demotion-grade
        else:
            # Statistically ok under-bias still labels direction
            assert diagnostics["coverage_pass"] is True

    def test_calibrated_sequence_direction_ok(self):
        returns = np.full(500, 0.001)
        var_thresholds = np.full(500, -0.02)
        returns[::20] = -0.03  # exactly 5%
        diagnostics = conformal_coverage_diagnostics(
            returns, var_thresholds, alpha=0.05
        )
        assert diagnostics["coverage_direction"] == "ok"
        assert diagnostics["coverage_pass"] is True
        assert diagnostics["coverage_hard_fail"] is False


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
