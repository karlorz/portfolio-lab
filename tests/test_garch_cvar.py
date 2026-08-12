#!/usr/bin/env python3
"""
Tests for garch_cvar.py — GARCH-filtered CVaR calculation.

Validates:
- GARCH(1,1) model fitting and parameter stability
- Standardized return calculation
- CVaR rescaling correctness
- Fallback behavior when arch unavailable
- Breach rate validation
- GARCH parameter validation
"""
import numpy as np


import pytest
from unittest.mock import patch, MagicMock

# Import after path setup
from src.monitor.garch_cvar import (
    GARCHFilteredCVaR,
    GARCHCVaRMetrics,
    GARCHParams,
    calculate_garch_cvar,
    compare_cvar_methods,
    ARCH_AVAILABLE,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_returns_garch_like(n=252, omega=0.000001, alpha=0.1, beta=0.85, seed=42):
    """Generate returns with GARCH-like volatility clustering."""
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    vol = 0.01
    
    for t in range(1, n):
        # GARCH(1,1) process
        vol = np.sqrt(omega + alpha * returns[t-1]**2 + beta * vol**2)
        returns[t] = rng.normal(0, vol)
    
    return returns


def _make_returns_iid(n=252, mean=0.0003, std=0.012, seed=42):
    """Generate i.i.d. normal returns (no clustering)."""
    rng = np.random.RandomState(seed)
    return rng.normal(mean, std, n)


# -----------------------------------------------------------------------------
# GARCHParams Tests
# -----------------------------------------------------------------------------

class TestGARCHParams:
    """Test GARCH parameter dataclass and validation."""
    
    def test_creation(self):
        params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.85, persistence=0.95)
        assert params.omega == 0.000001
        assert params.alpha == 0.1
        assert params.beta == 0.85
        assert params.persistence == 0.95
    
    def test_stable(self):
        # persistence < 0.99 should be stable
        params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.85, persistence=0.95)
        assert params.is_stable()
    
    def test_unstable_high_persistence(self):
        # persistence >= 0.9999 should be unstable
        params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.89, persistence=0.99995)
        assert not params.is_stable()
    
    def test_unstable_zero_omega(self):
        # omega <= 0 should be unstable
        params = GARCHParams(omega=0.0, alpha=0.1, beta=0.85, persistence=0.95)
        assert not params.is_stable()

    def test_stable_at_high_persistence(self):
        """Persistence just below 0.9999 boundary should be stable."""
        params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.8998, persistence=0.9998)
        assert params.is_stable()

    def test_unstable_at_boundary_persistence(self):
        """Persistence exactly 0.9999 is NOT stable (boundary exclusion)."""
        params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.8999, persistence=0.9999)
        assert not params.is_stable()

    def test_stable_with_minimal_omega(self):
        """Very small but positive omega is still stable."""
        params = GARCHParams(omega=1e-10, alpha=0.1, beta=0.85, persistence=0.95)
        assert params.is_stable()

    def test_zero_persistence_stable(self):
        """Zero persistence (no ARCH/GARCH effect) with positive omega is stable."""
        params = GARCHParams(omega=0.000001, alpha=0.0, beta=0.0, persistence=0.0)
        assert params.is_stable()

    def test_negative_alpha(self):
        """Negative alpha is physically invalid but the dataclass permits it; is_stable
        only checks persistence and omega."""
        params = GARCHParams(omega=0.000001, alpha=-0.1, beta=0.85, persistence=0.75)
        assert params.alpha == -0.1
        # persistence < 0.9999 and omega > 0 -> is_stable True despite negative alpha
        assert params.is_stable()


# -----------------------------------------------------------------------------
# GARCHFilteredCVaR Initialization Tests
# -----------------------------------------------------------------------------

class TestGARCHCVaRInitialization:
    """Test GARCHFilteredCVaR constructor and configuration."""
    
    def test_default_init(self):
        calc = GARCHFilteredCVaR()
        assert calc.window == 252
        assert calc.p == 1
        assert calc.q == 1
        assert calc.dist == "normal"
        assert calc.fallback_threshold == 0.03
        assert calc.convergence_retries == 3
    
    def test_custom_init(self):
        calc = GARCHFilteredCVaR(
            window=500,
            p=2,
            q=2,
            dist="t",
            fallback_threshold=0.1,
            convergence_retries=5
        )
        assert calc.window == 500
        assert calc.p == 2
        assert calc.q == 2
        assert calc.dist == "t"
        assert calc.fallback_threshold == 0.1
        assert calc.convergence_retries == 5
    
    def test_student_t_dist(self):
        calc = GARCHFilteredCVaR(dist="t")
        assert calc.dist == "t"
    
    def test_skewt_dist(self):
        calc = GARCHFilteredCVaR(dist="skewt")
        assert calc.dist == "skewt"

    def test_zero_window(self):
        """Zero window should be accepted (edge case for lookback)."""
        calc = GARCHFilteredCVaR(window=0)
        assert calc.window == 0

    def test_single_retry(self):
        """Single convergence retry is a valid configuration."""
        calc = GARCHFilteredCVaR(convergence_retries=1)
        assert calc.convergence_retries == 1

    def test_invalid_distribution_accepted(self):
        """Literal types are not enforced at runtime; the constructor accepts any string."""
        calc = GARCHFilteredCVaR(dist="invalid_dist")
        assert calc.dist == "invalid_dist"


# -----------------------------------------------------------------------------
# fit_garch Tests
# -----------------------------------------------------------------------------

class TestFitGARCH:
    """Test GARCH model fitting."""
    
    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_fit_success(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_garch_like(n=252)
        
        params, cond_vol = calc.fit_garch(returns)
        
        if params is not None:
            assert isinstance(params, GARCHParams)
            assert params.omega > 0
            assert 0 <= params.alpha < 1
            assert 0 < params.beta < 1
            assert params.persistence < 1.0
            assert cond_vol is not None
            assert len(cond_vol) == len(returns)
            assert np.all(cond_vol > 0)
    
    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_fit_insufficient_data(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_garch_like(n=10)  # Way below threshold
        
        params, cond_vol = calc.fit_garch(returns)
        
        assert params is None
        assert cond_vol is None
    
    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_fit_retries_on_failure(self):
        calc = GARCHFilteredCVaR(convergence_retries=3)
        returns = _make_returns_garch_like(n=252)
        
        # Should attempt up to 3 times
        with patch('src.monitor.garch_cvar.arch_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.fit.side_effect = ValueError("Convergence failed")
            mock_model.return_value = mock_instance
            
            params, cond_vol = calc.fit_garch(returns)
            assert mock_instance.fit.call_count == 3
    
    def test_fallback_arch_unavailable(self):
        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', False):
            with patch('src.monitor.garch_cvar.arch_model', None):
                calc = GARCHFilteredCVaR()
                returns = _make_returns_garch_like(n=252)

                params, cond_vol = calc.fit_garch(returns)

                assert params is None
                assert cond_vol is None

    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_fit_params_unstable_retries_exhausted(self):
        """When fit succeeds but parameters are unstable, retries are attempted.
        If all retries are exhausted, returns (None, None)."""
        calc = GARCHFilteredCVaR(convergence_retries=2)
        returns = _make_returns_garch_like(n=252)

        with patch('src.monitor.garch_cvar.arch_model') as mock_model:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            # persistence = 0.5 + 0.5 = 1.0 >= 0.9999 -> unstable
            mock_result.params = {'omega': 0.000001, 'alpha[1]': 0.5, 'beta[1]': 0.5}
            mock_result.conditional_volatility = np.ones(252) / 100.0
            mock_instance.fit.return_value = mock_result
            mock_model.return_value = mock_instance

            params, cond_vol = calc.fit_garch(returns)

            assert params is None
            assert cond_vol is None
            # Both retries attempted
            assert mock_instance.fit.call_count == 2

    def test_fit_empty_returns(self):
        """fit_garch should return (None, None) for empty returns regardless of ARCH."""
        calc = GARCHFilteredCVaR()
        returns = np.array([])

        params, cond_vol = calc.fit_garch(returns)

        assert params is None
        assert cond_vol is None


# -----------------------------------------------------------------------------
# standardize_returns Tests
# -----------------------------------------------------------------------------

class TestStandardizeReturns:
    """Test return standardization by conditional volatility."""
    
    def test_basic_standardization(self):
        calc = GARCHFilteredCVaR()
        returns = np.array([0.01, -0.02, 0.015, -0.01])
        cond_vol = np.array([0.012, 0.015, 0.011, 0.013])
        
        std_returns = calc.standardize_returns(returns, cond_vol)
        
        expected = returns / cond_vol
        np.testing.assert_array_almost_equal(std_returns, expected)
    
    def test_standardization_normalizes_vol(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_garch_like(n=252)
        cond_vol = np.full_like(returns, 0.01)  # Constant volatility
        
        std_returns = calc.standardize_returns(returns, cond_vol)
        
        # Should be roughly scaled
        assert np.std(std_returns) > 0
    
    def test_handles_zero_volatility(self):
        calc = GARCHFilteredCVaR()
        returns = np.array([0.01, 0.02])
        cond_vol = np.array([0.0, 0.0])  # Zero volatility (edge case)
        
        std_returns = calc.standardize_returns(returns, cond_vol, min_vol=1e-6)
        
        # Should use min_vol instead of zero
        assert np.all(np.isfinite(std_returns))

    def test_standardize_negative_vol(self):
        """Negative conditional volatility is floored to min_vol."""
        calc = GARCHFilteredCVaR()
        returns = np.array([0.01, -0.02])
        cond_vol = np.array([-0.01, 0.015])  # First element is negative

        std_returns = calc.standardize_returns(returns, cond_vol, min_vol=1e-6)

        assert np.all(np.isfinite(std_returns))
        # Negative vol floored to min_vol; positive vol used directly
        assert std_returns[0] == returns[0] / 1e-6
        assert std_returns[1] == returns[1] / 0.015

    def test_standardize_empty_arrays(self):
        """Empty returns and cond_vol should produce empty standardized returns."""
        calc = GARCHFilteredCVaR()
        std_returns = calc.standardize_returns(np.array([]), np.array([]))
        assert len(std_returns) == 0


# -----------------------------------------------------------------------------
# rescale_cvar Tests
# -----------------------------------------------------------------------------

class TestRescaleCVaR:
    """Test CVaR rescaling from standardized to return space."""
    
    def test_basic_rescaling(self):
        calc = GARCHFilteredCVaR()
        cvar_std = -2.0  # 2 std deviations
        current_vol = 0.015  # 1.5% daily vol
        
        rescaled = calc.rescale_cvar(cvar_std, current_vol)
        
        expected = -2.0 * 0.015
        assert rescaled == pytest.approx(expected)
    
    def test_rescaling_preserves_sign(self):
        calc = GARCHFilteredCVaR()
        cvar_std = -1.5  # Negative (loss)
        current_vol = 0.01
        
        rescaled = calc.rescale_cvar(cvar_std, current_vol)
        
        assert rescaled < 0  # Should remain negative (loss)

    def test_rescale_zero_volatility(self):
        """Scaling by zero volatility yields zero."""
        calc = GARCHFilteredCVaR()
        assert calc.rescale_cvar(-2.0, 0.0) == 0.0

    def test_rescale_negative_volatility(self):
        """Negative volatility flips the sign of the result."""
        calc = GARCHFilteredCVaR()
        result = calc.rescale_cvar(-2.0, -0.01)
        assert result > 0  # Negative * negative = positive

    def test_rescale_high_volatility(self):
        """Very high current volatility produces proportionally larger CVaR."""
        calc = GARCHFilteredCVaR()
        result = calc.rescale_cvar(-2.0, 0.1)  # 10 % daily vol
        assert result == pytest.approx(-0.2)


# -----------------------------------------------------------------------------
# compute Tests (Integration)
# -----------------------------------------------------------------------------

class TestCompute:
    """Test full GARCH-CVaR computation pipeline."""
    
    def test_returns_garch_cvar_metrics(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns, current_drawdown=-0.03, max_drawdown=-0.20)
        
        assert isinstance(metrics, GARCHCVaRMetrics)
        assert metrics.var_95 < 0
        assert metrics.cvar_95 < 0
        assert metrics.cvar_95 <= metrics.var_95  # CVaR more negative than VaR
        assert 1.0 <= metrics.cvar_ratio <= 3.0
        assert metrics.tail_severity in ("normal", "moderate", "elevated", "severe")
    
    def test_var_cvar_negative(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        assert metrics.var_95 < 0
        assert metrics.cvar_95 < 0
    
    def test_cvar_more_extreme_than_var(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        # CVaR should capture more tail risk (be more negative)
        assert metrics.cvar_95 <= metrics.var_95
    
    def test_cvar_ratio_bounded(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        assert 1.0 <= metrics.cvar_ratio <= 3.0
    
    def test_tail_severity_valid(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        assert metrics.tail_severity in ("normal", "moderate", "elevated", "severe")
    
    def test_drawdown_preserved(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns, current_drawdown=-0.05, max_drawdown=-0.25)
        
        assert metrics.current_drawdown == pytest.approx(-5.0, abs=0.1)
        assert metrics.max_drawdown == pytest.approx(-25.0, abs=0.1)
    
    def test_volatility_positive(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        assert metrics.volatility_annual > 0
    
    def test_garch_filtered_flag(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        
        metrics = calc.compute(returns)
        
        assert isinstance(metrics.garch_filtered, bool)
    
    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_garch_params_when_active(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_garch_like(n=252)
        
        metrics = calc.compute(returns)
        
        if metrics.filter_active:
            assert metrics.garch_omega is not None
            assert metrics.garch_alpha is not None
            assert metrics.garch_beta is not None
            assert metrics.garch_persistence is not None
            assert metrics.conditional_volatility_current is not None
    
    def test_fallback_when_insufficient_data(self):
        calc = GARCHFilteredCVaR(window=252, fallback_threshold=0.5)
        returns = _make_returns_iid(n=15)  # Below EWMA minimum of 20

        metrics = calc.compute(returns)

        assert not metrics.filter_active
        assert metrics.filter_reason is not None
        assert "insufficient" in metrics.filter_reason.lower() or "converge" in metrics.filter_reason.lower()
    
    def test_fallback_reason_set(self):
        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', False):
            with patch('src.monitor.garch_cvar.arch_model', None):
                calc = GARCHFilteredCVaR()
                returns = _make_returns_iid(n=15)  # Below EWMA minimum

                metrics = calc.compute(returns)

                assert not metrics.filter_active
                assert "not available" in metrics.filter_reason.lower()

    def test_compute_arch_available_fallback(self):
        """Verify fallback reason is set when arch is available but fitting
        raises an error and EWMA also unavailable."""
        calc = GARCHFilteredCVaR(convergence_retries=1)
        returns = _make_returns_garch_like(n=252)

        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', True):
            with patch('src.monitor.garch_cvar.arch_model') as mock_model:
                mock_instance = MagicMock()
                mock_instance.fit.side_effect = ValueError("Convergence failed")
                mock_model.return_value = mock_instance

                metrics = calc.compute(returns)

                # EWMA fallback activates since we have 252 returns
                assert metrics.filter_active
                assert "EWMA" in metrics.filter_reason

    def test_compute_unstable_params_fallback(self):
        """When fit succeeds but params are unstable, fallback to EWMA."""
        calc = GARCHFilteredCVaR(convergence_retries=1)
        returns = _make_returns_garch_like(n=252)

        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', True):
            with patch('src.monitor.garch_cvar.arch_model') as mock_model:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.params = {'omega': 0.000001, 'alpha[1]': 0.5, 'beta[1]': 0.5}
                mock_result.conditional_volatility = np.ones(252) / 100.0
                mock_instance.fit.return_value = mock_result
                mock_model.return_value = mock_instance

                metrics = calc.compute(returns)

                # EWMA fallback activates since GARCH params are unstable
                assert metrics.filter_active
                assert "EWMA" in metrics.filter_reason

    def test_compute_stores_state(self):
        """Verify _last_params and _last_volatility are preserved after compute."""
        calc = GARCHFilteredCVaR()
        returns = _make_returns_garch_like(n=252)

        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', True):
            with patch('src.monitor.garch_cvar.arch_model') as mock_model:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.params = {'omega': 0.01, 'alpha[1]': 0.15, 'beta[1]': 0.80}
                cond_vol_raw = np.linspace(0.5, 3.0, 252)
                mock_result.conditional_volatility = cond_vol_raw
                mock_instance.fit.return_value = mock_result
                mock_model.return_value = mock_instance

                metrics = calc.compute(returns)

                assert calc._last_params is not None
                assert calc._last_params.omega == 0.01 / 10000.0  # scaled by factor^2
                assert calc._last_params.alpha == 0.15
                assert calc._last_params.beta == 0.80
                assert calc._last_volatility is not None
                assert metrics.filter_active
                assert metrics.garch_filtered

    def test_compute_with_skewt_dist(self):
        """Compute works with skewt distribution."""
        calc = GARCHFilteredCVaR(dist="skewt")
        returns = _make_returns_iid(n=252)
        metrics = calc.compute(returns)
        assert isinstance(metrics, GARCHCVaRMetrics)


# -----------------------------------------------------------------------------
# calculate_garch_cvar Convenience Function
# -----------------------------------------------------------------------------

class TestCalculateGARCHCVaR:
    """Test convenience function for one-shot calculation."""
    
    def test_returns_metrics(self):
        returns = _make_returns_iid(n=252)
        
        metrics = calculate_garch_cvar(returns)
        
        assert isinstance(metrics, GARCHCVaRMetrics)
    
    def test_respects_window_param(self):
        returns = _make_returns_iid(n=500)
        
        # Should work with custom window
        metrics = calculate_garch_cvar(returns, window=252)
        assert isinstance(metrics, GARCHCVaRMetrics)
    
    def test_respects_dist_param(self):
        returns = _make_returns_iid(n=252)
        
        # Should accept different distributions
        metrics_t = calculate_garch_cvar(returns, dist="t")
        assert isinstance(metrics_t, GARCHCVaRMetrics)

    def test_calculate_garch_cvar_drawdowns(self):
        """Drawdown parameters are passed through to metrics."""
        returns = _make_returns_iid(n=252)
        metrics = calculate_garch_cvar(returns, current_drawdown=-0.04, max_drawdown=-0.20)
        assert metrics.current_drawdown == pytest.approx(-4.0, abs=0.1)
        assert metrics.max_drawdown == pytest.approx(-20.0, abs=0.1)


# -----------------------------------------------------------------------------
# compare_cvar_methods Tests
# -----------------------------------------------------------------------------

class TestCompareCvarMethods:
    """Test comparison between historical and GARCH-filtered methods."""
    
    def test_returns_comparison_dict(self):
        returns = _make_returns_iid(n=252)
        
        comparison = compare_cvar_methods(returns)
        
        assert "historical" in comparison
        assert "garch_filtered" in comparison
        assert "target_breach_rate" in comparison
        assert "accuracy_delta" in comparison
    
    def test_historical_has_var_cvar(self):
        returns = _make_returns_iid(n=252)
        
        comparison = compare_cvar_methods(returns)
        
        assert "var" in comparison["historical"]
        assert "cvar" in comparison["historical"]
        # Values should be negative (losses)
        assert comparison["historical"]["var"] < 0
        assert comparison["historical"]["cvar"] < 0
    
    def test_garch_has_metrics(self):
        returns = _make_returns_iid(n=252)
        
        comparison = compare_cvar_methods(returns)
        
        assert "var" in comparison["garch_filtered"]
        assert "cvar" in comparison["garch_filtered"]
        assert "cvar_ratio" in comparison["garch_filtered"]
        assert "tail_severity" in comparison["garch_filtered"]
        assert "filter_active" in comparison["garch_filtered"]
    
    def test_breach_rates_present(self):
        returns = _make_returns_iid(n=252)
        
        comparison = compare_cvar_methods(returns)
        
        assert "var_breach_rate" in comparison["historical"]
        assert "cvar_breach_rate" in comparison["historical"]
    
    def test_breach_rates_valid(self):
        returns = _make_returns_iid(n=252)
        
        comparison = compare_cvar_methods(returns)
        
        var_breach = comparison["historical"]["var_breach_rate"]
        cvar_breach = comparison["historical"]["cvar_breach_rate"]
        
        # Breach rates should be percentages (0-100)
        assert 0 <= var_breach <= 100
        assert 0 <= cvar_breach <= 100
        
        # CVaR breach rate should be <= VaR breach rate (tail average)
        assert cvar_breach <= var_breach

    def test_compare_cvar_methods_different_alpha(self):
        """Comparison works with custom alpha values."""
        returns = _make_returns_iid(n=252)
        comparison = compare_cvar_methods(returns, alpha=0.01)
        assert comparison["target_breach_rate"] == 1.0  # 1 %
        assert "var" in comparison["historical"]
        assert comparison["historical"]["var"] < 0

    def test_compare_cvar_methods_constant_returns(self):
        """Comparison handles constant returns (zero volatility)."""
        returns = np.full(252, 0.001)
        comparison = compare_cvar_methods(returns)
        assert "historical" in comparison
        assert "garch_filtered" in comparison


# -----------------------------------------------------------------------------
# Edge Cases and Stress Tests
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and stress scenarios."""
    
    def test_empty_returns(self):
        calc = GARCHFilteredCVaR()
        returns = np.array([])
        
        metrics = calc.compute(returns)
        
        # Should handle gracefully with defaults
        assert metrics.var_95 < 0  # Default should be negative
        assert metrics.cvar_95 < 0
    
    def test_single_return(self):
        calc = GARCHFilteredCVaR()
        returns = np.array([0.01])
        
        metrics = calc.compute(returns)
        
        # Should handle gracefully
        assert isinstance(metrics, GARCHCVaRMetrics)
    
    def test_all_zero_returns(self):
        calc = GARCHFilteredCVaR()
        returns = np.zeros(252)
        
        # This will cause issues with volatility calculation
        metrics = calc.compute(returns)
        
        # Should still return a result
        assert isinstance(metrics, GARCHCVaRMetrics)
    
    def test_extreme_returns(self):
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        # Add some extreme events
        returns[0] = -0.10  # -10% day
        returns[1] = 0.08   # +8% day
        
        metrics = calc.compute(returns)
        
        assert metrics.var_95 < 0
        assert metrics.cvar_95 < metrics.var_95  # CVaR captures more tail risk
    
    def test_constant_returns(self):
        calc = GARCHFilteredCVaR()
        returns = np.full(252, 0.001)  # Constant 0.1% daily return
        
        metrics = calc.compute(returns)
        
        # Zero volatility edge case
        assert isinstance(metrics, GARCHCVaRMetrics)


# -----------------------------------------------------------------------------
# Property-Based Validation
# -----------------------------------------------------------------------------

class TestProperties:
    """Validate mathematical properties of GARCH-CVaR."""
    
    def test_cvar_more_conservative_than_var(self):
        """CVaR should always be more conservative (more negative) than VaR."""
        for seed in range(10):
            returns = _make_returns_iid(n=252, seed=seed)
            calc = GARCHFilteredCVaR()
            metrics = calc.compute(returns)
            
            assert metrics.cvar_95 <= metrics.var_95, \
                f"CVaR ({metrics.cvar_95}) should be <= VaR ({metrics.var_95})"
    
    def test_ratio_bounds(self):
        """CVaR ratio should always be in valid range."""
        for seed in range(10):
            returns = _make_returns_iid(n=252, seed=seed)
            calc = GARCHFilteredCVaR()
            metrics = calc.compute(returns)
            
            assert 1.0 <= metrics.cvar_ratio <= 3.0, \
                f"Ratio {metrics.cvar_ratio} out of bounds"
    
    def test_severity_consistency(self):
        """Severity classification should be consistent with ratio."""
        calc = GARCHFilteredCVaR()
        returns = _make_returns_iid(n=252)
        metrics = calc.compute(returns)
        
        if metrics.cvar_ratio < 1.3:
            assert metrics.tail_severity == "normal"
        elif metrics.cvar_ratio < 1.5:
            assert metrics.tail_severity == "moderate"
        elif metrics.cvar_ratio < 1.8:
            assert metrics.tail_severity == "elevated"
        else:
            assert metrics.tail_severity == "severe"


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------

class TestPerformance:
    """Test computational performance requirements."""
    
    @pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch library not available")
    def test_calculation_time(self):
        """Calculation should complete in reasonable time (< 1 second for 252 days)."""
        import time
        
        returns = _make_returns_garch_like(n=252)
        calc = GARCHFilteredCVaR()
        
        start = time.time()
        metrics = calc.compute(returns)
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Calculation took {elapsed:.2f}s, expected < 1s"


# -----------------------------------------------------------------------------
# get_params Tests
# -----------------------------------------------------------------------------

class TestGetParams:
    """Test GARCHFilteredCVaR.get_params() method."""

    def test_get_params_none_before_fit(self):
        """get_params returns None when no fit has been performed."""
        calc = GARCHFilteredCVaR()
        assert calc.get_params() is None

    def test_get_params_dict_after_state_set(self):
        """get_params returns parameter dict when _last_params is set."""
        calc = GARCHFilteredCVaR()
        calc._last_params = GARCHParams(omega=0.000001, alpha=0.1, beta=0.85, persistence=0.95)

        params = calc.get_params()
        assert isinstance(params, dict)
        assert params["omega"] == 0.000001
        assert params["alpha"] == 0.1
        assert params["beta"] == 0.85
        assert params["persistence"] == 0.95


# -----------------------------------------------------------------------------
# GARCHCVaRMetrics Dataclass Direct Tests
# -----------------------------------------------------------------------------

class TestGARCHCVaRMetricsDirect:
    """Test direct construction of GARCHCVaRMetrics dataclass."""

    def test_direct_creation_all_fields(self):
        """Create GARCHCVaRMetrics directly with all fields populated."""
        from src.monitor.garch_cvar import GARCHCVaRMetrics

        metrics = GARCHCVaRMetrics(
            timestamp="2025-01-01T00:00:00",
            var_95=-1.5,
            cvar_95=-2.5,
            cvar_ratio=1.67,
            tail_severity="moderate",
            max_drawdown=-25.0,
            current_drawdown=-5.0,
            volatility_annual=12.5,
            garch_filtered=True,
            garch_omega=0.000001,
            garch_alpha=0.1,
            garch_beta=0.85,
            garch_persistence=0.95,
            conditional_volatility_current=1.5,
            historical_volatility=12.0,
            filter_active=True,
            filter_reason=None,
        )
        assert metrics.garch_filtered
        assert metrics.garch_alpha == 0.1
        assert metrics.var_95 == -1.5
        assert metrics.filter_active
        assert metrics.filter_reason is None
        assert metrics.garch_persistence == 0.95

    def test_direct_creation_fallback(self):
        """Create GARCHCVaRMetrics in fallback mode with null GARCH fields."""
        from src.monitor.garch_cvar import GARCHCVaRMetrics

        metrics = GARCHCVaRMetrics(
            timestamp="2025-01-01T00:00:00",
            var_95=-1.2,
            cvar_95=-1.8,
            cvar_ratio=1.5,
            tail_severity="moderate",
            max_drawdown=-25.0,
            current_drawdown=-3.0,
            volatility_annual=10.0,
            garch_filtered=False,
            garch_omega=None,
            garch_alpha=None,
            garch_beta=None,
            garch_persistence=None,
            conditional_volatility_current=None,
            historical_volatility=10.0,
            filter_active=False,
            filter_reason="arch library not available",
        )
        assert not metrics.garch_filtered
        assert metrics.garch_omega is None
        assert metrics.garch_alpha is None
        assert not metrics.filter_active
        assert metrics.filter_reason == "arch library not available"


if __name__ == "__main__":
    # Run basic smoke test
    print("Running GARCH-CVaR smoke test...")
    
    returns = _make_returns_iid(n=252)
    metrics = calculate_garch_cvar(returns)
    
    print(f"VaR (95%): {metrics.var_95:.2f}%")
    print(f"CVaR (95%): {metrics.cvar_95:.2f}%")
    print(f"Tail Severity: {metrics.tail_severity} ({metrics.cvar_ratio:.2f}x)")
    print(f"GARCH Filtered: {metrics.garch_filtered}")
    
    if metrics.filter_active:
        print(f"GARCH Parameters: ω={metrics.garch_omega:.2e}, α={metrics.garch_alpha:.3f}, β={metrics.garch_beta:.3f}")
    else:
        print(f"Filter inactive: {metrics.filter_reason}")
    
    print("\n✓ Smoke test passed")
