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
import sys
import os
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.monitor.cvar_metrics import calculate_var, calculate_cvar


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
        assert calc.fallback_threshold == 0.05
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
            assert 0 < params.alpha < 1
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
            mock_instance.fit.side_effect = Exception("Convergence failed")
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
        returns = _make_returns_iid(n=100)  # Below 50% threshold
        
        metrics = calc.compute(returns)
        
        assert not metrics.filter_active
        assert metrics.filter_reason is not None
        assert "insufficient" in metrics.filter_reason.lower() or "converge" in metrics.filter_reason.lower()
    
    def test_fallback_reason_set(self):
        with patch('src.monitor.garch_cvar.ARCH_AVAILABLE', False):
            with patch('src.monitor.garch_cvar.arch_model', None):
                calc = GARCHFilteredCVaR()
                returns = _make_returns_iid(n=252)
                
                metrics = calc.compute(returns)
                
                assert not metrics.filter_active
                assert "not available" in metrics.filter_reason.lower()


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
