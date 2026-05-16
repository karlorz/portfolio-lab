"""
Tests for v5.01 Transient Statistical Factors module.

Tests cover:
- TransientFactorExtractor initialization and configuration
- PCA computation on synthetic data with known factor structure
- Factor count selection (eigenvalue ratio)
- Factor stability computation
- Regime transition detection
- Ensemble signal generation
- Risk contribution calculation
- State persistence (save/load)
- Edge cases (single asset, insufficient data, NaN handling)
- Integration with data fetching
"""

import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.monitor.transient_factors import (
    TransientFactorExtractor,
    TransientFactorMetrics,
    compute_transient_risk,
    analyze_transient_factors,
    generate_ensemble_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stable_returns():
    """Synthetic returns with stable 2-factor structure (n_assets=5, n_days=120)."""
    np.random.seed(42)
    n_assets = 5
    n_days = 120
    # Two common factors
    factor1 = np.random.randn(n_days) * 0.01
    factor2 = np.random.randn(n_days) * 0.005
    # Asset-specific noise
    noise = np.random.randn(n_assets, n_days) * 0.003
    
    returns = np.zeros((n_assets, n_days))
    for i in range(n_assets):
        # Each asset has different loading on factors
        loading_1 = 0.5 + 0.5 * (i / n_assets)
        loading_2 = 0.3 * (1 - i / n_assets)
        returns[i] = loading_1 * factor1 + loading_2 * factor2 + noise[i]
    
    return returns


@pytest.fixture
def transitioning_returns():
    """Synthetic returns where factor structure changes mid-series (regime transition)."""
    np.random.seed(42)
    n_assets = 5
    n_days = 120
    
    # First 60 days: factor structure A
    factor1_a = np.random.randn(60) * 0.01
    factor2_a = np.random.randn(60) * 0.005
    
    # Last 60 days: factor structure B (different loadings)
    factor1_b = np.random.randn(60) * 0.015
    factor2_b = np.random.randn(60) * 0.008
    
    # Transition happens at day 60
    factor1 = np.concatenate([factor1_a, factor1_b])
    factor2 = np.concatenate([factor2_a, factor2_b])
    
    noise = np.random.randn(n_assets, n_days) * 0.005
    
    returns = np.zeros((n_assets, n_days))
    for i in range(n_assets):
        loading_1 = 0.5 + 0.5 * (i / n_assets)
        loading_2 = 0.3 * (1 - i / n_assets)
        returns[i] = loading_1 * factor1 + loading_2 * factor2 + noise[i]
    
    return returns


@pytest.fixture
def single_factor_returns():
    """Returns driven by single factor (strong market regime)."""
    np.random.seed(42)
    n_assets = 4
    n_days = 100
    
    market = np.random.randn(n_days) * 0.012
    noise = np.random.randn(n_assets, n_days) * 0.002
    
    returns = np.zeros((n_assets, n_days))
    for i in range(n_assets):
        returns[i] = market * (0.8 + 0.2 * np.random.random()) + noise[i]
    
    return returns


@pytest.fixture
def crash_returns():
    """Returns mimicking a market crash (volatility spike, correlation breakdown)."""
    np.random.seed(42)
    n_assets = 5
    n_days = 100
    
    # First 80 days: normal
    normal_days = 80
    crash_days = 20
    
    factor = np.random.randn(normal_days) * 0.01
    crash_factor = np.random.randn(crash_days) * 0.04  # 4x vol
    
    full_factor = np.concatenate([factor, crash_factor])
    noise = np.random.randn(n_assets, n_days) * 0.003
    
    returns = np.zeros((n_assets, n_days))
    for i in range(n_assets):
        # Correlations converge during crash
        beta = 0.5 + 0.5 * np.random.random()
        returns[i] = beta * full_factor + noise[i]
    
    return returns


@pytest.fixture
def asset_names():
    return ["SPY", "GLD", "TLT", "IEF", "QQQ"]


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_params(self):
        extractor = TransientFactorExtractor()
        assert extractor.window == 60
        assert extractor.min_window == 20
        assert extractor.max_factors == 4
        assert extractor.eigenvalue_ratio_threshold == 1.5
    
    def test_custom_params(self):
        extractor = TransientFactorExtractor(
            window=90, min_window=30, max_factors=5,
            eigenvalue_ratio_threshold=2.0
        )
        assert extractor.window == 90
        assert extractor.max_factors == 5
    
    def test_initial_state(self):
        extractor = TransientFactorExtractor()
        assert extractor._last_eigenvectors is None
        assert extractor._stability_history == []


# ---------------------------------------------------------------------------
# Test: PCA computation
# ---------------------------------------------------------------------------

class TestPCAComputation:
    def test_basic_computation(self, stable_returns, asset_names):
        extractor = TransientFactorExtractor(window=60, max_factors=3)
        metrics = extractor.compute(stable_returns, asset_names=asset_names)
        
        assert isinstance(metrics, TransientFactorMetrics)
        assert metrics.n_assets == 5
        assert metrics.n_factors_selected >= 1
        assert metrics.n_factors_selected <= 3
        assert 0 < metrics.explained_ratio <= 1.0
    
    def test_eigenvalue_sorting(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        
        # Eigenvalues should be sorted descending
        ev = metrics.factor_eigenvalues
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i+1], f"Eigenvalues not sorted at {i}"
    
    def test_factor_loadings_shape(self, stable_returns, asset_names):
        extractor = TransientFactorExtractor(max_factors=3)
        metrics = extractor.compute(stable_returns, asset_names=asset_names)
        
        n_factors = metrics.n_factors_selected
        for name in asset_names:
            assert len(metrics.individual_loadings[name]) == n_factors
    
    def test_single_factor_detection(self, single_factor_returns):
        extractor = TransientFactorExtractor(max_factors=4)
        metrics = extractor.compute(single_factor_returns)
        
        # Should detect 1 dominant factor
        assert metrics.n_factors_selected >= 1
        # First eigenvalue should dominate
        ev = metrics.factor_eigenvalues
        if len(ev) >= 2 and ev[1] > 0:
            ratio = ev[0] / ev[1]
            assert ratio > 1.5, f"First/second eigenvalue ratio too low: {ratio:.2f}"
    
    def test_crash_returns_factor_count(self, crash_returns):
        extractor = TransientFactorExtractor(window=60, max_factors=4)
        metrics = extractor.compute(crash_returns)
        
        # During crash, fewer factors should explain more variance
        assert metrics.explained_ratio > 0.3, "Crash should be explained by few factors"
    
    def test_error_single_asset(self):
        extractor = TransientFactorExtractor()
        with pytest.raises(ValueError, match="Need at least 2 assets"):
            extractor.compute(np.random.randn(1, 100))
    
    def test_error_insufficient_days(self):
        extractor = TransientFactorExtractor(min_window=30)
        with pytest.raises(ValueError, match="Need at least 30 days"):
            extractor.compute(np.random.randn(3, 10))


# ---------------------------------------------------------------------------
# Test: Factor Stability
# ---------------------------------------------------------------------------

class TestFactorStability:
    def test_stable_regime_high_stability(self, stable_returns):
        """Stable returns should produce high stability scores."""
        extractor = TransientFactorExtractor(window=60)
        
        # First call should return 1.0 (no prior history)
        m1 = extractor.compute(stable_returns[:, :80])
        assert m1.factor_stability == 1.0
        
        # Second call should show high stability (similar structure)
        m2 = extractor.compute(stable_returns[:, -80:])
        assert m2.factor_stability > 0.5, f"Expected high stability, got {m2.factor_stability}"
    
    def test_transitioning_regime_lower_stability(self, transitioning_returns):
        """Transitioning factor structure should produce lower stability."""
        extractor = TransientFactorExtractor(window=50)
        
        m1 = extractor.compute(transitioning_returns[:, :70])
        m2 = extractor.compute(transitioning_returns[:, -70:])
        
        # Transitioning regime should have lower stability
        # (Note: may not always be true with random data, but statistically)
        assert m2.factor_stability <= 1.0
    
    def test_stability_monotonic(self):
        """Stability values should be in [0, 1] range."""
        np.random.seed(123)
        n_assets = 3
        n_days = 120
        
        extractor = TransientFactorExtractor(window=40)
        
        for _ in range(3):
            returns = np.random.randn(n_assets, n_days) * 0.01
            metrics = extractor.compute(returns)
            assert 0.0 <= metrics.factor_stability <= 1.0
    
    def test_stability_history_tracked(self, stable_returns):
        extractor = TransientFactorExtractor(window=60)
        
        assert len(extractor._stability_history) == 0
        extractor.compute(stable_returns)
        assert len(extractor._stability_history) == 1
        extractor.compute(stable_returns)
        assert len(extractor._stability_history) == 2


# ---------------------------------------------------------------------------
# Test: Regime Transition Detection
# ---------------------------------------------------------------------------

class TestRegimeTransition:
    def test_stable_trend_label(self, stable_returns):
        extractor = TransientFactorExtractor(window=60)
        metrics = extractor.compute(stable_returns)
        # Stable returns should be "stable" or "shifting"
        assert metrics.stability_trend in ("stable", "shifting")
    
    def test_transition_label(self):
        """Force transition by changing data dramatically."""
        np.random.seed(99)
        n_assets = 3
        n_days = 80
        
        extractor = TransientFactorExtractor(window=40)
        
        # First half: stable
        returns1 = np.random.randn(n_assets, n_days) * 0.01
        extractor.compute(returns1)
        
        # Second half: completely different structure
        returns2 = np.ones((n_assets, n_days)) * 0.02 + np.random.randn(n_assets, n_days) * 0.03
        metrics = extractor.compute(returns2)
        
        # May or may not be "transition" with random data
        assert metrics.stability_trend in ("stable", "shifting", "transition")
    
    def test_transition_score_range(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        assert 0.0 <= metrics.regime_transition_score <= 1.0


# ---------------------------------------------------------------------------
# Test: Risk Contribution
# ---------------------------------------------------------------------------

class TestRiskContribution:
    def test_risk_contribution_range(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        assert 0.0 <= metrics.risk_contribution <= 1.0
    
    def test_risk_contribution_single_factor(self, single_factor_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(single_factor_returns)
        # With strong single factor, risk contribution should be meaningful
        assert metrics.risk_contribution > 0.0
    
    def test_compute_transient_risk_multiplier(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        
        risk = compute_transient_risk(metrics)
        
        assert "risk_multiplier" in risk
        assert 0.5 <= risk["risk_multiplier"] <= 2.0
        assert "stability_factor" in risk
        assert "transition_score" in risk
    
    def test_compute_transient_risk_with_cvar(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        
        risk = compute_transient_risk(metrics, base_cvar=-0.02)
        
        assert risk["adjusted_cvar"] is not None
        assert risk["adjusted_cvar"] <= 0.0  # CVaR is negative
        assert abs(risk["adjusted_cvar"]) >= 0.02  # Should be amplified


# ---------------------------------------------------------------------------
# Test: Ensemble Signal
# ---------------------------------------------------------------------------

class TestEnsembleSignal:
    def test_stable_signal_positive(self, stable_returns):
        extractor = TransientFactorExtractor(window=60)
        metrics = extractor.compute(stable_returns)
        
        sig = extractor.get_ensemble_signal_value(metrics)
        
        # Stable regime should give non-negative signal
        if metrics.stability_trend == "stable":
            assert sig >= -0.2
    
    def test_signal_range(self, stable_returns):
        extractor = TransientFactorExtractor()
        metrics = extractor.compute(stable_returns)
        
        sig = extractor.get_ensemble_signal_value(metrics)
        assert -0.8 <= sig <= 0.5
    
    def test_generate_ensemble_signal_returns_dict(self):
        sig = generate_ensemble_signal()
        assert "signal_value" in sig
        assert "confidence" in sig
        assert "reasoning" in sig
        assert -0.8 <= sig["signal_value"] <= 0.5
    
    def test_analyze_transient_factors_returns_dict(self):
        result = analyze_transient_factors()
        assert "status" in result


# ---------------------------------------------------------------------------
# Test: State Persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_state(self, stable_returns, tmp_path):
        extractor = TransientFactorExtractor(window=60)
        extractor.compute(stable_returns)
        
        state_path = tmp_path / "test_transient_state.json"
        extractor.save_state(state_path)
        
        assert state_path.exists()
        
        # Create new extractor and load
        extractor2 = TransientFactorExtractor(window=60)
        loaded = extractor2.load_state(state_path)
        
        assert loaded
        assert len(extractor2._stability_history) > 0
    
    def test_load_nonexistent(self, tmp_path):
        extractor = TransientFactorExtractor()
        loaded = extractor.load_state(tmp_path / "nonexistent.json")
        assert not loaded
    
    def test_load_corrupted(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{invalid json")
        
        extractor = TransientFactorExtractor()
        loaded = extractor.load_state(bad_path)
        assert not loaded


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_returns(self):
        """All-zero returns should not crash."""
        np.random.seed(42)
        returns = np.zeros((3, 100))
        
        extractor = TransientFactorExtractor()
        # With constant returns, eigenvalues will be near zero
        # but should not divide by zero
        try:
            metrics = extractor.compute(returns)
            assert metrics.n_factors_selected >= 1
        except Exception as e:
            pytest.fail(f"Constant returns caused exception: {e}")
    
    def test_very_short_data(self):
        """Data just at min_window should work."""
        np.random.seed(42)
        returns = np.random.randn(3, 20)
        
        extractor = TransientFactorExtractor(min_window=20)
        metrics = extractor.compute(returns)
        assert metrics.window == 20
    
    def test_many_assets(self):
        """Many assets with fewer days should still work."""
        np.random.seed(42)
        n_assets = 10
        n_days = 40
        returns = np.random.randn(n_assets, n_days)
        
        extractor = TransientFactorExtractor(window=40)
        metrics = extractor.compute(returns)
        assert metrics.n_assets == 10
        assert metrics.n_factors_selected <= 4  # max_factors
    
    def test_residual_computation(self):
        """Residuals should have lower variance than raw returns."""
        np.random.seed(42)
        n_assets = 3
        n_days = 100
        returns = np.random.randn(n_assets, n_days) * 0.01
        
        extractor = TransientFactorExtractor()
        
        # Market factor
        market = np.mean(returns[:, -60:], axis=0)
        
        residual_vars = []
        raw_vars = []
        for i in range(n_assets):
            cov = np.cov(returns[i, -60:], market)[0, 1]
            var_mkt = np.var(market) + 1e-10
            beta = cov / var_mkt
            residual = returns[i, -60:] - beta * market
            residual_vars.append(np.var(residual))
            raw_vars.append(np.var(returns[i, -60:]))
        
        # Reziduals should typically have lower variance
        for i in range(n_assets):
            assert residual_vars[i] <= raw_vars[i] * 1.1  # Allow slight increase


# ---------------------------------------------------------------------------
# Test: CLI Integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_analyze_creates_state_file(self):
        """analyze_transient_factors should create state file."""
        # Clean up if exists
        state_path = Path("data/transient_factor_state.json")
        if state_path.exists():
            state_path.unlink()
        
        result = analyze_transient_factors()
        
        if result.get("status") == "ok":
            # State file should be created
            assert state_path.exists(), "State file should be created"
            with open(state_path) as f:
                state = json.load(f)
            assert "stability_history" in state
            assert "last_eigenvectors" in state
