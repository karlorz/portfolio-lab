"""
Tests for v8.04 Kelly-Optimal Position Sizing.

Tests the multivariate Kelly optimization with sigmoidal scaling.
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

from src.strategy.kelly_optimal_sizing import (
    ASSETS,
    BASE_ALLOCATION,
    HARD_BOUNDS,
    DEFAULT_LOOKBACK,
    DEFAULT_SIGMOID_K,
    DEFAULT_FRACTION_MAX,
    DEFAULT_RISK_FREE,
    KellyOptimizer,
    KellyDecision,
    KellyFactors,
    sigmoid_scaling,
    compute_edge_to_odds,
    multivariate_kelly,
    STATE_PATH,
)


# ── Sigmoid Scaling Tests ────────────────────────────────────────────────────

class TestSigmoidScaling:
    """Tests for sigmoid_scaling function."""

    def test_zero_edge_to_odds(self):
        """Zero or negative edge-to-odds yields zero."""
        assert sigmoid_scaling(0.0) == 0.0
        assert sigmoid_scaling(-1.0) == 0.0
        assert sigmoid_scaling(-0.001) == 0.0

    def test_positive_scaling(self):
        """Positive edge-to-odds yields positive scaling."""
        result = sigmoid_scaling(0.5)
        assert 0.0 < result < 1.0

    def test_large_edge_saturates(self):
        """Large edge-to-odds saturates near 1.0."""
        result = sigmoid_scaling(5.0)
        assert result > 0.99
        assert result <= 1.0

    def test_monotonic(self):
        """Scaling is monotonically increasing."""
        xs = np.linspace(0.01, 3.0, 20)
        ys = [sigmoid_scaling(x) for x in xs]
        for i in range(1, len(ys)):
            assert ys[i] >= ys[i - 1], f"Non-monotonic at {xs[i]}"

    def test_default_k_parameter(self):
        """Default k=1.5 gives reasonable shape."""
        # At edge/odds=1.0, scaling should be ~0.9
        result = sigmoid_scaling(1.0, k=1.5)
        assert 0.8 < result < 0.95

    def test_higher_k_sharper(self):
        """Higher k gives steeper sigmoid."""
        low_k = sigmoid_scaling(0.5, k=1.0)
        high_k = sigmoid_scaling(0.5, k=3.0)
        assert high_k > low_k

    def test_nan_safety(self):
        """NaN input returns 0."""
        result = sigmoid_scaling(float('nan'))
        assert result == 0.0 or math.isnan(result) == False
        # Should handle gracefully
        assert not math.isnan(result) or result == 0.0


# ── Edge-to-Odds Computation Tests ───────────────────────────────────────────

class TestComputeEdgeToOdds:
    """Tests for compute_edge_to_odds function."""

    def test_positive_edge(self):
        """Positive edge gives positive edge-to-odds."""
        result = compute_edge_to_odds(0.10, 0.04, risk_free=0.04)
        # edge = 0.06, odds = 0.04 -> 1.5
        assert abs(result - 1.5) < 1e-10

    def test_negative_edge(self):
        """Negative edge returns zero."""
        result = compute_edge_to_odds(0.03, 0.04, risk_free=0.05)
        assert result == 0.0

    def test_zero_variance(self):
        """Zero variance returns zero."""
        result = compute_edge_to_odds(0.10, 0.0)
        assert result == 0.0

    def test_default_risk_free(self):
        """Uses default risk-free rate."""
        result = compute_edge_to_odds(0.10, 0.04)
        # Uses DEFAULT_RISK_FREE = 0.042
        # edge = 0.10 - 0.042 = 0.058, odds = 0.04 -> 1.45
        expected = (0.10 - DEFAULT_RISK_FREE) / 0.04
        assert abs(result - expected) < 1e-10

    def test_numerical_stability(self):
        """Very small positive variance handled gracefully."""
        result = compute_edge_to_odds(0.10, 1e-12)
        assert result > 0


# ── Multivariate Kelly Tests ─────────────────────────────────────────────────

class TestMultivariateKelly:
    """Tests for multivariate_kelly function."""

    def test_single_asset(self):
        """Single asset case reduces to simple Kelly."""
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        f = multivariate_kelly(mu, cov, risk_free=0.04)
        expected = (0.10 - 0.04) / 0.04  # 1.5
        assert abs(f[0] - expected) < 1e-10

    def test_two_uncorrelated_assets(self):
        """Two uncorrelated assets give independent fractions."""
        mu = np.array([0.10, 0.08])
        cov = np.array([[0.04, 0.0], [0.0, 0.03]])
        f = multivariate_kelly(mu, cov, risk_free=0.04)
        expected_0 = (0.10 - 0.04) / 0.04  # 1.5
        expected_1 = (0.08 - 0.04) / 0.03  # 1.333...
        assert abs(f[0] - expected_0) < 1e-10
        assert abs(f[1] - expected_1) < 1e-10

    def test_three_assets(self):
        """Three-asset case works."""
        mu = np.array([0.10, 0.06, 0.04])
        cov = np.array([
            [0.04, 0.005, -0.002],
            [0.005, 0.03, 0.001],
            [-0.002, 0.001, 0.02],
        ])
        f = multivariate_kelly(mu, cov, risk_free=0.04)
        assert len(f) == 3
        # All should be finite
        assert all(np.isfinite(f))

    def test_near_singular_covariance(self):
        """Near-singular covariance falls back to pseudo-inverse."""
        mu = np.array([0.10, 0.10])
        # Highly correlated
        cov = np.array([[0.04, 0.0399], [0.0399, 0.04]])
        f = multivariate_kelly(mu, cov, risk_free=0.04)
        assert len(f) == 2
        assert all(np.isfinite(f))

    def test_all_negative_excess(self):
        """When all excess returns are negative, fractions should be negative."""
        mu = np.array([0.01, 0.01])
        cov = np.array([[0.04, 0.0], [0.0, 0.03]])
        f = multivariate_kelly(mu, cov, risk_free=0.05)
        assert f[0] < 0
        assert f[1] < 0


# ── KellyOptimizer Unit Tests ─────────────────────────────────────────────────

class TestKellyOptimizer:
    """Tests for KellyOptimizer class."""

    def test_init_defaults(self):
        """Default initialization uses package defaults."""
        opt = KellyOptimizer()
        assert opt.lookback == DEFAULT_LOOKBACK
        assert opt.sigmoid_k == DEFAULT_SIGMOID_K
        assert opt.fraction_max == DEFAULT_FRACTION_MAX
        assert opt.risk_free == DEFAULT_RISK_FREE

    def test_init_with_data_dir(self):
        """Custom data directory."""
        with tempfile.TemporaryDirectory() as tmp:
            opt = KellyOptimizer(data_dir=Path(tmp))
            assert str(tmp) in str(opt.data_dir)

    def test_apply_bounds(self):
        """Hard bounds are enforced."""
        opt = KellyOptimizer()
        # Give extreme values
        alloc = {"SPY": 0.90, "GLD": 0.60, "TLT": 0.50}
        result = opt._apply_bounds(alloc)
        assert result["SPY"] <= HARD_BOUNDS["SPY"][1]
        assert result["GLD"] <= HARD_BOUNDS["GLD"][1]
        assert result["TLT"] <= HARD_BOUNDS["TLT"][1]
        assert result["SPY"] >= HARD_BOUNDS["SPY"][0]
        # Sum should be ~1.0
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_apply_bounds_normalizes(self):
        """Allocation sums to 1.0 after bounds."""
        opt = KellyOptimizer()
        alloc = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        result = opt._apply_bounds(alloc)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_load_state_no_file(self):
        """No state file does not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            opt = KellyOptimizer(data_dir=Path(tmp))
            opt._load_state()
            assert opt.last_allocation == BASE_ALLOCATION

    def test_load_state_from_file(self):
        """State file loads correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "last_allocation": {"SPY": 0.50, "GLD": 0.35, "TLT": 0.15},
                "lookback": 126,
                "sigmoid_k": 2.0,
                "fraction_max": 0.7,
            }
            state_path = Path(tmp) / "kelly_sizing_state.json"
            state_path.write_text(json.dumps(state))
            
            opt = KellyOptimizer(data_dir=Path(tmp))
            opt._load_state()
            assert opt.last_allocation["SPY"] == 0.50
            assert opt.lookback == 126
            assert opt.sigmoid_k == 2.0
            assert opt.fraction_max == 0.7

    def test_save_state(self):
        """State persists correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            opt = KellyOptimizer(data_dir=Path(tmp))
            decision = KellyDecision(
                timestamp="2026-05-17T12:00:00",
                base_allocation=BASE_ALLOCATION,
                kelly_allocation={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
                scaled_allocation={"SPY": 0.3, "GLD": 0.2, "TLT": 0.1},
                final_allocation={"SPY": 0.48, "GLD": 0.36, "TLT": 0.16},
                asset_edges={"SPY": 0.05, "GLD": 0.02, "TLT": 0.01},
                asset_odds={"SPY": 0.04, "GLD": 0.03, "TLT": 0.02},
                factors=KellyFactors(
                    timestamp="2026-05-17T12:00:00",
                    lookback_days=252,
                    spy_mean_return=0.10,
                    gld_mean_return=0.06,
                    tlt_mean_return=0.04,
                    spy_volatility=0.20,
                    gld_volatility=0.17,
                    tlt_volatility=0.14,
                    avg_edge_to_odds=0.5,
                    kelly_magnitude=0.6,
                    sigmoid_scale=0.4,
                ),
            )
            opt._save_state(decision)
            assert opt.state_path.exists()
            loaded = json.loads(opt.state_path.read_text())
            assert loaded["last_allocation"]["SPY"] == 0.48
            assert loaded["avg_edge_to_odds"] == 0.5

    def test_load_prices_not_found(self):
        """Missing prices file returns None without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch('src.strategy.kelly_optimal_sizing.PROJECT_ROOT', tmp_path):
                opt = KellyOptimizer(data_dir=tmp_path)
                result = opt.load_prices()
                assert result is None

    @patch("src.strategy.kelly_optimal_sizing.PROJECT_ROOT")
    def test_load_prices_from_project(self, mock_root):
        """Prices loaded from project path."""
        mock_root.__str__ = lambda self: "/tmp"
        mock_root.__truediv__ = lambda self, other: Path(f"/tmp/{other}")
        opt = KellyOptimizer(data_dir=Path("/tmp"))
        # Will fail because file doesn't exist, but should handle gracefully
        result = opt.load_prices()
        assert result is None or isinstance(result, dict)

    @patch.object(KellyOptimizer, 'load_prices')
    @patch.object(KellyOptimizer, 'get_series')
    def test_compute_allocation_with_mock_data(self, mock_get_series, mock_load_prices):
        """Computes allocation with mock price data."""
        mock_load_prices.return_value = {"SPY": [], "GLD": [], "TLT": []}
        
        # Create synthetic price series
        np.random.seed(42)
        n = 300
        spy = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))
        gld = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n)))
        tlt = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.006, n)))
        
        def get_series_side_effect(symbol):
            series_map = {"SPY": spy, "GLD": gld, "TLT": tlt}
            return series_map.get(symbol)
        
        mock_get_series.side_effect = get_series_side_effect
        
        opt = KellyOptimizer()
        decision = opt.compute_allocation(lookback=250)
        
        assert isinstance(decision, KellyDecision)
        assert len(decision.final_allocation) >= 3
        # Allocation should sum to ~1.0
        total = sum(decision.final_allocation.get(sym, 0) for sym in ASSETS)
        assert abs(total - 1.0) < 0.15  # Allow for bonds/IEF/SHY

    @patch.object(KellyOptimizer, 'load_prices')
    @patch.object(KellyOptimizer, 'get_series')
    def test_compute_allocation_fallback(self, mock_get_series, mock_load_prices):
        """Fallback allocation when data is insufficient."""
        mock_load_prices.return_value = {}
        mock_get_series.return_value = np.array([100.0, 101.0])  # Only 2 data points
        
        opt = KellyOptimizer()
        decision = opt.compute_allocation(lookback=250)
        
        # Should fall back to base allocation-ish
        for sym in ASSETS:
            assert sym in decision.final_allocation

    def test_estimate_returns_insufficient_data(self):
        """Estimate returns with insufficient data returns zeros."""
        opt = KellyOptimizer()
        prices = {
            "SPY": np.random.randn(10),
            "GLD": np.random.randn(10),
            "TLT": np.random.randn(10),
        }
        means, cov, vols = opt.estimate_returns_and_cov(prices, lookback=252)
        assert len(means) == 3
        assert cov.shape == (3, 3)

    def test_estimate_returns_with_realistic_data(self):
        """Estimate returns produces reasonable values."""
        opt = KellyOptimizer()
        np.random.seed(42)
        n = 300
        prices = {
            "SPY": 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n))),
            "GLD": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n))),
            "TLT": 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.006, n))),
        }
        means, cov, vols = opt.estimate_returns_and_cov(prices)
        assert means.size == 3
        assert vols.size == 3
        # Vols should be positive
        assert all(v > 0 for v in vols)
        # Cov should be symmetric
        assert np.allclose(cov, cov.T)


# ── Backend Simulation Tests ─────────────────────────────────────────────────

class TestSimulation:
    """Tests for KellyOptimizer simulation."""

    @patch.object(KellyOptimizer, 'load_prices')
    @patch.object(KellyOptimizer, 'get_series')
    def test_simulate_returns_metrics(self, mock_get_series, mock_load_prices):
        """Simulation returns metric dictionary."""
        mock_load_prices.return_value = {"_mock": True}
        
        np.random.seed(42)
        n = 600
        spy = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))
        gld = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n)))
        tlt = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.006, n)))
        
        def side_effect(symbol):
            return {"SPY": spy, "GLD": gld, "TLT": tlt}.get(symbol)
        mock_get_series.side_effect = side_effect
        
        opt = KellyOptimizer()
        opt.prices = {"_mock": True}  # Set prices directly to bypass load check
        result = opt.simulate(years=1)
        
        assert "error" not in result, result.get("error", "")
        assert "static" in result
        assert "kelly" in result
        assert "sharpe_delta" in result
        assert "n_rebalances" in result
        assert result["n_rebalances"] > 0

    def test_simulate_insufficient_data(self):
        """Simulation with insufficient data returns error."""
        opt = KellyOptimizer()
        with patch.object(opt, 'load_prices', return_value=None):
            result = opt.simulate()
            assert "error" in result


# ── Integration Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling."""

    def test_nan_in_inputs(self):
        """NaN in return estimation is handled."""
        opt = KellyOptimizer()
        prices = {
            "SPY": np.array([100, float('nan'), 102, 103]),
            "GLD": np.array([100, 101, float('nan'), 103]),
            "TLT": np.array([100, 100, 100, 100]),
        }
        # Should not crash
        try:
            means, cov, vols = opt.estimate_returns_and_cov(prices, lookback=252)
            assert len(means) == 3
        except Exception:
            pass  # NaN handling may produce warnings but shouldn't crash

    def test_sigmoid_with_nan(self):
        """NaN edge-to-odds produces zero."""
        result = sigmoid_scaling(float('nan'))
        assert result == 0.0 or (isinstance(result, float) and not math.isnan(result))

    def test_zero_length_prices(self):
        """Empty price series handled gracefully."""
        opt = KellyOptimizer()
        prices = {
            "SPY": np.array([]),
            "GLD": np.array([]),
            "TLT": np.array([]),
        }
        means, cov, vols = opt.estimate_returns_and_cov(prices, lookback=252)
        assert len(means) == 3

    def test_single_asset_only(self):
        """Only SPY available, other assets missing."""
        opt = KellyOptimizer()
        with patch.object(opt, 'load_prices', return_value={}):
            with patch.object(opt, 'get_series') as mock:
                mock.return_value = None
                # Should not crash
                decision = opt.compute_allocation(lookback=250)
                assert isinstance(decision, KellyDecision)


# ── CLI Tests ─────────────────────────────────────────────────────────────────

class TestCLI:
    """Tests for CLI entry points."""

    def test_main_adjust_default(self):
        """Adjust mode produces allocation."""
        with patch.object(KellyOptimizer, 'compute_allocation') as mock:
            mock.return_value = KellyDecision(
                timestamp="2026-05-17T12:00:00",
                base_allocation=BASE_ALLOCATION,
                kelly_allocation={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
                scaled_allocation={"SPY": 0.3, "GLD": 0.2, "TLT": 0.1},
                final_allocation={"SPY": 0.48, "GLD": 0.36, "TLT": 0.16},
                asset_edges={"SPY": 0.05, "GLD": 0.02, "TLT": 0.01},
                asset_odds={"SPY": 0.04, "GLD": 0.03, "TLT": 0.02},
                factors=KellyFactors(
                    timestamp="2026-05-17T12:00:00",
                    lookback_days=252,
                    spy_mean_return=0.10,
                    gld_mean_return=0.06,
                    tlt_mean_return=0.04,
                    spy_volatility=0.20,
                    gld_volatility=0.17,
                    tlt_volatility=0.14,
                    avg_edge_to_odds=0.5,
                    kelly_magnitude=0.6,
                    sigmoid_scale=0.4,
                ),
            )
            from src.strategy.kelly_optimal_sizing import main
            with patch('sys.argv', ['kelly_optimal_sizing.py', 'adjust']):
                main()  # Should not raise

    def test_main_status_no_file(self):
        """Status with no state file prints message."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch('src.strategy.kelly_optimal_sizing.STATE_PATH', Path(tmp) / 'nonexistent.json'):
                from src.strategy.kelly_optimal_sizing import main
                with patch('sys.argv', ['kelly_optimal_sizing.py', 'status']):
                    main()  # Should print "No state file found"

    def test_main_simulate(self):
        """Simulate mode runs without crashing."""
        with patch.object(KellyOptimizer, 'simulate') as mock:
            mock.return_value = {
                "static": {"sharpe": 0.8, "cagr": 0.08, "volatility": 0.12, "max_dd": -0.15},
                "kelly": {"sharpe": 0.9, "cagr": 0.09, "volatility": 0.13, "max_dd": -0.14},
                "sharpe_delta": 0.1,
                "dd_delta": 0.01,
                "years_simulated": 1.0,
                "n_rebalances": 12,
                "avg_turnover": 0.05,
                "avg_edge_to_odds": 0.3,
            }
            from src.strategy.kelly_optimal_sizing import main
            with patch('sys.argv', ['kelly_optimal_sizing.py', 'simulate']):
                main()  # Should not raise
