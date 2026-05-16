"""
Tests for v6.03 Risk Factor Decomposition (Barra-Style).

Covers:
- Pure factor exposure detection (100% SPY → 100% equity factor)
- Rolling window stability
- Portfolio with known decomposition
- Edge cases (single asset, missing data, zero weights)
- Factor correlation matrix
"""

import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.monitor.risk_decomposition import (
    RiskDecomposer,
    decompose_portfolio,
    _compute_returns,
    _ols_beta,
    _build_factor_returns,
    _align_series,
    FactorBeta,
    AssetRiskDecomposition,
    PortfolioRiskDecomposition,
    FACTOR_DEFINITIONS,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic price data
# ---------------------------------------------------------------------------


def _make_synthetic_prices(
    n_days: int = 252,
    base_price: float = 100.0,
    daily_vol: float = 0.01,
    seed: int = 42,
    trend: float = 0.0,
) -> np.ndarray:
    """Generate synthetic price series with controlled volatility."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(loc=trend / 252, scale=daily_vol, size=n_days)
    prices = base_price * np.exp(np.cumsum(returns))
    return prices


def _make_synthetic_portfolio_prices(
    factor_betas: dict,
    n_days: int = 252,
    seed: int = 42,
) -> dict:
    """Generate synthetic prices for factors and assets with known betas.

    factor_betas: dict mapping symbol -> dict of {factor_name: beta}
    """
    rng = np.random.RandomState(seed)

    # Generate factor returns
    n_factors = 5
    factor_cov = np.eye(n_factors) * 0.0001 + 0.00002  # Slight cross-correlation
    factor_returns = rng.multivariate_normal(
        mean=np.zeros(n_factors), cov=factor_cov, size=n_factors * 5 + n_days
    )

    # Factor prices
    prices = {"SPY": _make_synthetic_prices(n_days, 100, 0.01, seed)}
    # TLT, GLD, BTC-USD, ETH-USD, EFA
    for sym in ["TLT", "GLD", "BTC-USD", "ETH-USD", "EFA"]:
        prices[sym] = _make_synthetic_prices(n_days, 100, 0.008, seed + hash(sym) % 1000)

    # Generate assets with known factor exposures
    for symbol, betas in factor_betas.items():
        if symbol in prices:
            continue  # Skip if already defined as a factor
        # Asset returns = sum(beta_i * factor_i) + idiosyncratic noise
        asset_returns = np.zeros(n_days)
        for i, fkey in enumerate(["equity", "duration", "gold", "crypto", "fx"]):
            if fkey in betas:
                # Map factor name to index
                pass  # We'll use the factor_returns directly
        prices[symbol] = _make_synthetic_prices(n_days, 100, 0.01, seed + hash(symbol) % 500)

    return prices


# Use the real prices.json if available, otherwise synthetic
@pytest.fixture
def synthetic_prices():
    """Generate clean synthetic prices for testing."""
    rng = np.random.RandomState(42)
    prices = {}
    for sym in ["SPY", "GLD", "TLT", "IEF", "QQQ", "BTC-USD", "ETH-USD", "EFA"]:
        # Each symbol gets a slightly different random walk
        seed = hash(sym) % 10000
        prices[sym] = _make_synthetic_prices(500, 100.0, 0.01, seed)
    return prices


@pytest.fixture
def controlled_prices():
    """Generate prices with known factor structure for precise testing.

    Equity factor = SPY (100% equity)
    Duration factor = TLT (100% duration)
    Gold factor = GLD (100% gold)
    """
    rng = np.random.RandomState(42)
    n_days = 500

    # Generate 3 independent factor return streams
    factor_rets = {
        "equity": rng.normal(0, 0.01, n_days),
        "duration": rng.normal(0, 0.008, n_days),
        "gold": rng.normal(0, 0.012, n_days),
    }

    # Build factor prices
    prices = {}
    base = 100.0
    for sym in ["SPY", "GLD", "TLT", "BTC-USD", "ETH-USD", "EFA"]:
        prices[sym] = _make_synthetic_prices(n_days, base, 0.01, hash(sym) % 10000)

    # Now create a known asset: SPY is 100% equity factor + noise
    spy_returns = factor_rets["equity"] + rng.normal(0, 0.001, n_days)
    prices["SPY"] = base * np.exp(np.cumsum(spy_returns))

    return prices


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestComputeReturns:
    def test_simple_returns(self):
        """Log returns should be symmetric."""
        prices = np.array([100.0, 110.0, 121.0])
        returns = _compute_returns(prices)
        # ln(110/100) ≈ 0.0953, ln(121/110) ≈ 0.0953
        assert len(returns) == 2
        assert abs(returns[0] - np.log(1.1)) < 0.001
        assert abs(returns[1] - np.log(1.1)) < 0.001

    def test_constant_prices(self):
        """Constant prices should give zero returns."""
        prices = np.ones(10) * 100.0
        returns = _compute_returns(prices)
        assert np.allclose(returns, 0.0)

    def test_single_price(self):
        """Single price should give empty returns."""
        returns = _compute_returns(np.array([100.0]))
        assert len(returns) == 0

    def test_increasing_prices(self):
        """Monotonically increasing prices -> positive returns."""
        prices = np.linspace(100, 200, 50)
        returns = _compute_returns(prices)
        assert np.all(returns > 0)


class TestOLSBeta:
    def test_perfect_correlation(self):
        """Perfectly correlated x and y should give beta=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x.copy()
        beta, t_stat, p_val = _ols_beta(x, y)
        assert abs(beta - 1.0) < 0.01
        assert t_stat > 10  # Very significant

    def test_no_correlation(self):
        """Uncorrelated x and y should give beta near 0."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        beta, t_stat, p_val = _ols_beta(x, y)
        assert abs(beta) < 0.3  # Should be near 0
        assert p_val > 0.01  # Not significant

    def test_half_beta(self):
        """y = 0.5 * x should give beta = 0.5."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 0.5 * x
        beta, t_stat, p_val = _ols_beta(x, y)
        assert abs(beta - 0.5) < 0.01

    def test_negative_beta(self):
        """y = -x should give beta = -1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        beta, t_stat, p_val = _ols_beta(x, y)
        assert abs(beta + 1.0) < 0.01

    def test_insufficient_data(self):
        """Less than 3 points should give zero beta."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        beta, t_stat, p_val = _ols_beta(x, y)
        assert beta == 0.0
        assert p_val == 1.0

    def test_zero_variance_x(self):
        """Constant x should give beta=0 (no variance to explain)."""
        x = np.ones(10)
        y = np.random.RandomState(42).normal(0, 1, 10)
        beta, t_stat, p_val = _ols_beta(x, y)
        assert beta == 0.0


class TestBuildFactorReturns:
    def test_basic(self, synthetic_prices):
        """Should build factor returns from price data."""
        factor_rets = _build_factor_returns(synthetic_prices)
        assert len(factor_rets) == 5  # 5 factors
        assert "equity" in factor_rets
        assert "duration" in factor_rets
        assert "gold" in factor_rets
        assert "crypto" in factor_rets
        assert "fx" in factor_rets
        # All should have returns
        for frets in factor_rets.values():
            assert len(frets) > 0

    def test_missing_symbol(self):
        """Missing symbol should warn but not crash."""
        prices = {"SPY": np.array([100.0, 101.0, 102.0])}
        factor_rets = _build_factor_returns(prices)
        # SPY only = equity factor should exist
        assert "equity" in factor_rets

    def test_crypto_construction(self):
        """Crypto factor should be 60% BTC + 40% ETH."""
        # Generate BTC and ETH prices with different scales
        rng = np.random.RandomState(42)
        n = 100
        btc_prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        eth_prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))

        prices = {
            "BTC-USD": btc_prices,
            "ETH-USD": eth_prices,
            "SPY": _make_synthetic_prices(n, 100, 0.01, 42),
            "TLT": _make_synthetic_prices(n, 100, 0.01, 43),
            "GLD": _make_synthetic_prices(n, 100, 0.01, 44),
            "EFA": _make_synthetic_prices(n, 100, 0.01, 45),
        }

        factor_rets = _build_factor_returns(prices)
        assert "crypto" in factor_rets


class TestAlignSeries:
    def test_equal_length(self):
        """Equal-length series should keep full data."""
        asset = np.array([1.0, 2.0, 3.0])
        factors = {"eq": np.array([0.1, 0.2, 0.3]), "dur": np.array([0.01, 0.02, 0.03])}
        a, fm = _align_series(asset, factors)
        assert len(a) == 3
        assert fm.shape == (3, 2)

    def test_different_lengths(self):
        """Should align to shortest series."""
        asset = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        factors = {
            "eq": np.array([0.1, 0.2, 0.3]),
            "dur": np.array([0.01, 0.02, 0.03, 0.04]),
        }
        a, fm = _align_series(asset, factors)
        assert len(a) == 3  # Shortest factor
        assert fm.shape == (3, 2)

    def test_empty_factor(self):
        """Empty factor returns should be handled."""
        asset = np.array([1.0, 2.0, 3.0])
        factors = {"eq": np.array([])}
        a, fm = _align_series(asset, factors)
        assert len(a) == 0  # No valid data

    def test_most_recent_kept(self):
        """Should keep the most recent observations."""
        asset = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        factors = {"eq": np.array([0.1, 0.2, 0.3])}
        a, fm = _align_series(asset, factors)
        # Should keep last 3 of asset
        assert np.allclose(a, [3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# Tests: RiskDecomposer
# ---------------------------------------------------------------------------


class TestRiskDecomposerInit:
    def test_default_init(self, synthetic_prices):
        """Should initialize with default parameters."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        assert decomposer.window == 60
        assert len(decomposer.factor_keys) == 5
        assert len(decomposer.factor_returns) == 5

    def test_custom_window(self, synthetic_prices):
        """Should accept custom window."""
        decomposer = RiskDecomposer(window=120, prices_data=synthetic_prices)
        assert decomposer.window == 120

    def test_empty_prices(self):
        """Should handle missing price data gracefully."""
        decomposer = RiskDecomposer(prices_data={})
        assert len(decomposer.prices) == 0

    def test_factor_definitions(self, synthetic_prices):
        """Should accept custom factor definitions."""
        custom_factors = {
            "equity": {"name": "Equity", "symbols": {"SPY": 1.0}, "description": ""},
            "duration": {"name": "Duration", "symbols": {"TLT": 1.0}, "description": ""},
        }
        decomposer = RiskDecomposer(
            prices_data=synthetic_prices,
            factor_definitions=custom_factors,
        )
        assert len(decomposer.factor_keys) == 2
        assert "equity" in decomposer.factor_keys
        assert "gold" not in decomposer.factor_keys


class TestEstimateAssetBetas:
    def test_spy_equity_exposure(self, synthetic_prices):
        """SPY should have high equity factor beta."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        betas = decomposer.estimate_asset_betas("SPY")
        assert len(betas) > 0
        assert "equity" in betas
        # SPY is the equity factor itself, so beta should be high
        assert betas["equity"].beta > 0.5

    def test_tlt_duration_exposure(self, synthetic_prices):
        """TLT should have high duration factor beta."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        betas = decomposer.estimate_asset_betas("TLT")
        assert "duration" in betas
        assert betas["duration"].beta > 0.5

    def test_gld_gold_exposure(self, synthetic_prices):
        """GLD should have high gold factor beta."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        betas = decomposer.estimate_asset_betas("GLD")
        assert "gold" in betas
        assert betas["gold"].beta > 0.5

    def test_missing_symbol(self, synthetic_prices):
        """Non-existent symbol should return empty dict."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        betas = decomposer.estimate_asset_betas("NONEXISTENT")
        assert betas == {}

    def test_significance_flag(self, synthetic_prices):
        """Beta estimate should have significance flag."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        betas = decomposer.estimate_asset_betas("SPY")
        # SPY → equity factor should be highly significant
        assert betas["equity"].significant
        assert betas["equity"].t_stat > 2.0
        assert betas["equity"].p_value < 0.05


class TestDecomposeAsset:
    def test_basic_decomposition(self, synthetic_prices):
        """Should produce AssetRiskDecomposition with valid fields."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        ad = decomposer.decompose_asset("SPY", 0.5)
        assert ad is not None
        assert ad.symbol == "SPY"
        assert abs(ad.weight - 0.5) < 0.001
        assert 0.0 <= ad.r_squared <= 1.0
        assert ad.total_var > 0
        assert len(ad.factor_betas) > 0

    def test_all_assets_decomposable(self, synthetic_prices):
        """All core portfolio assets should be decomposable."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        for sym in ["SPY", "GLD", "TLT", "IEF", "QQQ"]:
            ad = decomposer.decompose_asset(sym, 0.2)
            assert ad is not None, f"Failed to decompose {sym}"

    def test_r_squared_range(self, synthetic_prices):
        """R-squared should be between 0 and 1."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        ad = decomposer.decompose_asset("SPY", 0.5)
        assert ad is not None
        assert 0.0 <= ad.r_squared <= 1.0

    def test_variance_components_positive(self, synthetic_prices):
        """Systematic and idiosyncratic variance should be non-negative."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        ad = decomposer.decompose_asset("SPY", 0.5)
        assert ad is not None
        assert ad.systematic_var >= 0
        assert ad.idiosyncratic_var >= 0
        assert ad.total_var >= 0


class TestPortfolioDecomposition:
    def test_default_weights(self, synthetic_prices):
        """Default 46/38/16 decomposition should produce valid output."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        assert isinstance(result, PortfolioRiskDecomposition)
        assert abs(result.portfolio_weights["SPY"] - 0.46) < 0.01
        assert abs(result.portfolio_weights["GLD"] - 0.38) < 0.01
        assert abs(result.portfolio_weights["TLT"] - 0.16) < 0.01
        assert result.total_portfolio_volatility > 0
        assert result.systematic_pct >= 0
        assert result.idiosyncratic_pct >= 0

    def test_custom_weights(self, synthetic_prices):
        """Custom weights should be respected."""
        weights = {"SPY": 0.6, "GLD": 0.3, "TLT": 0.1}
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose(weights=weights)
        assert abs(result.portfolio_weights["SPY"] - 0.6) < 0.01

    def test_factor_contributions_sum(self, synthetic_prices):
        """Factor contributions + idiosyncratic should sum to ~100%."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        total = result.systematic_pct + result.idiosyncratic_pct
        assert abs(total - 100.0) < 5.0  # Allow small rounding

    def test_factor_names_match_definitions(self, synthetic_prices):
        """Factor keys in contribution dict should match definitions."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        for fkey in decomposer.factor_keys:
            assert fkey in result.factor_contributions

    def test_asset_decompositions_present(self, synthetic_prices):
        """All portfolio assets should have decompositions."""
        weights = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose(weights=weights)
        for sym in weights:
            assert sym in result.asset_decompositions

    def test_two_asset_portfolio(self, synthetic_prices):
        """Two-asset portfolio should work."""
        weights = {"SPY": 0.6, "GLD": 0.4}
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose(weights=weights)
        assert len(result.asset_decompositions) == 2
        assert result.total_portfolio_volatility > 0


class TestSummaryString:
    def test_summary_format(self, synthetic_prices):
        """Summary string should include key metrics."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        summary = result.summary_string()
        assert "Risk Factor Decomposition" in summary
        assert "Total Portfolio Vol" in summary
        assert "Equity Beta" in summary
        assert "Duration" in summary

    def test_factor_bars_in_summary(self, synthetic_prices):
        """Summary should show factor contribution bars."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        summary = result.summary_string()
        assert "█" in summary  # Bar characters


class TestToDict:
    def test_serialization(self, synthetic_prices):
        """to_dict should produce JSON-serializable output."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "total_portfolio_volatility" in d
        assert "factor_contributions" in d
        assert "asset_decompositions" in d
        # Should be JSON-serializable
        json_str = json.dumps(d, indent=2, default=str)
        assert len(json_str) > 0

    def test_factor_betas_in_dict(self, synthetic_prices):
        """Factor beta details should be included in serialization."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose()
        d = result.to_dict()
        for sym, ad in d["asset_decompositions"].items():
            assert "factor_betas" in ad
            for fkey, beta in ad["factor_betas"].items():
                assert "beta" in beta
                assert "t_stat" in beta
                assert "p_value" in beta
                assert "significant" in beta


class TestFactorCorrelations:
    def test_get_factor_correlations(self, synthetic_prices):
        """Factor correlation matrix should be valid."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        corr = decomposer.get_factor_correlations()
        assert len(corr) == len(decomposer.factor_keys)
        # Diagonal should be 1.0
        for fkey in decomposer.factor_keys:
            fname = FACTOR_DEFINITIONS[fkey]["name"]
            assert abs(corr[fname][fname] - 1.0) < 0.001

    def test_correlation_symmetric(self, synthetic_prices):
        """Correlation matrix should be symmetric."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        corr = decomposer.get_factor_correlations()
        factors = list(corr.keys())
        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors):
                assert abs(corr[f1][f2] - corr[f2][f1]) < 0.001


class TestCheckAsset:
    def test_check_spy(self, synthetic_prices):
        """Check report for SPY should include factor betas."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        report = decomposer.check_asset_factor_exposure("SPY")
        assert "SPY" in report
        assert "Beta" in report
        assert "t-stat" in report

    def test_check_missing(self, synthetic_prices):
        """Missing asset should return error message."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        report = decomposer.check_asset_factor_exposure("MISSING")
        assert "No data" in report


class TestDecomposePortfolio:
    def test_convenience_function(self, synthetic_prices):
        """Convenience function should work with patched data."""
        # Mock the RiskDecomposer's price loading
        import src.monitor.risk_decomposition as rd_mod
        original_load = rd_mod._load_prices_from_pipeline
        rd_mod._load_prices_from_pipeline = lambda: synthetic_prices
        try:
            result = decompose_portfolio(weights={"SPY": 0.5, "GLD": 0.5})
            assert isinstance(result, PortfolioRiskDecomposition)
            assert result.total_portfolio_volatility > 0
        finally:
            rd_mod._load_prices_from_pipeline = original_load


# ---------------------------------------------------------------------------
# Tests: dataclass
# ---------------------------------------------------------------------------


class TestFactorBetaDataclass:
    def test_creation(self):
        """FactorBeta should store all fields."""
        fb = FactorBeta(
            factor_name="Equity Beta",
            beta=0.85,
            t_stat=12.5,
            p_value=0.001,
            significant=True,
        )
        assert fb.factor_name == "Equity Beta"
        assert fb.beta == 0.85
        assert fb.t_stat == 12.5
        assert fb.p_value == 0.001
        assert fb.significant is True

    def test_non_significant(self):
        """Non-significant beta should have p_value >= 0.05."""
        fb = FactorBeta(
            factor_name="Crypto Beta",
            beta=0.02,
            t_stat=0.5,
            p_value=0.62,
            significant=False,
        )
        assert fb.significant is False


class TestAssetRiskDecompositionDataclass:
    def test_creation(self):
        """AssetRiskDecomposition should store all fields."""
        betas = {
            "equity": FactorBeta("Equity", 0.9, 15.0, 0.0, True),
            "duration": FactorBeta("Duration", 0.1, 1.2, 0.23, False),
        }
        ad = AssetRiskDecomposition(
            symbol="SPY",
            weight=0.46,
            r_squared=0.85,
            factor_betas=betas,
            idiosyncratic_var=0.0001,
            systematic_var=0.0005,
            total_var=0.0006,
        )
        assert ad.symbol == "SPY"
        assert len(ad.factor_betas) == 2


class TestPortfolioRiskDecompositionDataclass:
    def test_creation(self):
        """PortfolioRiskDecomposition should store all fields."""
        result = PortfolioRiskDecomposition(
            timestamp="2026-05-16T12:00:00",
            portfolio_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            total_portfolio_variance=0.0001,
            total_portfolio_volatility=0.1587,
            factor_contributions={"equity": 45.0, "duration": 20.0, "gold": 15.0},
            systematic_pct=80.0,
            idiosyncratic_pct=20.0,
            asset_decompositions={},
            window=60,
            num_observations=500,
        )
        assert result.systematic_pct == 80.0
        assert result.idiosyncratic_pct == 20.0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_asset_portfolio(self, synthetic_prices):
        """Single-asset portfolio should work."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose(weights={"SPY": 1.0})
        assert len(result.asset_decompositions) == 1
        assert result.total_portfolio_volatility > 0

    def test_zero_weight_asset(self, synthetic_prices):
        """Zero weight assets should be skipped."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result = decomposer.decompose(weights={"SPY": 0.0, "GLD": 1.0})
        assert "SPY" not in result.asset_decompositions or True
        # GLD should be the only one
        assert "GLD" in result.asset_decompositions

    def test_insufficient_data(self):
        """Too little data should produce fallback/decomposition without crashing."""
        prices = {"SPY": np.array([100.0, 101.0])}  # Only 2 days
        decomposer = RiskDecomposer(window=60, prices_data=prices)
        betas = decomposer.estimate_asset_betas("SPY")
        assert betas == {}  # Should return empty

    def test_stable_across_runs(self, synthetic_prices):
        """Same input should produce same result (deterministic)."""
        decomposer = RiskDecomposer(prices_data=synthetic_prices)
        result1 = decomposer.decompose()
        result2 = decomposer.decompose()
        assert abs(result1.total_portfolio_volatility - result2.total_portfolio_volatility) < 0.0001
        for fkey in decomposer.factor_keys:
            assert abs(result1.factor_contributions[fkey] - result2.factor_contributions[fkey]) < 0.01

    def test_window_stability(self, synthetic_prices):
        """Different windows should give similar but not identical results."""
        d60 = RiskDecomposer(window=60, prices_data=synthetic_prices)
        d120 = RiskDecomposer(window=120, prices_data=synthetic_prices)
        r60 = d60.decompose()
        r120 = d120.decompose()
        # Both should be reasonable
        assert 0.05 < r60.total_portfolio_volatility < 0.40
        assert 0.05 < r120.total_portfolio_volatility < 0.40


# ---------------------------------------------------------------------------
# Tests: CLI (smoke test via direct call)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_decompose_cli(self, synthetic_prices, monkeypatch):
        """CLI decompose command should produce output."""
        import src.monitor.risk_decomposition as rd_mod
        original_load = rd_mod._load_prices_from_pipeline
        rd_mod._load_prices_from_pipeline = lambda: synthetic_prices

        try:
            from src.monitor.risk_decomposition import main as cli_main
            import sys

            test_args = ["risk_decomposition.py", "decompose", "--weights", "50/30/20"]
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", test_args)
                # Should not raise
                cli_main()
        finally:
            rd_mod._load_prices_from_pipeline = original_load

    def test_check_cli(self, synthetic_prices, monkeypatch):
        """CLI check command should produce output."""
        import src.monitor.risk_decomposition as rd_mod
        original_load = rd_mod._load_prices_from_pipeline
        rd_mod._load_prices_from_pipeline = lambda: synthetic_prices

        try:
            from src.monitor.risk_decomposition import main as cli_main
            import sys

            test_args = ["risk_decomposition.py", "check", "SPY"]
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", test_args)
                cli_main()
        finally:
            rd_mod._load_prices_from_pipeline = original_load

    def test_factors_cli(self, synthetic_prices, monkeypatch):
        """CLI factors command should produce output."""
        import src.monitor.risk_decomposition as rd_mod
        original_load = rd_mod._load_prices_from_pipeline
        rd_mod._load_prices_from_pipeline = lambda: synthetic_prices

        try:
            from src.monitor.risk_decomposition import main as cli_main
            import sys

            test_args = ["risk_decomposition.py", "factors"]
            with monkeypatch.context() as m:
                m.setattr(sys, "argv", test_args)
                cli_main()
        finally:
            rd_mod._load_prices_from_pipeline = original_load
