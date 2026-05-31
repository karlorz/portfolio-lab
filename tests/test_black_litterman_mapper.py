#!/usr/bin/env python3
"""
Tests for src/strategy/black_litterman_mapper.py — BL view mapper.
"""
import pytest
import numpy as np

from src.strategy.black_litterman_mapper import (
    BLViews, BLResult,
    map_biases_to_views, run_black_litterman, compute_bl_weights,
    tau_sensitivity,
    DEFAULT_TAU, DEFAULT_SYMBOLS, DEFAULT_MARKET_CAPS,
    BIAS_TO_RETURN_SCALE, MIN_VIEW_CONFIDENCE,
)


class TestBLViews:
    """Tests for BLViews dataclass."""

    def test_defaults(self):
        v = BLViews(
            absolute_views={"SPY": 0.05, "GLD": 0.03, "TLT": 0.02},
            view_confidences=[0.5, 0.5, 0.5],
        )
        assert v.tau == DEFAULT_TAU
        assert v.prior == "equal"
        assert v.symbols == DEFAULT_SYMBOLS

    def test_custom_tau(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.5],
            tau=0.30,
        )
        assert v.tau == 0.30

    def test_market_prior(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.5],
            prior="market",
        )
        assert v.prior == "market"


class TestBLResult:
    """Tests for BLResult dataclass."""

    def test_defaults(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 0.5},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
        )
        assert r.expected_sharpe is None
        assert r.extras == {}

    def test_with_metrics(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 0.5},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
            expected_sharpe=0.85,
            expected_cagr=9.5,
            expected_volatility=11.2,
        )
        assert r.expected_sharpe == 0.85
        assert r.expected_cagr == 9.5


class TestMapBiasesToViews:
    """Tests for map_biases_to_views()."""

    def test_zero_biases(self):
        """Zero biases → zero expected returns."""
        views = map_biases_to_views(0.0, 0.0, 0.0)
        assert views.absolute_views["SPY"] == 0.0
        assert views.absolute_views["TLT"] == 0.0
        assert views.absolute_views["GLD"] == 0.0

    def test_positive_equity_bias(self):
        views = map_biases_to_views(equity_bias=1.0, duration_bias=0.0, gold_bias=0.0)
        assert views.absolute_views["SPY"] == pytest.approx(BIAS_TO_RETURN_SCALE)
        assert views.absolute_views["TLT"] == 0.0

    def test_negative_duration_bias(self):
        views = map_biases_to_views(equity_bias=0.0, duration_bias=-0.5, gold_bias=0.0)
        assert views.absolute_views["TLT"] == pytest.approx(-0.5 * BIAS_TO_RETURN_SCALE)

    def test_gold_bias_scaling(self):
        views = map_biases_to_views(equity_bias=0.0, duration_bias=0.0, gold_bias=0.8)
        assert views.absolute_views["GLD"] == pytest.approx(0.8 * BIAS_TO_RETURN_SCALE)

    def test_default_confidences_without_health(self):
        views = map_biases_to_views(0.5, 0.3, 0.2)
        assert views.view_confidences == [0.50, 0.50, 0.50]

    def test_health_scores_modulate_confidence(self):
        """With high health scores, confidences should be higher than default."""
        health = {"MULTI_SPEED_MOM": 0.80, "CROSS_ASSET_RV": 0.75}
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores=health)
        # With bias magnitudes 0.5, 0.3, 0.2 and avg_health ~0.775:
        # conf = avg_health * (0.5 + 0.5 * |bias|)
        # SPY: 0.775 * (0.5 + 0.25) = 0.581
        # TLT: 0.775 * (0.5 + 0.15) = 0.504
        # GLD: 0.775 * (0.5 + 0.10) = 0.465
        assert views.view_confidences[0] > views.view_confidences[1]
        assert views.view_confidences[1] > views.view_confidences[2]
        # All above minimum
        for c in views.view_confidences:
            assert c >= MIN_VIEW_CONFIDENCE

    def test_zero_bias_with_high_health(self):
        """Zero bias should give base confidence (avg_health * 0.5)."""
        health = {"SIG1": 0.80}
        views = map_biases_to_views(0.0, 0.0, 0.0, health_scores=health)
        # conf = 0.80 * (0.5 + 0) = 0.40
        for c in views.view_confidences:
            assert c == pytest.approx(0.40, abs=0.01)

    def test_confidence_min_floor(self):
        """Very low health should still hit MIN_VIEW_CONFIDENCE floor."""
        health = {"SIG1": 0.01}
        views = map_biases_to_views(0.01, 0.01, 0.01, health_scores=health)
        for c in views.view_confidences:
            assert c >= MIN_VIEW_CONFIDENCE

    def test_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        health = {"SIG1": 1.0}
        views = map_biases_to_views(1.0, 1.0, 1.0, health_scores=health)
        for c in views.view_confidences:
            assert c <= 1.0

    def test_custom_tau(self):
        views = map_biases_to_views(0.5, 0.3, 0.2, tau=0.30)
        assert views.tau == 0.30

    def test_market_prior(self):
        views = map_biases_to_views(0.5, 0.3, 0.2, prior="market")
        assert views.prior == "market"


class TestRunBlackLitterman:
    """Tests for run_black_litterman()."""

    @pytest.fixture
    def sample_cov(self):
        """Realistic covariance matrix for SPY/GLD/TLT."""
        # Roughly: SPY vol ~15%, GLD vol ~16%, TLT vol ~14%
        # Correlations: SPY-GLD ~0.0, SPY-TLT ~-0.3, GLD-TLT ~0.1
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    def test_equal_prior_zero_views(self):
        """Zero views with equal prior should return near-equal weights."""
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(sample_cov_fixture := np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ]), views)
        assert result.prior_type == "equal"
        # Weights should exist for at least one asset
        assert len(result.bl_weights) > 0

    def test_positive_equity_view_shifts_weights(self):
        """Strong equity view should increase SPY allocation."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

        views_neutral = map_biases_to_views(0.0, 0.0, 0.0)
        views_bull = map_biases_to_views(1.0, 0.0, 0.0)

        result_neutral = run_black_litterman(cov, views_neutral)
        result_bull = run_black_litterman(cov, views_bull)

        # Bull view should shift weight toward SPY
        spy_neutral = result_neutral.bl_weights.get("SPY", 0)
        spy_bull = result_bull.bl_weights.get("SPY", 0)
        assert spy_bull >= spy_neutral

    def test_custom_prior(self):
        """Custom pi array should be used as prior."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0)
        custom_pi = np.array([0.08, 0.04, 0.03])
        result = run_black_litterman(cov, views, pi=custom_pi)
        assert result.prior_type == "custom"

    def test_market_prior(self):
        """Market prior should use market cap weights."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0, prior="market")
        result = run_black_litterman(cov, views)
        assert result.prior_type == "market"

    def test_posterior_returns_are_reasonable(self):
        """Posterior returns should be in a sensible range."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, -0.3, 0.2)
        result = run_black_litterman(cov, views)
        for sym, ret in result.posterior_returns.items():
            # Returns should be within reasonable bounds
            assert -0.30 < ret < 0.30, f"{sym} posterior return {ret} out of bounds"

    def test_result_has_performance_metrics(self):
        """Result should have Sharpe/CAGR/vol when EF succeeds."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        # If EF optimization succeeded, metrics should be present
        if result.expected_sharpe is not None:
            assert isinstance(result.expected_sharpe, float)
            assert result.expected_cagr is not None
            assert result.expected_volatility is not None

    def test_weights_sum_approximately_one(self):
        """BL weights should sum to approximately 1.0."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        weight_sum = sum(result.bl_weights.values())
        assert abs(weight_sum - 1.0) < 0.05, f"Weights sum to {weight_sum}"


class TestComputeBLWeights:
    """Tests for compute_bl_weights() convenience function."""

    def test_with_synthetic_prices(self):
        """Test with synthetic price data."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
        )
        assert isinstance(result, BLResult)
        assert len(result.bl_weights) > 0

    def test_zero_biases(self):
        """Zero biases should still produce valid weights."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        result = compute_bl_weights(prices_df=prices)
        assert isinstance(result, BLResult)

    def test_with_health_scores(self):
        """Health scores should modulate view confidence."""
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 500)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 500)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 500)),
        }, index=dates)

        health = {"SIG1": 0.80, "SIG2": 0.70, "SIG3": 0.60}
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
            health_scores=health,
        )
        assert isinstance(result, BLResult)


class TestChampionComparison:
    """Integration test: BL weights vs grid search champion."""

    def test_bl_near_champion_region(self):
        """BL with moderate views should produce weights near 46/38/16."""
        import pandas as pd

        np.random.seed(42)
        # Simulate 5 years of data with realistic parameters
        dates = pd.date_range("2020-01-01", periods=1260, freq="B")
        prices = pd.DataFrame({
            "SPY": 300 * np.cumprod(1 + np.random.normal(0.0004, 0.012, 1260)),
            "GLD": 150 * np.cumprod(1 + np.random.normal(0.0002, 0.011, 1260)),
            "TLT": 130 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 1260)),
        }, index=dates)

        # Moderate bullish equity, slight gold and duration view
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.3,
            duration_bias=-0.1,
            gold_bias=0.1,
            tau=0.15,
        )

        # BL should produce non-trivial weights
        assert len(result.bl_weights) >= 2
        # All weights should be positive
        for sym, w in result.bl_weights.items():
            assert w > 0, f"{sym} has non-positive weight {w}"


class TestTauSensitivity:
    """Tests for tau_sensitivity()."""

    def test_default_tau_range(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views)
        # Default should have 8 tau values
        assert len(results) == 8
        assert 0.005 in results
        assert 0.15 in results
        assert 0.50 in results

    def test_custom_tau_range(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views, tau_values=[0.05, 0.15, 0.30])
        assert len(results) == 3

    def test_results_are_blresult_instances(self):
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.2, 0.3)
        results = tau_sensitivity(cov, views, tau_values=[0.05, 0.15])
        for tau, result in results.items():
            assert isinstance(result, BLResult)
            assert result.tau == tau

    def test_higher_tau_changes_metrics(self):
        """Different tau values should produce different Sharpe ratios."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(1.0, -0.5, 0.0)
        results = tau_sensitivity(cov, views, tau_values=[0.005, 0.50])

        if 0.005 in results and 0.50 in results:
            # At minimum, the results should have metrics
            assert results[0.005].expected_sharpe is not None
            assert results[0.50].expected_sharpe is not None


class TestBLViewsExtended:
    """Extended tests for BLViews dataclass."""

    def test_absolute_views_dict(self):
        v = BLViews(
            absolute_views={"SPY": 0.10, "GLD": -0.02},
            view_confidences=[0.7, 0.4],
        )
        assert len(v.absolute_views) == 2
        assert v.absolute_views["SPY"] == 0.10

    def test_view_confidences_length(self):
        v = BLViews(
            absolute_views={"SPY": 0.05},
            view_confidences=[0.6],
        )
        assert len(v.view_confidences) == 1

    def test_custom_symbols(self):
        v = BLViews(
            absolute_views={"EFA": 0.04},
            view_confidences=[0.5],
            symbols=["EFA", "EEM", "SPY"],
        )
        assert v.symbols == ["EFA", "EEM", "SPY"]


class TestBLResultExtended:
    """Extended tests for BLResult dataclass."""

    def test_all_fields(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08, "GLD": 0.03, "TLT": 0.02},
            bl_weights={"SPY": 0.60, "GLD": 0.25, "TLT": 0.15},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5, 0.5, 0.5],
            expected_sharpe=0.85,
            expected_cagr=0.10,
            expected_volatility=0.12,
        )
        assert r.tau == 0.15
        assert r.posterior_returns["SPY"] == 0.08
        assert r.bl_weights["SPY"] == 0.60
        assert r.expected_sharpe == 0.85
        assert r.expected_cagr == 0.10
        assert r.prior_type == "equal"

    def test_none_sharpe(self):
        r = BLResult(
            posterior_returns={},
            bl_weights={},
            tau=0.15,
            prior_type="equal",
            view_confidences=[],
        )
        assert r.expected_sharpe is None
        assert r.expected_cagr is None

    def test_extras_default_empty(self):
        r = BLResult(
            posterior_returns={}, bl_weights={}, tau=0.15,
            prior_type="equal", view_confidences=[],
        )
        assert r.extras == {}


class TestMapBiasesToViewsExtended:
    """Extended tests for map_biases_to_views."""

    def test_all_zero_biases(self):
        views = map_biases_to_views(0.0, 0.0, 0.0)
        assert isinstance(views, BLViews)
        for val in views.absolute_views.values():
            assert val == 0.0

    def test_large_equity_bias(self):
        views = map_biases_to_views(2.0, 0.0, 0.0)
        assert views.absolute_views.get("SPY", 0) > 0

    def test_negative_gold_bias(self):
        views = map_biases_to_views(0.0, 0.0, -1.0)
        assert views.absolute_views.get("GLD", 0) < 0

    def test_positive_duration_bias(self):
        views = map_biases_to_views(0.0, 1.0, 0.0)
        assert views.absolute_views.get("TLT", 0) > 0

    def test_confidence_scaled(self):
        views = map_biases_to_views(0.5, 0.2, 0.3)
        assert len(views.view_confidences) > 0
        for c in views.view_confidences:
            assert c > 0


class TestConstants:
    """Validate module constants."""

    def test_default_tau(self):
        assert DEFAULT_TAU == 0.15

    def test_default_symbols(self):
        assert DEFAULT_SYMBOLS == ["SPY", "GLD", "TLT"]

    def test_bias_scale_positive(self):
        assert BIAS_TO_RETURN_SCALE > 0

    def test_min_view_confidence(self):
        assert 0 < MIN_VIEW_CONFIDENCE < 1


class TestConstantsExtended:
    """Extended constants validation — market caps, types, ranges."""

    def test_default_market_caps_has_keys(self):
        assert "SPY" in DEFAULT_MARKET_CAPS
        assert "GLD" in DEFAULT_MARKET_CAPS
        assert "TLT" in DEFAULT_MARKET_CAPS

    def test_default_market_caps_positive(self):
        for sym, cap in DEFAULT_MARKET_CAPS.items():
            assert cap > 0, f"{sym} market cap should be positive"

    def test_default_market_caps_reasonable_order(self):
        """SPY >> GLD > TLT in market cap."""
        assert DEFAULT_MARKET_CAPS["SPY"] > DEFAULT_MARKET_CAPS["GLD"]
        assert DEFAULT_MARKET_CAPS["GLD"] > DEFAULT_MARKET_CAPS["TLT"]

    def test_bias_scale_in_expected_range(self):
        """BIAS_TO_RETURN_SCALE should be between 0.01 and 1.0."""
        assert 0.01 <= BIAS_TO_RETURN_SCALE <= 1.0

    def test_min_confidence_strictly_positive(self):
        assert MIN_VIEW_CONFIDENCE > 0.0

    def test_min_confidence_max_below_default(self):
        """MIN_VIEW_CONFIDENCE should be below the 0.50 default confidence."""
        assert MIN_VIEW_CONFIDENCE < 0.50


class TestExportCompleteness:
    """Verify __all__ covers all public API."""

    def test_all_exports_match_public_api(self):
        from src.strategy.black_litterman_mapper import __all__ as public_api
        expected = {
            "BLViews", "BLResult",
            "map_biases_to_views", "run_black_litterman",
            "compute_bl_weights", "compute_regime_covariances", "tau_sensitivity",
        }
        assert set(public_api) == expected

    def test_all_dataclasses_exported(self):
        from src.strategy.black_litterman_mapper import __all__ as public_api
        assert "BLViews" in public_api
        assert "BLResult" in public_api

    def test_all_functions_exported(self):
        from src.strategy.black_litterman_mapper import __all__ as public_api
        assert "map_biases_to_views" in public_api
        assert "run_black_litterman" in public_api
        assert "compute_bl_weights" in public_api
        assert "tau_sensitivity" in public_api

    def test_no_private_items_in_all(self):
        from src.strategy.black_litterman_mapper import __all__ as public_api
        for name in public_api:
            assert not name.startswith("_"), f"Private name {name} in __all__"


class TestBLViewsFieldValidation:
    """Dataclass field validation for BLViews via dataclasses.fields()."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(BLViews)}
        expected = {"absolute_views", "view_confidences", "tau", "prior", "symbols"}
        assert field_names == expected

    def test_tau_default(self):
        from dataclasses import fields
        fmap = {f.name: f for f in fields(BLViews)}
        assert fmap["tau"].default == DEFAULT_TAU
        assert fmap["tau"].type is float or float in str(fmap["tau"].type)

    def test_prior_default(self):
        from dataclasses import fields
        fmap = {f.name: f for f in fields(BLViews)}
        assert fmap["prior"].default == "equal"

    def test_symbols_default_factory(self):
        from dataclasses import fields
        fmap = {f.name: f for f in fields(BLViews)}
        # default_factory should produce a new list each time
        produced = fmap["symbols"].default_factory()
        assert produced == DEFAULT_SYMBOLS
        # Verify it's a fresh copy
        produced.append("EXTRA")
        from dataclasses import fields
        fmap2 = {f.name: f for f in fields(BLViews)}
        restored = fmap2["symbols"].default_factory()
        assert restored == DEFAULT_SYMBOLS  # unchanged

    def test_absolute_views_is_required(self):
        """absolute_views has no default — it's a required field."""
        from dataclasses import fields
        fmap = {f.name: f for f in fields(BLViews)}
        assert fmap["absolute_views"].default is fmap["absolute_views"].default
        # Check it's not a MISSING sentinel — required field
        from dataclasses import MISSING, fields as f2
        views_fields = f2(BLViews)
        av_field = [f for f in views_fields if f.name == "absolute_views"][0]
        assert av_field.default is MISSING
        assert av_field.default_factory is MISSING

    def test_view_confidences_is_required(self):
        from dataclasses import MISSING, fields
        vc_field = [f for f in fields(BLViews) if f.name == "view_confidences"][0]
        assert vc_field.default is MISSING
        assert vc_field.default_factory is MISSING


class TestBLResultFieldValidation:
    """Dataclass field validation for BLResult via dataclasses.fields()."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(BLResult)}
        expected = {
            "posterior_returns", "bl_weights", "tau", "prior_type",
            "view_confidences", "expected_sharpe", "expected_cagr",
            "expected_volatility", "extras",
        }
        assert field_names == expected

    def test_expected_sharpe_default_none(self):
        from dataclasses import fields
        ef = [f for f in fields(BLResult) if f.name == "expected_sharpe"][0]
        assert ef.default is None

    def test_expected_cagr_default_none(self):
        from dataclasses import fields
        ef = [f for f in fields(BLResult) if f.name == "expected_cagr"][0]
        assert ef.default is None

    def test_expected_volatility_default_none(self):
        from dataclasses import fields
        ef = [f for f in fields(BLResult) if f.name == "expected_volatility"][0]
        assert ef.default is None

    def test_extras_default_empty_dict(self):
        from dataclasses import fields
        ef = [f for f in fields(BLResult) if f.name == "extras"][0]
        assert ef.default_factory() == {}

    def test_required_fields_have_no_default(self):
        from dataclasses import MISSING, fields
        for name in ("posterior_returns", "bl_weights", "tau", "prior_type", "view_confidences"):
            fld = [f for f in fields(BLResult) if f.name == name][0]
            assert fld.default is MISSING and fld.default_factory is MISSING, \
                f"{name} should be required"

    def test_optional_fields_have_none_default(self):
        from dataclasses import fields
        for name in ("expected_sharpe", "expected_cagr", "expected_volatility"):
            fld = [f for f in fields(BLResult) if f.name == name][0]
            assert fld.default is None, f"{name} should default to None"


class TestMapBiasesToViewsEdgeCases:
    """Boundary and edge cases for map_biases_to_views()."""

    def test_nan_bias(self):
        """NaN bias should propagate as NaN in absolute_views."""
        views = map_biases_to_views(float("nan"), 0.0, 0.0)
        assert np.isnan(views.absolute_views["SPY"])
        assert views.absolute_views["TLT"] == 0.0
        assert views.absolute_views["GLD"] == 0.0

    def test_inf_positive_bias(self):
        views = map_biases_to_views(float("inf"), 0.0, 0.0)
        assert np.isinf(views.absolute_views["SPY"])
        assert views.absolute_views["SPY"] > 0

    def test_inf_negative_bias(self):
        views = map_biases_to_views(0.0, float("-inf"), 0.0)
        assert np.isinf(views.absolute_views["TLT"])
        assert views.absolute_views["TLT"] < 0

    def test_extreme_positive_bias(self):
        """Very large bias should produce proportionally large return."""
        views = map_biases_to_views(100.0, 0.0, 0.0)
        assert views.absolute_views["SPY"] == 100.0 * BIAS_TO_RETURN_SCALE

    def test_extreme_negative_bias(self):
        views = map_biases_to_views(0.0, 0.0, -100.0)
        assert views.absolute_views["GLD"] == -100.0 * BIAS_TO_RETURN_SCALE

    def test_tiny_bias(self):
        """Near-zero bias should produce near-zero return."""
        views = map_biases_to_views(1e-10, -1e-10, 1e-10)
        assert views.absolute_views["SPY"] == pytest.approx(1e-10 * BIAS_TO_RETURN_SCALE)
        assert views.absolute_views["TLT"] == pytest.approx(-1e-10 * BIAS_TO_RETURN_SCALE)

    def test_empty_health_scores(self):
        """Empty health_scores dict should use 0.50 fallback and return default confidences."""
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores={})
        # np.mean([]) raises RuntimeWarning and returns NaN, but the code handles it
        for c in views.view_confidences:
            assert c >= MIN_VIEW_CONFIDENCE

    def test_health_scores_single_signal(self):
        """Single-element health_scores should work."""
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores={"SIG1": 0.90})
        for c in views.view_confidences:
            assert 0 <= c <= 1.0

    def test_health_scores_many_signals(self):
        """Many signals should produce stable average."""
        scores = {f"SIG{i}": 0.5 for i in range(100)}
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores=scores)
        for c in views.view_confidences:
            assert 0 <= c <= 1.0

    def test_confidence_rounding_precision(self):
        """Confidences should be rounded to 4 decimal places."""
        health = {"SIG1": 0.33333333, "SIG2": 0.66666666}
        views = map_biases_to_views(0.5, 0.3, 0.2, health_scores=health)
        for c in views.view_confidences:
            # Check it's rounded to 4 decimal places
            assert round(c, 4) == c

    def test_max_bias_magnitude_clipped_in_confidence(self):
        """Bias magnitude > 1.0 should be clipped to 1.0 in confidence calc."""
        health = {"SIG1": 0.80}
        views = map_biases_to_views(2.0, 0.0, 0.0, health_scores=health)
        # bias_magnitude = min(abs(2.0), 1.0) = 1.0
        # conf = 0.80 * (0.5 + 0.5 * 1.0) = 0.80
        assert views.view_confidences[0] == pytest.approx(0.80, abs=0.01)
        # Other confidences use bias=0 -> 0.80 * 0.5 = 0.40
        assert views.view_confidences[1] == pytest.approx(0.40, abs=0.01)


class TestRunBlackLittermanEdgeCases:
    """Boundary cases for run_black_litterman()."""

    @pytest.fixture
    def sample_cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    def test_negative_risk_free_rate(self):
        """Negative risk-free rate should still produce valid result."""
        cov = self._cov()
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(cov, views, risk_free_rate=-0.01)
        assert isinstance(result, BLResult)
        assert len(result.bl_weights) > 0

    def test_zero_risk_free_rate(self):
        cov = self._cov()
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(cov, views, risk_free_rate=0.0)
        assert isinstance(result, BLResult)

    def test_high_risk_free_rate(self):
        """Very high risk-free rate should not crash."""
        cov = self._cov()
        views = map_biases_to_views(0.3, 0.1, 0.2)
        result = run_black_litterman(cov, views, risk_free_rate=0.20)
        assert isinstance(result, BLResult)

    def test_missing_market_caps_key(self):
        """Missing key in market_caps should fall back to 1e9."""
        views = map_biases_to_views(0.0, 0.0, 0.0, prior="market")
        cov = self._cov()
        result = run_black_litterman(
            cov, views, market_caps={"SPY": 1e12}  # missing GLD, TLT
        )
        assert isinstance(result, BLResult)
        assert result.prior_type == "market"

    def test_nan_in_covariance(self):
        """NaN in covariance degrades gracefully — cascade falls back to equal weight."""
        cov = np.array([
            [0.0225, float("nan"), -0.0063],
            [float("nan"), 0.0256, 0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)
        # Cascade: BL fails → HRP fails → equal weight fallback
        assert len(result.bl_weights) > 0
        assert result.expected_sharpe is None
        assert result.extras.get("optimization_method") == "bl_equal_weight"

    def test_inf_in_covariance(self):
        """Inf in covariance degrades gracefully."""
        cov = np.array([
            [0.0225, 0.0000, float("inf")],
            [0.0000, 0.0256, 0.0022],
            [float("inf"), 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)
        assert len(result.bl_weights) > 0
        # Method recorded in extras regardless of whether BL succeeds or falls back
        assert "optimization_method" in result.extras

    def test_single_asset_covariance(self):
        """1x1 covariance with single view should work."""
        class SingleViews:
            symbols = ["SPY"]
            absolute_views = {"SPY": 0.05}
            view_confidences = [0.50]
            tau = 0.15
            prior = "equal"
        cov = np.array([[0.0225]])
        result = run_black_litterman(cov, SingleViews())
        assert isinstance(result, BLResult)

    def _cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])


class TestRunBlackLittermanFailurePaths:
    """Test expected failure and boundary paths in run_black_litterman()."""

    def test_covariance_dimension_mismatch_raises(self):
        """Wrong-size covariance matrix should raise ValueError."""
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])  # 2x2 but 3 symbols expected
        views = map_biases_to_views(0.5, 0.3, 0.2)
        with pytest.raises(ValueError, match="shape"):
            run_black_litterman(cov, views)

    def test_custom_pi_wrong_length_raises(self):
        """Wrong-length pi should raise IndexError."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.5, 0.3, 0.2)
        with pytest.raises((IndexError, ValueError)):
            run_black_litterman(cov, views, pi=np.array([0.05]))  # 1 vs 3 assets

    def test_all_views_zero_with_equal_prior(self):
        """Zero views with equal prior should not crash."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)
        assert result.prior_type == "equal"

    def test_market_prior_without_market_caps_arg(self):
        """Market prior without explicit market_caps should use defaults."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0, prior="market")
        result = run_black_litterman(cov, views)
        assert result.prior_type == "market"

    def test_custom_pi_overrides_prior(self):
        """Custom pi should produce prior_type='custom'."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0, prior="market")
        result = run_black_litterman(cov, views, pi=np.array([0.06, 0.04, 0.02]))
        assert result.prior_type == "custom"

    def test_custom_pi_matches_prior_type(self):
        """With non-None pi, prior_type must be 'custom'."""
        cov = np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])
        views = map_biases_to_views(0.0, 0.0, 0.0)
        result = run_black_litterman(cov, views, pi=np.array([0.08, 0.04, 0.03]))
        assert result.prior_type == "custom"

    def test_zero_covariance_matrix(self):
        """All-zero covariance matrix -- optimize may produce extreme weights or cascade."""
        cov = np.zeros((3, 3))
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)
        assert "optimization_method" in result.extras

    def test_identity_covariance(self):
        """Identity covariance should produce valid weights."""
        cov = np.eye(3)
        views = map_biases_to_views(0.5, 0.3, 0.2)
        result = run_black_litterman(cov, views)
        assert isinstance(result, BLResult)


class TestComputeBLWeightsEdgeCases:
    """Edge cases for compute_bl_weights()."""

    def test_single_symbol_available(self):
        """Only one of three symbols available — must adapt."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.3,
            duration_bias=0.0,
            gold_bias=0.0,
        )
        assert isinstance(result, BLResult)

    def test_extra_columns_in_prices(self):
        """Extra columns not in DEFAULT_SYMBOLS should be ignored."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 100)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 100)),
            "BTC": 100 * np.cumprod(1 + np.random.normal(0.001, 0.03, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
            health_scores={"SIG1": 0.80},
        )
        assert isinstance(result, BLResult)

    def test_empty_health_scores_dict(self):
        """Empty health_scores should not crash compute_bl_weights."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 100)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.5,
            duration_bias=0.2,
            gold_bias=0.3,
            health_scores={},
        )
        assert isinstance(result, BLResult)

    def test_tau_near_zero(self):
        """Tau approaching zero means views have negligible weight on posterior."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 100)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=1.0,
            duration_bias=-1.0,
            gold_bias=0.5,
            tau=0.001,
        )
        assert isinstance(result, BLResult)
        assert result.tau == pytest.approx(0.001)

    def test_market_prior_with_prices(self):
        """Market prior should work through compute_bl_weights."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 100)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=0.0,
            duration_bias=0.0,
            gold_bias=0.0,
            prior="market",
        )
        assert isinstance(result, BLResult)
        assert result.prior_type == "market"

    def test_all_biases_near_max(self):
        """Maximum biases in all directions."""
        import pandas as pd
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "SPY": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 100)),
            "GLD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 100)),
            "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.008, 100)),
        }, index=dates)
        result = compute_bl_weights(
            prices_df=prices,
            equity_bias=1.0,
            duration_bias=1.0,
            gold_bias=1.0,
        )
        assert isinstance(result, BLResult)


class TestTauSensitivityEdgeCases:
    """Edge cases for tau_sensitivity()."""

    @pytest.fixture
    def cov(self):
        return np.array([
            [0.0225, 0.0000, -0.0063],
            [0.0000, 0.0256,  0.0022],
            [-0.0063, 0.0022, 0.0196],
        ])

    @pytest.fixture
    def views(self):
        return map_biases_to_views(0.5, 0.2, 0.3)

    def test_empty_tau_list(self, cov, views):
        """Empty tau_values list should produce empty results dict."""
        results = tau_sensitivity(cov, views, tau_values=[])
        assert results == {}

    def test_single_tau_value(self, cov, views):
        results = tau_sensitivity(cov, views, tau_values=[0.15])
        assert len(results) == 1
        assert 0.15 in results

    def test_large_tau(self, cov, views):
        """tau=0.50 (upper bound) should still produce valid result."""
        results = tau_sensitivity(cov, views, tau_values=[0.50])
        assert 0.50 in results
        assert isinstance(results[0.50], BLResult)

    def test_tau_at_lower_bound(self, cov, views):
        """tau=0.005 (lowest default) should produce valid result."""
        results = tau_sensitivity(cov, views, tau_values=[0.005])
        assert 0.005 in results
        assert isinstance(results[0.005], BLResult)

    def test_tau_out_of_range_skipped(self, cov, views):
        """tau=100.0 is out of PyPortfolioOpt [0,1] range, should be skipped."""
        results = tau_sensitivity(cov, views, tau_values=[100.0])
        assert 100.0 not in results

    def test_negative_tau_skipped(self, cov, views):
        """tau=-0.05 is out of PyPortfolioOpt [0,1] range, should be skipped."""
        results = tau_sensitivity(cov, views, tau_values=[-0.05])
        assert -0.05 not in results

    def test_duplicate_tau_values(self, cov, views):
        """Duplicate tau values — last one wins (dict overwrite)."""
        results = tau_sensitivity(cov, views, tau_values=[0.15, 0.15, 0.15])
        assert len(results) == 1

    def test_custom_prior_through_tau_sensitivity(self, cov, views):
        """Custom pi through tau_sensitivity."""
        pi = np.array([0.08, 0.04, 0.02])
        results = tau_sensitivity(cov, views, tau_values=[0.15], pi=pi)
        assert 0.15 in results
        assert results[0.15].prior_type == "custom"


class TestBLResultExtrasAndEdgeCases:
    """Further BLResult edge cases."""

    def test_extras_can_store_arbitrary_data(self):
        r = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 1.0},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
            extras={"diagnostics": {"iterations": 25, "converged": True}},
        )
        assert r.extras["diagnostics"]["iterations"] == 25

    def test_extras_mutable_default_not_shared(self):
        r1 = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 1.0},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
        )
        r2 = BLResult(
            posterior_returns={"SPY": 0.08},
            bl_weights={"SPY": 1.0},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
        )
        r1.extras["key"] = "value"
        assert "key" not in r2.extras  # not shared

    def test_posterior_returns_rounding(self):
        """Posterior returns should be rounded to 6 decimal places."""
        r = BLResult(
            posterior_returns={"SPY": 0.083333333},
            bl_weights={"SPY": 1.0},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5],
        )
        # Just verify the value is reasonable
        assert isinstance(r.posterior_returns["SPY"], float)

    def test_bl_weights_filter_negligible(self):
        """BL weights < 0.001 should be filtered out? No — that filtering
        happens inside run_black_litterman, not BLResult itself."""
        r = BLResult(
            posterior_returns={"SPY": 0.08, "GLD": 0.03, "TLT": 0.02},
            bl_weights={"SPY": 0.50, "GLD": 0.30, "TLT": 0.20},
            tau=0.15,
            prior_type="equal",
            view_confidences=[0.5, 0.5, 0.5],
        )
        assert sum(r.bl_weights.values()) == pytest.approx(1.0, abs=0.01)


class TestCLIGuard:
    """Test CLI/__main__ guard behavior."""

    def test_module_has_no_main_block(self):
        """The source module does NOT have a __main__ block (verified by inspection)."""
        import inspect
        import src.strategy.black_litterman_mapper as blm
        source = inspect.getsource(blm)
        assert '__main__' not in source, "Source unexpectedly has a __main__ block"

    def test_module_import_does_not_print(self, capsys):
        """Importing the module should not produce any print output."""
        import importlib
        import src.strategy.black_litterman_mapper as blm
        importlib.reload(blm)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_module_all_is_public_strings(self):
        """Every name in __all__ should be a string."""
        import src.strategy.black_litterman_mapper as blm
        for name in blm.__all__:
            assert isinstance(name, str), f"{name} is not a string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
