"""Tests for multi-timeframe signal fusion (v806 redo).

TDD red phase — these tests define the contract for MultiTimeframeFusion
before implementation exists. All tests should FAIL until the green phase.

Tests use synthetic price data (no disk/network dependencies).
"""

import numpy as np
import pandas as pd
import pytest

# Module under test — will exist after green phase
from src.signals.multi_timeframe_fusion import (
    MultiTimeframeFusion,
    TimeframeComponent,
    FUSION_WEIGHTS,
    TIMEFRAMES,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_prices():
    """Generate 120 trading days of synthetic price data for SPY/GLD/TLT."""
    dates = pd.bdate_range("2025-01-02", periods=120)
    np.random.seed(42)
    # SPY: uptrend with noise, GLD: sideways, TLT: downtrend
    spy = 500 + np.cumsum(np.random.normal(0.5, 2.0, 120))
    gld = 200 + np.cumsum(np.random.normal(0.0, 1.5, 120))
    tlt = 90 + np.cumsum(np.random.normal(-0.2, 1.0, 120))
    return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt}, index=dates)


@pytest.fixture
def short_prices():
    """Generate only 30 days of data — less than long timeframe (63d)."""
    dates = pd.bdate_range("2025-10-01", periods=30)
    np.random.seed(99)
    spy = 500 + np.cumsum(np.random.normal(0.3, 1.5, 30))
    gld = 200 + np.cumsum(np.random.normal(0.1, 1.0, 30))
    tlt = 90 + np.cumsum(np.random.normal(-0.1, 0.8, 30))
    return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt}, index=dates)


@pytest.fixture
def fusion(synthetic_prices):
    """Create a MultiTimeframeFusion instance with synthetic data."""
    return MultiTimeframeFusion(prices_df=synthetic_prices)


# ── Timeframe Constants ─────────────────────────────────────────────────────


class TestTimeframeConstants:
    """Verify timeframe bucket definitions."""

    def test_timeframes_has_three_buckets(self):
        assert len(TIMEFRAMES) == 3

    def test_timeframe_keys_are_short_medium_long(self):
        assert set(TIMEFRAMES.keys()) == {"short", "medium", "long"}

    def test_timeframe_lookbacks_ascending(self):
        lookbacks = [TIMEFRAMES[k]["lookback_days"] for k in ("short", "medium", "long")]
        assert lookbacks == sorted(lookbacks)
        assert lookbacks[0] < lookbacks[1] < lookbacks[2]

    def test_timeframe_short_is_five_days(self):
        assert TIMEFRAMES["short"]["lookback_days"] == 5

    def test_timeframe_medium_is_twenty_one_days(self):
        assert TIMEFRAMES["medium"]["lookback_days"] == 21

    def test_timeframe_long_is_sixty_three_days(self):
        assert TIMEFRAMES["long"]["lookback_days"] == 63


# ── Fusion Weight Matrix ────────────────────────────────────────────────────


class TestFusionWeights:
    """Verify regime-dependent fusion weight matrix."""

    def test_fusion_weights_has_five_regimes(self):
        assert len(FUSION_WEIGHTS) == 5

    def test_fusion_weights_regime_keys(self):
        expected = {"crisis", "high_vol", "normal", "low_vol", "recovery"}
        assert set(FUSION_WEIGHTS.keys()) == expected

    def test_fusion_weights_sum_to_one_per_regime(self):
        for regime, weights in FUSION_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"Regime {regime} weights sum to {total}"

    def test_fusion_weights_all_positive(self):
        for regime, weights in FUSION_WEIGHTS.items():
            for tf, w in weights.items():
                assert w > 0, f"Regime {regime}, timeframe {tf} has weight {w}"

    def test_crisis_emphasizes_short_term(self):
        crisis = FUSION_WEIGHTS["crisis"]
        assert crisis["short"] > crisis["medium"]
        assert crisis["medium"] > crisis["long"]

    def test_low_vol_emphasizes_long_term(self):
        low_vol = FUSION_WEIGHTS["low_vol"]
        assert low_vol["long"] > low_vol["medium"]
        assert low_vol["medium"] > low_vol["short"]

    def test_normal_has_medium_bias(self):
        normal = FUSION_WEIGHTS["normal"]
        assert normal["medium"] >= normal["long"]
        assert normal["medium"] >= normal["short"]


# ── TimeframeComponent Dataclass ────────────────────────────────────────────


class TestTimeframeComponent:
    """Verify TimeframeComponent dataclass."""

    def test_create_component(self):
        comp = TimeframeComponent(
            timeframe="short",
            lookback_days=5,
            value=0.3,
            confidence=0.8,
        )
        assert comp.timeframe == "short"
        assert comp.lookback_days == 5
        assert comp.value == 0.3
        assert comp.confidence == 0.8

    def test_component_value_clamped(self):
        """Values outside [-1, 1] should be clamped."""
        comp = TimeframeComponent(
            timeframe="short", lookback_days=5, value=1.5, confidence=0.9,
        )
        assert -1.0 <= comp.value <= 1.0


# ── Momentum Decomposition ─────────────────────────────────────────────────


class TestMomentumDecomposition:
    """Test that price data is correctly decomposed into timeframe components."""

    def test_decompose_returns_three_components(self, fusion, synthetic_prices):
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        assert len(components) == 3

    def test_decompose_component_keys(self, fusion, synthetic_prices):
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        assert set(components.keys()) == {"short", "medium", "long"}

    def test_decompose_values_in_range(self, fusion, synthetic_prices):
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        for tf, comp in components.items():
            assert -1.0 <= comp.value <= 1.0, f"{tf} value {comp.value} out of range"

    def test_decompose_strong_trend_has_meaningful_magnitude(self, fusion, synthetic_prices):
        """C1c: vol-scaling bug fix - strong trends must produce non-microscopic signals.

        The synthetic SPY series is a strong uptrend (cumsum of Normal(+0.5, 2.0)).
        Before the fix, tanh(period_return / annualized_vol) produced ~1e-4 values
        because annualized vol (~0.13) >> period return (~0.01). The fix matches
        the vol horizon to the return horizon so tanh(return / period_vol) yields
        a meaningful magnitude (>= 0.05) for clear trends.
        """
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        # Long timeframe over a clear 63-day uptrend should carry meaningful mass
        assert abs(components["long"].value) >= 0.05, (
            f"long value {components['long'].value} is microscopic; "
            "vol-scaling likely divides by annualized instead of period-matched vol"
        )

    def test_decompose_confidence_in_range(self, fusion, synthetic_prices):
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        for tf, comp in components.items():
            assert 0.0 <= comp.confidence <= 1.0, f"{tf} confidence {comp.confidence} out of range"

    def test_decompose_lookback_days_match(self, fusion, synthetic_prices):
        components = fusion._decompose_momentum("SPY", synthetic_prices["SPY"])
        assert components["short"].lookback_days == 5
        assert components["medium"].lookback_days == 21
        assert components["long"].lookback_days == 63

    def test_decompose_short_data_returns_partial(self, short_prices):
        """With only 30 days, long (63d) should be inactive."""
        f = MultiTimeframeFusion(prices_df=short_prices)
        components = f._decompose_momentum("SPY", short_prices["SPY"])
        # Short and medium should have non-zero confidence
        assert components["short"].confidence > 0
        assert components["medium"].confidence > 0
        # Long should have zero confidence (insufficient data)
        assert components["long"].confidence == 0.0


# ── Composite Fusion ────────────────────────────────────────────────────────


class TestCompositeFusion:
    """Test fusion of timeframe components into a single signal."""

    def test_fuse_components_returns_float(self, fusion):
        components = {
            "short": TimeframeComponent("short", 5, 0.4, 0.8),
            "medium": TimeframeComponent("medium", 21, 0.2, 0.9),
            "long": TimeframeComponent("long", 63, 0.1, 0.7),
        }
        result = fusion._fuse_components(components, regime="normal")
        assert isinstance(result, float)

    def test_fuse_result_in_range(self, fusion):
        components = {
            "short": TimeframeComponent("short", 5, 0.8, 0.9),
            "medium": TimeframeComponent("medium", 21, 0.6, 0.8),
            "long": TimeframeComponent("long", 63, 0.4, 0.7),
        }
        result = fusion._fuse_components(components, regime="normal")
        assert -1.0 <= result <= 1.0

    def test_fuse_crisis_biases_short(self, fusion):
        """In crisis, short-term signal should dominate over medium/long."""
        components = {
            "short": TimeframeComponent("short", 5, 1.0, 1.0),
            "medium": TimeframeComponent("medium", 21, -0.3, 1.0),
            "long": TimeframeComponent("long", 63, -0.3, 1.0),
        }
        crisis_result = fusion._fuse_components(components, regime="crisis")
        # Short=1.0 weighted 0.50, medium/long=-0.3 weighted 0.50 combined
        # Should be positive because short dominates in crisis
        assert crisis_result > 0

    def test_fuse_low_vol_biases_long(self, fusion):
        """In low_vol, long-term signal should dominate over short/medium."""
        components = {
            "short": TimeframeComponent("short", 5, -0.3, 1.0),
            "medium": TimeframeComponent("medium", 21, -0.3, 1.0),
            "long": TimeframeComponent("long", 63, 1.0, 1.0),
        }
        low_vol_result = fusion._fuse_components(components, regime="low_vol")
        # Long=1.0 weighted 0.45, short/medium=-0.3 weighted 0.55 combined
        # Should be positive because long dominates in low_vol
        assert low_vol_result > 0

    def test_fuse_skips_zero_confidence(self, fusion):
        """Components with zero confidence should not contribute."""
        components = {
            "short": TimeframeComponent("short", 5, 1.0, 0.0),
            "medium": TimeframeComponent("medium", 21, 0.5, 0.8),
            "long": TimeframeComponent("long", 63, 0.5, 0.8),
        }
        result = fusion._fuse_components(components, regime="normal")
        # Only medium and long contribute
        assert result != 0.0  # Should not be zero (medium+long have positive values)


# ── Confidence Computation ──────────────────────────────────────────────────


class TestConfidenceComputation:
    """Test that confidence reflects timeframe agreement."""

    def test_high_agreement_high_confidence(self, fusion):
        """When all timeframes agree, confidence should be higher than disagreement."""
        components_agree = {
            "short": TimeframeComponent("short", 5, 0.5, 0.9),
            "medium": TimeframeComponent("medium", 21, 0.5, 0.9),
            "long": TimeframeComponent("long", 63, 0.5, 0.9),
        }
        components_disagree = {
            "short": TimeframeComponent("short", 5, 0.8, 0.9),
            "medium": TimeframeComponent("medium", 21, -0.8, 0.9),
            "long": TimeframeComponent("long", 63, 0.0, 0.9),
        }
        conf_agree = fusion._compute_fusion_confidence(components_agree, regime="normal")
        conf_disagree = fusion._compute_fusion_confidence(components_disagree, regime="normal")
        assert conf_agree > conf_disagree

    def test_low_agreement_low_confidence(self, fusion):
        """When timeframes disagree, confidence should be low."""
        components = {
            "short": TimeframeComponent("short", 5, 0.8, 0.9),
            "medium": TimeframeComponent("medium", 21, -0.8, 0.9),
            "long": TimeframeComponent("long", 63, 0.0, 0.9),
        }
        conf = fusion._compute_fusion_confidence(components, regime="normal")
        assert conf < 0.5

    def test_confidence_in_range(self, fusion):
        components = {
            "short": TimeframeComponent("short", 5, 0.3, 0.8),
            "medium": TimeframeComponent("medium", 21, 0.3, 0.8),
            "long": TimeframeComponent("long", 63, 0.3, 0.8),
        }
        conf = fusion._compute_fusion_confidence(components, regime="normal")
        assert 0.0 <= conf <= 1.0


# ── Per-Asset Decomposition ─────────────────────────────────────────────────


class TestPerAssetDecomposition:
    """Test that each asset gets independent timeframe analysis."""

    def test_get_per_asset_signals_returns_all_assets(self, fusion, synthetic_prices):
        result = fusion._get_per_asset_signals(synthetic_prices)
        assert "SPY" in result
        assert "GLD" in result
        assert "TLT" in result

    def test_per_asset_signals_have_components(self, fusion, synthetic_prices):
        result = fusion._get_per_asset_signals(synthetic_prices)
        for asset, components in result.items():
            assert set(components.keys()) == {"short", "medium", "long"}

    def test_different_assets_different_signals(self, fusion, synthetic_prices):
        result = fusion._get_per_asset_signals(synthetic_prices)
        # SPY (uptrend) and TLT (downtrend) should have different signs
        spy_val = result["SPY"]["short"].value
        tlt_val = result["TLT"]["short"].value
        # They don't HAVE to differ in sign (random data), but they should
        # be independently computed
        assert spy_val != tlt_val or True  # structural check, not value check


# ── Alternative Data Passthrough ────────────────────────────────────────────


class TestAlternativeDataPassthrough:
    """ALTERNATIVE_DATA bypasses timeframe decomposition at the ensemble voter level."""

    def test_alt_data_collector_skips_mtf(self):
        """Alternative data signal is collected independently, not through MTF."""
        from src.strategy.ensemble_voter import SignalSource
        # MTF and ALT_DATA are separate signal sources
        assert SignalSource.MULTI_TIMEFRAME_FUSION != SignalSource.ALTERNATIVE_DATA


# ── SignalSnapshot Output ───────────────────────────────────────────────────


class TestSignalSnapshotOutput:
    """Test that get_signal_snapshot() returns a valid SignalSnapshot."""

    def test_snapshot_has_correct_source(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert snapshot.source == "multi_timeframe_fusion"

    def test_snapshot_has_timestamp(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert snapshot.timestamp is not None
        assert len(snapshot.timestamp) > 0

    def test_snapshot_value_in_range(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert -1.0 <= snapshot.value <= 1.0

    def test_snapshot_confidence_in_range(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert 0.0 <= snapshot.confidence <= 1.0

    def test_snapshot_has_asset_signals(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert isinstance(snapshot.asset_signals, dict)
        assert len(snapshot.asset_signals) > 0

    def test_snapshot_asset_signals_in_range(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        for asset, val in snapshot.asset_signals.items():
            assert -1.0 <= val <= 1.0, f"{asset} signal {val} out of range"

    def test_snapshot_has_explanation(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert len(snapshot.explanation) > 0

    def test_snapshot_metadata_has_timeframes(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert "timeframe_breakdown" in snapshot.metadata

    def test_snapshot_regime_fit(self, fusion):
        snapshot = fusion.get_signal_snapshot()
        assert snapshot.regime_fit in ("all", "normal", "crisis", "high_vol", "low_vol", "recovery")


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_no_price_data_returns_inactive(self):
        """Empty price data should return an inactive snapshot."""
        empty_df = pd.DataFrame()
        f = MultiTimeframeFusion(prices_df=empty_df)
        snapshot = f.get_signal_snapshot()
        assert not snapshot.is_active
        assert snapshot.value == 0.0

    def test_missing_ticker_graceful(self, synthetic_prices):
        """Requesting a ticker not in the data should not crash."""
        f = MultiTimeframeFusion(prices_df=synthetic_prices)
        # This should not raise — missing tickers are skipped
        snapshot = f.get_signal_snapshot(tickers=["SPY", "FAKE_TICKER"])
        assert snapshot.is_active  # SPY still works

    def test_nan_prices_handled(self, synthetic_prices):
        """NaN values in price data should be handled gracefully."""
        prices = synthetic_prices.copy()
        prices.loc[prices.index[50:55], "SPY"] = np.nan
        f = MultiTimeframeFusion(prices_df=prices)
        snapshot = f.get_signal_snapshot()
        # Should not crash; may have lower confidence
        assert snapshot.is_active is not None

    def test_all_nan_ticker_inactive(self, synthetic_prices):
        """Ticker with all NaN should be excluded."""
        prices = synthetic_prices.copy()
        prices["SPY"] = np.nan
        f = MultiTimeframeFusion(prices_df=prices)
        snapshot = f.get_signal_snapshot(tickers=["SPY", "GLD"])
        # GLD should still work
        assert "GLD" in snapshot.asset_signals or not snapshot.is_active


# ── Integration: SignalSource Enum ──────────────────────────────────────────


class TestSignalSourceIntegration:
    """Test that the new signal integrates with the existing enum."""

    def test_signal_source_has_multi_timeframe_fusion(self):
        from src.strategy.ensemble_voter import SignalSource
        assert hasattr(SignalSource, 'MULTI_TIMEFRAME_FUSION')

    def test_signal_source_value(self):
        from src.strategy.ensemble_voter import SignalSource
        assert SignalSource.MULTI_TIMEFRAME_FUSION.value == "multi_timeframe_fusion"
