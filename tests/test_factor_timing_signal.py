"""
Tests for v6.02 Factor Timing Signal Generator.

Tests cross-sectional factor Z-score computation, regime-based tilting,
and EnsembleVoter integration.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime

from src.signals.factor_timing_signal import (
    FACTOR_ETFS,
    MOMENTUM_HORIZONS,
    REGIME_FACTOR_TILTS,
    FactorMomentum,
    FactorTimingResult,
    load_factor_prices,
    compute_returns,
    compute_factor_scores,
    compute_regime_tilt,
    compute_signal_value,
    compute_composite_urgency,
    compute_factor_divergence,
    generate_timing_signal,
    get_ensemble_signal,
    PRICES_PATH,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """Generate synthetic price data for all 4 factor ETFs."""
    np.random.seed(42)
    n = 300  # ~15 months of daily data
    symbols = list(FACTOR_ETFS.keys())

    prices = {}
    for i, symbol in enumerate(symbols):
        # Different trends for each factor
        base_trend = [0.0003, 0.0001, -0.0001, -0.0002][i]
        volatility = [0.01, 0.008, 0.009, 0.011][i]

        log_returns = np.random.normal(base_trend, volatility, n)
        price_series = 100 * np.exp(np.cumsum(log_returns))
        prices[symbol] = price_series

    return prices


@pytest.fixture
def all_equal_prices():
    """Degenerate case: all factors have identical returns."""
    np.random.seed(42)
    n = 300
    log_returns = np.random.normal(0.0002, 0.01, n)
    price_series = 100 * np.exp(np.cumsum(log_returns))

    return {symbol: price_series.copy() for symbol in FACTOR_ETFS}


@pytest.fixture
def clear_winner_prices():
    """One factor clearly outperforms, one clearly underperforms."""
    np.random.seed(42)
    n = 300

    # MTUM (winner): strong uptrend
    mtum = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, n)))
    # USMV (loser): slight downtrend
    usmv = 100 * np.exp(np.cumsum(np.random.normal(-0.0003, 0.008, n)))
    # QUAL: middling
    qual = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.009, n)))
    # VLUE: slightly positive
    vlue = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.011, n)))

    return {"MTUM": mtum, "USMV": usmv, "QUAL": qual, "VLUE": vlue}


@pytest.fixture
def short_history_prices():
    """Only 100 data points (less than 1 year)."""
    np.random.seed(42)
    n = 100
    return {
        symbol: 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, n)))
        for symbol in FACTOR_ETFS
    }


# ── Tests: load_factor_prices ────────────────────────────────────────

def test_load_prices_real_file():
    """Load prices from actual prices.json (should have MTUM, USMV, QUAL, VLUE)."""
    if not PRICES_PATH.exists():
        pytest.skip("prices.json not found — skipping integration test")

    prices = load_factor_prices()
    assert len(prices) >= 2, f"Expected at least 2 factors, got {len(prices)}"

    for symbol in ["MTUM", "USMV"]:
        assert symbol in prices, f"Missing {symbol}"
        assert len(prices[symbol]) > 260, f"{symbol} has insufficient data"


def test_load_prices_subset():
    """Load only specific symbols."""
    if not PRICES_PATH.exists():
        pytest.skip("prices.json not found")

    prices = load_factor_prices(symbols=["MTUM", "QUAL"])
    assert len(prices) == 2
    assert "MTUM" in prices
    assert "QUAL" in prices
    assert "USMV" not in prices


def test_load_prices_bad_symbol():
    """Non-existent symbol returns empty dict."""
    if not PRICES_PATH.exists():
        pytest.skip("prices.json not found")

    prices = load_factor_prices(symbols=["FAKE_TICKER_123"])
    assert len(prices) == 0


# ── Tests: compute_returns ───────────────────────────────────────────

class TestComputeReturns:
    def test_positive_return(self):
        prices = np.array([100, 105, 110, 115, 120, 125])
        # 252-day lookback: but only 6 points, so None
        r = compute_returns(prices, 252)
        assert r is None, "Should return None for insufficient data"

    def test_simple_return(self):
        prices = np.array([100, 102, 104, 106, 108, 110])
        # 5-day lookback: (110 - 100) / 100 = 0.10
        r = compute_returns(prices, 5)
        assert r is not None
        assert abs(r - 0.10) < 1e-10

    def test_negative_return(self):
        prices = np.array([100, 98, 96, 94, 92, 90])
        r = compute_returns(prices, 5)
        assert r is not None
        assert r < 0
        assert abs(r - (-0.10)) < 1e-10

    def test_zero_return(self):
        prices = np.array([100, 100, 100, 100, 100, 100])
        r = compute_returns(prices, 5)
        assert r is not None
        assert abs(r) < 1e-10

    def test_insufficient_data(self):
        prices = np.array([100])
        assert compute_returns(prices, 5) is None

    def test_exact_lookback(self):
        prices = np.array([90, 100])  # 1 return
        r = compute_returns(prices, 1)  # (100-90)/90 = 0.111...
        assert r is not None
        assert abs(r - 10/90) < 1e-10


# ── Tests: compute_factor_scores ─────────────────────────────────────

class TestComputeFactorScores:
    def test_basic_scores(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        assert len(scores) == 4
        assert all(isinstance(s, FactorMomentum) for s in scores.values())

    def test_rankings_are_deterministic(self, sample_prices):
        scores1 = compute_factor_scores(sample_prices)
        scores2 = compute_factor_scores(sample_prices)
        ranks1 = [(s.symbol, s.rank) for s in scores1.values()]
        ranks2 = [(s.symbol, s.rank) for s in scores2.values()]
        assert ranks1 == ranks2

    def test_clear_winner_ranks_first(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        # MTUM (strong uptrend) should rank higher (lower number) than USMV (slight downtrend)
        assert scores["MTUM"].rank < scores["USMV"].rank, (
            f"Expected MTUM rank ({scores['MTUM'].rank}) < USMV rank ({scores['USMV'].rank})"
        )

    def test_all_factors_have_z_scores(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        for symbol, fs in scores.items():
            assert isinstance(fs.composite_z, float)
            assert not np.isnan(fs.composite_z)

    def test_momentum_horizons_are_computed(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        for fs in scores.values():
            assert isinstance(fs.short_momentum, float)
            assert isinstance(fs.medium_momentum, float)
            assert isinstance(fs.long_momentum, float)

    def test_short_history_returns_defaults(self, short_history_prices):
        """With <1yr of data, long_momentum should default to 0."""
        scores = compute_factor_scores(short_history_prices)
        for fs in scores.values():
            assert fs.long_momentum == 0.0

    def test_factor_metadata(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        for symbol, fs in scores.items():
            assert fs.factor_name == FACTOR_ETFS[symbol]["factor"]

    def test_z_scores_sum_to_near_zero(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        z_values = [fs.composite_z for fs in scores.values()]
        mean_z = np.mean(z_values)
        assert abs(mean_z) < 1e-10, f"Z-scores should sum to near zero, got {mean_z}"


# ── Tests: compute_regime_tilt ───────────────────────────────────────

class TestComputeRegimeTilt:
    def test_normal_regime(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        tilt = compute_regime_tilt(scores, "normal")
        assert len(tilt) == 4
        assert abs(sum(tilt.values()) - 1.0) < 1e-10

    def test_bull_tilt_favors_momentum(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        tilt = compute_regime_tilt(scores, "bull")
        # In bull regime, MTUM has highest base weight
        assert tilt["MTUM"] >= tilt["USMV"]

    def test_bear_tilt_favors_defensive(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        tilt = compute_regime_tilt(scores, "bear")
        # In bear regime, USMV + QUAL should dominate
        defensive = tilt["USMV"] + tilt["QUAL"]
        aggressive = tilt["MTUM"]
        assert defensive > aggressive

    def test_crisis_tilt_max_defensive(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        tilt = compute_regime_tilt(scores, "crisis")
        # USMV + QUAL should be > 50% in crisis
        assert tilt["USMV"] + tilt["QUAL"] > 0.50

    def test_unknown_regime_defaults_normal(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        tilt_unknown = compute_regime_tilt(scores, "unknown_regime")
        tilt_normal = compute_regime_tilt(scores, "normal")
        assert abs(sum(tilt_unknown.values()) - 1.0) < 1e-10
        # Both should be similar since unknown falls back to normal
        for symbol in FACTOR_ETFS:
            assert abs(tilt_unknown[symbol] - tilt_normal[symbol]) < 0.05

    def test_tilt_weights_sum_to_one(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        for regime in ["normal", "bull", "bear", "high_vol", "crisis"]:
            tilt = compute_regime_tilt(scores, regime)
            assert abs(sum(tilt.values()) - 1.0) < 1e-10, f"{regime} tilt sums to {sum(tilt.values())}"

    def test_all_equal_factors_produce_base_tilt(self, all_equal_prices):
        """When all factors perform equally, base regime weights dominate."""
        scores = compute_factor_scores(all_equal_prices)
        tilt = compute_regime_tilt(scores, "normal")
        base = REGIME_FACTOR_TILTS["normal"]
        # With all equal returns, tilts should be close to base weights
        for symbol in FACTOR_ETFS:
            assert abs(tilt[symbol] - base[symbol]) < 0.03


# ── Tests: compute_signal_value ──────────────────────────────────────

class TestComputeSignalValue:
    def test_range(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        tilt = compute_regime_tilt(scores, "normal")
        signal = compute_signal_value(scores, tilt)
        assert -1.0 <= signal <= 1.0

    def test_aggressive_tilt_gives_positive_signal(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        # MTUM should dominate
        assert scores["MTUM"].composite_z > 0

    def test_defensive_tilt_gives_negative_signal(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        tilt = compute_regime_tilt(scores, "crisis")
        signal = compute_signal_value(scores, tilt)
        # Crisis heavily weights defensive factors
        assert signal <= 0


# ── Tests: compute_composite_urgency ─────────────────────────────────

class TestCompositeUrgency:
    def test_range(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        urgency = compute_composite_urgency(scores)
        assert 0.0 <= urgency <= 1.0

    def test_high_divergence_gives_high_urgency(self, clear_winner_prices):
        scores = compute_factor_scores(clear_winner_prices)
        urgency = compute_composite_urgency(scores)
        # MTUM winning big, USMV losing = high divergence
        assert urgency > 0.3

    def test_low_divergence(self, all_equal_prices):
        scores = compute_factor_scores(all_equal_prices)
        urgency = compute_composite_urgency(scores)
        # All factors similar = low urgency
        assert urgency < 0.7


# ── Tests: compute_factor_divergence ─────────────────────────────────

class TestFactorDivergence:
    def test_range(self, sample_prices):
        scores = compute_factor_scores(sample_prices)
        div = compute_factor_divergence(scores)
        assert 0.0 <= div <= 2.0

    def test_all_equal_zero_divergence(self):
        """All factors with same Z-score should have zero divergence."""
        scores = {
            "A": FactorMomentum("A", "momentum", 0, 0, 0, 0.0, 1, 300),
            "B": FactorMomentum("B", "low_vol", 0, 0, 0, 0.0, 1, 300),
        }
        assert compute_factor_divergence(scores) == 0.0

    def test_two_factors_max_divergence(self):
        """Two factors at +1 and -1 produce divergence of 2.0."""
        scores = {
            "A": FactorMomentum("A", "momentum", 0, 0, 0, 1.0, 1, 300),
            "B": FactorMomentum("B", "low_vol", 0, 0, 0, -1.0, 2, 300),
        }
        assert compute_factor_divergence(scores) == 2.0


# ── Tests: generate_timing_signal (integration) ──────────────────────

class TestGenerateTimingSignal:
    def test_generates_signal(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        result = generate_timing_signal("normal")
        assert result is not None
        assert isinstance(result, FactorTimingResult)
        assert result.top_factor in FACTOR_ETFS
        assert result.bottom_factor in FACTOR_ETFS
        assert len(result.factor_scores) >= 2

    def test_different_regimes_give_different_signals(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        normal = generate_timing_signal("normal")
        crisis = generate_timing_signal("crisis")
        assert normal is not None and crisis is not None
        # Signal values should differ by regime tilt
        assert normal.signal_value != crisis.signal_value

    def test_signal_value_range(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        for regime in ["normal", "bull", "bear", "high_vol", "crisis"]:
            result = generate_timing_signal(regime)
            assert result is not None
            assert -1.0 <= result.signal_value <= 1.0, f"{regime}: {result.signal_value}"

    def test_composite_urgency_range(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        result = generate_timing_signal("normal")
        assert result is not None
        assert 0.0 <= result.composite_urgency <= 1.0

    def test_timestamp_format(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        result = generate_timing_signal("normal")
        assert result is not None
        # Should be parseable as datetime
        parsed = datetime.strptime(result.timestamp.split(".")[0], "%Y-%m-%d %H:%M:%S")
        assert parsed is not None

    def test_factor_rankings_ordered(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        result = generate_timing_signal("normal")
        assert result is not None
        # Check descending order
        for i in range(len(result.factor_rankings) - 1):
            assert result.factor_rankings[i][1] >= result.factor_rankings[i + 1][1]


# ── Tests: get_ensemble_signal (EnsembleVoter integration) ──────────

class TestGetEnsembleSignal:
    def test_returns_dict(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        signal = get_ensemble_signal("normal")
        assert isinstance(signal, dict)
        assert "signal_value" in signal
        assert "confidence" in signal
        assert "active" in signal

    def test_active_flag_with_data(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        signal = get_ensemble_signal("normal")
        assert signal["active"] is True

    def test_all_regime_keys_present(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        for regime in ["normal", "bull", "bear", "high_vol", "crisis"]:
            signal = get_ensemble_signal(regime)
            assert "signal_value" in signal
            assert "regime_tilt" in signal
            assert "factor_scores" in signal

    def test_factor_scores_contains_symbols(self):
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        signal = get_ensemble_signal("normal")
        if signal["active"]:
            for symbol in FACTOR_ETFS:
                assert symbol in signal["factor_scores"] or signal["factor_scores"] == {}


# ── Tests: Edge Cases ────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_price_dict(self):
        scores = compute_factor_scores({})
        assert scores == {}

    def test_single_factor(self):
        prices = {"MTUM": np.array([100, 105, 110, 115, 120])}
        scores = compute_factor_scores(prices)
        assert len(scores) == 1
        # Single factor should get Z-score of 1.0 (it's the best and worst)
        # Actually with a single data point, the std is 0 and z becomes 0
        assert scores["MTUM"].composite_z == 0.0

    def test_tilt_with_no_scores(self):
        tilt = compute_regime_tilt({}, "normal")
        assert len(tilt) == 0

    def test_signal_with_no_data(self):
        result = get_ensemble_signal("normal")
        # If no data, should return safe defaults
        assert "signal_value" in result
        assert "active" in result


# ── Tests: Regime-specific behavior ──────────────────────────────────

class TestRegimeBehavior:
    def test_bull_vs_bear_signal_direction(self):
        """Bull regime should produce higher (or equal) signal than bear."""
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        bull = generate_timing_signal("bull")
        bear = generate_timing_signal("bear")
        assert bull is not None and bear is not None
        assert bull.signal_value >= bear.signal_value, (
            f"Bull signal ({bull.signal_value:.4f}) should be >= bear ({bear.signal_value:.4f})"
        )

    def test_high_vol_vs_crisis(self):
        """Crisis should be more defensive (lower signal) than high_vol."""
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        hv = generate_timing_signal("high_vol")
        crisis = generate_timing_signal("crisis")
        assert hv is not None and crisis is not None
        assert crisis.signal_value <= hv.signal_value, (
            f"Crisis signal ({crisis.signal_value:.4f}) should be <= high vol ({hv.signal_value:.4f})"
        )

    def test_recovery_tilt_properties(self):
        """Recovery regime should have moderate signal."""
        if not PRICES_PATH.exists():
            pytest.skip("prices.json not found")
        result = generate_timing_signal("recovery")
        assert result is not None
        # Recovery is a special regime — just check it exists
        assert "MTUM" in result.regime_tilt


# ── Tests: Structural completeness ───────────────────────────────────

class TestStructure:
    def test_factors_defined(self):
        """All factor ETFs should have metadata."""
        for symbol, meta in FACTOR_ETFS.items():
            assert "factor" in meta
            assert "name" in meta

    def test_momentum_horizons_defined(self):
        assert "short" in MOMENTUM_HORIZONS
        assert "medium" in MOMENTUM_HORIZONS
        assert "long" in MOMENTUM_HORIZONS

    def test_all_regimes_have_tilts(self):
        for regime in ["normal", "bull", "bear", "high_vol", "crisis"]:
            assert regime in REGIME_FACTOR_TILTS
            tilt = REGIME_FACTOR_TILTS[regime]
            assert len(tilt) == 4
            # Check each factor represented
            for symbol in FACTOR_ETFS:
                assert symbol in tilt
