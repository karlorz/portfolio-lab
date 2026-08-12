#!/usr/bin/env python3
"""
Tests for src/costs/etf_cost_table.py — centralized ETF transaction cost data.
"""


from src.costs.etf_cost_table import (
    ETF_COST_BPS,
    DEFAULT_COST_BPS,
    REGIME_COST_MULTIPLIER,
    get_cost_bps,
    estimate_round_trip_bps,
    estimate_cost_bps,
)


class TestETFCostTable:

    def test_spy_cheapest_equity(self):
        assert ETF_COST_BPS['SPY'] == 2.0

    def test_tlt_most_expensive_core(self):
        assert ETF_COST_BPS['TLT'] == 8.0

    def test_gld_between_spy_and_tlt(self):
        assert ETF_COST_BPS['SPY'] < ETF_COST_BPS['GLD'] < ETF_COST_BPS['TLT']

    def test_all_costs_positive(self):
        for sym, cost in ETF_COST_BPS.items():
            assert cost > 0, f"{sym} has non-positive cost"

    def test_all_costs_reasonable(self):
        for sym, cost in ETF_COST_BPS.items():
            assert 1.0 <= cost <= 15.0, f"{sym} cost {cost} out of range"

    def test_covers_core_symbols(self):
        for sym in ['SPY', 'GLD', 'TLT', 'IEF', 'QQQ']:
            assert sym in ETF_COST_BPS

    def test_default_cost(self):
        assert DEFAULT_COST_BPS == 5.0


class TestGetCostBps:

    def test_known_symbol(self):
        assert get_cost_bps('SPY') == 2.0

    def test_unknown_symbol_returns_default(self):
        assert get_cost_bps('UNKNOWN') == DEFAULT_COST_BPS


class TestEstimateRoundTripBps:

    def test_round_trip_is_double(self):
        assert estimate_round_trip_bps('SPY') == 4.0
        assert estimate_round_trip_bps('TLT') == 16.0

    def test_unknown_symbol(self):
        assert estimate_round_trip_bps('UNKNOWN') == DEFAULT_COST_BPS * 2


class TestRegimeCostMultiplier:

    def test_crisis_highest(self):
        assert REGIME_COST_MULTIPLIER['crisis'] > REGIME_COST_MULTIPLIER['normal']

    def test_low_vol_cheapest(self):
        assert REGIME_COST_MULTIPLIER['low_vol'] < REGIME_COST_MULTIPLIER['normal']

    def test_normal_is_baseline(self):
        assert REGIME_COST_MULTIPLIER['normal'] == 1.0


class TestEstimateCostBps:

    def test_normal_regime_equals_base(self):
        assert estimate_cost_bps('SPY', 'normal') == 2.0

    def test_crisis_increases_cost(self):
        base = estimate_cost_bps('SPY', 'normal')
        crisis = estimate_cost_bps('SPY', 'crisis')
        assert crisis > base

    def test_low_vol_decreases_cost(self):
        base = estimate_cost_bps('SPY', 'normal')
        low_vol = estimate_cost_bps('SPY', 'low_vol')
        assert low_vol < base

    def test_unknown_regime_uses_default(self):
        cost = estimate_cost_bps('SPY', 'unknown_regime')
        assert cost == 2.0  # default multiplier is 1.0

    def test_none_regime_uses_normal(self):
        assert estimate_cost_bps('SPY', None) == 2.0

    def test_unknown_symbol_with_crisis_regime(self):
        """Unknown symbol gets default cost * crisis multiplier."""
        cost = estimate_cost_bps('UNKNOWN', 'crisis')
        assert cost == round(DEFAULT_COST_BPS * 1.8, 2)

    def test_high_vol_between_normal_and_crisis(self):
        normal = estimate_cost_bps('GLD', 'normal')
        high_vol = estimate_cost_bps('GLD', 'high_vol')
        crisis = estimate_cost_bps('GLD', 'crisis')
        assert normal < high_vol < crisis

    def test_default_regime_multiplier_is_one(self):
        from src.costs.etf_cost_table import DEFAULT_REGIME_MULTIPLIER
        assert DEFAULT_REGIME_MULTIPLIER == 1.0

    def test_round_trip_with_regime_not_used(self):
        """estimate_round_trip_bps doesn't take regime — it's 2× base cost."""
        assert estimate_round_trip_bps('SPY') == get_cost_bps('SPY') * 2

    def test_all_regime_multipliers_positive(self):
        for regime, mult in REGIME_COST_MULTIPLIER.items():
            assert mult > 0, f"{regime} has non-positive multiplier"

    def test_dbc_highest_cost(self):
        """DBC (commodities) should have the highest cost."""
        max_sym = max(ETF_COST_BPS, key=ETF_COST_BPS.get)
        assert max_sym == 'DBC'

    def test_estimate_cost_result_is_rounded(self):
        """estimate_cost_bps returns 2 decimal places."""
        cost = estimate_cost_bps('GLD', 'crisis')
        assert cost == round(cost, 2)
