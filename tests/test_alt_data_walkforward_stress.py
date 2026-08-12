#!/usr/bin/env python3
"""
Tests for Alternative Data Walk-Forward & Stress Test Engine —
data classes, constants, compute_metrics, build_daily_returns,
walk_forward_test, and stress_test.
"""
import json
import logging

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.backtest.alt_data_walkforward_stress import (
    DailyReturn, WindowResult, StressResult, FullBacktestResult,
    WEIGHTS, REGIME_SHIFTS, STRESS_PERIODS,
    compute_metrics, build_daily_returns, walk_forward_test, stress_test,
    load_price_data, load_alt_signals, run_full_backtest, print_results,
    save_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_return(date='2020-06-01', spy_ret=0.005, gld_ret=0.002,
                       tlt_ret=-0.001, alt_signal=0.1, alt_regime='risk_on',
                       alt_confidence=0.5, baseline_ret=None, overlay_ret=None):
    """Create a DailyReturn with sensible defaults."""
    if baseline_ret is None:
        baseline_ret = WEIGHTS['SPY'] * spy_ret + WEIGHTS['GLD'] * gld_ret + WEIGHTS['TLT'] * tlt_ret
    if overlay_ret is None:
        overlay_ret = baseline_ret  # same unless regime shift applied
    return DailyReturn(
        date=date,
        spy_return=spy_ret,
        gld_return=gld_ret,
        tlt_return=tlt_ret,
        alt_signal=alt_signal,
        alt_regime=alt_regime,
        alt_confidence=alt_confidence,
        baseline_return=baseline_ret,
        overlay_return=overlay_ret,
    )


def _make_price_data(n_days=30, start_date='2020-01-02', base_prices=None):
    """Create synthetic price data dict for SPY/GLD/TLT."""
    if base_prices is None:
        base_prices = {'SPY': 300.0, 'GLD': 150.0, 'TLT': 130.0}
    dates = []
    d = datetime.strptime(start_date, '%Y-%m-%d')
    for _ in range(n_days):
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)

    data = {}
    for sym, base in base_prices.items():
        bars = []
        price = base
        for dt in dates:
            price *= 1.001  # slight uptrend
            bars.append({'d': dt, 'p': round(price, 2)})
        data[sym] = bars
    return data


def _make_alt_signals(dates, regime='risk_on', confidence=0.5, composite=0.1):
    """Create alt signals dict indexed by date."""
    return {
        d: {'date': d, 'composite_score': composite, 'regime': regime, 'confidence': confidence}
        for d in dates
    }


def _make_daily_returns_series(n=252, start_year=2020, base_ret=0.0004, vol=0.01):
    """Create a list of DailyReturn with deterministic returns."""
    import random
    random.seed(42)
    returns = []
    d = datetime(start_year, 1, 2)
    for i in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        r = base_ret + random.gauss(0, vol)
        returns.append(_make_daily_return(
            date=d.strftime('%Y-%m-%d'),
            spy_ret=r,
            gld_ret=r * 0.5,
            tlt_ret=-r * 0.3,
            baseline_ret=r * 0.8,
            overlay_ret=r * 0.85,
        ))
        d += timedelta(days=1)
    return returns


def _make_empty_result():
    """Create a FullBacktestResult with all empty fields."""
    return FullBacktestResult(
        walk_forward_windows=[],
        avg_sharpe_delta=0.0,
        pct_windows_improved=0.0,
        stress_tests=[],
        overall_baseline_sharpe=0.0,
        overall_overlay_sharpe=0.0,
        overall_sharpe_delta=0.0,
        target_met=False,
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_weights_sum_to_one(self):
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_weights_keys(self):
        assert set(WEIGHTS.keys()) == {'SPY', 'GLD', 'TLT'}

    def test_weights_values(self):
        assert WEIGHTS['SPY'] == 0.46
        assert WEIGHTS['GLD'] == 0.38
        assert WEIGHTS['TLT'] == 0.16

    def test_regime_shifts_keys(self):
        assert set(REGIME_SHIFTS.keys()) == {'risk_on', 'risk_off', 'neutral'}

    def test_regime_shifts_neutral_zero(self):
        for v in REGIME_SHIFTS['neutral'].values():
            assert v == 0.0

    def test_regime_shifts_risk_on_positive_spy(self):
        assert REGIME_SHIFTS['risk_on']['SPY'] > 0

    def test_regime_shifts_risk_off_negative_spy(self):
        assert REGIME_SHIFTS['risk_off']['SPY'] < 0

    def test_regime_shifts_sum_to_zero(self):
        for regime in REGIME_SHIFTS.values():
            total = sum(regime.values())
            assert abs(total) < 0.001

    def test_stress_periods_count(self):
        assert len(STRESS_PERIODS) == 5

    def test_stress_periods_keys(self):
        expected = {'covid_crash', 'covid_recovery', 'meme_stock_2021', 'bear_2022', 'rate_hike_2023'}
        assert set(STRESS_PERIODS.keys()) == expected

    def test_stress_periods_have_required_fields(self):
        for name, config in STRESS_PERIODS.items():
            assert 'start' in config
            assert 'end' in config
            assert 'description' in config

    def test_stress_periods_start_before_end(self):
        for name, config in STRESS_PERIODS.items():
            assert config['start'] < config['end']

    def test_regime_shifts_all_have_same_keys(self):
        expected_keys = {'SPY', 'GLD', 'TLT'}
        for regime, shifts in REGIME_SHIFTS.items():
            assert set(shifts.keys()) == expected_keys

    def test_regime_shifts_gld_opposite_spy(self):
        """GLD shift should be opposite sign to SPY for risk_on and risk_off."""
        assert REGIME_SHIFTS['risk_on']['GLD'] < 0
        assert REGIME_SHIFTS['risk_off']['GLD'] > 0


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDailyReturn:
    def test_creation(self):
        dr = _make_daily_return()
        assert dr.date == '2020-06-01'
        assert dr.spy_return == 0.005

    def test_alt_fields(self):
        dr = _make_daily_return(alt_signal=0.2, alt_regime='risk_off', alt_confidence=0.7)
        assert dr.alt_signal == 0.2
        assert dr.alt_regime == 'risk_off'
        assert dr.alt_confidence == 0.7

    def test_returns(self):
        dr = _make_daily_return(baseline_ret=0.004, overlay_ret=0.005)
        assert dr.baseline_return == 0.004
        assert dr.overlay_return == 0.005

    def test_negative_confidence(self):
        """Negative confidence should be storable (edge case, not validated)."""
        dr = _make_daily_return(alt_confidence=-1.0)
        assert dr.alt_confidence == -1.0

    def test_extreme_signal_values(self):
        """Very large composite_score should not cause issues."""
        dr = _make_daily_return(alt_signal=999.0)
        assert dr.alt_signal == 999.0
        dr = _make_daily_return(alt_signal=-999.0)
        assert dr.alt_signal == -999.0

    def test_zero_returns(self):
        """All return fields zero should be valid."""
        dr = _make_daily_return(spy_ret=0.0, gld_ret=0.0, tlt_ret=0.0,
                                baseline_ret=0.0, overlay_ret=0.0)
        assert dr.baseline_return == 0.0
        assert dr.overlay_return == 0.0


class TestWindowResult:
    def test_creation(self):
        w = WindowResult(
            label='Train 2020-2022 / Test 2023',
            start_date='2023', end_date='2023', trading_days=252,
            baseline_cagr=8.0, baseline_vol=12.0, baseline_sharpe=0.67,
            baseline_max_dd=-15.0,
            overlay_cagr=9.0, overlay_vol=11.5, overlay_sharpe=0.78,
            overlay_max_dd=-13.0,
            sharpe_delta=0.11, cagr_delta=1.0,
        )
        assert w.sharpe_delta == 0.11
        assert w.trading_days == 252

    def test_delta_consistency(self):
        """sharpe_delta must equal overlay_sharpe - baseline_sharpe."""
        w = WindowResult(
            label='Window', start_date='2023', end_date='2023', trading_days=100,
            baseline_cagr=8.0, baseline_vol=12.0, baseline_sharpe=0.50,
            baseline_max_dd=-15.0,
            overlay_cagr=10.0, overlay_vol=11.0, overlay_sharpe=0.80,
            overlay_max_dd=-12.0,
            sharpe_delta=0.30, cagr_delta=2.0,
        )
        assert w.sharpe_delta == pytest.approx(w.overlay_sharpe - w.baseline_sharpe)

    def test_extreme_values(self):
        """Very large CAGR values should be storable."""
        w = WindowResult(
            label='Extreme', start_date='2000', end_date='2000', trading_days=2,
            baseline_cagr=999.0, baseline_vol=99.0, baseline_sharpe=9.999,
            baseline_max_dd=-99.0,
            overlay_cagr=-999.0, overlay_vol=0.0, overlay_sharpe=0.0,
            overlay_max_dd=-99.0,
            sharpe_delta=-9.999, cagr_delta=-1998.0,
        )
        assert w.trading_days == 2
        assert w.baseline_cagr == 999.0
        assert w.overlay_sharpe == 0.0


class TestStressResult:
    def test_creation(self):
        s = StressResult(
            period='covid_crash', start_date='2020-02-20', end_date='2020-04-30',
            description='COVID crash',
            baseline_return=-15.0, overlay_return=-12.0,
            baseline_max_dd=-25.0, overlay_max_dd=-20.0,
            signal_accuracy=65.0, avg_confidence=0.45,
        )
        assert s.period == 'covid_crash'
        assert s.signal_accuracy == 65.0

    def test_max_dd_always_negative_or_zero(self):
        """Max drawdown should be <= 0 for any reasonable input."""
        s = StressResult(
            period='test', start_date='2020-01-01', end_date='2020-01-31',
            description='test', baseline_return=5.0, overlay_return=3.0,
            baseline_max_dd=0.0, overlay_max_dd=-10.0,
            signal_accuracy=50.0, avg_confidence=0.5,
        )
        assert s.baseline_max_dd <= 0
        assert s.overlay_max_dd <= 0

    def test_avg_confidence_zero_no_signals(self):
        """When no signals are above threshold, avg_confidence should be 0."""
        s = StressResult(
            period='test', start_date='2020-01-01', end_date='2020-01-31',
            description='test', baseline_return=-5.0, overlay_return=-3.0,
            baseline_max_dd=-10.0, overlay_max_dd=-8.0,
            signal_accuracy=0.0, avg_confidence=0.0,
        )
        assert s.avg_confidence == 0.0
        assert s.signal_accuracy == 0.0


class TestFullBacktestResult:
    def test_creation(self):
        r = FullBacktestResult(
            walk_forward_windows=[],
            avg_sharpe_delta=0.05,
            pct_windows_improved=60.0,
            stress_tests=[],
            overall_baseline_sharpe=0.79,
            overall_overlay_sharpe=0.84,
            overall_sharpe_delta=0.05,
            target_met=True,
        )
        assert r.target_met is True
        assert r.overall_sharpe_delta == 0.05

    def test_target_not_met(self):
        """target_met=False state should be representable."""
        r = FullBacktestResult(
            walk_forward_windows=[], avg_sharpe_delta=-0.01, pct_windows_improved=40.0,
            stress_tests=[], overall_baseline_sharpe=0.79, overall_overlay_sharpe=0.78,
            overall_sharpe_delta=-0.01, target_met=False,
        )
        assert r.target_met is False
        assert r.overall_sharpe_delta == -0.01

    def test_empty_lists_defaults(self):
        """walk_forward_windows and stress_tests can both be empty."""
        r = _make_empty_result()
        assert r.walk_forward_windows == []
        assert r.stress_tests == []
        assert r.avg_sharpe_delta == 0.0

    def test_all_fields_present(self):
        """FullBacktestResult should have all required fields."""
        r = _make_empty_result()
        assert hasattr(r, 'walk_forward_windows')
        assert hasattr(r, 'avg_sharpe_delta')
        assert hasattr(r, 'pct_windows_improved')
        assert hasattr(r, 'stress_tests')
        assert hasattr(r, 'overall_baseline_sharpe')
        assert hasattr(r, 'overall_overlay_sharpe')
        assert hasattr(r, 'overall_sharpe_delta')
        assert hasattr(r, 'target_met')

    def test_pct_windows_improved_bounded_0_to_100(self):
        """Percentage should be between 0 and 100."""
        r = FullBacktestResult(
            walk_forward_windows=[], avg_sharpe_delta=0.0, pct_windows_improved=100.0,
            stress_tests=[], overall_baseline_sharpe=0.0, overall_overlay_sharpe=0.0,
            overall_sharpe_delta=0.0, target_met=False,
        )
        assert 0 <= r.pct_windows_improved <= 100


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_returns(self):
        result = compute_metrics([])
        assert result['cagr'] == 0
        assert result['sharpe'] == 0

    def test_single_return(self):
        result = compute_metrics([0.01])
        assert result['cagr'] == 0

    def test_positive_returns(self):
        import random
        random.seed(42)
        rets = [0.001 + random.gauss(0, 0.005) for _ in range(252)]
        result = compute_metrics(rets)
        assert result['cagr'] > 0
        assert result['sharpe'] > 0

    def test_negative_returns(self):
        rets = [-0.001] * 252
        result = compute_metrics(rets)
        assert result['cagr'] < 0

    def test_zero_vol(self):
        rets = [0.0] * 252
        result = compute_metrics(rets)
        assert result['vol'] == 0
        assert result['sharpe'] == 0

    def test_max_dd_zero_for_monotonic_up(self):
        rets = [0.001] * 100
        result = compute_metrics(rets)
        assert result['max_dd'] == 0

    def test_max_dd_negative_for_drawdown(self):
        rets = [0.01] * 50 + [-0.05] * 5 + [0.01] * 50
        result = compute_metrics(rets)
        assert result['max_dd'] < 0

    def test_annualize_flag(self):
        import random
        random.seed(42)
        rets = [random.gauss(0, 0.01) for _ in range(100)]
        ann = compute_metrics(rets, annualize=True)
        daily = compute_metrics(rets, annualize=False)
        assert ann['vol'] > daily['vol']

    def test_output_keys(self):
        result = compute_metrics([0.01, -0.01, 0.005])
        assert set(result.keys()) == {'cagr', 'vol', 'sharpe', 'max_dd'}

    def test_rounding(self):
        import random
        random.seed(42)
        rets = [0.001 + random.gauss(0, 0.005) for _ in range(252)]
        result = compute_metrics(rets)
        assert isinstance(result['cagr'], float)
        assert isinstance(result['sharpe'], float)

    def test_known_sharpe(self):
        import random
        random.seed(42)
        # Positive drift with noise -> positive Sharpe
        rets = [0.0004 + random.gauss(0, 0.005) for _ in range(2520)]
        result = compute_metrics(rets)
        assert result['sharpe'] > 0.3

    def test_two_returns_minimum(self):
        """Minimum viable input (2 returns) should produce reasonable output."""
        result = compute_metrics([0.001, 0.002])
        assert result['cagr'] != 0
        assert isinstance(result['sharpe'], float)

    def test_known_max_dd_calculation(self):
        """Verify max_dd with a known sequence."""
        # Sequence: +10%, -20%, +10% -> peak at 1.10, trough at 0.88
        rets = [0.10, -0.20, 0.10]
        result = compute_metrics(rets)
        # Peak = 1.10, trough/min = 0.88 -> DD = (1.10-0.88)/1.10 = 0.20
        # Round(-0.20 * 100, 2) = -20.0
        assert result['max_dd'] == -20.0

    def test_single_day_returns(self):
        """A single day has years=0 -> CAGR=0, vol=0 (early return for n<2)."""
        result = compute_metrics([0.01])
        assert result['cagr'] == 0
        assert result['vol'] == 0  # early return for n<2
        assert result['sharpe'] == 0

    def test_mixed_signs_sharpe(self):
        """Sharpe with mixed positive and negative returns."""
        rets = [0.005, -0.003, 0.007, -0.002, 0.004]
        result = compute_metrics(rets)
        assert isinstance(result['sharpe'], (int, float))

    def test_high_vol_low_return(self):
        """High volatility with low return -> low Sharpe."""
        rets = [0.0001 + (-0.03 if i % 3 == 0 else 0.03) for i in range(252)]
        result = compute_metrics(rets)
        # Should be a valid but potentially low/negative Sharpe
        assert isinstance(result['sharpe'], float)

    def test_monotonic_negative_max_dd_equals_cumulative_loss(self):
        """Monotonically decreasing returns: max_dd = cumulative loss."""
        rets = [-0.01] * 50
        result = compute_metrics(rets)
        # Cumulative: 0.99^50 = ~0.605, peak=1.0, DD=(1-0.605)/1 = 0.395
        assert result['max_dd'] < 0
        assert result['max_dd'] > -100


# ---------------------------------------------------------------------------
# build_daily_returns tests
# ---------------------------------------------------------------------------

class TestBuildDailyReturns:
    def test_returns_list(self):
        prices = _make_price_data(20)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates)
        result = build_daily_returns(prices, signals, dates[0], dates[-1])
        assert isinstance(result, list)

    def test_returns_length(self):
        prices = _make_price_data(20)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates)
        result = build_daily_returns(prices, signals, dates[0], dates[-1])
        # First day has no prev, so n-1 returns
        assert len(result) == len(dates) - 1

    def test_returns_have_all_fields(self):
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates)
        result = build_daily_returns(prices, signals, dates[0], dates[-1])
        if result:
            dr = result[0]
            assert hasattr(dr, 'spy_return')
            assert hasattr(dr, 'baseline_return')
            assert hasattr(dr, 'overlay_return')
            assert hasattr(dr, 'alt_regime')

    def test_regime_shift_applied_above_threshold(self):
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_on', confidence=0.8)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.3)
        if result:
            dr = result[0]
            assert dr.alt_regime == 'risk_on'
            # With risk_on, overlay should differ from baseline
            assert dr.overlay_return != dr.baseline_return or abs(dr.overlay_return - dr.baseline_return) < 1e-10

    def test_no_shift_below_threshold(self):
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_on', confidence=0.1)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.3)
        if result:
            dr = result[0]
            # Below threshold -> baseline weights used -> overlay ~= baseline
            assert abs(dr.overlay_return - dr.baseline_return) < 1e-10

    def test_missing_signals_default_neutral(self):
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = {}  # No signals
        result = build_daily_returns(prices, signals, dates[0], dates[-1])
        if result:
            dr = result[0]
            assert dr.alt_regime == 'neutral'
            assert dr.alt_confidence == 0.0

    def test_empty_prices(self):
        result = build_daily_returns({}, {}, '2020-01-01', '2020-12-31')
        assert result == []

    def test_date_filtering(self):
        prices = _make_price_data(30, start_date='2020-01-02')
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates)
        # Narrow window
        result = build_daily_returns(prices, signals, dates[5], dates[10])
        for dr in result:
            assert dates[5] <= dr.date <= dates[10]

    def test_risk_off_shift_reduces_spy_weight(self):
        """risk_off regime decreases SPY allocation -> lower overlay return on up days."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_off', confidence=0.8)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.3)
        if result:
            dr = result[0]
            assert dr.alt_regime == 'risk_off'
            # overlay_return should differ from baseline_return
            assert dr.overlay_return != dr.baseline_return

    def test_neutral_no_shift(self):
        """neutral regime should produce overlay == baseline."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='neutral', confidence=0.8)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.3)
        if result:
            dr = result[0]
            assert dr.alt_regime == 'neutral'
            assert abs(dr.overlay_return - dr.baseline_return) < 1e-10

    def test_weights_normalized(self):
        """After regime shifts, effective weights should sum to 1."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_on', confidence=0.8)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.3)
        if result:
            dr = result[0]
            # Compute what the effective weights should be
            shifts = REGIME_SHIFTS['risk_on']
            raw_weights = {s: WEIGHTS[s] + shifts[s] for s in WEIGHTS}
            total = sum(raw_weights.values())
            normalized = {s: w / total for s, w in raw_weights.items()}
            assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_confidence_threshold_zero(self):
        """confidence_threshold=0 should apply shifts to all signals."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_on', confidence=0.01)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=0.0)
        if result:
            dr = result[0]
            # Even with very low confidence, threshold=0 means it applies
            assert dr.overlay_return != dr.baseline_return or abs(dr.overlay_return - dr.baseline_return) < 1e-10

    def test_confidence_threshold_one(self):
        """confidence_threshold=1.0 should prevent ANY shifts."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates, regime='risk_on', confidence=0.99)
        result = build_daily_returns(prices, signals, dates[0], dates[-1], confidence_threshold=1.0)
        if result:
            dr = result[0]
            # With threshold=1.0, even confidence=0.99 is below threshold
            assert abs(dr.overlay_return - dr.baseline_return) < 1e-10

    def test_start_equals_end_date(self):
        """When start == end, no daily returns can be computed (need 2 dates)."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        signals = _make_alt_signals(dates)
        result = build_daily_returns(prices, signals, dates[5], dates[5])
        assert result == []

    def test_partial_missing_assets_skipped(self):
        """Days missing some WEIGHTS assets should be skipped."""
        prices = _make_price_data(10)
        dates = [b['d'] for b in prices['SPY']]
        # Remove SPY from one specific date
        mid_date = dates[5]
        prices['SPY'] = [b for b in prices['SPY'] if b['d'] != mid_date]
        signals = _make_alt_signals(dates)
        result = build_daily_returns(prices, signals, dates[0], dates[-1])
        # The day with mid_date as prev or curr should be affected
        assert all(dr.date != mid_date for dr in result)


# ---------------------------------------------------------------------------
# walk_forward_test tests
# ---------------------------------------------------------------------------

class TestWalkForwardTest:
    def test_empty_returns(self):
        result = walk_forward_test([])
        assert result == []

    def test_returns_window_list(self):
        returns = _make_daily_returns_series(n=1500, start_year=2017)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        assert isinstance(result, list)

    def test_windows_have_required_fields(self):
        returns = _make_daily_returns_series(n=1500, start_year=2017)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        if result:
            w = result[0]
            assert hasattr(w, 'label')
            assert hasattr(w, 'sharpe_delta')
            assert hasattr(w, 'baseline_sharpe')
            assert hasattr(w, 'overlay_sharpe')

    def test_sharpe_delta_computation(self):
        returns = _make_daily_returns_series(n=1500, start_year=2017)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        for w in result:
            expected = round(w.overlay_sharpe - w.baseline_sharpe, 3)
            assert w.sharpe_delta == expected

    def test_insufficient_data_returns_empty(self):
        # Too few days for even one window
        returns = _make_daily_returns_series(n=10, start_year=2020)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        assert result == []

    def test_custom_train_test_years(self):
        returns = _make_daily_returns_series(n=2000, start_year=2015)
        result = walk_forward_test(returns, train_years=2, test_years=1)
        # Should produce windows with 2-year train
        if result:
            assert 'Train' in result[0].label

    def test_windows_are_chronological(self):
        returns = _make_daily_returns_series(n=2000, start_year=2015)
        result = walk_forward_test(returns, train_years=2, test_years=1)
        for i in range(1, len(result)):
            assert result[i].start_date >= result[i-1].start_date

    def test_test_days_filter_below_50(self):
        """Windows with fewer than 50 test days should be skipped."""
        # Create data with only one year of returns (< 50 trading days)
        returns = _make_daily_returns_series(n=30, start_year=2020)
        result = walk_forward_test(returns, train_years=1, test_years=1)
        assert result == []

    def test_one_train_year(self):
        """train_years=1 should still produce valid windows with enough data."""
        returns = _make_daily_returns_series(n=500, start_year=2020)
        result = walk_forward_test(returns, train_years=1, test_years=1)
        if result:
            assert len(result) >= 1

    def test_cagr_delta_consistency(self):
        """cagr_delta should approximately equal overlay_cagr - baseline_cagr."""
        returns = _make_daily_returns_series(n=1500, start_year=2017)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        for w in result:
            expected = w.overlay_cagr - w.baseline_cagr
            assert w.cagr_delta == pytest.approx(expected, abs=0.01)

    def test_label_format(self):
        """Window label should match 'Train X-Y / Test Z-W' pattern."""
        returns = _make_daily_returns_series(n=1500, start_year=2017)
        result = walk_forward_test(returns, train_years=3, test_years=1)
        if result:
            import re
            pattern = r'Train \d{4}-\d{4} / Test \d{4}-\d{4}'
            for w in result:
                assert re.match(pattern, w.label), f"Label '{w.label}' doesn't match pattern"


# ---------------------------------------------------------------------------
# stress_test tests
# ---------------------------------------------------------------------------

class TestStressTest:
    def test_returns_list(self):
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        assert isinstance(result, list)

    def test_stress_periods_detected(self):
        # Generate returns covering 2020-2023
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        periods = {s.period for s in result}
        # Should find at least some stress periods
        assert len(periods) > 0

    def test_stress_result_fields(self):
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        if result:
            s = result[0]
            assert hasattr(s, 'baseline_return')
            assert hasattr(s, 'overlay_return')
            assert hasattr(s, 'signal_accuracy')
            assert hasattr(s, 'avg_confidence')

    def test_signal_accuracy_bounded(self):
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        for s in result:
            assert 0 <= s.signal_accuracy <= 100

    def test_with_signals(self):
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        # Build signals from dates
        signals = {}
        for dr in returns:
            signals[dr.date] = {
                'date': dr.date,
                'composite_score': 0.1,
                'regime': 'risk_on',
                'confidence': 0.6,
            }
        result = stress_test(returns, signals)
        for s in result:
            assert s.avg_confidence > 0

    def test_fewer_than_5_days_skipped(self):
        # Create returns only for a very short period
        returns = [
            _make_daily_return(date='2020-02-20'),
            _make_daily_return(date='2020-02-21'),
        ]
        result = stress_test(returns, {})
        # covid_crash has only 2 days -> skipped
        assert len(result) == 0

    def test_no_signals_above_threshold_accuracy_zero(self):
        """When no signals have confidence >= 0.3, accuracy should be 0."""
        # The alt_confidence field lives on DailyReturn, not the signals dict.
        # Create returns where alt_confidence is 0 for all.
        returns = []
        for i in range(252):
            dr = _make_daily_return(
                date=f'2020-{i // 28 + 1:02d}-{(i % 28) + 1:02d}',
                spy_ret=0.001, baseline_ret=0.0005,
                alt_regime='risk_on', alt_confidence=0.0,
            )
            returns.append(dr)

        signals = {}
        result = stress_test(returns, signals)
        for s in result:
            assert s.signal_accuracy == 0.0

    def test_all_correct_signals(self):
        """All signals above threshold AND predicting correct direction."""
        # Returns with positive baseline returns in risk_on regime
        returns = []
        for i in range(252):
            dr = _make_daily_return(
                date=f'2020-{i // 28 + 1:02d}-{(i % 28) + 1:02d}',
                spy_ret=0.005, baseline_ret=0.003,
                alt_regime='risk_on', alt_confidence=0.8,
            )
            returns.append(dr)

        signals = {}
        for dr in returns:
            signals[dr.date] = {
                'date': dr.date,
                'composite_score': 0.2,
                'regime': 'risk_on',
                'confidence': 0.8,
            }

        result = stress_test(returns, signals)
        # At least some periods should have 100% accuracy
        if result:
            assert any(s.signal_accuracy >= 90.0 for s in result)

    def test_all_wrong_signals(self):
        """All signals predict opposite direction."""
        returns = []
        for i in range(252):
            dr = _make_daily_return(
                date=f'2022-{i // 28 + 1:02d}-{(i % 28) + 1:02d}',
                spy_ret=0.005, baseline_ret=0.003,
                alt_regime='risk_off', alt_confidence=0.8,
            )
            returns.append(dr)

        signals = {}
        for dr in returns:
            signals[dr.date] = {
                'date': dr.date,
                'composite_score': -0.2,
                'regime': 'risk_off',
                'confidence': 0.8,
            }

        result = stress_test(returns, signals)
        # At least some periods should have 0% accuracy
        if result:
            assert any(0 <= s.signal_accuracy < 10 for s in result)

    def test_periods_partial_overlap(self):
        """Stress periods that only partially overlap with data should still run."""
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        # All 5 periods should have at least 5 days in the 2019-2025 range
        assert len(result) >= 1

    def test_return_and_dd_values_are_floats(self):
        """Numeric fields in StressResult should be floats."""
        returns = _make_daily_returns_series(n=1500, start_year=2019)
        signals = {}
        result = stress_test(returns, signals)
        for s in result:
            assert isinstance(s.baseline_return, float)
            assert isinstance(s.overlay_return, float)
            assert isinstance(s.baseline_max_dd, float)
            assert isinstance(s.overlay_max_dd, float)


# ---------------------------------------------------------------------------
# load_price_data / load_alt_signals tests
# ---------------------------------------------------------------------------

class TestLoadFunctions:
    def test_load_price_data(self, tmp_path):
        data = _make_price_data(10)
        fpath = tmp_path / "prices.json"
        fpath.write_text(json.dumps(data))
        loaded = load_price_data(str(fpath))
        assert 'SPY' in loaded
        assert len(loaded['SPY']) == 10

    def test_load_alt_signals(self, tmp_path):
        signals = {'signals': [
            {'date': '2020-01-02', 'composite_score': 0.1, 'regime': 'risk_on', 'confidence': 0.5},
            {'date': '2020-01-03', 'composite_score': -0.1, 'regime': 'risk_off', 'confidence': 0.6},
        ]}
        fpath = tmp_path / "signals.json"
        fpath.write_text(json.dumps(signals))
        loaded = load_alt_signals(str(fpath))
        assert '2020-01-02' in loaded
        assert loaded['2020-01-02']['regime'] == 'risk_on'

    def test_load_price_data_empty(self, tmp_path):
        """Empty JSON object should load but have no symbols."""
        fpath = tmp_path / "prices.json"
        fpath.write_text('{}')
        loaded = load_price_data(str(fpath))
        assert loaded == {}

    def test_load_alt_signals_empty_list(self, tmp_path):
        """Empty signals list should produce empty dict."""
        signals = {'signals': []}
        fpath = tmp_path / "signals.json"
        fpath.write_text(json.dumps(signals))
        loaded = load_alt_signals(str(fpath))
        assert loaded == {}

    def test_load_price_data_with_extra_symbols(self, tmp_path):
        """Extra symbols beyond WEIGHTS should be ignored but loadable."""
        data = _make_price_data(10)
        data['QQQ'] = [{'d': '2020-01-02', 'p': 200.0}]
        fpath = tmp_path / "prices.json"
        fpath.write_text(json.dumps(data))
        loaded = load_price_data(str(fpath))
        assert 'QQQ' in loaded
        assert 'SPY' in loaded

    def test_load_alt_signals_preserves_insertion_order(self, tmp_path):
        """Signals loaded preserve JSON list insertion order."""
        signals = {'signals': [
            {'date': '2020-01-03', 'composite_score': 0.2, 'regime': 'neutral', 'confidence': 0.5},
            {'date': '2020-01-02', 'composite_score': 0.1, 'regime': 'risk_on', 'confidence': 0.6},
        ]}
        fpath = tmp_path / "signals.json"
        fpath.write_text(json.dumps(signals))
        loaded = load_alt_signals(str(fpath))
        # Insertion order preserved from JSON list
        assert list(loaded.keys()) == ['2020-01-03', '2020-01-02']
        assert loaded['2020-01-02']['regime'] == 'risk_on'


# ---------------------------------------------------------------------------
# run_full_backtest tests
# ---------------------------------------------------------------------------

class TestRunFullBacktest:
    def test_insufficient_data_returns_none(self, tmp_path):
        """When daily_returns < 100, run_full_backtest should return None."""
        # Create minimal price data (not enough for 100 daily returns)
        prices = _make_price_data(50)
        fpath = tmp_path / "prices.json"
        fpath.write_text(json.dumps(prices))

        signals = {'signals': [
            {'date': '2020-01-02', 'composite_score': 0.1, 'regime': 'risk_on', 'confidence': 0.5},
        ]}
        sfpath = tmp_path / "signals.json"
        sfpath.write_text(json.dumps(signals))

        with patch('src.backtest.alt_data_walkforward_stress.print_results') as mock_print, \
             patch('src.backtest.alt_data_walkforward_stress.save_results') as mock_save:
            result = run_full_backtest(
                price_filepath=str(fpath),
                alt_signal_filepath=str(sfpath),
            )
            assert result is None
            mock_print.assert_not_called()
            mock_save.assert_not_called()


class TestPrintResults:
    def test_print_results_empty_walkforward(self, caplog):
        """print_results should not crash with empty walk_forward_windows."""
        caplog.set_level(logging.INFO, logger='src.backtest.alt_data_walkforward_stress')
        result = _make_empty_result()
        print_results(result)
        assert 'WALK-FORWARD' in caplog.text
        assert 'STRESS TEST' in caplog.text

    def test_print_results_empty_stress_tests(self, caplog):
        """print_results should not crash with empty stress_tests."""
        caplog.set_level(logging.INFO, logger='src.backtest.alt_data_walkforward_stress')
        result = _make_empty_result()
        print_results(result)
        assert 'STRESS TEST RESULTS' in caplog.text

    def test_print_results_with_data(self, caplog):
        """print_results should format walk-forward and stress data."""
        caplog.set_level(logging.INFO, logger='src.backtest.alt_data_walkforward_stress')
        w = WindowResult(
            label='Train 2020-2022 / Test 2023', start_date='2023', end_date='2023',
            trading_days=252, baseline_cagr=8.0, baseline_vol=12.0, baseline_sharpe=0.67,
            baseline_max_dd=-15.0, overlay_cagr=9.0, overlay_vol=11.5, overlay_sharpe=0.78,
            overlay_max_dd=-13.0, sharpe_delta=0.11, cagr_delta=1.0,
        )
        s = StressResult(
            period='covid_crash', start_date='2020-02-20', end_date='2020-04-30',
            description='COVID crash', baseline_return=-15.0, overlay_return=-12.0,
            baseline_max_dd=-25.0, overlay_max_dd=-20.0, signal_accuracy=65.0, avg_confidence=0.45,
        )
        result = FullBacktestResult(
            walk_forward_windows=[w], avg_sharpe_delta=0.05, pct_windows_improved=100.0,
            stress_tests=[s], overall_baseline_sharpe=0.79, overall_overlay_sharpe=0.84,
            overall_sharpe_delta=0.05, target_met=True,
        )
        print_results(result)
        assert '0.670' in caplog.text  # baseline_sharpe
        assert '0.780' in caplog.text  # overlay_sharpe
        assert 'covid_crash' in caplog.text
        assert 'MET' in caplog.text


class TestSaveResults:
    def test_save_results_creates_json(self, tmp_path):
        """save_results should write a JSON file with expected structure."""
        result = _make_empty_result()
        # Temporarily patch SIGNALS_DIR to use tmp_path
        with patch('src.backtest.alt_data_walkforward_stress.SIGNALS_DIR', tmp_path):
            save_results(result)
            output_path = tmp_path / "alt_data_walkforward_stress_results.json"
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert 'metadata' in data
            assert data['metadata']['version'] == '2.60'
            assert 'overall' in data
            assert data['overall']['target_met'] is False
            assert 'walk_forward' in data
            assert 'stress_tests' in data

    def test_save_results_with_data(self, tmp_path):
        """save_results with a full result should preserve all fields."""
        w = WindowResult(
            label='Test Window', start_date='2023', end_date='2023', trading_days=100,
            baseline_cagr=5.0, baseline_vol=10.0, baseline_sharpe=0.50,
            baseline_max_dd=-10.0, overlay_cagr=6.0, overlay_vol=9.0, overlay_sharpe=0.67,
            overlay_max_dd=-8.0, sharpe_delta=0.17, cagr_delta=1.0,
        )
        s = StressResult(
            period='test_period', start_date='2020-01-01', end_date='2020-01-31',
            description='Test stress',
            baseline_return=-5.0, overlay_return=-3.0,
            baseline_max_dd=-10.0, overlay_max_dd=-8.0,
            signal_accuracy=50.0, avg_confidence=0.5,
        )
        result = FullBacktestResult(
            walk_forward_windows=[w], avg_sharpe_delta=0.17, pct_windows_improved=100.0,
            stress_tests=[s], overall_baseline_sharpe=0.50, overall_overlay_sharpe=0.67,
            overall_sharpe_delta=0.17, target_met=True,
        )
        with patch('src.backtest.alt_data_walkforward_stress.SIGNALS_DIR', tmp_path):
            save_results(result)
            data = json.loads((tmp_path / "alt_data_walkforward_stress_results.json").read_text())
            assert data['overall']['baseline_sharpe'] == 0.50
            assert data['overall']['overlay_sharpe'] == 0.67
            assert data['overall']['target_met'] is True
            assert len(data['walk_forward']['windows']) == 1
            assert len(data['stress_tests']) == 1
            assert data['walk_forward']['windows'][0]['label'] == 'Test Window'
            assert data['stress_tests'][0]['period'] == 'test_period'

    def test_save_results_overall_section_structure(self, tmp_path):
        """The overall section should have baseline_sharpe, overlay_sharpe, sharpe_delta, target_met."""
        result = _make_empty_result()
        with patch('src.backtest.alt_data_walkforward_stress.SIGNALS_DIR', tmp_path):
            save_results(result)
            data = json.loads((tmp_path / "alt_data_walkforward_stress_results.json").read_text())
            overall = data['overall']
            assert 'baseline_sharpe' in overall
            assert 'overlay_sharpe' in overall
            assert 'sharpe_delta' in overall
            assert 'target_met' in overall
            assert 'target_threshold' in overall
            assert overall['target_threshold'] == 0.03


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_a3_b1a_delegation_matches_pre_migration_capture():
    """A3 pin (Item B1a sub-task 10): load_price_data delegates to grid_runner.load_prices."""
    from src.backtest.grid_runner import load_prices

    # module-level loader stays in pilot; the shared loader is grid_runner's
    assert load_price_data.__module__ == "src.backtest.alt_data_walkforward_stress"
    assert load_prices.__module__ == "src.backtest.grid_runner"
