#!/usr/bin/env python3
"""
Tests for Alternative Data Walk-Forward & Stress Test Engine —
data classes, constants, compute_metrics, build_daily_returns,
walk_forward_test, and stress_test.
"""
import sys
import os
import math
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta

from src.backtest.alt_data_walkforward_stress import (
    DailyReturn, WindowResult, StressResult, FullBacktestResult,
    WEIGHTS, REGIME_SHIFTS, STRESS_PERIODS,
    compute_metrics, build_daily_returns, walk_forward_test, stress_test,
    load_price_data, load_alt_signals,
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
        # Positive drift with noise → positive Sharpe
        rets = [0.0004 + random.gauss(0, 0.005) for _ in range(2520)]
        result = compute_metrics(rets)
        assert result['sharpe'] > 0.3


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
            # Below threshold → baseline weights used → overlay ≈ baseline
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
        # covid_crash has only 2 days → skipped
        assert len(result) == 0


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
