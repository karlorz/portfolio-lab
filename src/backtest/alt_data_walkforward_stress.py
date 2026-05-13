"""
Alternative Data Walk-Forward & Stress Test Engine
v2.60 Phase 4.2-4.3 - Portfolio-Lab

Walk-forward validation: 3-year train, 1-year test, rolling windows
Stress testing: COVID crash, 2022 bear market, 2021 meme stock noise

Integrates alternative data signals with price data for backtest.
"""

import json
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class DailyReturn:
    date: str
    spy_return: float
    gld_return: float
    tlt_return: float
    alt_signal: float         # composite_score from alt data
    alt_regime: str           # risk_on/risk_off/neutral
    alt_confidence: float
    # Baseline portfolio return (46/38/16)
    baseline_return: float
    # Alt-data overlay portfolio return
    overlay_return: float


@dataclass
class WindowResult:
    label: str
    start_date: str
    end_date: str
    trading_days: int
    # Baseline metrics
    baseline_cagr: float
    baseline_vol: float
    baseline_sharpe: float
    baseline_max_dd: float
    # Overlay metrics
    overlay_cagr: float
    overlay_vol: float
    overlay_sharpe: float
    overlay_max_dd: float
    # Delta
    sharpe_delta: float
    cagr_delta: float


@dataclass
class StressResult:
    period: str
    start_date: str
    end_date: str
    description: str
    baseline_return: float
    overlay_return: float
    baseline_max_dd: float
    overlay_max_dd: float
    signal_accuracy: float   # % of days signal correctly predicted direction
    avg_confidence: float


@dataclass
class FullBacktestResult:
    # Walk-forward results
    walk_forward_windows: List[WindowResult]
    avg_sharpe_delta: float
    pct_windows_improved: float
    # Stress test results
    stress_tests: List[StressResult]
    # Overall
    overall_baseline_sharpe: float
    overall_overlay_sharpe: float
    overall_sharpe_delta: float
    target_met: bool  # +0.03 Sharpe


# Base portfolio weights: SPY/GLD/TLT 46/38/16
WEIGHTS = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}

# Alt-data regime allocation shifts
REGIME_SHIFTS = {
    'risk_on':  {'SPY': +0.05, 'GLD': -0.03, 'TLT': -0.02},
    'risk_off': {'SPY': -0.05, 'GLD': +0.03, 'TLT': +0.02},
    'neutral':  {'SPY': 0.00,  'GLD': 0.00,  'TLT': 0.00},
}

# Stress test periods
STRESS_PERIODS = {
    'covid_crash': {
        'start': '2020-02-20',
        'end': '2020-04-30',
        'description': 'COVID-19 crash: rapid market selloff, news sentiment surge',
    },
    'covid_recovery': {
        'start': '2020-05-01',
        'end': '2020-12-31',
        'description': 'COVID recovery: V-shaped rebound, mixed signals',
    },
    'meme_stock_2021': {
        'start': '2021-01-15',
        'end': '2021-03-31',
        'description': 'Meme stock craze: GME/AMC, social media noise spike',
    },
    'bear_2022': {
        'start': '2022-01-01',
        'end': '2022-10-31',
        'description': '2022 bear market: rate hikes, inflation, earnings guidance cuts',
    },
    'rate_hike_2023': {
        'start': '2023-01-01',
        'end': '2023-10-31',
        'description': 'Rate hike plateau: banking stress (SVB), peak rates',
    },
}


def load_price_data(filepath: str) -> Dict[str, List[Dict]]:
    """Load price data from JSON."""
    with open(filepath) as f:
        return json.load(f)


def load_alt_signals(filepath: str) -> Dict[str, Dict]:
    """Load alternative data signals indexed by date."""
    with open(filepath) as f:
        data = json.load(f)
    return {s['date']: s for s in data['signals']}


def build_daily_returns(
    prices: Dict[str, List[Dict]],
    alt_signals: Dict[str, Dict],
    start_date: str,
    end_date: str,
    confidence_threshold: float = 0.3,
) -> List[DailyReturn]:
    """Build daily returns with alt-data overlay."""
    # Build price index by date
    price_idx = {}
    for sym in WEIGHTS:
        if sym not in prices:
            continue
        for bar in prices[sym]:
            d = bar['d']
            if d not in price_idx:
                price_idx[d] = {}
            price_idx[d][sym] = bar['p']

    # Get sorted trading dates
    all_dates = sorted(price_idx.keys())
    dates_in_range = [d for d in all_dates if start_date <= d <= end_date]

    results = []
    prev_prices = None

    for date in dates_in_range:
        if date not in price_idx:
            continue
        curr_prices = price_idx[date]

        # Need all 3 assets
        if not all(s in curr_prices for s in WEIGHTS):
            continue

        if prev_prices is not None and all(s in prev_prices for s in WEIGHTS):
            # Calculate returns
            rets = {}
            for sym in WEIGHTS:
                rets[sym] = (curr_prices[sym] - prev_prices[sym]) / prev_prices[sym]

            # Baseline portfolio return
            baseline_ret = sum(WEIGHTS[s] * rets[s] for s in WEIGHTS)

            # Alt-data signal
            alt = alt_signals.get(date, {})
            composite = alt.get('composite_score', 0.0)
            regime = alt.get('regime', 'neutral')
            confidence = alt.get('confidence', 0.0)

            # Apply regime shift if confidence above threshold
            shifts = REGIME_SHIFTS.get(regime, REGIME_SHIFTS['neutral'])
            if confidence >= confidence_threshold:
                effective_weights = {s: WEIGHTS[s] + shifts[s] for s in WEIGHTS}
            else:
                effective_weights = WEIGHTS.copy()

            # Normalize weights to sum to 1
            total_w = sum(effective_weights.values())
            effective_weights = {s: w / total_w for s, w in effective_weights.items()}

            overlay_ret = sum(effective_weights[s] * rets[s] for s in WEIGHTS)

            results.append(DailyReturn(
                date=date,
                spy_return=rets['SPY'],
                gld_return=rets['GLD'],
                tlt_return=rets['TLT'],
                alt_signal=composite,
                alt_regime=regime,
                alt_confidence=confidence,
                baseline_return=baseline_ret,
                overlay_return=overlay_ret,
            ))

        prev_prices = curr_prices

    return results


def compute_metrics(returns: List[float], annualize: bool = True) -> Dict:
    """Compute CAGR, volatility, Sharpe, max drawdown."""
    if not returns or len(returns) < 2:
        return {'cagr': 0, 'vol': 0, 'sharpe': 0, 'max_dd': 0}

    n = len(returns)
    total_ret = 1.0
    for r in returns:
        total_ret *= (1 + r)

    years = n / 252
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0

    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    vol_daily = math.sqrt(var_r)
    vol_annual = vol_daily * math.sqrt(252) if annualize else vol_daily

    sharpe = cagr / vol_annual if vol_annual > 0 else 0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    cum = 1.0
    for r in returns:
        cum *= (1 + r)
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'cagr': round(cagr * 100, 2),
        'vol': round(vol_annual * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(-max_dd * 100, 2),
    }


def walk_forward_test(
    daily_returns: List[DailyReturn],
    train_years: int = 3,
    test_years: int = 1,
) -> List[WindowResult]:
    """Walk-forward validation with rolling windows."""
    if not daily_returns:
        return []

    # Group by year
    by_year = {}
    for dr in daily_returns:
        year = dr.date[:4]
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(dr)

    years = sorted(by_year.keys())
    windows = []

    i = 0
    while i + train_years + test_years <= len(years):
        train_start = years[i]
        test_end_year = i + train_years + test_years - 1
        test_end = years[test_end_year]

        # Collect test period returns
        test_returns_base = []
        test_returns_overlay = []
        for y_idx in range(i + train_years, i + train_years + test_years):
            y = years[y_idx]
            if y in by_year:
                for dr in by_year[y]:
                    test_returns_base.append(dr.baseline_return)
                    test_returns_overlay.append(dr.overlay_return)

        if len(test_returns_base) < 50:
            i += 1
            continue

        # Compute train period stats (for context)
        train_base = []
        for y_idx in range(i, i + train_years):
            y = years[y_idx]
            if y in by_year:
                for dr in by_year[y]:
                    train_base.append(dr.baseline_return)

        base_metrics = compute_metrics(test_returns_base)
        overlay_metrics = compute_metrics(test_returns_overlay)

        windows.append(WindowResult(
            label=f"Train {train_start}-{years[i+train_years-1]} / Test {years[i+train_years]}-{test_end}",
            start_date=years[i + train_years],
            end_date=test_end,
            trading_days=len(test_returns_base),
            baseline_cagr=base_metrics['cagr'],
            baseline_vol=base_metrics['vol'],
            baseline_sharpe=base_metrics['sharpe'],
            baseline_max_dd=base_metrics['max_dd'],
            overlay_cagr=overlay_metrics['cagr'],
            overlay_vol=overlay_metrics['vol'],
            overlay_sharpe=overlay_metrics['sharpe'],
            overlay_max_dd=overlay_metrics['max_dd'],
            sharpe_delta=round(overlay_metrics['sharpe'] - base_metrics['sharpe'], 3),
            cagr_delta=round(overlay_metrics['cagr'] - base_metrics['cagr'], 2),
        ))

        i += 1

    return windows


def stress_test(
    daily_returns: List[DailyReturn],
    alt_signals: Dict[str, Dict],
) -> List[StressResult]:
    """Stress test across known crisis periods."""
    results = []
    dr_by_date = {dr.date: dr for dr in daily_returns}

    for period_name, config in STRESS_PERIODS.items():
        period_returns = [
            dr for dr in daily_returns
            if config['start'] <= dr.date <= config['end']
        ]

        if len(period_returns) < 5:
            continue

        base_rets = [dr.baseline_return for dr in period_returns]
        overlay_rets = [dr.overlay_return for dr in period_returns]

        # Cumulative returns
        base_cum = 1.0
        overlay_cum = 1.0
        for r in base_rets:
            base_cum *= (1 + r)
        for r in overlay_rets:
            overlay_cum *= (1 + r)

        # Max drawdown for each
        def max_dd(returns):
            peak = 1.0
            mdd = 0.0
            cum = 1.0
            for r in returns:
                cum *= (1 + r)
                if cum > peak:
                    peak = cum
                dd = (peak - cum) / peak
                if dd > mdd:
                    mdd = dd
            return round(-mdd * 100, 2)

        # Signal accuracy: % of days where signal direction matches actual return
        correct = 0
        total = 0
        confidences = []
        for dr in period_returns:
            if dr.alt_confidence >= 0.3:
                # Signal says risk_on → expect positive, risk_off → expect negative
                signal_direction = 1 if dr.alt_regime == 'risk_on' else (-1 if dr.alt_regime == 'risk_off' else 0)
                actual_direction = 1 if dr.baseline_return > 0 else -1
                if signal_direction != 0:
                    total += 1
                    if signal_direction == actual_direction:
                        correct += 1
                confidences.append(dr.alt_confidence)

        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0

        results.append(StressResult(
            period=period_name,
            start_date=config['start'],
            end_date=config['end'],
            description=config['description'],
            baseline_return=round((base_cum - 1) * 100, 2),
            overlay_return=round((overlay_cum - 1) * 100, 2),
            baseline_max_dd=max_dd(base_rets),
            overlay_max_dd=max_dd(overlay_rets),
            signal_accuracy=accuracy,
            avg_confidence=avg_conf,
        ))

    return results


def run_full_backtest(
    price_filepath: str = '/root/projects/portfolio-lab/public/data/prices.json',
    alt_signal_filepath: str = '/root/projects/portfolio-lab/data/signals/alternative_data_historical_2020_2026.json',
    confidence_threshold: float = 0.3,
) -> FullBacktestResult:
    """Run complete Phase 4.2 + 4.3 backtest."""
    print("=" * 70)
    print("v2.60 Alternative Data - Phase 4.2-4.3 Walk-Forward & Stress Test")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading price data...")
    prices = load_price_data(price_filepath)
    print(f"  Loaded {len(prices)} symbols")

    print("[2/5] Loading alternative data signals...")
    alt_signals = load_alt_signals(alt_signal_filepath)
    print(f"  Loaded {len(alt_signals)} daily signals")

    # Build daily returns (2020-2026, matching alt data range)
    print("[3/5] Building daily returns with alt-data overlay...")
    daily_returns = build_daily_returns(
        prices, alt_signals,
        start_date='2020-01-01',
        end_date='2026-05-08',
        confidence_threshold=confidence_threshold,
    )
    print(f"  Built {len(daily_returns)} daily return records")

    if len(daily_returns) < 100:
        print("ERROR: Insufficient data for backtest")
        return None

    # Walk-forward test
    print("[4/5] Running walk-forward validation (3yr train / 1yr test)...")
    wf_windows = walk_forward_test(daily_returns, train_years=3, test_years=1)
    print(f"  Completed {len(wf_windows)} walk-forward windows")

    # Stress test
    print("[5/5] Running stress tests...")
    stress_results = stress_test(daily_returns, alt_signals)
    print(f"  Completed {len(stress_results)} stress test periods")

    # Compute overall metrics
    all_base = [dr.baseline_return for dr in daily_returns]
    all_overlay = [dr.overlay_return for dr in daily_returns]
    overall_base = compute_metrics(all_base)
    overall_overlay = compute_metrics(all_overlay)
    sharpe_delta = round(overall_overlay['sharpe'] - overall_base['sharpe'], 3)

    # Walk-forward summary
    if wf_windows:
        avg_wf_delta = round(sum(w.sharpe_delta for w in wf_windows) / len(wf_windows), 3)
        improved = sum(1 for w in wf_windows if w.sharpe_delta > 0)
        pct_improved = round(improved / len(wf_windows) * 100, 1)
    else:
        avg_wf_delta = 0
        pct_improved = 0

    target_met = sharpe_delta >= 0.03

    result = FullBacktestResult(
        walk_forward_windows=wf_windows,
        avg_sharpe_delta=avg_wf_delta,
        pct_windows_improved=pct_improved,
        stress_tests=stress_results,
        overall_baseline_sharpe=overall_base['sharpe'],
        overall_overlay_sharpe=overall_overlay['sharpe'],
        overall_sharpe_delta=sharpe_delta,
        target_met=target_met,
    )

    # Print results
    print_results(result)

    # Save results
    save_results(result)

    return result


def print_results(result: FullBacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    if result.walk_forward_windows:
        print(f"\n{'Window':<50} {'Base':>6} {'Overlay':>7} {'Delta':>6}")
        print("-" * 70)
        for w in result.walk_forward_windows:
            print(f"  {w.label:<48} {w.baseline_sharpe:>6.3f} {w.overlay_sharpe:>7.3f} {w.sharpe_delta:>+6.3f}")
        print("-" * 70)
        print(f"  {'Average':<48} {'':>6} {'':>7} {result.avg_sharpe_delta:>+6.3f}")
        print(f"  Windows improved: {result.pct_windows_improved}%")

    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)

    if result.stress_tests:
        for s in result.stress_tests:
            print(f"\n  [{s.period}] {s.description}")
            print(f"    Period: {s.start_date} to {s.end_date}")
            print(f"    Baseline return: {s.baseline_return:+.2f}%  |  Overlay: {s.overlay_return:+.2f}%  |  Delta: {s.overlay_return - s.baseline_return:+.2f}pp")
            print(f"    Baseline MaxDD:  {s.baseline_max_dd:.2f}%   |  Overlay: {s.overlay_max_dd:.2f}%   |  Delta: {s.overlay_max_dd - s.baseline_max_dd:+.2f}pp")
            print(f"    Signal accuracy: {s.signal_accuracy}%  |  Avg confidence: {s.avg_confidence:.3f}")

    print("\n" + "=" * 70)
    print("OVERALL RESULTS (2020-2026)")
    print("=" * 70)
    print(f"  Baseline (46/38/16) Sharpe: {result.overall_baseline_sharpe:.3f}")
    print(f"  Overlay (alt-data) Sharpe:  {result.overall_overlay_sharpe:.3f}")
    print(f"  Sharpe delta:               {result.overall_sharpe_delta:+.3f}")
    print(f"  Target (+0.03):             {'MET' if result.target_met else 'NOT MET'}")
    print("=" * 70)


def save_results(result: FullBacktestResult):
    """Save results to JSON."""
    output = {
        'metadata': {
            'version': '2.60',
            'phase': '4.2-4.3',
            'generated_at': datetime.now().isoformat(),
            'description': 'Walk-forward validation and stress test for alternative data overlay',
            'base_portfolio': 'SPY/GLD/TLT 46/38/16',
            'overlay': 'Alternative data regime signal (10% weight shift)',
            'confidence_threshold': 0.3,
        },
        'overall': {
            'baseline_sharpe': result.overall_baseline_sharpe,
            'overlay_sharpe': result.overall_overlay_sharpe,
            'sharpe_delta': result.overall_sharpe_delta,
            'target_met': result.target_met,
            'target_threshold': 0.03,
        },
        'walk_forward': {
            'avg_sharpe_delta': result.avg_sharpe_delta,
            'pct_windows_improved': result.pct_windows_improved,
            'windows': [
                {
                    'label': w.label,
                    'start': w.start_date,
                    'end': w.end_date,
                    'trading_days': w.trading_days,
                    'baseline': {'cagr': w.baseline_cagr, 'vol': w.baseline_vol, 'sharpe': w.baseline_sharpe, 'max_dd': w.baseline_max_dd},
                    'overlay': {'cagr': w.overlay_cagr, 'vol': w.overlay_vol, 'sharpe': w.overlay_sharpe, 'max_dd': w.overlay_max_dd},
                    'sharpe_delta': w.sharpe_delta,
                    'cagr_delta': w.cagr_delta,
                }
                for w in result.walk_forward_windows
            ],
        },
        'stress_tests': [
            {
                'period': s.period,
                'start': s.start_date,
                'end': s.end_date,
                'description': s.description,
                'baseline_return': s.baseline_return,
                'overlay_return': s.overlay_return,
                'baseline_max_dd': s.baseline_max_dd,
                'overlay_max_dd': s.overlay_max_dd,
                'signal_accuracy': s.signal_accuracy,
                'avg_confidence': s.avg_confidence,
            }
            for s in result.stress_tests
        ],
    }

    output_path = '/root/projects/portfolio-lab/data/signals/alt_data_walkforward_stress_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    run_full_backtest()
