#!/usr/bin/env python3
"""
Actual UBT/TMF Historical Backtest - Phase 1C Validation
v2.35 Capital Efficiency Strategy

Uses real historical UBT/TMF data to validate synthetic simulation accuracy.
"""

import json
import os
from datetime import datetime
from math import sqrt, pow
import numpy as np

def load_historical_data():
    """Load historical data from JSON"""
    data_path = '/root/projects/portfolio-lab/public/data/historical.json'
    with open(data_path, 'r') as f:
        return json.load(f)

def extract_prices(data, symbol):
    """Extract price series (using adjClose if available, else close)"""
    series = data.get(symbol, [])
    prices = []
    dates = []
    for entry in series:
        date = entry.get('date')
        price = entry.get('adjClose', entry.get('close', entry.get('p')))
        if date and price:
            prices.append(float(price))
            dates.append(date)
    return dates, prices

def calculate_returns(prices):
    """Calculate daily returns"""
    returns = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(r)
    return returns

def find_overlap(dates1, dates2):
    """Find overlapping date range"""
    set1 = set(dates1)
    set2 = set(dates2)
    overlap = sorted(set1 & set2)
    if not overlap:
        return None
    return overlap[0], overlap[-1], len(overlap)

def align_series(dates1, prices1, dates2, prices2):
    """Align two price series by date"""
    map1 = dict(zip(dates1, prices1))
    map2 = dict(zip(dates2, prices2))
    
    aligned_dates = []
    aligned_p1 = []
    aligned_p2 = []
    
    for d in dates1:
        if d in map2:
            aligned_dates.append(d)
            aligned_p1.append(map1[d])
            aligned_p2.append(map2[d])
    
    return aligned_dates, aligned_p1, aligned_p2

def calculate_metrics(returns, dates, scenario, base_returns=None, expected_multiple=1):
    """Calculate backtest metrics"""
    total_return = np.prod([1 + r for r in returns]) - 1
    years = len(returns) / 252
    cagr = pow(1 + total_return, 1/years) - 1 if years > 0 else 0
    
    mean_return = np.mean(returns)
    daily_vol = np.std(returns, ddof=1)
    annualized_vol = daily_vol * sqrt(252)
    
    # Max drawdown
    max_dd = 0
    peak = 1
    current = 1
    for r in returns:
        current *= (1 + r)
        if current > peak:
            peak = current
        dd = (current - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    # Sharpe (assuming 4% risk-free)
    sharpe = (cagr - 0.04) / annualized_vol if annualized_vol > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else cagr
    
    # Tracking error
    tracking_error = 0
    if base_returns and len(base_returns) == len(returns):
        differences = []
        for i in range(len(returns)):
            expected = base_returns[i] * expected_multiple
            diff = returns[i] - expected
            differences.append(diff)
        te = np.std(differences, ddof=1) * sqrt(252)
        tracking_error = te
    
    # Volatility decay estimate
    variance = daily_vol ** 2
    decay = -0.5 * (expected_multiple ** 2) * variance * 252 if expected_multiple > 1 else 0
    
    # Expense impact
    expense_ratio = 0.0080 if 'UBT' in scenario else 0.0091 if 'TMF' in scenario else 0.0015
    expense_impact = pow(1 - expense_ratio, years) - 1
    
    return {
        'scenario': scenario,
        'startDate': dates[0] if dates else '',
        'endDate': dates[-1] if dates else '',
        'days': len(returns),
        'cagr': cagr,
        'volatility': annualized_vol,
        'sharpe': sharpe,
        'maxDrawdown': max_dd,
        'calmar': calmar,
        'totalReturn': total_return,
        'trackingErrorVsTLT': tracking_error,
        'volatilityDecayEstimate': decay,
        'annualizedExpenseImpact': expense_impact
    }

def calculate_correlation(returns1, returns2):
    """Calculate correlation between two return series"""
    n = min(len(returns1), len(returns2))
    r1 = returns1[:n]
    r2 = returns2[:n]
    return np.corrcoef(r1, r2)[0, 1]

def main():
    print('[INFO] Loading historical data...')
    data = load_historical_data()
    
    symbols = list(data.keys())
    print(f'[INFO] Available symbols: {", ".join(symbols[:10])}...')
    
    has_ubt = 'UBT' in symbols
    has_tmf = 'TMF' in symbols
    has_tlt = 'TLT' in symbols
    
    print(f'[INFO] UBT available: {has_ubt}, TMF available: {has_tmf}, TLT available: {has_tlt}')
    
    if not has_tlt:
        print('[ERROR] TLT data required but not found')
        return
    
    # Extract data
    tlt_dates, tlt_prices = extract_prices(data, 'TLT')
    ubt_dates, ubt_prices = extract_prices(data, 'UBT') if has_ubt else (None, None)
    tmf_dates, tmf_prices = extract_prices(data, 'TMF') if has_tmf else (None, None)
    
    print(f'[INFO] TLT: {len(tlt_prices)} days')
    if ubt_dates:
        print(f'[INFO] UBT: {len(ubt_prices)} days ({ubt_dates[0]} to {ubt_dates[-1]})')
    if tmf_dates:
        print(f'[INFO] TMF: {len(tmf_prices)} days ({tmf_dates[0]} to {tmf_dates[-1]})')
    
    # Find overlaps
    results = []
    
    # Scenario 1: TLT baseline
    print('\n[INFO] Calculating TLT baseline...')
    tlt_returns = calculate_returns(tlt_prices)
    tlt_metrics = calculate_metrics(tlt_returns, tlt_dates[1:], 'Baseline_TLT')
    results.append(tlt_metrics)
    
    # Scenario 2: UBT actual
    if has_ubt and ubt_dates:
        print('[INFO] Calculating UBT actual returns...')
        overlap = find_overlap(tlt_dates, ubt_dates)
        if overlap:
            start, end, days = overlap
            print(f'[INFO] TLT-UBT overlap: {days} days ({start} to {end})')
            
            aligned_dates, aligned_tlt, aligned_ubt = align_series(
                tlt_dates, tlt_prices, ubt_dates, ubt_prices
            )
            
            tlt_overlap_returns = calculate_returns(aligned_tlt)
            ubt_returns = calculate_returns(aligned_ubt)
            
            ubt_metrics = calculate_metrics(
                ubt_returns, aligned_dates[1:], 'Actual_UBT', 
                tlt_overlap_returns, expected_multiple=2
            )
            results.append(ubt_metrics)
    
    # Scenario 3: TMF actual
    if has_tmf and tmf_dates:
        print('[INFO] Calculating TMF actual returns...')
        overlap = find_overlap(tlt_dates, tmf_dates)
        if overlap:
            start, end, days = overlap
            print(f'[INFO] TLT-TMF overlap: {days} days ({start} to {end})')
            
            aligned_dates, aligned_tlt, aligned_tmf = align_series(
                tlt_dates, tlt_prices, tmf_dates, tmf_prices
            )
            
            tlt_overlap_returns = calculate_returns(aligned_tlt)
            tmf_returns = calculate_returns(aligned_tmf)
            
            tmf_metrics = calculate_metrics(
                tmf_returns, aligned_dates[1:], 'Actual_TMF',
                tlt_overlap_returns, expected_multiple=3
            )
            results.append(tmf_metrics)
    
    # Calculate synthetic comparison
    print('\n[INFO] Calculating synthetic returns for comparison...')
    tlt_for_syn = tlt_returns[:min(len(tlt_returns), 2520)]  # 10 years
    synthetic_ubt = [r * 2 - 0.0080/252 for r in tlt_for_syn]
    synthetic_tmf = [r * 3 - 0.0091/252 for r in tlt_for_syn]
    
    ubt_result = next((r for r in results if r['scenario'] == 'Actual_UBT'), None)
    tmf_result = next((r for r in results if r['scenario'] == 'Actual_TMF'), None)
    tlt_result = next((r for r in results if r['scenario'] == 'Baseline_TLT'), None)
    
    # Calculate correlations
    ubt_corr = 0
    tmf_corr = 0
    if ubt_result and len(tlt_returns) >= ubt_result['days']:
        actual_ubt_returns = calculate_returns(ubt_prices)[:ubt_result['days']]
        tlt_slice = tlt_returns[:len(actual_ubt_returns)]
        ubt_corr = calculate_correlation(actual_ubt_returns, [r * 2 for r in tlt_slice])
    
    if tmf_result and len(tlt_returns) >= tmf_result['days']:
        actual_tmf_returns = calculate_returns(tmf_prices)[:tmf_result['days']]
        tlt_slice = tlt_returns[:len(actual_tmf_returns)]
        tmf_corr = calculate_correlation(actual_tmf_returns, [r * 3 for r in tlt_slice])
    
    # Determine recommendation
    proceed_to_paper = False
    recommended_scenario = 'Baseline_TLT'
    reasoning = ''
    confidence = 'low'
    
    if ubt_result and tlt_result:
        cagr_improvement = ubt_result['cagr'] - tlt_result['cagr']
        vol_increase = ubt_result['volatility'] - tlt_result['volatility']
        tracking_ok = ubt_result['trackingErrorVsTLT'] < 0.02
        
        if cagr_improvement > 0.005 and vol_increase < 0.15 and tracking_ok:
            proceed_to_paper = True
            recommended_scenario = 'Capital_Efficient_UBT'
            reasoning = f"Actual UBT delivers +{cagr_improvement*100:.1f}% CAGR vs TLT with {vol_increase*100:.1f}% vol increase and {ubt_result['trackingErrorVsTLT']*100:.2f}% tracking error. Meets criteria for paper trading."
            confidence = 'high'
        elif cagr_improvement > 0:
            recommended_scenario = 'Capital_Efficient_UBT'
            reasoning = f"UBT shows +{cagr_improvement*100:.1f}% CAGR improvement but vol increase of {vol_increase*100:.1f}% requires monitoring. Paper trading deferred."
            confidence = 'medium'
        else:
            reasoning = f"UBT underperforms TLT by {abs(cagr_improvement)*100:.1f}% CAGR. Strategy not validated."
            confidence = 'low'
    else:
        reasoning = 'Insufficient data to validate capital efficiency strategy.'
    
    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.35',
        'dataQuality': {
            'tltDays': len(tlt_prices),
            'ubtDays': len(ubt_prices) if ubt_prices else 0,
            'tmfDays': len(tmf_prices) if tmf_prices else 0,
            'dataStart': tlt_dates[0] if tlt_dates else '',
            'dataEnd': tlt_dates[-1] if tlt_dates else ''
        },
        'results': results,
        'syntheticVsActual': {
            'ubtCorrelation': ubt_corr,
            'tmfCorrelation': tmf_corr,
            'ubtAnnualizedTrackingError': ubt_result['trackingErrorVsTLT'] if ubt_result else 0,
            'tmfAnnualizedTrackingError': tmf_result['trackingErrorVsTLT'] if tmf_result else 0,
            'syntheticAccuracy': 'high' if ubt_corr > 0.95 else 'medium' if ubt_corr > 0.85 else 'low'
        },
        'recommendation': {
            'proceedToPaperTrading': proceed_to_paper,
            'recommendedScenario': recommended_scenario,
            'reasoning': reasoning,
            'confidence': confidence
        }
    }
    
    # Save report
    output_path = '/root/projects/portfolio-lab/data/ubt_actual_validation.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'\n[SUCCESS] Validation complete! Results saved to: {output_path}')
    print('\n=== VALIDATION SUMMARY ===')
    print(f"Data Quality:")
    print(f"  TLT: {report['dataQuality']['tltDays']} days")
    print(f"  UBT: {report['dataQuality']['ubtDays']} days")
    print(f"  TMF: {report['dataQuality']['tmfDays']} days")
    print(f"\nSynthetic vs Actual Accuracy: {report['syntheticVsActual']['syntheticAccuracy'].upper()}")
    print(f"  UBT Correlation: {report['syntheticVsActual']['ubtCorrelation']*100:.1f}%")
    print(f"  UBT Tracking Error: {report['syntheticVsActual']['ubtAnnualizedTrackingError']*100:.2f}%")
    print(f"\n=== BACKTEST RESULTS ===")
    for r in results:
        print(f"{r['scenario']}:")
        print(f"  Period: {r['startDate']} to {r['endDate']} ({r['days']} days)")
        print(f"  CAGR: {r['cagr']*100:.2f}%")
        print(f"  Volatility: {r['volatility']*100:.1f}%")
        print(f"  Sharpe: {r['sharpe']:.2f}")
        print(f"  Max DD: {r['maxDrawdown']*100:.1f}%")
        if r['trackingErrorVsTLT'] != 0:
            print(f"  Tracking Error vs TLT: {r['trackingErrorVsTLT']*100:.2f}%")
        print(f"  Est. Vol Decay: {r['volatilityDecayEstimate']*100:.2f}%")
    print(f"\n=== RECOMMENDATION ===")
    print(f"Proceed to Paper Trading: {'YES' if proceed_to_paper else 'NO'}")
    print(f"Recommended Scenario: {recommended_scenario}")
    print(f"Confidence: {confidence.upper()}")
    print(f"Reasoning: {reasoning}")

if __name__ == '__main__':
    main()
