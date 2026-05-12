"""
Factor Correlation Analysis for v2.43 Dynamic Factor Timing
Phase 1.2: Calculate correlations between factor ETFs and base portfolio
"""
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_prices():
    with open('public/data/prices.json', 'r') as f:
        return json.load(f)

def align_series(symbols_data, start_date='2013-07-18'):
    """Align price series to common date range"""
    # Get all dates for each symbol
    symbol_dates = {}
    for sym, data in symbols_data.items():
        symbol_dates[sym] = {entry['d']: entry['p'] for entry in data}
    
    # Find common dates
    all_dates = set()
    for dates in symbol_dates.values():
        all_dates.update(dates.keys())
    
    # Filter to start_date and sort
    common_dates = sorted([d for d in all_dates if d >= start_date])
    
    # Build aligned matrix
    aligned = {}
    for sym, date_prices in symbol_dates.items():
        series = []
        for d in common_dates:
            if d in date_prices:
                series.append(date_prices[d])
            else:
                series.append(None)
        aligned[sym] = series
    
    return aligned, common_dates

def calculate_returns(prices):
    """Calculate daily log returns"""
    returns = []
    for i in range(1, len(prices)):
        if prices[i] is not None and prices[i-1] is not None and prices[i-1] > 0:
            ret = np.log(prices[i] / prices[i-1])
            returns.append(ret)
        else:
            returns.append(0.0)
    return returns

def correlation_matrix(symbols, returns_dict):
    """Calculate correlation matrix"""
    corr = {}
    for s1 in symbols:
        corr[s1] = {}
        for s2 in symbols:
            if s1 in returns_dict and s2 in returns_dict:
                r1 = np.array(returns_dict[s1])
                r2 = np.array(returns_dict[s2])
                if len(r1) == len(r2) and len(r1) > 1:
                    corr[s1][s2] = np.corrcoef(r1, r2)[0, 1]
                else:
                    corr[s1][s2] = 0.0
            else:
                corr[s1][s2] = 0.0
    return corr

def analyze_factor_regime_sensitivity(prices, signals_path='public/data/signals.json'):
    """Analyze factor performance by regime"""
    try:
        with open(signals_path, 'r') as f:
            signals = json.load(f)
        regime = signals.get('regime', 'unknown')
    except:
        regime = 'unknown'
    
    return regime

def main():
    print("=" * 70)
    print("Factor Correlation Analysis - v2.43 Phase 1.2")
    print("=" * 70)
    
    # Load data
    prices = load_prices()
    
    # Base portfolio components
    base = ['SPY', 'GLD', 'TLT']
    
    # Factor ETFs (all 5)
    factors = ['MTUM', 'VLUE', 'USMV', 'QUAL', 'IJR']
    
    # Analysis period (QUAL has earliest launch among factor ETFs: 2013-07-18)
    analysis_start = '2013-07-18'
    
    print(f"\nAnalysis period: {analysis_start} to present")
    print(f"Base portfolio: {base}")
    print(f"Factor ETFs: {factors}")
    
    # Align series
    aligned, dates = align_series(prices, analysis_start)
    print(f"\nCommon trading days: {len(dates)}")
    
    # Calculate returns
    returns = {}
    for sym in base + factors:
        if sym in aligned:
            returns[sym] = calculate_returns(aligned[sym])
    
    # Calculate correlations with base portfolio
    print("\n" + "=" * 70)
    print("Factor Correlations with Base Portfolio Components")
    print("=" * 70)
    
    for factor in factors:
        if factor not in returns:
            continue
        print(f"\n{factor}:")
        for base_sym in base:
            if base_sym in returns:
                corr = np.corrcoef(returns[factor], returns[base_sym])[0, 1]
                print(f"  vs {base_sym}: {corr:+.3f}")
    
    # Full correlation matrix
    all_symbols = base + factors
    corr_matrix = correlation_matrix(all_symbols, returns)
    
    print("\n" + "=" * 70)
    print("Full Correlation Matrix (Factors + Base Portfolio)")
    print("=" * 70)
    
    # Print header
    header = "      " + "".join([f"{s:>8}" for s in all_symbols])
    print(header)
    
    for s1 in all_symbols:
        row = f"{s1:>6}"
        for s2 in all_symbols:
            if s2 in corr_matrix.get(s1, {}):
                row += f" {corr_matrix[s1][s2]:+7.2f}"
            else:
                row += "    N/A"
        print(row)
    
    # Factor-factor correlations
    print("\n" + "=" * 70)
    print("Factor-Factor Cross-Correlations (Style Diversification)")
    print("=" * 70)
    
    factor_pairs = [
        ('MTUM', 'VLUE'),   # Momentum vs Value
        ('MTUM', 'USMV'),   # Momentum vs Low Vol
        ('MTUM', 'QUAL'),   # Momentum vs Quality
        ('VLUE', 'QUAL'),   # Value vs Quality
        ('VLUE', 'IJR'),    # Value vs Small
        ('USMV', 'QUAL'),   # Low Vol vs Quality
        ('USMV', 'IJR'),    # Low Vol vs Small
        ('QUAL', 'IJR'),    # Quality vs Small
    ]
    
    for f1, f2 in factor_pairs:
        if f1 in returns and f2 in returns:
            corr = np.corrcoef(returns[f1], returns[f2])[0, 1]
            print(f"{f1:5} vs {f2:5}: {corr:+.3f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_period': {'start': analysis_start, 'days': len(dates)},
        'correlation_matrix': corr_matrix,
        'factor_style_pairs': {f"{f1}-{f2}": corr_matrix[f1][f2] for f1, f2 in factor_pairs if f1 in corr_matrix and f2 in corr_matrix[f1]},
        'base_vs_factor': {}
    }
    
    for factor in factors:
        if factor in returns:
            results['base_vs_factor'][factor] = {
                base_sym: corr_matrix[factor][base_sym] for base_sym in base if base_sym in corr_matrix.get(factor, {})
            }
    
    with open('data/factor_correlation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to data/factor_correlation_analysis.json")
    
    # Regime detection
    regime = analyze_factor_regime_sensitivity(prices)
    print(f"\nCurrent regime signal: {regime}")
    
    # Summary insights
    print("\n" + "=" * 70)
    print("Key Insights for Factor Timing Overlay")
    print("=" * 70)
    
    # Calculate average factor correlation to SPY
    spy_corrs = []
    for factor in factors:
        if factor in corr_matrix and 'SPY' in corr_matrix[factor]:
            spy_corrs.append((factor, corr_matrix[factor]['SPY']))
    
    spy_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nFactor-Equity Correlation Rankings (to SPY):")
    for factor, corr in spy_corrs:
        diversification = "High" if abs(corr) < 0.7 else "Medium" if abs(corr) < 0.85 else "Low"
        print(f"  {factor}: {corr:+.3f} ({diversification} diversification)")
    
    print("\nFactor Timing Implications:")
    print("  - Low correlation factors offer better timing opportunities")
    print("  - MTUM typically has highest SPY correlation (momentum herding)")
    print("  - VLUE, USMV may provide better defensive timing signals")
    print("  - QUAL and IJR offer intermediate diversification")

if __name__ == '__main__':
    main()
