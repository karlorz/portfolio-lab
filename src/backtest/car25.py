"""
CAR25 Performance Metric - Bandy's risk-normalized objective function

CAR25 = Compound Annual Rate of Return at the 25th percentile
after position-sizing via safe-f (max drawdown-constrained).

Two-stage process:
  1. Safe-f: Binary search for position size where 95th %ile max DD = tolerance
  2. CAR25: Monte Carlo at safe-f, extract 25th percentile CAGR

Companion metric: Correlation to SPY benchmark
"""

import json
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Default simulation parameters
DEFAULT_SIMULATIONS = 1000
DEFAULT_HORIZON_YEARS = 2
DEFAULT_RISK_TOLERANCE = 0.20  # 20% max DD
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BLOCK_SIZE = 20  # ~1 month blocks for autocorrelation
TRADING_DAYS_PER_YEAR = 252
MAX_ITERATIONS = 20
F_TOLERANCE = 0.005  # 0.5% convergence threshold


@dataclass
class SafeFResult:
    safe_f: float  # Position size fraction (0.01 to 4.0)
    drawdown95: float  # Actual 95th %ile max DD achieved
    iterations: int
    converged: bool
    tolerance_used: float


@dataclass
class CAR25Result:
    car25: float  # 25th percentile annualized return
    car50: float  # Median annualized return
    car75: float  # 75th percentile (optimistic)
    twr25: float  # 25th percentile terminal wealth ratio
    twr50: float
    twr75: float
    safe_f: float
    final_equity25: float  # 25th percentile final portfolio value
    final_equity50: float
    final_equity75: float


@dataclass
class MarketCorrelationResult:
    correlation: float  # Pearson's ρ (-1 to 1)
    classification: str  # 'low', 'moderate', 'high'
    common_days: int


@dataclass
class CAR25FullResult:
    portfolio: str
    safe_f: SafeFResult
    car25: CAR25Result
    correlation: MarketCorrelationResult
    config: Dict
    input_days: int


def block_bootstrap_returns(
    daily_returns: np.ndarray,
    num_days: int,
    block_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Block bootstrap: resample daily returns in blocks to preserve autocorrelation.
    """
    n = len(daily_returns)
    
    # Handle edge case: not enough data for requested block size
    effective_block_size = min(block_size, n)
    
    num_blocks = int(np.ceil(num_days / effective_block_size))
    
    result = []
    for _ in range(num_blocks):
        # Pick random starting point for block
        # Ensure valid range for random selection
        max_start = max(1, n - effective_block_size + 1)
        start_idx = rng.integers(0, max_start)
        block = daily_returns[start_idx:start_idx + effective_block_size]
        result.extend(block)
    
    return np.array(result[:num_days])


def simulate_equity_curve(
    daily_returns: np.ndarray,
    position_size: float,
    initial_equity: float = 1.0
) -> np.ndarray:
    """
    Simulate equity curve with given position size.
    """
    equity = np.zeros(len(daily_returns) + 1)
    equity[0] = initial_equity
    
    for i, ret in enumerate(daily_returns):
        # Position-sized return
        sized_return = ret * position_size
        equity[i + 1] = equity[i] * (1 + sized_return)
    
    return equity


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)


def safe_f(
    daily_returns: np.ndarray,
    risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
    n_sims: int = DEFAULT_SIMULATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: Optional[int] = None
) -> SafeFResult:
    """
    Binary search for position size where 95th %ile max DD = tolerance.
    
    Args:
        daily_returns: Array of daily returns
        risk_tolerance: Maximum acceptable drawdown (e.g., 0.20 = 20%)
        horizon_years: Simulation horizon in years
        n_sims: Number of Monte Carlo simulations
        confidence: Confidence level for drawdown (e.g., 0.95 = 95th percentile)
        block_size: Block size for bootstrap resampling
        seed: Random seed for reproducibility
    
    Returns:
        SafeFResult with converged position size
    """
    rng = np.random.default_rng(seed)
    num_days = int(horizon_years * TRADING_DAYS_PER_YEAR)
    
    # Binary search bounds
    f_low = 0.01
    f_high = 4.0
    
    best_f = f_low
    best_dd = 0.0
    converged = False
    iteration = 0
    
    for iteration in range(MAX_ITERATIONS):
        f_mid = (f_low + f_high) / 2.0
        
        # Run Monte Carlo simulations
        max_drawdowns = []
        for _ in range(n_sims):
            bootstrapped = block_bootstrap_returns(daily_returns, num_days, block_size, rng)
            equity = simulate_equity_curve(bootstrapped, f_mid)
            max_dd = calculate_max_drawdown(equity)
            max_drawdowns.append(max_dd)
        
        # Get confidence level drawdown (e.g., 95th percentile)
        dd_at_confidence = np.percentile(max_drawdowns, confidence * 100)
        
        # Store best result
        if abs(dd_at_confidence + risk_tolerance) < abs(best_dd + risk_tolerance):
            best_f = f_mid
            best_dd = dd_at_confidence
        
        # Check convergence
        if abs(abs(dd_at_confidence) - risk_tolerance) < F_TOLERANCE:
            converged = True
            best_f = f_mid
            best_dd = dd_at_confidence
            break
        
        # Adjust search bounds
        if abs(dd_at_confidence) > risk_tolerance:
            # Drawdown too high, reduce position size
            f_high = f_mid
        else:
            # Drawdown below tolerance, can increase position size
            f_low = f_mid
    
    return SafeFResult(
        safe_f=best_f,
        drawdown95=abs(best_dd),
        iterations=iteration + 1,
        converged=converged,
        tolerance_used=risk_tolerance
    )


def car25(
    daily_returns: np.ndarray,
    safe_f_value: float,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
    n_sims: int = DEFAULT_SIMULATIONS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: Optional[int] = None,
    initial_equity: float = 100000.0
) -> CAR25Result:
    """
    Monte Carlo at safe-f, extract 25th percentile CAGR.
    
    Args:
        daily_returns: Array of daily returns
        safe_f_value: Position size from safe_f calculation
        horizon_years: Simulation horizon in years
        n_sims: Number of Monte Carlo simulations
        block_size: Block size for bootstrap resampling
        seed: Random seed for reproducibility
        initial_equity: Starting portfolio value
    
    Returns:
        CAR25Result with percentile returns
    """
    rng = np.random.default_rng(seed)
    num_days = int(horizon_years * TRADING_DAYS_PER_YEAR)
    
    # Run Monte Carlo simulations
    final_equities = []
    twrs = []
    
    for _ in range(n_sims):
        bootstrapped = block_bootstrap_returns(daily_returns, num_days, block_size, rng)
        equity = simulate_equity_curve(bootstrapped, safe_f_value, initial_equity)
        final_equity = equity[-1]
        twr = final_equity / initial_equity
        
        final_equities.append(final_equity)
        twrs.append(twr)
    
    # Calculate percentiles
    twr25 = np.percentile(twrs, 25)
    twr50 = np.percentile(twrs, 50)
    twr75 = np.percentile(twrs, 75)
    
    final_equity25 = np.percentile(final_equities, 25)
    final_equity50 = np.percentile(final_equities, 50)
    final_equity75 = np.percentile(final_equities, 75)
    
    # Annualize: CAR = TWR^(1/years) - 1
    car25_value = twr25 ** (1.0 / horizon_years) - 1.0
    car50_value = twr50 ** (1.0 / horizon_years) - 1.0
    car75_value = twr75 ** (1.0 / horizon_years) - 1.0
    
    return CAR25Result(
        car25=car25_value,
        car50=car50_value,
        car75=car75_value,
        twr25=twr25,
        twr50=twr50,
        twr75=twr75,
        safe_f=safe_f_value,
        final_equity25=final_equity25,
        final_equity50=final_equity50,
        final_equity75=final_equity75
    )


def market_correlation(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> MarketCorrelationResult:
    """
    Calculate Pearson correlation between portfolio and benchmark.
    
    Args:
        portfolio_returns: Array of portfolio daily returns
        benchmark_returns: Array of benchmark daily returns
    
    Returns:
        MarketCorrelationResult with correlation metrics
    """
    # Ensure same length
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    p_ret = portfolio_returns[:min_len]
    b_ret = benchmark_returns[:min_len]
    
    # Calculate Pearson correlation
    if len(p_ret) < 2:
        return MarketCorrelationResult(
            correlation=0.0,
            classification='low',
            common_days=len(p_ret)
        )
    
    correlation = np.corrcoef(p_ret, b_ret)[0, 1]
    
    # Handle NaN
    if np.isnan(correlation):
        correlation = 0.0
    
    # Classify
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        classification = 'low'
    elif abs_corr < 0.7:
        classification = 'moderate'
    else:
        classification = 'high'
    
    return MarketCorrelationResult(
        correlation=correlation,
        classification=classification,
        common_days=len(p_ret)
    )


def load_prices_data(data_path: Optional[str] = None) -> Dict:
    """
    Load prices.json data from the data directory.
    Format: {symbol: [{d: date, p: price}, ...], ...}
    """
    if data_path is None:
        # Default to project data directory
        data_file = Path(__file__).parent.parent.parent / 'data' / 'prices.json'
    else:
        data_file = Path(data_path)
    
    with open(str(data_file), 'r') as f:
        return json.load(f)


def calculate_portfolio_returns(
    prices_data: Dict,
    weights: Dict[str, float]
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate daily returns for a weighted portfolio.
    
    Args:
        prices_data: Dict with symbol keys, each containing list of {d, p} dicts
        weights: Dict mapping symbol to weight (should sum to ~1.0)
    
    Returns:
        Tuple of (daily_returns_array, dates_list)
    """
    # Get dates from first symbol's data
    first_symbol = next(iter(weights.keys()))
    if first_symbol not in prices_data:
        raise ValueError(f"Symbol {first_symbol} not found in price data")
    
    dates = [entry['d'] for entry in prices_data[first_symbol]]
    
    # Calculate returns for each symbol
    symbol_returns = {}
    for symbol, weight in weights.items():
        if weight == 0:
            continue
        
        if symbol not in prices_data:
            raise ValueError(f"Symbol {symbol} not found in price data")
        
        symbol_prices = np.array([entry['p'] for entry in prices_data[symbol]])
        daily_rets = np.diff(symbol_prices) / symbol_prices[:-1]
        symbol_returns[symbol] = daily_rets
    
    # Calculate weighted portfolio returns
    # All symbols should have same length
    n_days = len(dates) - 1  # One less return than prices
    portfolio_returns = np.zeros(n_days)
    
    for symbol, weight in weights.items():
        if symbol in symbol_returns:
            # Ensure matching length
            n_rets = min(len(symbol_returns[symbol]), n_days)
            portfolio_returns[:n_rets] += symbol_returns[symbol][:n_rets] * weight
    
    return portfolio_returns, dates[1:]


def parse_portfolio_string(portfolio_str: str) -> Dict[str, float]:
    """
    Parse portfolio string like 'SPY/GLD/TLT 46/38/16' into weights dict.
    Also handles single-asset portfolios like 'SPY' or 'SPY 100'.
    """
    parts = portfolio_str.split()
    
    # Handle single-asset without weight (implicit 100%)
    if len(parts) == 1:
        return {parts[0]: 1.0}
    
    if len(parts) != 2:
        raise ValueError(f"Invalid portfolio string: {portfolio_str}")
    
    symbols = parts[0].split('/')
    weights_str = parts[1].split('/')
    
    # Handle case where single asset has explicit weight like 'SPY 100'
    if len(symbols) == 1 and len(weights_str) == 1:
        weight = float(weights_str[0])
        if weight > 1.0:  # Assume percentage
            weight = weight / 100.0
        return {symbols[0]: weight}
    
    if len(symbols) != len(weights_str):
        raise ValueError(f"Symbol count mismatch: {portfolio_str}")
    
    weights = [float(w) / 100.0 for w in weights_str]
    
    return dict(zip(symbols, weights))


def compute_car25_for_portfolio(
    portfolio_str: str,
    prices_data: Optional[Dict] = None,
    risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
    horizon_years: float = DEFAULT_HORIZON_YEARS,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: Optional[int] = 42
) -> CAR25FullResult:
    """
    Compute CAR25 for a named portfolio.
    
    Args:
        portfolio_str: Portfolio string like 'SPY/GLD/TLT 46/38/16' or 'SPY'
        prices_data: Optional pre-loaded prices data
        risk_tolerance: Maximum acceptable drawdown
        horizon_years: Simulation horizon
        n_sims: Number of simulations
        seed: Random seed
    
    Returns:
        CAR25FullResult with all metrics
    """
    if prices_data is None:
        prices_data = load_prices_data()
    
    # Parse portfolio
    weights = parse_portfolio_string(portfolio_str)
    
    # Calculate portfolio returns
    portfolio_returns, _ = calculate_portfolio_returns(prices_data, weights)
    
    # Get SPY benchmark returns
    spy_weights = {'SPY': 1.0}
    spy_returns, _ = calculate_portfolio_returns(prices_data, spy_weights)
    
    # Stage 1: Safe-f calculation
    safe_f_result = safe_f(
        portfolio_returns,
        risk_tolerance=risk_tolerance,
        horizon_years=horizon_years,
        n_sims=n_sims,
        seed=seed
    )
    
    # Stage 2: CAR25 calculation
    car25_result = car25(
        portfolio_returns,
        safe_f_result.safe_f,
        horizon_years=horizon_years,
        n_sims=n_sims,
        seed=seed
    )
    
    # Correlation
    correlation_result = market_correlation(portfolio_returns, spy_returns)
    
    return CAR25FullResult(
        portfolio=portfolio_str,
        safe_f=safe_f_result,
        car25=car25_result,
        correlation=correlation_result,
        config={
            'simulations': n_sims,
            'horizon_years': horizon_years,
            'risk_tolerance': risk_tolerance,
            'confidence': DEFAULT_CONFIDENCE,
            'block_size': DEFAULT_BLOCK_SIZE
        },
        input_days=len(portfolio_returns)
    )


def print_car25_result(result: CAR25FullResult, json_output: bool = False):
    """
    Print CAR25 result in human-readable or JSON format.
    """
    if json_output:
        output = {
            'portfolio': result.portfolio,
            'safe_f': {
                'safe_f': round(result.safe_f.safe_f, 4),
                'drawdown95': round(result.safe_f.drawdown95, 4),
                'iterations': result.safe_f.iterations,
                'converged': result.safe_f.converged,
                'tolerance_used': result.safe_f.tolerance_used
            },
            'car25': {
                'car25': round(result.car25.car25 * 100, 2),
                'car50': round(result.car25.car50 * 100, 2),
                'car75': round(result.car25.car75 * 100, 2),
                'twr25': round(result.car25.twr25, 4),
                'twr50': round(result.car25.twr50, 4),
                'safe_f': round(result.car25.safe_f, 4)
            },
            'correlation': {
                'correlation': round(result.correlation.correlation, 4),
                'classification': result.correlation.classification,
                'common_days': result.correlation.common_days
            },
            'input_days': result.input_days
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"CAR25 Analysis: {result.portfolio}")
        print(f"{'='*60}")
        print(f"\nStage 1: Safe-f (Risk Normalization)")
        print(f"  Position Size (safe-f): {result.safe_f.safe_f:.4f}")
        print(f"  95th %ile Max Drawdown: {result.safe_f.drawdown95*100:.2f}%")
        print(f"  Iterations: {result.safe_f.iterations}")
        print(f"  Converged: {result.safe_f.converged}")
        
        print(f"\nStage 2: CAR25 (Profit Estimation)")
        print(f"  CAR25 (25th percentile): {result.car25.car25*100:.2f}%")
        print(f"  CAR50 (median):          {result.car25.car50*100:.2f}%")
        print(f"  CAR75 (75th percentile): {result.car25.car75*100:.2f}%")
        print(f"  TWR25: {result.car25.twr25:.4f}")
        
        print(f"\nMarket Correlation (SPY)")
        print(f"  Correlation: {result.correlation.correlation:.4f}")
        print(f"  Classification: {result.correlation.classification}")
        print(f"  Common Days: {result.correlation.common_days}")
        
        print(f"\nConfig")
        print(f"  Simulations: {result.config['simulations']}")
        print(f"  Horizon: {result.config['horizon_years']} years")
        print(f"  Risk Tolerance: {result.config['risk_tolerance']*100:.0f}%")
        print(f"  Input Days: {result.input_days}")
        print(f"{'='*60}\n")


def main():
    """
    CLI entry point for CAR25 calculations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CAR25 Performance Metric Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.backtest.car25 --portfolio "SPY/GLD/TLT 46/38/16"
  python -m src.backtest.car25 --portfolio "SPY" --json
  python -m src.backtest.car25 --compare-all --json > car25_results.json
        """
    )
    
    parser.add_argument(
        '--portfolio',
        type=str,
        help='Portfolio string like "SPY/GLD/TLT 46/38/16" or "SPY"'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=DEFAULT_RISK_TOLERANCE,
        help=f'Risk tolerance (max drawdown), default {DEFAULT_RISK_TOLERANCE}'
    )
    
    parser.add_argument(
        '--horizon',
        type=float,
        default=DEFAULT_HORIZON_YEARS,
        help=f'Simulation horizon in years, default {DEFAULT_HORIZON_YEARS}'
    )
    
    parser.add_argument(
        '--sims',
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f'Number of simulations, default {DEFAULT_SIMULATIONS}'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare CAR25 for all standard portfolios'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to prices.json file'
    )
    
    args = parser.parse_args()
    
    # Define standard portfolios
    standard_portfolios = [
        'SPY',
        'QQQ',
        'SPY/GLD 55/45',
        'SPY/GLD/TLT 46/38/16',
        'SPY/GLD/TLT 50/35/15',
        'SPY/GLD/TLT 48/32/20',
        'SPY/GLD/IEF 50/35/15',
    ]
    
    if args.compare_all:
        # Compare all standard portfolios
        prices_data = load_prices_data(args.data_path)
        results = []
        
        for portfolio_str in standard_portfolios:
            try:
                result = compute_car25_for_portfolio(
                    portfolio_str,
                    prices_data=prices_data,
                    risk_tolerance=args.tolerance,
                    horizon_years=args.horizon,
                    n_sims=args.sims
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {portfolio_str}: {e}", file=__import__('sys').stderr)
        
        # Sort by CAR25 descending
        results.sort(key=lambda r: r.car25.car25, reverse=True)
        
        if args.json:
            output = []
            for r in results:
                output.append({
                    'portfolio': r.portfolio,
                    'car25': round(r.car25.car25 * 100, 2),
                    'car50': round(r.car25.car50 * 100, 2),
                    'safe_f': round(r.safe_f.safe_f, 4),
                    'correlation': round(r.correlation.correlation, 4),
                    'converged': r.safe_f.converged
                })
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*80}")
            print(f"CAR25 Comparison - All Portfolios (Tolerance: {args.tolerance*100:.0f}%)")
            print(f"{'='*80}")
            print(f"{'Portfolio':<25} {'CAR25':>8} {'CAR50':>8} {'Safe-f':>8} {'Corr':>8} {'Status':>10}")
            print(f"{'-'*80}")
            for r in results:
                status = "✓" if r.safe_f.converged else "~"
                print(f"{r.portfolio:<25} {r.car25.car25*100:>7.2f}% {r.car25.car50*100:>7.2f}% {r.safe_f.safe_f:>7.3f} {r.correlation.correlation:>7.3f} {status:>10}")
            print(f"{'='*80}\n")
    
    elif args.portfolio:
        # Single portfolio analysis
        prices_data = load_prices_data(args.data_path) if args.data_path else None
        result = compute_car25_for_portfolio(
            args.portfolio,
            prices_data=prices_data,
            risk_tolerance=args.tolerance,
            horizon_years=args.horizon,
            n_sims=args.sims
        )
        print_car25_result(result, json_output=args.json)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
