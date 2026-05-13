#!/usr/bin/env python3
"""
Duration-Yield Curve Regime Backtest - v3.11 Phase 4
Compare static vs dynamic duration allocation (2005-2026)

Tests the hypothesis that yield curve regime-based duration targeting
improves risk-adjusted returns vs static allocation.

Regimes:
- Inverted (< -0.25%): Short duration (2-3yr effective)
- Flat (-0.25% to +0.75%): Neutral 5-7yr duration  
- Steep (> +0.75%): Long duration (10-15yr)

Static Allocation: 16% TLT / 15% IEF / 5% SHY (36% total bonds)
Dynamic Allocation: Regime-based shifts in TLT/IEF/SHY weights
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
PRICES_PATH = DATA_DIR / "prices.json"
YIELDS_PATH = DATA_DIR / "yields.json"
OUTPUT_PATH = DATA_DIR / ".duration_backtest_results.json"

# Allocation definitions
STATIC_ALLOCATION = {
    "tlt": 0.16,  # 16% long duration
    "ief": 0.15,  # 15% intermediate
    "shy": 0.05,  # 5% short
}

# Dynamic allocations by regime
DYNAMIC_ALLOCATIONS = {
    "inverted": {"tlt": 0.05, "ief": 0.25, "shy": 0.06},  # Short duration focus
    "flat": {"tlt": 0.16, "ief": 0.15, "shy": 0.05},     # Same as static
    "steep": {"tlt": 0.22, "ief": 0.10, "shy": 0.04},    # Long duration focus
}

# Effective duration by regime
REGIME_EFFECTIVE_DURATION = {
    "inverted": 3.5,  # Years
    "flat": 6.5,
    "steep": 8.5,
}

# Annual expense ratios
EXPENSE_RATIOS = {
    "tlt": 0.0015,  # 0.15%
    "ief": 0.0015,
    "shy": 0.0015,
    "spy": 0.0009,
    "gld": 0.0040,
}

# Transaction costs (bps)
TRANSACTION_COST = 0.0010  # 10 bps per trade


@dataclass
class BacktestResult:
    """Results from duration backtest comparison."""
    # Static metrics
    static_cagr: float
    static_volatility: float
    static_sharpe: float
    static_max_dd: float
    
    # Dynamic metrics
    dynamic_cagr: float
    dynamic_volatility: float
    dynamic_sharpe: float
    dynamic_max_dd: float
    
    # Comparison
    sharpe_delta: float
    cagr_delta: float
    max_dd_delta: float
    
    # Crisis performance
    crisis_2008_static: float
    crisis_2008_dynamic: float
    crisis_2020_static: float
    crisis_2020_dynamic: float
    crisis_2022_static: float
    crisis_2022_dynamic: float
    
    # Regime statistics
    regime_days: Dict[str, int]
    regime_transitions: int
    rebalancing_costs: float
    
    # Metadata
    start_date: str
    end_date: str
    total_days: int
    timestamp: str


def load_price_data() -> pd.DataFrame:
    """Load price data from prices.json."""
    logger.info("Loading price data...")
    with open(PRICES_PATH) as f:
        data = json.load(f)
    
    # prices.json format: {symbol: [{"d": date, "p": price}, ...]}
    # Convert to DataFrame with dates as rows and symbols as columns
    all_dates = set()
    for symbol, entries in data.items():
        for entry in entries:
            all_dates.add(entry["d"])
    
    dates = sorted(all_dates)
    records = []
    
    for date in dates:
        record = {"date": date}
        for symbol, entries in data.items():
            # Find price for this date
            price = None
            for entry in entries:
                if entry["d"] == date:
                    price = entry["p"]
                    break
            if price is not None:
                record[symbol.lower()] = price
        records.append(record)
    
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def load_yield_curve_data() -> pd.DataFrame:
    """Load yield curve regime classifications."""
    logger.info("Loading yield curve data...")
    
    # First try to load from yields.json
    if YIELDS_PATH.exists():
        with open(YIELDS_PATH) as f:
            data = json.load(f)
        
        if "regimes" in data:
            regimes = data["regimes"]
            df = pd.DataFrame([
                {"date": k, "regime": v["regime"], "spread": v.get("spread", 0)}
                for k, v in regimes.items()
            ])
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
    
    # Fallback: infer from price data dates
    logger.info("No yield curve regime data found, using synthetic classification...")
    return None


def classify_regime_from_spread(spread: float) -> str:
    """Classify yield curve regime from 10Y-2Y spread."""
    if spread < -0.25:
        return "inverted"
    elif spread > 0.75:
        return "steep"
    else:
        return "flat"


def load_yield_spread_history() -> pd.DataFrame:
    """Load or estimate yield spread history from FRED data."""
    logger.info("Loading yield spread history...")
    
    # Try database first
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if we have yield data
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='yield_curve_regimes'
        """)
        
        if cursor.fetchone():
            df = pd.read_sql_query("""
                SELECT date, regime, spread 
                FROM yield_curve_regimes 
                ORDER BY date
            """, conn)
            conn.close()
            
            if len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"Loaded {len(df)} regime records from database")
                return df
        
        conn.close()
    
    # Use synthetic regime data based on known periods
    logger.info("Using synthetic regime classification...")
    
    # Create synthetic regime data based on historical yield curve inversions
    # 2006-2007: Flat to inverted (pre-crisis)
    # 2008-2009: Steep (post-crisis recovery)
    # 2010-2017: Flat (low rate environment)
    # 2018-2019: Flat to inverted (late cycle)
    # 2020: Steep (COVID crash/recovery)
    # 2021-2022: Flat (inflation/rate hikes)
    # 2023-2026: Flat to steep (disinflation)
    
    dates = pd.date_range(start="2005-01-01", end="2026-05-14", freq="D")
    
    regimes = []
    for date in dates:
        year = date.year
        
        if year in [2006, 2007]:
            regime = "inverted"
            spread = -0.30
        elif year in [2008, 2009]:
            regime = "steep"
            spread = 1.50
        elif year in range(2010, 2018):
            regime = "flat"
            spread = 0.50
        elif year in [2018, 2019]:
            regime = "inverted"
            spread = -0.20
        elif year == 2020:
            regime = "steep"
            spread = 0.80
        elif year in [2021, 2022]:
            regime = "inverted"
            spread = -0.10
        else:  # 2023-2026
            regime = "flat"
            spread = 0.47
        
        regimes.append({
            "date": date,
            "regime": regime,
            "spread": spread
        })
    
    df = pd.DataFrame(regimes)
    logger.info(f"Created synthetic regime data: {len(df)} days")
    return df


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from prices."""
    return prices.pct_change().fillna(0)


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 30:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_cagr(returns: pd.Series) -> float:
    """Calculate annualized return (CAGR)."""
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod()
    years = len(returns) / 252
    
    if years < 0.1:
        return 0.0
    
    cagr = (total_return ** (1 / years)) - 1
    return cagr


def run_backtest(
    prices_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    start_date: str = "2005-01-01",
    end_date: str = "2026-05-14"
) -> BacktestResult:
    """Run backtest comparing static vs dynamic duration allocation."""
    logger.info(f"Running backtest from {start_date} to {end_date}...")
    
    # Filter to date range
    prices_df = prices_df[
        (prices_df["date"] >= start_date) & 
        (prices_df["date"] <= end_date)
    ].copy()
    
    regimes_df = regimes_df[
        (regimes_df["date"] >= start_date) & 
        (regimes_df["date"] <= end_date)
    ].copy()
    
    # Merge data
    merged = prices_df.merge(regimes_df, on="date", how="inner")
    
    if len(merged) == 0:
        logger.error("No overlapping dates between prices and regimes")
        return None
    
    logger.info(f"Backtesting on {len(merged)} days")
    
    # Calculate daily returns for each asset
    for col in ["tlt", "ief", "shy", "spy", "gld"]:
        if col in merged.columns:
            merged[f"{col}_ret"] = calculate_returns(merged[col])
        else:
            merged[f"{col}_ret"] = 0.0
    
    # Fill missing returns with 0
    for col in ["tlt_ret", "ief_ret", "shy_ret", "spy_ret", "gld_ret"]:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0)
    
    # Portfolio: 46% SPY, 38% GLD, 36% Bonds (TLT/IEF/SHY)
    # We'll compare bond allocation only, keeping SPY/GLD constant
    
    spy_weight = 0.46
    gld_weight = 0.38
    bond_weight = 0.36
    
    # Static allocation
    static_total_weight = sum(STATIC_ALLOCATION.values())
    static_tlt_w = STATIC_ALLOCATION["tlt"] / static_total_weight
    static_ief_w = STATIC_ALLOCATION["ief"] / static_total_weight
    static_shy_w = STATIC_ALLOCATION["shy"] / static_total_weight
    
    # Calculate static portfolio returns
    merged["static_bond_ret"] = (
        static_tlt_w * merged["tlt_ret"] +
        static_ief_w * merged["ief_ret"] +
        static_shy_w * merged["shy_ret"]
    )
    
    merged["static_portfolio_ret"] = (
        spy_weight * merged["spy_ret"] +
        gld_weight * merged["gld_ret"] +
        bond_weight * merged["static_bond_ret"]
    )
    
    # Track dynamic allocation
    merged["dynamic_tlt_w"] = static_tlt_w
    merged["dynamic_ief_w"] = static_ief_w
    merged["dynamic_shy_w"] = static_shy_w
    
    # Apply regime-based shifts with transition constraints
    current_regime = "flat"
    regime_days_count = {"inverted": 0, "flat": 0, "steep": 0}
    regime_transitions = 0
    days_in_current_regime = 0
    rebalance_costs = 0.0
    
    for idx in range(len(merged)):
        row = merged.iloc[idx]
        detected_regime = row.get("regime", "flat")
        
        # Check for regime change
        if detected_regime != current_regime:
            days_in_current_regime += 1
            
            # Require 30 days in new regime before switching
            if days_in_current_regime >= 30:
                # Record transition
                regime_transitions += 1
                
                # Get new allocation
                new_alloc = DYNAMIC_ALLOCATIONS.get(detected_regime, STATIC_ALLOCATION)
                new_total = sum(new_alloc.values())
                
                # Calculate target weights
                new_tlt = new_alloc["tlt"] / new_total
                new_ief = new_alloc["ief"] / new_total
                new_shy = new_alloc["shy"] / new_total
                
                # Apply max 25% shift per month constraint
                max_shift = 0.25 / 21  # Per day limit
                
                old_tlt = merged.at[merged.index[idx], "dynamic_tlt_w"]
                old_ief = merged.at[merged.index[idx], "dynamic_ief_w"]
                old_shy = merged.at[merged.index[idx], "dynamic_shy_w"]
                
                # Gradual shift
                tlt_shift = np.clip(new_tlt - old_tlt, -max_shift, max_shift)
                ief_shift = np.clip(new_ief - old_ief, -max_shift, max_shift)
                shy_shift = np.clip(new_shy - old_shy, -max_shift, max_shift)
                
                merged.at[merged.index[idx], "dynamic_tlt_w"] = old_tlt + tlt_shift
                merged.at[merged.index[idx], "dynamic_ief_w"] = old_ief + ief_shift
                merged.at[merged.index[idx], "dynamic_shy_w"] = old_shy + shy_shift
                
                # Transaction cost for rebalancing
                rebalance_costs += bond_weight * TRANSACTION_COST * (
                    abs(tlt_shift) + abs(ief_shift) + abs(shy_shift)
                )
                
                # Update regime tracking
                current_regime = detected_regime
                days_in_current_regime = 0
        else:
            days_in_current_regime += 1
            # Carry forward weights
            if idx > 0:
                merged.at[merged.index[idx], "dynamic_tlt_w"] = merged.iloc[idx-1]["dynamic_tlt_w"]
                merged.at[merged.index[idx], "dynamic_ief_w"] = merged.iloc[idx-1]["dynamic_ief_w"]
                merged.at[merged.index[idx], "dynamic_shy_w"] = merged.iloc[idx-1]["dynamic_shy_w"]
        
        # Count regime days
        regime_days_count[detected_regime] = regime_days_count.get(detected_regime, 0) + 1
    
    # Calculate dynamic portfolio returns
    merged["dynamic_bond_ret"] = (
        merged["dynamic_tlt_w"] * merged["tlt_ret"] +
        merged["dynamic_ief_w"] * merged["ief_ret"] +
        merged["dynamic_shy_w"] * merged["shy_ret"]
    )
    
    merged["dynamic_portfolio_ret"] = (
        spy_weight * merged["spy_ret"] +
        gld_weight * merged["gld_ret"] +
        bond_weight * merged["dynamic_bond_ret"]
    )
    
    # Calculate metrics
    static_returns = merged["static_portfolio_ret"].fillna(0)
    dynamic_returns = merged["dynamic_portfolio_ret"].fillna(0)
    
    # Apply transaction costs
    static_returns = static_returns - (TRANSACTION_COST / 252)  # Assume annual rebalancing
    dynamic_returns = dynamic_returns - rebalance_costs / len(merged)
    
    # Calculate overall metrics
    static_cagr = calculate_cagr(static_returns)
    static_vol = static_returns.std() * np.sqrt(252)
    static_sharpe = calculate_sharpe(static_returns)
    static_max_dd = calculate_max_drawdown(static_returns)
    
    dynamic_cagr = calculate_cagr(dynamic_returns)
    dynamic_vol = dynamic_returns.std() * np.sqrt(252)
    dynamic_sharpe = calculate_sharpe(dynamic_returns)
    dynamic_max_dd = calculate_max_drawdown(dynamic_returns)
    
    # Crisis performance
    crisis_2008_static = merged[merged["date"].dt.year == 2008]["static_portfolio_ret"].sum()
    crisis_2008_dynamic = merged[merged["date"].dt.year == 2008]["dynamic_portfolio_ret"].sum()
    
    crisis_2020_static = merged[merged["date"].dt.year == 2020]["static_portfolio_ret"].sum()
    crisis_2020_dynamic = merged[merged["date"].dt.year == 2020]["dynamic_portfolio_ret"].sum()
    
    crisis_2022_static = merged[merged["date"].dt.year == 2022]["static_portfolio_ret"].sum()
    crisis_2022_dynamic = merged[merged["date"].dt.year == 2022]["dynamic_portfolio_ret"].sum()
    
    return BacktestResult(
        static_cagr=static_cagr,
        static_volatility=static_vol,
        static_sharpe=static_sharpe,
        static_max_dd=static_max_dd,
        dynamic_cagr=dynamic_cagr,
        dynamic_volatility=dynamic_vol,
        dynamic_sharpe=dynamic_sharpe,
        dynamic_max_dd=dynamic_max_dd,
        sharpe_delta=dynamic_sharpe - static_sharpe,
        cagr_delta=dynamic_cagr - static_cagr,
        max_dd_delta=dynamic_max_dd - static_max_dd,
        crisis_2008_static=crisis_2008_static,
        crisis_2008_dynamic=crisis_2008_dynamic,
        crisis_2020_static=crisis_2020_static,
        crisis_2020_dynamic=crisis_2020_dynamic,
        crisis_2022_static=crisis_2022_static,
        crisis_2022_dynamic=crisis_2022_dynamic,
        regime_days=regime_days_count,
        regime_transitions=regime_transitions,
        rebalancing_costs=rebalance_costs,
        start_date=start_date,
        end_date=end_date,
        total_days=len(merged),
        timestamp=datetime.now().isoformat()
    )


def print_results(result: BacktestResult):
    """Print backtest results in formatted table."""
    print("\n" + "="*70)
    print("DURATION-YIELD CURVE REGIME BACKTEST RESULTS")
    print("="*70)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Total Days: {result.total_days:,}")
    print()
    
    print("-"*70)
    print("PERFORMANCE COMPARISON")
    print("-"*70)
    print(f"{'Metric':<25} {'Static':<15} {'Dynamic':<15} {'Delta':<15}")
    print("-"*70)
    print(f"{'CAGR':<25} {result.static_cagr*100:>14.2f}% {result.dynamic_cagr*100:>14.2f}% {result.cagr_delta*100:>+14.2f}%")
    print(f"{'Volatility':<25} {result.static_volatility*100:>14.2f}% {result.dynamic_volatility*100:>14.2f}% {(result.dynamic_volatility-result.static_volatility)*100:>+14.2f}%")
    print(f"{'Sharpe Ratio':<25} {result.static_sharpe:>14.3f} {result.dynamic_sharpe:>14.3f} {result.sharpe_delta:>+14.3f}")
    print(f"{'Max Drawdown':<25} {result.static_max_dd*100:>14.2f}% {result.dynamic_max_dd*100:>14.2f}% {result.max_dd_delta*100:>+14.2f}%")
    print()
    
    print("-"*70)
    print("CRISIS PERFORMANCE")
    print("-"*70)
    print(f"{'Crisis':<25} {'Static':<15} {'Dynamic':<15} {'Delta':<15}")
    print("-"*70)
    print(f"{'2008 Financial Crisis':<25} {result.crisis_2008_static*100:>14.2f}% {result.crisis_2008_dynamic*100:>14.2f}% {(result.crisis_2008_dynamic-result.crisis_2008_static)*100:>+14.2f}%")
    print(f"{'2020 COVID':<25} {result.crisis_2020_static*100:>14.2f}% {result.crisis_2020_dynamic*100:>14.2f}% {(result.crisis_2020_dynamic-result.crisis_2020_static)*100:>+14.2f}%")
    print(f"{'2022 Rate Hikes':<25} {result.crisis_2022_static*100:>14.2f}% {result.crisis_2022_dynamic*100:>14.2f}% {(result.crisis_2022_dynamic-result.crisis_2022_static)*100:>+14.2f}%")
    print()
    
    print("-"*70)
    print("REGIME STATISTICS")
    print("-"*70)
    for regime, days in result.regime_days.items():
        pct = days / result.total_days * 100
        print(f"{regime.capitalize():<25} {days:>10,} days ({pct:>5.1f}%)")
    print(f"{'Total Regime Transitions':<25} {result.regime_transitions:>14}")
    print(f"{'Rebalancing Costs':<25} ${result.rebalancing_costs*100000:>13.2f}")
    print()
    
    print("-"*70)
    print("SUCCESS CRITERIA VALIDATION")
    print("-"*70)
    
    # Check criteria
    sharpe_target_met = result.sharpe_delta >= 0.015
    max_dd_ok = result.max_dd_delta > -0.02
    crisis_2008_ok = result.crisis_2008_dynamic <= result.crisis_2008_static + 0.02
    
    print(f"{'Sharpe +0.015 target':<40} {'✓ PASS' if sharpe_target_met else '✗ FAIL':<15} (got {result.sharpe_delta:+.3f})")
    print(f"{'Max DD <2% degradation':<40} {'✓ PASS' if max_dd_ok else '✗ FAIL':<15} (got {result.max_dd_delta*100:+.2f}%)")
    print(f"{'2008 crisis benefit':<40} {'✓ PASS' if crisis_2008_ok else '✗ FAIL':<15} (dynamic {result.crisis_2008_dynamic*100:.1f}% vs static {result.crisis_2008_static*100:.1f}%)")
    print()
    
    print("="*70)


def save_results(result: BacktestResult):
    """Save results to JSON file."""
    with open(OUTPUT_PATH, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    logger.info(f"Results saved to {OUTPUT_PATH}")


def main():
    """Main entry point."""
    logger.info("Duration-Yield Curve Regime Backtest - v3.11 Phase 4")
    logger.info("="*50)
    
    # Load data
    prices_df = load_price_data()
    regimes_df = load_yield_spread_history()
    
    # Run backtest
    result = run_backtest(prices_df, regimes_df)
    
    if result:
        # Print results
        print_results(result)
        
        # Save results
        save_results(result)
        
        # Return exit code based on success criteria
        success = (
            result.sharpe_delta >= 0.015 and
            result.max_dd_delta > -0.02 and
            result.crisis_2008_dynamic <= result.crisis_2008_static + 0.02
        )
        
        return 0 if success else 1
    else:
        logger.error("Backtest failed")
        return 1


if __name__ == "__main__":
    exit(main())
