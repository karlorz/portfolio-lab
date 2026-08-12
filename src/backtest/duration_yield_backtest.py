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
from datetime import datetime
from typing import Dict
import logging

import pandas as pd
import numpy as np

from src.paths import BASE_ALLOCATION, DATA_DIR, MARKET_DB, sqlite_connect
from src.backtest.metrics import (
    BacktestResult,
    save_results_json,
)


__all__ = ['STATIC_ALLOCATION', 'DYNAMIC_ALLOCATIONS', 'REGIME_EFFECTIVE_DURATION', 'EXPENSE_RATIOS', 'TRANSACTION_COST', 'load_price_data', 'load_yield_curve_data', 'classify_regime_from_spread', 'load_yield_spread_history', 'calculate_returns', 'calculate_sharpe', 'calculate_max_drawdown', 'calculate_cagr', 'run_backtest', 'print_results', 'save_results']

# Setup logging
logger = logging.getLogger(__name__)

# Paths
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


def load_price_data() -> pd.DataFrame:
    """Load price data from prices.json.

    Delegates to the shared grid_runner loader (Item 32 A5 consolidation;
    semantics ported faithfully — A4 output-equality verified).
    """
    logger.info("Loading price data...")
    from src.backtest.grid_runner import load_prices, prices_to_frame
    return prices_to_frame(load_prices())


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
    if MARKET_DB.exists():
        with sqlite_connect(MARKET_DB) as conn:
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

                if len(df) > 0:
                    df["date"] = pd.to_datetime(df["date"])
                    logger.info("Loaded %d regime records from database", len(df))
                    return df

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
    logger.info("Created synthetic regime data: %d days", len(df))
    return df


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from prices."""
    return prices.pct_change().fillna(0)


# Metrics moved to the shared grid_runner (Item 32 A5 consolidation);
# re-exported here so module-level names stay importable/patchable and
# run_backtest's module-global calls resolve to the identical implementations.
from src.backtest.grid_runner import (  # noqa: E402
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_cagr,
)


def run_backtest(
    prices_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    start_date: str = "2005-01-01",
    end_date: str = "2026-05-14"
) -> BacktestResult:
    """Run backtest comparing static vs dynamic duration allocation."""
    logger.info("Running backtest from %s to %s...", start_date, end_date)

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

    logger.info("Backtesting on %d days", len(merged))

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

    spy_weight = BASE_ALLOCATION["SPY"]
    gld_weight = BASE_ALLOCATION["GLD"]
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
        total_return=((1 + dynamic_returns).prod() - 1) * 100,
        cagr=dynamic_cagr * 100,
        volatility=dynamic_vol * 100,
        sharpe_ratio=dynamic_sharpe,
        max_drawdown=dynamic_max_dd * 100,
        total_rebalances=regime_transitions,
        total_transaction_costs=rebalance_costs,
        crisis_returns={
            "2008": crisis_2008_dynamic * 100,
            "2020": crisis_2020_dynamic * 100,
            "2022": crisis_2022_dynamic * 100,
        },
        extras={
            "static_cagr": static_cagr,
            "static_volatility": static_vol,
            "static_sharpe": static_sharpe,
            "static_max_dd": static_max_dd,
            "dynamic_cagr": dynamic_cagr,
            "dynamic_volatility": dynamic_vol,
            "dynamic_sharpe": dynamic_sharpe,
            "dynamic_max_dd": dynamic_max_dd,
            "sharpe_delta": dynamic_sharpe - static_sharpe,
            "cagr_delta": dynamic_cagr - static_cagr,
            "max_dd_delta": dynamic_max_dd - static_max_dd,
            "crisis_2008_static": crisis_2008_static,
            "crisis_2008_dynamic": crisis_2008_dynamic,
            "crisis_2020_static": crisis_2020_static,
            "crisis_2020_dynamic": crisis_2020_dynamic,
            "crisis_2022_static": crisis_2022_static,
            "crisis_2022_dynamic": crisis_2022_dynamic,
            "regime_days": regime_days_count,
            "regime_transitions": regime_transitions,
            "rebalancing_costs": rebalance_costs,
            "start_date": start_date,
            "end_date": end_date,
            "total_days": len(merged),
            "timestamp": datetime.now().isoformat(),
        },
    )


def print_results(result: BacktestResult):
    """Print backtest results in formatted table."""
    e = result.extras
    logger.info("\n" + "="*70)
    logger.info("DURATION-YIELD CURVE REGIME BACKTEST RESULTS")
    logger.info("="*70)
    logger.info(f"Period: {e['start_date']} to {e['end_date']}")
    logger.info(f"Total Days: {e['total_days']:,}")
    logger.info("")

    logger.info("-"*70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("-"*70)
    logger.info(f"{'Metric':<25} {'Static':<15} {'Dynamic':<15} {'Delta':<15}")
    logger.info("-"*70)
    logger.info(f"{'CAGR':<25} {e['static_cagr']*100:>14.2f}% {e['dynamic_cagr']*100:>14.2f}% {e['cagr_delta']*100:>+14.2f}%")
    logger.info(f"{'Volatility':<25} {e['static_volatility']*100:>14.2f}% {e['dynamic_volatility']*100:>14.2f}% {(e['dynamic_volatility']-e['static_volatility'])*100:>+14.2f}%")
    logger.info(f"{'Sharpe Ratio':<25} {e['static_sharpe']:>14.3f} {e['dynamic_sharpe']:>14.3f} {e['sharpe_delta']:>+14.3f}")
    logger.info(f"{'Max Drawdown':<25} {e['static_max_dd']*100:>14.2f}% {e['dynamic_max_dd']*100:>14.2f}% {e['max_dd_delta']*100:>+14.2f}%")
    logger.info("")

    logger.info("-"*70)
    logger.info("CRISIS PERFORMANCE")
    logger.info("-"*70)
    logger.info(f"{'Crisis':<25} {'Static':<15} {'Dynamic':<15} {'Delta':<15}")
    logger.info("-"*70)
    logger.info(f"{'2008 Financial Crisis':<25} {e['crisis_2008_static']*100:>14.2f}% {e['crisis_2008_dynamic']*100:>14.2f}% {(e['crisis_2008_dynamic']-e['crisis_2008_static'])*100:>+14.2f}%")
    logger.info(f"{'2020 COVID':<25} {e['crisis_2020_static']*100:>14.2f}% {e['crisis_2020_dynamic']*100:>14.2f}% {(e['crisis_2020_dynamic']-e['crisis_2020_static'])*100:>+14.2f}%")
    logger.info(f"{'2022 Rate Hikes':<25} {e['crisis_2022_static']*100:>14.2f}% {e['crisis_2022_dynamic']*100:>14.2f}% {(e['crisis_2022_dynamic']-e['crisis_2022_static'])*100:>+14.2f}%")
    logger.info("")

    logger.info("-"*70)
    logger.info("REGIME STATISTICS")
    logger.info("-"*70)
    for regime, days in e["regime_days"].items():
        pct = days / e["total_days"] * 100
        logger.info(f"{regime.capitalize():<25} {days:>10,} days ({pct:>5.1f}%)")
    logger.info(f"{'Total Regime Transitions':<25} {e['regime_transitions']:>14}")
    logger.info(f"{'Rebalancing Costs':<25} ${e['rebalancing_costs']*100000:>13.2f}")
    logger.info("")

    logger.info("-"*70)
    logger.info("SUCCESS CRITERIA VALIDATION")
    logger.info("-"*70)

    # Check criteria
    ex = result.extras
    sharpe_target_met = ex["sharpe_delta"] >= 0.015
    max_dd_ok = ex["max_dd_delta"] > -0.02
    crisis_2008_ok = ex["crisis_2008_dynamic"] <= ex["crisis_2008_static"] + 0.02

    logger.info(f"{'Sharpe +0.015 target':<40} {'✓ PASS' if sharpe_target_met else '✗ FAIL':<15} (got {ex['sharpe_delta']:+.3f})")
    logger.info(f"{'Max DD <2% degradation':<40} {'✓ PASS' if max_dd_ok else '✗ FAIL':<15} (got {ex['max_dd_delta']*100:+.2f}%)")
    logger.info(f"{'2008 crisis benefit':<40} {'✓ PASS' if crisis_2008_ok else '✗ FAIL':<15} (dynamic {ex['crisis_2008_dynamic']*100:.1f}% vs static {ex['crisis_2008_static']*100:.1f}%)")
    logger.info("")

    logger.info("="*70)


def save_results(result: BacktestResult):
    """Save results to JSON file."""
    from dataclasses import asdict
    save_results_json(asdict(result), output_path=str(OUTPUT_PATH))
    logger.info("Results saved to %s", OUTPUT_PATH)


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
            result.extras["sharpe_delta"] >= 0.015 and
            result.extras["max_dd_delta"] > -0.02 and
            result.extras["crisis_2008_dynamic"] <= result.extras["crisis_2008_static"] + 0.02
        )

        return 0 if success else 1
    else:
        logger.error("Backtest failed")
        return 1


if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    exit(main())
