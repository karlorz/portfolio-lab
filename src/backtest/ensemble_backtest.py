#!/usr/bin/env python3
"""
Portfolio-Lab v2.82: 8-Source Ensemble Backtest Engine

Validates the integrated SignalIntegrator with all 8 signal sources:
- technical, macro, alternative_data, llm_sentiment (legacy 4)
- tsmom (v2.52), multi_speed (v2.56), risk_parity (v2.57), network_momentum (v2.58)

Target: Sharpe > 0.95 on 2005-2026 data
Usage:
    python -m src.backtest.ensemble_backtest run --portfolio 46/38/16
    python -m src.backtest.ensemble_backtest validate --target-sharpe 0.95
    python -m src.backtest.ensemble_backtest plot --output results.png
"""

import json
import sqlite3
import numpy as np
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.signals.integrator import SignalIntegrator


@dataclass
class EnsembleBacktestResult:
    """Complete ensemble backtest metrics"""
    start_date: str
    end_date: str
    portfolio: str
    
    # Returns
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk
    max_drawdown: float
    max_dd_duration: int
    calmar_ratio: float
    var_95: float
    cvar_95: float
    
    # Signal stats
    num_rebalances: int
    avg_signal_confidence: float
    regime_distribution: Dict[str, float]
    
    # Crisis performance
    crisis_alpha_2008: float
    crisis_alpha_2020: float
    crisis_alpha_2022: float
    
    # Component contributions
    source_contributions: Dict[str, Dict[str, float]]
    
    # Rolling metrics
    rolling_sharpe_1y: List[Tuple[str, float]]
    
    def to_dict(self) -> dict:
        return asdict(self)


class EnsembleBacktestEngine:
    """
    Full SignalIntegrator backtest with 8 signal sources
    Simulates daily signal generation and allocation shifts
    """
    
    # Crisis periods for validation
    CRISIS_PERIODS = {
        "2008": ("2008-09-01", "2008-12-31"),
        "2020": ("2020-02-19", "2020-04-30"),
        "2022": ("2022-01-01", "2022-10-31"),
    }
    
    # Transaction costs
    TX_COST_BPS = 5.0  # 5 bps per trade
    
    def __init__(
        self, 
        db_path: Path = None,
        integrator: SignalIntegrator = None
    ):
        if db_path is None:
            db_path = Path("/root/projects/portfolio-lab/data/market.db")
        self.db_path = db_path
        
        # Lazy init integrator (creates all 8 sources)
        self.integrator = integrator or SignalIntegrator()
        
        # Cache for historical data
        self._price_cache: Dict[str, List[Dict]] = {}
        self._signal_cache: Dict[str, List[Dict]] = {}
        
    def _fetch_historical_prices(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> List[Dict]:
        """Fetch prices from database with caching"""
        cache_key = f"{symbol}:{start_date}:{end_date}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
            
        if not self.db_path.exists():
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, close, open, high, low, volume
            FROM prices
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, (symbol, start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        data = [
            {
                "date": row[0], 
                "close": row[1], 
                "open": row[2],
                "high": row[3],
                "low": row[4],
                "volume": row[5]
            }
            for row in rows
        ]
        
        self._price_cache[cache_key] = data
        return data
    
    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Daily log returns"""
        prices_arr = np.array(prices)
        log_returns = np.diff(np.log(prices_arr))
        return log_returns
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int]:
        """Returns (max_dd, max_dd_duration_days)"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = np.min(drawdown)
        
        # Find duration of longest drawdown
        in_dd = drawdown < -0.001  # In drawdown if > 0.1% below peak
        max_duration = 0
        current_duration = 0
        
        for i in range(len(in_dd)):
            if in_dd[i]:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_dd, max_duration
    
    def _calculate_crisis_alpha(
        self,
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
        crisis_period: Tuple[str, str]
    ) -> float:
        """Calculate alpha during crisis period"""
        start, end = crisis_period
        
        # Get returns within crisis window
        crisis_portfolio = [
            r for date, r in portfolio_returns.items()
            if start <= date <= end
        ]
        crisis_benchmark = [
            r for date, r in benchmark_returns.items()
            if start <= date <= end
        ]
        
        if not crisis_portfolio or not crisis_benchmark:
            return 0.0
            
        port_cum = np.prod([1 + r for r in crisis_portfolio]) - 1
        bench_cum = np.prod([1 + r for r in crisis_benchmark]) - 1
        
        return port_cum - bench_cum
    
    def _generate_daily_signals(
        self,
        date: str,
        portfolio: Dict[str, float]
    ) -> Dict[str, any]:
        """Generate signals from all 8 sources for a given date"""
        signals = {}
        
        # Get composite signals for each asset
        for asset in portfolio.keys():
            try:
                composite = self.integrator.get_composite_signal(asset)
                signals[asset] = {
                    "score": composite.score,
                    "confidence": composite.confidence,
                    "regime": composite.regime,
                    "sources": composite.sources
                }
            except Exception as e:
                # Fallback to neutral signal
                signals[asset] = {
                    "score": 0.0,
                    "confidence": 0.0,
                    "regime": "neutral",
                    "sources": []
                }
        
        return signals
    
    def _calculate_allocation_deltas(
        self,
        current_alloc: Dict[str, float],
        signals: Dict[str, any],
        max_delta: float = 0.10
    ) -> Dict[str, float]:
        """Calculate target allocation based on signals"""
        target = current_alloc.copy()
        
        # Normalize signal scores
        total_score = sum(abs(s["score"]) for s in signals.values())
        if total_score > 0:
            weights = {
                asset: abs(s["score"]) / total_score
                for asset, s in signals.items()
            }
        else:
            weights = {asset: 1.0/len(signals) for asset in signals}
        
        # Calculate target based on signal strength and direction
        for asset, signal in signals.items():
            direction = np.sign(signal["score"])
            strength = abs(signal["score"]) * signal["confidence"]
            
            # Base adjustment
            adjustment = direction * strength * max_delta
            
            # Apply to target (will be constrained later)
            target[asset] = current_alloc[asset] + adjustment
        
        # Normalize to sum to 1.0
        total = sum(target.values())
        if total > 0:
            target = {k: v/total for k, v in target.items()}
        
        return target
    
    def run_backtest(
        self,
        portfolio: Dict[str, float],
        start_date: str = "2005-01-01",
        end_date: str = "2026-05-13",
        rebalance_freq: str = "monthly"
    ) -> EnsembleBacktestResult:
        """
        Run full ensemble backtest
        
        Args:
            portfolio: Base allocation e.g. {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
            start_date: Backtest start
            end_date: Backtest end
            rebalance_freq: monthly, weekly, or threshold (drift-based)
        """
        assets = list(portfolio.keys())
        
        # Fetch all price data
        price_data = {}
        for asset in assets:
            price_data[asset] = self._fetch_historical_prices(asset, start_date, end_date)
        
        # Find common dates
        all_dates = set()
        for asset, data in price_data.items():
            for d in data:
                all_dates.add(d["date"])
        
        dates = sorted(all_dates)
        if len(dates) < 30:
            raise ValueError(f"Insufficient data: only {len(dates)} trading days")
        
        # Initialize
        current_alloc = portfolio.copy()
        portfolio_value = 1.0
        equity_curve = [portfolio_value]
        daily_returns = {}
        
        rebalance_dates = []
        signal_history = []
        
        # Determine rebalance dates
        if rebalance_freq == "monthly":
            last_month = None
            for d in dates:
                month = d[:7]  # YYYY-MM
                if month != last_month:
                    rebalance_dates.append(d)
                    last_month = month
        elif rebalance_freq == "weekly":
            # Weekly on Mondays
            last_week = None
            for d in dates:
                dt = datetime.strptime(d, "%Y-%m-%d")
                week = dt.strftime("%Y-W%U")
                if week != last_week:
                    rebalance_dates.append(d)
                    last_week = week
        else:
            rebalance_dates = dates[::20]  # Every ~20 days for threshold-based
        
        # Backtest loop
        for i, date in enumerate(dates[1:], 1):
            prev_date = dates[i-1]
            
            # Calculate daily portfolio return
            daily_ret = 0.0
            for asset in assets:
                if asset in price_data and len(price_data[asset]) > i:
                    prev_price = price_data[asset][i-1]["close"]
                    curr_price = price_data[asset][i]["close"]
                    if prev_price > 0:
                        asset_ret = (curr_price - prev_price) / prev_price
                        daily_ret += current_alloc.get(asset, 0) * asset_ret
            
            # Apply transaction costs on rebalance days
            if date in rebalance_dates and i > 1:
                # Calculate turnover
                turnover = sum(abs(current_alloc.get(a, 0) - portfolio.get(a, 0)) 
                              for a in assets) / 2
                cost = turnover * self.TX_COST_BPS / 10000
                daily_ret -= cost
                
                # Generate new signals and update allocation
                signals = self._generate_daily_signals(date, current_alloc)
                signal_history.append({
                    "date": date,
                    "signals": signals,
                    "allocation": current_alloc.copy()
                })
                
                # Calculate new target allocation
                new_alloc = self._calculate_allocation_deltas(current_alloc, signals)
                current_alloc = new_alloc
            
            portfolio_value *= (1 + daily_ret)
            equity_curve.append(portfolio_value)
            daily_returns[date] = daily_ret
        
        # Calculate metrics
        returns_arr = np.array(list(daily_returns.values()))
        
        total_return = portfolio_value - 1.0
        n_years = len(dates) / 252
        annualized_return = (portfolio_value ** (1/n_years)) - 1 if n_years > 0 else 0
        
        volatility = np.std(returns_arr) * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns_arr[returns_arr < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        
        max_dd, max_dd_duration = self._calculate_max_drawdown(np.array(equity_curve))
        calmar = annualized_return / abs(max_dd) if max_dd < 0 else 0
        
        # VaR/CVaR
        var_95 = np.percentile(returns_arr, 5)
        cvar_95 = np.mean(returns_arr[returns_arr <= var_95])
        
        # Fetch SPY for benchmark comparison
        spy_data = self._fetch_historical_prices("SPY", start_date, end_date)
        spy_returns = {}
        for i in range(1, len(spy_data)):
            prev = spy_data[i-1]["close"]
            curr = spy_data[i]["close"]
            if prev > 0:
                spy_returns[spy_data[i]["date"]] = (curr - prev) / prev
        
        # Crisis alphas
        crisis_alpha = {}
        for crisis_name, period in self.CRISIS_PERIODS.items():
            crisis_alpha[crisis_name] = self._calculate_crisis_alpha(
                daily_returns, spy_returns, period
            )
        
        # Signal source contributions (aggregated from history)
        source_contributions = defaultdict(lambda: {"hits": 0, "total": 0, "avg_confidence": 0.0})
        for day in signal_history:
            for asset, sig in day["signals"].items():
                if "sources" in sig:
                    for src in sig["sources"]:
                        name = src.get("source", "unknown")
                        source_contributions[name]["hits"] += 1
                        source_contributions[name]["total"] += 1
                        source_contributions[name]["avg_confidence"] += sig.get("confidence", 0)
        
        for name, stats in source_contributions.items():
            if stats["total"] > 0:
                stats["avg_confidence"] /= stats["total"]
        
        # Regime distribution
        regime_counts = defaultdict(int)
        for day in signal_history:
            for asset, sig in day["signals"].items():
                regime = sig.get("regime", "unknown")
                regime_counts[regime] += 1
        
        total_regime = sum(regime_counts.values())
        regime_dist = {r: c/total_regime for r, c in regime_counts.items()} if total_regime > 0 else {}
        
        # Rolling 1-year Sharpe
        rolling_sharpe = []
        window = 252
        if len(returns_arr) >= window:
            for i in range(window, len(returns_arr)):
                window_rets = returns_arr[i-window:i]
                window_annual_ret = np.mean(window_rets) * 252
                window_vol = np.std(window_rets) * np.sqrt(252)
                window_sharpe = window_annual_ret / window_vol if window_vol > 0 else 0
                rolling_sharpe.append((dates[i], window_sharpe))
        
        return EnsembleBacktestResult(
            start_date=dates[0],
            end_date=dates[-1],
            portfolio="/".join(f"{k}:{v:.2f}" for k, v in portfolio.items()),
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_dd_duration=max_dd_duration,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
            num_rebalances=len(rebalance_dates),
            avg_signal_confidence=np.mean([s["signals"].get("SPY", {}).get("confidence", 0) 
                                          for s in signal_history]) if signal_history else 0,
            regime_distribution=dict(regime_dist),
            crisis_alpha_2008=crisis_alpha.get("2008", 0),
            crisis_alpha_2020=crisis_alpha.get("2020", 0),
            crisis_alpha_2022=crisis_alpha.get("2022", 0),
            source_contributions=dict(source_contributions),
            rolling_sharpe_1y=rolling_sharpe[-10:] if rolling_sharpe else []
        )
    
    def validate_target(self, result: EnsembleBacktestResult, target_sharpe: float = 0.95) -> bool:
        """Validate against target Sharpe ratio"""
        passed = result.sharpe_ratio >= target_sharpe
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE BACKTEST VALIDATION (Target Sharpe: {target_sharpe:.2f})")
        print(f"{'='*60}")
        print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f} {'✓' if passed else '✗'}")
        print(f"CAGR:              {result.annualized_return*100:.2f}%")
        print(f"Volatility:        {result.volatility*100:.2f}%")
        print(f"Max Drawdown:      {result.max_drawdown*100:.1f}%")
        print(f"Sortino:           {result.sortino_ratio:.2f}")
        print(f"Calmar:            {result.calmar_ratio:.2f}")
        print(f"\nCrisis Alpha:")
        print(f"  2008 GFC:        {result.crisis_alpha_2008*100:+.1f}%")
        print(f"  2020 COVID:      {result.crisis_alpha_2020*100:+.1f}%")
        print(f"  2022 Bear:       {result.crisis_alpha_2022*100:+.1f}%")
        print(f"\nSource Contributions:")
        for src, stats in sorted(result.source_contributions.items(), 
                                 key=lambda x: x[1].get("hits", 0), reverse=True)[:5]:
            print(f"  {src:20s}: {stats['hits']:4d} hits, conf={stats['avg_confidence']:.2f}")
        print(f"\nRegime Distribution:")
        for regime, pct in sorted(result.regime_distribution.items(), key=lambda x: -x[1]):
            print(f"  {regime:12s}: {pct*100:.1f}%")
        print(f"{'='*60}")
        
        return passed


def main():
    parser = argparse.ArgumentParser(
        description="8-Source Ensemble Backtest Engine v2.82"
    )
    parser.add_argument(
        "command",
        choices=["run", "validate", "benchmark"],
        help="Run backtest, validate against target, or benchmark vs static"
    )
    parser.add_argument(
        "--portfolio", "-p",
        default="46/38/16",
        help="Portfolio allocation (e.g., 46/38/16 for SPY/GLD/TLT)"
    )
    parser.add_argument(
        "--start", "-s",
        default="2006-01-01",
        help="Start date (default: 2006-01-01, integrator data availability)"
    )
    parser.add_argument(
        "--end", "-e",
        default="2026-05-13",
        help="End date"
    )
    parser.add_argument(
        "--target-sharpe", "-t",
        type=float,
        default=0.95,
        help="Target Sharpe ratio (default: 0.95)"
    )
    parser.add_argument(
        "--rebalance", "-r",
        default="monthly",
        choices=["monthly", "weekly", "threshold"],
        help="Rebalance frequency"
    )
    parser.add_argument(
        "--output", "-o",
        help="JSON output file for results"
    )
    
    args = parser.parse_args()
    
    # Parse portfolio
    weights = [float(w) for w in args.portfolio.split("/")]
    assets = ["SPY", "GLD", "TLT"][:len(weights)]
    portfolio = {assets[i]: weights[i]/100 if weights[i] > 1 else weights[i] 
                  for i in range(len(weights))}
    
    # Normalize
    total = sum(portfolio.values())
    portfolio = {k: v/total for k, v in portfolio.items()}
    
    print(f"Portfolio-Lab v2.82: 8-Source Ensemble Backtest")
    print(f"Portfolio: {portfolio}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Rebalance: {args.rebalance}")
    print(f"Initializing SignalIntegrator with 8 sources...")
    
    engine = EnsembleBacktestEngine()
    
    if args.command == "run" or args.command == "validate":
        result = engine.run_backtest(
            portfolio=portfolio,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq=args.rebalance
        )
        
        engine.validate_target(result, args.target_sharpe)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
    
    elif args.command == "benchmark":
        # Compare ensemble vs static allocation
        print("\n" + "="*60)
        print("ENSEMBLE VS STATIC BENCHMARK")
        print("="*60)
        
        ensemble_result = engine.run_backtest(
            portfolio=portfolio,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq=args.rebalance
        )
        
        # Run static (no signal integration)
        print("\nRunning static allocation comparison...")
        static_engine = EnsembleBacktestEngine()
        # Override signal generation to return neutral
        original_generate = static_engine._generate_daily_signals
        static_engine._generate_daily_signals = lambda date, port: {
            asset: {"score": 0, "confidence": 0, "regime": "neutral", "sources": []}
            for asset in port.keys()
        }
        
        static_result = static_engine.run_backtest(
            portfolio=portfolio,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq="yearly"  # Static rebalances only yearly for drift
        )
        
        print(f"\nEnsemble (8-source):")
        print(f"  Sharpe: {ensemble_result.sharpe_ratio:.2f}")
        print(f"  CAGR:   {ensemble_result.annualized_return*100:.2f}%")
        print(f"  MaxDD:  {ensemble_result.max_drawdown*100:.1f}%")
        
        print(f"\nStatic (46/38/16):")
        print(f"  Sharpe: {static_result.sharpe_ratio:.2f}")
        print(f"  CAGR:   {static_result.annualized_return*100:.2f}%")
        print(f"  MaxDD:  {static_result.max_drawdown*100:.1f}%")
        
        improvement = ensemble_result.sharpe_ratio - static_result.sharpe_ratio
        print(f"\nImprovement: {improvement:+.2f} Sharpe ({improvement/static_result.sharpe_ratio*100:+.1f}%)")
        print("="*60)


if __name__ == "__main__":
    main()
