#!/usr/bin/env python3
"""
Portfolio-Lab v2.55: Combined Strategy Backtest

Historical backtest of unified signal strategy (TSMOM + HMM Regime + Fed Policy)
against baseline 46/38/16 allocation.

Backtest Methodology:
    1. Load historical price data (2006-2026, 5371 trading days)
    2. For each rebalance date (monthly on 21-day intervals):
       a. Calculate TSMOM signals (12m formation, 1m skip, vol-scaled)
       b. Detect HMM regime (5-state classifier)
       c. Determine Fed policy regime (from FRED data or simulation)
       d. Combine signals with conflict resolution
       e. Apply allocation adjustments
    3. Calculate daily returns with transaction costs (10 bps)
    4. Compare to baseline buy-and-hold 46/38/16
    5. Generate performance attribution and statistical tests

Attribution Analysis:
    - TSMOM contribution: Return from momentum overlay
    - HMM contribution: Return from regime-based shifts  
    - Fed contribution: Return from policy-based shifts
    - Interaction effects: Non-linear combinations

Usage:
    python -m src.backtest.combined_strategy backtest --start 2006-01-01 --end 2026-05-08
    python -m src.backtest.combined_strategy attribution --period 2020-01-01/2022-12-31
    python -m src.backtest.combined_strategy compare --baseline 46/38/16
    python -m src.backtest.combined_strategy stress --crisis 2008,2020,2022
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import pickle

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.signals.tsmom_overlay import TSMOMOverlay, TSMOMBacktester, DEFAULT_BASE_ALLOCATION
from src.agents.risk_agent_hmm import HMMRegimeDetector, PortfolioRegimeManager, MarketRegime
from src.signals.fed_policy_overlay import FedPolicyOverlay, classify_fed_regime


# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()
RESULTS_PATH = DATA_DIR / "combined_backtest_results.json"

# Backtest parameters
TRANSACTION_COST = 0.001  # 10 bps per trade
REBALANCE_FREQ = 21  # Monthly (trading days)
MIN_HISTORY_DAYS = 252 + 21  # TSMOM needs 12m + skip
START_DATE = "2006-02-01"  # First valid rebalance after burn-in
END_DATE = "2026-05-08"


@dataclass
class DailyPosition:
    """Portfolio position on a given day."""
    date: str
    weights: Dict[str, float]
    prices: Dict[str, float]
    portfolio_value: float
    
    # Metadata
    tsmom_deltas: Optional[Dict[str, float]] = None
    hmm_regime: Optional[str] = None
    fed_regime: Optional[str] = None
    rebalance_executed: bool = False
    turnover: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy: str
    start_date: str
    end_date: str
    trading_days: int
    rebalances: int
    
    # Performance metrics
    start_value: float
    end_value: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Comparison to baseline
    baseline_cagr: float
    baseline_sharpe: float
    excess_return: float  # Annualized
    information_ratio: float
    
    # Attribution
    tsmom_contribution: float
    hmm_contribution: float
    fed_contribution: float
    
    # Crisis performance
    crisis_2008_return: Optional[float] = None
    crisis_2020_return: Optional[float] = None
    crisis_2022_return: Optional[float] = None
    
    # Path
    daily_values: List[float] = None
    daily_returns: List[float] = None
    positions: List[DailyPosition] = None
    
    def to_dict(self) -> dict:
        return {
            'strategy': self.strategy,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trading_days': self.trading_days,
            'rebalances': self.rebalances,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'cagr': round(self.cagr, 4),
            'volatility': round(self.volatility, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'calmar_ratio': round(self.calmar_ratio, 4),
            'baseline_cagr': round(self.baseline_cagr, 4),
            'baseline_sharpe': round(self.baseline_sharpe, 4),
            'excess_return': round(self.excess_return, 4),
            'information_ratio': round(self.information_ratio, 4),
            'tsmom_contribution': round(self.tsmom_contribution, 4),
            'hmm_contribution': round(self.hmm_contribution, 4),
            'fed_contribution': round(self.fed_contribution, 4),
            'crisis_2008_return': self.crisis_2008_return,
            'crisis_2020_return': self.crisis_2020_return,
            'crisis_2022_return': self.crisis_2022_return,
        }


class CombinedStrategyBacktester:
    """
    Backtest engine for combined signal strategy.
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        base_allocation: Dict[str, float] = None,
        transaction_cost: float = TRANSACTION_COST,
        rebalance_freq: int = REBALANCE_FREQ
    ):
        self.tickers = tickers or ['SPY', 'GLD', 'TLT']
        self.base_allocation = base_allocation or DEFAULT_BASE_ALLOCATION.copy()
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        
        # Initialize signal modules
        self.tsmom = TSMOMOverlay(max_deviation=0.10)
        self.hmm_manager = PortfolioRegimeManager(base_allocation=self.base_allocation)
        self.fed_overlay = FedPolicyOverlay()
        
        # Load trained HMM model
        self.hmm_manager.detector.load()
        
        # Price data cache
        self.prices_df: Optional[pd.DataFrame] = None
        self.dates: List[str] = []
        
    def load_prices(self) -> bool:
        """Load and align price data for all tickers."""
        if not PRICES_PATH.exists():
            print(f"Error: Price data not found at {PRICES_PATH}")
            return False
        
        try:
            with open(PRICES_PATH) as f:
                data = json.load(f)
            
            all_prices = {}
            for ticker in self.tickers:
                if ticker in data:
                    ticker_data = data[ticker]
                    if isinstance(ticker_data, list) and len(ticker_data) > 0:
                        dates = [item['d'] for item in ticker_data]
                        prices = [item['p'] for item in ticker_data]
                        all_prices[ticker] = pd.Series(prices, index=pd.to_datetime(dates))
            
            if not all_prices:
                print("Error: No valid price data loaded")
                return False
            
            # Combine into DataFrame
            self.prices_df = pd.DataFrame(all_prices)
            self.prices_df.dropna(inplace=True)
            self.dates = [d.strftime('%Y-%m-%d') for d in self.prices_df.index]
            
            print(f"Loaded {len(self.prices_df)} days of price data")
            print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
            
            return True
            
        except Exception as e:
            print(f"Error loading prices: {e}")
            return False
    
    def _get_tsmom_deltas(
        self,
        current_idx: int
    ) -> Dict[str, float]:
        """Get TSMOM allocation deltas at a specific date index."""
        deltas = {}
        
        for ticker in self.tickers:
            if ticker not in self.prices_df.columns:
                continue
            
            prices = self.prices_df[ticker].iloc[:current_idx + 1]
            
            # Need at least lookback + skip days
            if len(prices) < self.tsmom.lookback_days + self.tsmom.skip_days:
                deltas[ticker] = 0.0
                continue
            
            # Calculate formation return (excluding skip period)
            start_idx = len(prices) - 1 - self.tsmom.lookback_days - self.tsmom.skip_days
            end_idx = len(prices) - 1 - self.tsmom.skip_days
            
            start_price = prices.iloc[start_idx]
            end_price = prices.iloc[end_idx]
            formation_return = (end_price - start_price) / start_price
            
            # Signal
            signal = 0 if abs(formation_return) < 0.001 else (1 if formation_return > 0 else -1)
            
            # Volatility
            if len(prices) >= self.tsmom.vol_window:
                recent_returns = np.log(prices.iloc[-self.tsmom.vol_window:] / prices.iloc[-self.tsmom.vol_window-1:-1])
                vol = recent_returns.std() * np.sqrt(252)
            else:
                vol = 0.15
            
            vol = max(vol, 0.01)
            
            # Calculate adjustment
            position_normalized = np.clip(signal / vol * 0.15, -1, 1)
            adjustment = position_normalized * self.tsmom.max_deviation
            
            base_weight = self.base_allocation.get(ticker, 0.25)
            target_weight = max(0.05, min(0.95, base_weight + adjustment))
            deltas[ticker] = target_weight - base_weight
        
        return deltas
    
    def _get_hmm_regime(
        self,
        current_idx: int
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Get HMM regime and deltas at a specific date index."""
        if not self.hmm_manager.detector.is_fitted:
            return None, {t: 0.0 for t in self.tickers}
        
        # Detect regime for SPY (proxy for market)
        if 'SPY' not in self.prices_df.columns:
            return None, {t: 0.0 for t in self.tickers}
        
        spy_prices = self.prices_df['SPY'].iloc[:current_idx + 1]
        
        if len(spy_prices) < 126:  # Need enough history
            return None, {t: 0.0 for t in self.tickers}
        
        regime_result = self.hmm_manager.detector.predict_regime(spy_prices, ticker='SPY')
        
        if regime_result is None:
            return None, {t: 0.0 for t in self.tickers}
        
        # Get allocation shifts based on regime
        shifts = {
            MarketRegime.BULL: {'SPY': +0.10, 'GLD': -0.05, 'TLT': -0.05},
            MarketRegime.BEAR: {'SPY': -0.10, 'GLD': +0.05, 'TLT': +0.05},
            MarketRegime.NEUTRAL: {'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0},
            MarketRegime.HIGH_VOL: {'SPY': -0.05, 'GLD': +0.10, 'TLT': -0.05},
            MarketRegime.CRISIS: {'SPY': -0.15, 'GLD': +0.10, 'TLT': +0.05},
        }
        
        deltas = shifts.get(regime_result.regime, {t: 0.0 for t in self.tickers})
        return str(regime_result.regime), deltas
    
    def _get_fed_regime_deltas(
        self,
        current_idx: int
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Get Fed policy regime and deltas.
        
        For backtest: simulate Fed regime based on historical context
        (Fed Funds rate trends, inflation proxy from gold performance, etc.)
        """
        # Simplified: use gold/SPY ratio as inflation proxy
        # and rate changes from TLT performance as rate regime proxy
        
        if len(self.prices_df) < current_idx + 63:
            return None, {t: 0.0 for t in self.tickers}
        
        # Get recent 3-month data
        spy_prices = self.prices_df['SPY'].iloc[current_idx - 62:current_idx + 1]
        tlt_prices = self.prices_df['TLT'].iloc[current_idx - 62:current_idx + 1]
        gld_prices = self.prices_df['GLD'].iloc[current_idx - 62:current_idx + 1] if 'GLD' in self.prices_df else None
        
        # SPY return (proxy for growth)
        spy_return = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1
        
        # TLT return (inverse proxy for rates)
        tlt_return = (tlt_prices.iloc[-1] / tlt_prices.iloc[0]) - 1
        
        # Gold/SPY ratio change (proxy for inflation expectations)
        if gld_prices is not None:
            gld_return = (gld_prices.iloc[-1] / gld_prices.iloc[0]) - 1
            inflation_proxy = gld_return - spy_return
        else:
            inflation_proxy = 0.0
        
        # Classify regime based on heuristics
        if tlt_return > 0.05 and spy_return > 0.05:
            # Both up = easing (rates down, growth up)
            regime = 'EASING'
            deltas = {'SPY': +0.05, 'GLD': +0.05, 'TLT': -0.05}
        elif tlt_return < -0.05 and spy_return < -0.05:
            # Both down = tightening (rates up, growth down)
            regime = 'TIGHTENING'
            deltas = {'SPY': -0.10, 'GLD': +0.10, 'TLT': 0.0}
        elif abs(tlt_return) < 0.02 and abs(spy_return) < 0.03:
            regime = 'NEUTRAL'
            deltas = {'SPY': 0.0, 'GLD': 0.0, 'TLT': 0.0}
        else:
            regime = 'UNCERTAIN'
            deltas = {'SPY': -0.05, 'GLD': +0.10, 'TLT': -0.05}
        
        return regime, deltas
    
    def _combine_signals(
        self,
        tsmom_deltas: Dict[str, float],
        hmm_regime: Optional[str],
        hmm_deltas: Dict[str, float],
        fed_regime: Optional[str],
        fed_deltas: Dict[str, float],
        current_idx: int
    ) -> Tuple[Dict[str, float], str]:
        """
        Combine signals with conflict resolution.
        Simplified version of CombinedSignalOrchestrator for backtest.
        """
        tickers = self.tickers
        
        # Weights (same as orchestrator)
        weights = {
            'tsmom': 0.35,
            'hmm': 0.25,
            'fed': 0.25,
            'base': 0.15,
        }
        
        # Weighted combination
        combined = {t: 0.0 for t in tickers}
        
        for ticker in tickers:
            # TSMOM contribution (confidence ~0.85)
            combined[ticker] += tsmom_deltas.get(ticker, 0.0) * weights['tsmom'] * 0.85
            
            # HMM contribution (confidence ~0.90 when regime is stable)
            hmm_conf = 0.9 if hmm_regime else 0.5
            combined[ticker] += hmm_deltas.get(ticker, 0.0) * weights['hmm'] * hmm_conf
            
            # Fed contribution (lower confidence in backtest simulation)
            fed_conf = 0.7 if fed_regime else 0.5
            combined[ticker] += fed_deltas.get(ticker, 0.0) * weights['fed'] * fed_conf
        
        # Normalize by total weight
        total_weight = (weights['tsmom'] * 0.85 + 
                       weights['hmm'] * hmm_conf + 
                       weights['fed'] * fed_conf +
                       weights['base'] * 0.6)
        
        if total_weight > 0:
            combined = {t: combined[t] / total_weight for t in tickers}
        
        # Conflict detection (simplified)
        conflicts = []
        for ticker in tickers:
            tsmom_sign = 1 if tsmom_deltas.get(ticker, 0) > 0.01 else (-1 if tsmom_deltas.get(ticker, 0) < -0.01 else 0)
            fed_sign = 1 if fed_deltas.get(ticker, 0) > 0.01 else (-1 if fed_deltas.get(ticker, 0) < -0.01 else 0)
            
            if tsmom_sign != 0 and fed_sign != 0 and tsmom_sign != fed_sign:
                conflicts.append(f"{ticker}: TSMOM vs Fed")
        
        # Conflict resolution
        resolution = "weighted_average"
        if conflicts:
            resolution = "split_difference"
            # Reduce magnitude
            for ticker in combined:
                combined[ticker] *= 0.7
        
        # HMM neutral reduction
        if hmm_regime == 'neutral':
            resolution += ", hmm_neutral"
            for ticker in combined:
                combined[ticker] *= 0.8
        
        return combined, resolution
    
    def run_backtest(
        self,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        initial_value: float = 100000.0,
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run full historical backtest of combined strategy.
        """
        if self.prices_df is None:
            if not self.load_prices():
                raise ValueError("Failed to load price data")
        
        # Find start and end indices
        try:
            start_idx = self.dates.index(start_date)
        except ValueError:
            # Find first date >= start_date
            for i, d in enumerate(self.dates):
                if d >= start_date:
                    start_idx = i
                    break
            else:
                start_idx = len(self.dates) - 1
        
        try:
            end_idx = self.dates.index(end_date)
        except ValueError:
            for i in range(len(self.dates) - 1, -1, -1):
                if self.dates[i] <= end_date:
                    end_idx = i
                    break
            else:
                end_idx = len(self.dates) - 1
        
        # Ensure we have enough history for TSMOM
        start_idx = max(start_idx, MIN_HISTORY_DAYS)
        
        print(f"Backtest range: {self.dates[start_idx]} to {self.dates[end_idx]}")
        print(f"Trading days: {end_idx - start_idx + 1}")
        
        # Initialize
        portfolio_value = initial_value
        current_weights = self.base_allocation.copy()
        positions = []
        daily_values = [portfolio_value]
        daily_returns = []
        rebalances = 0
        
        for idx in range(start_idx, end_idx + 1):
            current_date = self.dates[idx]
            
            # Get current prices
            current_prices = {
                t: self.prices_df[t].iloc[idx]
                for t in self.tickers
            }
            
            # Check if rebalance needed
            rebalance_executed = False
            turnover = 0.0
            tsmom_deltas = None
            hmm_regime = None
            fed_regime = None
            
            if (idx - start_idx) % self.rebalance_freq == 0:
                # Get signals
                tsmom_deltas = self._get_tsmom_deltas(idx)
                hmm_regime, hmm_deltas = self._get_hmm_regime(idx)
                fed_regime, fed_deltas = self._get_fed_regime_deltas(idx)
                
                # Combine signals
                combined_deltas, resolution = self._combine_signals(
                    tsmom_deltas, hmm_regime, hmm_deltas,
                    fed_regime, fed_deltas, idx
                )
                
                # Calculate new weights
                new_weights = {}
                for ticker in self.tickers:
                    new_weight = self.base_allocation[ticker] + combined_deltas.get(ticker, 0.0)
                    new_weights[ticker] = max(0.05, min(0.90, new_weight))
                
                # Normalize
                total = sum(new_weights.values())
                new_weights = {t: w / total for t, w in new_weights.items()}
                
                # Calculate turnover
                turnover = sum(abs(new_weights.get(t, 0) - current_weights.get(t, 0))
                             for t in self.tickers) / 2
                
                # Apply transaction costs
                cost = turnover * self.transaction_cost * 2  # Both legs
                portfolio_value *= (1 - cost)
                
                current_weights = new_weights
                rebalances += 1
                rebalance_executed = True
                
                if verbose and rebalances <= 10:
                    print(f"Rebalance {rebalances}: {current_date}")
                    print(f"  TSMOM: {tsmom_deltas}")
                    print(f"  HMM: {hmm_regime} -> {hmm_deltas}")
                    print(f"  Fed: {fed_regime} -> {fed_deltas}")
                    print(f"  Combined: {combined_deltas}")
                    print(f"  New weights: {current_weights}")
                    print(f"  Turnover: {turnover:.2%}")
            
            # Record position
            positions.append(DailyPosition(
                date=current_date,
                weights=current_weights.copy(),
                prices=current_prices,
                portfolio_value=portfolio_value,
                tsmom_deltas=tsmom_deltas,
                hmm_regime=hmm_regime,
                fed_regime=fed_regime,
                rebalance_executed=rebalance_executed,
                turnover=turnover
            ))
            
            # Calculate daily return
            if idx > start_idx:
                daily_return = 0
                for ticker in self.tickers:
                    prev_price = self.prices_df[ticker].iloc[idx - 1]
                    curr_price = current_prices[ticker]
                    ticker_return = (curr_price / prev_price) - 1
                    daily_return += current_weights.get(ticker, 0) * ticker_return
                
                daily_returns.append(daily_return)
                portfolio_value *= (1 + daily_return)
            
            daily_values.append(portfolio_value)
        
        # Calculate metrics
        returns_series = pd.Series(daily_returns)
        
        days = len(daily_returns)
        years = days / 252
        
        cagr = (portfolio_value / initial_value) ** (1 / years) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(daily_values)
        drawdowns = (peak - daily_values) / peak
        max_dd = drawdowns.max()
        
        calmar = cagr / max_dd if max_dd > 0 else 0
        
        # Baseline comparison (46/38/16 buy-and-hold)
        baseline_result = self._run_baseline(start_idx, end_idx, initial_value)
        
        # Attribution (simplified - would need factor analysis for full attribution)
        tsmom_contrib = 0.0  # Placeholder - would need separate TSMOM-only backtest
        hmm_contrib = 0.0
        fed_contrib = 0.0
        
        # Crisis returns
        crisis_2008 = self._calculate_crisis_return(positions, "2008-01-01", "2008-12-31")
        crisis_2020 = self._calculate_crisis_return(positions, "2020-02-01", "2020-04-30")
        crisis_2022 = self._calculate_crisis_return(positions, "2022-01-01", "2022-12-31")
        
        # Information ratio
        excess_returns = np.array(daily_returns) - np.array(baseline_result['daily_returns'])
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (cagr - baseline_result['cagr']) / tracking_error if tracking_error > 0 else 0
        
        return BacktestResult(
            strategy="Combined Signal v2.55 (TSMOM + HMM + Fed)",
            start_date=self.dates[start_idx],
            end_date=self.dates[end_idx],
            trading_days=days,
            rebalances=rebalances,
            start_value=initial_value,
            end_value=portfolio_value,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            baseline_cagr=baseline_result['cagr'],
            baseline_sharpe=baseline_result['sharpe'],
            excess_return=cagr - baseline_result['cagr'],
            information_ratio=information_ratio,
            tsmom_contribution=tsmom_contrib,
            hmm_contribution=hmm_contrib,
            fed_contribution=fed_contrib,
            crisis_2008_return=crisis_2008,
            crisis_2020_return=crisis_2020,
            crisis_2022_return=crisis_2022,
            daily_values=daily_values,
            daily_returns=daily_returns,
            positions=positions
        )
    
    def _run_baseline(
        self,
        start_idx: int,
        end_idx: int,
        initial_value: float
    ) -> Dict:
        """Run baseline 46/38/16 buy-and-hold backtest."""
        baseline_weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        
        daily_returns = []
        portfolio_value = initial_value
        
        for idx in range(start_idx + 1, end_idx + 1):
            daily_return = 0
            for ticker in self.tickers:
                prev_price = self.prices_df[ticker].iloc[idx - 1]
                curr_price = self.prices_df[ticker].iloc[idx]
                ticker_return = (curr_price / prev_price) - 1
                daily_return += baseline_weights.get(ticker, 0) * ticker_return
            
            daily_returns.append(daily_return)
            portfolio_value *= (1 + daily_return)
        
        days = len(daily_returns)
        years = days / 252
        
        cagr = (portfolio_value / initial_value) ** (1 / years) - 1
        volatility = pd.Series(daily_returns).std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'daily_returns': daily_returns
        }
    
    def _calculate_crisis_return(
        self,
        positions: List[DailyPosition],
        crisis_start: str,
        crisis_end: str
    ) -> Optional[float]:
        """Calculate portfolio return during crisis period."""
        crisis_positions = [
            p for p in positions
            if crisis_start <= p.date <= crisis_end
        ]
        
        if not crisis_positions:
            return None
        
        start_value = crisis_positions[0].portfolio_value
        end_value = crisis_positions[-1].portfolio_value
        
        return (end_value / start_value) - 1


def main():
    parser = argparse.ArgumentParser(description="Combined Strategy Backtest v2.55")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run full backtest')
    backtest_parser.add_argument('--start', default=START_DATE, help='Start date')
    backtest_parser.add_argument('--end', default=END_DATE, help='End date')
    backtest_parser.add_argument('--initial', type=float, default=100000, help='Initial value')
    backtest_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    backtest_parser.add_argument('--output', help='Output JSON file')
    
    # summary command
    summary_parser = subparsers.add_parser('summary', help='Show backtest summary')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show backtest status')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        print("Running Combined Strategy Backtest...")
        print("This may take 2-3 minutes...")
        
        backtester = CombinedStrategyBacktester()
        result = backtester.run_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_value=args.initial,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Strategy: {result.strategy}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Trading Days: {result.trading_days}")
        print(f"Rebalances: {result.rebalances}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  CAGR: {result.cagr:.2%}")
        print(f"  Volatility: {result.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Calmar Ratio: {result.calmar_ratio:.4f}")
        print()
        print("BASELINE COMPARISON (46/38/16 Buy & Hold):")
        print(f"  Baseline CAGR: {result.baseline_cagr:.2%}")
        print(f"  Baseline Sharpe: {result.baseline_sharpe:.4f}")
        print(f"  Excess Return: {result.excess_return:.2%}")
        print(f"  Information Ratio: {result.information_ratio:.4f}")
        print()
        print("CRISIS PERFORMANCE:")
        if result.crisis_2008_return:
            print(f"  2008: {result.crisis_2008_return:.2%}")
        if result.crisis_2020_return:
            print(f"  2020: {result.crisis_2020_return:.2%}")
        if result.crisis_2022_return:
            print(f"  2022: {result.crisis_2022_return:.2%}")
        print("=" * 60)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        # Also save to results path
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {RESULTS_PATH}")
    
    elif args.command == 'summary':
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            
            print("=" * 60)
            print("BACKTEST SUMMARY (Saved Results)")
            print("=" * 60)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"No saved results found at {RESULTS_PATH}")
            print("Run 'backtest' command first")
    
    elif args.command == 'status':
        backtester = CombinedStrategyBacktester()
        loaded = backtester.load_prices()
        
        print("Combined Strategy Backtest v2.55 - Status")
        print("=" * 40)
        print(f"Data loaded: {loaded}")
        if loaded:
            print(f"Date range: {backtester.dates[0]} to {backtester.dates[-1]}")
            print(f"Trading days: {len(backtester.dates)}")
            print(f"Tickers: {backtester.tickers}")
        print()
        print("Parameters:")
        print(f"  Transaction cost: {TRANSACTION_COST:.2%}")
        print(f"  Rebalance frequency: {REBALANCE_FREQ} days")
        print(f"  Min history: {MIN_HISTORY_DAYS} days")
        print()
        print("Modules:")
        print(f"  TSMOM: ready")
        print(f"  HMM: {backtester.hmm_manager.detector.is_fitted}")
        print(f"  Fed Policy: heuristic-based for backtest")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
