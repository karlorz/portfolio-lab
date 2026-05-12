#!/usr/bin/env python3
"""
Portfolio-Lab v2.52: Time-Series Momentum (TSMOM) Overlay

AQR-Style Time-Series Momentum implementation based on:
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
- Hurst, Ooi, Pedersen (2017): "A Century of Evidence on Trend-Following Investing"
- Brooks et al. (2024): "Economic Trend" - fundamental momentum complement

Signal Generation:
    Signal_i(t) = sign(Return_i(t-12m to t-1m))  # 12-month formation, skip 1 month

Volatility Scaling:
    Position_i(t) = Signal_i(t) / σ_i(t)         # Equal risk contribution
    σ_i(t) = 20-day realized volatility (annualized)

Allocation Overlay:
    - Base allocation: 46/38/16 (SPY/GLD/TLT)
    - Max deviation: ±10% per asset
    - Min weight: 5% per asset
    - Rebalance frequency: Monthly or drift-based (10% threshold)

Performance Target:
    - Baseline Sharpe: 0.79 (46/38/16)
    - TSMOM Target: 0.88 (+0.09 improvement)
    - Crisis performance: Positive returns in 2008, 2020, 2022 stress

Usage:
    python -m src.signals.tsmom_overlay compute --ticker SPY --days 252
    python -m src.signals.tsmom_overlay backtest --portfolio 46/38/16
    python -m src.signals.tsmom_overlay live --portfolio 46/38/16
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
import sqlite3

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.signals.integrator import SignalIntegrator, CompositeSignal, SignalSourceResult


# Constants
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

# TSMOM Parameters
LOOKBACK_DAYS = 252        # 12 months (trading days)
SKIP_DAYS = 21             # Skip most recent month (avoid short-term reversal)
VOL_WINDOW = 20            # 20-day volatility estimation
VOL_TARGET = 0.15          # 15% target volatility (annualized)
MAX_DEVIATION = 0.10       # ±10% max deviation from base allocation
MIN_WEIGHT = 0.05          # Minimum 5% per asset
REBALANCE_FREQ = 21        # Monthly rebalancing (or drift-based)

# Asset mapping
ASSET_TICKERS = {
    'SPY': 'SPY',    # US Equities
    'GLD': 'GLD',    # Gold
    'TLT': 'TLT',    # Long Treasuries
    'CASH': 'CASH',  # Cash/money market
}

DEFAULT_BASE_ALLOCATION = {
    'SPY': 0.46,
    'GLD': 0.38,
    'TLT': 0.16,
    'CASH': 0.0,
}


@dataclass
class TSMOMSignal:
    """Time-series momentum signal for an asset."""
    ticker: str
    timestamp: str
    
    # Raw momentum
    lookback_return: float        # Return over lookback period (excl skip)
    recent_return: float          # Return over skip period
    signal: int                   # -1, 0, or +1
    
    # Risk scaling
    realized_vol: float           # 20-day annualized volatility
    vol_scaled_position: float    # Signal / volatility
    
    # Allocation adjustment
    base_weight: float            # Base allocation weight
    adjustment: float             # TSMOM adjustment (-10% to +10%)
    target_weight: float          # Base + adjustment
    
    # Metadata
    lookback_start_price: float
    lookback_end_price: float
    formation_days: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TSMOMPortfolio:
    """TSMOM-adjusted portfolio allocation."""
    timestamp: str
    base_allocation: Dict[str, float]
    tsmom_adjustments: Dict[str, float]
    target_allocation: Dict[str, float]
    
    # Risk metrics
    predicted_volatility: float
    max_drawdown_estimate: float
    
    # Consensus
    tsmom_signals: Dict[str, TSMOMSignal]
    overall_confidence: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'base_allocation': self.base_allocation,
            'tsmom_adjustments': self.tsmom_adjustments,
            'target_allocation': self.target_allocation,
            'predicted_volatility': self.predicted_volatility,
            'max_drawdown_estimate': self.max_drawdown_estimate,
            'overall_confidence': self.overall_confidence,
            'tsmom_signals': {k: v.to_dict() for k, v in self.tsmom_signals.items()}
        }


class TSMOMOverlay:
    """
    Time-Series Momentum overlay for portfolio allocation.
    
    Implements AQR-style TSMOM with:
    - 12-month formation period
    - 1-month skip (avoid short-term reversal)
    - 20-day volatility scaling
    - Position limits relative to base allocation
    """
    
    def __init__(
        self,
        lookback_days: int = LOOKBACK_DAYS,
        skip_days: int = SKIP_DAYS,
        vol_window: int = VOL_WINDOW,
        max_deviation: float = MAX_DEVIATION,
        min_weight: float = MIN_WEIGHT,
        data_source: str = "yahoo"
    ):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.vol_window = vol_window
        self.max_deviation = max_deviation
        self.min_weight = min_weight
        self.data_source = data_source
        
        # Cache for price data
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.signal_history: List[TSMOMSignal] = []
        
    def load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data for a ticker."""
        if ticker in self.price_cache:
            return self.price_cache[ticker]
        
        # Try to load from prices.json
        if PRICES_PATH.exists():
            try:
                with open(PRICES_PATH) as f:
                    data = json.load(f)
                
                if ticker in data:
                    ticker_data = data[ticker]
                    # Format: [{'d': '2005-01-03', 'p': 81.38}, ...]
                    if isinstance(ticker_data, list) and len(ticker_data) > 0:
                        dates = [item['d'] for item in ticker_data]
                        prices = [item['p'] for item in ticker_data]
                        df = pd.DataFrame({
                            'date': pd.to_datetime(dates),
                            'close': prices
                        })
                        df.set_index('date', inplace=True)
                        self.price_cache[ticker] = df
                        return df
            except Exception as e:
                print(f"Error loading prices for {ticker}: {e}")
        
        return None
    
    def calculate_formation_return(
        self,
        prices: pd.Series,
        current_idx: int
    ) -> Tuple[float, float, float, int]:
        """
        Calculate return over formation period (excl. skip).
        
        Returns: (formation_return, start_price, end_price, actual_days)
        """
        if current_idx < self.lookback_days + self.skip_days:
            return 0.0, prices.iloc[0], prices.iloc[current_idx], current_idx
        
        # Formation period: [t-12m, t-1m]
        start_idx = current_idx - self.lookback_days - self.skip_days
        end_idx = current_idx - self.skip_days
        
        start_price = prices.iloc[start_idx]
        end_price = prices.iloc[end_idx]
        
        formation_return = (end_price - start_price) / start_price
        actual_days = end_idx - start_idx
        
        return formation_return, start_price, end_price, actual_days
    
    def calculate_realized_volatility(
        self,
        prices: pd.Series,
        current_idx: int
    ) -> float:
        """
        Calculate realized volatility over vol_window.
        Annualized, assuming 252 trading days.
        """
        if current_idx < self.vol_window + 1:
            return 0.15  # Default to 15%
        
        recent_prices = prices.iloc[current_idx - self.vol_window:current_idx + 1]
        returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
        
        vol = returns.std() * np.sqrt(252)
        return max(vol, 0.01)  # Min 1% vol to avoid division by zero
    
    def compute_signal(self, ticker: str, timestamp: Optional[str] = None) -> Optional[TSMOMSignal]:
        """
        Compute TSMOM signal for a single asset.
        
        Implements AQR TSMOM:
        Signal = sign(formation_return)
        Position = Signal / volatility
        """
        df = self.load_prices(ticker)
        if df is None or len(df) < self.lookback_days + self.skip_days:
            return None
        
        prices = df['close']
        current_idx = len(prices) - 1
        
        # Formation return (excl. skip)
        formation_return, start_price, end_price, actual_days = \
            self.calculate_formation_return(prices, current_idx)
        
        # Recent return (skip period)
        if current_idx >= self.skip_days:
            skip_start_price = prices.iloc[current_idx - self.skip_days]
            current_price = prices.iloc[current_idx]
            recent_return = (current_price - skip_start_price) / skip_start_price
        else:
            recent_return = 0.0
        
        # Signal
        if abs(formation_return) < 0.001:  # Threshold for neutrality
            signal = 0
        else:
            signal = 1 if formation_return > 0 else -1
        
        # Volatility scaling
        realized_vol = self.calculate_realized_volatility(prices, current_idx)
        vol_scaled_position = signal / realized_vol if realized_vol > 0 else 0
        
        # Base weight
        base_weight = DEFAULT_BASE_ALLOCATION.get(ticker, 0.25)
        
        # Adjustment: scale by max deviation
        # Normalize vol_scaled_position to [-1, 1] for adjustment calculation
        position_normalized = np.clip(vol_scaled_position * 0.15, -1, 1)
        adjustment = position_normalized * self.max_deviation
        
        # Target weight with bounds
        target_weight = base_weight + adjustment
        target_weight = max(self.min_weight, min(0.95, target_weight))
        
        # Final adjustment after bounds
        adjustment = target_weight - base_weight
        
        timestamp = timestamp or datetime.now().isoformat()
        
        return TSMOMSignal(
            ticker=ticker,
            timestamp=timestamp,
            lookback_return=formation_return,
            recent_return=recent_return,
            signal=signal,
            realized_vol=realized_vol,
            vol_scaled_position=vol_scaled_position,
            base_weight=base_weight,
            adjustment=adjustment,
            target_weight=target_weight,
            lookback_start_price=start_price,
            lookback_end_price=end_price,
            formation_days=actual_days
        )
    
    def compute_portfolio(
        self,
        tickers: List[str] = None,
        base_allocation: Dict[str, float] = None,
        timestamp: Optional[str] = None
    ) -> Optional[TSMOMPortfolio]:
        """
        Compute TSMOM-adjusted portfolio allocation.
        """
        tickers = tickers or ['SPY', 'GLD', 'TLT']
        base_allocation = base_allocation or DEFAULT_BASE_ALLOCATION.copy()
        timestamp = timestamp or datetime.now().isoformat()
        
        # Compute signals for all assets
        signals = {}
        adjustments = {}
        target_allocs = {}
        
        for ticker in tickers:
            signal = self.compute_signal(ticker, timestamp)
            if signal:
                signals[ticker] = signal
                adjustments[ticker] = signal.adjustment
                target_allocs[ticker] = signal.target_weight
        
        if not signals:
            return None
        
        # Normalize to sum to 1.0 (accounting for CASH)
        total_weight = sum(target_allocs.values())
        if total_weight < 1.0:
            target_allocs['CASH'] = 1.0 - total_weight
        else:
            # Normalize down if over-allocated
            scale = 0.95 / total_weight
            for k in target_allocs:
                target_allocs[k] *= scale
            target_allocs['CASH'] = 0.05
        
        # Calculate predicted portfolio volatility
        # Simplified: weighted average of individual vols
        pred_vol = sum(
            s.realized_vol * target_allocs.get(t, 0)
            for t, s in signals.items()
        )
        
        # Max drawdown estimate (simplified)
        # Based on historical TSMOM drawdown characteristics
        max_dd_estimate = -0.15 if all(s.signal >= 0 for s in signals.values()) else -0.20
        
        # Confidence based on signal consistency
        signal_values = [s.signal for s in signals.values()]
        if len(signal_values) >= 2:
            # Higher confidence when signals agree
            agreement = abs(sum(signal_values)) / len(signal_values)
            confidence = 0.5 + agreement * 0.5
        else:
            confidence = 0.5
        
        return TSMOMPortfolio(
            timestamp=timestamp,
            base_allocation=base_allocation,
            tsmom_adjustments=adjustments,
            target_allocation=target_allocs,
            predicted_volatility=pred_vol,
            max_drawdown_estimate=max_dd_estimate,
            tsmom_signals=signals,
            overall_confidence=confidence
        )
    
    def get_current_recommendation(
        self,
        base_allocation: Dict[str, float] = None
    ) -> Dict:
        """Get current TSMOM recommendation."""
        portfolio = self.compute_portfolio(base_allocation=base_allocation)
        if portfolio is None:
            return {"error": "Unable to compute TSMOM signals"}
        
        # Calculate deltas
        deltas = {
            ticker: portfolio.target_allocation.get(ticker, 0) - base_weight
            for ticker, base_weight in portfolio.base_allocation.items()
        }
        
        return {
            "strategy": "TSMOM Overlay v2.52",
            "timestamp": portfolio.timestamp,
            "base_allocation": portfolio.base_allocation,
            "tsmom_allocation": portfolio.target_allocation,
            "deltas": deltas,
            "predicted_volatility": round(portfolio.predicted_volatility, 4),
            "confidence": round(portfolio.overall_confidence, 4),
            "signals": {
                ticker: {
                    "formation_return": round(s.lookback_return, 4),
                    "signal": s.signal,
                    "volatility": round(s.realized_vol, 4),
                    "adjustment": round(s.adjustment, 4)
                }
                for ticker, s in portfolio.tsmom_signals.items()
            }
        }


class TSMOMBacktester:
    """
    Backtesting engine for TSMOM overlay strategy.
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        base_allocation: Dict[str, float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        transaction_cost: float = 0.001  # 10 bps
    ):
        self.tickers = tickers or ['SPY', 'GLD', 'TLT']
        self.base_allocation = base_allocation or DEFAULT_BASE_ALLOCATION.copy()
        self.start_date = start_date
        self.end_date = end_date
        self.transaction_cost = transaction_cost
        
        self.overlay = TSMOMOverlay()
    
    def run_backtest(self, rebalance_freq: int = 21) -> Dict:
        """
        Run full historical backtest.
        """
        # Load all price data
        prices_df = self._load_all_prices()
        if prices_df is None or len(prices_df) < LOOKBACK_DAYS + SKIP_DAYS + 100:
            return {"error": "Insufficient price data"}
        
        # Filter dates
        if self.start_date:
            prices_df = prices_df[prices_df.index >= self.start_date]
        if self.end_date:
            prices_df = prices_df[prices_df.index <= self.end_date]
        
        if len(prices_df) < 100:
            return {"error": "Insufficient data after date filtering"}
        
        # Initialize portfolio
        portfolio_values = [100000.0]
        current_weights = self.base_allocation.copy()
        rebalance_dates = []
        
        # Run simulation
        for i in range(LOOKBACK_DAYS + SKIP_DAYS, len(prices_df)):
            date = prices_df.index[i]
            
            # Check if rebalance needed
            if (i - LOOKBACK_DAYS - SKIP_DAYS) % rebalance_freq == 0:
                # Compute TSMOM signals
                signals = self._compute_signals_at_date(prices_df, i)
                if signals:
                    new_weights = self._weights_from_signals(signals)
                    
                    # Calculate turnover
                    turnover = sum(abs(new_weights.get(t, 0) - current_weights.get(t, 0))
                                 for t in self.tickers + ['CASH']) / 2
                    
                    # Apply transaction costs
                    cost = turnover * self.transaction_cost * 2  # Both legs
                    portfolio_values[-1] *= (1 - cost)
                    
                    current_weights = new_weights
                    rebalance_dates.append({
                        'date': date.isoformat(),
                        'turnover': turnover,
                        'weights': new_weights.copy()
                    })
            
            # Calculate daily return
            daily_return = 0
            for ticker in self.tickers:
                if ticker in prices_df.columns:
                    ticker_return = prices_df.iloc[i][ticker] / prices_df.iloc[i-1][ticker] - 1
                    daily_return += current_weights.get(ticker, 0) * ticker_return
            
            # Cash return (assume 0 for simplicity, or add T-bill rates)
            daily_return += current_weights.get('CASH', 0) * 0.0
            
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        # Calculate metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        max_dd = drawdowns.max()
        
        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else 0
        
        return {
            "strategy": "TSMOM Overlay Backtest v2.52",
            "start_date": prices_df.index[LOOKBACK_DAYS + SKIP_DAYS].isoformat(),
            "end_date": prices_df.index[-1].isoformat(),
            "trading_days": len(returns),
            "rebalances": len(rebalance_dates),
            "start_value": portfolio_values[0],
            "end_value": portfolio_values[-1],
            "cagr": round(cagr, 4),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "rebalance_history": rebalance_dates[-10:],  # Last 10
            "parameters": {
                "lookback_days": LOOKBACK_DAYS,
                "skip_days": SKIP_DAYS,
                "vol_window": VOL_WINDOW,
                "max_deviation": MAX_DEVIATION,
                "transaction_cost_bps": self.transaction_cost * 10000
            }
        }
    
    def _load_all_prices(self) -> Optional[pd.DataFrame]:
        """Load and align price data for all tickers."""
        all_prices = {}
        
        for ticker in self.tickers:
            df = self.overlay.load_prices(ticker)
            if df is not None:
                all_prices[ticker] = df['close']
        
        if not all_prices:
            return None
        
        # Combine into single DataFrame
        prices_df = pd.DataFrame(all_prices)
        prices_df.dropna(inplace=True)
        
        return prices_df
    
    def _compute_signals_at_date(
        self,
        prices_df: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, TSMOMSignal]:
        """Compute TSMOM signals at a specific date index."""
        signals = {}
        timestamp = prices_df.index[current_idx].isoformat()
        
        for ticker in self.tickers:
            if ticker not in prices_df.columns:
                continue
            
            prices = prices_df[ticker]
            
            # Check if enough history
            if current_idx < LOOKBACK_DAYS + SKIP_DAYS:
                continue
            
            # Formation return
            start_idx = current_idx - LOOKBACK_DAYS - SKIP_DAYS
            end_idx = current_idx - SKIP_DAYS
            
            start_price = prices.iloc[start_idx]
            end_price = prices.iloc[end_idx]
            formation_return = (end_price - start_price) / start_price
            
            # Signal
            signal = 0 if abs(formation_return) < 0.001 else (1 if formation_return > 0 else -1)
            
            # Volatility
            if current_idx >= VOL_WINDOW:
                recent_returns = np.log(prices.iloc[current_idx-VOL_WINDOW:current_idx+1] / 
                                       prices.iloc[current_idx-VOL_WINDOW-1:current_idx])
                vol = recent_returns.std() * np.sqrt(252)
            else:
                vol = 0.15
            
            vol = max(vol, 0.01)
            
            # Adjustment
            position_normalized = np.clip(signal / vol * 0.15, -1, 1)
            adjustment = position_normalized * MAX_DEVIATION
            
            base_weight = self.base_allocation.get(ticker, 0.25)
            target_weight = max(MIN_WEIGHT, min(0.95, base_weight + adjustment))
            
            signals[ticker] = TSMOMSignal(
                ticker=ticker,
                timestamp=timestamp,
                lookback_return=formation_return,
                recent_return=0.0,
                signal=signal,
                realized_vol=vol,
                vol_scaled_position=signal / vol,
                base_weight=base_weight,
                adjustment=target_weight - base_weight,
                target_weight=target_weight,
                lookback_start_price=start_price,
                lookback_end_price=end_price,
                formation_days=LOOKBACK_DAYS
            )
        
        return signals
    
    def _weights_from_signals(self, signals: Dict[str, TSMOMSignal]) -> Dict[str, float]:
        """Convert signals to portfolio weights."""
        weights = {ticker: s.target_weight for ticker, s in signals.items()}
        
        # Add cash
        total = sum(weights.values())
        weights['CASH'] = max(0, 1.0 - total)
        
        # Normalize if over-allocated
        if total > 1.0:
            scale = 0.95 / total
            for k in weights:
                if k != 'CASH':
                    weights[k] *= scale
            weights['CASH'] = 0.05
        
        return weights


def main():
    parser = argparse.ArgumentParser(description="TSMOM Overlay v2.52")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # compute command
    compute_parser = subparsers.add_parser('compute', help='Compute TSMOM signal for a ticker')
    compute_parser.add_argument('--ticker', required=True, help='Ticker symbol (e.g., SPY)')
    compute_parser.add_argument('--days', type=int, default=252, help='Lookback days')
    compute_parser.add_argument('--output', help='Output JSON file')
    
    # backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run TSMOM backtest')
    backtest_parser.add_argument('--portfolio', default='46/38/16', 
                                help='Base allocation (e.g., 46/38/16)')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--freq', type=int, default=21, help='Rebalance frequency (days)')
    backtest_parser.add_argument('--output', help='Output JSON file')
    
    # live command
    live_parser = subparsers.add_parser('live', help='Get current TSMOM recommendation')
    live_parser.add_argument('--portfolio', default='46/38/16', help='Base allocation')
    live_parser.add_argument('--output', help='Output JSON file')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show TSMOM status')
    
    args = parser.parse_args()
    
    if args.command == 'compute':
        overlay = TSMOMOverlay(lookback_days=args.days)
        signal = overlay.compute_signal(args.ticker)
        if signal:
            result = signal.to_dict()
            print(json.dumps(result, indent=2))
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
        else:
            print(json.dumps({"error": f"Could not compute signal for {args.ticker}"}))
    
    elif args.command == 'backtest':
        # Parse allocation
        parts = args.portfolio.split('/')
        base_alloc = {
            'SPY': float(parts[0]) / 100 if float(parts[0]) > 1 else float(parts[0]),
            'GLD': float(parts[1]) / 100 if float(parts[1]) > 1 else float(parts[1]),
            'TLT': float(parts[2]) / 100 if float(parts[2]) > 1 else float(parts[2]),
        }
        base_alloc['CASH'] = 0.0
        
        backtester = TSMOMBacktester(
            base_allocation=base_alloc,
            start_date=args.start,
            end_date=args.end
        )
        result = backtester.run_backtest(rebalance_freq=args.freq)
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.command == 'live':
        parts = args.portfolio.split('/')
        base_alloc = {
            'SPY': float(parts[0]) / 100 if float(parts[0]) > 1 else float(parts[0]),
            'GLD': float(parts[1]) / 100 if float(parts[1]) > 1 else float(parts[1]),
            'TLT': float(parts[2]) / 100 if float(parts[2]) > 1 else float(parts[2]),
        }
        base_alloc['CASH'] = 0.0
        
        overlay = TSMOMOverlay()
        result = overlay.get_current_recommendation(base_alloc)
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.command == 'status':
        print("TSMOM Overlay v2.52 - Status")
        print("=" * 40)
        print(f"Lookback: {LOOKBACK_DAYS} days (12 months)")
        print(f"Skip: {SKIP_DAYS} days (1 month)")
        print(f"Volatility window: {VOL_WINDOW} days")
        print(f"Max deviation: {MAX_DEVIATION * 100}%")
        print(f"Min weight: {MIN_WEIGHT * 100}%")
        print(f"Target volatility: {VOL_TARGET * 100}%")
        print()
        print(f"Data source: {DATA_DIR}")
        print(f"Prices path: {PRICES_PATH}")
        print(f"Prices exist: {PRICES_PATH.exists()}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
