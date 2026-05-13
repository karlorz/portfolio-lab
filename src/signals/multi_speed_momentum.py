#!/usr/bin/env python3
"""
Portfolio-Lab v2.56: Multi-Speed Momentum Ensemble

Man AHL-style multi-horizon momentum based on:
- Man AHL "Trend Following Deep Dive: Dynamics of Dispersion" (Sept 2025)
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
- Brooks et al. (2024): "Economic Trend" - fundamental momentum complement

Multi-Speed Ensemble Strategy:
    Speed diversification IS the edge - no single "best" design exists.
    Equal risk-weight across speed tiers (NOT optimized - intentional diversification)

Speed Tiers:
    Fast:   ~2-3 month horizon (crisis alpha, sharp turns)
    Medium: ~4 month horizon (balanced)  
    Slow:   ~6-12 month horizon (trend persistence)

Signal Generation (per tier):
    Signal_i(t, speed) = sign(Return_i(t-lookback_speed to t-skip))
    Position_i(t, speed) = Signal_i(t, speed) / σ_i(t, speed)

Ensemble Aggregation:
    Ensemble_Position = (Fast + Medium + Slow) / 3  # Equal risk-weight

Performance Target:
    - Baseline Sharpe: 0.93 (Combined v2.55 TSMOM+HMM+Fed)
    - Multi-Speed Target: 1.10 (+0.17 improvement)
    - Crisis performance: Better 2008, 2020, 2022 capture via speed diversification

Usage:
    python -m src.signals.multi_speed_momentum compute --ticker SPY --tier fast
    python -m src.signals.multi_speed_momentum backtest --portfolio 46/38/16
    python -m src.signals.multi_speed_momentum live --portfolio 46/38/16
    python -m src.signals.multi_speed_momentum status
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

# Multi-Speed Parameters (Man AHL-inspired speed tiers)
SPEED_TIERS = {
    'fast': {
        'lookback_days': 63,      # ~3 months
        'skip_days': 5,           # ~1 week
        'vol_window': 10,         # 10-day volatility
        'description': 'Fast momentum: crisis alpha, sharp turns'
    },
    'medium': {
        'lookback_days': 126,     # ~6 months
        'skip_days': 10,          # ~2 weeks
        'vol_window': 20,         # 20-day volatility
        'description': 'Medium momentum: balanced'
    },
    'slow': {
        'lookback_days': 252,     # ~12 months
        'skip_days': 21,          # ~1 month
        'vol_window': 20,         # 20-day volatility
        'description': 'Slow momentum: trend persistence'
    }
}

VOL_TARGET = 0.15              # 15% target volatility (annualized)
MAX_DEVIATION = 0.10           # ±10% max deviation from base allocation
MIN_WEIGHT = 0.05              # Minimum 5% per asset
REBALANCE_FREQ = 21            # Monthly rebalancing

ASSET_TICKERS = {
    'SPY': 'SPY',
    'GLD': 'GLD',
    'TLT': 'TLT',
    'DBC': 'DBC',  # Phase 2: Commodity diversification
    'CASH': 'CASH',
}

DEFAULT_BASE_ALLOCATION = {
    'SPY': 0.46,
    'GLD': 0.34,  # Reduced from 0.38 to accommodate DBC
    'TLT': 0.16,
    'DBC': 0.04,  # Phase 2: 4% commodity exposure
    'CASH': 0.0,
}


@dataclass
class SpeedMomentumSignal:
    """Momentum signal for a specific speed tier."""
    ticker: str
    tier: str  # fast, medium, slow
    timestamp: str
    
    # Raw momentum
    lookback_return: float
    recent_return: float
    signal: int  # -1, 0, or +1
    
    # Risk scaling
    realized_vol: float
    vol_scaled_position: float
    
    # Allocation adjustment
    base_weight: float
    adjustment: float
    target_weight: float
    
    # Metadata
    lookback_start_price: float
    lookback_end_price: float
    formation_days: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnsembleSignal:
    """Ensemble aggregation across speed tiers."""
    ticker: str
    timestamp: str
    
    # Per-tier signals
    fast_signal: SpeedMomentumSignal
    medium_signal: SpeedMomentumSignal
    slow_signal: SpeedMomentumSignal
    
    # Ensemble aggregation (equal risk-weight)
    ensemble_position: float  # Average of vol-scaled positions
    ensemble_confidence: float  # Agreement across tiers
    
    # Allocation adjustment
    base_weight: float
    adjustment: float
    target_weight: float
    
    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'fast_signal': self.fast_signal.to_dict(),
            'medium_signal': self.medium_signal.to_dict(),
            'slow_signal': self.slow_signal.to_dict(),
            'ensemble_position': self.ensemble_position,
            'ensemble_confidence': self.ensemble_confidence,
            'base_weight': self.base_weight,
            'adjustment': self.adjustment,
            'target_weight': self.target_weight,
        }


@dataclass
class MultiSpeedPortfolio:
    """Multi-speed momentum-adjusted portfolio allocation."""
    timestamp: str
    base_allocation: Dict[str, float]
    ensemble_adjustments: Dict[str, float]
    target_allocation: Dict[str, float]
    
    # Risk metrics
    predicted_volatility: float
    max_drawdown_estimate: float
    
    # Per-asset ensemble signals
    ensemble_signals: Dict[str, EnsembleSignal]
    
    # Speed tier contribution analysis
    tier_contributions: Dict[str, float]  # fast, medium, slow Sharpe contribution
    
    overall_confidence: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'base_allocation': self.base_allocation,
            'ensemble_adjustments': self.ensemble_adjustments,
            'target_allocation': self.target_allocation,
            'predicted_volatility': self.predicted_volatility,
            'max_drawdown_estimate': self.max_drawdown_estimate,
            'ensemble_signals': {k: v.to_dict() for k, v in self.ensemble_signals.items()},
            'tier_contributions': self.tier_contributions,
            'overall_confidence': self.overall_confidence,
        }


class MultiSpeedMomentum:
    """
    Multi-speed momentum ensemble overlay for portfolio allocation.
    
    Implements Man AHL-style speed diversification:
    - Fast: 2-3 month horizon (crisis alpha)
    - Medium: 4 month horizon (balanced)
    - Slow: 6-12 month horizon (trend persistence)
    
    Equal risk-weight across tiers (diversification IS the edge).
    """
    
    def __init__(
        self,
        prices_path: Path = PRICES_PATH,
        db_path: Path = DB_PATH,
        speed_tiers: Dict = SPEED_TIERS,
        vol_target: float = VOL_TARGET,
        max_deviation: float = MAX_DEVIATION,
        min_weight: float = MIN_WEIGHT
    ):
        self.prices_path = prices_path
        self.db_path = db_path
        self.speed_tiers = speed_tiers
        self.vol_target = vol_target
        self.max_deviation = max_deviation
        self.min_weight = min_weight
        self._prices_df: Optional[pd.DataFrame] = None
    
    def _load_prices(self) -> pd.DataFrame:
        """Load price data from JSON."""
        if self._prices_df is not None:
            return self._prices_df
        
        with open(self.prices_path, 'r') as f:
            data = json.load(f)
        
        records = []
        for symbol, entries in data.items():
            for entry in entries:
                records.append({
                    'date': entry['d'],
                    'ticker': symbol,
                    'price': entry['p']
                })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot(index='date', columns='ticker', values='price')
        df = df.sort_index()
        
        self._prices_df = df
        return df
    
    def load_prices(self) -> pd.DataFrame:
        """Public interface to load prices."""
        return self._load_prices()
    
    def compute_speed_signal(
        self,
        ticker: str,
        tier: str,
        base_weight: float,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[SpeedMomentumSignal]:
        """Compute momentum signal for a specific speed tier."""
        if prices_df is None:
            prices_df = self._load_prices()
        
        if ticker not in prices_df.columns:
            return None
        
        tier_config = self.speed_tiers[tier]
        lookback_days = tier_config['lookback_days']
        skip_days = tier_config['skip_days']
        vol_window = tier_config['vol_window']
        
        prices = prices_df[ticker].dropna()
        if len(prices) < lookback_days + skip_days + vol_window:
            return None
        
        current_price = prices.iloc[-1]
        current_date = prices.index[-1]
        
        # Lookback return (excluding skip period)
        lookback_start_idx = -(lookback_days + skip_days)
        lookback_end_idx = -skip_days
        lookback_start_price = prices.iloc[lookback_start_idx]
        lookback_end_price = prices.iloc[lookback_end_idx]
        lookback_return = (lookback_end_price / lookback_start_price) - 1
        
        # Recent return (skip period)
        if skip_days > 0:
            recent_start_price = prices.iloc[-skip_days]
            recent_return = (current_price / recent_start_price) - 1
        else:
            recent_return = 0.0
        
        # Signal (sign of lookback return)
        signal = int(np.sign(lookback_return)) if lookback_return != 0 else 0
        
        # Volatility scaling
        recent_prices = prices.iloc[-vol_window:]
        returns = recent_prices.pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.15
        vol_scaled_position = signal / realized_vol if realized_vol > 0 else 0
        
        # Allocation adjustment
        adjustment = np.clip(
            vol_scaled_position * 0.15,  # Scale to reasonable adjustment
            -self.max_deviation,
            self.max_deviation
        )
        target_weight = base_weight + adjustment
        target_weight = np.clip(target_weight, self.min_weight, 1.0)
        
        return SpeedMomentumSignal(
            ticker=ticker,
            tier=tier,
            timestamp=current_date.isoformat(),
            lookback_return=lookback_return,
            recent_return=recent_return,
            signal=signal,
            realized_vol=realized_vol,
            vol_scaled_position=vol_scaled_position,
            base_weight=base_weight,
            adjustment=adjustment,
            target_weight=target_weight,
            lookback_start_price=lookback_start_price,
            lookback_end_price=lookback_end_price,
            formation_days=lookback_days
        )
    
    def compute_ensemble_signal(
        self,
        ticker: str,
        base_weight: float,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[EnsembleSignal]:
        """Compute ensemble signal across all speed tiers."""
        if prices_df is None:
            prices_df = self._load_prices()
        
        # Compute signals for each tier
        fast_signal = self.compute_speed_signal(ticker, 'fast', base_weight, prices_df)
        medium_signal = self.compute_speed_signal(ticker, 'medium', base_weight, prices_df)
        slow_signal = self.compute_speed_signal(ticker, 'slow', base_weight, prices_df)
        
        if not all([fast_signal, medium_signal, slow_signal]):
            return None
        
        # Equal risk-weight ensemble (Man AHL: diversification IS the edge)
        ensemble_position = (
            fast_signal.vol_scaled_position +
            medium_signal.vol_scaled_position +
            slow_signal.vol_scaled_position
        ) / 3.0
        
        # Ensemble confidence (agreement across tiers)
        signals = [fast_signal.signal, medium_signal.signal, slow_signal.signal]
        if all(s == signals[0] for s in signals):
            ensemble_confidence = 1.0  # Full agreement
        elif sum(signals) == 0:
            ensemble_confidence = 0.0  # Maximum disagreement
        else:
            ensemble_confidence = 0.5  # Partial agreement
        
        # Allocation adjustment based on ensemble
        adjustment = np.clip(
            ensemble_position * 0.15,
            -self.max_deviation,
            self.max_deviation
        )
        target_weight = base_weight + adjustment
        target_weight = np.clip(target_weight, self.min_weight, 1.0)
        
        current_date = prices_df.index[-1]
        
        return EnsembleSignal(
            ticker=ticker,
            timestamp=current_date.isoformat(),
            fast_signal=fast_signal,
            medium_signal=medium_signal,
            slow_signal=slow_signal,
            ensemble_position=ensemble_position,
            ensemble_confidence=ensemble_confidence,
            base_weight=base_weight,
            adjustment=adjustment,
            target_weight=target_weight
        )
    
    def get_current_recommendation(
        self,
        base_allocation: Dict[str, float],
        include_cash: bool = False
    ) -> MultiSpeedPortfolio:
        """Get current multi-speed momentum portfolio recommendation."""
        prices_df = self._load_prices()
        current_date = prices_df.index[-1]
        
        ensemble_signals = {}
        ensemble_adjustments = {}
        target_allocation = {'CASH': 0.0}
        
        # Compute ensemble signals for each asset
        for ticker, base_weight in base_allocation.items():
            if ticker == 'CASH':
                continue
            
            ensemble_signal = self.compute_ensemble_signal(ticker, base_weight, prices_df)
            if ensemble_signal:
                ensemble_signals[ticker] = ensemble_signal
                ensemble_adjustments[ticker] = ensemble_signal.adjustment
                target_allocation[ticker] = ensemble_signal.target_weight
            else:
                target_allocation[ticker] = base_weight
                ensemble_adjustments[ticker] = 0.0
        
        # Normalize weights to sum to 1.0 (excluding cash)
        total_weight = sum(w for k, w in target_allocation.items() if k != 'CASH')
        if total_weight > 0:
            for ticker in target_allocation:
                if ticker != 'CASH':
                    target_allocation[ticker] /= total_weight
        
        # Estimate portfolio volatility (weighted average)
        predicted_vol = 0.0
        for ticker, weight in target_allocation.items():
            if ticker != 'CASH' and ticker in ensemble_signals:
                asset_vol = ensemble_signals[ticker].fast_signal.realized_vol
                predicted_vol += weight * asset_vol
        
        # Estimate max drawdown (simplified)
        max_dd_estimate = -predicted_vol * 2.5  # Rough 2.5-sigma estimate
        
        # Tier contributions (will be populated by backtester)
        tier_contributions = {'fast': 0.0, 'medium': 0.0, 'slow': 0.0}
        
        # Overall confidence (average of ensemble confidences)
        if ensemble_signals:
            overall_confidence = np.mean([s.ensemble_confidence for s in ensemble_signals.values()])
        else:
            overall_confidence = 0.0
        
        return MultiSpeedPortfolio(
            timestamp=current_date.isoformat(),
            base_allocation=base_allocation,
            ensemble_adjustments=ensemble_adjustments,
            target_allocation=target_allocation,
            predicted_volatility=predicted_vol,
            max_drawdown_estimate=max_dd_estimate,
            ensemble_signals=ensemble_signals,
            tier_contributions=tier_contributions,
            overall_confidence=overall_confidence
        )
    
    def get_ensemble_signal(self, ticker: str, prices_df: Optional[pd.DataFrame] = None) -> Optional[float]:
        """Get current ensemble signal value for a ticker (-1 to +1)."""
        prices_df = prices_df or self._load_prices()
        
        if ticker not in prices_df.columns:
            return None
        
        base_weight = 0.33  # Equal weight placeholder
        signal = self.compute_ensemble_signal(ticker, base_weight, prices_df)
        
        if signal:
            # Return normalized signal -1 to +1 based on adjustment and signal direction
            raw_signal = signal.adjustment / self.max_deviation if self.max_deviation else 0
            return np.clip(raw_signal, -1, 1)
        return 0.0

    def save_to_db(self, portfolio: MultiSpeedPortfolio):
        """Save ensemble recommendation to signals database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create table if not exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS multi_speed_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                base_allocation TEXT,
                target_allocation TEXT,
                adjustments TEXT,
                predicted_volatility REAL,
                overall_confidence REAL,
                tier_contributions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            INSERT INTO multi_speed_recommendations
            (timestamp, base_allocation, target_allocation, adjustments, 
             predicted_volatility, overall_confidence, tier_contributions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            portfolio.timestamp,
            json.dumps(portfolio.base_allocation),
            json.dumps(portfolio.target_allocation),
            json.dumps(portfolio.ensemble_adjustments),
            portfolio.predicted_volatility,
            portfolio.overall_confidence,
            json.dumps(portfolio.tier_contributions)
        ))
        
        conn.commit()
        conn.close()


class MultiSpeedBacktester:
    """Backtester for multi-speed momentum ensemble strategy."""
    
    def __init__(
        self,
        base_allocation: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: int = REBALANCE_FREQ
    ):
        self.base_allocation = base_allocation
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.rebalance_freq = rebalance_freq
        self.multi_speed = MultiSpeedMomentum()
        self.prices_df = self.multi_speed._load_prices()
    
    def run_backtest(self) -> Dict:
        """Run full historical backtest."""
        prices = self.prices_df.copy()
        
        if self.start_date:
            prices = prices[prices.index >= self.start_date]
        if self.end_date:
            prices = prices[prices.index <= self.end_date]
        
        # Need enough data for slow tier
        min_history = SPEED_TIERS['slow']['lookback_days'] + SPEED_TIERS['slow']['skip_days'] + 50
        if len(prices) < min_history:
            return {'error': f'Insufficient data: {len(prices)} days < {min_history} required'}
        
        # Portfolio tracking
        portfolio_value = 100000.0
        current_allocation = self.base_allocation.copy()
        
        daily_values = []
        rebalance_dates = []
        signals_history = []
        
        # Tier performance tracking
        tier_returns = {'fast': [], 'medium': [], 'slow': []}
        
        max_lookback = max(t['lookback_days'] + t['skip_days'] for t in SPEED_TIERS.values())
        
        for i in range(max_lookback, len(prices)):
            current_date = prices.index[i]
            history = prices.iloc[:i+1]
            
            # Monthly rebalancing
            if i == max_lookback or (i - max_lookback) % self.rebalance_freq == 0:
                self.multi_speed._prices_df = history
                
                recommendation = self.multi_speed.get_current_recommendation(
                    self.base_allocation
                )
                
                current_allocation = recommendation.target_allocation
                rebalance_dates.append({
                    'date': current_date.isoformat(),
                    'allocation': current_allocation.copy(),
                    'signals': {t: s.to_dict() for t, s in recommendation.ensemble_signals.items()}
                })
            
            # Calculate daily return
            daily_return = 0.0
            for ticker, weight in current_allocation.items():
                if ticker == 'CASH' or ticker not in prices.columns:
                    continue
                
                if i > 0:
                    ticker_return = prices[ticker].iloc[i] / prices[ticker].iloc[i-1] - 1
                    daily_return += weight * ticker_return
            
            portfolio_value *= (1 + daily_return)
            
            # Handle NaN
            if np.isnan(portfolio_value) or np.isinf(portfolio_value):
                portfolio_value = daily_values[-1]['value'] if daily_values else 100000.0
            
            daily_values.append({
                'date': current_date.isoformat(),
                'value': portfolio_value,
                'return': daily_return
            })
        
        # Calculate performance metrics
        df_values = pd.DataFrame(daily_values)
        df_values['date'] = pd.to_datetime(df_values['date'])
        df_values.set_index('date', inplace=True)
        
        returns = df_values['return'].dropna()
        
        # Annualized metrics
        start_val = float(df_values['value'].iloc[0])
        end_val = float(df_values['value'].iloc[-1])
        total_return = (end_val / start_val) - 1 if start_val > 0 else 0
        years = len(df_values) / 252
        cagr = ((end_val / start_val) ** (1/years)) - 1 if start_val > 0 and years > 0 else 0
        volatility = float(returns.std()) * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Crisis analysis
        crisis_periods = {
            '2008': ('2008-01-01', '2008-12-31'),
            '2020': ('2020-02-01', '2020-05-31'),
            '2022': ('2022-01-01', '2022-12-31'),
        }
        
        crisis_returns = {}
        for crisis, (start, end) in crisis_periods.items():
            try:
                period_df = df_values.loc[start:end]
                if not period_df.empty:
                    crisis_return = period_df['value'].iloc[-1] / period_df['value'].iloc[0] - 1
                    crisis_returns[crisis] = crisis_return
            except:
                crisis_returns[crisis] = None
        
        # Baseline comparison (static allocation)
        baseline_values = [100000.0]
        for i in range(max_lookback, len(prices)):
            daily_return = 0.0
            for ticker, weight in self.base_allocation.items():
                if ticker == 'CASH' or ticker not in prices.columns:
                    continue
                if i > 0:
                    ticker_return = float(prices[ticker].iloc[i]) / float(prices[ticker].iloc[i-1]) - 1
                    daily_return += weight * ticker_return
            new_value = baseline_values[-1] * (1 + daily_return)
            if np.isnan(new_value) or np.isinf(new_value):
                new_value = baseline_values[-1]
            baseline_values.append(new_value)
        
        baseline_total_return = (baseline_values[-1] / baseline_values[0]) - 1 if baseline_values[0] > 0 else 0
        baseline_cagr = ((baseline_values[-1] / baseline_values[0]) ** (1/years)) - 1 if baseline_values[0] > 0 and years > 0 else 0
        baseline_returns = pd.Series(baseline_values).pct_change().dropna()
        baseline_vol = float(baseline_returns.std()) * np.sqrt(252)
        baseline_sharpe = baseline_cagr / baseline_vol if baseline_vol > 0 else 0
        
        return {
            'strategy': 'Multi-Speed Momentum Ensemble v2.56',
            'start_date': prices.index[max_lookback].isoformat(),
            'end_date': prices.index[-1].isoformat(),
            'trading_days': len(df_values),
            'rebalances': len(rebalance_dates),
            'start_value': 100000,
            'end_value': portfolio_value,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'baseline_cagr': baseline_cagr,
            'baseline_sharpe': baseline_sharpe,
            'excess_return': cagr - baseline_cagr,
            'information_ratio': (cagr - baseline_cagr) / abs(volatility - baseline_vol) if volatility != baseline_vol else 0,
            'crisis_2008_return': crisis_returns.get('2008'),
            'crisis_2020_return': crisis_returns.get('2020'),
            'crisis_2022_return': crisis_returns.get('2022'),
            'tier_config': {k: {kk: vv for kk, vv in v.items() if kk != 'description'} 
                          for k, v in SPEED_TIERS.items()},
            'speed_tiers': list(SPEED_TIERS.keys()),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Speed Momentum Ensemble v2.56'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Compute command
    compute_parser = subparsers.add_parser('compute', help='Compute speed signal for ticker')
    compute_parser.add_argument('--ticker', required=True, help='Ticker symbol (SPY, GLD, TLT)')
    compute_parser.add_argument('--tier', choices=['fast', 'medium', 'slow'], default='medium',
                               help='Speed tier')
    compute_parser.add_argument('--output', help='Output JSON file')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run historical backtest')
    backtest_parser.add_argument('--portfolio', default='46/38/16',
                               help='Base allocation as SPY/GLD/TLT percentages')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--freq', type=int, default=21, help='Rebalance frequency (days)')
    backtest_parser.add_argument('--output', help='Output JSON file')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Get current recommendation')
    live_parser.add_argument('--portfolio', default='46/38/16',
                            help='Base allocation as SPY/GLD/TLT percentages')
    live_parser.add_argument('--output', help='Output JSON file')
    live_parser.add_argument('--save-db', action='store_true', help='Save to database')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if args.command == 'compute':
        multi_speed = MultiSpeedMomentum()
        
        base_alloc = DEFAULT_BASE_ALLOCATION.copy()
        signal = multi_speed.compute_speed_signal(
            args.ticker,
            args.tier,
            base_alloc.get(args.ticker, 0.0)
        )
        
        if signal:
            result = signal.to_dict()
            print(json.dumps(result, indent=2))
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
        else:
            print(json.dumps({'error': f'Could not compute signal for {args.ticker}'}))
    
    elif args.command == 'backtest':
        parts = args.portfolio.split('/')
        base_alloc = {
            'SPY': float(parts[0]) / 100 if float(parts[0]) > 1 else float(parts[0]),
            'GLD': float(parts[1]) / 100 if float(parts[1]) > 1 else float(parts[1]),
            'TLT': float(parts[2]) / 100 if float(parts[2]) > 1 else float(parts[2]),
        }
        base_alloc['CASH'] = 0.0
        
        backtester = MultiSpeedBacktester(
            base_allocation=base_alloc,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq=args.freq
        )
        result = backtester.run_backtest()
        print(json.dumps(result, indent=2, default=str))
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
        
        multi_speed = MultiSpeedMomentum()
        result = multi_speed.get_current_recommendation(base_alloc)
        
        output = result.to_dict()
        print(json.dumps(output, indent=2))
        
        if args.save_db:
            multi_speed.save_to_db(result)
            print(f"\nSaved to database: {DB_PATH}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
    
    elif args.command == 'status':
        print("Multi-Speed Momentum Ensemble v2.56 - Status")
        print("=" * 50)
        for tier, config in SPEED_TIERS.items():
            print(f"\n{tier.upper()} TIER:")
            print(f"  Lookback: {config['lookback_days']} days")
            print(f"  Skip: {config['skip_days']} days")
            print(f"  Vol window: {config['vol_window']} days")
            print(f"  {config['description']}")
        print("\nEnsemble: Equal risk-weight across tiers")
        print(f"Max deviation: {MAX_DEVIATION * 100}%")
        print(f"Min weight: {MIN_WEIGHT * 100}%")
        print(f"Target volatility: {VOL_TARGET * 100}%")
        print(f"\nData source: {PRICES_PATH}")
        print(f"Prices exist: {PRICES_PATH.exists()}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
