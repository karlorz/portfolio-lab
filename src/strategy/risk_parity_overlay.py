#!/usr/bin/env python3
"""
Portfolio-Lab v2.57: Risk Parity Volatility Targeting Overlay

Bridgewater/BlackRock-style risk parity based on:
- Bridgewater All Weather (Dalio): Equal risk contribution across asset classes
- BlackRock Factor Framework: Four-pillar factor timing
- Asness (1996): "Value and Momentum Everywhere"

Risk Parity Principle:
    Instead of equal capital weights (46/38/16), allocate by equal RISK contribution.
    Assets with lower volatility receive higher capital allocation (via leverage).

Volatility Targeting:
    Target portfolio volatility (e.g., 10%) via dynamic leverage adjustment.
    Higher leverage in low-vol regimes, lower in high-vol.

Implementation:
    w_i = (1/σ_i) / Σ(1/σ_j)           # Inverse volatility weights
    σ_portfolio = Σ(w_i * σ_i)          # Portfolio volatility
    Leverage = TargetVol / σ_portfolio  # Scale to target
    Final_w_i = w_i * Leverage

Performance Target:
    - Baseline Sharpe: 0.94 (Multi-Speed v2.56)
    - Risk Parity Target: 1.05-1.10 (+0.11 to +0.16 improvement)
    - Lower max drawdown through vol targeting

Usage:
    python -m src.strategy.risk_parity_overlay backtest --target-vol 0.10
    python -m src.strategy.risk_parity_overlay live --target-vol 0.10 --save-db
    python -m src.strategy.risk_parity_overlay status
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Constants
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

# Risk Parity Parameters
VOL_LOOKBACK = 252           # 1-year volatility estimation
VOL_TARGET_DEFAULT = 0.10    # 10% annualized volatility target
MAX_LEVERAGE = 2.0           # Max 2x leverage
MIN_LEVERAGE = 0.5           # Min 0.5x leverage (de-risking)
REBALANCE_FREQ = 21          # Monthly rebalancing
MIN_WEIGHT = 0.05            # Minimum 5% per asset

# Asset mapping with risk characteristics
ASSETS = {
    'SPY': {'type': 'equity', 'base_vol': 0.15, 'desc': 'US Equities'},
    'GLD': {'type': 'commodity', 'base_vol': 0.12, 'desc': 'Gold'},
    'TLT': {'type': 'bond', 'base_vol': 0.10, 'desc': 'Long Treasuries'},
    'CASH': {'type': 'cash', 'base_vol': 0.005, 'desc': 'Cash'},
}

DEFAULT_BASE_ALLOCATION = {
    'SPY': 0.46,
    'GLD': 0.38,
    'TLT': 0.16,
    'CASH': 0.0,
}


@dataclass
class RiskParityAllocation:
    """Risk parity allocation for a portfolio."""
    timestamp: str
    
    # Raw calculations
    asset_vols: Dict[str, float]          # Realized vol per asset
    inverse_vols: Dict[str, float]        # 1/vol for each asset
    raw_rp_weights: Dict[str, float]      # Unleveraged RP weights
    
    # Leverage calculation
    portfolio_vol_unlevered: float        # Vol without leverage
    leverage: float                       # TargetVol / unlevered_vol
    
    # Final allocation
    target_weights: Dict[str, float]    # Final risk parity weights
    target_vol: float                     # Target volatility
    actual_vol_estimated: float           # Expected portfolio vol
    
    # Risk metrics
    risk_contribution: Dict[str, float]   # Marginal risk contribution
    risk_parity_quality: float            # How close to equal risk (1.0 = perfect)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VolatilityTargetState:
    """Volatility targeting state with regime detection."""
    timestamp: str
    current_portfolio_vol: float          # Current realized portfolio vol
    target_vol: float                     # Target volatility
    vol_regime: str                       # low, normal, high, crisis
    leverage_adjustment: float            # Dynamic leverage based on regime
    
    # Regime thresholds (annualized vol)
    regime_thresholds: Dict[str, float]  # Boundaries for regime classification
    
    def to_dict(self) -> dict:
        return asdict(self)


class RiskParityOverlay:
    """
    Risk parity overlay with volatility targeting.
    
    Implements Bridgewater-style risk parity:
    - Equal risk contribution across assets (not equal capital)
    - Dynamic leverage to hit volatility target
    - Monthly rebalancing with drift thresholds
    """
    
    def __init__(
        self,
        prices_path: Path = PRICES_PATH,
        db_path: Path = DB_PATH,
        vol_lookback: int = VOL_LOOKBACK,
        target_vol: float = VOL_TARGET_DEFAULT,
        max_leverage: float = MAX_LEVERAGE,
        min_leverage: float = MIN_LEVERAGE
    ):
        self.prices_path = prices_path
        self.db_path = db_path
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
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
    
    def calculate_realized_vol(
        self,
        ticker: str,
        lookback_days: Optional[int] = None,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """Calculate realized volatility for an asset."""
        if prices_df is None:
            prices_df = self._load_prices()
        
        if ticker not in prices_df.columns:
            return None
        
        lookback = lookback_days or self.vol_lookback
        prices = prices_df[ticker].dropna()
        
        if len(prices) < lookback + 10:
            return None
        
        recent_prices = prices.iloc[-lookback:]
        returns = recent_prices.pct_change().dropna()
        
        if len(returns) < 20:
            return None
        
        # Annualized volatility
        vol = returns.std() * np.sqrt(252)
        return float(vol)
    
    def detect_vol_regime(
        self,
        portfolio_vol: float,
        target_vol: float
    ) -> Tuple[str, float]:
        """
        Detect volatility regime and suggest leverage adjustment.
        
        Regimes:
        - low: vol < 0.6 * target (opportunity for more leverage)
        - normal: 0.6 * target <= vol <= 1.4 * target
        - high: 1.4 * target < vol <= 2.0 * target (reduce leverage)
        - crisis: vol > 2.0 * target (de-risk significantly)
        """
        ratio = portfolio_vol / target_vol if target_vol > 0 else 1.0
        
        if ratio < 0.6:
            regime = 'low'
            adj = 1.2  # Increase leverage slightly
        elif ratio < 1.4:
            regime = 'normal'
            adj = 1.0  # Normal operation
        elif ratio < 2.0:
            regime = 'high'
            adj = 0.85  # Reduce leverage
        else:
            regime = 'crisis'
            adj = 0.7  # De-risk significantly
        
        return regime, adj
    
    def calculate_risk_parity_allocation(
        self,
        target_vol: Optional[float] = None,
        prices_df: Optional[pd.DataFrame] = None,
        assets: List[str] = None
    ) -> Optional[RiskParityAllocation]:
        """
        Calculate risk parity allocation across assets.
        
        Formula:
        w_i = (1/σ_i) / Σ(1/σ_j)
        Leverage = TargetVol / Σ(w_i * σ_i)
        """
        if prices_df is None:
            prices_df = self._load_prices()
        
        if assets is None:
            assets = ['SPY', 'GLD', 'TLT']
        
        target = target_vol or self.target_vol
        
        # Calculate volatilities
        asset_vols = {}
        for asset in assets:
            vol = self.calculate_realized_vol(asset, prices_df=prices_df)
            if vol and vol > 0:
                asset_vols[asset] = vol
        
        if len(asset_vols) < len(assets):
            return None
        
        # Inverse volatility weights (unleveraged)
        inverse_vols = {k: 1.0 / v for k, v in asset_vols.items()}
        sum_inv_vol = sum(inverse_vols.values())
        
        if sum_inv_vol == 0:
            return None
        
        raw_rp_weights = {k: v / sum_inv_vol for k, v in inverse_vols.items()}
        
        # Calculate unlevered portfolio volatility
        # Assuming zero correlation for conservative estimate
        # In practice: σ_p = sqrt(w'Σw) where Σ is covariance matrix
        portfolio_vol_unlevered = sum(
            raw_rp_weights[k] * asset_vols[k] 
            for k in raw_rp_weights.keys()
        )
        
        # Detect regime and adjust
        regime, regime_adj = self.detect_vol_regime(portfolio_vol_unlevered, target)
        adjusted_target = target * regime_adj
        
        # Calculate leverage to hit target
        if portfolio_vol_unlevered > 0:
            leverage = adjusted_target / portfolio_vol_unlevered
        else:
            leverage = 1.0
        
        # Clip leverage
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        # Apply leverage to get final weights
        target_weights = {k: v * leverage for k, v in raw_rp_weights.items()}
        
        # Ensure minimum weights
        for k in target_weights:
            target_weights[k] = max(target_weights[k], MIN_WEIGHT)
        
        # Renormalize to sum to leverage (allowing cash to absorb difference)
        weight_sum = sum(target_weights.values())
        if weight_sum > 0:
            scale = leverage / weight_sum
            target_weights = {k: v * scale for k, v in target_weights.items()}
        
        # Add cash residual
        target_weights['CASH'] = max(0.0, 1.0 - sum(target_weights.values()))
        
        # Calculate risk contribution (simplified: weight * vol)
        risk_contribution = {
            k: target_weights[k] * asset_vols[k] 
            for k in target_weights.keys() if k != 'CASH'
        }
        
        # Risk parity quality (1.0 = equal risk contribution)
        if len(risk_contribution) > 0:
            risk_values = list(risk_contribution.values())
            mean_risk = np.mean(risk_values)
            if mean_risk > 0:
                # Coefficient of variation inverse (lower CV = better parity)
                cv = np.std(risk_values) / mean_risk
                risk_parity_quality = max(0.0, 1.0 - cv)
            else:
                risk_parity_quality = 0.0
        else:
            risk_parity_quality = 0.0
        
        # Estimated actual portfolio volatility with leverage
        actual_vol_estimated = portfolio_vol_unlevered * leverage
        
        current_date = prices_df.index[-1]
        
        return RiskParityAllocation(
            timestamp=current_date.isoformat(),
            asset_vols=asset_vols,
            inverse_vols=inverse_vols,
            raw_rp_weights=raw_rp_weights,
            portfolio_vol_unlevered=portfolio_vol_unlevered,
            leverage=leverage,
            target_weights=target_weights,
            target_vol=target,
            actual_vol_estimated=actual_vol_estimated,
            risk_contribution=risk_contribution,
            risk_parity_quality=risk_parity_quality
        )
    
    def save_to_db(self, allocation: RiskParityAllocation):
        """Save risk parity allocation to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create table if not exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS risk_parity_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                target_vol REAL,
                leverage REAL,
                target_weights TEXT,
                asset_vols TEXT,
                risk_parity_quality REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            INSERT INTO risk_parity_allocations
            (timestamp, target_vol, leverage, target_weights, asset_vols, risk_parity_quality)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            allocation.timestamp,
            allocation.target_vol,
            allocation.leverage,
            json.dumps(allocation.target_weights),
            json.dumps(allocation.asset_vols),
            allocation.risk_parity_quality
        ))
        
        conn.commit()
        conn.close()


class RiskParityBacktester:
    """Backtester for risk parity volatility targeting strategy."""
    
    def __init__(
        self,
        target_vol: float = VOL_TARGET_DEFAULT,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: int = REBALANCE_FREQ
    ):
        self.target_vol = target_vol
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.rebalance_freq = rebalance_freq
        self.rp_overlay = RiskParityOverlay(target_vol=target_vol)
        self.prices_df = self.rp_overlay._load_prices()
    
    def run_backtest(self) -> Dict:
        """Run full historical backtest."""
        prices = self.prices_df.copy()
        
        if self.start_date:
            prices = prices[prices.index >= self.start_date]
        if self.end_date:
            prices = prices[prices.index <= self.end_date]
        
        # Need enough data for volatility estimation
        min_history = VOL_LOOKBACK + 50
        if len(prices) < min_history:
            return {'error': f'Insufficient data: {len(prices)} days < {min_history} required'}
        
        # Portfolio tracking
        portfolio_value = 100000.0
        current_weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        
        daily_values = []
        rebalance_dates = []
        leverage_history = []
        
        for i in range(min_history, len(prices)):
            current_date = prices.index[i]
            history = prices.iloc[:i+1]
            
            # Monthly rebalancing
            if (i - min_history) % self.rebalance_freq == 0:
                self.rp_overlay._prices_df = history
                
                allocation = self.rp_overlay.calculate_risk_parity_allocation(
                    target_vol=self.target_vol
                )
                
                if allocation:
                    current_weights = allocation.target_weights
                    leverage_history.append({
                        'date': current_date.isoformat(),
                        'leverage': allocation.leverage,
                        'vol_unlevered': allocation.portfolio_vol_unlevered,
                        'quality': allocation.risk_parity_quality
                    })
                    rebalance_dates.append({
                        'date': current_date.isoformat(),
                        'weights': current_weights.copy(),
                        'allocation': allocation.to_dict()
                    })
            
            # Calculate daily return
            daily_return = 0.0
            for ticker, weight in current_weights.items():
                if ticker == 'CASH' or ticker not in prices.columns:
                    continue
                
                if i > 0:
                    ticker_return = float(prices[ticker].iloc[i]) / float(prices[ticker].iloc[i-1]) - 1
                    daily_return += weight * ticker_return
            
            new_value = portfolio_value * (1 + daily_return)
            if np.isnan(new_value) or np.isinf(new_value):
                new_value = portfolio_value
            portfolio_value = new_value
            
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
        years = len(df_values) / 252
        
        total_return = (end_val / start_val) - 1 if start_val > 0 else 0
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
        
        # Baseline comparison (static 46/38/16)
        baseline_values = [100000.0]
        static_weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        
        for i in range(min_history, len(prices)):
            daily_return = 0.0
            for ticker, weight in static_weights.items():
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
        
        # Average leverage used
        avg_leverage = np.mean([x['leverage'] for x in leverage_history]) if leverage_history else 1.0
        
        return {
            'strategy': f'Risk Parity Vol Target {self.target_vol:.0%} v2.57',
            'start_date': prices.index[min_history].isoformat(),
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
            'avg_leverage': avg_leverage,
            'baseline_cagr': baseline_cagr,
            'baseline_sharpe': baseline_sharpe,
            'baseline_volatility': baseline_vol,
            'excess_return': cagr - baseline_cagr,
            'sharpe_improvement': sharpe - baseline_sharpe,
            'crisis_2008_return': crisis_returns.get('2008'),
            'crisis_2020_return': crisis_returns.get('2020'),
            'crisis_2022_return': crisis_returns.get('2022'),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Risk Parity Volatility Targeting v2.57'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run historical backtest')
    backtest_parser.add_argument('--target-vol', type=float, default=0.10,
                               help='Target annualized volatility (e.g., 0.10 for 10%%)')
    backtest_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--freq', type=int, default=21, help='Rebalance frequency (days)')
    backtest_parser.add_argument('--output', help='Output JSON file')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Get current allocation')
    live_parser.add_argument('--target-vol', type=float, default=0.10,
                            help='Target annualized volatility')
    live_parser.add_argument('--output', help='Output JSON file')
    live_parser.add_argument('--save-db', action='store_true', help='Save to database')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        backtester = RiskParityBacktester(
            target_vol=args.target_vol,
            start_date=args.start,
            end_date=args.end,
            rebalance_freq=args.freq
        )
        result = backtester.run_backtest()
        print(json.dumps(result, indent=2, default=str))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    
    elif args.command == 'live':
        overlay = RiskParityOverlay(target_vol=args.target_vol)
        allocation = overlay.calculate_risk_parity_allocation(target_vol=args.target_vol)
        
        if allocation:
            output = allocation.to_dict()
            print(json.dumps(output, indent=2))
            
            if args.save_db:
                overlay.save_to_db(allocation)
                print(f"\nSaved to database: {DB_PATH}")
        else:
            print(json.dumps({'error': 'Could not calculate allocation'}))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(allocation.to_dict() if allocation else {}, f, indent=2)
    
    elif args.command == 'status':
        print("Risk Parity Volatility Targeting v2.57 - Status")
        print("=" * 50)
        print(f"Volatility lookback: {VOL_LOOKBACK} days")
        print(f"Target volatility default: {VOL_TARGET_DEFAULT:.0%}")
        print(f"Max leverage: {MAX_LEVERAGE}x")
        print(f"Min leverage: {MIN_LEVERAGE}x")
        print(f"Rebalance frequency: {REBALANCE_FREQ} days")
        print()
        print("Regime thresholds (relative to target):")
        print("  < 0.6x: LOW (increase leverage)")
        print("  0.6-1.4x: NORMAL")
        print("  1.4-2.0x: HIGH (reduce leverage)")
        print("  > 2.0x: CRISIS (de-risk)")
        print()
        print(f"Data source: {PRICES_PATH}")
        print(f"Prices exist: {PRICES_PATH.exists()}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
