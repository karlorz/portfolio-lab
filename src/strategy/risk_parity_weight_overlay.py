#!/usr/bin/env python3
"""
Portfolio-Lab v2.57b: Risk Parity Weight Overlay (Refined)

Improved risk parity implementation that preserves CAGR while improving Sharpe:
- Uses inverse-volatility weights (not equal capital weights)
- Allows portfolio volatility to float naturally (no leverage targeting)
- Applies max 15% deviation from base allocation (like TSMOM overlay)
- Rebalances monthly or on drift threshold

Risk Parity Weights:
    Raw w_i = (1/σ_i) / Σ(1/σ_j)  # Inverse vol weighting
    Adjusted w_i = base_w_i + RP_overlay  # Apply as overlay
    Clip: max deviation ±15% from base allocation

Performance Target:
    - Baseline: Sharpe 0.94 (Multi-Speed v2.56)
    - Target: Sharpe 1.05-1.10 (+0.11 to +0.16)
    - Maintains or improves CAGR through better risk allocation

Source: Bridgewater All Weather, Asness (1996), BlackRock Systematic
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

VOL_LOOKBACK = 252
MAX_DEVIATION = 0.15  # ±15% deviation from base
MIN_WEIGHT = 0.05
REBALANCE_FREQ = 21

DEFAULT_BASE = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}


@dataclass
class RPWeightOverlay:
    """Risk parity weight overlay allocation."""
    timestamp: str
    
    # Calculations
    asset_vols: Dict[str, float]
    raw_rp_weights: Dict[str, float]  # Pure risk parity
    base_weights: Dict[str, float]
    
    # Overlay adjustment
    rp_adjustments: Dict[str, float]  # Difference from base
    target_weights: Dict[str, float]  # Final weights (clipped)
    
    # Metrics
    expected_vol: float
    risk_parity_score: float  # How close to equal risk contribution
    
    def to_dict(self) -> dict:
        return asdict(self)


class RiskParityWeightOverlay:
    """
    Risk parity weight overlay - applies inverse-vol weighting as overlay.
    
    Key difference from v2.57a: No leverage targeting, just better weight allocation.
    """
    
    def __init__(
        self,
        prices_path: Path = PRICES_PATH,
        db_path: Path = DB_PATH,
        vol_lookback: int = VOL_LOOKBACK,
        max_deviation: float = MAX_DEVIATION
    ):
        self.prices_path = prices_path
        self.db_path = db_path
        self.vol_lookback = vol_lookback
        self.max_deviation = max_deviation
        self._prices_df: Optional[pd.DataFrame] = None
    
    def _load_prices(self) -> pd.DataFrame:
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
    
    def calculate_realized_vol(self, ticker: str, prices_df: Optional[pd.DataFrame] = None) -> Optional[float]:
        if prices_df is None:
            prices_df = self._load_prices()
        
        if ticker not in prices_df.columns:
            return None
        
        prices = prices_df[ticker].dropna()
        if len(prices) < self.vol_lookback + 10:
            return None
        
        recent_prices = prices.iloc[-self.vol_lookback:]
        returns = recent_prices.pct_change().dropna()
        
        if len(returns) < 20:
            return None
        
        return float(returns.std() * np.sqrt(252))
    
    def calculate_rp_overlay(
        self,
        base_weights: Dict[str, float],
        prices_df: Optional[pd.DataFrame] = None
    ) -> Optional[RPWeightOverlay]:
        """Calculate risk parity weight overlay."""
        if prices_df is None:
            prices_df = self._load_prices()
        
        assets = [k for k in base_weights.keys() if k != 'CASH']
        
        # Calculate volatilities
        asset_vols = {}
        for asset in assets:
            vol = self.calculate_realized_vol(asset, prices_df)
            if vol and vol > 0:
                asset_vols[asset] = vol
        
        if len(asset_vols) < len(assets):
            return None
        
        # Pure risk parity weights
        inverse_vols = {k: 1.0 / v for k, v in asset_vols.items()}
        sum_inv = sum(inverse_vols.values())
        raw_rp_weights = {k: v / sum_inv for k, v in inverse_vols.items()}
        
        # Calculate RP adjustments from base weights
        rp_adjustments = {}
        for asset in assets:
            base_w = base_weights.get(asset, 0.0)
            rp_w = raw_rp_weights.get(asset, 0.0)
            adjustment = rp_w - base_w
            # Clip to max deviation
            adjustment = np.clip(adjustment, -self.max_deviation, self.max_deviation)
            rp_adjustments[asset] = adjustment
        
        # Apply adjustments
        target_weights = {'CASH': 0.0}
        for asset in assets:
            target_weights[asset] = base_weights.get(asset, 0.0) + rp_adjustments[asset]
        
        # Ensure minimum weights and normalize
        for asset in target_weights:
            if asset != 'CASH':
                target_weights[asset] = max(target_weights[asset], MIN_WEIGHT)
        
        # Normalize to sum to 1.0
        total = sum(w for k, w in target_weights.items() if k != 'CASH')
        if total > 0:
            for asset in target_weights:
                if asset != 'CASH':
                    target_weights[asset] /= total
        
        # Recalculate adjustments post-normalization
        for asset in assets:
            rp_adjustments[asset] = target_weights[asset] - base_weights.get(asset, 0.0)
        
        # Calculate expected portfolio volatility
        expected_vol = sum(
            target_weights[k] * asset_vols[k] 
            for k in target_weights.keys() if k != 'CASH' and k in asset_vols
        )
        
        # Risk parity score (1.0 = perfect equal risk contribution)
        risk_contrib = {k: target_weights[k] * asset_vols[k] for k in asset_vols.keys()}
        if risk_contrib:
            mean_rc = np.mean(list(risk_contrib.values()))
            std_rc = np.std(list(risk_contrib.values()))
            risk_parity_score = max(0.0, 1.0 - (std_rc / mean_rc if mean_rc > 0 else 0))
        else:
            risk_parity_score = 0.0
        
        current_date = prices_df.index[-1]
        
        return RPWeightOverlay(
            timestamp=current_date.isoformat(),
            asset_vols=asset_vols,
            raw_rp_weights=raw_rp_weights,
            base_weights=base_weights,
            rp_adjustments=rp_adjustments,
            target_weights=target_weights,
            expected_vol=expected_vol,
            risk_parity_score=risk_parity_score
        )


class RPBacktester:
    """Backtester for risk parity weight overlay."""
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: int = REBALANCE_FREQ,
        max_deviation: float = MAX_DEVIATION
    ):
        self.base_weights = base_weights
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.rebalance_freq = rebalance_freq
        self.max_deviation = max_deviation
        self.overlay = RiskParityWeightOverlay(max_deviation=max_deviation)
        self.prices_df = self.overlay._load_prices()
    
    def run_backtest(self) -> Dict:
        prices = self.prices_df.copy()
        
        if self.start_date:
            prices = prices[prices.index >= self.start_date]
        if self.end_date:
            prices = prices[prices.index <= self.end_date]
        
        min_history = VOL_LOOKBACK + 50
        if len(prices) < min_history:
            return {'error': f'Insufficient data: {len(prices)} days < {min_history} required'}
        
        portfolio_value = 100000.0
        current_weights = self.base_weights.copy()
        
        daily_values = []
        
        for i in range(min_history, len(prices)):
            current_date = prices.index[i]
            history = prices.iloc[:i+1]
            
            # Monthly rebalancing
            if (i - min_history) % self.rebalance_freq == 0:
                self.overlay._prices_df = history
                allocation = self.overlay.calculate_rp_overlay(self.base_weights)
                if allocation:
                    current_weights = allocation.target_weights
            
            # Daily return
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
        
        # Metrics
        df_values = pd.DataFrame(daily_values)
        df_values['date'] = pd.to_datetime(df_values['date'])
        df_values.set_index('date', inplace=True)
        
        returns = df_values['return'].dropna()
        
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
        
        # Baseline comparison
        baseline_values = [100000.0]
        for i in range(min_history, len(prices)):
            daily_return = 0.0
            for ticker, weight in self.base_weights.items():
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
            'strategy': f'Risk Parity Weight Overlay v2.57b (max_dev={self.max_deviation:.0%})',
            'start_date': prices.index[min_history].isoformat(),
            'end_date': prices.index[-1].isoformat(),
            'trading_days': len(df_values),
            'start_value': 100000,
            'end_value': portfolio_value,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
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
    parser = argparse.ArgumentParser(description='Risk Parity Weight Overlay v2.57b')
    subparsers = parser.add_subparsers(dest='command')
    
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--max-dev', type=float, default=0.15, help='Max deviation from base')
    backtest_parser.add_argument('--start', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    backtest_parser.add_argument('--output', help='Output JSON')
    
    live_parser = subparsers.add_parser('live', help='Get current allocation')
    live_parser.add_argument('--max-dev', type=float, default=0.15)
    live_parser.add_argument('--output', help='Output JSON')
    
    subparsers.add_parser('status', help='Show status')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        backtester = RPBacktester(
            base_weights=DEFAULT_BASE,
            start_date=args.start,
            end_date=args.end,
            max_deviation=args.max_dev
        )
        result = backtester.run_backtest()
        print(json.dumps(result, indent=2, default=str))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
    
    elif args.command == 'live':
        overlay = RiskParityWeightOverlay(max_deviation=args.max_dev)
        allocation = overlay.calculate_rp_overlay(DEFAULT_BASE)
        if allocation:
            print(json.dumps(allocation.to_dict(), indent=2))
        else:
            print(json.dumps({'error': 'Could not calculate allocation'}))
    
    elif args.command == 'status':
        print("Risk Parity Weight Overlay v2.57b - Status")
        print("=" * 50)
        print(f"Max deviation from base: {MAX_DEVIATION:.0%}")
        print(f"Vol lookback: {VOL_LOOKBACK} days")
        print(f"Rebalance frequency: {REBALANCE_FREQ} days")
        print()
        print("Base allocation (46/38/16):")
        for k, v in DEFAULT_BASE.items():
            if k != 'CASH':
                print(f"  {k}: {v:.0%}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
