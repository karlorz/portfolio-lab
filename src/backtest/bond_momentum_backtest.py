"""
Bond Momentum Overlay Backtest - v3.30 Phase 3 Implementation

Portfolio-level walk-forward backtest for bond momentum tactical overlay.
Tests SPY/GLD/TLT/IEF allocation with momentum-based duration timing.

Target: Validate +0.02 to +0.03 Sharpe improvement vs baseline 46/38/16.
Period: 2010-2026 (16+ years including 2022 rate hiking cycle)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for bond momentum overlay backtest."""
    start_date: str = "2010-01-01"
    end_date: str = "2026-05-15"
    initial_capital: float = 100000.0
    
    # Baseline allocation (46/38/16)
    base_spy_weight: float = 0.46
    base_gld_weight: float = 0.38
    base_tlt_weight: float = 0.16
    
    # Bond momentum parameters
    formation_months: int = 18  # 18m optimal from research
    skip_months: int = 1
    vol_target: float = 0.06  # 6% for bonds
    
    # Rebalancing
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 10.0  # 10 bps per trade
    
    # Overlay constraints
    max_duration_shift: float = 0.10  # Max 10% shift in bond allocation
    min_holding_days: int = 21  # Monthly minimum hold


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic metrics
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Overlay-specific
    baseline_sharpe: float
    sharpe_improvement: float
    overlay_trades: int
    
    # Crisis performance
    return_2008: Optional[float]
    return_2020: Optional[float]
    return_2022: Optional[float]
    
    # Allocation history
    avg_tlt_weight: float
    avg_ief_weight: float
    avg_shy_weight: float
    
    # Full equity curve
    equity_curve: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'baseline_sharpe': self.baseline_sharpe,
            'sharpe_improvement': self.sharpe_improvement,
            'overlay_trades': self.overlay_trades,
            'return_2008': self.return_2008,
            'return_2020': self.return_2020,
            'return_2022': self.return_2022,
            'avg_tlt_weight': self.avg_tlt_weight,
            'avg_ief_weight': self.avg_ief_weight,
            'avg_shy_weight': self.avg_shy_weight,
        }


class BondMomentumOverlayBacktester:
    """
    Walk-forward backtest for bond momentum tactical overlay.
    
    Strategy: Use momentum signals to dynamically allocate between
    TLT (long duration), IEF (intermediate), and SHY (short duration).
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.prices: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        
    def load_data(self, data_path: Path = None) -> bool:
        """Load price data from prices.json."""
        try:
            if data_path is None:
                data_path = Path(__file__).parent.parent.parent / "public" / "data" / "prices.json"
            
            with open(data_path) as f:
                data = json.load(f)
            
            # Load all symbols needed
            symbols = ['SPY', 'GLD', 'TLT', 'IEF', 'SHY']
            records = []
            
            for symbol in symbols:
                if symbol in data:
                    for entry in data[symbol]:
                        records.append({
                            'date': entry['d'],
                            'symbol': symbol,
                            'price': entry['p']
                        })
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='symbol', values='price')
            
            self.prices = df
            logger.info(f"Loaded data: {df.index[0]} to {df.index[-1]}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def calculate_momentum_signals(self) -> pd.DataFrame:
        """Calculate momentum signals for bond ETFs."""
        if self.prices is None:
            raise ValueError("No price data loaded")
        
        prices = self.prices
        formation_days = self.config.formation_months * 21
        skip_days = self.config.skip_months * 21
        
        signals = pd.DataFrame(index=prices.index)
        
        for etf in ['TLT', 'IEF', 'SHY']:
            if etf not in prices.columns:
                continue
            
            # Trailing return (skip most recent month)
            momentum = prices[etf].pct_change(formation_days).shift(skip_days)
            
            # Realized volatility (63-day)
            realized_vol = prices[etf].pct_change().rolling(63).std() * np.sqrt(252)
            
            # Position sizing (inverse vol)
            position_size = self.config.vol_target / (realized_vol + 0.01)
            position_size = position_size.clip(0, 2.0)
            
            # Signal: positive momentum gets position, zero otherwise
            signals[f'{etf}_signal'] = np.where(momentum > 0, position_size, 0)
            signals[f'{etf}_momentum'] = momentum
        
        self.signals = signals
        return signals
    
    def get_bond_allocation(self, date: datetime) -> Tuple[float, float, float]:
        """
        Get bond allocation based on momentum signals.
        Returns (tlt_weight, ief_weight, shy_weight) summing to 1.0
        """
        if self.signals is None or date not in self.signals.index:
            # Default: all TLT (baseline)
            return (1.0, 0.0, 0.0)
        
        signals = self.signals.loc[date]
        
        # Get momentum scores
        tlt_score = signals.get('TLT_signal', 0)
        ief_score = signals.get('IEF_signal', 0)
        shy_score = signals.get('SHY_signal', 0)
        
        # Normalize to weights
        total = tlt_score + ief_score + shy_score
        
        if total > 0:
            tlt_w = tlt_score / total
            ief_w = ief_score / total
            shy_w = shy_score / total
        else:
            # All negative momentum - go to shortest duration
            tlt_w, ief_w, shy_w = 0.0, 0.0, 1.0
        
        return (tlt_w, ief_w, shy_w)
    
    def run_backtest(self) -> BacktestResult:
        """Execute full walk-forward backtest."""
        if not self.load_data():
            raise ValueError("Failed to load data")
        
        self.calculate_momentum_signals()
        
        prices = self.prices
        config = self.config
        
        # Filter date range
        start_date = pd.to_datetime(config.start_date)
        end_date = pd.to_datetime(config.end_date)
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        
        # Calculate daily returns
        returns = prices.pct_change().fillna(0)
        
        # Initialize
        capital = config.initial_capital
        equity_curve = []
        
        # Track allocation
        current_tlt_w = 1.0  # Start with all TLT for bond sleeve
        current_ief_w = 0.0
        current_shy_w = 0.0
        
        bond_weight = config.base_tlt_weight  # Total bond allocation (16%)
        spy_weight = config.base_spy_weight
        gld_weight = config.base_gld_weight
        
        overlay_trades = 0
        last_rebalance = None
        
        # Track weights over time
        tlt_weights = []
        ief_weights = []
        shy_weights = []
        
        for date, row in returns.iterrows():
            # Monthly rebalancing check
            if last_rebalance is None or (date - last_rebalance).days >= 21:
                # Get new bond allocation based on momentum
                new_tlt_w, new_ief_w, new_shy_w = self.get_bond_allocation(date)
                
                # Check if change is significant (>5% shift in any component)
                max_change = max(
                    abs(new_tlt_w - current_tlt_w),
                    abs(new_ief_w - current_ief_w),
                    abs(new_shy_w - current_shy_w)
                )
                
                if max_change > 0.05:
                    current_tlt_w, current_ief_w, current_shy_w = new_tlt_w, new_ief_w, new_shy_w
                    overlay_trades += 1
                
                last_rebalance = date
            
            # Record weights
            tlt_weights.append(current_tlt_w)
            ief_weights.append(current_ief_w)
            shy_weights.append(current_shy_w)
            
            # Calculate portfolio return
            # Bond sleeve = bond_weight * (tlt_w * TLT_return + ief_w * IEF_return + shy_w * SHY_return)
            bond_return = (
                current_tlt_w * row.get('TLT', 0) +
                current_ief_w * row.get('IEF', 0) +
                current_shy_w * row.get('SHY', 0)
            )
            
            portfolio_return = (
                spy_weight * row.get('SPY', 0) +
                gld_weight * row.get('GLD', 0) +
                bond_weight * bond_return
            )
            
            # Transaction costs (10 bps per rebalance)
            if overlay_trades > 0 and date == last_rebalance:
                portfolio_return -= 0.0010 * bond_weight  # Cost on bond sleeve
            
            capital *= (1 + portfolio_return)
            
            equity_curve.append({
                'date': date.strftime('%Y-%m-%d'),
                'equity': capital,
                'return': portfolio_return,
                'tlt_weight': current_tlt_w * bond_weight,
                'ief_weight': current_ief_w * bond_weight,
                'shy_weight': current_shy_w * bond_weight,
            })
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # Total return
        total_return = (capital - config.initial_capital) / config.initial_capital
        
        # CAGR
        n_years = len(equity_df) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility and Sharpe
        daily_returns = equity_df['return']
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Crisis returns
        def get_year_return(year: int) -> Optional[float]:
            year_data = equity_df[equity_df.index.year == year]
            if len(year_data) > 0:
                return (1 + year_data['return']).prod() - 1
            return None
        
        return_2008 = get_year_return(2008)
        return_2020 = get_year_return(2020)
        return_2022 = get_year_return(2022)
        
        # Average weights
        avg_tlt_weight = np.mean(tlt_weights) * bond_weight if tlt_weights else bond_weight
        avg_ief_weight = np.mean(ief_weights) * bond_weight if ief_weights else 0
        avg_shy_weight = np.mean(shy_weights) * bond_weight if shy_weights else 0
        
        # Baseline Sharpe (static 46/38/16 allocation)
        baseline_returns = (
            spy_weight * returns['SPY'] +
            gld_weight * returns['GLD'] +
            bond_weight * returns['TLT']  # Static TLT only
        )
        baseline_cagr = (1 + baseline_returns).prod() ** (1 / n_years) - 1
        baseline_vol = baseline_returns.std() * np.sqrt(252)
        baseline_sharpe = baseline_cagr / baseline_vol if baseline_vol > 0 else 0
        
        sharpe_improvement = sharpe - baseline_sharpe
        
        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            baseline_sharpe=baseline_sharpe,
            sharpe_improvement=sharpe_improvement,
            overlay_trades=overlay_trades,
            return_2008=return_2008,
            return_2020=return_2020,
            return_2022=return_2022,
            avg_tlt_weight=avg_tlt_weight,
            avg_ief_weight=avg_ief_weight,
            avg_shy_weight=avg_shy_weight,
            equity_curve=equity_curve,
        )


def run_sensitivity_analysis() -> pd.DataFrame:
    """Test multiple formation periods and bond weights."""
    results = []
    
    formation_periods = [12, 18, 24]
    bond_weights = [0.10, 0.16, 0.20]
    
    for formation in formation_periods:
        for bond_w in bond_weights:
            config = BacktestConfig(
                formation_months=formation,
                base_tlt_weight=bond_w,
                base_spy_weight=0.50 - bond_w * 0.25,  # Maintain equity proportion
                base_gld_weight=0.40 - bond_w * 0.15,
            )
            
            backtester = BondMomentumOverlayBacktester(config)
            try:
                result = backtester.run_backtest()
                results.append({
                    'formation_months': formation,
                    'bond_weight': bond_w,
                    'cagr': result.cagr,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown,
                    'sharpe_improvement': result.sharpe_improvement,
                    'overlay_trades': result.overlay_trades,
                })
                logger.info(f"Formation={formation}m, Bond={bond_w:.0%}: "
                          f"Sharpe={result.sharpe_ratio:.2f} "
                          f"(Δ{result.sharpe_improvement:+.3f})")
            except Exception as e:
                logger.error(f"Failed for formation={formation}, bond={bond_w}: {e}")
    
    return pd.DataFrame(results)


def main():
    """Run backtest and save results."""
    print("=" * 70)
    print("Bond Momentum Overlay Backtest - v3.30 Phase 3")
    print("=" * 70)
    
    # Run main backtest
    print("\nRunning baseline configuration...")
    config = BacktestConfig()
    backtester = BondMomentumOverlayBacktester(config)
    result = backtester.run_backtest()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"CAGR: {result.cagr:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"\nBaseline Sharpe: {result.baseline_sharpe:.2f}")
    print(f"Sharpe Improvement: {result.sharpe_improvement:+.3f}")
    print(f"Overlay Trades: {result.overlay_trades}")
    print(f"\nAverage Bond Allocation:")
    print(f"  TLT: {result.avg_tlt_weight:.1%}")
    print(f"  IEF: {result.avg_ief_weight:.1%}")
    print(f"  SHY: {result.avg_shy_weight:.1%}")
    
    print(f"\n{'='*70}")
    print("CRISIS PERFORMANCE")
    print(f"{'='*70}")
    if result.return_2008:
        print(f"2008: {result.return_2008:.2%}")
    if result.return_2020:
        print(f"2020: {result.return_2020:.2%}")
    if result.return_2022:
        print(f"2022: {result.return_2022:.2%}")
    
    # Sensitivity analysis
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    sensitivity = run_sensitivity_analysis()
    
    print("\nFormation Period Comparison:")
    pivot = sensitivity.pivot_table(
        index='formation_months',
        columns='bond_weight',
        values='sharpe'
    )
    print(pivot.round(2))
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    output_dir = Path(__file__).parent.parent.parent / "research"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "bond_momentum_overlay_backtest.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'formation_months': config.formation_months,
                'base_tlt_weight': config.base_tlt_weight,
                'base_spy_weight': config.base_spy_weight,
                'base_gld_weight': config.base_gld_weight,
            },
            'results': result.to_dict(),
            'sensitivity': sensitivity.to_dict('records'),
            'success_criteria': {
                'sharpe_improvement_target': 0.02,
                'sharpe_improvement_achieved': result.sharpe_improvement,
                'meets_target': result.sharpe_improvement >= 0.02,
            }
        }, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    if result.sharpe_improvement >= 0.02:
        print(f"✓ SUCCESS: Sharpe improvement {result.sharpe_improvement:+.3f} meets target (+0.02)")
        print("  Recommendation: PROCEED to Phase 4 (dashboard integration)")
    elif result.sharpe_improvement >= 0.01:
        print(f"⚠ MARGINAL: Sharpe improvement {result.sharpe_improvement:+.3f} below target but positive")
        print("  Recommendation: Consider Phase 4 with lower ensemble weight")
    else:
        print(f"✗ FAILED: Sharpe improvement {result.sharpe_improvement:+.3f} negative or minimal")
        print("  Recommendation: REJECT - momentum overlay degrades performance")
    
    print(f"{'='*70}")
    
    return result


if __name__ == '__main__':
    main()
