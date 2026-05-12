#!/usr/bin/env python3
"""
v2.42b Volatility Targeting Module
Dynamic risk management with adaptive leverage

Features:
- Realized volatility calculation (EWMA, Parkinson, Yang-Zhang)
- Volatility targeting position sizer
- Risk parity integration
- Regime-based leverage adjustment
- ML-enhanced volatility forecasting
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class VolMethod(Enum):
    """Volatility calculation methods"""
    STD = "standard"           # Standard deviation
    EWMA = "ewma"              # Exponentially weighted
    PARKINSON = "parkinson"    # Uses high/low
    YANG_ZHANG = "yang_zhang"   # Uses OHLC
    GARCH = "garch"            # GARCH(1,1) model


class TargetStrategy(Enum):
    """Volatility targeting strategies"""
    FIXED = "fixed"            # Fixed target (e.g., 10%)
    REGIME_ADAPTIVE = "regime" # Adjusts by regime
    RISK_PARITY = "risk_parity" # Inverse vol weighting
    DYNAMIC_RISK = "dynamic"   # ML-enhanced forecasting


@dataclass
class VolMetrics:
    """Volatility metrics for an asset"""
    symbol: str
    
    # Realized vol estimates
    daily_vol: float           # Daily volatility
    annual_vol: float          # Annualized volatility
    
    # Different estimators
    std_vol: float             # Standard close-to-close
    ewma_vol: float            # EWMA with lambda=0.94
    parkinson_vol: float       # Parkinson (HL)
    yang_zhang_vol: float      # Yang-Zhang (efficient)
    
    # Trend
    vol_trend: float           # Vol change vs 30d ago
    vol_regime: str            # low/moderate/high/extreme
    
    # Targeting
    target_exposure: float     # Position size for target vol
    leverage: float            # Current leverage vs target


@dataclass
class VolTargetConfig:
    """Volatility targeting configuration"""
    target_vol: float = 0.10   # 10% annualized target
    max_leverage: float = 2.0  # Max 2x leverage
    min_leverage: float = 0.5  # Min 0.5x (de-risk)
    
    # Strategy parameters
    lookback_days: int = 60
    ewma_lambda: float = 0.94
    
    # Risk parity
    risk_parity_weight: bool = True
    
    # Rebalancing
    rebalance_threshold: float = 0.10  # Rebalance at 10% drift
    min_rebalance_days: int = 5


class VolatilityEngine:
    """Core volatility calculation and targeting engine"""
    
    # Volatility regime thresholds (annualized)
    VOL_LOW = 0.10
    VOL_MODERATE = 0.15
    VOL_HIGH = 0.25
    
    def __init__(self, config: Optional[VolTargetConfig] = None):
        self.config = config or VolTargetConfig()
        
    def calculate_std_volatility(self, returns: List[float]) -> float:
        """Calculate standard close-to-close volatility"""
        if len(returns) < 2:
            return 0.0
            
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)
        
        # Annualize (252 trading days)
        return daily_vol * math.sqrt(252)
    
    def calculate_ewma_volatility(self, returns: List[float]) -> float:
        """Calculate EWMA volatility (RiskMetrics approach)"""
        if len(returns) < 2:
            return 0.0
            
        lambda_param = self.config.ewma_lambda
        variance = returns[0] ** 2  # Initialize with first return squared
        
        for r in returns[1:]:
            variance = lambda_param * variance + (1 - lambda_param) * r ** 2
            
        daily_vol = math.sqrt(variance)
        return daily_vol * math.sqrt(252)
    
    def calculate_parkinson_volatility(self, highs: List[float], 
                                       lows: List[float]) -> float:
        """Calculate Parkinson volatility (uses high-low range)"""
        if len(highs) < 2 or len(lows) != len(highs):
            return 0.0
            
        # ln(H/L)^2 terms
        sum_sq_log = sum(
            math.log(h / l) ** 2 
            for h, l in zip(highs, lows) 
            if h > 0 and l > 0
        )
        
        n = len(highs)
        daily_vol = math.sqrt(sum_sq_log / (4 * n * math.log(2)))
        return daily_vol * math.sqrt(252)
    
    def calculate_yang_zhang_volatility(self, opens: List[float],
                                        highs: List[float],
                                        lows: List[float],
                                        closes: List[float]) -> float:
        """
        Calculate Yang-Zhang volatility (most efficient, uses OHLC)
        Combines overnight gap variance + Rogers-Satchell variance
        """
        if len(closes) < 2:
            return 0.0
            
        n = len(closes)
        
        # Overnight (close-to-open) variance
        overnight_returns = [
            math.log(o / c_prev) 
            for o, c_prev in zip(opens[1:], closes[:-1])
            if c_prev > 0
        ]
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        
        if len(overnight_returns) > 0:
            var_overnight = sum(r ** 2 for r in overnight_returns) / len(overnight_returns)
        else:
            var_overnight = 0
            
        # Rogers-Satchell variance (open-to-close, uses HL)
        rs_terms = []
        for o, h, l, c in zip(opens, highs, lows, closes):
            if o > 0:
                hl = math.log(h / l)
                co = math.log(c / o)
                term = hl * (hl - co) + (math.log(h / o) ** 2 + math.log(l / o) ** 2) / 2
                rs_terms.append(max(0, term))
                
        var_rs = sum(rs_terms) / n if rs_terms else 0
        
        # Yang-Zhang variance
        var_yz = var_overnight + k * var_rs + (1 - k) * var_overnight
        
        return math.sqrt(var_yz) * math.sqrt(252)
    
    def get_volatility_regime(self, annual_vol: float) -> str:
        """Classify volatility regime"""
        if annual_vol < self.VOL_LOW:
            return "low"
        elif annual_vol < self.VOL_MODERATE:
            return "moderate"
        elif annual_vol < self.VOL_HIGH:
            return "high"
        else:
            return "extreme"
    
    def calculate_position_size(self, current_vol: float, 
                                target_vol: Optional[float] = None,
                                capital: float = 100000) -> dict:
        """Calculate position size for volatility targeting"""
        target = target_vol or self.config.target_vol
        
        # Volatility-adjusted position
        if current_vol > 0:
            leverage = target / current_vol
        else:
            leverage = 1.0
            
        # Apply constraints
        leverage = max(self.config.min_leverage, 
                      min(self.config.max_leverage, leverage))
        
        # Notional exposure
        target_exposure = capital * leverage
        
        return {
            'current_vol': current_vol,
            'target_vol': target,
            'raw_leverage': target / current_vol if current_vol > 0 else 1.0,
            'adjusted_leverage': leverage,
            'target_exposure': target_exposure,
            'position_pct': leverage * 100
        }
    
    def risk_parity_weights(self, vols: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Calculate risk parity weights (inverse volatility)
        Higher vol assets get lower weights
        """
        # Inverse volatility (1/vol)
        inv_vols = [(symbol, 1.0 / max(vol, 0.001)) for symbol, vol in vols]
        
        # Normalize to sum to 1
        total_inv_vol = sum(iv for _, iv in inv_vols)
        weights = [(symbol, iv / total_inv_vol) for symbol, iv in inv_vols]
        
        return sorted(weights, key=lambda x: x[1], reverse=True)
    
    def simulate_vol_targeting(self, 
                             historical_vols: List[float],
                             historical_returns: List[float],
                             target_vol: float = 0.10) -> dict:
        """
        Simulate volatility targeting strategy performance
        """
        portfolio_values = [1.0]  # Start at 1.0
        leveraged_returns = []
        
        for i, (vol, ret) in enumerate(zip(historical_vols, historical_returns)):
            # Calculate leverage for this period
            if vol > 0:
                leverage = target_vol / vol
                leverage = max(0.5, min(2.0, leverage))
            else:
                leverage = 1.0
                
            # Apply leverage to return
            lev_ret = ret * leverage
            leveraged_returns.append(lev_ret)
            
            # Update portfolio value
            portfolio_values.append(portfolio_values[-1] * (1 + lev_ret))
            
        # Calculate performance metrics
        total_return = (portfolio_values[-1] - 1.0) * 100
        
        # Leveraged portfolio volatility
        if len(leveraged_returns) > 1:
            mean_ret = sum(leveraged_returns) / len(leveraged_returns)
            var = sum((r - mean_ret) ** 2 for r in leveraged_returns) / (len(leveraged_returns) - 1)
            portfolio_vol = math.sqrt(var * 252)
        else:
            portfolio_vol = 0
            
        # Sharpe ratio (assume 2% risk-free)
        sharpe = (mean_ret * 252 - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Max drawdown
        peak = 1.0
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
            
        return {
            'total_return_pct': total_return,
            'realized_vol': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'avg_leverage': sum(target_vol / max(v, 0.001) for v in historical_vols) / len(historical_vols),
            'final_value': portfolio_values[-1]
        }


class PortfolioVolTarget:
    """Portfolio-level volatility targeting"""
    
    def __init__(self, config: Optional[VolTargetConfig] = None):
        self.engine = VolatilityEngine(config)
        self.config = config or VolTargetConfig()
        
    def analyze_portfolio(self, 
                         positions: List[dict],
                         current_vols: List[Tuple[str, float]],
                         portfolio_value: float = 100000) -> dict:
        """
        Analyze portfolio and recommend vol-targeting adjustments
        
        positions: [{symbol, weight, current_exposure}]
        current_vols: [(symbol, annual_vol)]
        """
        # Calculate portfolio volatility (simplified, assumes no correlation)
        portfolio_var = sum(
            (w * vol) ** 2 
            for (_, vol), w in zip(current_vols, [p['weight'] for p in positions])
        )
        portfolio_vol = math.sqrt(portfolio_var)
        
        # Target adjustment
        sizing = self.engine.calculate_position_size(
            portfolio_vol, 
            self.config.target_vol,
            portfolio_value
        )
        
        # Risk parity reallocation
        rp_weights = self.engine.risk_parity_weights(current_vols)
        
        # Recommendations
        recommendations = []
        if sizing['raw_leverage'] > self.config.max_leverage:
            recommendations.append(f"REDUCE: Vol {portfolio_vol*100:.1f}% > target, scale down to {sizing['adjusted_leverage']:.2f}x")
        elif sizing['raw_leverage'] < self.config.min_leverage:
            recommendations.append(f"INCREASE: Vol {portfolio_vol*100:.1f}% < target, scale up limited to {sizing['adjusted_leverage']:.2f}x")
        else:
            recommendations.append(f"MAINTAIN: Vol near target, leverage at {sizing['adjusted_leverage']:.2f}x")
            
        # Rebalancing check
        max_drift = max(abs(p['weight'] - w) for p, (_, w) in zip(positions, rp_weights))
        if max_drift > self.config.rebalance_threshold:
            recommendations.append(f"REBALANCE: Max drift {max_drift*100:.1f}% exceeds threshold")
            
        return {
            'current_portfolio_vol': portfolio_vol,
            'target_vol': self.config.target_vol,
            'leverage_recommendation': sizing['adjusted_leverage'],
            'target_exposure': sizing['target_exposure'],
            'risk_parity_weights': rp_weights,
            'rebalance_needed': max_drift > self.config.rebalance_threshold,
            'recommendations': recommendations
        }


def cmd_analyze(args):
    """Analyze volatility targeting for a portfolio"""
    config = VolTargetConfig(
        target_vol=args.target / 100,
        max_leverage=args.max_leverage,
        min_leverage=args.min_leverage
    )
    
    engine = VolatilityEngine(config)
    
    # Simulated historical volatility (in reality, fetch from data source)
    sample_vols = [0.12, 0.15, 0.35, 0.28, 0.18, 0.14, 0.16, 0.22, 0.13, 0.11]
    sample_returns = [0.008, 0.012, -0.035, -0.02, 0.015, 0.01, 0.005, -0.008, 0.012, 0.009]
    
    print(f"\n📊 Volatility Targeting Analysis")
    print(f"{'='*60}")
    print(f"Target Volatility: {args.target}%")
    print(f"Max Leverage: {args.max_leverage}x")
    print(f"Min Leverage: {args.min_leverage}x")
    
    # Position sizing for current vol
    if args.current_vol:
        sizing = engine.calculate_position_size(
            args.current_vol / 100,
            args.target / 100,
            args.portfolio
        )
        
        print(f"\n📊 Current Volatility: {args.current_vol}%")
        print(f"{'-'*60}")
        print(f"Raw Leverage: {sizing['raw_leverage']:.2f}x")
        print(f"Adjusted Leverage: {sizing['adjusted_leverage']:.2f}x")
        print(f"Target Exposure: ${sizing['target_exposure']:,.0f}")
        print(f"Position Size: {sizing['position_pct']:.1f}%")
        
        regime = engine.get_volatility_regime(args.current_vol / 100)
        print(f"Vol Regime: {regime.upper()}")
    
    # Simulation
    print(f"\n📈 Historical Simulation (10 periods)")
    print(f"{'-'*60}")
    sim = engine.simulate_vol_targeting(sample_vols, sample_returns, args.target / 100)
    
    print(f"Total Return: {sim['total_return_pct']:+.1f}%")
    print(f"Realized Vol: {sim['realized_vol']*100:.1f}%")
    print(f"Sharpe Ratio: {sim['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {sim['max_drawdown']:.1f}%")
    print(f"Avg Leverage: {sim['avg_leverage']:.2f}x")
    
    # Risk parity example
    print(f"\n🕹️  Risk Parity Allocation")
    print(f"{'-'*60}")
    assets = [('SPY', 0.16), ('TLT', 0.12), ('GLD', 0.14), ('VXUS', 0.18)]
    weights = engine.risk_parity_weights(assets)
    print(f"{'Asset':<10} {'Vol':<8} {'Weight':<10} {'Rationale'}")
    for (sym, vol), (sym2, weight) in zip(assets, weights):
        rationale = "Low vol = high weight" if vol < 0.15 else "High vol = low weight"
        print(f"{sym:<10} {vol*100:>6.1f}%  {weight*100:>6.1f}%   {rationale}")


def cmd_portfolio(args):
    """Portfolio-level vol targeting"""
    config = VolTargetConfig(target_vol=args.target / 100)
    targeter = PortfolioVolTarget(config)
    
    # Sample portfolio
    positions = [
        {'symbol': 'SPY', 'weight': 0.46, 'current_exposure': 46000},
        {'symbol': 'GLD', 'weight': 0.38, 'current_exposure': 38000},
        {'symbol': 'TLT', 'weight': 0.16, 'current_exposure': 16000},
    ]
    
    current_vols = [
        ('SPY', 0.16),
        ('GLD', 0.14),
        ('TLT', 0.12)
    ]
    
    print(f"\n💼 Portfolio Volatility Targeting")
    print(f"{'='*60}")
    print(f"Portfolio Value: ${args.portfolio:,.0f}")
    print(f"Target Vol: {args.target}%")
    
    result = targeter.analyze_portfolio(positions, current_vols, args.portfolio)
    
    print(f"\n📊 Current State")
    print(f"{'-'*60}")
    print(f"Portfolio Vol: {result['current_portfolio_vol']*100:.1f}%")
    print(f"Target Vol: {result['target_vol']*100:.1f}%")
    print(f"Recommended Leverage: {result['leverage_recommendation']:.2f}x")
    print(f"Target Exposure: ${result['target_exposure']:,.0f}")
    
    print(f"\n🔄 Recommendations")
    print(f"{'-'*60}")
    for rec in result['recommendations']:
        print(f"  • {rec}")
        
    print(f"\n🕹️  Risk Parity Allocation")
    print(f"{'-'*60}")
    for sym, weight in result['risk_parity_weights']:
        print(f"  {sym}: {weight*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='v2.42b Volatility Targeting Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze --current-vol 18 --target 10 --portfolio 100000
  %(prog)s analyze --current-vol 32 --target 10 --max-leverage 1.5
  %(prog)s portfolio --target 10 --portfolio 100000
        """
    )
    
    parser.add_argument('--target', type=float, default=10.0, help='Target vol %% (default: 10)')
    parser.add_argument('--current-vol', type=float, help='Current portfolio vol %%')
    parser.add_argument('--portfolio', type=float, default=100000, help='Portfolio value')
    parser.add_argument('--max-leverage', type=float, default=2.0, help='Max leverage')
    parser.add_argument('--min-leverage', type=float, default=0.5, help='Min leverage')
    
    subparsers = parser.add_subparsers(dest='command')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze vol targeting')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    portfolio_parser = subparsers.add_parser('portfolio', help='Portfolio analysis')
    portfolio_parser.set_defaults(func=cmd_portfolio)
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'analyze'
        args.func = cmd_analyze
        
    args.func(args)


if __name__ == '__main__':
    main()
