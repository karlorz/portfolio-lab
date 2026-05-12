#!/usr/bin/env python3
"""
v2.42 Tail Risk Hedging Module
Hybrid protective puts + VIX call overlay for portfolio protection

Features:
- Protective put sizing (delta-based strike selection)
- VIX call overlay (volatility spike capture)
- Hybrid optimizer (cost vs convexity trade-off)
- Rolling hedge management
- Cost/benefit analytics
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class HedgeType(Enum):
    PROTECTIVE_PUT = "protective_put"
    VIX_CALL = "vix_call"
    HYBRID = "hybrid"
    COLLAR = "collar"
    PUT_SPREAD = "put_spread"


class MarketRegime(Enum):
    LOW_VOL = "low_vol"          # VIX < 15 - optimal hedge entry
    MODERATE_VOL = "mod_vol"     # VIX 15-25 - moderate cost
    ELEVATED_VOL = "elev_vol"    # VIX 25-35 - expensive, reduce size
    CRISIS = "crisis"            # VIX > 35 - wait for decay


@dataclass
class PutHedge:
    """Protective put position specification"""
    underlying: str              # Ticker (SPY, QQQ, etc.)
    notional: float              # $ to protect
    spot_price: float            # Current underlying price
    strike_pct: float            # Strike as % of spot (e.g., 0.95 = 95%)
    delta: float                 # Approximate delta (e.g., -0.15 for 15 delta)
    days_to_expiry: int          # Days until expiration
    implied_vol: float           # IV at strike
    premium_pct: float           # Premium as % of notional
    
    # Calculated fields
    num_contracts: int = 0       # Number of contracts (100 shares each)
    strike_price: float = 0.0    # Absolute strike price
    total_premium: float = 0.0   # Total premium cost
    annual_cost_pct: float = 0.0 # Annualized cost
    
    def calculate(self) -> 'PutHedge':
        """Calculate position sizing and costs"""
        # Standard contract size
        contract_value = self.spot_price * 100
        self.num_contracts = math.ceil(self.notional / contract_value)
        
        # Strike price
        self.strike_price = self.spot_price * self.strike_pct
        
        # Estimate premium using BSM approximation
        # Simplified: premium ≈ intrinsic + time value
        intrinsic = max(0, self.strike_price - self.spot_price)
        time_value = self.spot_price * self.implied_vol * math.sqrt(self.days_to_expiry / 365)
        premium_per_share = intrinsic + time_value * abs(self.delta)
        
        self.total_premium = premium_per_share * self.num_contracts * 100
        self.premium_pct = self.total_premium / self.notional
        
        # Annualized cost (assuming quarterly rolls)
        rolls_per_year = 365 / self.days_to_expiry
        self.annual_cost_pct = self.premium_pct * rolls_per_year
        
        return self


@dataclass
class VixHedge:
    """VIX call overlay specification"""
    portfolio_value: float       # Total portfolio $ value
    vix_spot: float              # Current VIX level
    vix_futures: float           # VIX futures price (basis adjusted)
    strike: float                # VIX call strike
    days_to_expiry: int          # Days until VIX futures expiry
    
    # Sizing parameters
    target_allocation_pct: float = 0.01  # Target 1% allocation
    
    # Calculated fields
    num_contracts: int = 0
    notional_exposure: float = 0.0
    premium_cost: float = 0.0
    convexity_score: float = 0.0  # Expected payoff during vol spike
    
    def calculate(self) -> 'VixHedge':
        """Calculate VIX call position sizing"""
        # VIX futures contract = $1000 per VIX point
        target_notional = self.portfolio_value * self.target_allocation_pct
        
        # Estimate call premium (simplified model)
        # VIX calls are expensive due to upward skew
        moneyness = self.vix_futures / self.strike
        if moneyness < 1.0:  # OTM
            # OTM VIX calls: premium increases with time and inverse moneyness
            premium_per_point = 0.5 + (1.0 - moneyness) * 2.0
        else:  # ITM
            premium_per_point = moneyness * 0.8
            
        contract_premium = premium_per_point * 1000
        
        # Number of contracts based on target allocation
        self.num_contracts = math.ceil(target_notional / (contract_premium * 100))
        self.premium_cost = self.num_contracts * contract_premium
        
        # Notional exposure
        self.notional_exposure = self.num_contracts * self.vix_futures * 1000
        
        # Convexity: expected payoff if VIX doubles
        vix_spike = self.vix_spot * 2
        if vix_spike > self.strike:
            payoff_per_contract = (vix_spike - self.strike) * 1000 - contract_premium
        else:
            payoff_per_contract = -contract_premium
        self.convexity_score = (payoff_per_contract * self.num_contracts) / self.premium_cost
        
        return self


@dataclass
class HybridHedge:
    """Combined put + VIX hedge configuration"""
    portfolio_value: float
    equity_allocation_pct: float
    
    # Component weights (sum to 1.0)
    put_weight: float = 0.6
    vix_weight: float = 0.4
    
    # Parameters
    target_annual_cost_pct: float = 0.01  # 1% annual drag target
    
    # Sub-positions
    put_hedge: Optional[PutHedge] = None
    vix_hedge: Optional[VixHedge] = None
    
    # Aggregated metrics
    total_annual_cost: float = 0.0
    expected_payoff_crisis: float = 0.0     # If SPY -20%, VIX +100%
    efficiency_score: float = 0.0           # Cost/benefit ratio
    
    def optimize(self, market_regime: MarketRegime, spy_price: float, 
                 vix_level: float) -> 'HybridHedge':
        """Optimize hedge mix based on market regime"""
        
        equity_notional = self.portfolio_value * self.equity_allocation_pct
        
        # Regime-based allocation adjustments
        regime_configs = {
            MarketRegime.LOW_VOL: {
                'put_weight': 0.5, 'vix_weight': 0.5,
                'put_delta': -0.20, 'put_dte': 90,
                'vix_strike_mult': 1.5,
                'max_cost': 0.015  # 1.5% acceptable in cheap vol
            },
            MarketRegime.MODERATE_VOL: {
                'put_weight': 0.6, 'vix_weight': 0.4,
                'put_delta': -0.25, 'put_dte': 60,
                'vix_strike_mult': 1.3,
                'max_cost': 0.012
            },
            MarketRegime.ELEVATED_VOL: {
                'put_weight': 0.7, 'vix_weight': 0.3,
                'put_delta': -0.30, 'put_dte': 30,
                'vix_strike_mult': 1.2,
                'max_cost': 0.008  # Reduce size when expensive
            },
            MarketRegime.CRISIS: {
                'put_weight': 0.8, 'vix_weight': 0.2,
                'put_delta': -0.35, 'put_dte': 30,
                'vix_strike_mult': 1.0,
                'max_cost': 0.005  # Minimal hedges in crisis
            }
        }
        
        config = regime_configs.get(market_regime, regime_configs[MarketRegime.MODERATE_VOL])
        
        self.put_weight = config['put_weight']
        self.vix_weight = config['vix_weight']
        
        # Allocate budget
        put_budget = self.target_annual_cost_pct * self.put_weight
        vix_budget = self.target_annual_cost_pct * self.vix_weight
        
        # Build put hedge
        put_iv = vix_level / 100  # Simplified IV estimate
        put_premium_estimate = spy_price * put_iv * math.sqrt(config['put_dte'] / 365) * abs(config['put_delta'])
        
        # Back-calculate contracts from budget
        put_contract_value = spy_price * 100
        put_contracts = math.ceil(equity_notional / put_contract_value)
        put_premium_total = put_premium_estimate * put_contracts * 100
        put_annual_cost = put_premium_total / equity_notional * (365 / config['put_dte'])
        
        # Scale down if over budget
        if put_annual_cost > put_budget:
            scale_factor = put_budget / put_annual_cost
            put_contracts = max(1, int(put_contracts * scale_factor))
            put_premium_total = put_premium_estimate * put_contracts * 100
            put_annual_cost = put_premium_total / equity_notional * (365 / config['put_dte'])
        
        self.put_hedge = PutHedge(
            underlying="SPY",
            notional=equity_notional,
            spot_price=spy_price,
            strike_pct=1.0 + (config['put_delta'] * 0.3),  # Delta proxy for strike
            delta=config['put_delta'],
            days_to_expiry=config['put_dte'],
            implied_vol=put_iv,
            premium_pct=0.0
        )
        self.put_hedge.num_contracts = put_contracts
        self.put_hedge.total_premium = put_premium_total
        self.put_hedge.premium_pct = put_premium_total / equity_notional
        self.put_hedge.annual_cost_pct = put_annual_cost
        
        # Build VIX hedge
        vix_strike = vix_level * config['vix_strike_mult']
        vix_target_notional = self.portfolio_value * vix_budget * 0.5  # Conservative sizing
        
        # Estimate VIX call premium
        vix_futures = vix_level * 1.1  # Typical contango
        moneyness = vix_futures / vix_strike
        vix_premium_per_point = 0.5 + max(0, (1.0 - moneyness)) * 2.0
        vix_contract_premium = vix_premium_per_point * 1000
        
        vix_contracts = math.ceil(vix_target_notional / (vix_contract_premium * 10))
        vix_cost = vix_contracts * vix_contract_premium
        
        self.vix_hedge = VixHedge(
            portfolio_value=self.portfolio_value,
            vix_spot=vix_level,
            vix_futures=vix_futures,
            strike=vix_strike,
            days_to_expiry=config['put_dte'],
            target_allocation_pct=vix_budget * 0.5
        )
        self.vix_hedge.num_contracts = vix_contracts
        self.vix_hedge.premium_cost = vix_cost
        
        # Aggregate metrics
        self.total_annual_cost = put_annual_cost + (vix_cost / self.portfolio_value * (365 / config['put_dte']))
        
        # Expected payoff if SPY -20%, VIX doubles to 2x
        spy_drop = 0.20
        vix_spike = vix_level * 2
        
        put_intrinsic = spy_price * (self.put_hedge.strike_pct - (1 - spy_drop))
        put_payoff = max(0, put_intrinsic) * put_contracts * 100 - put_premium_total
        
        if vix_spike > vix_strike:
            vix_payoff = (vix_spike - vix_strike) * 1000 * vix_contracts - vix_cost
        else:
            vix_payoff = -vix_cost
            
        self.expected_payoff_crisis = put_payoff + vix_payoff
        self.efficiency_score = self.expected_payoff_crisis / (self.total_annual_cost * self.portfolio_value)
        
        return self


@dataclass
class HedgeAnalytics:
    """Comprehensive hedge cost/benefit analysis"""
    hedge_type: HedgeType
    
    # Cost metrics
    annual_premium: float
    annual_premium_pct: float
    
    # Protection metrics
    max_protection_notional: float
    protection_pct: float  # % of portfolio covered
    
    # Scenario payoffs
    payoff_5pct_drop: float
    payoff_10pct_drop: float
    payoff_20pct_drop: float
    payoff_black_swan: float  # -40% equity crash
    
    # Risk-adjusted metrics
    cost_benefit_ratio: float  # Premium / Expected payoff
    sharpe_improvement: float  # Estimated impact on portfolio Sharpe
    max_dd_reduction: float    # Estimated max drawdown reduction
    
    # Breakeven
    breakeven_move_pct: float  # Underlying move needed to breakeven


class TailRiskHedger:
    """Main tail risk hedging engine"""
    
    # Constants
    VIX_CRISIS_THRESHOLD = 35.0
    VIX_ELEVATED_THRESHOLD = 25.0
    VIX_LOW_THRESHOLD = 15.0
    
    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        
    def detect_regime(self, vix_level: float, spy_realized_vol: Optional[float] = None) -> MarketRegime:
        """Determine current market regime for hedge sizing"""
        if vix_level > self.VIX_CRISIS_THRESHOLD:
            return MarketRegime.CRISIS
        elif vix_level > self.VIX_ELEVATED_THRESHOLD:
            return MarketRegime.ELEVATED_VOL
        elif vix_level < self.VIX_LOW_THRESHOLD:
            return MarketRegime.LOW_VOL
        else:
            return MarketRegime.MODERATE_VOL
    
    def calculate_protective_put(self, underlying: str, shares: int, 
                                current_price: float, days_to_expiry: int = 90,
                                delta_target: float = -0.20) -> PutHedge:
        """Calculate protective put position"""
        # Estimate strike from delta target (simplified)
        # 20-delta puts typically 5-8% OTM
        otm_pct = abs(delta_target) * 0.3  # ~6% for 20 delta
        strike_pct = 1.0 - otm_pct
        
        # Estimate IV from VIX proxy
        implied_vol = 0.20  # Default 20% IV
        
        notional = shares * current_price
        
        hedge = PutHedge(
            underlying=underlying,
            notional=notional,
            spot_price=current_price,
            strike_pct=strike_pct,
            delta=delta_target,
            days_to_expiry=days_to_expiry,
            implied_vol=implied_vol,
            premium_pct=0.0
        )
        return hedge.calculate()
    
    def calculate_vix_overlay(self, vix_current: float, 
                             target_allocation_pct: float = 0.01) -> VixHedge:
        """Calculate VIX call overlay"""
        # VIX futures typically trade at premium to spot (contango)
        vix_futures = vix_current * 1.1
        
        # Strike selection: 1.3-1.5x current VIX
        if vix_current < 15:
            strike_mult = 1.5
        elif vix_current < 25:
            strike_mult = 1.3
        else:
            strike_mult = 1.2
            
        hedge = VixHedge(
            portfolio_value=self.portfolio_value,
            vix_spot=vix_current,
            vix_futures=vix_futures,
            strike=vix_current * strike_mult,
            days_to_expiry=60,
            target_allocation_pct=target_allocation_pct
        )
        return hedge.calculate()
    
    def optimize_hybrid(self, spy_price: float, vix_level: float,
                       equity_allocation_pct: float = 0.50,
                       max_annual_cost_pct: float = 0.015) -> HybridHedge:
        """Optimize hybrid hedge based on current conditions"""
        regime = self.detect_regime(vix_level)
        
        hedge = HybridHedge(
            portfolio_value=self.portfolio_value,
            equity_allocation_pct=equity_allocation_pct,
            target_annual_cost_pct=max_annual_cost_pct
        )
        return hedge.optimize(regime, spy_price, vix_level)
    
    def analytics(self, hedge: HybridHedge) -> HedgeAnalytics:
        """Generate comprehensive hedge analytics"""
        equity_notional = hedge.portfolio_value * hedge.equity_allocation_pct
        
        # Calculate protection coverage
        if hedge.put_hedge:
            put_protection = hedge.put_hedge.strike_price * hedge.put_hedge.num_contracts * 100
            protection_pct = put_protection / equity_notional
        else:
            put_protection = 0
            protection_pct = 0
            
        # Scenario payoffs
        scenarios = {
            'payoff_5pct_drop': -0.05,
            'payoff_10pct_drop': -0.10,
            'payoff_20pct_drop': -0.20,
            'payoff_black_swan': -0.40
        }
        
        payoffs = {}
        for name, drop in scenarios.items():
            # Put payoff
            if hedge.put_hedge:
                put_intrinsic = max(0, hedge.put_hedge.strike_price - (hedge.put_hedge.spot_price * (1 + drop)))
                put_payoff = put_intrinsic * hedge.put_hedge.num_contracts * 100 - hedge.put_hedge.total_premium
            else:
                put_payoff = 0
                
            # VIX payoff (assume VIX spikes 0.5x per 10% equity drop)
            vix_spike = 1 + abs(drop) * 5  # 2.5x for -50% drop
            if hedge.vix_hedge:
                vix_level_spike = hedge.vix_hedge.vix_spot * vix_spike
                if vix_level_spike > hedge.vix_hedge.strike:
                    vix_intrinsic = (vix_level_spike - hedge.vix_hedge.strike) * 1000 * hedge.vix_hedge.num_contracts
                else:
                    vix_intrinsic = 0
                vix_payoff = vix_intrinsic - hedge.vix_hedge.premium_cost
            else:
                vix_payoff = 0
                
            payoffs[name] = put_payoff + vix_payoff
            
        # Cost/benefit (annual cost vs expected 20% drop payoff)
        expected_benefit = payoffs['payoff_20pct_drop']
        annual_cost = hedge.total_annual_cost * hedge.portfolio_value
        cost_benefit = annual_cost / expected_benefit if expected_benefit > 0 else float('inf')
        
        # Estimate Sharpe improvement (simplified)
        # Hedge drag reduces returns by cost but reduces vol
        vol_reduction = 0.15  # Assume 15% vol reduction from hedge
        sharpe_improvement = vol_reduction * 0.5 - hedge.total_annual_cost
        
        # Breakeven (move needed for hedge to pay for itself)
        breakeven = 0.0
        if hedge.put_hedge:
            breakeven = hedge.put_hedge.premium_pct / abs(hedge.put_hedge.delta)
            
        return HedgeAnalytics(
            hedge_type=HedgeType.HYBRID,
            annual_premium=annual_cost,
            annual_premium_pct=hedge.total_annual_cost,
            max_protection_notional=put_protection,
            protection_pct=protection_pct,
            **payoffs,
            cost_benefit_ratio=cost_benefit,
            sharpe_improvement=sharpe_improvement,
            max_dd_reduction=vol_reduction,
            breakeven_move_pct=breakeven * 100
        )
    
    def rolling_schedule(self, current_dte: int = 30) -> dict:
        """Generate rolling hedge management schedule"""
        now = datetime.now()
        
        if current_dte <= 7:
            action = "ROLL IMMEDIATELY"
            urgency = "HIGH"
        elif current_dte <= 21:
            action = "PLAN ROLL"
            urgency = "MEDIUM"
        else:
            action = "MONITOR"
            urgency = "LOW"
            
        # Optimal roll timing
        optimal_roll = now + timedelta(days=max(0, current_dte - 21))
        
        return {
            'current_dte': current_dte,
            'action': action,
            'urgency': urgency,
            'optimal_roll_date': optimal_roll.strftime('%Y-%m-%d'),
            'next_expiry_options': self._get_quarterly_expiry(optimal_roll),
            'notes': [
                'Roll before 21 DTE to minimize gamma risk',
                'Avoid rolling during monthly OpEx week (third Friday)',
                'Consider volatility skew when selecting strikes'
            ]
        }
    
    def _get_quarterly_expiry(self, target_date: datetime) -> str:
        """Get next quarterly expiration (3rd Friday of Mar/Jun/Sep/Dec)"""
        month = target_date.month
        year = target_date.year
        
        # Find quarterly months
        quarterly = [3, 6, 9, 12]
        next_q = next((m for m in quarterly if m >= month), 3)
        if next_q < month:
            year += 1
            
        # Third Friday
        import calendar
        cal = calendar.monthcalendar(year, next_q)
        fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
        third_friday = fridays[2]
        
        return f"{year}-{next_q:02d}-{third_friday:02d}"


def cmd_analyze(args):
    """Run tail risk hedge analysis"""
    hedger = TailRiskHedger(portfolio_value=args.portfolio)
    
    regime = hedger.detect_regime(args.vix)
    print(f"\n📊 Tail Risk Hedge Analysis")
    print(f"{'='*60}")
    print(f"Portfolio Value: ${args.portfolio:,.0f}")
    print(f"SPY Price: ${args.spy:.2f}")
    print(f"VIX Level: {args.vix:.2f}")
    print(f"Detected Regime: {regime.value.upper().replace('_', ' ')}")
    
    # Optimize hybrid hedge
    hedge = hedger.optimize_hybrid(
        spy_price=args.spy,
        vix_level=args.vix,
        equity_allocation_pct=args.equity_pct,
        max_annual_cost_pct=args.max_cost
    )
    
    print(f"\n🛡️  Hybrid Hedge Configuration")
    print(f"{'-'*60}")
    
    if hedge.put_hedge:
        print(f"\nProtective Put (SPY):")
        print(f"  Contracts: {hedge.put_hedge.num_contracts}")
        print(f"  Strike: ${hedge.put_hedge.strike_price:.2f} ({hedge.put_hedge.strike_pct*100:.1f}%)")
        print(f"  Days to Expiry: {hedge.put_hedge.days_to_expiry}")
        print(f"  Premium: ${hedge.put_hedge.total_premium:,.0f} ({hedge.put_hedge.premium_pct*100:.2f}%)")
        print(f"  Annual Cost: {hedge.put_hedge.annual_cost_pct*100:.2f}%")
    
    if hedge.vix_hedge:
        print(f"\nVIX Call Overlay:")
        print(f"  Contracts: {hedge.vix_hedge.num_contracts}")
        print(f"  Strike: {hedge.vix_hedge.strike:.1f}")
        print(f"  Cost: ${hedge.vix_hedge.premium_cost:,.0f}")
        print(f"  Convexity: {hedge.vix_hedge.convexity_score:.1f}x")
    
    print(f"\n📈 Aggregate Metrics")
    print(f"{'-'*60}")
    print(f"Total Annual Cost: {hedge.total_annual_cost*100:.2f}%")
    print(f"Expected Crisis Payoff: ${hedge.expected_payoff_crisis:,.0f}")
    print(f"Efficiency Score: {hedge.efficiency_score:.2f}x")
    
    # Analytics
    analytics = hedger.analytics(hedge)
    print(f"\n📊 Scenario Payoffs")
    print(f"{'-'*60}")
    print(f"  -5% Market Drop: ${analytics.payoff_5pct_drop:,.0f}")
    print(f"  -10% Market Drop: ${analytics.payoff_10pct_drop:,.0f}")
    print(f"  -20% Market Drop: ${analytics.payoff_20pct_drop:,.0f}")
    print(f"  Black Swan (-40%): ${analytics.payoff_black_swan:,.0f}")
    
    print(f"\n🎯 Risk Metrics")
    print(f"{'-'*60}")
    print(f"Cost/Benefit Ratio: {analytics.cost_benefit_ratio:.2f}")
    print(f"Sharpe Improvement: {analytics.sharpe_improvement*100:+.1f}%")
    print(f"Max DD Reduction: {analytics.max_dd_reduction*100:.1f}%")
    print(f"Breakeven Move: {analytics.breakeven_move_pct:.1f}%")
    
    # Rolling schedule
    schedule = hedger.rolling_schedule(hedge.put_hedge.days_to_expiry if hedge.put_hedge else 30)
    print(f"\n🔄 Rolling Schedule")
    print(f"{'-'*60}")
    print(f"Action: {schedule['action']}")
    print(f"Urgency: {schedule['urgency']}")
    print(f"Optimal Roll: {schedule['optimal_roll_date']}")
    print(f"Next Expiry: {schedule['next_expiry_options']}")
    
    if args.output:
        result = {
            'regime': regime.value,
            'hedge': {
                'put': asdict(hedge.put_hedge) if hedge.put_hedge else None,
                'vix': asdict(hedge.vix_hedge) if hedge.vix_hedge else None,
                'total_annual_cost': hedge.total_annual_cost,
                'efficiency_score': hedge.efficiency_score
            },
            'analytics': asdict(analytics),
            'schedule': schedule
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n📄 Output saved to: {args.output}")


def cmd_compare(args):
    """Compare different hedge strategies"""
    hedger = TailRiskHedger(portfolio_value=args.portfolio)
    
    print(f"\n🔍 Hedge Strategy Comparison")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'Annual Cost':<12} {'20% Crash':<12} {'Efficiency':<12}")
    print(f"{'-'*70}")
    
    # Test different configurations
    configs = [
        ('Conservative Puts', 0.7, 0.3, 0.012),
        ('Balanced Hybrid', 0.5, 0.5, 0.015),
        ('VIX Heavy', 0.3, 0.7, 0.015),
        ('Minimal Hedge', 0.5, 0.5, 0.008),
    ]
    
    for name, put_w, vix_w, cost in configs:
        hedge = HybridHedge(
            portfolio_value=args.portfolio,
            equity_allocation_pct=0.50,
            put_weight=put_w,
            vix_weight=vix_w,
            target_annual_cost_pct=cost
        )
        hedge.optimize(hedger.detect_regime(args.vix), args.spy, args.vix)
        
        print(f"{name:<25} {hedge.total_annual_cost*100:>6.2f}%      "
              f"${hedge.expected_payoff_crisis/1000:>6.1f}K     "
              f"{hedge.efficiency_score:.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description='v2.42 Tail Risk Hedging Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze --spy 580 --vix 18 --portfolio 100000
  %(prog)s analyze --spy 580 --vix 32 --equity-pct 0.6
  %(prog)s compare --spy 580 --vix 22 --portfolio 100000
        """
    )
    
    parser.add_argument('--spy', type=float, default=580, help='SPY price (default: 580)')
    parser.add_argument('--vix', type=float, default=20, help='VIX level (default: 20)')
    parser.add_argument('--portfolio', type=float, default=100000, help='Portfolio value')
    parser.add_argument('--equity-pct', type=float, default=0.50, help='Equity allocation %')
    parser.add_argument('--max-cost', type=float, default=0.015, help='Max annual hedge cost %')
    parser.add_argument('-o', '--output', type=str, help='JSON output file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze hedge configuration')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare strategies')
    compare_parser.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if not args.command:
        # Default to analyze
        args.command = 'analyze'
        args.func = cmd_analyze
    
    args.func(args)


if __name__ == '__main__':
    main()
