#!/usr/bin/env python3
"""
Portfolio-Lab v2.42: Tail Risk Hedge Calculator

Phase 1 implementation of protective put and VIX call overlay strategies
for portfolio tail risk hedging.

Based on synthesis from deep research:
- CBOE VXTH Index methodology
- CME Group tail risk hedging research 2024
- Graham Capital crisis alpha analysis

Usage:
    python -m src.risk.tail_hedge_calculator analyze --portfolio 46/38/16 --notional 100000
    python -m src.risk.tail_hedge_calculator vix-signal --current-vix 18.5
    python -m src.risk.tail_hedge_calculator cost --strike-pct 0.95 --dte 60
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class HedgeStrategy(Enum):
    """Available tail hedge strategies"""
    PROTECTIVE_PUT = "protective_put"      # Long OTM puts on underlying
    VIX_CALL = "vix_call"                  # Long VIX calls
    VIX_CALL_SPREAD = "vix_call_spread"    # Long VIX call spreads (cheaper)
    COLLAR = "collar"                      # Short call + long put
    HYBRID = "hybrid"                      # Protective puts + VIX calls


class HedgeAction(Enum):
    """Recommended hedge actions"""
    ENTER = "enter"                        # Initiate hedge position
    HOLD = "hold"                          # Maintain existing hedge
    ROLL = "roll"                          # Roll to new expiration
    TAKE_PROFIT = "take_profit"           # Close profitable hedge
    NO_ACTION = "no_action"               # No hedge warranted


@dataclass
class ProtectivePutConfig:
    """Configuration for protective put hedge"""
    underlying: str                       # Ticker (SPY, QQQ, etc.)
    strike_pct: float = 0.95             # Strike as % of spot (5% OTM)
    delta_target: float = 0.30           # Target delta (30 delta typical)
    dte: int = 60                        # Days to expiration
    max_hedge_notional: float = 0.02    # Max 2% of portfolio for hedges
    
    # Cost parameters (estimated for liquid ETFs)
    implied_vol_premium: float = 0.18     # ~18% IV for SPY 60d OTM
    theta_decay_daily: float = 0.0015     # Estimated daily theta


@dataclass
class VIXCallConfig:
    """Configuration for VIX call hedge"""
    strike_vix: float = 22.0             # Strike level
    dte: int = 90                        # VIX calls need longer (vol mean reversion)
    max_contracts: int = 10              # Position limit
    position_size_pct: float = 0.01      # 1% of portfolio max
    
    # Entry/exit thresholds
    vix_entry_low: float = 15.0          # Enter when VIX < 15
    vix_entry_high: float = 22.0         # Don't enter above 22
    vix_exit_profit: float = 35.0        # Take profit when VIX > 35
    vix_exit_stop: float = 40.0         # Emergency exit


@dataclass
class HedgeRecommendation:
    """Complete hedge recommendation"""
    timestamp: str
    portfolio_value: float
    
    # Current market conditions
    vix_spot: float
    vix_percentile: float               # VIX percentile (0-100)
    underlying_spot: Dict[str, float]
    portfolio_ath_distance: float      # % from all-time high
    
    # Recommended action
    action: HedgeAction
    strategy: HedgeStrategy
    
    # Position details
    contracts: int
    strike: float
    expiry: str
    premium: float                      # Estimated cost
    premium_pct: float                  # As % of portfolio
    
    # Risk metrics
    max_loss: float                    # Max loss (premium)
    breakeven: float                   # Underlying level for profit
    expected_payout_crisis: float       # Estimated payout if VIX > 35
    
    # Rationale
    rationale: str
    alternative_actions: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


class TailRiskHedgeCalculator:
    """
    Calculates optimal tail risk hedges for portfolio protection.
    
    Implements:
    - Protective put sizing (strike, expiration, delta selection)
    - VIX call overlay sizing and timing
    - Hybrid optimization (puts + VIX)
    - Cost/benefit analytics
    """
    
    # Historical VIX data for percentile calculations
    VIX_HISTORICAL = {
        "mean": 19.5,
        "median": 17.8,
        "p10": 12.5,
        "p25": 14.2,
        "p75": 23.4,
        "p90": 28.6,
        "min": 9.1,
        "max": 82.7,  # Mar 2020
    }
    
    # VIX futures term structure (typical)
    VIX_TERM_STRUCTURE = {
        30: 1.05,    # Front month ~5% contango
        60: 1.12,
        90: 1.18,
        180: 1.25,
    }
    
    def __init__(
        self,
        portfolio_value: float = 100000.0,
        base_allocation: Dict[str, float] = None
    ):
        self.portfolio_value = portfolio_value
        self.base_allocation = base_allocation or {
            "SPY": 0.46,
            "GLD": 0.38,
            "TLT": 0.16
        }
        
    def _calculate_vix_percentile(self, vix: float) -> float:
        """Calculate VIX percentile (0-100) based on historical distribution"""
        # Simplified percentile using normal approximation of log(VIX)
        import math
        log_vix = math.log(vix)
        log_mean = math.log(self.VIX_HISTORICAL["mean"])
        log_std = 0.35  # Approximate from historical data
        
        z_score = (log_vix - log_mean) / log_std
        # Convert to percentile using normal CDF approximation
        percentile = 50 * (1 + math.erf(z_score / math.sqrt(2)))
        return min(100, max(0, percentile))
    
    def _estimate_put_premium(
        self,
        underlying: str,
        spot: float,
        strike: float,
        dte: int,
        iv: float = None
    ) -> float:
        """
        Estimate put option premium using Black-Scholes approximation.
        Returns per-contract premium (100 shares).
        """
        if iv is None:
            # Default IV by underlying and DTE
            iv_map = {
                "SPY": {30: 0.16, 60: 0.18, 90: 0.19},
                "QQQ": {30: 0.19, 60: 0.21, 90: 0.22},
                "GLD": {30: 0.15, 60: 0.16, 90: 0.17},
                "TLT": {30: 0.12, 60: 0.13, 90: 0.14},
            }
            iv = iv_map.get(underlying, {}).get(dte, 0.18)
        
        # Simplified B-S for ATM/OTM puts
        # Premium ≈ strike * IV * sqrt(T) for OTM puts
        t = dte / 365.0
        moneyness = strike / spot
        
        # Base premium
        base_premium = strike * iv * math.sqrt(t)
        
        # Adjust for moneyness (OTM cheaper)
        if moneyness < 1.0:
            base_premium *= (0.5 + 0.5 * moneyness)  # 95% strike = ~97.5% of ATM price
        
        # Contract multiplier (100 shares)
        return base_premium * 100
    
    def _estimate_vix_call_premium(
        self,
        vix_spot: float,
        strike: float,
        dte: int
    ) -> float:
        """
        Estimate VIX call premium.
        VIX options have unique dynamics due to futures basis.
        """
        # VIX call pricing is complex due to term structure
        # Simplified model: premium increases with (strike - futures price)
        
        # Estimate VIX futures price from term structure
        term_mult = self.VIX_TERM_STRUCTURE.get(dte, 1.15)
        futures_price = vix_spot * term_mult
        
        # Intrinsic value approximation
        intrinsic = max(0, futures_price - strike)
        
        # Time value (high for VIX due to volatility of volatility)
        time_value = vix_spot * 0.25 * math.sqrt(dte / 365)
        
        premium = (intrinsic + time_value) * 100  # Per contract
        return premium
    
    def analyze_protective_put(
        self,
        underlying: str = "SPY",
        current_price: float = None,
        config: ProtectivePutConfig = None
    ) -> Dict:
        """Analyze protective put hedge for single underlying"""
        if config is None:
            config = ProtectivePutConfig(underlying=underlying)
        
        if current_price is None:
            # Use recent prices as default
            default_prices = {"SPY": 585.0, "QQQ": 490.0, "GLD": 240.0, "TLT": 95.0}
            current_price = default_prices.get(underlying, 500.0)
        
        # Calculate position size
        position_value = self.portfolio_value * self.base_allocation.get(underlying, 0.33)
        shares_held = position_value / current_price
        
        # Determine strike
        strike = current_price * config.strike_pct
        
        # Estimate contracts needed (100 shares per contract)
        contracts = int(shares_held / 100)
        if contracts < 1:
            contracts = 1  # Minimum hedge
        
        # Estimate premium
        premium_per_contract = self._estimate_put_premium(
            underlying, current_price, strike, config.dte
        )
        total_premium = premium_per_contract * contracts
        premium_pct = total_premium / self.portfolio_value
        
        # Cost check
        if premium_pct > config.max_hedge_notional:
            # Scale down
            max_premium = self.portfolio_value * config.max_hedge_notional
            contracts = int(max_premium / premium_per_contract)
            total_premium = premium_per_contract * contracts
            premium_pct = total_premium / self.portfolio_value
        
        return {
            "underlying": underlying,
            "current_price": current_price,
            "shares_held": int(shares_held),
            "strike": round(strike, 2),
            "strike_pct": config.strike_pct,
            "contracts": contracts,
            "dte": config.dte,
            "expiry": (datetime.now() + timedelta(days=config.dte)).strftime("%Y-%m-%d"),
            "premium_per_contract": round(premium_per_contract, 2),
            "total_premium": round(total_premium, 2),
            "premium_pct": round(premium_pct * 100, 2),
            "max_loss": round(total_premium, 2),
            "breakeven": round(strike - (total_premium / (contracts * 100)), 2),
            "cost_ok": premium_pct <= config.max_hedge_notional,
        }
    
    def analyze_vix_overlay(
        self,
        vix_spot: float,
        config: VIXCallConfig = None
    ) -> Dict:
        """Analyze VIX call overlay hedge"""
        if config is None:
            config = VIXCallConfig()
        
        # Determine entry recommendation
        if vix_spot < config.vix_entry_low:
            action = HedgeAction.ENTER
            rationale = f"VIX {vix_spot:.1f} below entry threshold ({config.vix_entry_low})"
        elif vix_spot < config.vix_entry_high:
            action = HedgeAction.ENTER
            rationale = f"VIX {vix_spot:.1f} in entry zone ({config.vix_entry_low}-{config.vix_entry_high})"
        elif vix_spot >= config.vix_exit_profit and vix_spot < config.vix_exit_stop:
            action = HedgeAction.TAKE_PROFIT
            rationale = f"VIX {vix_spot:.1f} elevated - take profit if holding"
        elif vix_spot >= config.vix_exit_stop:
            action = HedgeAction.NO_ACTION
            rationale = f"VIX {vix_spot:.1f} too high for entry"
        else:
            action = HedgeAction.NO_ACTION
            rationale = f"VIX {vix_spot:.1f} neutral zone"
        
        # Calculate position size
        max_position_value = self.portfolio_value * config.position_size_pct
        
        # Estimate premium
        premium_per_contract = self._estimate_vix_call_premium(
            vix_spot, config.strike_vix, config.dte
        )
        
        contracts = int(max_position_value / premium_per_contract)
        contracts = min(contracts, config.max_contracts)
        
        total_premium = premium_per_contract * contracts
        
        # Expected payout in crisis (VIX > 35)
        crisis_vix = 40.0
        intrinsic_crisis = max(0, crisis_vix - config.strike_vix)
        expected_payout = intrinsic_crisis * 100 * contracts  # Simplified
        
        # VIX percentile
        vix_pct = self._calculate_vix_percentile(vix_spot)
        
        return {
            "vix_spot": vix_spot,
            "vix_percentile": round(vix_pct, 1),
            "action": action.value,
            "rationale": rationale,
            "strike": config.strike_vix,
            "dte": config.dte,
            "expiry": (datetime.now() + timedelta(days=config.dte)).strftime("%Y-%m-%d"),
            "contracts": contracts,
            "premium_per_contract": round(premium_per_contract, 2),
            "total_premium": round(total_premium, 2),
            "premium_pct": round(total_premium / self.portfolio_value * 100, 2),
            "max_loss": round(total_premium, 2),
            "expected_payout_crisis": round(expected_payout, 2),
            "cost_benefit_ratio": round(expected_payout / total_premium, 2) if total_premium > 0 else 0,
        }
    
    def get_full_recommendation(
        self,
        vix_spot: float,
        portfolio_distance_from_ath: float = 0.0,
        underlying_prices: Dict[str, float] = None
    ) -> HedgeRecommendation:
        """
        Generate complete hedge recommendation based on market conditions.
        
        Args:
            vix_spot: Current VIX level
            portfolio_distance_from_ath: % distance from all-time high (negative if below)
            underlying_prices: Dict of {ticker: price}
        """
        if underlying_prices is None:
            underlying_prices = {"SPY": 585.0, "QQQ": 490.0, "GLD": 240.0, "TLT": 95.0}
        
        # Analyze both strategies
        vix_analysis = self.analyze_vix_overlay(vix_spot)
        
        # Select primary hedge
        if vix_analysis["action"] == "enter":
            strategy = HedgeStrategy.VIX_CALL
            action = HedgeAction.ENTER
            primary = vix_analysis
            rationale = f"VIX {vix_spot:.1f} ({vix_analysis['vix_percentile']:.0f}th percentile) - favorable for VIX call entry"
        elif portfolio_distance_from_ath < -0.05:  # >5% drawdown
            strategy = HedgeStrategy.PROTECTIVE_PUT
            action = HedgeAction.ENTER
            primary = self.analyze_protective_put("SPY", underlying_prices.get("SPY"))
            rationale = f"Portfolio in drawdown ({portfolio_distance_from_ath:.1%}) - protective puts recommended"
        else:
            strategy = HedgeStrategy.HYBRID
            action = HedgeAction.NO_ACTION
            primary = vix_analysis
            rationale = "No immediate tail risk signals - maintain standard risk management"
        
        return HedgeRecommendation(
            timestamp=datetime.now().isoformat(),
            portfolio_value=self.portfolio_value,
            vix_spot=vix_spot,
            vix_percentile=vix_analysis["vix_percentile"],
            underlying_spot=underlying_prices,
            portfolio_ath_distance=portfolio_distance_from_ath,
            action=action,
            strategy=strategy,
            contracts=primary.get("contracts", 0),
            strike=primary.get("strike", 0),
            expiry=primary.get("expiry", ""),
            premium=primary.get("total_premium", 0),
            premium_pct=primary.get("premium_pct", 0),
            max_loss=primary.get("max_loss", 0),
            breakeven=primary.get("breakeven", 0),
            expected_payout_crisis=primary.get("expected_payout_crisis", 0),
            rationale=rationale,
            alternative_actions=["Monitor VIX daily", "Review if VIX > 22 or < 15"]
        )


def main():
    parser = argparse.ArgumentParser(
        description="v2.42 Tail Risk Hedge Calculator"
    )
    parser.add_argument(
        "--portfolio", "-p",
        default="46/38/16",
        help="Portfolio allocation (SPY/GLD/TLT)"
    )
    parser.add_argument(
        "--notional", "-n",
        type=float,
        default=100000,
        help="Portfolio notional value"
    )
    parser.add_argument(
        "--vix", "-v",
        type=float,
        default=18.0,
        help="Current VIX level"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Generate full hedge recommendation'
    )
    analyze_parser.add_argument(
        '--distance-from-ath',
        type=float,
        default=0.0,
        help='Portfolio % from all-time high (negative if below)'
    )
    
    # VIX signal command
    vix_parser = subparsers.add_parser(
        'vix-signal',
        help='Analyze VIX call overlay only'
    )
    vix_parser.add_argument(
        '--current-vix',
        type=float,
        required=True,
        help='Current VIX spot'
    )
    
    # Cost command
    cost_parser = subparsers.add_parser(
        'cost',
        help='Calculate protective put cost'
    )
    cost_parser.add_argument(
        '--underlying',
        default='SPY',
        help='Underlying ticker'
    )
    cost_parser.add_argument(
        '--strike-pct',
        type=float,
        default=0.95,
        help='Strike as % of spot'
    )
    cost_parser.add_argument(
        '--dte',
        type=int,
        default=60,
        help='Days to expiration'
    )
    
    args = parser.parse_args()
    
    # Parse portfolio
    weights = [float(w) for w in args.portfolio.split("/")]
    assets = ["SPY", "GLD", "TLT"][:len(weights)]
    allocation = {assets[i]: weights[i]/100 if weights[i] > 1 else weights[i] 
                  for i in range(len(weights))}
    
    calc = TailRiskHedgeCalculator(
        portfolio_value=args.notional,
        base_allocation=allocation
    )
    
    if args.command == 'analyze' or args.command is None:
        rec = calc.get_full_recommendation(
            vix_spot=args.vix,
            portfolio_distance_from_ath=getattr(args, 'distance_from_ath', 0.0)
        )
        
        print("="*70)
        print("TAIL RISK HEDGE RECOMMENDATION v2.42")
        print("="*70)
        print(f"Timestamp: {rec.timestamp}")
        print(f"Portfolio Value: ${rec.portfolio_value:,.0f}")
        print()
        print("MARKET CONDITIONS:")
        print(f"  VIX Spot: {rec.vix_spot:.2f} ({rec.vix_percentile:.0f}th percentile)")
        print(f"  Distance from ATH: {rec.portfolio_ath_distance:+.1%}")
        print()
        print("RECOMMENDATION:")
        print(f"  Action: {rec.action.value.upper()}")
        print(f"  Strategy: {rec.strategy.value}")
        print(f"  Rationale: {rec.rationale}")
        print()
        
        if rec.action != HedgeAction.NO_ACTION:
            print("POSITION DETAILS:")
            print(f"  Contracts: {rec.contracts}")
            print(f"  Strike: {rec.strike:.2f}")
            print(f"  Expiry: {rec.expiry}")
            print(f"  Premium: ${rec.premium:,.0f} ({rec.premium_pct:.2f}% of portfolio)")
            print(f"  Max Loss: ${rec.max_loss:,.0f}")
            print(f"  Expected Payout (Crisis): ${rec.expected_payout_crisis:,.0f}")
        
        print()
        print("ALTERNATIVE ACTIONS:")
        for action in rec.alternative_actions:
            print(f"  - {action}")
        print("="*70)
    
    elif args.command == 'vix-signal':
        analysis = calc.analyze_vix_overlay(args.current_vix)
        
        print("="*70)
        print("VIX CALL OVERLAY ANALYSIS")
        print("="*70)
        print(f"VIX Spot: {analysis['vix_spot']:.2f}")
        print(f"VIX Percentile: {analysis['vix_percentile']:.0f}th")
        print(f"Recommendation: {analysis['action'].upper()}")
        print(f"Rationale: {analysis['rationale']}")
        print()
        print("POSITION:")
        print(f"  Strike: {analysis['strike']:.1f}")
        print(f"  DTE: {analysis['dte']}")
        print(f"  Contracts: {analysis['contracts']}")
        print(f"  Premium: ${analysis['total_premium']:,.0f} ({analysis['premium_pct']:.2f}%)")
        print(f"  Cost-Benefit Ratio: {analysis['cost_benefit_ratio']:.1f}x")
        print("="*70)
    
    elif args.command == 'cost':
        result = calc.analyze_protective_put(
            underlying=args.underlying,
            config=ProtectivePutConfig(
                underlying=args.underlying,
                strike_pct=args.strike_pct,
                dte=args.dte
            )
        )
        
        print("="*70)
        print(f"PROTECTIVE PUT COST ANALYSIS: {args.underlying}")
        print("="*70)
        print(f"Underlying Price: ${result['current_price']:.2f}")
        print(f"Shares Held: {result['shares_held']}")
        print()
        print("HEDGE CONFIGURATION:")
        print(f"  Strike: ${result['strike']:.2f} ({result['strike_pct']:.0%})")
        print(f"  Contracts: {result['contracts']}")
        print(f"  DTE: {result['dte']}")
        print(f"  Expiry: {result['expiry']}")
        print()
        print("COSTS:")
        print(f"  Premium/Contract: ${result['premium_per_contract']:.2f}")
        print(f"  Total Premium: ${result['total_premium']:.2f}")
        print(f"  % of Portfolio: {result['premium_pct']:.2f}%")
        print(f"  Breakeven: ${result['breakeven']:.2f}")
        print(f"  Within Budget: {'✓' if result['cost_ok'] else '✗'}")
        print("="*70)


if __name__ == "__main__":
    main()
