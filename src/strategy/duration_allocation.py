#!/usr/bin/env python3
"""
Duration Allocation Engine - v2.35 Phase 2
Capital Efficiency via Leveraged Treasury ETFs (UBT/TMF)

Integrates leveraged ETF support into the core allocation system:
- UBT (2x): 50% capital requirement for same duration exposure
- TMF (3x): 33% capital requirement (higher risk)
- Automatic fallback to unlevered TLT if limits exceeded

References:
- ProShares UBT Fact Sheet (2025): 0.95% expense, 2x daily leverage
- Direxion TMF Fact Sheet (2025): 1.06% expense, 3x daily leverage
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
ALLOCATION_STATE_PATH = DATA_DIR / ".duration_allocation_state.json"


@dataclass
class LeveragedETFConfig:
    """Configuration for leveraged ETF parameters."""
    symbol: str
    leverage: float
    expense_ratio: float  # Annual fee
    tracking_error: float  # Estimated annual tracking vs theoretical
    volatility_decay: float  # Estimated annual decay in choppy markets
    max_portfolio_pct: float  # Maximum allocation in portfolio
    

# Leveraged ETF registry with risk parameters
LEVERAGED_ETF_REGISTRY: Dict[str, LeveragedETFConfig] = {
    "TLT": LeveragedETFConfig(
        symbol="TLT", leverage=1.0, expense_ratio=0.0015,
        tracking_error=0.0005, volatility_decay=0.0, max_portfolio_pct=1.0
    ),
    "UBT": LeveragedETFConfig(
        symbol="UBT", leverage=2.0, expense_ratio=0.0095,
        tracking_error=0.0015, volatility_decay=0.008, max_portfolio_pct=0.10
    ),
    "TMF": LeveragedETFConfig(
        symbol="TMF", leverage=3.0, expense_ratio=0.0106,
        tracking_error=0.0025, volatility_decay=0.015, max_portfolio_pct=0.05
    ),
    "IEF": LeveragedETFConfig(
        symbol="IEF", leverage=1.0, expense_ratio=0.0015,
        tracking_error=0.0005, volatility_decay=0.0, max_portfolio_pct=1.0
    ),
    "SHY": LeveragedETFConfig(
        symbol="SHY", leverage=1.0, expense_ratio=0.0015,
        tracking_error=0.0003, volatility_decay=0.0, max_portfolio_pct=1.0
    ),
    "BIL": LeveragedETFConfig(
        symbol="BIL", leverage=1.0, expense_ratio=0.0014,
        tracking_error=0.0003, volatility_decay=0.0, max_portfolio_pct=1.0
    ),
}


class DurationAllocationEngine:
    """
    Duration allocation with leveraged ETF support for capital efficiency.
    
    Replaces TLT-only duration exposure with UBT/TMF where appropriate,
    freeing capital for other allocations while maintaining target duration.
    """
    
    # Base duration allocations by regime
    BASE_ALLOCATIONS = {
        "steep": {"TLT": 0.70, "IEF": 0.25, "SHY": 0.05, "BIL": 0.00},  # Long duration
        "normal": {"TLT": 0.50, "IEF": 0.35, "SHY": 0.15, "BIL": 0.00},
        "flat": {"TLT": 0.30, "IEF": 0.40, "SHY": 0.25, "BIL": 0.05},
        "inverted": {"TLT": 0.15, "IEF": 0.25, "SHY": 0.35, "BIL": 0.25},  # Short duration
    }
    
    def __init__(self, portfolio_value: float = 100000.0):
        self.portfolio_value = portfolio_value
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load allocation state from disk."""
        if ALLOCATION_STATE_PATH.exists():
            with open(ALLOCATION_STATE_PATH) as f:
                return json.load(f)
        return {
            "current_regime": "unknown",
            "base_allocation": {},
            "leveraged_allocation": {},
            "capital_freed": 0.0,
            "last_updated": None,
            "ubt_utilized": False,
            "tmf_utilized": False,
        }
    
    def _save_state(self):
        """Save allocation state to disk."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(ALLOCATION_STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_yield_curve_regime(self) -> str:
        """Get current yield curve regime from database or yields.json."""
        yields_file = DATA_DIR / "yields.json"
        
        if yields_file.exists():
            with open(yields_file) as f:
                data = json.load(f)
                if data and "current" in data:
                    return data["current"].get("regime", "normal")
        
        # Fallback: calculate from database
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT dgs2, dgs10 FROM yields 
                    ORDER BY date DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    dgs2, dgs10 = row
                    spread = (dgs10 - dgs2) * 100  # Convert to bps
                    return self._classify_regime(spread)
            except sqlite3.OperationalError:
                pass
            finally:
                conn.close()
        
        return "normal"  # Default
    
    def _classify_regime(self, spread_bps: float) -> str:
        """Classify yield curve regime from 2s10s spread."""
        if spread_bps > 100:
            return "steep"
        elif spread_bps > 50:
            return "normal"
        elif spread_bps > 0:
            return "flat"
        else:
            return "inverted"
    
    def calculate_base_allocation(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Get base (unlevered) allocation for regime."""
        regime = regime or self.get_yield_curve_regime()
        return self.BASE_ALLOCATIONS.get(regime, self.BASE_ALLOCATIONS["normal"]).copy()
    
    def calculate_leveraged_allocation(
        self,
        regime: Optional[str] = None,
        leverage_preference: str = "ubt",  # 'none', 'ubt', 'tmf', 'optimal'
        portfolio_pct: float = 1.0  # Duration allocation as % of total portfolio
    ) -> Dict[str, any]:
        """
        Calculate capital-efficient allocation using leveraged ETFs.
        
        Args:
            regime: Yield curve regime (or auto-detect)
            leverage_preference: 'none', 'ubt' (2x), 'tmf' (3x), 'optimal'
            portfolio_pct: Total duration allocation as % of portfolio (default 16%)
        
        Returns:
            Dict with allocation, capital_freed, and risk metrics
        """
        regime = regime or self.get_yield_curve_regime()
        base = self.calculate_base_allocation(regime)
        
        # Scale base allocation by portfolio_pct
        scaled_base = {k: v * portfolio_pct for k, v in base.items()}
        
        if leverage_preference == "none":
            return {
                "regime": regime,
                "allocation": scaled_base,
                "capital_freed": 0.0,
                "leverage_used": False,
                "expense_drag": self._calculate_expense_drag(scaled_base),
                "duration_exposure": self._calculate_duration_exposure(scaled_base),
            }
        
        # Calculate leveraged allocation
        leveraged = self._apply_leverage(scaled_base, leverage_preference)
        capital_freed = self._calculate_capital_freed(scaled_base, leveraged)
        
        return {
            "regime": regime,
            "allocation": leveraged,
            "capital_freed": capital_freed,
            "leverage_used": leveraged.get("UBT", 0) > 0 or leveraged.get("TMF", 0) > 0,
            "expense_drag": self._calculate_expense_drag(leveraged),
            "volatility_decay_estimate": self._estimate_volatility_decay(leveraged),
            "duration_exposure": self._calculate_duration_exposure(leveraged),
            "risk_score": self._calculate_risk_score(leveraged),
        }
    
    def _apply_leverage(
        self,
        base_allocation: Dict[str, float],
        preference: str
    ) -> Dict[str, float]:
        """Apply leveraged ETFs to long-duration allocation."""
        result = base_allocation.copy()
        tlt_target = base_allocation.get("TLT", 0)
        
        if tlt_target == 0:
            return result  # No long duration to leverage
        
        # Clear TLT, will replace with leveraged equivalent
        result["TLT"] = 0.0
        
        if preference in ["ubt", "optimal"]:
            ubt_config = LEVERAGED_ETF_REGISTRY["UBT"]
            # Capital needed for same exposure: target / leverage
            ubt_capital = tlt_target / ubt_config.leverage
            
            # Respect maximum allocation limit
            max_ubt = ubt_config.max_portfolio_pct
            if ubt_capital <= max_ubt:
                result["UBT"] = ubt_capital
            else:
                # Use max UBT, fill remainder with TLT
                result["UBT"] = max_ubt
                remaining_exposure = tlt_target - (max_ubt * ubt_config.leverage)
                if remaining_exposure > 0:
                    result["TLT"] = remaining_exposure
                    
        elif preference == "tmf":
            tmf_config = LEVERAGED_ETF_REGISTRY["TMF"]
            ubt_config = LEVERAGED_ETF_REGISTRY["UBT"]
            
            tmf_capital = tlt_target / tmf_config.leverage
            max_tmf = tmf_config.max_portfolio_pct
            
            if tmf_capital <= max_tmf:
                result["TMF"] = tmf_capital
            else:
                # Use max TMF, try UBT for remainder, then TLT
                result["TMF"] = max_tmf
                remaining = tlt_target - (max_tmf * tmf_config.leverage)
                
                ubt_for_remainder = remaining / ubt_config.leverage
                max_ubt = ubt_config.max_portfolio_pct
                
                if ubt_for_remainder <= max_ubt:
                    result["UBT"] = ubt_for_remainder
                else:
                    result["UBT"] = max_ubt
                    result["TLT"] = remaining - (max_ubt * ubt_config.leverage)
        
        # Normalize to ensure sum equals original portfolio_pct
        return self._normalize_allocation(result, sum(base_allocation.values()))
    
    def _normalize_allocation(
        self,
        allocation: Dict[str, float],
        target_sum: float
    ) -> Dict[str, float]:
        """Normalize allocation to sum to target."""
        current_sum = sum(allocation.values())
        if current_sum == 0:
            return allocation
        
        scale = target_sum / current_sum
        return {k: v * scale for k, v in allocation.items()}
    
    def _calculate_capital_freed(
        self,
        base: Dict[str, float],
        leveraged: Dict[str, float]
    ) -> float:
        """Calculate capital freed by using leveraged ETFs."""
        base_tlt = base.get("TLT", 0)
        leveraged_ubt = leveraged.get("UBT", 0)
        leveraged_tmf = leveraged.get("TMF", 0)
        leveraged_tlt = leveraged.get("TLT", 0)
        
        # Capital used for long duration in each case
        base_capital = base_tlt
        leveraged_capital = leveraged_ubt + leveraged_tmf + leveraged_tlt
        
        return max(0, base_capital - leveraged_capital)
    
    def _calculate_expense_drag(self, allocation: Dict[str, float]) -> float:
        """Calculate weighted expense ratio for allocation."""
        total = sum(allocation.values())
        if total == 0:
            return 0.0
        
        weighted_expense = sum(
            allocation.get(symbol, 0) * config.expense_ratio
            for symbol, config in LEVERAGED_ETF_REGISTRY.items()
        )
        
        return weighted_expense / total if total > 0 else 0.0
    
    def _estimate_volatility_decay(self, allocation: Dict[str, float]) -> float:
        """Estimate annual volatility decay for leveraged positions."""
        decay = 0.0
        for symbol, weight in allocation.items():
            if symbol in LEVERAGED_ETF_REGISTRY:
                config = LEVERAGED_ETF_REGISTRY[symbol]
                decay += weight * config.volatility_decay
        return decay
    
    def _calculate_duration_exposure(self, allocation: Dict[str, float]) -> float:
        """Calculate effective duration exposure (years)."""
        # Approximate durations: TLT/UBT/TMF ~18.5y, IEF ~7.5y, SHY ~1.9y, BIL ~0.1y
        durations = {
            "TLT": 18.5, "UBT": 18.5, "TMF": 18.5,  # Same underlying
            "IEF": 7.5, "SHY": 1.9, "BIL": 0.1
        }
        
        total_exposure = sum(
            allocation.get(symbol, 0) * durations.get(symbol, 0)
            for symbol in allocation
        )
        
        # For leveraged ETFs, multiply exposure by leverage
        for symbol in ["UBT", "TMF"]:
            if symbol in allocation and allocation[symbol] > 0:
                leverage = LEVERAGED_ETF_REGISTRY[symbol].leverage
                # Adjust: we're already counting the capital amount, 
                # so we need to multiply by leverage to get true exposure
                total_exposure += allocation[symbol] * durations[symbol] * (leverage - 1)
        
        return total_exposure
    
    def _calculate_risk_score(self, allocation: Dict[str, float]) -> float:
        """Calculate risk score based on leverage concentration."""
        score = 0.0
        
        # Higher score = higher risk
        ubt_weight = allocation.get("UBT", 0)
        tmf_weight = allocation.get("TMF", 0)
        
        # UBT at 10% max = +1.0 risk
        score += ubt_weight * 10  # 10% UBT = 1.0 risk
        
        # TMF at 5% max = +1.5 risk (higher due to 3x leverage)
        score += tmf_weight * 30  # 5% TMF = 1.5 risk
        
        return min(score, 3.0)  # Cap at 3.0
    
    def generate_recommendation(self) -> Dict[str, any]:
        """Generate allocation recommendation for current market conditions."""
        regime = self.get_yield_curve_regime()
        
        # Calculate both options
        unlevered = self.calculate_leveraged_allocation(regime, "none")
        levered = self.calculate_leveraged_allocation(regime, "ubt")
        
        # Decision logic
        if regime == "inverted":
            recommendation = "Avoid leverage during inverted curve"
            preferred = unlevered
        elif levered["risk_score"] > 1.5:
            recommendation = "Limit leverage due to risk concentration"
            preferred = unlevered
        elif levered["capital_freed"] < 0.02:  # Less than 2% freed
            recommendation = "Minimal benefit, use unlevered for simplicity"
            preferred = unlevered
        else:
            recommendation = f"Use UBT to free {levered['capital_freed']:.1%} capital"
            preferred = levered
        
        return {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "recommendation": recommendation,
            "preferred_allocation": preferred,
            "unlevered_comparison": unlevered,
            "capital_deployment_options": self._suggest_capital_deployment(levered.get("capital_freed", 0)),
        }
    
    def _suggest_capital_deployment(self, capital_freed: float) -> List[Dict[str, str]]:
        """Suggest ways to deploy freed capital."""
        if capital_freed < 0.01:
            return [{"action": "insufficient_freed_capital", "rationale": "<1% freed, maintain as cash buffer"}]
        
        suggestions = []
        
        if capital_freed >= 0.05:
            suggestions.append({
                "action": "add_ief_barbell",
                "allocation": min(capital_freed * 0.5, 0.10),
                "rationale": "Duration barbell: short + long exposure"
            })
        
        suggestions.append({
            "action": "increase_spy",
            "allocation": min(capital_freed * 0.3, 0.05),
            "rationale": "Tilt toward growth in favorable conditions"
        })
        
        suggestions.append({
            "action": "cash_buffer",
            "allocation": capital_freed * 0.2,
            "rationale": "Maintain liquidity for rebalancing"
        })
        
        return suggestions
    
    def status(self) -> Dict:
        """Get current allocation engine status."""
        return {
            "engine_version": "2.35",
            "current_regime": self.get_yield_curve_regime(),
            "state": self.state,
            "leveraged_etf_support": True,
            "available_etfs": list(LEVERAGED_ETF_REGISTRY.keys()),
        }


class DurationAllocationCLI:
    """Command-line interface for duration allocation engine."""
    
    def __init__(self):
        self.engine = DurationAllocationEngine()
    
    def cmd_status(self):
        """Show current allocation engine status."""
        status = self.engine.status()
        print("Duration Allocation Engine v2.35")
        print("=" * 40)
        print(f"Current Regime: {status['current_regime'].upper()}")
        print(f"Leveraged ETF Support: {'✓' if status['leveraged_etf_support'] else '✗'}")
        print(f"Available ETFs: {', '.join(status['available_etfs'])}")
        
        if status['state'].get('last_updated'):
            print(f"Last Updated: {status['state']['last_updated']}")
    
    def cmd_allocate(self, regime: Optional[str] = None, preference: str = "ubt"):
        """Calculate and display allocation."""
        result = self.engine.calculate_leveraged_allocation(regime, preference)
        
        print(f"\nDuration Allocation ({preference.upper()} preference)")
        print("=" * 50)
        print(f"Regime: {result['regime'].upper()}")
        print(f"Leverage Used: {'Yes' if result['leverage_used'] else 'No'}")
        print(f"Capital Freed: {result['capital_freed']:.2%}")
        print(f"Expense Drag: {result['expense_drag']:.3%}")
        
        if 'volatility_decay_estimate' in result:
            print(f"Est. Volatility Decay: {result['volatility_decay_estimate']:.2%}")
        
        print(f"Duration Exposure: {result['duration_exposure']:.1f} years")
        print(f"Risk Score: {result['risk_score']:.2f}/3.0")
        
        print("\nAllocation:")
        for symbol, weight in sorted(result['allocation'].items()):
            if weight > 0:
                config = LEVERAGED_ETF_REGISTRY.get(symbol)
                leverage_tag = f" ({config.leverage}x)" if config and config.leverage > 1 else ""
                print(f"  {symbol}{leverage_tag}: {weight:.2%}")
    
    def cmd_recommend(self):
        """Show allocation recommendation."""
        rec = self.engine.generate_recommendation()
        
        print("\nAllocation Recommendation")
        print("=" * 40)
        print(f"Regime: {rec['regime'].upper()}")
        print(f"Recommendation: {rec['recommendation']}")
        
        pref = rec['preferred_allocation']
        print(f"\nPreferred Allocation:")
        for symbol, weight in sorted(pref['allocation'].items()):
            if weight > 0:
                print(f"  {symbol}: {weight:.2%}")
        
        if pref.get('capital_freed', 0) > 0:
            print(f"\nCapital Deployment Suggestions:")
            for opt in rec['capital_deployment_options']:
                print(f"  → {opt['action']}: {opt.get('allocation', 0):.2%} - {opt['rationale']}")
    
    def cmd_compare(self):
        """Compare unlevered vs levered allocations."""
        regime = self.engine.get_yield_curve_regime()
        
        print(f"\nAllocation Comparison ({regime.upper()} regime)")
        print("=" * 60)
        
        for pref in ["none", "ubt", "tmf"]:
            result = self.engine.calculate_leveraged_allocation(regime, pref)
            leverage = "1x" if pref == "none" else ("2x" if pref == "ubt" else "3x")
            
            freed = result.get('capital_freed', 0)
            freed_str = f" (+{freed:.1%} freed)" if freed > 0 else ""
            
            print(f"\n{pref.upper()} ({leverage}){freed_str}:")
            for sym, wt in sorted(result['allocation'].items()):
                if wt > 0:
                    marker = "✓" if wt > 0.01 else " "
                    print(f"  {marker} {sym}: {wt:.2%}")


def main():
    """CLI entry point."""
    import sys
    
    cli = DurationAllocationCLI()
    
    if len(sys.argv) < 2:
        cli.cmd_status()
        print("\nUsage: python3 -m src.strategy.duration_allocation <command>")
        print("\nCommands:")
        print("  status       - Show engine status")
        print("  allocate     - Calculate allocation (default: auto-detect regime)")
        print("  recommend    - Show recommendation with capital deployment")
        print("  compare      - Compare unlevered vs levered options")
        return
    
    command = sys.argv[1]
    
    if command == "status":
        cli.cmd_status()
    elif command == "allocate":
        preference = sys.argv[2] if len(sys.argv) > 2 else "ubt"
        cli.cmd_allocate(preference=preference)
    elif command == "recommend":
        cli.cmd_recommend()
    elif command == "compare":
        cli.cmd_compare()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
