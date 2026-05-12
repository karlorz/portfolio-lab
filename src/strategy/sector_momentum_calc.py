"""
Sector Momentum Calculator
v2.40 - Sector Rotation Momentum Implementation
Calculates momentum signals for SPDR sector ETFs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Sector ETF definitions
SECTOR_ETF_DEFINITIONS = [
    {"symbol": "XLK", "name": "Technology", "beta": 1.10, "sectorGroup": "sensitive"},
    {"symbol": "XLV", "name": "Healthcare", "beta": 0.85, "sectorGroup": "defensive"},
    {"symbol": "XLF", "name": "Financials", "beta": 1.05, "sectorGroup": "cyclical"},
    {"symbol": "XLY", "name": "Consumer Discretionary", "beta": 1.15, "sectorGroup": "cyclical"},
    {"symbol": "XLI", "name": "Industrials", "beta": 1.00, "sectorGroup": "cyclical"},
    {"symbol": "XLE", "name": "Energy", "beta": 0.95, "sectorGroup": "sensitive"},
    {"symbol": "XLP", "name": "Consumer Staples", "beta": 0.65, "sectorGroup": "defensive"},
    {"symbol": "XLU", "name": "Utilities", "beta": 0.55, "sectorGroup": "defensive"},
    {"symbol": "XLB", "name": "Materials", "beta": 1.05, "sectorGroup": "sensitive"},
    {"symbol": "XLRE", "name": "Real Estate", "beta": 0.75, "sectorGroup": "sensitive"},
    {"symbol": "XLC", "name": "Communication Services", "beta": 1.00, "sectorGroup": "sensitive"},
]

SECTOR_ETF_MAP = {s["symbol"]: s for s in SECTOR_ETF_DEFINITIONS}

# Regime-aware sector preferences
REGIME_SECTOR_PREFERENCES = {
    "early_expansion": {"preferred": ["XLK", "XLY", "XLF"], "avoid": ["XLU", "XLP"]},
    "late_expansion": {"preferred": ["XLE", "XLB", "XLI"], "avoid": ["XLK", "XLY"]},
    "contraction": {"preferred": ["XLP", "XLV", "XLU"], "avoid": ["XLY", "XLB", "XLE"]},
    "recovery": {"preferred": ["XLF", "XLRE", "XLK"], "avoid": ["XLU", "XLP"]},
    "neutral": {"preferred": [], "avoid": []},
}


class SectorMomentumCalculator:
    """Calculate momentum scores for sector ETFs"""
    
    def __init__(self, historical_data: Dict):
        self.data = historical_data
        
    def calculate_momentum(self, symbol: str, lookback_days: int = 252) -> Optional[Dict]:
        """Calculate momentum for a single sector ETF"""
        if symbol not in self.data:
            return None
            
        prices = self.data[symbol]
        if len(prices) < lookback_days:
            return None
            
        # Sort by date (most recent last)
        sorted_prices = sorted(prices, key=lambda x: x.get("date", x.get("d", "")))
        
        current_price = sorted_prices[-1].get("adjClose", sorted_prices[-1].get("close", 0))
        long_price = sorted_prices[-lookback_days].get("adjClose", sorted_prices[-lookback_days].get("close", 0))
        short_lookback = max(1, lookback_days // 4)  # ~63 days (3 months)
        short_price = sorted_prices[-short_lookback].get("adjClose", sorted_prices[-short_lookback].get("close", 0))
        
        if current_price == 0 or long_price == 0 or short_price == 0:
            return None
            
        long_momentum = (current_price / long_price) - 1
        short_momentum = (current_price / short_price) - 1
        
        # Calculate volatility (annualized)
        returns = []
        for i in range(max(1, len(sorted_prices) - lookback_days), len(sorted_prices)):
            prev = sorted_prices[i-1].get("adjClose", sorted_prices[i-1].get("close", 0))
            curr = sorted_prices[i].get("adjClose", sorted_prices[i].get("close", 0))
            if prev > 0:
                returns.append((curr / prev) - 1)
        
        if len(returns) > 1:
            daily_vol = np.std(returns)
            volatility = daily_vol * np.sqrt(252)  # Annualized
        else:
            volatility = 0.2  # Default 20% vol
            
        # Dual momentum: require both long and short positive
        if long_momentum > 0 and short_momentum > 0:
            composite = (long_momentum + short_momentum) / 2
        else:
            composite = min(long_momentum, short_momentum)
            
        return {
            "symbol": symbol,
            "name": SECTOR_ETF_MAP.get(symbol, {}).get("name", symbol),
            "longMomentum": long_momentum,
            "shortMomentum": short_momentum,
            "compositeMomentum": composite,
            "volatility": volatility,
            "riskAdjustedMomentum": composite / volatility if volatility > 0 else 0,
        }
    
    def calculate_all_momentum(self, lookback_days: int = 252) -> List[Dict]:
        """Calculate momentum for all sector ETFs"""
        results = []
        
        for symbol in SECTOR_ETF_MAP.keys():
            momentum = self.calculate_momentum(symbol, lookback_days)
            if momentum:
                results.append(momentum)
        
        # Sort by composite momentum descending
        results.sort(key=lambda x: x["compositeMomentum"], reverse=True)
        
        # Assign ranks
        for i, r in enumerate(results):
            r["rank"] = i + 1
            r["percentile"] = round(((len(results) - i) / len(results)) * 100)
            
        return results
    
    def adjust_for_regime(self, momentum_scores: List[Dict], regime: str, 
                          preference_boost: float = 0.02) -> List[Dict]:
        """Adjust sector rankings based on regime preferences"""
        prefs = REGIME_SECTOR_PREFERENCES.get(regime, REGIME_SECTOR_PREFERENCES["neutral"])
        
        adjusted = []
        for score in momentum_scores:
            adjusted_momentum = score["compositeMomentum"]
            
            if score["symbol"] in prefs.get("preferred", []):
                adjusted_momentum += preference_boost
            elif score["symbol"] in prefs.get("avoid", []):
                adjusted_momentum -= preference_boost
                
            adjusted.append({
                **score,
                "compositeMomentum": adjusted_momentum,
                "regimeAdjusted": True,
            })
        
        # Re-sort
        adjusted.sort(key=lambda x: x["compositeMomentum"], reverse=True)
        for i, a in enumerate(adjusted):
            a["rank"] = i + 1
            
        return adjusted
    
    def get_allocation(self, momentum_scores: List[Dict], top_n: int = 3,
                      overlay_pct: float = 0.25, spy_weight: float = 0.46,
                      min_momentum: float = 0.0, vix: float = 0,
                      vix_threshold: float = 30) -> Dict:
        """Generate sector overlay allocation"""
        
        # Check VIX threshold - disable rotation in high vol
        if vix > vix_threshold:
            return {
                "spAllocation": spy_weight,
                "sectorAllocations": [],
                "totalEquityWeight": spy_weight,
                "regimeAdjusted": False,
                "regime": None,
                "rebalanceRecommended": False,
                "rebalanceReason": f"VIX {vix:.1f} > threshold {vix_threshold} - sector rotation disabled"
            }
        
        # Filter positive momentum
        positive_sectors = [s for s in momentum_scores if s["compositeMomentum"] >= min_momentum]
        
        if len(positive_sectors) < 1:
            return {
                "spAllocation": spy_weight,
                "sectorAllocations": [],
                "totalEquityWeight": spy_weight,
                "regimeAdjusted": False,
                "regime": None,
                "rebalanceRecommended": False,
                "rebalanceReason": f"Only {len(positive_sectors)} sectors meet momentum threshold"
            }
        
        # Take top N
        top_sectors = positive_sectors[:top_n]
        
        # Calculate allocations
        sector_portion = spy_weight * overlay_pct
        sp_allocation = spy_weight - sector_portion
        
        # Equal weight among top sectors
        sector_weight = sector_portion / len(top_sectors)
        
        sector_allocations = []
        for s in top_sectors:
            sector_allocations.append({
                "symbol": s["symbol"],
                "name": s["name"],
                "weight": sector_weight,
                "momentum": s["compositeMomentum"],
                "rank": s["rank"],
                "volatility": s["volatility"],
            })
        
        total_weight = sp_allocation + sum(s["weight"] for s in sector_allocations)
        
        # Check if rebalance recommended (top momentum > 10%)
        rebalance_needed = top_sectors[0]["compositeMomentum"] > 0.10
        
        return {
            "spAllocation": sp_allocation,
            "sectorAllocations": sector_allocations,
            "totalEquityWeight": total_weight,
            "regimeAdjusted": any(s.get("regimeAdjusted", False) for s in momentum_scores),
            "regime": None,
            "rebalanceRecommended": rebalance_needed,
            "rebalanceReason": f"Top momentum {top_sectors[0]['symbol']} at {top_sectors[0]['compositeMomentum']*100:.1f}%" if rebalance_needed else None
        }


def generate_sector_signals(historical_path: Path, vix: float = 0, regime: str = None) -> Optional[Dict]:
    """Generate complete sector rotation signals for dashboard"""
    try:
        if not historical_path.exists():
            return None
            
        with open(historical_path) as f:
            data = json.load(f)
        
        calculator = SectorMomentumCalculator(data)
        
        # Calculate momentum
        momentum_scores = calculator.calculate_all_momentum(lookback_days=252)
        
        if not momentum_scores:
            return None
        
        # Apply regime adjustment if provided
        if regime and regime != "neutral":
            momentum_scores = calculator.adjust_for_regime(momentum_scores, regime)
        
        # Get allocation
        allocation = calculator.get_allocation(momentum_scores, top_n=3, vix=vix)
        
        # Format output
        top_sectors = momentum_scores[:5]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "vix": vix,
            "regime": regime,
            "methodology": "12-month momentum lookback, top 3 sectors, quarterly rebalancing",
            "overlay_pct": 0.25,
            "top_sectors": [
                {
                    "symbol": s["symbol"],
                    "name": s["name"],
                    "momentumScore": round(s["compositeMomentum"], 4),
                    "allocation": next((a["weight"] for a in allocation["sectorAllocations"] if a["symbol"] == s["symbol"]), 0),
                    "rank": s["rank"],
                    "longMomentum": round(s["longMomentum"], 4),
                    "shortMomentum": round(s["shortMomentum"], 4),
                    "volatility": round(s["volatility"], 4),
                }
                for s in top_sectors
            ],
            "allocation": {
                "spy_core": round(allocation["spAllocation"], 4),
                "spy_total": round(allocation["totalEquityWeight"], 4),
                "sector_overlay": round(sum(a["weight"] for a in allocation["sectorAllocations"]), 4),
                "sectors": [
                    {
                        "symbol": a["symbol"],
                        "weight": round(a["weight"], 4),
                        "momentum": round(a["momentum"], 4),
                        "rank": a["rank"]
                    }
                    for a in allocation["sectorAllocations"]
                ]
            },
            "rebalanceRecommended": allocation["rebalanceRecommended"],
            "rebalanceReason": allocation["rebalanceReason"]
        }
        
    except Exception as e:
        print(f"Error generating sector signals: {e}")
        return None


if __name__ == "__main__":
    # Test
    path = Path("/root/projects/portfolio-lab/public/data/historical.json")
    signals = generate_sector_signals(path, vix=18.5)
    print(json.dumps(signals, indent=2))
