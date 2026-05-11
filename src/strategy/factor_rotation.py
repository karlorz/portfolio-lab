"""
Factor Momentum Rotation Strategy
Rotates between equity factor ETFs based on 12-month momentum

Factors:
- Value (VTV, VLUE): Mean reversion, counter-cyclical
- Momentum (MTUM): Trend continuation
- Quality (QUAL, SPHQ): Profitability, stability
- Low Volatility (USMV, SPLV): Defensive
- Small Cap (IJR, VBR): Size premium

Strategy Logic:
1. Calculate 12-month return for each factor ETF
2. Rank by momentum score
3. Select top N factors (default: 2)
4. Allocate using inverse-volatility weighting
5. Rebalance monthly or on significant momentum shifts
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FactorScore:
    """Momentum and quality metrics for a factor ETF"""
    symbol: str
    factor_name: str
    price: float
    return_12m: float
    return_6m: float
    return_3m: float
    volatility: float  # 20-day annualized
    sharpe_12m: float  # Return/volatility
    momentum_score: float  # Composite score
    rank: int


class FactorMomentumEngine:
    """
    Factor rotation engine based on relative momentum
    
    Universe: US equity factor ETFs
    Lookback: 12 months (primary), 6 months (confirmation)
    Selection: Top 2-3 factors by momentum score
    Weighting: Inverse volatility (risk parity style)
    """
    
    # Factor definitions
    FACTORS = {
        # Value
        "VTV": {"name": "Value", "category": "value", "alternative": "VLUE"},
        "VLUE": {"name": "Value (Alpha)", "category": "value", "alternative": "VTV"},
        # Momentum
        "MTUM": {"name": "Momentum", "category": "momentum", "alternative": None},
        # Quality
        "QUAL": {"name": "Quality", "category": "quality", "alternative": "SPHQ"},
        "SPHQ": {"name": "Quality (S&P)", "category": "quality", "alternative": "QUAL"},
        # Low Volatility
        "USMV": {"name": "Low Vol", "category": "low_vol", "alternative": "SPLV"},
        "SPLV": {"name": "Low Vol (S&P)", "category": "low_vol", "alternative": "USMV"},
        # Small Cap
        "IJR": {"name": "Small Cap", "category": "small", "alternative": "VBR"},
        "VBR": {"name": "Small Value", "category": "small", "alternative": "IJR"},
        # Core equity (benchmark)
        "SPY": {"name": "S&P 500", "category": "core", "alternative": "QQQ"},
        "QQQ": {"name": "Nasdaq 100", "category": "core", "alternative": "SPY"},
    }
    
    # Default allocation weights when factor is selected
    # (inverse volatility weighted at runtime)
    
    def __init__(
        self,
        db_path: Path = Path("~/projects/portfolio-lab/data/market.db").expanduser(),
        lookback_months: int = 12,
        top_n: int = 2,
        min_momentum: float = 0.0,  # Minimum 12m return to qualify
        vol_lookback: int = 20
    ):
        self.db_path = db_path
        self.lookback_months = lookback_months
        self.top_n = top_n
        self.min_momentum = min_momentum
        self.vol_lookback = vol_lookback
        
        # Universe symbols
        self.universe = list(self.FACTORS.keys())
        
        # Factor categories for diversity constraint
        self.max_per_category = 1  # Max 1 per category in selection
    
    def _fetch_price_data(self, symbol: str, days: int = 300) -> List[Dict]:
        """Fetch historical price data from SQLite"""
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, close, volume
            FROM prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """, (symbol, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"date": row[0], "close": row[1], "volume": row[2]}
            for row in reversed(rows)
        ]
    
    def _calculate_factor_score(self, symbol: str) -> Optional[FactorScore]:
        """Calculate momentum metrics for a factor ETF"""
        data = self._fetch_price_data(symbol, 300)
        
        if len(data) < 252:  # Need at least 1 year
            return None
        
        closes = np.array([d["close"] for d in data])
        current_price = closes[-1]
        
        # Returns (approximately 252 trading days/year)
        days_12m = min(252, len(closes) - 1)
        days_6m = min(126, len(closes) - 1)
        days_3m = min(63, len(closes) - 1)
        
        return_12m = (current_price / closes[-days_12m]) - 1 if days_12m > 0 else 0
        return_6m = (current_price / closes[-days_6m]) - 1 if days_6m > 0 else 0
        return_3m = (current_price / closes[-days_3m]) - 1 if days_3m > 0 else 0
        
        # Volatility (20-day)
        returns_daily = np.diff(closes[-self.vol_lookback-1:]) / closes[-self.vol_lookback-1:-1]
        volatility = np.std(returns_daily) * np.sqrt(252) if len(returns_daily) > 1 else 0.15
        
        # Sharpe (using 12m return / 12m volatility)
        returns_12m_period = np.diff(closes[-days_12m:]) / closes[-days_12m:-1]
        vol_12m = np.std(returns_12m_period) * np.sqrt(252) if len(returns_12m_period) > 20 else 0.15
        sharpe_12m = return_12m / vol_12m if vol_12m > 0 else 0
        
        # Composite momentum score
        # 50% 12m + 30% 6m + 20% 3m, adjusted by volatility
        momentum_score = (
            return_12m * 0.5 +
            return_6m * 0.3 +
            return_3m * 0.2
        ) * (0.15 / max(volatility, 0.10))  # Volatility adjustment
        
        return FactorScore(
            symbol=symbol,
            factor_name=self.FACTORS[symbol]["name"],
            price=current_price,
            return_12m=return_12m,
            return_6m=return_6m,
            return_3m=return_3m,
            volatility=volatility,
            sharpe_12m=sharpe_12m,
            momentum_score=momentum_score,
            rank=0  # Set later
        )
    
    def evaluate(self) -> Dict:
        """
        Run factor momentum evaluation and generate rotation signal
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate scores for all factors
        factor_scores = {}
        for symbol in self.universe:
            score = self._calculate_factor_score(symbol)
            if score:
                factor_scores[symbol] = score
        
        if not factor_scores:
            return {
                "timestamp": timestamp,
                "error": "Insufficient data for factor analysis",
                "selected_factors": [],
                "allocation": {},
                "current_scores": {}
            }
        
        # Rank by momentum score
        sorted_factors = sorted(
            factor_scores.items(),
            key=lambda x: x[1].momentum_score,
            reverse=True
        )
        
        # Update ranks
        for rank, (symbol, score) in enumerate(sorted_factors, 1):
            score.rank = rank
        
        # Select top N with diversity constraint
        selected = []
        category_counts = {}
        
        for symbol, score in sorted_factors:
            category = self.FACTORS[symbol]["category"]
            
            # Check minimum momentum
            if score.return_12m < self.min_momentum:
                continue
            
            # Check category diversity
            if category_counts.get(category, 0) >= self.max_per_category:
                continue
            
            selected.append((symbol, score))
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if len(selected) >= self.top_n:
                break
        
        # Generate allocation (inverse volatility weighting)
        allocation = self._generate_allocation(selected)
        
        # Build output
        return {
            "timestamp": timestamp,
            "selected_factors": [s[0] for s in selected],
            "allocation": allocation,
            "current_scores": {
                symbol: {
                    "factor_name": score.factor_name,
                    "category": self.FACTORS[symbol]["category"],
                    "return_12m": float(score.return_12m),
                    "return_6m": float(score.return_6m),
                    "return_3m": float(score.return_3m),
                    "volatility": float(score.volatility),
                    "sharpe_12m": float(score.sharpe_12m),
                    "momentum_score": float(score.momentum_score),
                    "rank": score.rank
                }
                for symbol, score in factor_scores.items()
            },
            "diversity": {
                "categories_used": list(category_counts.keys()),
                "category_distribution": category_counts
            },
            "signal_strength": self._calculate_signal_strength(selected),
            "recommendation": self._generate_recommendation(selected, factor_scores)
        }
    
    def _generate_allocation(self, selected: List[Tuple[str, FactorScore]]) -> Dict[str, float]:
        """Generate inverse-volatility weighted allocation"""
        if not selected:
            return {"SPY": 1.0}  # Default to market if no factors selected
        
        # Calculate inverse volatility weights
        inv_vols = {}
        for symbol, score in selected:
            inv_vols[symbol] = 1 / max(score.volatility, 0.05)
        
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol > 0:
            weights = {sym: inv_vols[sym] / total_inv_vol for sym, _ in selected}
        else:
            weights = {sym: 1/len(selected) for sym, _ in selected}
        
        return weights
    
    def _calculate_signal_strength(self, selected: List[Tuple[str, FactorScore]]) -> float:
        """Calculate overall signal confidence (0-1)"""
        if not selected:
            return 0.0
        
        # Factors: score spread, momentum consistency, volatility levels
        scores = [s[1].momentum_score for s in selected]
        vols = [s[1].volatility for s in selected]
        
        # Score spread (difference between #1 and #3)
        all_scores = sorted([s[1].momentum_score for s in selected], reverse=True)
        if len(all_scores) >= 3:
            spread = all_scores[0] - all_scores[2]
        elif len(all_scores) >= 2:
            spread = all_scores[0] - all_scores[1]
        else:
            spread = 0.1
        
        # Momentum direction (all positive is stronger signal)
        momentum_direction = sum(1 for s in scores if s > 0) / len(scores)
        
        # Volatility reasonableness (< 25% is good)
        vol_score = sum(1 for v in vols if v < 0.25) / len(vols)
        
        # Composite strength (0-1)
        strength = (
            min(spread * 2, 0.4) +  # Spread contribution
            momentum_direction * 0.4 +  # Direction contribution
            vol_score * 0.2  # Vol contribution
        )
        
        return min(1.0, max(0.0, strength))
    
    def _generate_recommendation(
        self,
        selected: List[Tuple[str, FactorScore]],
        all_scores: Dict[str, FactorScore]
    ) -> str:
        """Generate human-readable recommendation"""
        if not selected:
            return "No factor ETFs showing positive momentum. Hold SPY."
        
        factor_names = [self.FACTORS[sym]["name"] for sym, _ in selected]
        categories = [self.FACTORS[sym]["category"] for sym, _ in selected]
        
        # Check for concentration risk
        if len(set(categories)) == 1:
            return f"Rotate to {', '.join(factor_names)} (concentrated in {categories[0]} category - monitor for diversification)"
        
        # Check for quality of signal
        avg_momentum = np.mean([s[1].return_12m for s in selected])
        if avg_momentum > 0.20:
            strength = "strong"
        elif avg_momentum > 0.10:
            strength = "moderate"
        else:
            strength = "weak"
        
        return f"Rotate to {', '.join(factor_names)} ({strength} momentum: {avg_momentum:+.1%} avg 12m return)"


class FactorRotationBacktest:
    """
    Backtest engine for factor momentum strategy
    Compares against buy-and-hold SPY and equal-weight factors
    """
    
    def __init__(self, engine: FactorMomentumEngine):
        self.engine = engine
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        rebalance_frequency: str = "monthly"
    ) -> Dict:
        """
        Run historical backtest of factor rotation strategy
        """
        return {
            "strategy": "factor_momentum_rotation",
            "period": f"{start_date} to {end_date}",
            "rebalance_frequency": rebalance_frequency,
            "status": "placeholder",
            "note": "Full backtest requires historical simulation framework"
        }


# CLI Interface
def main():
    import sys
    
    engine = FactorMomentumEngine()
    
    if len(sys.argv) < 2:
        print("Factor Momentum Rotation Strategy")
        print("=" * 70)
        print("\nCommands:")
        print("  evaluate      - Run factor evaluation and generate rotation signal")
        print("  status        - Quick status check")
        print("  compare       - Compare all factors by momentum")
        print("\nUniverse:")
        for symbol, info in engine.FACTORS.items():
            print(f"  {symbol:6} - {info['name']:<15} ({info['category']})")
        print()
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "evaluate":
        result = engine.evaluate()
        
        print(f"\n{'='*70}")
        print("FACTOR MOMENTUM ROTATION SIGNAL")
        print(f"{'='*70}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Signal Strength: {result.get('signal_strength', 0):.0%}")
        print(f"\nSelected Factors: {', '.join(result['selected_factors'])}")
        print(f"\nRecommendation: {result['recommendation']}")
        
        print(f"\n{'-'*70}")
        print("ALLOCATION")
        print(f"{'-'*70}")
        for symbol, weight in result['allocation'].items():
            print(f"  {symbol}: {weight:>6.1%}")
        
        print(f"\n{'-'*70}")
        print("ALL FACTOR SCORES (Ranked by Momentum)")
        print(f"{'-'*70}")
        print(f"{'Rank':<6} {'Factor':<12} {'12m':<8} {'6m':<8} {'3m':<8} {'Vol':<8} {'Score':<10}")
        print(f"{'-'*70}")
        
        scores = result['current_scores']
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1]['momentum_score'],
            reverse=True
        )
        
        for symbol, data in sorted_scores:
            print(f"{data['rank']:<6} {data['factor_name']:<12} "
                  f"{data['return_12m']:>+7.1%} {data['return_6m']:>+7.1%} "
                  f"{data['return_3m']:>+7.1%} {data['volatility']:>7.1%} "
                  f"{data['momentum_score']:>+9.3f}")
        
        print(f"{'='*70}\n")
        
        # Output JSON for integration
        print(json.dumps(result, indent=2))
        
    elif cmd == "status":
        result = engine.evaluate()
        print(json.dumps({
            "available": len(result.get('current_scores', {})) > 0,
            "selected": result.get('selected_factors', []),
            "signal_strength": result.get('signal_strength', 0),
            "recommendation": result.get('recommendation', ''),
            "factor_count": len(result.get('current_scores', {}))
        }, indent=2))
    
    elif cmd == "compare":
        result = engine.evaluate()
        
        print(f"\nFactor Comparison (sorted by momentum score)")
        print(f"{'='*70}")
        
        scores = result['current_scores']
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1]['momentum_score'],
            reverse=True
        )
        
        print(f"{'Rank':<6} {'Symbol':<8} {'Factor':<15} {'Category':<10} {'Score':<10} {'Action'}")
        print(f"{'-'*70}")
        
        for symbol, data in sorted_scores:
            action = "HOLD" if symbol in result.get('selected_factors', []) else "AVOID"
            marker = "✓" if action == "HOLD" else " "
            print(f"{marker} {data['rank']:<5} {symbol:<8} {data['factor_name']:<15} "
                  f"{data['category']:<10} {data['momentum_score']:>+9.3f} {action}")
        
        print(f"{'='*70}\n")
    
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python factor_rotation.py [evaluate|status|compare]")


if __name__ == "__main__":
    main()
