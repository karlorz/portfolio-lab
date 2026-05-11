"""
Strategy Comparison Engine
Compares multiple allocation strategies and provides recommendation
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    name: str
    description: str
    allocation: Dict[str, float]
    expected_return: float  # Based on historical CAGR
    expected_volatility: float  # Expected annual volatility
    sharpe_estimate: float
    max_drawdown_estimate: float
    crisis_performance: Dict[str, float]  # Returns in crisis years
    rebalance_frequency: str
    complexity: str  # low, medium, high
    signal_required: bool  # Does it need real-time signals?


class StrategyComparisonEngine:
    """
    Compares different portfolio strategies:
    - All-Season Static (46/38/16)
    - All-Season + Trend Following
    - Dual Momentum (Absolute + Relative)
    - Risk Parity (Inverse Volatility)
    - 60/40 Benchmark
    - 100% SPY Benchmark
    """
    
    def __init__(self):
        self.strategies = {
            "all_season_static": StrategyPerformance(
                name="All-Season Static",
                description="SPY/GLD/TLT 46/38/16 - Fixed allocation, drift-based rebalancing",
                allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                expected_return=0.106,
                expected_volatility=0.111,
                sharpe_estimate=0.79,
                max_drawdown_estimate=-0.262,
                crisis_performance={
                    "2008": -0.123,
                    "2020": -0.071,
                    "2022": -0.130
                },
                rebalance_frequency="On 10% drift",
                complexity="low",
                signal_required=False
            ),
            
            "all_season_trend": StrategyPerformance(
                name="All-Season + Trend",
                description="Base 46/38/16 with 10-month SMA overlay per asset",
                allocation={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},  # Base
                expected_return=0.095,
                expected_volatility=0.105,
                sharpe_estimate=0.75,
                max_drawdown_estimate=-0.180,
                crisis_performance={
                    "2008": -0.080,
                    "2020": -0.050,
                    "2022": -0.100
                },
                rebalance_frequency="Monthly",
                complexity="medium",
                signal_required=True
            ),
            
            "dual_momentum": StrategyPerformance(
                name="Dual Momentum",
                description="Gary Antonacci's GEM: Absolute + Relative momentum",
                allocation={"SPY": 0.50, "QQQ": 0.30, "TLT": 0.10, "GLD": 0.10},  # Example
                expected_return=0.140,
                expected_volatility=0.125,
                sharpe_estimate=0.90,
                max_drawdown_estimate=-0.180,
                crisis_performance={
                    "2008": -0.050,  # TLT saved it
                    "2020": -0.120,
                    "2022": -0.150
                },
                rebalance_frequency="Monthly",
                complexity="high",
                signal_required=True
            ),
            
            "risk_parity": StrategyPerformance(
                name="Risk Parity",
                description="Inverse volatility weighting, ~2x leverage on bonds",
                allocation={"SPY": 0.25, "TLT": 0.50, "GLD": 0.25},  # Vol-targeted
                expected_return=0.090,
                expected_volatility=0.085,
                sharpe_estimate=0.85,
                max_drawdown_estimate=-0.150,
                crisis_performance={
                    "2008": +0.100,  # Bonds rallied
                    "2020": -0.050,
                    "2022": -0.250  # Rate hike disaster
                },
                rebalance_frequency="Monthly",
                complexity="high",
                signal_required=True
            ),
            
            "sixty_forty": StrategyPerformance(
                name="60/40 Portfolio",
                description="Classic stocks/bonds benchmark",
                allocation={"SPY": 0.60, "TLT": 0.40},
                expected_return=0.085,
                expected_volatility=0.110,
                sharpe_estimate=0.55,
                max_drawdown_estimate=-0.350,
                crisis_performance={
                    "2008": -0.200,
                    "2020": -0.120,
                    "2022": -0.180
                },
                rebalance_frequency="Quarterly",
                complexity="low",
                signal_required=False
            ),
            
            "spy_only": StrategyPerformance(
                name="100% SPY",
                description="S&P 500 benchmark",
                allocation={"SPY": 1.0},
                expected_return=0.105,
                expected_volatility=0.190,
                sharpe_estimate=0.42,
                max_drawdown_estimate=-0.550,
                crisis_performance={
                    "2008": -0.370,
                    "2020": -0.340,
                    "2022": -0.250
                },
                rebalance_frequency="None",
                complexity="low",
                signal_required=False
            )
        }
    
    def compare_strategies(
        self,
        criteria: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare all strategies across specified criteria
        
        Criteria: sharpe, return, volatility, drawdown, crisis_resilience
        """
        if criteria is None:
            criteria = ["sharpe", "drawdown", "crisis_resilience"]
        
        results = []
        for key, strategy in self.strategies.items():
            score = self._calculate_score(strategy, criteria)
            results.append({
                "key": key,
                "strategy": strategy,
                "score": score
            })
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "rankings": results,
            "criteria": criteria,
            "best_overall": results[0]["key"] if results else None,
            "recommendation": self._generate_recommendation(results[0] if results else None)
        }
    
    def _calculate_score(
        self,
        strategy: StrategyPerformance,
        criteria: List[str]
    ) -> float:
        """Calculate composite score based on criteria"""
        scores = []
        
        for criterion in criteria:
            if criterion == "sharpe":
                # Normalize Sharpe (0.4 = 0, 1.0 = 1)
                s = (strategy.sharpe_estimate - 0.4) / 0.6
                scores.append(max(0, min(1, s)))
            
            elif criterion == "return":
                # Normalize return (6% = 0, 15% = 1)
                r = (strategy.expected_return - 0.06) / 0.09
                scores.append(max(0, min(1, r)))
            
            elif criterion == "volatility":
                # Lower is better (invert), 20% = 0, 8% = 1
                v = 1 - (strategy.expected_volatility - 0.08) / 0.12
                scores.append(max(0, min(1, v)))
            
            elif criterion == "drawdown":
                # Lower is better (invert), -55% = 0, -10% = 1
                d = 1 - (abs(strategy.max_drawdown_estimate) - 0.10) / 0.45
                scores.append(max(0, min(1, d)))
            
            elif criterion == "crisis_resilience":
                # Average of crisis performance (positive = good)
                crises = list(strategy.crisis_performance.values())
                if crises:
                    avg_crisis = sum(crises) / len(crises)
                    # Normalize: -20% = 0, +10% = 1
                    c = (avg_crisis + 0.20) / 0.30
                    scores.append(max(0, min(1, c)))
            
            elif criterion == "simplicity":
                # Lower complexity = higher score
                complexity_map = {"low": 1.0, "medium": 0.6, "high": 0.3}
                scores.append(complexity_map.get(strategy.complexity, 0.5))
        
        return sum(scores) / len(scores) if scores else 0
    
    def _generate_recommendation(self, top_result: Optional[Dict]) -> str:
        """Generate human-readable recommendation"""
        if not top_result:
            return "No recommendation available"
        
        strategy = top_result["strategy"]
        score = top_result["score"]
        
        if score > 0.8:
            return f"{strategy.name} is strongly recommended based on your criteria. Excellent risk-adjusted returns with strong crisis resilience."
        elif score > 0.6:
            return f"{strategy.name} is a good choice with balanced characteristics. Consider your comfort with {strategy.complexity} complexity."
        else:
            return f"{strategy.name} scores moderately. Review if trade-offs align with your goals."
    
    def get_strategy_details(self, strategy_key: str) -> Optional[Dict]:
        """Get detailed information about a specific strategy"""
        strategy = self.strategies.get(strategy_key)
        if not strategy:
            return None
        
        return {
            "key": strategy_key,
            "name": strategy.name,
            "description": strategy.description,
            "allocation": strategy.allocation,
            "expected_metrics": {
                "cagr": f"{strategy.expected_return:.1%}",
                "volatility": f"{strategy.expected_volatility:.1%}",
                "sharpe": f"{strategy.sharpe_estimate:.2f}",
                "max_drawdown": f"{strategy.max_drawdown_estimate:.1%}"
            },
            "crisis_performance": {
                year: f"{ret:.1%}" for year, ret in strategy.crisis_performance.items()
            },
            "implementation": {
                "rebalance_frequency": strategy.rebalance_frequency,
                "complexity": strategy.complexity,
                "requires_signals": strategy.signal_required
            }
        }
    
    def recommend_for_user_profile(
        self,
        risk_tolerance: str,  # conservative, moderate, aggressive
        time_horizon: str,  # short, medium, long
        tech_comfort: str  # low, medium, high
    ) -> Dict:
        """
        Recommend strategy based on user profile
        """
        # Define scoring weights by profile
        profiles = {
            ("conservative", "long", "low"): {
                "sharpe": 0.3, "drawdown": 0.4, "volatility": 0.2, "simplicity": 0.1
            },
            ("conservative", "long", "high"): {
                "sharpe": 0.3, "drawdown": 0.3, "volatility": 0.2, "simplicity": 0.0, "crisis_resilience": 0.2
            },
            ("moderate", "long", "low"): {
                "sharpe": 0.3, "return": 0.2, "drawdown": 0.2, "volatility": 0.1, "simplicity": 0.2
            },
            ("moderate", "long", "high"): {
                "sharpe": 0.3, "return": 0.3, "drawdown": 0.2, "crisis_resilience": 0.2
            },
            ("aggressive", "long", "low"): {
                "return": 0.4, "sharpe": 0.2, "drawdown": 0.2, "simplicity": 0.2
            },
            ("aggressive", "long", "high"): {
                "return": 0.5, "sharpe": 0.3, "crisis_resilience": 0.2
            }
        }
        
        weights = profiles.get(
            (risk_tolerance, time_horizon, tech_comfort),
            {"sharpe": 0.3, "drawdown": 0.3, "return": 0.2, "volatility": 0.2}  # Default
        )
        
        # Calculate weighted scores
        scored = []
        for key, strategy in self.strategies.items():
            score = 0
            for criterion, weight in weights.items():
                criterion_score = self._calculate_score(strategy, [criterion])
                score += criterion_score * weight
            
            scored.append({
                "key": key,
                "strategy": strategy,
                "score": score,
                "rationale": f"Aligns with {risk_tolerance} risk profile and {tech_comfort} tech comfort"
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "profile": {
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "tech_comfort": tech_comfort
            },
            "recommendation": scored[0] if scored else None,
            "alternatives": scored[1:3] if len(scored) > 1 else [],
            "all_ranked": scored
        }


def main():
    import sys
    
    engine = StrategyComparisonEngine()
    
    if len(sys.argv) < 2:
        print("Strategy Comparison Engine")
        print("=" * 60)
        print("\nCommands:")
        print("  compare [criteria...]  - Compare all strategies")
        print("  details <strategy>     - Show strategy details")
        print("  recommend              - Interactive recommendation")
        print("\nCriteria: sharpe, return, volatility, drawdown, crisis_resilience, simplicity")
        print("\nAvailable strategies:")
        for key in engine.strategies.keys():
            print(f"  - {key}")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "compare":
        criteria = sys.argv[2:] if len(sys.argv) > 2 else None
        result = engine.compare_strategies(criteria)
        
        print(f"\n{'='*60}")
        print("STRATEGY COMPARISON")
        print(f"{'='*60}")
        print(f"Criteria: {', '.join(criteria) if criteria else 'sharpe, drawdown, crisis_resilience'}")
        print()
        
        for i, item in enumerate(result["rankings"][:5], 1):
            s = item["strategy"]
            score = item["score"]
            print(f"{i}. {s.name}")
            print(f"   Score: {score:.1%} | Sharpe: {s.sharpe_estimate:.2f} | Max DD: {s.max_drawdown_estimate:.1%}")
            print(f"   Complexity: {s.complexity} | Rebalance: {s.rebalance_frequency}")
            print()
        
        print(f"Recommendation: {result['recommendation']}")
        print(f"{'='*60}\n")
        
        # Output JSON
        output = {
            "rankings": [
                {
                    "rank": i+1,
                    "key": item["key"],
                    "name": item["strategy"].name,
                    "score": round(item["score"], 3),
                    "sharpe": item["strategy"].sharpe_estimate,
                    "max_drawdown": item["strategy"].max_drawdown_estimate,
                    "complexity": item["strategy"].complexity
                }
                for i, item in enumerate(result["rankings"])
            ],
            "recommendation": result["recommendation"],
            "best": result["best_overall"]
        }
        print(json.dumps(output, indent=2))
    
    elif cmd == "details" and len(sys.argv) > 2:
        strategy_key = sys.argv[2]
        details = engine.get_strategy_details(strategy_key)
        if details:
            print(json.dumps(details, indent=2))
        else:
            print(f"Strategy not found: {strategy_key}")
    
    elif cmd == "recommend":
        # Demo recommendation
        result = engine.recommend_for_user_profile(
            risk_tolerance="moderate",
            time_horizon="long",
            tech_comfort="high"
        )
        print(json.dumps({
            "profile": result["profile"],
            "top_recommendation": {
                "name": result["recommendation"]["strategy"].name,
                "score": result["recommendation"]["score"],
                "rationale": result["recommendation"]["rationale"]
            },
            "alternatives": [
                {"name": alt["strategy"].name, "score": alt["score"]} 
                for alt in result["alternatives"]
            ]
        }, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
