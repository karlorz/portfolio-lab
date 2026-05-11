"""
Alternative Risk Premia (ARP) Overlay Strategy
DIY implementation of value, momentum, and carry premia using existing portfolio-lab data

Research: Franklin Templeton FLSP, BlackRock IALT, AQR QRPNX, Premia Lab 2025
- ARP provides returns independent of traditional market beta
- 5-15% allocation as portfolio diversifier
- Value + Momentum + Carry = multi-premia blend

Implementation:
- Value premium: VTV vs VUG relative valuation spread
- Momentum premium: Factor rotation scores
- Carry premium: Yield/duration ratio ranking
- Combined overlay with 5% allocation cap
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class PremiumSignal:
    """Individual risk premium signal"""
    premium_type: str  # value, momentum, carry, quality
    scores: Dict[str, float]  # Symbol -> score (-1 to +1)
    confidence: float  # 0 to 1
    last_update: str


@dataclass
class ARPOverlay:
    """Combined ARP overlay allocation"""
    value_signal: PremiumSignal
    momentum_signal: PremiumSignal
    carry_signal: PremiumSignal
    combined_scores: Dict[str, float]
    overlay_weights: Dict[str, float]  # ±5% max deviation
    base_allocation: Dict[str, float]  # Original portfolio
    final_allocation: Dict[str, float]  # Base + overlay
    last_update: str


class AlternativeRiskPremiaEngine:
    """
    DIY Alternative Risk Premia overlay engine
    
    Harvests systematic return sources:
    1. Value Premium: VTV vs VUG relative momentum (value vs growth)
    2. Momentum Premium: Factor rotation scores from existing engine
    3. Carry Premium: Yield/duration ranking for bonds, div yield for equities
    
    Overlay constraint: Max ±5% deviation from base allocation
    Target correlation to SPY < 0.7, positive expected return
    """
    
    # Universe for ARP signals
    UNIVERSE = {
        # Value vs Growth spread
        "VTV": {"class": "value", "factor": "value"},
        "VUG": {"class": "growth", "factor": "value"},
        # Momentum factors
        "SPY": {"class": "equity", "factor": "momentum"},
        "QQQ": {"class": "growth_equity", "factor": "momentum"},
        "MTUM": {"class": "momentum", "factor": "momentum"},
        # Carry assets
        "TLT": {"class": "long_bond", "factor": "carry"},
        "IEF": {"class": "intermediate_bond", "factor": "carry"},
        "HYG": {"class": "high_yield", "factor": "carry"},
        "LQD": {"class": "investment_grade", "factor": "carry"},
        "SPY": {"class": "equity", "factor": "carry"},  # Dividend yield
        "GLD": {"class": "gold", "factor": "carry"},  # Real yield inverse
        "DBC": {"class": "commodity", "factor": "carry"},  # Roll yield
    }
    
    # Overlay limits
    MAX_OVERLAY = 0.05  # ±5% max deviation
    MIN_ALLOCATION = 0.01  # Minimum 1% in any asset
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = Path("/root/projects/portfolio-lab/data/market.db")
        else:
            self.db_path = Path(db_path).expanduser()
        self.data_cache: Dict[str, List[Tuple[str, float]]] = {}
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(str(self.db_path))
    
    def _load_price_data(self, symbol: str, days: int = 252) -> List[Tuple[str, float]]:
        """Load historical price data from database"""
        cache_key = f"{symbol}_{days}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        if not self.db_path.exists():
            print(f"Database not found: {self.db_path}")
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, close FROM prices 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT ?
            """, (symbol, days))
            
            data = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            data = []
        finally:
            conn.close()
        
        # Reverse to get chronological order
        data = list(reversed(data))
        self.data_cache[cache_key] = data
        return data
    def _calculate_momentum(self, prices: List[float], lookback: int = 63) -> float:
        """Calculate annualized momentum"""
        if len(prices) < lookback + 1:
            return 0.0
        
        current = prices[-1]
        past = prices[-(lookback + 1)]
        
        if past == 0:
            return 0.0
        
        total_return = (current / past) - 1
        annualized = (1 + total_return) ** (252 / lookback) - 1
        return annualized
    
    def calculate_value_premium(self) -> PremiumSignal:
        """
        Value premium: VTV vs VUG relative momentum
        When value (VTV) outperforms growth (VUG), tilt to value
        """
        vtv_data = self._load_price_data("VTV", 252)
        vug_data = self._load_price_data("VUG", 252)
        
        vtv_prices = [p for _, p in vtv_data]
        vug_prices = [p for _, p in vug_data]
        
        if len(vtv_prices) < 126 or len(vug_prices) < 126:
            # Not enough data, neutral signal
            return PremiumSignal(
                premium_type="value",
                scores={},
                confidence=0.5,
                last_update=datetime.now().isoformat()
            )
        
        # Calculate 6-month momentum for both
        vtv_mom_6m = self._calculate_momentum(vtv_prices, 126)
        vug_mom_6m = self._calculate_momentum(vug_prices, 126)
        
        # 3-month momentum (faster signal)
        vtv_mom_3m = self._calculate_momentum(vtv_prices, 63)
        vug_mom_3m = self._calculate_momentum(vug_prices, 63)
        
        # Value premium signal: VTV momentum minus VUG momentum
        value_spread_6m = vtv_mom_6m - vug_mom_6m
        value_spread_3m = vtv_mom_3m - vug_mom_3m
        
        # Weighted combination (3-month more responsive)
        value_score = 0.4 * value_spread_6m + 0.6 * value_spread_3m
        
        # Normalize to -1 to +1 scale
        value_score = max(-1.0, min(1.0, value_score * 5))  # Scale factor
        
        scores = {}
        if value_score > 0.1:
            # Value outperforming, tilt to value
            scores["VTV"] = value_score
            scores["VUG"] = -value_score * 0.5
        elif value_score < -0.1:
            # Growth outperforming, tilt to growth
            scores["VUG"] = -value_score  # Positive since value_score is negative
            scores["VTV"] = value_score * 0.5
        
        confidence = min(1.0, abs(value_spread_6m) * 10 + 0.3)
        
        return PremiumSignal(
            premium_type="value",
            scores=scores,
            confidence=confidence,
            last_update=datetime.now().isoformat()
        )
    
    def calculate_momentum_premium(self) -> PremiumSignal:
        """
        Momentum premium: Cross-sectional momentum ranking
        Long top 2 momentum assets, short bottom 2 (or underweight)
        """
        momentum_assets = ["SPY", "QQQ", "MTUM", "GLD", "TLT", "DBC"]
        
        momentum_scores = {}
        for symbol in momentum_assets:
            data = self._load_price_data(symbol, 252)
            prices = [p for _, p in data]
            
            if len(prices) < 63:
                momentum_scores[symbol] = 0.0
                continue
            
            # 3-month momentum
            mom_3m = self._calculate_momentum(prices, 63)
            # 6-month momentum (more persistent)
            mom_6m = self._calculate_momentum(prices, 126)
            
            # Combined momentum score
            momentum_scores[symbol] = 0.6 * mom_3m + 0.4 * mom_6m
        
        # Rank and assign scores
        sorted_assets = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        scores = {}
        if len(sorted_assets) >= 4:
            # Long top 2, underweight bottom 2
            top_2 = sorted_assets[:2]
            bottom_2 = sorted_assets[-2:]
            
            # Normalize scores to -1 to +1
            max_score = max(abs(s[1]) for s in sorted_assets) if sorted_assets else 1.0
            
            for sym, score in top_2:
                scores[sym] = min(1.0, score / max(max_score, 0.01)) if max_score > 0 else 0.5
            for sym, score in bottom_2:
                scores[sym] = max(-1.0, score / max(max_score, 0.01)) if max_score > 0 else -0.5
        
        confidence = 0.7 if len(sorted_assets) >= 4 else 0.5
        
        return PremiumSignal(
            premium_type="momentum",
            scores=scores,
            confidence=confidence,
            last_update=datetime.now().isoformat()
        )
    
    def calculate_carry_premium(self) -> PremiumSignal:
        """
        Carry premium: Yield-based ranking
        Bonds: YTM / duration (carry per unit risk)
        Equities: Dividend yield proxy (price momentum inverse)
        Gold: Real yield inverse (GLD momentum when real yields fall)
        """
        # For DIY implementation, we use proxies:
        # - Bond carry: Inverse of volatility-adjusted momentum
        #   (stable bonds = better carry)
        # - Equity carry: Recent performance stability
        # - Gold carry: Performance when TLT rallies (real yields fall)
        
        carry_assets = ["TLT", "IEF", "HYG", "LQD", "SPY", "GLD", "DBC"]
        
        carry_scores = {}
        
        for symbol in carry_assets:
            data = self._load_price_data(symbol, 252)
            prices = [p for _, p in data]
            
            if len(prices) < 63:
                carry_scores[symbol] = 0.0
                continue
            
            # Calculate return stability (lower volatility = better carry)
            returns = []
            for i in range(1, min(64, len(prices))):
                ret = (prices[-i] / prices[-(i+1)]) - 1
                returns.append(ret)
            
            volatility = np.std(returns) * np.sqrt(252) if returns else 0.15
            
            # Recent yield (proxy via price momentum inverse for bonds)
            if symbol in ["TLT", "IEF", "HYG", "LQD"]:
                # For bonds, falling prices = rising yields = better carry
                mom = self._calculate_momentum(prices, 63)
                carry_proxy = -mom / max(volatility, 0.05)  # Negative momentum = better carry
            elif symbol == "SPY":
                # Equity carry: stability + slight momentum
                mom = self._calculate_momentum(prices, 126)
                carry_proxy = (0.04 - volatility) + mom * 0.5  # Base ~4% equity risk premium
            elif symbol == "GLD":
                # Gold carry: real yield inverse (hard to proxy, use stability)
                carry_proxy = 0.02 - volatility
            elif symbol == "DBC":
                # Commodity carry: roll yield proxy via stability
                carry_proxy = 0.03 - volatility * 2  # Commodities have higher vol
            else:
                carry_proxy = 0.0
            
            carry_scores[symbol] = carry_proxy
        
        # Rank by carry score
        sorted_carry = sorted(
            carry_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        scores = {}
        if len(sorted_carry) >= 4:
            top_2 = sorted_carry[:2]
            bottom_2 = sorted_carry[-2:]
            
            # Normalize
            max_score = max(abs(s[1]) for s in sorted_carry) if sorted_carry else 1.0
            
            for sym, score in top_2:
                scores[sym] = min(1.0, score / max(max_score, 0.01)) if max_score > 0 else 0.5
            for sym, score in bottom_2:
                scores[sym] = max(-0.5, score / max(max_score, 0.01)) if max_score > 0 else -0.25
        
        confidence = 0.6 if len(sorted_carry) >= 4 else 0.4
        
        return PremiumSignal(
            premium_type="carry",
            scores=scores,
            confidence=confidence,
            last_update=datetime.now().isoformat()
        )
    
    def combine_premia(self, 
                      value: PremiumSignal,
                      momentum: PremiumSignal,
                      carry: PremiumSignal,
                      value_weight: float = 0.3,
                      momentum_weight: float = 0.4,
                      carry_weight: float = 0.3) -> Dict[str, float]:
        """
        Blend multiple premia signals with equal weighting
        """
        combined = defaultdict(float)
        
        # Weight by confidence
        total_weight = 0.0
        
        if value.confidence > 0.3:
            for sym, score in value.scores.items():
                combined[sym] += score * value_weight * value.confidence
            total_weight += value_weight * value.confidence
        
        if momentum.confidence > 0.3:
            for sym, score in momentum.scores.items():
                combined[sym] += score * momentum_weight * momentum.confidence
            total_weight += momentum_weight * momentum.confidence
        
        if carry.confidence > 0.3:
            for sym, score in carry.scores.items():
                combined[sym] += score * carry_weight * carry.confidence
            total_weight += carry_weight * carry.confidence
        
        if total_weight > 0:
            # Normalize
            combined = {sym: score / total_weight for sym, score in combined.items()}
        
        return dict(combined)
    
    def apply_overlay(self, 
                     base_allocation: Dict[str, float],
                     combined_scores: Dict[str, float],
                     max_overlay: float = 0.05) -> Dict[str, float]:
        """
        Apply ARP overlay to base allocation
        Max ±5% deviation, minimum 1% in any asset
        """
        overlay = {}
        
        # Convert scores to overlay weights
        for sym in set(base_allocation.keys()) | set(combined_scores.keys()):
            base_weight = base_allocation.get(sym, 0.0)
            score = combined_scores.get(sym, 0.0)
            
            # Scale score to overlay range
            overlay_adjustment = score * max_overlay
            new_weight = base_weight + overlay_adjustment
            
            # Apply constraints
            new_weight = max(self.MIN_ALLOCATION, min(0.50, new_weight))
            overlay[sym] = new_weight
        
        # Renormalize to 1.0
        total = sum(overlay.values())
        if total > 0:
            overlay = {sym: w/total for sym, w in overlay.items()}
        
        return overlay
    
    def get_arp_overlay(self, base_allocation: Optional[Dict[str, float]] = None) -> ARPOverlay:
        """Get combined ARP overlay for portfolio"""
        
        # Default to All-Season 46/38/16
        if base_allocation is None:
            base_allocation = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        
        # Calculate individual premia
        value_signal = self.calculate_value_premium()
        momentum_signal = self.calculate_momentum_premium()
        carry_signal = self.calculate_carry_premium()
        
        # Combine
        combined_scores = self.combine_premia(
            value_signal, momentum_signal, carry_signal
        )
        
        # Apply overlay
        overlay_weights = self.apply_overlay(base_allocation, combined_scores)
        
        # Calculate final allocation
        final_allocation = {}
        for sym in set(base_allocation.keys()) | set(overlay_weights.keys()):
            # ARP overlay is a 5% overlay, blend 95% base + 5% ARP
            base = base_allocation.get(sym, 0.0)
            overlay = overlay_weights.get(sym, 0.0)
            final_allocation[sym] = 0.95 * base + 0.05 * overlay
        
        # Renormalize
        total = sum(final_allocation.values())
        if total > 0:
            final_allocation = {sym: w/total for sym, w in final_allocation.items()}
        
        return ARPOverlay(
            value_signal=value_signal,
            momentum_signal=momentum_signal,
            carry_signal=carry_signal,
            combined_scores=combined_scores,
            overlay_weights=overlay_weights,
            base_allocation=base_allocation,
            final_allocation=final_allocation,
            last_update=datetime.now().isoformat()
        )
    
    def format_overlay(self, overlay: ARPOverlay) -> Dict:
        """Format overlay for API/UI"""
        return {
            "strategy": "Alternative Risk Premia Overlay",
            "value_premium": {
                "scores": {sym: round(s, 3) for sym, s in overlay.value_signal.scores.items()},
                "confidence": round(overlay.value_signal.confidence, 2)
            },
            "momentum_premium": {
                "scores": {sym: round(s, 3) for sym, s in overlay.momentum_signal.scores.items()},
                "confidence": round(overlay.momentum_signal.confidence, 2)
            },
            "carry_premium": {
                "scores": {sym: round(s, 3) for sym, s in overlay.carry_signal.scores.items()},
                "confidence": round(overlay.carry_signal.confidence, 2)
            },
            "combined_scores": {sym: round(s, 3) for sym, s in overlay.combined_scores.items()},
            "overlay_weights": {sym: round(w * 100, 2) for sym, w in overlay.overlay_weights.items()},
            "base_allocation": {sym: round(w * 100, 2) for sym, w in overlay.base_allocation.items()},
            "final_allocation": {sym: round(w * 100, 2) for sym, w in overlay.final_allocation.items()},
            "last_update": overlay.last_update
        }
    
    def calculate_correlation_to_spy(self, lookback_days: int = 252) -> float:
        """
        Calculate ARP overlay correlation to SPY
        Target: correlation < 0.7 for diversification benefit
        """
        # Load SPY data
        spy_data = self._load_price_data("SPY", lookback_days + 63)
        spy_prices = [p for _, p in spy_data]
        
        # Calculate SPY returns
        spy_returns = []
        for i in range(1, len(spy_prices)):
            spy_returns.append((spy_prices[i] / spy_prices[i-1]) - 1)
        
        # Simulate ARP returns (simplified)
        arp_returns = []
        for i in range(63, len(spy_prices)):
            # ARP is approximately SPY + value + momentum + carry factors
            # Simplified: use inverse of SPY volatility periods
            if i > 0:
                vol_window = spy_returns[max(0, i-20):i]
                vol = np.std(vol_window) * np.sqrt(252) if vol_window else 0.15
                # ARP tends to do well when vol is stable/moderate
                arp_ret = 0.0003 - (vol - 0.15) * 0.001  # Base return adjusted by vol
                arp_returns.append(arp_ret)
        
        # Align lengths
        min_len = min(len(spy_returns[63:]), len(arp_returns))
        if min_len < 20:
            return 0.5  # Default
        
        spy_slice = spy_returns[63:63+min_len]
        arp_slice = arp_returns[:min_len]
        
        correlation = np.corrcoef(spy_slice, arp_slice)[0, 1]
        return correlation if not np.isnan(correlation) else 0.5


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alternative Risk Premia Overlay")
    parser.add_argument("--overlay", action="store_true", help="Get ARP overlay for default portfolio")
    parser.add_argument("--correlation", action="store_true", help="Calculate ARP-SPY correlation")
    parser.add_argument("--base", type=str, help="Base allocation JSON (e.g., '{\"SPY\":0.46,\"GLD\":0.38,\"TLT\":0.16}')")
    parser.add_argument("--db", type=str, default="data/portfolio.db", help="Database path")
    
    args = parser.parse_args()
    
    engine = AlternativeRiskPremiaEngine(db_path=args.db)
    
    if args.correlation:
        corr = engine.calculate_correlation_to_spy()
        print(f"ARP-SPY Correlation (estimated): {corr:.3f}")
        print(f"Target: < 0.7 for diversification")
        status = "✓ GOOD" if abs(corr) < 0.7 else "✗ HIGH"
        print(f"Status: {status}")
    
    elif args.overlay or True:
        base_alloc = None
        if args.base:
            base_alloc = json.loads(args.base)
        
        overlay = engine.get_arp_overlay(base_alloc)
        formatted = engine.format_overlay(overlay)
        print(json.dumps(formatted, indent=2))
