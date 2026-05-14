#!/usr/bin/env python3
"""
Portfolio-Lab v3.12 Phase 1: 0DTE Yield Enhancement - Strike Selector

Strike selection algorithm for 0DTE call writing with:
- 30-delta target selection
- Premium threshold validation (min 0.4%)
- Liquidity check (volume, bid-ask spread)
- Strike ladder management

Usage:
    from src.options.odte_yield_selector import StrikeSelector, StrikeCandidate
    
    selector = StrikeSelector()
    candidate = selector.select_strike(spot=550, vix=16, options_chain=chain)
    if candidate.is_valid:
        print(f"Selected strike: {candidate.strike} @ {candidate.premium:.2f}")
"""

import json
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum

from src.options.odte_yield_calculator import (
    ZeroDTECalculator, ZeroDTEConfig, OptionType, MarketCondition
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrikeQuality(Enum):
    EXCELLENT = "excellent"      # Meets all criteria
    GOOD = "good"                # Minor deviations
    ACCEPTABLE = "acceptable"    # At limits
    POOR = "poor"                # Below minimums
    INVALID = "invalid"          # Fails hard constraints


@dataclass
class StrikeCandidate:
    """Represents a candidate strike for 0DTE call writing."""
    underlying: str
    strike: float
    expiration: datetime
    
    # Quote data
    bid: float
    ask: float
    mid: float
    last: Optional[float] = None
    
    # Greeks (if available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    implied_vol: Optional[float] = None
    
    # Liquidity
    volume: int = 0
    open_interest: int = 0
    
    # Calculated metrics
    premium_pct: float = 0.0       # Premium / spot
    spread_pct: float = 0.0        # Bid-ask spread as % of mid
    delta_estimated: float = 0.0   # Estimated delta
    
    # Quality assessment
    quality: StrikeQuality = StrikeQuality.INVALID
    score: float = 0.0             # 0-100 scoring
    rejection_reasons: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return self.quality in (StrikeQuality.EXCELLENT, StrikeQuality.GOOD, StrikeQuality.ACCEPTABLE)
    
    @property
    def premium(self) -> float:
        return self.mid
    
    def to_dict(self) -> dict:
        return {
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "premium_pct": self.premium_pct,
            "delta": self.delta,
            "delta_estimated": self.delta_estimated,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "spread_pct": self.spread_pct,
            "quality": self.quality.value,
            "score": self.score,
        }


@dataclass
class SelectionCriteria:
    """Criteria for strike selection."""
    target_delta: float = 0.30
    delta_tolerance: float = 0.05
    min_premium_pct: float = 0.004
    max_spread_pct: float = 0.10
    min_volume: int = 100
    min_open_interest: int = 500
    
    # Scoring weights
    delta_weight: float = 0.30
    premium_weight: float = 0.25
    liquidity_weight: float = 0.25
    spread_weight: float = 0.20


class StrikeSelector:
    """
    Selects optimal strike for 0DTE call writing.
    
    Evaluates strikes based on:
    1. Delta alignment (target 30-delta OTM)
    2. Premium adequacy (min 0.4% of spot)
    3. Liquidity (volume, OI, spread)
    4. Overall quality scoring
    """
    
    def __init__(
        self,
        config: Optional[ZeroDTEConfig] = None,
        criteria: Optional[SelectionCriteria] = None,
        calculator: Optional[ZeroDTECalculator] = None
    ):
        self.config = config or ZeroDTEConfig()
        self.criteria = criteria or SelectionCriteria()
        self.calculator = calculator or ZeroDTECalculator(self.config)
    
    def evaluate_strike(self, candidate: StrikeCandidate, 
                       spot: float, vix: float) -> StrikeCandidate:
        """
        Evaluate a single strike candidate against criteria.
        
        Returns updated candidate with quality and score.
        """
        rejection_reasons = []
        
        # Calculate metrics
        candidate.premium_pct = candidate.mid / spot if spot > 0 else 0
        candidate.spread_pct = (candidate.ask - candidate.bid) / candidate.mid if candidate.mid > 0 else 1.0
        
        # Estimate delta if not provided
        if candidate.delta is None:
            candidate.delta_estimated = self.calculator.delta_approximation(
                spot, candidate.strike, vix
            )
            effective_delta = candidate.delta_estimated
        else:
            effective_delta = candidate.delta
        
        # Check delta constraint
        delta_deviation = abs(effective_delta - self.criteria.target_delta)
        if delta_deviation > self.criteria.delta_tolerance:
            rejection_reasons.append(
                f"Delta {effective_delta:.2f} deviates {delta_deviation:.2f} from target"
            )
        
        # Check premium constraint
        if candidate.premium_pct < self.criteria.min_premium_pct:
            rejection_reasons.append(
                f"Premium {candidate.premium_pct:.2%} below minimum {self.criteria.min_premium_pct:.2%}"
            )
        
        # Check spread constraint
        if candidate.spread_pct > self.criteria.max_spread_pct:
            rejection_reasons.append(
                f"Spread {candidate.spread_pct:.1%} exceeds max {self.criteria.max_spread_pct:.1%}"
            )
        
        # Check liquidity
        if candidate.volume < self.criteria.min_volume:
            rejection_reasons.append(
                f"Volume {candidate.volume} below minimum {self.criteria.min_volume}"
            )
        
        if candidate.open_interest < self.criteria.min_open_interest:
            rejection_reasons.append(
                f"OI {candidate.open_interest} below minimum {self.criteria.min_open_interest}"
            )
        
        # Calculate score (0-100)
        score = self._calculate_score(candidate, effective_delta, delta_deviation)
        candidate.score = score
        candidate.rejection_reasons = rejection_reasons
        
        # Determine quality
        if len(rejection_reasons) == 0:
            candidate.quality = StrikeQuality.EXCELLENT if score >= 80 else StrikeQuality.GOOD
        elif all(r.startswith("Volume") or r.startswith("OI") for r in rejection_reasons):
            # Minor liquidity issues - still acceptable
            candidate.quality = StrikeQuality.ACCEPTABLE
        elif len(rejection_reasons) <= 1 and candidate.premium_pct >= self.criteria.min_premium_pct * 0.9:
            # One minor issue, close on premium
            candidate.quality = StrikeQuality.ACCEPTABLE
        elif len(rejection_reasons) <= 2 and candidate.premium_pct >= self.criteria.min_premium_pct:
            # Some issues but good premium
            candidate.quality = StrikeQuality.POOR
        else:
            candidate.quality = StrikeQuality.INVALID
        
        return candidate
    
    def _calculate_score(self, candidate: StrikeCandidate, 
                        effective_delta: float, delta_deviation: float) -> float:
        """Calculate quality score (0-100)."""
        
        # Delta score (30% weight) - closer to target is better
        delta_score = max(0, 100 - (delta_deviation / self.criteria.delta_tolerance) * 50)
        delta_score *= self.criteria.delta_weight
        
        # Premium score (25% weight) - higher is better, with diminishing returns
        premium_ratio = candidate.premium_pct / self.criteria.min_premium_pct
        premium_score = min(100, premium_ratio * 50 + 50)  # 50 points at minimum, up to 100
        premium_score *= self.criteria.premium_weight
        
        # Liquidity score (25% weight) - volume and OI
        vol_score = min(100, candidate.volume / self.criteria.min_volume * 50)
        oi_score = min(100, candidate.open_interest / self.criteria.min_open_interest * 50)
        liquidity_score = (vol_score + oi_score) / 2 * self.criteria.liquidity_weight
        
        # Spread score (20% weight) - tighter is better
        spread_score = max(0, 100 - (candidate.spread_pct / self.criteria.max_spread_pct) * 100)
        spread_score *= self.criteria.spread_weight
        
        return delta_score + premium_score + liquidity_score + spread_score
    
    def select_strike(self, spot: float, vix: float,
                     options_chain: Optional[List[Dict]] = None,
                     underlying: str = "SPY") -> Optional[StrikeCandidate]:
        """
        Select optimal strike from available options.
        
        If options_chain is None, estimates theoretical strikes.
        
        Returns best StrikeCandidate or None if no valid strikes.
        """
        candidates = []
        
        if options_chain:
            # Parse provided chain
            for opt in options_chain:
                if opt.get("option_type", "call") != "call":
                    continue
                
                candidate = StrikeCandidate(
                    underlying=underlying,
                    strike=opt.get("strike", 0),
                    expiration=datetime.fromisoformat(opt.get("expiration", datetime.now().isoformat())),
                    bid=opt.get("bid", 0),
                    ask=opt.get("ask", 0),
                    mid=opt.get("mid", (opt.get("bid", 0) + opt.get("ask", 0)) / 2),
                    delta=opt.get("delta"),
                    volume=opt.get("volume", 0),
                    open_interest=opt.get("open_interest", 0),
                )
                
                evaluated = self.evaluate_strike(candidate, spot, vix)
                if evaluated.is_valid:
                    candidates.append(evaluated)
        else:
            # Generate theoretical candidates
            candidates = self._generate_theoretical_candidates(spot, vix, underlying)
        
        if not candidates:
            logger.warning(f"No valid strike candidates found for {underlying} @ ${spot:.2f}, VIX {vix:.1f}")
            return None
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Selected strike {candidates[0].strike:.2f} with score {candidates[0].score:.1f}")
        return candidates[0]
    
    def _generate_theoretical_candidates(self, spot: float, vix: float,
                                        underlying: str) -> List[StrikeCandidate]:
        """Generate theoretical strike candidates for analysis."""
        candidates = []
        
        # Find target strike
        target_strike, _ = self.calculator.find_target_strike(spot, vix)
        
        # Generate candidates around target (±5 strikes)
        strike_step = 1.0 if underlying == "SPY" else 5.0
        
        # Set expiration to end of today
        from datetime import datetime
        now = datetime.now()
        exp = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        for offset in range(-5, 6):
            strike = target_strike + (offset * strike_step)
            if strike <= spot:  # Skip ITM and ATM
                continue
            
            # Estimate premium
            premium = self.calculator.estimate_premium(spot, strike, vix, OptionType.CALL)
            
            # Estimate delta
            delta = self.calculator.delta_approximation(spot, strike, vix)
            
            # Theoretical bid/ask (10% spread)
            bid = premium * 0.95
            ask = premium * 1.05
            
            # Create candidate with explicit expiration to pass freshness check
            candidate = StrikeCandidate(
                underlying=underlying,
                strike=strike,
                expiration=exp,
                bid=bid,
                ask=ask,
                mid=premium,
                delta_estimated=delta,
                volume=1000,  # Assume liquid
                open_interest=5000,
            )
            
            # Pre-populate metrics
            candidate.premium_pct = premium / spot if spot > 0 else 0
            candidate.spread_pct = 0.05
            
            evaluated = self.evaluate_strike(candidate, spot, vix)
            if evaluated.is_valid:
                candidates.append(evaluated)
        
        return candidates
    
    def get_strike_ladder(self, spot: float, vix: float,
                         underlying: str = "SPY") -> List[StrikeCandidate]:
        """
        Get ordered list of strike candidates (ladder view).
        
        Returns all evaluated candidates sorted by quality and score.
        """
        candidates = self._generate_theoretical_candidates(spot, vix, underlying)
        
        # Include some rejected strikes for context
        strike_step = 1.0 if underlying == "SPY" else 5.0
        target_strike, _ = self.calculator.find_target_strike(spot, vix)
        
        # Set expiration to end of today
        from datetime import datetime
        now = datetime.now()
        exp = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        for offset in [10, 15, 20]:  # Further OTM
            strike = target_strike + (offset * strike_step)
            premium = self.calculator.estimate_premium(spot, strike, vix, OptionType.CALL)
            delta = self.calculator.delta_approximation(spot, strike, vix)
            
            candidate = StrikeCandidate(
                underlying=underlying,
                strike=strike,
                expiration=exp,
                bid=premium * 0.95,
                ask=premium * 1.05,
                mid=premium,
                delta_estimated=delta,
                volume=500,
                open_interest=2000,
                quality=StrikeQuality.POOR,
            )
            candidate.premium_pct = premium / spot
            candidate.spread_pct = 0.05
            candidate.score = self._calculate_score(candidate, delta, abs(delta - 0.30))
            candidates.append(candidate)
        
        # Sort by strike
        candidates.sort(key=lambda x: x.strike)
        return candidates
    
    def validate_selection(self, candidate: StrikeCandidate, 
                          spot: float, vix: float,
                          portfolio_delta: float) -> Tuple[bool, List[str]]:
        """
        Final validation before executing trade.
        
        Returns:
            (is_valid, list of warnings)
        """
        warnings = []
        
        # Re-check premium threshold
        if candidate.premium_pct < self.config.min_premium_pct:
            return False, [f"Premium {candidate.premium_pct:.2%} below minimum {self.config.min_premium_pct:.2%}"]
        
        # Check portfolio delta impact
        estimated_contracts = 1  # Conservative estimate
        delta_impact = -candidate.delta_estimated * estimated_contracts * 100 / (spot * 100)
        new_portfolio_delta = portfolio_delta + delta_impact
        
        if new_portfolio_delta > self.config.max_delta_exposure:
            warnings.append(
                f"Portfolio delta would be {new_portfolio_delta:.2f}, near limit {self.config.max_delta_exposure}"
            )
        
        # Check market condition
        condition = self.calculator.classify_market_condition(vix)
        if condition in (MarketCondition.HIGH_VOL, MarketCondition.EXTREME):
            warnings.append(f"Elevated volatility: {condition.value}")
        
        return len(warnings) == 0, warnings


if __name__ == "__main__":
    # Example usage
    selector = StrikeSelector()
    
    spot = 550.0
    vix = 16.0
    
    print(f"=== 0DTE Strike Selection ===")
    print(f"Spot: ${spot:.2f}, VIX: {vix:.1f}")
    print()
    
    # Select strike
    candidate = selector.select_strike(spot, vix)
    
    if candidate:
        print(f"✓ Selected Strike: ${candidate.strike:.2f}")
        print(f"  Premium: ${candidate.premium:.2f} ({candidate.premium_pct:.2%})")
        print(f"  Delta: {candidate.delta_estimated:.2f}")
        print(f"  Quality: {candidate.quality.value}")
        print(f"  Score: {candidate.score:.1f}/100")
        print()
        
        # Show ladder
        print("Strike Ladder:")
        ladder = selector.get_strike_ladder(spot, vix)
        for c in ladder:
            indicator = "← SELECTED" if c.strike == candidate.strike else ""
            print(f"  ${c.strike:>6.2f} | Δ{c.delta_estimated:.2f} | ${c.premium:>5.2f} | {c.quality.value:12} {indicator}")
    else:
        print("✗ No valid strike found")
