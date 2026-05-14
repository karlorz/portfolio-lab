#!/usr/bin/env python3
"""
Portfolio-Lab v3.12 Phase 1: 0DTE Yield Enhancement - Options Calculator

Black-Scholes and delta approximation for 0DTE option pricing.
Provides premium estimation, delta calculation, and position sizing.

Usage:
    from src.options.odte_yield_calculator import ZeroDTECalculator, ZeroDTEConfig
    
    config = ZeroDTEConfig()
    calc = ZeroDTECalculator(config)
    premium = calc.estimate_premium(spot=550, strike=555, vix=16)
    delta = calc.delta_approximation(spot, strike, vix, time_to_expiry=1/365)
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class MarketCondition(Enum):
    NORMAL = "normal"
    ELEVATED_VOL = "elevated_vol"
    HIGH_VOL = "high_vol"
    EXTREME = "extreme"


@dataclass
class ZeroDTEConfig:
    """Configuration for 0DTE yield enhancement strategy."""
    # Allocation limits
    max_portfolio_allocation: float = 0.02       # 2% max of portfolio
    position_size_pct: float = 0.005              # 0.5% per trade
    max_weekly_positions: int = 2                 # Max 2 trades per week
    max_concurrent_positions: int = 1             # Only 1 0DTE position at a time
    
    # Entry filters
    min_vix: float = 15.0                         # Minimum VIX for entry
    max_vix: float = 35.0                         # Maximum VIX (avoid chaos)
    entry_time_start: time = time(10, 30)         # 10:30 AM ET earliest
    entry_time_end: time = time(14, 0)          # 2:00 PM ET latest
    
    # Strike selection
    delta_target: float = 0.30                    # Target 30-delta OTM
    delta_tolerance: float = 0.05                 # Accept 25-35 delta
    min_premium_pct: float = 0.004                # 0.4% min premium received
    
    # Risk limits
    max_delta_exposure: float = 0.08              # Portfolio delta limit
    emergency_close_delta: float = 0.50           # Roll/close trigger
    max_loss_pct: float = 0.015                   # 1.5% max loss per trade
    profit_take_pct: float = 0.50                 # Close at 50% profit
    
    # Days to avoid (earnings, events)
    blocked_dates: Optional[list] = None          # List of "YYYY-MM-DD" strings


@dataclass
class OptionQuote:
    """Represents an option quote with calculated metrics."""
    underlying: str                               # SPY, SPX, etc.
    option_type: OptionType
    strike: float
    expiration: datetime
    bid: float
    ask: float
    mid: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    implied_vol: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    
    @property
    def premium(self) -> float:
        """Premium received (for short) or paid (for long)."""
        return self.mid
    
    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid price."""
        if self.mid == 0:
            return 0.0
        return (self.ask - self.bid) / self.mid


@dataclass
class PositionMetrics:
    """Metrics for an active 0DTE position."""
    entry_premium: float
    current_premium: float
    delta: float
    unrealized_pnl: float
    pnl_pct: float
    time_to_expiry_hours: float
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0
    
    @property
    def profit_pct_of_max(self) -> float:
        """What percentage of max profit has been captured."""
        if self.entry_premium == 0:
            return 0.0
        return abs(self.unrealized_pnl) / self.entry_premium


class ZeroDTECalculator:
    """
    Calculator for 0DTE option pricing and position sizing.
    
    Uses simplified Black-Scholes approximations suitable for 0DTE
    where time decay is the dominant factor.
    """
    
    def __init__(self, config: Optional[ZeroDTEConfig] = None):
        self.config = config or ZeroDTEConfig()
    
    def classify_market_condition(self, vix: float) -> MarketCondition:
        """Classify market volatility condition."""
        if vix < 15:
            return MarketCondition.NORMAL
        elif vix < 22:
            return MarketCondition.ELEVATED_VOL
        elif vix < 30:
            return MarketCondition.HIGH_VOL
        else:
            return MarketCondition.EXTREME
    
    def is_entry_allowed(self, vix: float, current_time: time, 
                         portfolio_delta: float) -> Tuple[bool, str]:
        """
        Check if new position entry is allowed.
        
        Returns:
            (allowed, reason)
        """
        # VIX check
        if vix < self.config.min_vix:
            return False, f"VIX {vix:.1f} below minimum {self.config.min_vix}"
        
        if vix > self.config.max_vix:
            return False, f"VIX {vix:.1f} above maximum {self.config.max_vix}"
        
        # Time check
        if current_time < self.config.entry_time_start:
            return False, f"Too early: {current_time} before {self.config.entry_time_start}"
        
        if current_time > self.config.entry_time_end:
            return False, f"Too late: {current_time} after {self.config.entry_time_end}"
        
        # Portfolio delta check
        if portfolio_delta > self.config.max_delta_exposure:
            return False, f"Portfolio delta {portfolio_delta:.2f} exceeds limit {self.config.max_delta_exposure}"
        
        return True, "Entry permitted"
    
    def estimate_premium(self, spot: float, strike: float, vix: float,
                        option_type: OptionType = OptionType.CALL,
                        time_to_expiry: float = 1/365) -> float:
        """
        Estimate option premium using simplified Black-Scholes.
        
        For 0DTE, uses the approximation that premium ≈ intrinsic + time value
        where time value decays rapidly.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            vix: VIX index (annualized volatility %)
            option_type: Call or put
            time_to_expiry: Fraction of year (default 1/365 for 1 day)
        
        Returns:
            Estimated premium in dollars per share
        """
        # Intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        
        # Time value approximation for 0DTE
        # σ * S * √(T) * 0.4 for ATM options
        vol = vix / 100  # Convert to decimal
        atm_time_value = spot * vol * np.sqrt(time_to_expiry) * 0.4
        
        # Distance from strike (moneyness adjustment)
        distance = abs(strike - spot) / spot
        moneyness_factor = max(0.1, 1 - distance * 5)  # Decays as we go OTM
        
        time_value = atm_time_value * moneyness_factor
        
        return intrinsic + time_value
    
    def delta_approximation(self, spot: float, strike: float, 
                           vix: float, time_to_expiry: float = 1/365) -> float:
        """
        Approximate option delta for strike selection.
        
        Uses the approximation:
        Delta ≈ N(d1) where d1 ≈ ln(S/K) / (σ√T) + 0.5σ√T
        
        For 0DTE, this simplifies to roughly step function around strike.
        """
        vol = vix / 100
        
        # Simplified d1 calculation for 0DTE
        if spot == strike:
            d1 = 0
        else:
            d1 = np.log(spot / strike) / (vol * np.sqrt(time_to_expiry))
        
        # Approximate N(d1) using error function approximation
        # N(x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
        delta = 0.5 * (1 + np.sign(d1))  # Simplified step function
        
        # Smooth the transition
        k = 2.0  # Steepness factor
        delta = 1 / (1 + np.exp(-k * d1))
        
        return delta
    
    def find_target_strike(self, spot: float, vix: float,
                          target_delta: Optional[float] = None) -> Tuple[float, float]:
        """
        Find strike price that gives target delta.
        
        Returns:
            (strike_price, estimated_delta)
        """
        if target_delta is None:
            target_delta = self.config.delta_target
        
        # For OTM calls (delta < 0.5), strike > spot
        # Start with OTM by default (selling calls above current price)
        delta_sign = 1 if target_delta > 0 else -1
        
        # Binary search for target delta
        low_strike = spot * 0.95
        high_strike = spot * 1.10
        
        best_strike = spot * 1.01  # Default 1% OTM
        best_delta = 0.5
        
        for _ in range(20):  # Binary search iterations
            mid_strike = (low_strike + high_strike) / 2
            mid_delta = self.delta_approximation(spot, mid_strike, vix)
            
            if abs(mid_delta - target_delta) < abs(best_delta - target_delta):
                best_strike = mid_strike
                best_delta = mid_delta
            
            if mid_delta > target_delta:
                # Too ITM, need higher strike
                low_strike = mid_strike
            else:
                # Too OTM, need lower strike  
                high_strike = mid_strike
        
        # Round to standard SPY strike intervals ($1 for SPY, $5 for SPX)
        if spot > 400:  # Likely SPX
            best_strike = round(best_strike / 5) * 5
        else:  # Likely SPY
            best_strike = round(best_strike)
        
        return best_strike, best_delta
    
    def calculate_position_size(self, portfolio_value: float,
                               max_position_value: Optional[float] = None) -> int:
        """
        Calculate number of contracts to sell.
        
        Args:
            portfolio_value: Total portfolio value
            max_position_value: Max dollar value for this position
        
        Returns:
            Number of contracts (each = 100 shares)
        """
        if max_position_value is None:
            max_position_value = portfolio_value * self.config.position_size_pct
        
        # Each contract covers 100 shares
        notional_per_contract = 100
        
        # Calculate number of contracts
        num_contracts = int(max_position_value / notional_per_contract)
        
        # Minimum 1 contract if allocation permits
        if num_contracts < 1 and max_position_value >= notional_per_contract:
            num_contracts = 1
        
        return max(0, num_contracts)
    
    def calculate_notional_exposure(self, strike: float, 
                                   num_contracts: int) -> float:
        """Calculate total notional exposure of position."""
        return strike * num_contracts * 100
    
    def calculate_portfolio_delta_impact(self, option_delta: float,
                                        num_contracts: int,
                                        portfolio_value: float) -> float:
        """
        Calculate portfolio-level delta from this position.
        
        Short calls have negative delta (bearish exposure).
        """
        # Short call: negative delta
        position_delta = -option_delta * num_contracts * 100  # Shares equivalent
        return position_delta / portfolio_value
    
    def check_emergency_close(self, position_delta: float,
                             current_premium: float,
                             entry_premium: float,
                             current_time: time) -> Tuple[bool, str]:
        """
        Check if position needs emergency close.
        
        Returns:
            (should_close, reason)
        """
        # Delta-based stop
        if abs(position_delta) > self.config.emergency_close_delta:
            return True, f"Delta {position_delta:.2f} exceeded limit {self.config.emergency_close_delta}"
        
        # Loss-based stop
        loss_pct = (current_premium - entry_premium) / entry_premium
        if loss_pct > self.config.max_loss_pct:
            return True, f"Loss {loss_pct:.1%} exceeded limit {self.config.max_loss_pct:.1%}"
        
        # Time-based close (3:30 PM cutoff)
        cutoff_time = time(15, 30)
        if current_time > cutoff_time:
            return True, f"Time exit: {current_time} after cutoff {cutoff_time}"
        
        return False, "Position within normal parameters"
    
    def calculate_expected_return(self, premium: float, strike: float,
                                 spot: float, vix: float,
                                 win_rate: float = 0.68) -> dict:
        """
        Calculate expected return statistics.
        
        Args:
            premium: Premium received
            strike: Strike price (short call)
            spot: Current spot price
            vix: VIX level
            win_rate: Historical win rate for this setup
        
        Returns:
            Dictionary with expected value metrics
        """
        # Max gain = premium received
        max_gain = premium
        
        # Max loss = theoretically unlimited for short calls
        # Estimate using 2-sigma move
        vol = vix / 100
        sigma_move = spot * vol * np.sqrt(1/365) * 2
        max_loss_estimate = max(0, (strike + sigma_move - spot) - premium)
        
        # Expected value
        ev_win = win_rate * max_gain
        ev_loss = (1 - win_rate) * max_loss_estimate
        expected_value = ev_win - ev_loss
        
        # Risk-reward
        risk_reward = max_gain / max_loss_estimate if max_loss_estimate > 0 else float('inf')
        
        return {
            "max_gain": max_gain,
            "max_loss_estimate": max_loss_estimate,
            "expected_value": expected_value,
            "risk_reward_ratio": risk_reward,
            "win_rate_assumed": win_rate,
            "breakeven": strike + premium,  # For short call
        }
    
    def format_position_summary(self, metrics: PositionMetrics) -> str:
        """Format position metrics for display."""
        return (
            f"0DTE Position: "
            f"P&L ${metrics.unrealized_pnl:+.2f} ({metrics.pnl_pct:+.1%}) | "
            f"Delta: {metrics.delta:.2f} | "
            f"Time: {metrics.time_to_expiry_hours:.1f}h remaining"
        )


if __name__ == "__main__":
    # Example usage
    config = ZeroDTEConfig()
    calc = ZeroDTECalculator(config)
    
    # Example: SPY at $550, VIX 16
    spot = 550.0
    vix = 16.0
    
    print(f"=== 0DTE Yield Enhancement Calculator ===")
    print(f"Spot: ${spot:.2f}, VIX: {vix:.1f}")
    print()
    
    # Check entry conditions
    allowed, reason = calc.is_entry_allowed(vix, time(11, 0), portfolio_delta=0.02)
    print(f"Entry allowed: {allowed} - {reason}")
    print()
    
    # Find target strike
    strike, delta = calc.find_target_strike(spot, vix, target_delta=0.30)
    print(f"Target strike: ${strike:.2f} (delta: {delta:.2f})")
    
    # Estimate premium
    premium = calc.estimate_premium(spot, strike, vix, OptionType.CALL)
    print(f"Estimated premium: ${premium:.2f}")
    print(f"Premium as % of strike: {premium/strike:.2%}")
    print()
    
    # Position sizing for $100K portfolio
    portfolio_value = 100000
    num_contracts = calc.calculate_position_size(portfolio_value)
    notional = calc.calculate_notional_exposure(strike, num_contracts)
    
    print(f"Portfolio: ${portfolio_value:,.0f}")
    print(f"Contracts to sell: {num_contracts}")
    print(f"Notional exposure: ${notional:,.0f}")
    print(f"Position size: {notional/portfolio_value:.2%} of portfolio")
    print()
    
    # Expected return
    expected = calc.calculate_expected_return(premium, strike, spot, vix)
    print(f"Expected Return Analysis:")
    print(f"  Max gain: ${expected['max_gain']:.2f}")
    print(f"  Max loss (est): ${expected['max_loss_estimate']:.2f}")
    print(f"  Risk/reward: {expected['risk_reward_ratio']:.2f}")
    print(f"  Breakeven: ${expected['breakeven']:.2f}")
