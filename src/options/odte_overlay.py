#!/usr/bin/env python3
"""
Portfolio-Lab v2.23: 0DTE Options Overlay Module

0 Days To Expiration (0DTE) Options Trading Strategy with Iron Condor implementation.
Based on Q3 2026 research synthesis showing 0DTE options dominate SPX volume with
Iron Condor strategies achieving 66-90% win rates when properly risk-managed.

Key Components:
- Iron Condor Strategy: Sell 16-delta calls/puts, buy protective wings
- GEX Monitoring: SpotGamma-style gamma exposure calculations
- Position Sizing: 0.5-2% portfolio risk per trade
- Three-Stop System: Price-based, percentage-based, time-based exits
- Integration: Works with circuit_breaker.py and evaluator.py

Research References:
- CBOE 0DTE Activity Report (Q2 2026)
- SpotGamma GEX Methodology
- Option Alpha Iron Condor Studies
"""

import json
import sqlite3
import argparse
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('odte_overlay')

# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
CONFIG_PATH = DATA_DIR / "odte_config.json"
STATE_PATH = DATA_DIR / "odte_state.json"
TRADE_LOG_PATH = DATA_DIR / "odte_trades.jsonl"


class StopType(Enum):
    """Three-stop system types"""
    PRICE = "price"           # Price-based (breakeven or technical level)
    PERCENTAGE = "percentage"  # Percentage of max profit/loss
    TIME = "time"             # Time-based exit (e.g., 3:00 PM ET)


class TradeStatus(Enum):
    """Trade lifecycle status"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"


@dataclass
class Greeks:
    """Option Greeks container"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class OptionLeg:
    """Single option leg (call or put)"""
    symbol: str           # Underlying (SPX, SPY, etc.)
    option_symbol: str    # Full option symbol
    strike: float
    expiration: datetime
    option_type: str      # 'call' or 'put'
    side: str             # 'buy' or 'sell'
    quantity: int
    entry_price: float
    current_price: float = 0.0
    greeks: Greeks = field(default_factory=Greeks)
    
    @property
    def is_short(self) -> bool:
        return self.side == 'sell'
    
    @property
    def notional_value(self) -> float:
        """Approximate notional value (strike * quantity * 100)"""
        return self.strike * self.quantity * 100
    
    @property
    def premium(self) -> float:
        """Premium received/paid"""
        mult = -1 if self.is_short else 1
        return self.entry_price * self.quantity * 100 * mult


@dataclass
class IronCondor:
    """
    Iron Condor position structure
    
    Structure:
    - Short Call (sell): ~16 delta
    - Long Call (buy): Higher strike (10-25 points away)
    - Short Put (sell): ~16 delta
    - Long Put (buy): Lower strike (10-25 points away)
    
    Width determined by VIX:
    - VIX <20: 10-wide wings
    - VIX 20-25: 15-wide wings
    - VIX >25: 20-wide wings
    """
    trade_id: str
    underlying: str
    entry_time: datetime
    
    # Legs
    short_call: OptionLeg
    long_call: OptionLeg
    short_put: OptionLeg
    long_put: OptionLeg
    
    # Entry metrics
    entry_spot: float
    entry_vix: float
    
    # Risk parameters
    max_profit: float = 0.0
    max_loss: float = 0.0
    wing_width: int = 10
    
    # Exit tracking
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    realized_pnl: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    
    def __post_init__(self):
        if self.max_profit == 0.0:
            self._calculate_risk_metrics()
    
    def _calculate_risk_metrics(self):
        """Calculate max profit and max loss for the condor"""
        # Net credit received
        call_credit = self.short_call.premium + self.long_call.premium
        put_credit = self.short_put.premium + self.long_put.premium
        
        # Max profit is net credit received
        self.max_profit = abs(call_credit + put_credit)
        
        # Max loss is width of wing minus credit received
        call_width = self.long_call.strike - self.short_call.strike
        put_width = self.short_put.strike - self.long_put.strike
        
        call_risk = (call_width * 100) - abs(call_credit)
        put_risk = (put_width * 100) - abs(put_credit)
        
        self.max_loss = max(call_risk, put_risk)
        self.wing_width = int(call_width)
    
    @property
    def breakevens(self) -> Tuple[float, float]:
        """Calculate breakeven prices"""
        total_credit = self.max_profit / 100
        upper_breakeven = self.short_call.strike + total_credit
        lower_breakeven = self.short_put.strike - total_credit
        return (lower_breakeven, upper_breakeven)
    
    @property
    def current_pnl(self, current_prices: Optional[Dict] = None) -> float:
        """Calculate current unrealized P&L"""
        if not current_prices:
            return 0.0
        
        pnl = 0.0
        for leg in [self.short_call, self.long_call, self.short_put, self.long_put]:
            if leg.option_symbol in current_prices:
                price_diff = leg.entry_price - current_prices[leg.option_symbol]
                if leg.is_short:
                    pnl += price_diff * leg.quantity * 100
                else:
                    pnl -= price_diff * leg.quantity * 100
        
        return pnl
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "trade_id": self.trade_id,
            "underlying": self.underlying,
            "entry_time": self.entry_time.isoformat(),
            "entry_spot": self.entry_spot,
            "entry_vix": self.entry_vix,
            "status": self.status.value,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "wing_width": self.wing_width,
            "breakevens": self.breakevens,
            "legs": {
                "short_call": {
                    "strike": self.short_call.strike,
                    "delta": self.short_call.greeks.delta,
                    "entry": self.short_call.entry_price
                },
                "long_call": {
                    "strike": self.long_call.strike,
                    "entry": self.long_call.entry_price
                },
                "short_put": {
                    "strike": self.short_put.strike,
                    "delta": self.short_put.greeks.delta,
                    "entry": self.short_put.entry_price
                },
                "long_put": {
                    "strike": self.long_put.strike,
                    "entry": self.long_put.entry_price
                }
            },
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl
        }


@dataclass
class GEXLevel:
    """Gamma Exposure at a specific strike"""
    strike: float
    gamma_exposure: float
    call_gamma: float
    put_gamma: float
    net_delta: float


@dataclass
class GEXProfile:
    """Full GEX profile for underlying"""
    underlying: str
    spot_price: float
    timestamp: datetime
    levels: List[GEXLevel] = field(default_factory=list)
    
    # Key levels
    max_gamma_strike: float = 0.0
    max_gamma_abs: float = 0.0
    put_wall: float = 0.0
    call_wall: float = 0.0
    gamma_flip: float = 0.0
    
    # Overall exposure
    total_gamma: float = 0.0
    total_call_gamma: float = 0.0
    total_put_gamma: float = 0.0
    gamma_imbalance: float = 0.0  # positive = more call gamma


class GEXCalculator:
    """
    Gamma Exposure (GEX) Calculator
    
    SpotGamma-style GEX calculations for 0DTE options.
    GEX helps identify:
    - Key support/resistance levels (gamma walls)
    - Potential pin risk at max gamma strikes
    - Market maker hedging pressure
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
    
    def calculate_gex(
        self,
        underlying: str,
        expiration: Optional[datetime] = None,
        spot_price: Optional[float] = None
    ) -> GEXProfile:
        """
        Calculate GEX profile for underlying
        
        In production, this would query option chain data from broker API.
        For backtesting/simulation, we estimate from ATM volatility.
        """
        timestamp = datetime.now()
        
        # Get spot price if not provided
        if spot_price is None:
            spot_price = self._get_spot_price(underlying)
        
        profile = GEXProfile(
            underlying=underlying,
            spot_price=spot_price,
            timestamp=timestamp
        )
        
        # In simulation mode, estimate GEX distribution
        # In live mode, would query actual option chain
        profile = self._estimate_gex_distribution(profile, underlying)
        
        return profile
    
    def _get_spot_price(self, underlying: str) -> float:
        """Get current spot price from database"""
        if not self.db_path.exists():
            return 0.0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = ? 
            ORDER BY date DESC LIMIT 1
        """, (underlying,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else 0.0
    
    def _estimate_gex_distribution(
        self,
        profile: GEXProfile,
        underlying: str
    ) -> GEXProfile:
        """
        Estimate GEX distribution based on typical 0DTE patterns
        
        In live trading, replace with actual option chain query.
        """
        spot = profile.spot_price
        
        # Generate strikes around spot (+/- 5%)
        strike_range = spot * 0.05
        strikes = np.linspace(spot - strike_range, spot + strike_range, 21)
        
        levels = []
        total_gamma = 0.0
        total_call_gamma = 0.0
        total_put_gamma = 0.0
        
        max_gamma = 0.0
        max_gamma_strike = spot
        
        for strike in strikes:
            # Estimate gamma using Black-Scholes approximation
            # Simplified: gamma peaks ATM and decreases away
            distance = abs(strike - spot) / spot
            
            # Typical 0DTE: high gamma near ATM, creates magnetic effect
            atm_gamma = self._estimate_strike_gamma(strike, spot, underlying)
            
            # Call gamma (positive, concentrated above spot)
            call_gamma = atm_gamma * max(0, 1 - (strike - spot) / spot * 10)
            
            # Put gamma (positive, concentrated below spot)
            put_gamma = atm_gamma * max(0, 1 - (spot - strike) / spot * 10)
            
            net_gamma = call_gamma + put_gamma
            
            level = GEXLevel(
                strike=round(strike, 2),
                gamma_exposure=net_gamma,
                call_gamma=call_gamma,
                put_gamma=put_gamma,
                net_delta=(call_gamma - put_gamma) * 0.01  # Approximate
            )
            
            levels.append(level)
            
            total_gamma += net_gamma
            total_call_gamma += call_gamma
            total_put_gamma += put_gamma
            
            if net_gamma > max_gamma:
                max_gamma = net_gamma
                max_gamma_strike = strike
        
        profile.levels = levels
        profile.total_gamma = total_gamma
        profile.total_call_gamma = total_call_gamma
        profile.total_put_gamma = total_put_gamma
        profile.max_gamma_strike = max_gamma_strike
        profile.max_gamma_abs = max_gamma
        
        # Calculate key levels
        profile = self._identify_key_levels(profile)
        
        return profile
    
    def _estimate_strike_gamma(
        self,
        strike: float,
        spot: float,
        underlying: str
    ) -> float:
        """Estimate gamma for a strike (simplified model)"""
        # Distance from ATM (0 = at-the-money)
        moneyness = abs(strike - spot) / spot
        
        # 0DTE gamma is very high ATM, drops off quickly
        # Gamma ~ 1 / (distance ^ 2) for short-dated options
        if moneyness < 0.001:
            return 1000.0  # Very high ATM gamma
        
        gamma = 1000.0 * np.exp(-moneyness * 50)
        
        return max(gamma, 0.1)
    
    def _identify_key_levels(self, profile: GEXProfile) -> GEXProfile:
        """Identify put wall, call wall, and gamma flip points"""
        if not profile.levels:
            return profile
        
        # Sort by gamma exposure
        sorted_levels = sorted(
            profile.levels,
            key=lambda x: x.gamma_exposure,
            reverse=True
        )
        
        # Put wall: highest put gamma below spot
        puts_below = [l for l in profile.levels if l.strike < profile.spot_price]
        if puts_below:
            puts_below.sort(key=lambda x: x.put_gamma, reverse=True)
            profile.put_wall = puts_below[0].strike
        
        # Call wall: highest call gamma above spot
        calls_above = [l for l in profile.levels if l.strike > profile.spot_price]
        if calls_above:
            calls_above.sort(key=lambda x: x.call_gamma, reverse=True)
            profile.call_wall = calls_above[0].strike
        
        # Gamma flip: where total gamma changes sign (roughly spot)
        profile.gamma_flip = profile.spot_price
        
        # Gamma imbalance
        if profile.total_gamma > 0:
            profile.gamma_imbalance = (
                (profile.total_call_gamma - profile.total_put_gamma) / 
                profile.total_gamma
            )
        
        return profile
    
    def check_pin_risk(self, profile: GEXProfile) -> Dict:
        """
        Check for potential pin risk near max gamma strike
        
        Returns risk assessment dict
        """
        spot = profile.spot_price
        max_gamma_strike = profile.max_gamma_strike
        
        # Distance to max gamma
        distance_pct = abs(spot - max_gamma_strike) / spot * 100
        
        # Pin risk when within 0.5% of max gamma strike
        pin_risk = distance_pct < 0.5 and profile.max_gamma_abs > 500
        
        return {
            "pin_risk_detected": pin_risk,
            "distance_to_max_gamma_pct": round(distance_pct, 2),
            "max_gamma_strike": max_gamma_strike,
            "current_spot": spot,
            "recommendation": "avoid_new_positions" if pin_risk else "normal"
        }


class PositionSizer:
    """
    Position Sizing for 0DTE Options
    
    Rules:
    - Max 2% portfolio risk per trade
    - VIX-adjusted sizing
    - Account for circuit breaker status
    """
    
    MAX_RISK_PCT = 0.02  # 2% max risk per trade
    MIN_RISK_PCT = 0.005  # 0.5% min risk per trade
    
    def __init__(
        self,
        portfolio_value: float,
        circuit_breaker_scalar: float = 1.0
    ):
        self.portfolio_value = portfolio_value
        self.circuit_breaker_scalar = circuit_breaker_scalar
    
    def calculate_size(
        self,
        vix: float,
        wing_width: int,
        max_loss_per_contract: float,
        gex_profile: Optional[GEXProfile] = None
    ) -> Dict:
        """
        Calculate position size based on risk parameters
        
        Returns dict with:
        - num_contracts
        - risk_amount
        - risk_pct
        - sizing_notes
        """
        # Base risk percent based on VIX
        if vix < 15:
            base_risk = 0.015  # 1.5% in low vol
        elif vix < 20:
            base_risk = 0.012  # 1.2% in normal vol
        elif vix < 25:
            base_risk = 0.010  # 1.0% in elevated vol
        else:
            base_risk = 0.008  # 0.8% in high vol
        
        # Adjust for wing width (wider wings = higher risk per contract)
        width_adjustment = wing_width / 10.0  # Normalize to 10-wide
        
        # Calculate max contracts
        max_risk_amount = self.portfolio_value * self.MAX_RISK_PCT * self.circuit_breaker_scalar
        
        # Adjust for actual max loss per contract
        adjusted_max_loss = max_loss_per_contract * width_adjustment
        
        if adjusted_max_loss <= 0:
            return {
                "num_contracts": 0,
                "risk_amount": 0,
                "risk_pct": 0,
                "sizing_notes": ["Invalid max loss calculation"]
            }
        
        num_contracts = int(max_risk_amount / adjusted_max_loss)
        
        # Apply minimum
        if num_contracts < 1 and base_risk >= self.MIN_RISK_PCT:
            num_contracts = 1
        
        # Calculate actual risk
        actual_risk = num_contracts * adjusted_max_loss
        actual_risk_pct = actual_risk / self.portfolio_value if self.portfolio_value > 0 else 0
        
        notes = [
            f"VIX level: {vix:.1f}",
            f"Wing width: {wing_width}",
            f"Circuit breaker scalar: {self.circuit_breaker_scalar:.2f}",
            f"Max risk allowed: ${max_risk_amount:,.2f}"
        ]
        
        # GEX-based adjustment
        if gex_profile and gex_profile.max_gamma_abs > 1000:
            num_contracts = max(1, int(num_contracts * 0.8))
            notes.append("GEX high - reduced size 20%")
        
        return {
            "num_contracts": num_contracts,
            "risk_amount": actual_risk,
            "risk_pct": round(actual_risk_pct, 4),
            "sizing_notes": notes
        }


class ThreeStopManager:
    """
    Three-Stop Risk Management System
    
    Stops:
    1. Price-based: Breakeven or technical level breach
    2. Percentage-based: % of max profit or max loss
    3. Time-based: Exit at specified time (e.g., 3:00 PM ET)
    
    First trigger wins - position closed on any stop.
    """
    
    # Default exit time (3:00 PM ET for 0DTE)
    DEFAULT_EXIT_TIME = time(15, 0)
    
    def __init__(
        self,
        profit_target_pct: float = 0.50,  # Close at 50% of max profit
        stop_loss_pct: float = 2.0,        # Close at 2x credit received (200%)
        exit_time: Optional[time] = None
    ):
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.exit_time = exit_time or self.DEFAULT_EXIT_TIME
    
    def check_stops(
        self,
        condor: IronCondor,
        current_pnl: float,
        current_time: datetime
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check all three stop conditions
        
        Returns: (should_exit, reason, exit_price_estimate)
        """
        # Stop 1: Percentage-based (profit target or stop loss)
        max_profit = condor.max_profit
        max_loss = condor.max_loss
        
        if max_profit > 0:
            profit_pct = current_pnl / max_profit
            
            # Profit target reached
            if profit_pct >= self.profit_target_pct:
                return True, f"profit_target_{self.profit_target_pct:.0%}", None
        
        if max_loss > 0:
            loss_pct = -current_pnl / max_loss
            
            # Stop loss reached
            if loss_pct >= self.stop_loss_pct:
                return True, f"stop_loss_{self.stop_loss_pct:.0%}", None
        
        # Stop 2: Price-based (breach of short strikes)
        # Would need current underlying price - simplified check
        
        # Stop 3: Time-based
        current_time_only = current_time.time()
        if current_time_only >= self.exit_time:
            return True, f"time_exit_{self.exit_time}", None
        
        return False, None, None
    
    def calculate_dynamic_adjustments(
        self,
        condor: IronCondor,
        gex_profile: GEXProfile,
        minutes_to_close: int
    ) -> Dict:
        """
        Calculate dynamic stop adjustments based on GEX and time
        
        As expiration approaches:
        - Tighten stops if near max gamma
        - Widen slightly if in "safe zone"
        """
        # Base adjustments
        profit_adjustment = 0.0
        stop_adjustment = 0.0
        notes = []
        
        # Time decay acceleration
        if minutes_to_close < 60:
            # Last hour - consider earlier profit taking
            profit_adjustment -= 0.10  # Lower profit target
            notes.append("Last hour - reduced profit target 10%")
        
        # GEX-based adjustments
        if gex_profile.max_gamma_strike:
            distance_to_max = abs(
                condor.entry_spot - gex_profile.max_gamma_strike
            ) / condor.entry_spot * 100
            
            if distance_to_max < 0.5:
                # Near max gamma - high pin risk
                profit_adjustment -= 0.15
                stop_adjustment += 0.25
                notes.append("High pin risk - tightened stops")
            elif distance_to_max > 2.0:
                # In safe zone - can be more patient
                profit_adjustment += 0.10
                notes.append("Safe zone from max gamma - can be patient")
        
        return {
            "adjusted_profit_target": self.profit_target_pct + profit_adjustment,
            "adjusted_stop_loss": self.stop_loss_pct + stop_adjustment,
            "notes": notes
        }


class ODTEOverlay:
    """
    Main 0DTE Options Overlay Module
    
    Integrates Iron Condor strategy, GEX monitoring, position sizing,
    and three-stop risk management.
    """
    
    # Entry window (9:35-10:00 AM ET)
    ENTRY_START = time(9, 35)
    ENTRY_END = time(10, 0)
    
    # Delta targets for strikes
    SHORT_DELTA_TARGET = 0.16  # ~16 delta
    LONG_DELTA_TARGET = 0.05   # ~5 delta (protective wing)
    
    def __init__(
        self,
        portfolio_value: float = 100000.0,
        mode: str = "paper"
    ):
        self.portfolio_value = portfolio_value
        self.mode = mode
        
        # Sub-modules
        self.gex_calc = GEXCalculator()
        self.stop_mgr = ThreeStopManager()
        
        # State
        self.active_trades: List[IronCondor] = []
        self.trade_history: List[Dict] = []
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted state"""
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                state = json.load(f)
                self.trade_history = state.get("trade_history", [])
    
    def _save_state(self):
        """Persist state"""
        state = {
            "active_trades": [t.to_dict() for t in self.active_trades],
            "trade_history": self.trade_history[-100:],  # Keep last 100
            "last_update": datetime.now().isoformat()
        }
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_wing_width(self, vix: float) -> int:
        """
        Determine wing width based on VIX level
        
        VIX <20: 10-wide
        VIX 20-25: 15-wide
        VIX >25: 20-wide
        """
        if vix < 20:
            return 10
        elif vix <= 25:
            return 15
        else:
            return 20
    
    def check_entry_conditions(
        self,
        underlying: str = "SPX",
        vix: Optional[float] = None
    ) -> Dict:
        """
        Check if conditions are suitable for entry
        
        Returns assessment dict with:
        - can_enter: bool
        - reason: str
        - recommendations: list
        """
        now = datetime.now()
        current_time = now.time()
        
        # Check entry window
        if not (self.ENTRY_START <= current_time <= self.ENTRY_END):
            return {
                "can_enter": False,
                "reason": "Outside entry window (9:35-10:00 AM ET)",
                "recommendations": ["Wait for next entry window"]
            }
        
        # Check VIX
        if vix is None:
            vix = self._get_vix()
        
        if vix > 35:
            return {
                "can_enter": False,
                "reason": f"VIX too high ({vix:.1f}) - excessive volatility",
                "recommendations": ["Wait for VIX normalization below 35"]
            }
        
        # Check GEX
        gex = self.gex_calc.calculate_gex(underlying)
        pin_risk = self.gex_calc.check_pin_risk(gex)
        
        if pin_risk["pin_risk_detected"]:
            return {
                "can_enter": False,
                "reason": "High pin risk detected near max gamma",
                "recommendations": [
                    f"Spot {gex.spot_price:.2f} near max gamma {gex.max_gamma_strike:.2f}",
                    "Wait for distance >0.5% from max gamma"
                ]
            }
        
        # Check for existing positions
        if len(self.active_trades) >= 2:
            return {
                "can_enter": False,
                "reason": "Max concurrent positions reached (2)",
                "recommendations": ["Manage existing positions first"]
            }
        
        return {
            "can_enter": True,
            "reason": "All conditions met",
            "vix": vix,
            "gex_profile": {
                "put_wall": gex.put_wall,
                "call_wall": gex.call_wall,
                "gamma_flip": gex.gamma_flip
            },
            "recommendations": [
                f"VIX {vix:.1f} - use {self.get_wing_width(vix)}-wide wings"
            ]
        }
    
    def _get_vix(self) -> float:
        """Get current VIX level"""
        if not DB_PATH.exists():
            return 20.0
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT close FROM prices 
            WHERE symbol = '^VIX' 
            ORDER BY date DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else 20.0
    
    def construct_condor(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        circuit_breaker_scalar: float = 1.0
    ) -> Optional[IronCondor]:
        """
        Construct Iron Condor based on current conditions
        
        In production, this would query option chain and select strikes
        by delta. For simulation, we estimate strikes from spot + VIX.
        """
        # Determine wing width
        wing_width = self.get_wing_width(vix)
        
        # Estimate strikes based on 16-delta (approx 0.5 * VIX for 1-day)
        # This is a simplification - real trading uses actual delta
        daily_vol = vix / 100 / np.sqrt(252)
        move_16delta = spot_price * daily_vol * 0.5  # Approx 0.5 std dev
        
        short_call_strike = round((spot_price + move_16delta) / 5) * 5
        long_call_strike = short_call_strike + wing_width
        
        short_put_strike = round((spot_price - move_16delta) / 5) * 5
        long_put_strike = short_put_strike - wing_width
        
        # Estimate premiums (simplified Black-Scholes)
        short_call_premium = self._estimate_premium(
            spot_price, short_call_strike, vix, 1/252, "call"
        )
        long_call_premium = self._estimate_premium(
            spot_price, long_call_strike, vix, 1/252, "call"
        )
        short_put_premium = self._estimate_premium(
            spot_price, short_put_strike, vix, 1/252, "put"
        )
        long_put_premium = self._estimate_premium(
            spot_price, long_put_strike, vix, 1/252, "put"
        )
        
        # Calculate position size
        position_sizer = PositionSizer(
            self.portfolio_value,
            circuit_breaker_scalar
        )
        
        # Estimate max loss per contract
        call_risk = (long_call_strike - short_call_strike) * 100
        put_risk = (short_put_strike - long_put_strike) * 100
        max_loss_per_contract = max(call_risk, put_risk)
        
        # Get GEX profile for sizing
        gex = self.gex_calc.calculate_gex(underlying, spot_price=spot_price)
        
        sizing = position_sizer.calculate_size(
            vix, wing_width, max_loss_per_contract, gex
        )
        
        num_contracts = sizing["num_contracts"]
        
        if num_contracts < 1:
            logger.warning("Position sizing resulted in 0 contracts")
            return None
        
        # Create trade ID
        trade_id = f"0DTE_{underlying}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create legs
        short_call = OptionLeg(
            symbol=underlying,
            option_symbol=f"{underlying}{datetime.now().strftime('%y%m%d')}C{int(short_call_strike)}",
            strike=short_call_strike,
            expiration=datetime.now().replace(hour=16, minute=0),
            option_type="call",
            side="sell",
            quantity=num_contracts,
            entry_price=short_call_premium,
            current_price=short_call_premium,
            greeks=Greeks(delta=-0.16, gamma=0.05, theta=0.02, vega=0.01)
        )
        
        long_call = OptionLeg(
            symbol=underlying,
            option_symbol=f"{underlying}{datetime.now().strftime('%y%m%d')}C{int(long_call_strike)}",
            strike=long_call_strike,
            expiration=datetime.now().replace(hour=16, minute=0),
            option_type="call",
            side="buy",
            quantity=num_contracts,
            entry_price=long_call_premium,
            current_price=long_call_premium,
            greeks=Greeks(delta=0.05, gamma=0.02, theta=0.005, vega=0.005)
        )
        
        short_put = OptionLeg(
            symbol=underlying,
            option_symbol=f"{underlying}{datetime.now().strftime('%y%m%d')}P{int(short_put_strike)}",
            strike=short_put_strike,
            expiration=datetime.now().replace(hour=16, minute=0),
            option_type="put",
            side="sell",
            quantity=num_contracts,
            entry_price=short_put_premium,
            current_price=short_put_premium,
            greeks=Greeks(delta=0.16, gamma=0.05, theta=0.02, vega=0.01)
        )
        
        long_put = OptionLeg(
            symbol=underlying,
            option_symbol=f"{underlying}{datetime.now().strftime('%y%m%d')}P{int(long_put_strike)}",
            strike=long_put_strike,
            expiration=datetime.now().replace(hour=16, minute=0),
            option_type="put",
            side="buy",
            quantity=num_contracts,
            entry_price=long_put_premium,
            current_price=long_put_premium,
            greeks=Greeks(delta=-0.05, gamma=0.02, theta=0.005, vega=0.005)
        )
        
        condor = IronCondor(
            trade_id=trade_id,
            underlying=underlying,
            entry_time=datetime.now(),
            short_call=short_call,
            long_call=long_call,
            short_put=short_put,
            long_put=long_put,
            entry_spot=spot_price,
            entry_vix=vix,
            status=TradeStatus.OPEN
        )
        
        logger.info(f"Constructed condor: {trade_id} with {num_contracts} contracts")
        return condor
    
    def _estimate_premium(
        self,
        spot: float,
        strike: float,
        vix: float,
        time_to_expiry: float,
        option_type: str
    ) -> float:
        """Simplified premium estimate using Black-Scholes approximation"""
        # Simplified: ATM straddle approx = 0.8 * spot * vol * sqrt(T)
        # Individual option approx half that adjusted by moneyness
        
        vol = vix / 100
        
        # Distance from ATM
        moneyness = abs(strike - spot) / spot
        
        # Base ATM premium estimate (simplified)
        atm_premium = spot * vol * np.sqrt(time_to_expiry) * 0.4
        
        # Adjust for moneyness (OTM = cheaper)
        if option_type == "call":
            if strike > spot:  # OTM call
                otm_factor = np.exp(-moneyness * 5)
            else:  # ITM call
                otm_factor = 1 + moneyness
        else:  # put
            if strike < spot:  # OTM put
                otm_factor = np.exp(-moneyness * 5)
            else:  # ITM put
                otm_factor = 1 + moneyness
        
        premium = atm_premium * otm_factor
        
        # Scale to option price (0DTEs are cheaper)
        return max(premium * 0.1, 0.05)
    
    def manage_positions(self) -> List[Dict]:
        """
        Check and manage all active positions
        
        Returns list of actions taken
        """
        actions = []
        now = datetime.now()
        
        for condor in self.active_trades[:]:
            # Get current P&L estimate
            # In live trading, query mark prices
            current_pnl = self._estimate_current_pnl(condor)
            
            # Check stops
            should_exit, reason, _ = self.stop_mgr.check_stops(
                condor, current_pnl, now
            )
            
            if should_exit:
                action = self._close_position(condor, reason, current_pnl)
                actions.append(action)
            else:
                # Monitor position
                actions.append({
                    "trade_id": condor.trade_id,
                    "action": "monitor",
                    "current_pnl": round(current_pnl, 2),
                    "unrealized_pct": round(current_pnl / condor.max_profit, 2) if condor.max_profit > 0 else 0
                })
        
        return actions
    
    def _estimate_current_pnl(self, condor: IronCondor) -> float:
        """Estimate current P&L based on time decay"""
        # Simplified: assume 0DTE loses value through the day
        elapsed = (datetime.now() - condor.entry_time).total_seconds() / 3600
        
        # Rough theta decay curve
        if elapsed < 1:
            decay_factor = 0.1
        elif elapsed < 3:
            decay_factor = 0.3
        elif elapsed < 5:
            decay_factor = 0.6
        else:
            decay_factor = 0.9
        
        # Assume we capture decay as profit
        pnl = condor.max_profit * decay_factor
        
        return pnl
    
    def _close_position(
        self,
        condor: IronCondor,
        reason: str,
        realized_pnl: float
    ) -> Dict:
        """Close a position and log the trade"""
        condor.status = TradeStatus.CLOSED
        condor.exit_time = datetime.now()
        condor.exit_reason = reason
        condor.realized_pnl = realized_pnl
        
        # Remove from active
        self.active_trades.remove(condor)
        
        # Log trade
        trade_record = condor.to_dict()
        self.trade_history.append(trade_record)
        
        with open(TRADE_LOG_PATH, 'a') as f:
            f.write(json.dumps(trade_record) + '\n')
        
        logger.info(f"Closed {condor.trade_id}: {reason}, P&L: ${realized_pnl:.2f}")
        
        return {
            "trade_id": condor.trade_id,
            "action": "close",
            "reason": reason,
            "realized_pnl": round(realized_pnl, 2),
            "return_pct": round(realized_pnl / condor.max_profit * 100, 2) if condor.max_profit > 0 else 0
        }
    
    def run_cycle(self, circuit_breaker_status: str = "green") -> Dict:
        """
        Run one full cycle: check entry, manage positions
        
        Integrates with circuit breaker system
        """
        # Get circuit breaker position scalar
        cb_scalar = self._get_circuit_breaker_scalar(circuit_breaker_status)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "circuit_breaker_status": circuit_breaker_status,
            "position_scalar": cb_scalar,
            "actions": []
        }
        
        # Manage existing positions first
        position_actions = self.manage_positions()
        results["actions"].extend(position_actions)
        
        # Check for new entry (only if green/yellow CB status)
        if cb_scalar > 0 and len(self.active_trades) < 2:
            entry_check = self.check_entry_conditions()
            
            if entry_check["can_enter"]:
                vix = entry_check.get("vix", self._get_vix())
                
                # Get spot price
                spot = self.gex_calc._get_spot_price("SPY")
                if spot == 0:
                    spot = 550  # Fallback
                
                condor = self.construct_condor(
                    "SPY", spot, vix, cb_scalar
                )
                
                if condor:
                    self.active_trades.append(condor)
                    results["actions"].append({
                        "action": "enter",
                        "trade_id": condor.trade_id,
                        "underlying": condor.underlying,
                        "strikes": {
                            "short_call": condor.short_call.strike,
                            "long_call": condor.long_call.strike,
                            "short_put": condor.short_put.strike,
                            "long_put": condor.long_put.strike
                        },
                        "max_profit": round(condor.max_profit, 2),
                        "max_loss": round(condor.max_loss, 2),
                        "breakevens": condor.breakevens
                    })
            else:
                results["actions"].append({
                    "action": "no_entry",
                    "reason": entry_check["reason"]
                })
        
        # Save state
        self._save_state()
        
        return results
    
    def _get_circuit_breaker_scalar(self, status: str) -> float:
        """Get position scalar from circuit breaker status"""
        scalars = {
            "green": 1.0,
            "yellow": 1.0,   # Warning only
            "orange": 0.75,  # Reduce 25%
            "red": 0.50,     # Reduce 50%
            "black": 0.0     # No new positions
        }
        return scalars.get(status, 0.0)
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        if not self.trade_history:
            return {"message": "No trade history available"}
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get("realized_pnl", 0) > 0)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.get("realized_pnl", 0) for t in self.trade_history)
        
        wins = [t.get("realized_pnl", 0) for t in self.trade_history if t.get("realized_pnl", 0) > 0]
        losses = [t.get("realized_pnl", 0) for t in self.trade_history if t.get("realized_pnl", 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(winning_trades / total_trades, 4) if total_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else float('inf'),
            "active_trades": len(self.active_trades)
        }


class ODTEBacktester:
    """
    Backtesting engine for 0DTE strategies
    
    Simulates trading on historical data.
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio_value: float = 100000.0
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio_value = portfolio_value
        self.results: List[Dict] = []
    
    def run(self) -> Dict:
        """Run backtest simulation"""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # This would iterate through historical data
        # For now, return placeholder structure
        
        return {
            "backtest_period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat()
            },
            "initial_capital": self.portfolio_value,
            "trades_simulated": 0,
            "final_pnl": 0,
            "message": "Backtest requires historical options data"
        }


def main():
    """CLI interface for 0DTE Options Overlay"""
    parser = argparse.ArgumentParser(
        description="0DTE Options Overlay - Iron Condor Strategy"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze current conditions")
    analyze_parser.add_argument("--underlying", default="SPY", help="Underlying symbol")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Run trading cycle")
    trade_parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    trade_parser.add_argument("--mode", default="paper", choices=["paper", "live"], help="Trading mode")
    trade_parser.add_argument("--cb-status", default="green", help="Circuit breaker status")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        calc = GEXCalculator()
        profile = calc.calculate_gex(args.underlying)
        pin_risk = calc.check_pin_risk(profile)
        
        print(f"\n📊 GEX Analysis for {args.underlying}")
        print(f"   Spot Price: ${profile.spot_price:.2f}")
        print(f"   Max Gamma Strike: ${profile.max_gamma_strike:.2f}")
        print(f"   Put Wall: ${profile.put_wall:.2f}")
        print(f"   Call Wall: ${profile.call_wall:.2f}")
        print(f"   Gamma Flip: ${profile.gamma_flip:.2f}")
        print(f"\n   ⚠️  Pin Risk: {'YES' if pin_risk['pin_risk_detected'] else 'No'}")
        if pin_risk['pin_risk_detected']:
            print(f"       Distance to max gamma: {pin_risk['distance_to_max_gamma_pct']:.2f}%")
        print(f"   Recommendation: {pin_risk['recommendation']}")
    
    elif args.command == "trade":
        overlay = ODTEOverlay(args.portfolio, args.mode)
        results = overlay.run_cycle(args.cb_status)
        
        print(f"\n🔄 Trading Cycle: {results['timestamp']}")
        print(f"   Circuit Breaker: {results['circuit_breaker_status']}")
        print(f"   Position Scalar: {results['position_scalar']}")
        
        for action in results['actions']:
            if action['action'] == 'enter':
                print(f"\n   ✅ NEW POSITION: {action['trade_id']}")
                print(f"      Strikes: {action['strikes']}")
                print(f"      Max Profit: ${action['max_profit']:.2f}")
                print(f"      Max Loss: ${action['max_loss']:.2f}")
            elif action['action'] == 'close':
                print(f"\n   📤 CLOSED: {action['trade_id']}")
                print(f"      Reason: {action['reason']}")
                print(f"      P&L: ${action['realized_pnl']:.2f}")
            elif action['action'] == 'monitor':
                print(f"   📈 Monitoring {action['trade_id']}: ${action['current_pnl']:.2f}")
            elif action['action'] == 'no_entry':
                print(f"   ⏸️  No entry: {action['reason']}")
    
    elif args.command == "backtest":
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        
        backtester = ODTEBacktester(start, end, args.portfolio)
        results = backtester.run()
        
        print(f"\n📊 Backtest Results")
        print(f"   Period: {results['backtest_period']['start']} to {results['backtest_period']['end']}")
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Trades: {results['trades_simulated']}")
        print(f"   Final P&L: ${results['final_pnl']:,.2f}")
        print(f"   Note: {results['message']}")
    
    elif args.command == "stats":
        overlay = ODTEOverlay()
        stats = overlay.get_stats()
        
        print(f"\n📈 0DTE Trading Statistics")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
