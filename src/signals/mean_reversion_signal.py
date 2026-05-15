#!/usr/bin/env python3
"""
Portfolio-Lab v4.81: VIX-Gated Mean-Reversion Signal Generator

Implements mean-reversion overlay triggered by VIX regime:
- VIX < 20: Trend-following regime (no mean-reversion)
- VIX 20-30: Mixed regime (reduced mean-reversion sizing)
- VIX > 30: Mean-reversion regime (activate dip-buying)
- VIX > 40: Crisis regime (freeze — use circuit breaker instead)

Entry Conditions:
1. VIX > 30 (elevated fear)
2. SPY down >2% in 3 trading days (short-term oversold)
3. SPY > 200-day MA (no secular bear)
4. VPIN < 0.6 (no toxic flow)

Exit Conditions:
1. SPY recovers to entry price (trade to breakeven or better)
2. VIX drops below 25 (fear subsiding)
3. Stop loss: -3% from entry
4. Max hold: 10 trading days

Usage:
    python -m src.signals.mean_reversion_signal signal      # Generate current signal
    python -m src.signals.mean_reversion_signal backtest    # Historical backtest
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_PATH = PROJECT_ROOT / "public/data/prices.json"
STATE_PATH = DATA_DIR / "mean_reversion_state.json"
VIX_STATE_PATH = DATA_DIR / "signals/vix_term_structure_signal.json"
VIX_INSURANCE_PATH = DATA_DIR / "signals/vix_insurance_signal.json"

# Constants
VIX_LOW = 20.0          # Below this: trend-following regime
VIX_MIXED = 30.0        # 20-30: mixed regime (reduced sizing)
VIX_CRISIS = 40.0       # Above this: crisis freeze
VIX_EXIT = 25.0         # VIX must drop below this to exit

ENTRY_SPY_DROP_PCT = -2.0       # SPY must drop >2% in 3 days
ENTRY_LOOKBACK_DAYS = 3         # Days to check SPY drop
ENTRY_VPIN_MAX = 0.6             # Max VPIN to allow entry

STOP_LOSS_PCT = -3.0            # -3% stop loss from entry
MAX_HOLD_DAYS = 10              # Max 10 trading days hold
EXIT_RECOVERY_RATIO = 1.0       # Exit when SPY recovers to entry price

BASE_ALLOC_PCT = 2.0            # Base allocation 2% of portfolio
SCALE_ALLOC_PCT = 1.0           # Additional 1% per -2% SPY drop
MAX_ALLOC_PCT = 5.0             # Max 5% total allocation

VPIN_THRESHOLD = 0.6            # VPIN toxicity threshold


class VIXRegime(Enum):
    """VIX regime classification for gating."""
    TREND_FOLLOW = "trend_follow"      # VIX < 20
    MIXED = "mixed"                     # VIX 20-30
    MEAN_REVERSION = "mean_reversion"  # VIX 30-40
    CRISIS_FREEZE = "crisis_freeze"    # VIX > 40


class MeanReversionState(Enum):
    """Current mean-reversion trade state."""
    IDLE = "idle"               # No active position
    ENTERING = "entering"       # Entry conditions met, allocating
    ACTIVE = "active"           # In a mean-reversion trade
    STOPPED = "stopped"         # Hit stop loss
    EXITED = "exited"           # Exited normally
    EXPIRED = "expired"         # Max hold reached


@dataclass
class MeanReversionSignal:
    """Mean-reversion signal with entry/exit conditions."""
    timestamp: str
    vix_level: float
    vix_regime: str
    spy_price: float
    spy_3d_return: float
    spy_above_200ma: bool
    vpin_level: float
    vpin_ok: bool
    
    # Entry conditions
    entry_triggered: bool
    entry_reason: str
    
    # Active trade state
    trade_state: str
    trade_entry_price: Optional[float]
    trade_hold_days: int
    trade_return_pct: float
    
    # Recommended allocation
    recommended_allocation_pct: float
    allocation_rationale: str
    
    # Overall signal
    signal_value: float  # -1 to +1, negative = bullish mean-reversion (buy dip)
    signal_strength: float  # 0.0 to 1.0


@dataclass
class MeanReversionTrade:
    """Record of a completed mean-reversion trade."""
    entry_date: str
    exit_date: str
    entry_spy: float
    exit_spy: float
    return_pct: float
    hold_days: int
    vix_at_entry: float
    allocation_pct: float
    exit_reason: str  # "recovery", "vix_drop", "stop_loss", "expired"


class VIXMeanReversionCalculator:
    """Computes VIX-gated mean-reversion signals from market data."""

    def __init__(self, prices_path: Path = PRICES_PATH):
        self.prices_path = prices_path
        self._data: Dict[str, List[Dict]] = {}
        self._prices: Dict[str, np.ndarray] = {}
        self._dates: List[str] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load prices from JSON."""
        if not self.prices_path.exists():
            logger.error(f"Prices file not found: {self.prices_path}")
            return
        with open(self.prices_path) as f:
            self._data = json.load(f)

        # Build price arrays for known tickers
        for ticker in ["SPY", "GLD", "TLT", "^VIX", "VIX"]:
            raw = self._data.get(ticker, [])
            if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
                self._prices[ticker] = np.array([r["p"] for r in raw], dtype=np.float64)
                if self._dates and len(raw) == len(self._dates):
                    pass
                elif not self._dates:
                    self._dates = [r["d"] for r in raw]

    def get_vix_level(self) -> Optional[float]:
        """Get latest VIX level from existing VIX state files."""
        # Try VIX term structure state first
        if VIX_STATE_PATH.exists():
            try:
                with open(VIX_STATE_PATH) as f:
                    state = json.load(f)
                vix = state.get("vix_spot")
                if vix is not None:
                    return float(vix)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        # Try VIX insurance state
        if VIX_INSURANCE_PATH.exists():
            try:
                with open(VIX_INSURANCE_PATH) as f:
                    state = json.load(f)
                vix = state.get("spot_vix")
                if vix is not None:
                    return float(vix)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        # Fallback: estimate VIX from SPY volatility
        spy_prices = self._prices.get("SPY", np.array([]))
        if len(spy_prices) > 20:
            spy_returns = np.diff(spy_prices[-21:]) / spy_prices[-21:-1]
            realized_vol = float(np.std(spy_returns) * np.sqrt(252) * 100)
            return max(realized_vol, 10.0)  # Floor at 10
        
        return None

    def get_vix_level_at(self, idx: int) -> Optional[float]:
        """Get VIX level at a specific index using SPY vol proxy."""
        spy_prices = self._prices.get("SPY", np.array([]))
        if len(spy_prices) < 21 or idx < 20:
            return 20.0  # Default to moderate VIX
        window = spy_prices[max(0, idx - 20):idx + 1]
        if len(window) < 5:
            return 20.0
        spy_returns = np.diff(window) / window[:-1]
        realized_vol = float(np.std(spy_returns) * np.sqrt(252) * 100)
        return max(realized_vol, 10.0)

    def get_vix_historical(self) -> np.ndarray:
        """Get VIX history approximated from SPY realized volatility."""
        spy_prices = self._prices.get("SPY", np.array([]))
        if len(spy_prices) < 21:
            return np.array([])
        
        # Compute 20-day rolling annualized volatility as VIX proxy
        vix_proxy = np.full(len(spy_prices), fill_value=np.nan)
        for i in range(20, len(spy_prices)):
            window = spy_prices[i - 20:i + 1]
            returns = np.diff(window) / window[:-1]
            vix_proxy[i] = max(np.std(returns) * np.sqrt(252) * 100, 10.0)
        
        # Forward-fill NaN at beginning
        first_valid = np.where(~np.isnan(vix_proxy))[0]
        if len(first_valid) > 0:
            vix_proxy[:first_valid[0]] = vix_proxy[first_valid[0]]
        
        return vix_proxy

    def get_spy_prices(self) -> np.ndarray:
        """Get SPY price history."""
        return self._prices.get("SPY", np.array([]))

    def classify_vix_regime(self, vix: float) -> VIXRegime:
        """Classify VIX level into a regime."""
        if vix >= VIX_CRISIS:
            return VIXRegime.CRISIS_FREEZE
        elif vix >= VIX_MIXED:
            return VIXRegime.MEAN_REVERSION
        elif vix >= VIX_LOW:
            return VIXRegime.MIXED
        return VIXRegime.TREND_FOLLOW

    def compute_spy_3d_return(self, spy_prices: np.ndarray) -> float:
        """Compute 3-trading-day return for SPY."""
        if len(spy_prices) < 4:
            return 0.0
        return float((spy_prices[-1] / spy_prices[-4] - 1) * 100)

    def compute_spy_return(self, spy_prices: np.ndarray, lookback: int) -> float:
        """Compute return over lookback trading days."""
        if len(spy_prices) <= lookback:
            return 0.0
        return float((spy_prices[-1] / spy_prices[-(lookback + 1)] - 1) * 100)

    def check_spy_above_200ma(self, spy_prices: np.ndarray) -> bool:
        """Check if SPY is above its 200-day moving average."""
        if len(spy_prices) < 200:
            return True  # Not enough data, assume bullish
        ma200 = float(np.mean(spy_prices[-200:]))
        return spy_prices[-1] > ma200

    def get_spy_200ma(self, spy_prices: np.ndarray) -> float:
        """Get 200-day moving average of SPY."""
        if len(spy_prices) < 200:
            return float(spy_prices[-1])
        return float(np.mean(spy_prices[-200:]))

    def spy_above_200ma_at(self, spy_prices: np.ndarray, idx: int) -> bool:
        """Check if SPY is above its 200-day MA at a specific index."""
        if idx < 200:
            return True
        ma200 = float(np.mean(spy_prices[idx - 200:idx]))
        return spy_prices[idx] > ma200

    def compute_vpin(self, idx: int = -1) -> Tuple[float, bool]:
        """
        Estimate VPIN (Volume-synchronized Probability of Informed Trading).
        Uses price change as a proxy for volume when true volume data unavailable.
        Returns (vpin_estimate, is_ok).
        """
        spy_prices = self.get_spy_prices()
        if len(spy_prices) < 60:
            return 0.3, True  # Default: low VPIN

        if idx < 0:
            idx = len(spy_prices) - 1

        window = spy_prices[max(0, idx - 60):idx + 1]
        if len(window) < 20:
            return 0.3, True

        # VPIN proxy: ratio of absolute returns to total range
        daily_returns = np.diff(window) / window[:-1]
        abs_returns = np.abs(daily_returns)
        total_range = np.max(window) - np.min(window)
        
        if total_range == 0:
            return 0.3, True

        # Normalize: higher absolute returns relative to range = higher toxicity
        vpin_estimate = float(np.mean(abs_returns) * len(window) / (total_range / np.mean(window)))
        vpin_estimate = min(max(vpin_estimate, 0.0), 1.0)
        
        return vpin_estimate, vpin_estimate < VPIN_THRESHOLD

    def _load_vpin_state(self) -> float:
        """Load VPIN level from vix_overlay state or compute fresh."""
        vpin_path = DATA_DIR / "vix_overlay_state.json"
        if vpin_path.exists():
            try:
                with open(vpin_path) as f:
                    state = json.load(f)
                return state.get("vpin_level", 0.5)
            except (json.JSONDecodeError, KeyError):
                pass
        vpin, _ = self.compute_vpin()
        return vpin

    def compute_trade_state(self, state_path: Path = STATE_PATH) -> Dict:
        """Load or initialize persistent trade state."""
        default_state = {
            "active": False,
            "entry_date": None,
            "entry_price": None,
            "entry_vix": None,
            "hold_days": 0,
            "allocation_pct": 0.0,
        }
        if state_path.exists():
            try:
                with open(state_path) as f:
                    saved = json.load(f)
                return {**default_state, **saved}
            except (json.JSONDecodeError, TypeError):
                pass
        return default_state

    def save_trade_state(self, state: Dict, state_path: Path = STATE_PATH) -> None:
        """Save trade state to persistent storage."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def generate_signal(self) -> MeanReversionSignal:
        """Generate current mean-reversion signal based on market conditions."""
        spy_prices = self.get_spy_prices()
        vix = self.get_vix_level()
        
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Default signal (no mean-reversion opportunity)
        signal = MeanReversionSignal(
            timestamp=now,
            vix_level=vix or 15.0,
            vix_regime="unknown",
            spy_price=float(spy_prices[-1]) if len(spy_prices) > 0 else 0.0,
            spy_3d_return=0.0,
            spy_above_200ma=True,
            vpin_level=0.5,
            vpin_ok=True,
            entry_triggered=False,
            entry_reason="",
            trade_state="idle",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=0.0,
            allocation_rationale="No mean-reversion opportunity",
            signal_value=0.0,
            signal_strength=0.0,
        )

        if vix is None or len(spy_prices) < 5:
            signal.allocation_rationale = "Insufficient data"
            return signal

        # Determine VIX regime
        vix_regime = self.classify_vix_regime(vix)
        signal.vix_regime = vix_regime.value

        # Compute SPY metrics
        spy_3d_return = self.compute_spy_3d_return(spy_prices)
        signal.spy_3d_return = spy_3d_return
        signal.spy_price = float(spy_prices[-1])
        signal.spy_above_200ma = self.check_spy_above_200ma(spy_prices)

        # Load VPIN
        vpin = self._load_vpin_state()
        signal.vpin_level = vpin
        signal.vpin_ok = vpin < VPIN_THRESHOLD

        # Load trade state
        trade_state = self.compute_trade_state()
        current_hold = trade_state.get("hold_days", 0) + 1  # Increment each day

        # --- Entry Logic ---
        entry_conditions_met = (
            vix >= VIX_MIXED and vix < VIX_CRISIS  # VIX 30-40
            and spy_3d_return <= ENTRY_SPY_DROP_PCT  # SPY down >2% in 3 days
            and signal.spy_above_200ma  # Above 200D MA (no secular bear)
            and vpin < VPIN_THRESHOLD  # No toxic flow
            and not trade_state["active"]  # Not in a trade already
        )

        if entry_conditions_met:
            # Calculate allocation: base + scale
            additional_drop = abs(spy_3d_return) - abs(ENTRY_SPY_DROP_PCT)
            scale_units = int(additional_drop / abs(SCALE_ALLOC_PCT))
            allocation = min(BASE_ALLOC_PCT + scale_units * SCALE_ALLOC_PCT, MAX_ALLOC_PCT)
            
            signal.entry_triggered = True
            signal.entry_reason = (
                f"VIX={vix:.1f} >30 (fear elevated), "
                f"SPY 3d return={spy_3d_return:.1f}% (oversold), "
                f"VPIN={vpin:.2f} (clean)"
            )
            signal.trade_state = "entering"
            signal.recommended_allocation_pct = allocation
            signal.allocation_rationale = (
                f"ENTRY: SPY oversold ({spy_3d_return:.1f}% in 3d) during "
                f"elevated VIX ({vix:.1f}). Allocate {allocation:.0f}% from GLD/Cash reserve."
            )
            signal.signal_value = -0.5  # Negative = bullish mean-reversion
            signal.signal_strength = min(allocation / MAX_ALLOC_PCT, 1.0)

        # --- Active Trade Logic ---
        elif trade_state["active"]:
            entry_price = trade_state["entry_price"]
            entry_vix = trade_state.get("entry_vix", vix)
            current_return = float((spy_prices[-1] / entry_price - 1) * 100)
            
            signal.trade_state = "active"
            signal.trade_entry_price = entry_price
            signal.trade_hold_days = trade_state.get("hold_days", 0)
            signal.trade_return_pct = current_return
            signal.recommended_allocation_pct = trade_state.get("allocation_pct", BASE_ALLOC_PCT)

            # Check exit conditions
            exit_reason = None
            
            # Exit 1: SPY recovered to entry price
            if current_return >= 0:
                exit_reason = f"recovery: SPY returned {current_return:.1f}% from entry"
            
            # Exit 2: VIX dropped below 25
            elif vix < VIX_EXIT:
                exit_reason = f"vix_drop: VIX fell to {vix:.1f} (below {VIX_EXIT})"
            
            # Exit 3: Stop loss hit
            elif current_return <= STOP_LOSS_PCT:
                exit_reason = f"stop_loss: SPY down {current_return:.1f}% from entry (limit {STOP_LOSS_PCT}%)"
            
            # Exit 4: Max hold reached
            elif signal.trade_hold_days >= MAX_HOLD_DAYS:
                exit_reason = f"expired: max hold of {MAX_HOLD_DAYS} days reached"

            if exit_reason:
                signal.trade_state = "exited"
                signal.allocation_rationale = f"EXIT: {exit_reason}"
                signal.recommended_allocation_pct = 0.0
                signal.signal_value = 0.0
                signal.signal_strength = 0.0
                
                # Save trade record
                trade_record = MeanReversionTrade(
                    entry_date=trade_state.get("entry_date", "unknown"),
                    exit_date=now,
                    entry_spy=entry_price or 0.0,
                    exit_spy=float(spy_prices[-1]),
                    return_pct=current_return,
                    hold_days=trade_state.get("hold_days", 0),
                    vix_at_entry=entry_vix,
                    allocation_pct=trade_state.get("allocation_pct", 0.0),
                    exit_reason=exit_reason.split(":")[0],
                )
                self._save_trade_record(trade_record)
                
                # Reset trade state
                self.save_trade_state({
                    "active": False,
                    "entry_date": None,
                    "entry_price": None,
                    "entry_vix": None,
                    "hold_days": 0,
                    "allocation_pct": 0.0,
                })
            else:
                # Trade continues
                self.save_trade_state({
                    "active": True,
                    "entry_date": trade_state["entry_date"],
                    "entry_price": entry_price,
                    "entry_vix": entry_vix,
                    "hold_days": signal.trade_hold_days + 1,
                    "allocation_pct": trade_state.get("allocation_pct", BASE_ALLOC_PCT),
                })
                signal.allocation_rationale = (
                    f"HOLDING: SPY {current_return:.1f}% from entry "
                    f"(day {signal.trade_hold_days}/{MAX_HOLD_DAYS}), VIX={vix:.1f}"
                )
                signal.signal_value = -0.3  # Still mean-reversion bias
                signal.signal_strength = 0.5

        # --- No Entry + No Active Trade ---
        else:
            if vix >= VIX_CRISIS:
                signal.allocation_rationale = f"CRISIS FREEZE: VIX={vix:.1f} >40. Circuit breaker active."
            elif vix_regime == VIXRegime.TREND_FOLLOW:
                signal.allocation_rationale = (
                    f"TREND MODE: VIX={vix:.1f} <20. "
                    f"Trend-following regime — no mean-reversion."
                )
            elif spy_3d_return > ENTRY_SPY_DROP_PCT:
                signal.allocation_rationale = (
                    f"WAITING: SPY 3d return {spy_3d_return:.1f}% "
                    f"(not oversold, need <{abs(ENTRY_SPY_DROP_PCT)}%)"
                )
            elif not signal.spy_above_200ma:
                signal.allocation_rationale = (
                    f"NO LONG: SPY below 200D MA ({self.get_spy_200ma(spy_prices):.1f}). "
                    f"Secular bear — no dip-buying."
                )
            elif not signal.vpin_ok:
                signal.allocation_rationale = f"TOXIC FLOW: VPIN={vpin:.2f} >=0.6. Deferring entry."

        return signal

    def _save_trade_record(self, trade: MeanReversionTrade):
        """Save a completed trade record to the trade log."""
        trades_path = DATA_DIR / "mean_reversion_trades.json"
        existing = []
        if trades_path.exists():
            try:
                with open(trades_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, TypeError):
                existing = []
        existing.append(asdict(trade))
        with open(trades_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def run_backtest(self) -> Dict:
        """
        Historical backtest of the VIX-gated mean-reversion strategy.
        Tests entry/exit rules across full price history.
        Returns performance metrics.
        """
        spy_prices = self.get_spy_prices()
        vix_prices = self.get_vix_historical()
        
        if len(spy_prices) == 0 or len(vix_prices) == 0:
            logger.error("Insufficient data for backtest")
            return {"error": "Insufficient data"}
        
        # Align lengths
        min_len = min(len(spy_prices), len(vix_prices))
        spy = spy_prices[:min_len]
        vix = vix_prices[:min_len]
        
        trades: List[MeanReversionTrade] = []
        in_trade = False
        entry_price = 0.0
        entry_idx = 0
        entry_vix = 0.0
        hold_days = 0
        current_alloc = 0.0
        entry_date = ""
        
        dates = self._dates[:min_len] if len(self._dates) >= min_len else [f"day_{i}" for i in range(min_len)]
        
        for i in range(200, min_len - 1):  # Need 200 days for MA
            current_spy = spy[i]
            current_vix = vix[i]
            current_date = dates[i] if i < len(dates) else f"day_{i}"
            
            if in_trade:
                hold_days += 1
                current_return = float((current_spy / entry_price - 1) * 100)
                
                # Check exits
                exit_reason = None
                
                # Stop loss
                if current_return <= STOP_LOSS_PCT:
                    exit_reason = "stop_loss"
                # Recovery
                elif current_return >= 0:
                    exit_reason = "recovery"
                # VIX drop
                elif current_vix < VIX_EXIT:
                    exit_reason = "vix_drop"
                # Max hold
                elif hold_days >= MAX_HOLD_DAYS:
                    exit_reason = "expired"
                
                if exit_reason:
                    trades.append(MeanReversionTrade(
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_spy=entry_price,
                        exit_spy=current_spy,
                        return_pct=current_return,
                        hold_days=hold_days,
                        vix_at_entry=entry_vix,
                        allocation_pct=current_alloc,
                        exit_reason=exit_reason,
                    ))
                    in_trade = False
                    hold_days = 0
                    current_alloc = 0.0
            else:
                # Check entry
                if i >= 4:
                    spy_3d_return = float((spy[i] / spy[i - 3] - 1) * 100)
                else:
                    spy_3d_return = 0.0
                
                above_200ma = self.spy_above_200ma_at(spy, i)
                
                # VPIN proxy at this index
                vpin, vpin_ok = self.compute_vpin(i)
                
                entry_ok = (
                    VIX_MIXED <= current_vix < VIX_CRISIS
                    and spy_3d_return <= ENTRY_SPY_DROP_PCT
                    and above_200ma
                    and vpin_ok
                )
                
                if entry_ok:
                    # Calculate allocation
                    additional_drop = abs(spy_3d_return) - abs(ENTRY_SPY_DROP_PCT)
                    scale_units = int(additional_drop / abs(SCALE_ALLOC_PCT))
                    current_alloc = min(BASE_ALLOC_PCT + scale_units * SCALE_ALLOC_PCT, MAX_ALLOC_PCT)
                    
                    in_trade = True
                    entry_price = current_spy
                    entry_idx = i
                    entry_vix = current_vix
                    hold_days = 0
                    entry_date = current_date
        
        # Close any open trade at end
        if in_trade:
            final_return = float((spy[-1] / entry_price - 1) * 100)
            trades.append(MeanReversionTrade(
                entry_date=entry_date,
                exit_date=dates[-1] if dates else "end",
                entry_spy=entry_price,
                exit_spy=spy[-1],
                return_pct=final_return,
                hold_days=hold_days,
                vix_at_entry=entry_vix,
                allocation_pct=current_alloc,
                exit_reason="end_of_data",
            ))
        
        # Compute metrics
        if not trades:
            return {
                "total_trades": 0,
                "message": "No trades generated in backtest period",
                "date_range": f"{dates[200]} to {dates[-1]}" if len(dates) > 200 else "insufficient data",
            }
        
        trade_returns = [t.return_pct for t in trades]
        win_trades = [r for r in trade_returns if r > 0]
        loss_trades = [r for r in trade_returns if r <= 0]
        
        win_rate = len(win_trades) / len(trade_returns) * 100 if trade_returns else 0
        avg_win = np.mean(win_trades) if win_trades else 0.0
        avg_loss = np.mean(loss_trades) if loss_trades else 0.0
        total_return = np.sum(trade_returns)
        
        # Count by exit reason
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        # Sharpe-like metric (using trade returns)
        sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(12)) if np.std(trade_returns) > 0 else 0.0
        
        # Avg hold period
        avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0
        
        return {
            "total_trades": len(trades),
            "date_range": f"{dates[200]} to {dates[-1]}" if len(dates) > 200 else "insufficient data",
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(float(avg_win), 2),
            "avg_loss_pct": round(float(avg_loss), 2),
            "total_return_pct": round(float(total_return), 2),
            "avg_return_per_trade_pct": round(float(np.mean(trade_returns)), 2),
            "trade_sharpe_annualized": round(float(sharpe), 3),
            "avg_hold_days": round(float(avg_hold), 1),
            "exit_reasons": exit_reasons,
            "best_trade_pct": round(float(np.max(trade_returns)), 2),
            "worst_trade_pct": round(float(np.min(trade_returns)), 2),
            "median_trade_pct": round(float(np.median(trade_returns)), 2),
            "total_return_sum_pct": round(float(total_return), 2),
        }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VIX-Gated Mean-Reversion Signal Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Signal command
    signal_parser = subparsers.add_parser("signal", help="Generate current mean-reversion signal")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run historical backtest")
    
    args = parser.parse_args()
    
    calc = VIXMeanReversionCalculator()
    
    if args.command == "signal":
        signal = calc.generate_signal()
        print(json.dumps(asdict(signal), indent=2, default=str))
    
    elif args.command == "backtest":
        results = calc.run_backtest()
        print(json.dumps(results, indent=2, default=str))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
