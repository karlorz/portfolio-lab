#!/usr/bin/env python3
"""
v5.73: ML-Light Regime Predictor

A lightweight, deterministic regime classifier that runs without ML dependencies
(no sklearn, hmmlearn, torch). Uses threshold-based classification from market data.

Pure numpy/pandas — runs in all environments (safe mode, crontab, no-ML).

Output Regimes:
  - LOW_VOL: VIX < 14 proxy, positive slope, low spreads
  - NORMAL: Low vol, positive momentum, normal conditions
  - HIGH_VOL: Vol > 20% or bearish momentum shift
  - CRISIS: Vol > 30% AND bearish momentum AND drawdown
  - RECOVERY: Recent crisis + improving indicators

Usage:
    python -m src.signals.regime_classifier scan       # Current regime scan
    python -m src.signals.regime_classifier history    # Historical regime scan
    python -m src.signals.regime_classifier signal     # Get signal value
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_PATH = PROJECT_ROOT / "public" / "data" / "prices.json"
STATE_PATH = DATA_DIR / "regime_classifier_state.json"

# ── Parameters ──────────────────────────────────────────────────────────────

# Vol thresholds (annualized)
LOW_VOL_THRESH = 0.14      # Below this = LOW_VOL
HIGH_VOL_THRESH = 0.20     # Above this = HIGH_VOL
CRISIS_VOL_THRESH = 0.30   # Above this = CRISIS (with confirmation)

# Momentum thresholds (20-day returns)
MOM_NEGATIVE = -0.03       # Below this = bearish
MOM_POSITIVE = 0.02        # Above this = bullish

# Drawdown thresholds
DD_CAUTION = -0.05         # -5% drawdown caution
DD_CRISIS = -0.08          # -8% drawdown crisis threshold

# Gold/Equity ratio thresholds (60-day return differential)
GE_DEFLATION = -0.05       # GLD underperforming SPY by 5%
GE_SAFETY = 0.05           # GLD outperforming SPY by 5%

# Lookback windows
VOL_WINDOW = 20
MOM_WINDOW = 20
TREND_WINDOW = 60
DD_WINDOW = 60
GE_WINDOW = 60

# Confirmation periods
CRISIS_COOLDOWN_DAYS = 10  # Min days after crisis before transition
RECOVERY_MIN_MOMENTUM = 0.03  # Min 20-day return to confirm recovery


class Regime(Enum):
    """Deterministic regime classification."""
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


REGIME_CONFIDENCE_MAP = {
    Regime.LOW_VOL: 0.75,
    Regime.NORMAL: 0.70,
    Regime.HIGH_VOL: 0.80,
    Regime.CRISIS: 0.90,
    Regime.RECOVERY: 0.70,
    Regime.UNKNOWN: 0.30,
}

REGIME_DESCRIPTION = {
    Regime.LOW_VOL: "Low volatility environment — risk-on favorable",
    Regime.NORMAL: "Normal market conditions — standard allocation",
    Regime.HIGH_VOL: "Elevated volatility — defensive positioning",
    Regime.CRISIS: "Crisis conditions — maximum risk reduction",
    Regime.RECOVERY: "Post-crisis recovery — gradual re-risking",
    Regime.UNKNOWN: "Insufficient data for classification",
}


@dataclass
class RegimeFactors:
    """Raw factor readings used for classification."""
    timestamp: str
    spy_vol_20d: float          # Annualized 20-day SPY volatility
    spy_mom_20d: float          # SPY 20-day return
    spy_mom_60d: float          # SPY 60-day return
    spy_drawdown_60d: float     # Max drawdown over 60 days
    gld_spy_ratio_60d: float    # GLD return - SPY return over 60d
    tlt_ief_ratio_60d: float    # TLT return - IEF return (flight to safety)
    last_crisis_days_ago: Optional[int] = None  # Days since last crisis regime
    spy_current_price: Optional[float] = None
    gld_current_price: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegimeReading:
    """Complete regime classification reading."""
    timestamp: str
    regime: Regime
    confidence: float
    factors: RegimeFactors
    regime_reason: str = ""
    regime_duration_days: Optional[int] = None
    previous_regime: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.value
        d["previous_regime"] = self.previous_regime
        d["regime_description"] = REGIME_DESCRIPTION.get(self.regime, "")
        return d


class RegimeClassifier:
    """
    Deterministic regime classifier using market data thresholds.
    No ML dependencies — pure numpy/pandas.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.prices: Optional[Dict[str, List[Dict]]] = None
        self.state_path = self.data_dir / "regime_classifier_state.json"

        # State persistence
        self.current_regime: Regime = Regime.NORMAL
        self.previous_regime: Optional[Regime] = None
        self.regime_start_date: Optional[str] = None
        self.regime_history: List[RegimeReading] = []
        self._load_state()

    # ── Data Loading ────────────────────────────────────────────────────────

    def load_prices(self) -> Optional[Dict[str, List[Dict]]]:
        """Load price data from JSON.

        Resolution order:
        1. (self.data_dir parent) / public / data / prices.json
        2. self.data_dir / prices.json
        3. Default PROJECT_ROOT / public / data / prices.json (only if using default data_dir)
        """
        # If a custom data_dir is provided, only look in paths relative to it
        using_default = self.data_dir.resolve() == Path(PROJECT_ROOT / "data").resolve()

        candidates = [
            self.data_dir.parent / "public" / "data" / "prices.json",
            self.data_dir / "prices.json",
        ]

        if using_default:
            candidates.append(PROJECT_ROOT / "public" / "data" / "prices.json")
            if PRICES_PATH.exists():
                candidates.append(PRICES_PATH)

        prices_path = None
        for path in candidates:
            if path.exists():
                prices_path = path
                break

        if prices_path is None:
            logger.warning("Prices file not found in any expected location")
            return None

        try:
            with open(prices_path) as f:
                self.prices = json.load(f)
            return self.prices
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            return None

    def _get_series(self, symbol: str) -> Optional[np.ndarray]:
        """Get price series for a symbol as numpy array."""
        if self.prices is None:
            self.load_prices()
        if self.prices is None or symbol not in self.prices:
            return None
        return np.array([p["p"] for p in self.prices[symbol]])

    def _get_dates(self, symbol: str = "SPY") -> Optional[List[str]]:
        """Get date strings for a symbol."""
        if self.prices is None:
            self.load_prices()
        if self.prices is None or symbol not in self.prices:
            return None
        return [p["d"] for p in self.prices[symbol]]

    # ── Factor Computation ──────────────────────────────────────────────────

    def compute_factors(self, lookback_days: int = 60) -> Optional[RegimeFactors]:
        """
        Compute all regime factors from market data.
        Returns None if insufficient data.
        """
        spy = self._get_series("SPY")
        if spy is None or len(spy) < lookback_days + VOL_WINDOW:
            logger.warning("Insufficient SPY data for factor computation")
            return None

        dates = self._get_dates()
        if dates is None:
            return None

        # SPY returns
        spy_returns = np.diff(spy) / spy[:-1]

        # Latest factors
        spy_recent_20 = spy_returns[-VOL_WINDOW:]
        spy_recent_60 = spy_returns[-TREND_WINDOW:]

        # 20-day realized vol (annualized)
        spy_vol_20d = float(np.std(spy_recent_20) * np.sqrt(252))

        # Momentum
        spy_mom_20d = float(np.sum(spy_recent_20))  # 20-day return
        spy_mom_60d = float(np.sum(spy_recent_60))  # 60-day return

        # Max drawdown over 60 days
        spy_window = spy[-DD_WINDOW:]
        running_max = np.maximum.accumulate(spy_window)
        drawdowns = spy_window / running_max - 1
        spy_drawdown_60d = float(np.min(drawdowns))

        # Gold/Equity ratio
        gld = self._get_series("GLD")
        gld_spy_ratio_60d = 0.0
        tlt_ief_ratio_60d = 0.0
        spy_current_price = float(spy[-1])
        gld_current_price = None

        if gld is not None and len(gld) >= TREND_WINDOW and len(spy) >= TREND_WINDOW:
            gld_return = (gld[-1] - gld[-TREND_WINDOW]) / gld[-TREND_WINDOW]
            spy_return_60d = (spy[-1] - spy[-TREND_WINDOW]) / spy[-TREND_WINDOW]
            gld_spy_ratio_60d = float(gld_return - spy_return_60d)
            gld_current_price = float(gld[-1])

        # TLT/IEF ratio (flight to safety)
        tlt = self._get_series("TLT")
        ief = self._get_series("IEF")
        if tlt is not None and ief is not None and len(tlt) >= TREND_WINDOW and len(ief) >= TREND_WINDOW:
            tlt_return = (tlt[-1] - tlt[-TREND_WINDOW]) / tlt[-TREND_WINDOW]
            ief_return = (ief[-1] - ief[-TREND_WINDOW]) / ief[-TREND_WINDOW]
            tlt_ief_ratio_60d = float(tlt_return - ief_return)

        # Days since last crisis
        last_crisis_days_ago = self._days_since_regime(Regime.CRISIS)

        return RegimeFactors(
            timestamp=dates[-1] if dates else datetime.now().isoformat(),
            spy_vol_20d=spy_vol_20d,
            spy_mom_20d=spy_mom_20d,
            spy_mom_60d=spy_mom_60d,
            spy_drawdown_60d=spy_drawdown_60d,
            gld_spy_ratio_60d=gld_spy_ratio_60d,
            tlt_ief_ratio_60d=tlt_ief_ratio_60d,
            last_crisis_days_ago=last_crisis_days_ago,
            spy_current_price=spy_current_price,
            gld_current_price=gld_current_price,
        )

    # ── Classification ──────────────────────────────────────────────────────

    def classify(self, factors: Optional[RegimeFactors] = None) -> RegimeReading:
        """
        Classify current regime from market factors.
        """
        if factors is None:
            factors = self.compute_factors()

        if factors is None:
            return RegimeReading(
                timestamp=datetime.now().isoformat(),
                regime=Regime.UNKNOWN,
                confidence=0.3,
                factors=RegimeFactors(
                    timestamp=datetime.now().isoformat(),
                    spy_vol_20d=0.0,
                    spy_mom_20d=0.0,
                    spy_mom_60d=0.0,
                    spy_drawdown_60d=0.0,
                    gld_spy_ratio_60d=0.0,
                    tlt_ief_ratio_60d=0.0,
                ),
                regime_reason="Insufficient data for classification",
            )

        # -- Decision Tree (deterministic, no ML) --

        regime = Regime.NORMAL
        reason_parts = []
        confidence = REGIME_CONFIDENCE_MAP[Regime.NORMAL]

        # Check CRISIS first (highest priority)
        is_crisis = (
            factors.spy_vol_20d > CRISIS_VOL_THRESH
            and factors.spy_mom_20d < MOM_NEGATIVE
            and factors.spy_drawdown_60d < DD_CRISIS
        )
        # Crisis can also be triggered by extreme vol alone with bearish momentum
        is_vol_crisis = (
            factors.spy_vol_20d > CRISIS_VOL_THRESH * 1.2  # 36% vol
            and factors.spy_mom_20d < 0
        )
        # Or extreme drawdown alone
        is_dd_crisis = factors.spy_drawdown_60d < DD_CRISIS * 1.5  # -12% drawdown

        if is_crisis or is_vol_crisis or is_dd_crisis:
            regime = Regime.CRISIS
            confidence = REGIME_CONFIDENCE_MAP[Regime.CRISIS]
            triggers = []
            if factors.spy_vol_20d > CRISIS_VOL_THRESH:
                triggers.append(f"vol={factors.spy_vol_20d:.1%}")
            if factors.spy_mom_20d < MOM_NEGATIVE:
                triggers.append(f"mom={factors.spy_mom_20d:.2%}")
            if factors.spy_drawdown_60d < DD_CRISIS:
                triggers.append(f"dd={factors.spy_drawdown_60d:.2%}")
            reason_parts.append(f"Crisis: {', '.join(triggers)}")
            confidence = min(confidence + 0.05, 0.95)

        # Check RECOVERY if we were recently in crisis
        elif (self.current_regime == Regime.CRISIS or self.current_regime == Regime.RECOVERY):
            if (factors.spy_mom_60d > RECOVERY_MIN_MOMENTUM
                    and factors.spy_vol_20d < HIGH_VOL_THRESH
                    and factors.spy_drawdown_60d > DD_CAUTION):
                regime = Regime.RECOVERY
                confidence = REGIME_CONFIDENCE_MAP[Regime.RECOVERY]
                reason_parts.append(
                    f"Recovery: mom={factors.spy_mom_60d:.2%}, vol={factors.spy_vol_20d:.1%}"
                )
            else:
                # Stay in CRISIS if conditions haven't improved
                if (factors.spy_vol_20d > HIGH_VOL_THRESH
                        or factors.spy_mom_20d < MOM_NEGATIVE):
                    regime = Regime.CRISIS
                    confidence = REGIME_CONFIDENCE_MAP[Regime.CRISIS] * 0.8
                    reason_parts.append("Continued crisis conditions")

        # Check HIGH_VOL
        if regime == Regime.NORMAL:
            if (factors.spy_vol_20d > HIGH_VOL_THRESH
                    or (factors.spy_vol_20d > LOW_VOL_THRESH
                        and factors.spy_mom_20d < MOM_NEGATIVE)):
                regime = Regime.HIGH_VOL
                confidence = REGIME_CONFIDENCE_MAP[Regime.HIGH_VOL]
                if factors.spy_vol_20d > HIGH_VOL_THRESH:
                    reason_parts.append(f"High vol: {factors.spy_vol_20d:.1%}")
                if factors.spy_mom_20d < MOM_NEGATIVE:
                    reason_parts.append(f"Bearish mom: {factors.spy_mom_20d:.2%}")

        # Check LOW_VOL
        if regime == Regime.NORMAL:
            if (factors.spy_vol_20d < LOW_VOL_THRESH
                    and factors.spy_mom_20d > 0
                    and factors.spy_drawdown_60d > -0.03):
                regime = Regime.LOW_VOL
                confidence = REGIME_CONFIDENCE_MAP[Regime.LOW_VOL]
                reason_parts.append(
                    f"Low vol: {factors.spy_vol_20d:.1%}, mom={factors.spy_mom_20d:.2%}"
                )

        # Fallback to NORMAL
        if regime == Regime.NORMAL:
            reason_parts.append(
                f"Normal: vol={factors.spy_vol_20d:.1%}, mom={factors.spy_mom_20d:.2%}"
            )

        # Compute regime duration
        regime_duration = self._get_regime_duration()

        # Update state
        self.previous_regime = self.current_regime
        if regime != self.current_regime:
            self.current_regime = regime
            self.regime_start_date = factors.timestamp

        reading = RegimeReading(
            timestamp=factors.timestamp,
            regime=regime,
            confidence=confidence,
            factors=factors,
            regime_reason="; ".join(reason_parts),
            regime_duration_days=regime_duration,
            previous_regime=self.previous_regime.value if self.previous_regime else None,
        )

        # Store history
        self.regime_history.append(reading)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]

        # Persist state
        self._save_state(reading)

        return reading

    # ── State Persistence ───────────────────────────────────────────────────

    def _load_state(self):
        """Load persisted regime state."""
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.current_regime = Regime(state.get("current_regime", "normal"))
            self.regime_start_date = state.get("regime_start_date")
            prev = state.get("previous_regime")
            self.previous_regime = Regime(prev) if prev else None
            # Restore limited history
            history = state.get("history", [])
            for h in history[-10:]:
                try:
                    factors = RegimeFactors(**h.get("factors", {}))
                    self.regime_history.append(RegimeReading(
                        timestamp=h.get("timestamp", ""),
                        regime=Regime(h.get("regime", "normal")),
                        confidence=h.get("confidence", 0.5),
                        factors=factors,
                        regime_reason=h.get("regime_reason", ""),
                    ))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load regime state: {e}")

    def _save_state(self, reading: Optional[RegimeReading] = None):
        """Save regime state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "current_regime": self.current_regime.value,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "regime_start_date": self.regime_start_date,
            "last_updated": datetime.now().isoformat(),
            "history": [r.to_dict() for r in self.regime_history[-20:]],
        }
        if reading:
            state["last_reading"] = reading.to_dict()
        try:
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save regime state: {e}")

    def _days_since_regime(self, regime: Regime) -> Optional[int]:
        """Count trading days since last occurrence of a regime."""
        if self.prices is None:
            self.load_prices()
        dates = self._get_dates()
        if not dates or not self.regime_history:
            return None
        # Find most recent occurrence in history
        for reading in reversed(self.regime_history):
            if reading.regime == regime:
                try:
                    last_date = datetime.strptime(reading.timestamp, "%Y-%m-%d")
                    now = datetime.strptime(dates[-1], "%Y-%m-%d")
                    return (now - last_date).days
                except (ValueError, IndexError):
                    pass
        return None

    def _get_regime_duration(self) -> Optional[int]:
        """Get duration of current regime in trading days."""
        dates = self._get_dates()
        if not dates or not self.regime_start_date:
            return None
        try:
            start = datetime.strptime(self.regime_start_date, "%Y-%m-%d")
            end = datetime.strptime(dates[-1], "%Y-%m-%d")
            return (end - start).days
        except (ValueError, IndexError):
            return None

    # ── Signal Output ───────────────────────────────────────────────────────

    def get_signal_value(self, reading: Optional[RegimeReading] = None) -> float:
        """
        Convert regime to a -1 to +1 signal value for EnsembleVoter.
        Positive = risk-on, Negative = risk-off.
        """
        if reading is None:
            reading = self.classify()

        signal_map = {
            Regime.LOW_VOL: 0.6,    # Mildly bullish — low vol environment
            Regime.NORMAL: 0.2,      # Slightly bullish — normal conditions
            Regime.HIGH_VOL: -0.3,   # Slightly bearish — defensive
            Regime.CRISIS: -0.8,     # Strongly bearish — risk-off
            Regime.RECOVERY: 0.4,    # Mildly bullish — improving conditions
            Regime.UNKNOWN: 0.0,     # Neutral — insufficient data
        }
        return signal_map.get(reading.regime, 0.0)

    def get_asset_signals(self, reading: Optional[RegimeReading] = None) -> Dict[str, float]:
        """
        Get per-asset regime-based signals.
        """
        if reading is None:
            reading = self.classify()

        # Base signals by regime
        regime_asset_signals = {
            Regime.LOW_VOL: {"SPY": 0.3, "GLD": -0.1, "TLT": -0.2, "IEF": -0.1},
            Regime.NORMAL: {"SPY": 0.1, "GLD": 0.0, "TLT": 0.0, "IEF": 0.0},
            Regime.HIGH_VOL: {"SPY": -0.3, "GLD": 0.2, "TLT": 0.3, "IEF": 0.2},
            Regime.CRISIS: {"SPY": -0.8, "GLD": 0.5, "TLT": 0.6, "IEF": 0.4},
            Regime.RECOVERY: {"SPY": 0.4, "GLD": -0.2, "TLT": -0.3, "IEF": -0.1},
            Regime.UNKNOWN: {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0, "IEF": 0.0},
        }
        return regime_asset_signals.get(reading.regime, {})


# ── CLI ────────────────────────────────────────────────────────────────────

def print_scan(reading: RegimeReading):
    """Print formatted regime scan output."""
    print("=" * 60)
    print("ML-LIGHT REGIME PREDICTOR v5.73")
    print("=" * 60)
    print(f"Timestamp:     {reading.timestamp}")
    print(f"Regime:        {reading.regime.value.upper()}")
    print(f"Confidence:    {reading.confidence:.0%}")
    print(f"Description:   {REGIME_DESCRIPTION.get(reading.regime, '')}")
    print(f"Reason:        {reading.regime_reason}")
    if reading.regime_duration_days is not None:
        print(f"Duration:      {reading.regime_duration_days} days")
    if reading.previous_regime:
        print(f"Previous:      {reading.previous_regime}")
    print()
    print("── Market Factors ──")
    print(f"SPY 20d Vol:   {reading.factors.spy_vol_20d:.1%}")
    print(f"SPY 20d Mom:   {reading.factors.spy_mom_20d:.2%}")
    print(f"SPY 60d Mom:   {reading.factors.spy_mom_60d:.2%}")
    print(f"SPY 60d DD:    {reading.factors.spy_drawdown_60d:.2%}")
    print(f"GLD-SPY 60d:   {reading.factors.gld_spy_ratio_60d:.2%}")
    print(f"TLT-IEF 60d:   {reading.factors.tlt_ief_ratio_60d:.2%}")
    if reading.factors.spy_current_price:
        print(f"SPY Price:     ${reading.factors.spy_current_price:.2f}")
    if reading.factors.gld_current_price:
        print(f"GLD Price:     ${reading.factors.gld_current_price:.2f}")
    print()
    print(f"Signal Value:  {0.0:+.3f} → {reading.regime.value}, {reading.confidence:.0%} confidence")


def main():
    """CLI entry point."""
    import sys

    classifier = RegimeClassifier()
    classifier.load_prices()

    if len(sys.argv) < 2 or sys.argv[1] == "scan":
        reading = classifier.classify()
        print_scan(reading)
        classifier._save_state(reading)

    elif sys.argv[1] == "history":
        # Scan last N days and show regime transitions
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 252
        if classifier.prices is None:
            classifier.load_prices()
        spy = classifier._get_series("SPY")
        if spy is None:
            print("No price data available")
            return

        print(f"Scanning last {n} days for regime transitions...")
        print()

        # We need to run classification sequentially to track regime transitions
        # Re-initialize for clean history
        classifier2 = RegimeClassifier()
        classifier2.load_prices()

        transitions = []
        # Sample every 5 days for speed
        step = max(1, len(spy) // min(n, len(spy)))
        for i in range(len(spy) - 60, len(spy), step):
            reading = classifier2.classify()
            transitions.append(reading)

        # Print summary
        regimes = {}
        for r in transitions:
            key = r.regime.value
            regimes[key] = regimes.get(key, 0) + 1

        print("Regime Distribution (last N samples):")
        for regime, count in sorted(regimes.items(), key=lambda x: -x[1]):
            pct = count / len(transitions) * 100
            print(f"  {regime.upper():12s}: {count:3d} samples ({pct:.1f}%)")

        # Show latest reading
        if transitions:
            print()
            print_scan(transitions[-1])

    elif sys.argv[1] == "signal":
        # Get numeric signal for ensemble voter
        reading = classifier.classify()
        signal_value = classifier.get_signal_value(reading)
        print(f"{signal_value:.4f}")

    else:
        print("Usage: python -m src.signals.regime_classifier [scan|history|signal]")


if __name__ == "__main__":
    main()
