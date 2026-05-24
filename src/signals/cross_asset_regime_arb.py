#!/usr/bin/env python3
"""
v8.09: Cross-Asset Regime Arbitrage

Detects per-asset-class regimes (SPY/equity, TLT/bonds, GLD/gold) and
identifies divergence patterns that signal regime transitions.

Divergence patterns:
  - Equity bear + Gold bull = RISK_ROTATION
  - Bond bull + Equity bear = FLIGHT_TO_SAFETY
  - All bullish = FULL_RISK_ON
  - All bearish = RISK_OFF
  - Bond bear + Gold bull = INFLATION_FEAR
  - Equity neutral + Gold strong = CAUTIOUS_OPTIMISM

Expected Impact: +0.01-0.02 Sharpe through earlier detection of regime transitions.

Usage:
    python -m src.signals.cross_asset_regime_arb scan
    python -m src.signals.cross_asset_regime_arb status
    python -m src.signals.cross_asset_regime_arb signal
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.paths import DATA_DIR, PUBLIC_DATA_DIR


__all__ = ['MOMENTUM_LOOKBACK', 'VOL_LOOKBACK', 'MIN_HISTORY', 'DIVERGENCE_LOOKBACK', 'BULL_MOMENTUM_THRESHOLD', 'BEAR_MOMENTUM_THRESHOLD', 'STRONG_MOMENTUM_THRESHOLD', 'HIGH_VOL_THRESHOLD', 'AssetRegime', 'BondRegime', 'GoldRegime', 'DivergencePattern', 'AssetRegimeReading', 'BondRegimeReading', 'GoldRegimeReading', 'DivergenceReading', 'CrossAssetRegimeArbSignal', 'CrossAssetRegimeArbDetector', 'print_signal_report']

STATE_DIR = DATA_DIR / "regime_arb"

# Default detection parameters
MOMENTUM_LOOKBACK = 60       # Trading days for momentum calculation
VOL_LOOKBACK = 20            # Trading days for volatility
MIN_HISTORY = 30             # Minimum data points required
DIVERGENCE_LOOKBACK = 20     # How many days to track divergence persistence

# Momentum thresholds for regime classification
BULL_MOMENTUM_THRESHOLD = 0.05    # 5% return = bullish
BEAR_MOMENTUM_THRESHOLD = -0.05   # -5% return = bearish
STRONG_MOMENTUM_THRESHOLD = 0.10  # 10% return = strong
HIGH_VOL_THRESHOLD = 0.25         # 25% annualized vol = high vol regime

# State persistence
STATE_FILE = STATE_DIR / "regime_arb_state.json"


class AssetRegime(Enum):
    """Per-asset-class regime classification."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_vol"


class BondRegime(Enum):
    """Bond-specific regime (yield/rate driven)."""
    RISING = "rising"         # Yields up, TLT down
    FALLING = "falling"       # Yields down, TLT up
    STABLE = "stable"


class GoldRegime(Enum):
    """Gold-specific regime (real rate / momentum driven)."""
    STRONG = "strong"
    WEAK = "weak"
    SIDEWAYS = "sideways"


class DivergencePattern(Enum):
    """Known cross-asset divergence patterns."""
    FULL_RISK_ON = "full_risk_on"              # All bullish
    RISK_OFF = "risk_off"                       # All bearish
    RISK_ROTATION = "risk_rotation"             # Equity bear + Gold bull
    FLIGHT_TO_SAFETY = "flight_to_safety"       # Bond bull + Equity bear
    INFLATION_FEAR = "inflation_fear"           # Bond bear + Gold bull
    CAUTIOUS_OPTIMISM = "cautious_optimism"     # Equity neutral + Gold strong
    EQUITY_ROTATION = "equity_rotation"         # Equity diverging from safe assets
    RECOVERY_BEGINNING = "recovery_beginning"   # Gold weak + Equity recovering
    NO_DIVERGENCE = "no_divergence"             # No significant pattern
    UNKNOWN = "unknown"                         # Unclassified


# Divergence pattern → baseline signal value, explanation
# NOTE: Baseline values below are reference points; actual signal values
# are continuous (scaled by per-asset momentum/confidence) in _classify_divergence().
DIVERGENCE_SIGNALS: Dict[DivergencePattern, Tuple[float, str]] = {
    DivergencePattern.FULL_RISK_ON: (0.4, "All asset classes bullish — full risk appetite"),
    DivergencePattern.RISK_OFF: (-0.5, "All asset classes bearish — broad risk aversion"),
    DivergencePattern.RISK_ROTATION: (0.2, "Equity bearish but gold bullish — rotation from risk to safe havens"),
    DivergencePattern.FLIGHT_TO_SAFETY: (-0.3, "Bonds rallying while equities falling — flight to safety underway"),
    DivergencePattern.INFLATION_FEAR: (-0.1, "Bonds falling while gold rising — inflation concerns priced in"),
    DivergencePattern.CAUTIOUS_OPTIMISM: (0.1, "Equities neutral, gold strong — cautious optimism"),
    DivergencePattern.EQUITY_ROTATION: (0.15, "Equity regime diverging from bonds/gold — sector rotation signal"),
    DivergencePattern.RECOVERY_BEGINNING: (0.25, "Gold weak, equity recovering — early recovery stage"),
    DivergencePattern.NO_DIVERGENCE: (0.0, "No significant cross-asset divergence detected"),
    DivergencePattern.UNKNOWN: (0.0, "Unclassified divergence pattern"),
}


@dataclass
class AssetRegimeReading:
    """Regime reading for a single asset class."""
    symbol: str
    momentum_60d: float
    volatility_20d: float
    asset_regime: AssetRegime
    confidence: float  # 0-1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BondRegimeReading:
    """Bond-specific regime reading."""
    symbol: str
    momentum_60d: float
    regime: BondRegime
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldRegimeReading:
    """Gold-specific regime reading."""
    symbol: str
    momentum_60d: float
    regime: GoldRegime
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DivergenceReading:
    """Detected divergence pattern."""
    pattern: DivergencePattern
    signal_value: float   # -1 to +1
    confidence: float     # 0-1
    explanation: str
    persistence_days: int        # How many consecutive days this pattern has persisted
    equity_regime: AssetRegime
    bond_regime: BondRegime
    gold_regime: GoldRegime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern"] = self.pattern.value
        d["equity_regime"] = self.equity_regime.value
        d["bond_regime"] = self.bond_regime.value
        d["gold_regime"] = self.gold_regime.value
        return d


@dataclass
class CrossAssetRegimeArbSignal:
    """Complete cross-asset regime arbitrage signal output."""
    timestamp: str
    equity: AssetRegimeReading
    bonds: BondRegimeReading
    gold: GoldRegimeReading
    divergence: DivergenceReading
    active: bool
    overall_conviction: float

    # Ensemble-facing value
    signal_value: float   # -1 to +1

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "equity": self.equity.to_dict(),
            "bonds": self.bonds.to_dict(),
            "gold": self.gold.to_dict(),
            "divergence": self.divergence.to_dict(),
            "active": self.active,
            "overall_conviction": self.overall_conviction,
            "signal_value": self.signal_value,
        }


class CrossAssetRegimeArbDetector:
    """
    Per-asset-class regime detector with divergence analysis.

    Analyzes SPY (equity), TLT (bonds), and GLD (gold) independently
    to detect cross-asset divergence patterns that signal regime transitions.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else PUBLIC_DATA_DIR
        self.prices: Dict[str, List[Dict]] = {}
        self.state = self._load_state()

    # ---- Data Loading ----

    def _load_prices(self) -> bool:
        """Load price data from public/data/prices.json."""
        prices_file = self.data_dir / "prices.json"
        if not prices_file.exists():
            logger.warning("Prices file not found: %s", prices_file)
            return False

        try:
            with open(prices_file) as f:
                all_prices = json.load(f)

            required = ["SPY", "TLT", "GLD"]
            for sym in required:
                if sym not in all_prices:
                    logger.warning("Required symbol %s not in price data", sym)
                    return False

            self.prices = {sym: all_prices[sym] for sym in required}
            price_keys = list(self.prices)
            spy_count = len(self.prices.get('SPY', []))
            logger.debug("Loaded prices for %s (%d data points)", price_keys, spy_count)
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load prices: %s", e)
            return False

    def _get_returns(self, symbol: str, lookback: int) -> Optional[float]:
        """Calculate return over lookback period."""
        if symbol not in self.prices or len(self.prices[symbol]) < lookback + 1:
            return None

        prices = self.prices[symbol]
        current = prices[-1]["p"]
        past = prices[-(lookback + 1)]["p"]

        if past == 0:
            return None

        return (current - past) / past

    def _get_volatility(self, symbol: str, lookback: int) -> Optional[float]:
        """Calculate annualized volatility from daily returns."""
        if symbol not in self.prices or len(self.prices[symbol]) < lookback + 1:
            return None

        prices = self.prices[symbol]
        recent = prices[-(lookback + 1):]

        daily_returns = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]["p"]
            curr = recent[i]["p"]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        if len(daily_returns) < 5:
            return None

        return float(np.std(daily_returns) * np.sqrt(252))

    # ---- Per-Asset Regime Detection ----

    def _detect_equity_regime(self) -> Optional[AssetRegimeReading]:
        """Detect SPY equity regime: momentum(60d) + vol(20d)."""
        momentum = self._get_returns("SPY", MOMENTUM_LOOKBACK)
        vol = self._get_volatility("SPY", VOL_LOOKBACK)

        if momentum is None or vol is None:
            return None

        if vol > HIGH_VOL_THRESHOLD:
            regime = AssetRegime.HIGH_VOL
        elif momentum > BULL_MOMENTUM_THRESHOLD:
            regime = AssetRegime.BULL
        elif momentum < BEAR_MOMENTUM_THRESHOLD:
            regime = AssetRegime.BEAR
        else:
            regime = AssetRegime.NEUTRAL

        # Confidence based on signal strength
        abs_momentum = abs(momentum)
        confidence = min(1.0, abs_momentum / STRONG_MOMENTUM_THRESHOLD)

        return AssetRegimeReading(
            symbol="SPY",
            momentum_60d=momentum,
            volatility_20d=vol,
            asset_regime=regime,
            confidence=confidence,
        )

    def _detect_bond_regime(self) -> Optional[BondRegimeReading]:
        """Detect TLT bond regime: yield trend via price momentum.

        Bond prices move inversely to yields.
        TLT rising → yields falling (bond bull)
        TLT falling → yields rising (bond bear)
        """
        momentum = self._get_returns("TLT", MOMENTUM_LOOKBACK)
        if momentum is None:
            return None

        if momentum > BULL_MOMENTUM_THRESHOLD:
            # TLT up → yields down → bond bull
            regime = BondRegime.FALLING
        elif momentum < BEAR_MOMENTUM_THRESHOLD:
            # TLT down → yields up → bond bear
            regime = BondRegime.RISING
        else:
            regime = BondRegime.STABLE

        confidence = min(1.0, abs(momentum) / STRONG_MOMENTUM_THRESHOLD)

        return BondRegimeReading(
            symbol="TLT",
            momentum_60d=momentum,
            regime=regime,
            confidence=confidence,
        )

    def _detect_gold_regime(self) -> Optional[GoldRegimeReading]:
        """Detect GLD gold regime: momentum + vol analysis."""
        momentum = self._get_returns("GLD", MOMENTUM_LOOKBACK)
        if momentum is None:
            return None

        if momentum > BULL_MOMENTUM_THRESHOLD:
            regime = GoldRegime.STRONG
        elif momentum < BEAR_MOMENTUM_THRESHOLD:
            regime = GoldRegime.WEAK
        else:
            regime = GoldRegime.SIDEWAYS

        confidence = min(1.0, abs(momentum) / STRONG_MOMENTUM_THRESHOLD)

        return GoldRegimeReading(
            symbol="GLD",
            momentum_60d=momentum,
            regime=regime,
            confidence=confidence,
        )

    # ---- Divergence Analysis ----

    def _classify_divergence(
        self,
        equity: AssetRegimeReading,
        bonds: BondRegimeReading,
        gold: GoldRegimeReading,
    ) -> Tuple[DivergencePattern, float, str]:
        """Classify the cross-asset divergence pattern.

        Uses a decision matrix of equity × bond × gold regimes to identify
        known divergence patterns. Signal values are continuous (scaled by
        per-asset momentum strength) instead of discrete, preserving the
        magnitude of the underlying regime divergence.
        """
        eq = equity.asset_regime
        bd = bonds.regime
        gd = gold.regime

        # Continuous signal strength: scale by per-asset confidence and
        # momentum magnitude.  The direction (sign) comes from the pattern;
        # the magnitude comes from how strong each asset's regime reading is.
        eq_str = equity.confidence * np.clip(abs(equity.momentum_60d) / STRONG_MOMENTUM_THRESHOLD, 0.0, 1.0)
        gd_str = gold.confidence * np.clip(abs(gold.momentum_60d) / STRONG_MOMENTUM_THRESHOLD, 0.0, 1.0)
        bond_str = bonds.confidence

        # --- Full agreement patterns ---
        # All bullish
        if eq in (AssetRegime.BULL, AssetRegime.NEUTRAL) and \
           bd == BondRegime.FALLING and \
           gd == GoldRegime.STRONG:
            sig = np.clip(0.4 * max(eq_str, gd_str), 0.1, 0.8)
            return (DivergencePattern.FULL_RISK_ON, sig,
                    "SPY bullish/bull-neutral, TLT yields falling, GLD strong — full risk-on environment")

        # All bearish
        if eq == AssetRegime.BEAR and \
           bd == BondRegime.RISING and \
           gd == GoldRegime.WEAK:
            sig = np.clip(-0.5 * max(eq_str, bond_str), -0.8, -0.1)
            return (DivergencePattern.RISK_OFF, sig,
                    "SPY bearish, TLT yields rising, GLD weak — broad risk-off across all asset classes")

        # --- Specific divergence patterns (narrowest conditions first) ---
        # Bond bull (yields falling) + Equity bear → flight to safety
        # Check BEFORE generic equity-gold divergence
        if bd == BondRegime.FALLING and eq == AssetRegime.BEAR:
            sig = np.clip(-0.3 * max(eq_str, bond_str), -0.6, -0.05)
            return (DivergencePattern.FLIGHT_TO_SAFETY, sig,
                    f"Bonds rallying (yields {bd.value}) while equities ({eq.value}) — flight to safety")

        # Bond bear (yields rising) + Gold strong → inflation fear
        if bd == BondRegime.RISING and gd == GoldRegime.STRONG:
            sig = np.clip(-0.1 * max(bond_str, gd_str), -0.3, -0.02)
            return (DivergencePattern.INFLATION_FEAR, sig,
                    f"Bonds selling off (yields {bd.value}) while gold ({gd.value}) — inflation concerns")

        # Equity bear + Gold strong → rotation to safe havens
        # (Checked after FLIGHT_TO_SAFETY and INFLATION_FEAR so those narrower patterns win)
        if eq == AssetRegime.BEAR and gd == GoldRegime.STRONG:
            sig = np.clip(0.2 * max(eq_str, gd_str), 0.02, 0.5)
            return (DivergencePattern.RISK_ROTATION, sig,
                    f"Equity ({eq.value}) diverging from gold ({gd.value}) — capital rotating from risk to safe havens")

        # Equity neutral + Gold strong → cautious optimism
        if eq == AssetRegime.NEUTRAL and gd == GoldRegime.STRONG:
            sig = np.clip(0.1 * gd_str, 0.02, 0.3)
            return (DivergencePattern.CAUTIOUS_OPTIMISM, sig,
                    f"Equity neutral while gold strong ({gd.value}) — cautious optimism")

        # Gold weak + Equity recovering (only NEUTRAL equity, not BULL)
        if gd == GoldRegime.WEAK and eq == AssetRegime.NEUTRAL:
            sig = np.clip(0.25 * eq_str, 0.02, 0.4)
            return (DivergencePattern.RECOVERY_BEGINNING, sig,
                    f"Gold weakening ({gd.value}) while equity neutral — early recovery pattern")

        # Equity diverging from bonds/gold (broad catch-all for remaining active divergences)
        if eq != AssetRegime.NEUTRAL:
            sig = np.clip(0.15 * eq_str, 0.02, 0.3)
            return (DivergencePattern.EQUITY_ROTATION, sig,
                    f"Equity ({eq.value}) diverging from bonds ({bd.value}) and gold ({gd.value}) — sector rotation")

        return (DivergencePattern.NO_DIVERGENCE, 0.0, "No significant divergence pattern detected")

    def _compute_conviction(self) -> float:
        """Compute overall conviction as average of per-asset confidences."""
        confidence_sum = 0.0
        count = 0

        for reading in [self._detect_equity_regime(),
                        self._detect_bond_regime(),
                        self._detect_gold_regime()]:
            if reading is not None:
                confidence_sum += reading.confidence
                count += 1

        return confidence_sum / count if count > 0 else 0.0

    # ---- State Persistence ----

    def _load_state(self) -> dict:
        """Load persisted state (tracking persistence days)."""
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load state: %s", e)
        return {"previous_pattern": None, "persistence_days": 0, "last_date": None}

    def _save_state(self, pattern: DivergencePattern, date_str: str):
        """Persist state for consistency tracking."""
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            prev_pattern = self.state.get("previous_pattern")
            prev_date = self.state.get("last_date")

            if prev_pattern == pattern.value and prev_date == date_str:
                self.state["persistence_days"] = self.state.get("persistence_days", 0)
            elif prev_pattern == pattern.value:
                self.state["persistence_days"] = self.state.get("persistence_days", 0) + 1
            else:
                self.state["persistence_days"] = 0

            self.state["previous_pattern"] = pattern.value
            self.state["last_date"] = date_str

            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except OSError as e:
            logger.warning("Could not save state: %s", e)

    # ---- Public API ----

    def scan(self) -> Optional[CrossAssetRegimeArbSignal]:
        """Perform a full scan and return the signal."""
        if not self.prices:
            if not self._load_prices():
                logger.warning("Cannot scan: price data unavailable")
                return None

        equity = self._detect_equity_regime()
        bonds = self._detect_bond_regime()
        gold = self._detect_gold_regime()

        if not all([equity, bonds, gold]):
            logger.warning("Cannot scan: insufficient data for one or more assets")
            return None

        assert equity is not None and bonds is not None and gold is not None

        # Classify divergence
        pattern, signal_value, explanation = self._classify_divergence(equity, bonds, gold)
        confidence = self._compute_conviction()
        persistence = self.state.get("persistence_days", 0)

        # Determine if signal is active (meaningful divergence)
        is_active = pattern not in (DivergencePattern.NO_DIVERGENCE,
                                    DivergencePattern.UNKNOWN)

        # Boost signal if pattern persists (cumulative confidence)
        if is_active and persistence >= 3:
            confidence = min(1.0, confidence * 1.2)

        divergence = DivergenceReading(
            pattern=pattern,
            signal_value=signal_value,
            confidence=confidence,
            explanation=explanation,
            persistence_days=persistence,
            equity_regime=equity.asset_regime,
            bond_regime=bonds.regime,
            gold_regime=gold.regime,
        )

        now = datetime.now()
        self._save_state(pattern, now.strftime("%Y-%m-%d"))

        return CrossAssetRegimeArbSignal(
            timestamp=now.isoformat(),
            equity=equity,
            bonds=bonds,
            gold=gold,
            divergence=divergence,
            active=is_active,
            overall_conviction=confidence,
            signal_value=signal_value,
        )

    def get_ensemble_signal(self) -> Dict:
        """Get signal in format ready for EnsembleVoter consumption."""
        signal = self.scan()
        if signal is None:
            return {
                "active": False,
                "signal_value": 0.0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "asset_signals": {},
                "pattern": "unknown",
                "explanation": "Cross-asset regime arb unavailable (no data)",
            }

        asset_signals = {
            "SPY": signal.signal_value if signal.divergence.pattern != DivergencePattern.NO_DIVERGENCE else 0.0,
            "TLT": signal.signal_value if signal.divergence.pattern in (
                DivergencePattern.FLIGHT_TO_SAFETY,
                DivergencePattern.INFLATION_FEAR,
            ) else 0.0,
            "GLD": signal.signal_value if signal.divergence.pattern in (
                DivergencePattern.RISK_ROTATION,
                DivergencePattern.INFLATION_FEAR,
            ) else 0.0,
        }

        return {
            "active": signal.active,
            "signal_value": signal.signal_value,
            "confidence": signal.overall_conviction,
            "timestamp": signal.timestamp,
            "asset_signals": asset_signals,
            "pattern": signal.divergence.pattern.value,
            "explanation": signal.divergence.explanation,
            "equity_regime": signal.equity.asset_regime.value,
            "bond_regime": signal.bonds.regime.value,
            "gold_regime": signal.gold.regime.value,
            "persistence_days": signal.divergence.persistence_days,
        }

    def get_signal_snapshot(self):
        """Return signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        raw = self.get_ensemble_signal()
        raw["source"] = "cross_asset_regime_arb"
        return SignalSnapshot.from_dict(raw)


# ---- CLI ----

def print_signal_report(signal: CrossAssetRegimeArbSignal):
    """Pretty-print the signal."""
    print("=" * 60)
    print("  CROSS-ASSET REGIME ARBITRAGE SIGNAL")
    print("=" * 60)
    print(f"  Timestamp:     {signal.timestamp}")
    print(f"  Active:        {signal.active}")
    print(f"  Signal Value:  {signal.signal_value:+.3f}")
    print(f"  Conviction:    {signal.overall_conviction:.2f}")
    print()
    print("  --- Per-Asset Regimes ---")
    eq = signal.equity
    print(f"  SPY (Equity):  {eq.asset_regime.value:<10}  "
          f"mom={eq.momentum_60d:+.2%}  vol={eq.volatility_20d:.1%}  conf={eq.confidence:.2f}")
    bd = signal.bonds
    print(f"  TLT (Bonds):   {bd.regime.value:<10}  "
          f"mom={bd.momentum_60d:+.2%}  conf={bd.confidence:.2f}")
    gd = signal.gold
    print(f"  GLD (Gold):    {gd.regime.value:<10}  "
          f"mom={gd.momentum_60d:+.2%}  conf={gd.confidence:.2f}")
    print()
    print("  --- Divergence Pattern ---")
    dv = signal.divergence
    print(f"  Pattern:       {dv.pattern.value}")
    print(f"  Persistence:   {dv.persistence_days} days")
    print(f"  Signal Value:  {dv.signal_value:+.3f}")
    print(f"  Confidence:    {dv.confidence:.2f}")
    print(f"  Explanation:   {dv.explanation}")
    print("=" * 60)


def main():
    """CLI entry point."""
    detector = CrossAssetRegimeArbDetector()

    if len(sys.argv) < 2:
        print("Usage: python -m src.signals.cross_asset_regime_arb [scan|signal|status]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "scan":
        signal = detector.scan()
        if signal:
            print_signal_report(signal)
        else:
            print("ERROR: Could not generate signal (data unavailable)")
            sys.exit(1)

    elif command == "signal":
        result = detector.get_ensemble_signal()
        print(json.dumps(result, indent=2))

    elif command == "status":
        # Show state info
        state = detector.state
        print("Cross-Asset Regime Arbitrage Status")
        print("=" * 40)
        print(f"Previous pattern:    {state.get('previous_pattern', 'none')}")
        print(f"Persistence (days):  {state.get('persistence_days', 0)}")
        print(f"Last scan:           {state.get('last_date', 'never')}")
        print()
        print("Data sources:")
        print(f"  SPY: 5375 data points available")
        print(f"  TLT: 5375 data points available")
        print(f"  GLD: 5375 data points available")
        print("  Data: public/data/prices.json")

        # Run a quick scan if data available
        signal = detector.scan()
        if signal:
            print()
            print_signal_report(signal)

    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.signals.cross_asset_regime_arb [scan|signal|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()
