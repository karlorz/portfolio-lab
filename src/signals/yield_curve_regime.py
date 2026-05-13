#!/usr/bin/env python3
"""
Yield Curve Regime Classifier - v3.11 Phase 1
Dynamic duration targeting based on 10Y-2Y yield curve regime

Regimes:
- Inverted (< -0.25%): Short duration (2-3yr) outperforms by 150-200bps
- Flat (-0.25% to +0.75%): Neutral 5-7yr duration optimal
- Steep (> +0.75%): Long duration (10-15yr) adds 100-140bps excess return

References:
- AQR 2025: "Duration Timing and the Yield Curve"
- Campbell Harvey: "Yield Curve Inversions and Economic Growth"
- Fed Policy v2.54: FRED integration patterns
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
STATE_PATH = DATA_DIR / ".yield_curve_regime_state.json"

# FRED series IDs
FRED_SERIES = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS30": "30-Year Treasury Constant Maturity Rate",
    "DGS5": "5-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year minus 2-Year Treasury Yield Spread",
}


class YieldCurveRegime(Enum):
    """Yield curve regime classification."""
    INVERTED = "inverted"      # 2s10s < -0.25%
    FLAT = "flat"              # -0.25% <= 2s10s <= +0.75%
    STEEP = "steep"            # 2s10s > +0.75%
    UNKNOWN = "unknown"       # Data unavailable


@dataclass
class YieldCurveData:
    """Yield curve data snapshot."""
    timestamp: str
    dgs10: float  # 10-year yield (%)
    dgs2: float   # 2-year yield (%)
    dgs30: Optional[float]  # 30-year yield (%)
    dgs5: Optional[float]   # 5-year yield (%)
    spread_2s10s: float  # 10Y - 2Y spread (percentage points)
    spread_10s30s: Optional[float]  # 30Y - 10Y spread
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class RegimeClassification:
    """Complete regime classification result."""
    timestamp: str
    regime: YieldCurveRegime
    spread_2s10s: float
    dgs10: float
    dgs2: float
    days_in_regime: int
    regime_start_date: str
    is_transition_pending: bool  # Min 30-day rule
    days_until_eligible: int
    confidence: str  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime.value,
            "spread_2s10s": self.spread_2s10s,
            "dgs10": self.dgs10,
            "dgs2": self.dgs2,
            "days_in_regime": self.days_in_regime,
            "regime_start_date": self.regime_start_date,
            "is_transition_pending": self.is_transition_pending,
            "days_until_eligible": self.days_until_eligible,
            "confidence": self.confidence,
        }


class YieldCurveRegimeClassifier:
    """
    Classifies yield curve regime based on 10Y-2Y spread.
    
    Thresholds (based on AQR/Fed research):
    - Inverted: spread < -0.25% (recession signal)
    - Flat: -0.25% <= spread <= +0.75% (neutral)
    - Steep: spread > +0.75% (growth signal)
    
    Transition rules:
    - Minimum 30 days in new regime before allocation shift
    - 20-day moving average smoothing to avoid whipsaws
    """
    
    # Regime thresholds (percentage points converted to decimals)
    THRESHOLD_INVERTED = -0.0025  # -0.25%
    THRESHOLD_STEEP = 0.0075     # +0.75%
    
    # Transition rules
    MIN_DAYS_IN_REGIME = 30
    SMOOTHING_WINDOW = 20
    
    def __init__(self):
        self.state = self._load_state()
        self.spread_history: List[Tuple[str, float]] = []  # [(date, spread), ...]
        self._load_spread_history()
    
    def _load_state(self) -> Dict:
        """Load classifier state from disk."""
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                state = json.load(f)
                # Convert string regime back to enum
                if "current_regime" in state:
                    state["current_regime"] = YieldCurveRegime(state["current_regime"])
                return state
        return {
            "current_regime": YieldCurveRegime.UNKNOWN,
            "regime_start_date": None,
            "last_update": None,
            "pending_regime": None,
            "pending_since": None,
        }
    
    def _save_state(self):
        """Save classifier state to disk."""
        state = self.state.copy()
        # Convert enum to string for JSON serialization
        if isinstance(state.get("current_regime"), YieldCurveRegime):
            state["current_regime"] = state["current_regime"].value
        if isinstance(state.get("pending_regime"), YieldCurveRegime):
            state["pending_regime"] = state["pending_regime"].value
        
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_spread_history(self):
        """Load spread history from database or FRED cache."""
        try:
            if DB_PATH.exists():
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Try to load from yield_curve_data table first
                cursor.execute("""
                    SELECT date, spread_2s10s FROM yield_curve_data
                    WHERE date >= date('now', '-90 days')
                    ORDER BY date DESC
                    LIMIT 30
                """)
                rows = cursor.fetchall()
                conn.close()
                
                if rows:
                    self.spread_history = [(row[0], row[1]) for row in rows]
                    return
        except Exception as e:
            logger.warning(f"Could not load spread history from DB: {e}")
        
        # Initialize empty if no data
        self.spread_history = []
    
    @staticmethod
    def classify_regime(spread_2s10s: float) -> YieldCurveRegime:
        """
        Classify regime based on 2s10s spread.
        
        Args:
            spread_2s10s: 10Y - 2Y spread in percentage points (e.g., 0.47 for 47bps)
        
        Returns:
            YieldCurveRegime classification
        """
        if spread_2s10s < YieldCurveRegimeClassifier.THRESHOLD_INVERTED:
            return YieldCurveRegime.INVERTED
        elif spread_2s10s > YieldCurveRegimeClassifier.THRESHOLD_STEEP:
            return YieldCurveRegime.STEEP
        else:
            return YieldCurveRegime.FLAT
    
    def get_smoothed_regime(self, spread_2s10s: float) -> YieldCurveRegime:
        """
        Get regime using 20-day moving average smoothing.
        
        This prevents whipsaws during volatile periods.
        """
        # Add current spread to history
        today = datetime.now().strftime("%Y-%m-%d")
        self.spread_history.insert(0, (today, spread_2s10s))
        
        # Keep only last 30 days
        self.spread_history = self.spread_history[:30]
        
        # Need at least smoothing window days
        if len(self.spread_history) < self.SMOOTHING_WINDOW:
            return self.classify_regime(spread_2s10s)
        
        # Calculate moving average
        recent_spreads = [s for _, s in self.spread_history[:self.SMOOTHING_WINDOW]]
        avg_spread = sum(recent_spreads) / len(recent_spreads)
        
        return self.classify_regime(avg_spread)
    
    def _calculate_confidence(self, spread_2s10s: float, 
                             days_in_regime: int) -> str:
        """Calculate confidence level of regime classification."""
        # Distance from threshold (in decimal, e.g., 0.003 = 30bps)
        dist_inverted = abs(spread_2s10s - self.THRESHOLD_INVERTED)
        dist_steep = abs(spread_2s10s - self.THRESHOLD_STEEP)
        min_dist = min(dist_inverted, dist_steep) if spread_2s10s < self.THRESHOLD_STEEP else dist_steep
        
        # High confidence: far from threshold (>30bps) and established regime (>45 days)
        if min_dist > 0.0030 and days_in_regime >= 45:
            return "high"
        # Low confidence: near threshold (<15bps) or new regime (<15 days)
        elif min_dist < 0.0015 or days_in_regime < 15:
            return "low"
        else:
            return "medium"
    
    def classify(self, yield_data: YieldCurveData,
                 use_smoothing: bool = True) -> RegimeClassification:
        """
        Classify current yield curve regime.
        
        Args:
            yield_data: Current yield curve data
            use_smoothing: Whether to apply 20-day MA smoothing
        
        Returns:
            RegimeClassification with transition status
        """
        spread = yield_data.spread_2s10s
        
        # Get raw and smoothed regimes
        raw_regime = self.classify_regime(spread)
        smoothed_regime = self.get_smoothed_regime(spread) if use_smoothing else raw_regime
        
        # Determine effective regime
        effective_regime = smoothed_regime
        
        # Check for regime change
        current_regime = self.state.get("current_regime", YieldCurveRegime.UNKNOWN)
        pending_regime = self.state.get("pending_regime")
        regime_start = self.state.get("regime_start_date")
        
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        is_transition_pending = False
        days_until_eligible = 0
        days_in_regime = 0
        
        if current_regime == YieldCurveRegime.UNKNOWN:
            # First classification
            self.state["current_regime"] = effective_regime
            self.state["regime_start_date"] = today_str
            self.state["last_update"] = today_str
            current_regime = effective_regime
            regime_start = today_str
        
        elif effective_regime != current_regime:
            # Potential regime change
            if pending_regime == effective_regime:
                # Already pending this regime - check if eligible
                pending_since = self.state.get("pending_since")
                if pending_since:
                    pending_date = datetime.strptime(pending_since, "%Y-%m-%d")
                    days_pending = (today - pending_date).days
                    
                    if days_pending >= self.MIN_DAYS_IN_REGIME:
                        # Transition confirmed
                        self.state["current_regime"] = effective_regime
                        self.state["regime_start_date"] = today_str
                        self.state["pending_regime"] = None
                        self.state["pending_since"] = None
                        current_regime = effective_regime
                        regime_start = today_str
                    else:
                        # Still pending
                        is_transition_pending = True
                        days_until_eligible = self.MIN_DAYS_IN_REGIME - days_pending
                else:
                    # Start pending period
                    self.state["pending_regime"] = effective_regime
                    self.state["pending_since"] = today_str
                    is_transition_pending = True
                    days_until_eligible = self.MIN_DAYS_IN_REGIME
            else:
                # New pending regime
                self.state["pending_regime"] = effective_regime
                self.state["pending_since"] = today_str
                is_transition_pending = True
                days_until_eligible = self.MIN_DAYS_IN_REGIME
        else:
            # Still in current regime - clear any pending
            if self.state.get("pending_regime"):
                self.state["pending_regime"] = None
                self.state["pending_since"] = None
        
        self.state["last_update"] = today_str
        self._save_state()
        
        # Calculate days in current regime
        if regime_start:
            regime_start_date = datetime.strptime(regime_start, "%Y-%m-%d")
            days_in_regime = (today - regime_start_date).days
        
        confidence = self._calculate_confidence(spread, days_in_regime)
        
        return RegimeClassification(
            timestamp=today_str,
            regime=current_regime,
            spread_2s10s=spread,
            dgs10=yield_data.dgs10,
            dgs2=yield_data.dgs2,
            days_in_regime=days_in_regime,
            regime_start_date=regime_start or today_str,
            is_transition_pending=is_transition_pending,
            days_until_eligible=max(0, days_until_eligible),
            confidence=confidence,
        )
    
    def get_regime_description(self, regime: YieldCurveRegime) -> str:
        """Get human-readable regime description."""
        descriptions = {
            YieldCurveRegime.INVERTED: (
                f"Inverted Curve (< {self.THRESHOLD_INVERTED}%) - "
                "Short duration preferred (2-3yr). Historically signals recession."
            ),
            YieldCurveRegime.FLAT: (
                f"Flat Curve ({self.THRESHOLD_INVERTED}% to {self.THRESHOLD_STEEP}%) - "
                "Neutral 5-7yr duration optimal. Transitional phase."
            ),
            YieldCurveRegime.STEEP: (
                f"Steep Curve (> {self.THRESHOLD_STEEP}%) - "
                "Long duration (10-15yr) adds 100-140bps excess return."
            ),
            YieldCurveRegime.UNKNOWN: "Unknown - Insufficient data",
        }
        return descriptions.get(regime, "Unknown")
    
    def get_expected_alpha(self, regime: YieldCurveRegime) -> float:
        """
        Get expected annual alpha from dynamic duration vs static.
        Based on historical backtesting research (AQR).
        """
        alphas = {
            YieldCurveRegime.INVERTED: 1.8,   # +180bps vs static
            YieldCurveRegime.FLAT: 0.5,       # +50bps vs static
            YieldCurveRegime.STEEP: 1.2,      # +120bps vs static
            YieldCurveRegime.UNKNOWN: 0.0,
        }
        return alphas.get(regime, 0.0)


def fetch_fred_yield_data() -> Optional[YieldCurveData]:
    """
    Fetch latest yield curve data from FRED or local cache.
    
    Returns:
        YieldCurveData or None if unavailable
    """
    try:
        # Try to fetch from FRED via fed_policy_overlay infrastructure
        from src.signals.fed_policy_overlay import fetch_fred_series
        
        # Fetch last 30 days of data
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        data = fetch_fred_series("DGS10", start_date=start_date)
        
        if data is None or len(data) == 0:
            logger.warning("No FRED data available, checking local cache")
            return _load_cached_yield_data()
        
        dgs10 = data['value'].iloc[-1]
        
        data2 = fetch_fred_series("DGS2", start_date=start_date)
        dgs2 = data2['value'].iloc[-1] if data2 is not None and len(data2) > 0 else None
        
        data30 = fetch_fred_series("DGS30", start_date=start_date)
        dgs30 = data30['value'].iloc[-1] if data30 is not None and len(data30) > 0 else None
        
        data5 = fetch_fred_series("DGS5", start_date=start_date)
        dgs5 = data5['value'].iloc[-1] if data5 is not None and len(data5) > 0 else None
        
        if dgs2 is None:
            logger.error("Cannot calculate spread - DGS2 unavailable")
            return _load_cached_yield_data()
        
        spread_2s10s = (dgs10 - dgs2) / 100  # Convert from percentage to decimal
        spread_10s30s = (dgs30 - dgs10) / 100 if dgs30 else None
        
        return YieldCurveData(
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            dgs10=dgs10 / 100,  # Convert to decimal
            dgs2=dgs2 / 100,
            dgs30=dgs30 / 100 if dgs30 else None,
            dgs5=dgs5 / 100 if dgs5 else None,
            spread_2s10s=spread_2s10s,
            spread_10s30s=spread_10s30s,
        )
        
    except Exception as e:
        logger.error(f"Error fetching FRED data: {e}")
        return _load_cached_yield_data()


def _load_cached_yield_data() -> Optional[YieldCurveData]:
    """Load from cache as fallback."""
    try:
        cache_path = DATA_DIR / ".yield_curve_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
                return YieldCurveData(**data)
    except Exception as e:
        logger.error(f"Error loading cached yield data: {e}")
    
    return None


def save_yield_cache(data: YieldCurveData):
    """Save yield data to cache."""
    cache_path = DATA_DIR / ".yield_curve_cache.json"
    with open(cache_path, 'w') as f:
        json.dump(data.to_dict(), f, indent=2)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Yield Curve Regime Classifier")
    parser.add_argument("--status", action="store_true", help="Show current regime status")
    parser.add_argument("--classify", type=float, help="Classify specific spread value (bps)")
    
    args = parser.parse_args()
    
    classifier = YieldCurveRegimeClassifier()
    
    if args.classify:
        spread = args.classify / 100  # Convert bps to percentage
        regime = classifier.classify_regime(spread)
        print(f"Spread: {args.classify}bps → Regime: {regime.value}")
        print(f"Description: {classifier.get_regime_description(regime)}")
        print(f"Expected Alpha: {classifier.get_expected_alpha(regime):.2f}%")
    
    elif args.status or True:
        # Default to status
        data = fetch_fred_yield_data()
        if data:
            result = classifier.classify(data)
            print(f"\n=== Yield Curve Regime Status ===")
            print(f"Timestamp: {result.timestamp}")
            print(f"Regime: {result.regime.value.upper()}")
            print(f"2s10s Spread: {result.spread_2s10s:.2%}")
            print(f"10Y Yield: {result.dgs10:.2%}")
            print(f"2Y Yield: {result.dgs2:.2%}")
            print(f"Days in Regime: {result.days_in_regime}")
            print(f"Confidence: {result.confidence.upper()}")
            print(f"Transition Pending: {'Yes' if result.is_transition_pending else 'No'}")
            if result.is_transition_pending:
                print(f"Days until eligible: {result.days_until_eligible}")
            print(f"\nDescription: {classifier.get_regime_description(result.regime)}")
            print(f"Expected Alpha: {classifier.get_expected_alpha(result.regime):.2f}%")
        else:
            print("ERROR: No yield data available")
            exit(1)
