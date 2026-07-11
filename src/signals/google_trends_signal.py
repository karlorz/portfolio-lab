"""Google Trends sentiment signal.

Replaces the net-negative behavioral sentiment signal (VIX-proxy, -0.216 Sharpe,
65.8% false positive rate) with search volume data for macro fear indicators.

Data source: data/google_trends.json (cached, populated by fetch script or cron).
Signal construction: Z-score of recent search volume relative to 90-day baseline.
Fear spike (high Z-score) = negative signal. Low fear = positive signal.

Academic basis:
- Da, Engelberg & Gao (2015) "The Sum of All Fears" — search volume predicts returns
- Preis, Moat & Stanley (2013) — Google Trends data predicted 2009 market bottom
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.signals.signal_snapshot import SignalSnapshot

logger = logging.getLogger(__name__)

__all__ = [
    "GoogleTrendsSignal",
    "TREND_TERMS",
    "FEAR_THRESHOLD",
    "GREED_THRESHOLD",
]

# Macro fear indicators to track
TREND_TERMS: List[str] = [
    "recession",
    "inflation",
    "stock market crash",
    "interest rates",
]

# Z-score thresholds for signal generation
FEAR_THRESHOLD = 1.5    # Z-score above this = fear (negative signal)
GREED_THRESHOLD = -0.5  # Z-score below this = complacency (positive signal)

# Minimum data points required
MIN_DATA_POINTS = 14

# Staleness: data older than this many days is rejected
MAX_STALE_DAYS = 14

# Baseline window for Z-score computation
BASELINE_WINDOW = 90


class GoogleTrendsSignal:
    """Read cached Google Trends data and produce a sentiment signal.

    Args:
        data_path: Path to google_trends.json. Defaults to data/google_trends.json.
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = str(Path(__file__).parent.parent.parent / "data" / "google_trends.json")
        self._data_path = data_path
        self._data: Optional[Dict[str, Dict[str, int]]] = None
        self._last_load: Optional[datetime] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load trend data from JSON file."""
        try:
            path = Path(self._data_path)
            if not path.exists():
                self._data = None
                return
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict) or not raw:
                self._data = None
                return
            self._data = raw
            self._last_load = datetime.now()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load Google Trends data: %s", e)
            self._data = None

    def get_signal_snapshot(
        self,
        tickers: Optional[List[str]] = None,
        date: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> SignalSnapshot:
        """Generate a SignalSnapshot from Google Trends data.

        Args:
            tickers: Unused (interface compatibility).
            date: Unused (interface compatibility).
            regime: Unused (interface compatibility).

        Returns:
            SignalSnapshot with Google Trends sentiment value.
        """
        # Reload if file has been updated
        self._load_data()

        if not self._data:
            return self._inactive_snapshot("No Google Trends data available")

        # Check data freshness
        latest_date = self._find_latest_date()
        if latest_date is None:
            return self._inactive_snapshot("No dates in trend data")

        days_old = (datetime.now() - latest_date).days
        if days_old > MAX_STALE_DAYS:
            return self._inactive_snapshot(f"Data is {days_old} days old (max {MAX_STALE_DAYS})")

        # Compute Z-scores for each available term
        z_scores = []
        term_signals = {}
        for term in TREND_TERMS:
            if term not in self._data:
                continue
            z = self._compute_z_score(self._data[term])
            if z is not None:
                z_scores.append(z)
                term_signals[term] = round(z, 3)

        if not z_scores:
            return self._inactive_snapshot("Insufficient data for Z-score computation")

        # Average Z-score across terms
        mean_z = sum(z_scores) / len(z_scores)

        # Convert Z-score to signal value [-1, 1]
        # High Z-score (fear spike) = negative signal (contrarian: fear = buy opportunity)
        # Low Z-score (complacency) = positive signal
        signal_value = float(max(-1.0, min(1.0, -mean_z / 3.0)))

        # Confidence based on data coverage
        data_points = self._count_recent_data_points()
        freshness = max(0, 1.0 - days_old / MAX_STALE_DAYS)
        coverage = min(1.0, data_points / BASELINE_WINDOW)
        confidence = float(min(1.0, freshness * 0.6 + coverage * 0.4))

        # Classify regime
        if mean_z > FEAR_THRESHOLD:
            label = "fear"
        elif mean_z < GREED_THRESHOLD:
            label = "greed"
        else:
            label = "neutral"

        explanation = (
            f"Google Trends ({label}): Z-score={mean_z:.2f}, "
            + ", ".join(f"{k}={v}" for k, v in term_signals.items())
            + f" | signal={signal_value:.3f}, conf={confidence:.3f}"
        )

        return SignalSnapshot(
            source="google_trends",
            timestamp=str(datetime.now()),
            value=signal_value,
            confidence=confidence,
            regime_fit="all",
            is_active=True,
            explanation=explanation,
            metadata={
                "z_scores": term_signals,
                "mean_z": round(mean_z, 3),
                "label": label,
                "data_age_days": days_old,
            },
        )

    def _find_latest_date(self) -> Optional[datetime]:
        """Find the latest date across all trend terms."""
        latest = None
        for term_data in self._data.values():
            if not isinstance(term_data, dict):
                continue
            for date_str in term_data:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    if latest is None or dt > latest:
                        latest = dt
                except ValueError:
                    continue
        return latest

    def _compute_z_score(self, term_data: Dict[str, int]) -> Optional[float]:
        """Compute Z-score of recent search volume vs 90-day baseline.

        Returns None if insufficient data.
        """
        # Sort by date
        try:
            sorted_items = sorted(
                term_data.items(),
                key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"),
            )
        except ValueError:
            return None

        if len(sorted_items) < MIN_DATA_POINTS:
            return None

        values = [v for _, v in sorted_items]

        # Recent: last 7 days
        recent = values[-7:] if len(values) >= 7 else values[-3:]
        recent_mean = sum(recent) / len(recent)

        # Baseline: last 90 days
        baseline = values[-BASELINE_WINDOW:] if len(values) >= BASELINE_WINDOW else values
        baseline_mean = sum(baseline) / len(baseline)
        baseline_std = self._std(baseline)

        if baseline_std < 1e-6:
            return 0.0  # No variation = neutral

        return (recent_mean - baseline_mean) / baseline_std

    def _count_recent_data_points(self) -> int:
        """Count data points in the last 90 days."""
        total = 0
        for term_data in self._data.values():
            if not isinstance(term_data, dict):
                continue
            total += min(len(term_data), BASELINE_WINDOW)
        return total

    @staticmethod
    def _std(values: List[int | float]) -> float:
        """Compute standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)

    @staticmethod
    def _inactive_snapshot(reason: str) -> SignalSnapshot:
        """Create an inactive snapshot."""
        reason_lower = reason.lower()
        if "days old" in reason_lower or "stale" in reason_lower:
            inactive_category = "stale"
        elif "no " in reason_lower:
            inactive_category = "missing"
        elif "insufficient" in reason_lower:
            inactive_category = "insufficient_data"
        else:
            inactive_category = "inactive"

        return SignalSnapshot(
            source="google_trends",
            timestamp=str(datetime.now()),
            value=0.0,
            confidence=0.0,
            regime_fit="all",
            is_active=False,
            explanation=f"Google Trends: {reason}",
            metadata={
                "inactive_reason": reason,
                "inactive_category": inactive_category,
            },
        )
