"""
Behavioral Sentiment Signal Generator — Portfolio-Lab v2.70 Phase 2
Wraps BehavioralSentimentFetcher with z-score normalization,
regime-gated suppression, and contrarian allocation signals.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentFetcher,
    BehavioralSentimentSnapshot,
)

logger = logging.getLogger(__name__)

# Signal constants
MIN_HOLDING_DAYS = 5
MAX_EQUITY_SHIFT_PCT = 5.0
ZSCORE_WINDOW_DAYS = 90
VIX_CRISIS_THRESHOLD = 35.0
VIX_HIGH_THRESHOLD = 30.0
VIX_ELEVATED_THRESHOLD = 25.0

# Default cache DB for rolling z-score history (only used when no explicit db passed)
_DEFAULT_CACHE_DB = None


def _resolve_cache_db() -> Path:
    """Resolve the default cache DB path lazily to avoid module-level hardcoding."""
    global _DEFAULT_CACHE_DB
    if _DEFAULT_CACHE_DB is None:
        _DEFAULT_CACHE_DB = Path(__file__).resolve().parent.parent.parent / "data" / "market.db"
    return _DEFAULT_CACHE_DB


@dataclass
class BehavioralSignal:
    """Behavioral sentiment signal output"""

    signal_type: str  # contrarian_buy | contrarian_sell | moderate_buy | moderate_sell | neutral
    confidence: float  # 0-1
    equity_shift_pct: float  # recommended allocation change (capped at ±5%)
    holding_period_days: int  # minimum days before next signal
    z_score: float  # normalized composite score
    composite_score: float  # raw composite score (-3 to +3)
    vix: float  # current VIX level
    regime_suppressed: bool  # True if signal suppressed due to regime
    rationale: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class BehavioralSentimentSignal:
    """Generates contrarian behavioral sentiment signals with regime gating"""

    def __init__(self, cache_db: Path = None):
        if cache_db is None:
            cache_db = _resolve_cache_db()
        self.cache_db = cache_db
        self.fetcher = BehavioralSentimentFetcher(cache_db=cache_db)
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_type: Optional[str] = None
        self._signal_count_5d: int = 0
        self._pause_until: Optional[datetime] = None
        self._init_zscore_table()

    def _init_zscore_table(self):
        """Ensure z-score history table exists"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS behavioral_zscore_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        composite_score REAL,
                        signal_type TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init zscore table: {e}")

    def _get_zscore(self, composite_score: float) -> float:
        """Compute z-score of composite_score against 90-day rolling window"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cutoff = (datetime.now() - timedelta(days=ZSCORE_WINDOW_DAYS)).isoformat()
                cursor = conn.execute(
                    """SELECT composite_score FROM behavioral_zscore_history
                       WHERE created_at >= ?""",
                    (cutoff,),
                )
                scores = [row[0] for row in cursor.fetchall()]

                if len(scores) < 10:
                    # Insufficient history — use heuristic z-score
                    return composite_score / 1.5

                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                std = variance ** 0.5

                if std < 0.01:
                    return 0.0

                return (composite_score - mean) / std
        except Exception as e:
            logger.warning(f"Z-score computation failed: {e}")
            return composite_score / 1.5

    def _record_score(self, composite_score: float, signal_type: str):
        """Record a score to the rolling history table"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    """INSERT INTO behavioral_zscore_history
                       (timestamp, composite_score, signal_type)
                       VALUES (?, ?, ?)""",
                    (datetime.now().isoformat(), composite_score, signal_type),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record zscore: {e}")

    def _regime_check(self, vix: float) -> Tuple[bool, str]:
        """Check if current regime should suppress behavioral signals"""
        if vix >= VIX_CRISIS_THRESHOLD:
            return True, f"VIX {vix:.1f} >= {VIX_CRISIS_THRESHOLD}: crisis regime, signal suppressed"

        if vix >= VIX_HIGH_THRESHOLD:
            return True, f"VIX {vix:.1f} >= {VIX_HIGH_THRESHOLD}: high volatility, signal suppressed"

        return False, ""

    def _circuit_breaker_check(self, signal_type: str) -> Tuple[bool, str]:
        """Check circuit breakers: churn control, earnings blackout, pause"""
        now = datetime.now()

        # Pause check
        if self._pause_until and now < self._pause_until:
            remaining = (self._pause_until - now).total_seconds() / 3600
            return True, f"Paused until {self._pause_until.isoformat()[:19]} ({remaining:.1f}h remaining)"

        # Duplicate signal within 5 days
        if (
            self._last_signal_time
            and signal_type == self._last_signal_type
            and signal_type != "neutral"
            and (now - self._last_signal_time) < timedelta(days=5)
        ):
            return True, "Duplicate signal within 5 days — rejecting to prevent churn"

        # Two signals within 5 days (any type, non-neutral)
        if (
            self._last_signal_time
            and self._signal_count_5d >= 2
            and signal_type != "neutral"
        ):
            return True, "Two non-neutral signals within 5 days — churn control"

        return False, ""

    def get_signal(self, snapshot: Optional[BehavioralSentimentSnapshot] = None) -> BehavioralSignal:
        """Generate behavioral sentiment signal with all checks applied"""
        if snapshot is None:
            snapshot = self.fetcher.fetch_snapshot()

        composite = snapshot.composite_score
        vix = snapshot.options.vix
        z_score = self._get_zscore(composite)

        # Determine raw signal type from composite score
        raw_type = snapshot.signal_type  # extreme_fear, fear, neutral, greed, extreme_greed

        # Map to contrarian action
        if raw_type == "extreme_fear":
            signal_type = "contrarian_buy"
            equity_shift = 5.0
        elif raw_type == "fear":
            signal_type = "moderate_buy"
            equity_shift = 3.0
        elif raw_type == "extreme_greed":
            signal_type = "contrarian_sell"
            equity_shift = -5.0
        elif raw_type == "greed":
            signal_type = "moderate_sell"
            equity_shift = -3.0
        else:
            signal_type = "neutral"
            equity_shift = 0.0

        confidence = snapshot.confidence

        # Regime gate: suppress in high vol / crisis
        regime_suppressed, regime_reason = self._regime_check(vix)
        if regime_suppressed:
            signal_type = "neutral"
            equity_shift = 0.0
            confidence *= 0.5

        # VIX elevated: half weight
        if not regime_suppressed and vix >= VIX_ELEVATED_THRESHOLD:
            equity_shift *= 0.5
            confidence *= 0.8

        # Circuit breaker check
        blocked, block_reason = self._circuit_breaker_check(signal_type)
        if blocked:
            signal_type = "neutral"
            equity_shift = 0.0
            confidence *= 0.3

        # Build rationale
        parts = []
        if regime_suppressed:
            parts.append(regime_reason)
        else:
            parts.append(
                f"Composite: {composite:+.2f} (z={z_score:+.2f}), "
                f"VIX: {vix:.1f}, Signal: {raw_type}"
            )
        if blocked:
            parts.append(block_reason)
        if signal_type == "neutral" and not regime_suppressed and not blocked:
            parts.append("No extreme sentiment detected — neutral allocation")

        # Update state for circuit breaker tracking
        if signal_type != "neutral":
            now = datetime.now()
            if self._last_signal_time and (now - self._last_signal_time) < timedelta(days=5):
                self._signal_count_5d += 1
            else:
                self._signal_count_5d = 1
            self._last_signal_time = now
            self._last_signal_type = signal_type

        # Record score for rolling z-score window
        self._record_score(composite, signal_type)

        return BehavioralSignal(
            signal_type=signal_type,
            confidence=round(confidence, 4),
            equity_shift_pct=round(equity_shift, 2),
            holding_period_days=MIN_HOLDING_DAYS,
            z_score=round(z_score, 4),
            composite_score=round(composite, 4),
            vix=round(vix, 2),
            regime_suppressed=regime_suppressed,
            rationale=" | ".join(parts) if parts else "No signal",
            timestamp=snapshot.timestamp,
        )

    def trigger_pause(self, hours: int = 72, reason: str = ""):
        """Manually trigger a circuit breaker pause"""
        self._pause_until = datetime.now() + timedelta(hours=hours)
        logger.info(f"Pause triggered for {hours}h: {reason}")

    def clear_pause(self):
        """Clear an active circuit breaker pause"""
        self._pause_until = None
        logger.info("Pause cleared")

    def get_status(self) -> Dict:
        """Return current signal generator status"""
        return {
            "paused": self._pause_until is not None and datetime.now() < self._pause_until,
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
            "last_signal_time": self._last_signal_time.isoformat() if self._last_signal_time else None,
            "last_signal_type": self._last_signal_type,
            "signal_count_5d": self._signal_count_5d,
        }

    def historical_backfill(self, start_date: str = "2020-01-01", end_date: str = None) -> List[Dict]:
        """Generate synthetic historical sentiment signals for backtesting.

        Uses VIX history from market.db (prices table) as a proxy for sentiment
        extremes. This is a simplified reconstruction — no real SKEW/PCR data
        available pre-2024.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        results = []
        try:
            with sqlite3.connect(self.cache_db) as conn:
                # Try to get VIX data from prices table
                cursor = conn.execute(
                    """SELECT date, close FROM prices
                       WHERE symbol = '^VIX'
                       AND date >= ? AND date <= ?
                       ORDER BY date""",
                    (start_date, end_date),
                )
                rows = cursor.fetchall()

                if not rows:
                    logger.warning("No VIX price data found for historical backfill")
                    return results

                for date_str, vix_close in rows:
                    # Synthesize composite score from VIX
                    # VIX >30 → fear (-1.5 to -3.0), VIX <15 → greed (+1.0 to +2.0)
                    if vix_close >= 35:
                        composite = -2.5
                        signal_type = "extreme_fear"
                    elif vix_close >= 30:
                        composite = -1.5
                        signal_type = "fear"
                    elif vix_close >= 25:
                        composite = -0.5
                        signal_type = "fear"
                    elif vix_close <= 12:
                        composite = 2.0
                        signal_type = "extreme_greed"
                    elif vix_close <= 15:
                        composite = 1.0
                        signal_type = "greed"
                    else:
                        composite = 0.0
                        signal_type = "neutral"

                    z = composite / 1.5
                    equity_shift = 0.0
                    action = "neutral"
                    if signal_type == "extreme_fear":
                        action = "contrarian_buy"
                        equity_shift = 5.0
                    elif signal_type == "fear":
                        action = "moderate_buy"
                        equity_shift = 3.0
                    elif signal_type == "extreme_greed":
                        action = "contrarian_sell"
                        equity_shift = -5.0
                    elif signal_type == "greed":
                        action = "moderate_sell"
                        equity_shift = -3.0

                    # VIX >30 suppresses signal
                    regime_suppressed = vix_close >= 30

                    results.append({
                        "date": date_str,
                        "vix": round(vix_close, 2),
                        "composite_score": round(composite, 2),
                        "z_score": round(z, 4),
                        "signal_type": "neutral" if regime_suppressed else action,
                        "equity_shift_pct": 0.0 if regime_suppressed else equity_shift,
                        "regime_suppressed": regime_suppressed,
                    })
        except Exception as e:
            logger.warning(f"Historical backfill failed: {e}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral Sentiment Signal Generator")
    parser.add_argument("--signal", action="store_true", help="Generate current signal")
    parser.add_argument("--status", action="store_true", help="Show generator status")
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Backfill start date")
    parser.add_argument("--end", type=str, default=None, help="Backfill end date")
    parser.add_argument("--pause", type=int, help="Trigger pause for N hours")
    parser.add_argument("--clear", action="store_true", help="Clear active pause")

    args = parser.parse_args()

    signal_gen = BehavioralSentimentSignal()

    if args.signal or (not args.status and not args.backfill and not args.pause and not args.clear):
        sig = signal_gen.get_signal()
        print("\n=== Behavioral Sentiment Signal ===")
        print(f"Timestamp: {sig.timestamp}")
        print(f"Signal Type: {sig.signal_type}")
        print(f"Confidence: {sig.confidence:.2%}")
        print(f"Equity Shift: {sig.equity_shift_pct:+.1f}%")
        print(f"Z-Score: {sig.z_score:+.2f}")
        print(f"Composite: {sig.composite_score:+.2f}")
        print(f"VIX: {sig.vix:.1f}")
        print(f"Regime Suppressed: {sig.regime_suppressed}")
        print(f"Holding Period: {sig.holding_period_days}d")
        print(f"Rationale: {sig.rationale}")

    if args.status:
        status = signal_gen.get_status()
        print("\n=== Signal Generator Status ===")
        for k, v in status.items():
            print(f"  {k}: {v}")

    if args.backfill:
        results = signal_gen.historical_backfill(args.start, args.end)
        print(f"\n=== Historical Backfill: {len(results)} days ===")
        # Summarize
        buy_days = sum(1 for r in results if "buy" in r["signal_type"])
        sell_days = sum(1 for r in results if "sell" in r["signal_type"])
        neutral_days = sum(1 for r in results if r["signal_type"] == "neutral")
        print(f"  Buy signals: {buy_days}")
        print(f"  Sell signals: {sell_days}")
        print(f"  Neutral: {neutral_days}")
        if results:
            print(f"  Sample (first): {results[0]}")
            print(f"  Sample (last): {results[-1]}")

    if args.pause:
        signal_gen.trigger_pause(args.pause, "Manual CLI trigger")
        print(f"Paused for {args.pause}h")

    if args.clear:
        signal_gen.clear_pause()
        print("Pause cleared")
