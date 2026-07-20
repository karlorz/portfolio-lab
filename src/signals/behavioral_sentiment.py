"""
Behavioral Sentiment Signal Generator — Portfolio-Lab v2.70 Phase 2
Wraps BehavioralSentimentFetcher with z-score normalization,
regime-gated suppression, and contrarian allocation signals.
"""

from src.paths import sqlite_connect
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from src.paths import MARKET_DB
from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentFetcher,
    BehavioralSentimentSnapshot,
)

__all__ = [
    'MIN_HOLDING_DAYS', 'MAX_EQUITY_SHIFT_PCT', 'ZSCORE_WINDOW_DAYS',
    'VIX_CRISIS_THRESHOLD', 'VIX_HIGH_THRESHOLD', 'VIX_ELEVATED_THRESHOLD',
    'BehavioralSignal', 'BehavioralSentimentSignal',
]

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
        _DEFAULT_CACHE_DB = MARKET_DB
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

    def to_signal_snapshot(self):
        """Convert to canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot

        # Map signal_type to directional value
        type_map = {
            "contrarian_buy": 0.5,
            "moderate_buy": 0.3,
            "neutral": 0.0,
            "moderate_sell": -0.3,
            "contrarian_sell": -0.5,
        }
        value = type_map.get(self.signal_type, 0.0)
        is_active = not self.regime_suppressed and self.confidence >= 0.3

        return SignalSnapshot(
            source="behavioral_sentiment",
            timestamp=self.timestamp,
            value=value,
            confidence=self.confidence,
            asset_signals={"SPY": self.equity_shift_pct},
            regime_fit="all",
            is_active=is_active,
            explanation=f"Behavioral: {self.signal_type}, "
                        f"composite={self.composite_score:.2f}, "
                        f"z={self.z_score:.2f}, "
                        f"VIX={self.vix:.1f}, "
                        f"suppressed={self.regime_suppressed}",
            metadata={
                "signal_type": self.signal_type,
                "composite_score": self.composite_score,
                "z_score": self.z_score,
                "vix": self.vix,
                "regime_suppressed": self.regime_suppressed,
            },
        )


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
            with sqlite_connect(self.cache_db) as conn:
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
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to init zscore table: %s", e)

    def _get_zscore(self, composite_score: float) -> float:
        """Compute z-score of composite_score against 90-day rolling window"""
        try:
            with sqlite_connect(self.cache_db) as conn:
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
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning("Z-score computation failed: %s", e)
            return composite_score / 1.5

    def _record_score(self, composite_score: float, signal_type: str):
        """Record a score to the rolling history table"""
        try:
            with sqlite_connect(self.cache_db) as conn:
                conn.execute(
                    """INSERT INTO behavioral_zscore_history
                       (timestamp, composite_score, signal_type)
                       VALUES (?, ?, ?)""",
                    (datetime.now().isoformat(), composite_score, signal_type),
                )
                conn.commit()
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to record zscore: %s", e)

    def _resolve_named_regime(self) -> Optional[str]:
        """Load named market regime from regime_state SSOT (NORMAL/HIGH_VOL/…)."""
        try:
            from src.paths import DATA_DIR
            path = DATA_DIR / "regime_state.json"
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            regime = payload.get("regime") or payload.get("current_regime")
            if regime is None:
                return None
            return str(regime).upper()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to resolve named regime for behavioral gate: %s", e)
            return None

    def _regime_check(
        self,
        vix: float,
        named_regime: Optional[str] = ...,  # type: ignore[assignment]
    ) -> Tuple[bool, str]:
        """Check if current regime should suppress behavioral signals.

        Prefer RegimeGate (named regime SSOT) so signals.behavioral_sentiment.active
        agrees with regime_gate.json. Fall back to VIX thresholds when named
        regime is unavailable.

        Pass ``named_regime=None`` to skip named-regime resolution (VIX-only).
        Omit the arg to resolve from ``regime_state.json``.
        """
        if named_regime is ...:  # type: ignore[comparison-overlap]
            regime = self._resolve_named_regime()
        else:
            regime = named_regime
        if regime:
            try:
                from src.signals.regime_gate import RegimeGate

                gate = RegimeGate()
                if not gate.is_active("behavioral_sentiment", regime):
                    return (
                        True,
                        f"RegimeGate OFF for behavioral_sentiment in {regime}",
                    )
                # Named regime allows signal (e.g. LOW_VOL); still apply VIX crisis/high as extra safety
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                logger.warning("RegimeGate check failed, falling back to VIX: %s", e)

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

    def get_signal_snapshot(self):
        """Return SignalSnapshot for the typed pipeline."""
        from src.signals.signal_snapshot import SignalSnapshot

        signal = self.get_signal()
        if signal is not None:
            return signal.to_signal_snapshot()
        return SignalSnapshot(
            source="behavioral_sentiment",
            timestamp=str(datetime.now()),
            value=0.0,
            confidence=0.0,
            regime_fit="all",
            is_active=False,
            explanation="Behavioral sentiment: no signal data available",
        )

    def trigger_pause(self, hours: int = 72, reason: str = ""):
        """Manually trigger a circuit breaker pause"""
        self._pause_until = datetime.now() + timedelta(hours=hours)
        logger.info("Pause triggered for %dh: %s", hours, reason)

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
            with sqlite_connect(self.cache_db) as conn:
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
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError, RuntimeError) as e:
            logger.warning("Historical backfill failed: %s", e)

        return results


if __name__ == "__main__":
    import argparse
    from src.utils.log_config import configure_logging

    configure_logging()

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
        logger.info("\n=== Behavioral Sentiment Signal ===")
        logger.info("Timestamp: %s", sig.timestamp)
        logger.info("Signal Type: %s", sig.signal_type)
        logger.info("Confidence: %.2f%%", sig.confidence * 100)
        logger.info("Equity Shift: %+.1f%%", sig.equity_shift_pct)
        logger.info("Z-Score: %+.2f", sig.z_score)
        logger.info("Composite: %+.2f", sig.composite_score)
        logger.info("VIX: %.1f", sig.vix)
        logger.info("Regime Suppressed: %s", sig.regime_suppressed)
        logger.info("Holding Period: %dd", sig.holding_period_days)
        logger.info("Rationale: %s", sig.rationale)

    if args.status:
        status = signal_gen.get_status()
        logger.info("\n=== Signal Generator Status ===")
        for k, v in status.items():
            logger.info("  %s: %s", k, v)

    if args.backfill:
        results = signal_gen.historical_backfill(args.start, args.end)
        logger.info("\n=== Historical Backfill: %d days ===", len(results))
        # Summarize
        buy_days = sum(1 for r in results if "buy" in r["signal_type"])
        sell_days = sum(1 for r in results if "sell" in r["signal_type"])
        neutral_days = sum(1 for r in results if r["signal_type"] == "neutral")
        logger.info("  Buy signals: %d", buy_days)
        logger.info("  Sell signals: %d", sell_days)
        logger.info("  Neutral: %d", neutral_days)
        if results:
            logger.info("  Sample (first): %s", results[0])
            logger.info("  Sample (last): %s", results[-1])

    if args.pause:
        signal_gen.trigger_pause(args.pause, "Manual CLI trigger")
        logger.info("Paused for %dh", args.pause)

    if args.clear:
        signal_gen.clear_pause()
        logger.info("Pause cleared")
