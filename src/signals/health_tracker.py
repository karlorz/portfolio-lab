#!/usr/bin/env python3
"""
Signal Health Decay Tracking - v3.12 Phase 1
Data infrastructure and health calculator for ensemble voter

Tracks rolling accuracy of signal sources to enable dynamic weight adjustment
when signals show degradation (health < 0.5 triggers weight reduction).

References:
- trending-quant-strategies-2026-mid-may-update: Quality over quantity pivot
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from src.paths import DATA_DIR, MARKET_DB, sqlite_connect
from src.backtest.metrics import save_results_json


from src.signals.signal_source import SignalSource  # canonical, consolidated May 2026
__all__ = [
    'SignalSource',
    'SignalHealthStatus',
    'SignalPrediction',
    'HealthScore',
    'DecayAlert',
    'SignalHealthTracker',
    'backfill_predictions',
    'DEFAULT_RESOLVE_MAX_DAYS',
]

# Max distinct unresolved prediction dates to label per health/cron cycle.
DEFAULT_RESOLVE_MAX_DAYS = 30

# Setup logging
logger = logging.getLogger(__name__)

# Paths
DB_PATH = MARKET_DB
STATE_PATH = DATA_DIR / ".signal_health_state.json"


class SignalHealthStatus(Enum):
    """Health status classification.

    Absolute cutoffs depend on ``weight_scheme`` (see
    ``status_thresholds_for_scheme``). Full multi-window history uses the
    classic 0.7 / 0.5 bands; collapsed 90/60 recency schemes use lower
    healthy bounds so 0/N is not structural when max score ~0.58.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Full independent 90/60/30 windows (Batch classic)
HEALTH_THRESHOLD_HEALTHY_FULL = 0.70
HEALTH_THRESHOLD_DEGRADED_FULL = 0.50
# Collapsed recency (Batch CP / c328): 90d≡60d so score is 0.4*a60+0.6*a30.
# Empirically fleet tops ~0.58; 0.7 is unreachable → permanent 0/N healthy.
HEALTH_THRESHOLD_HEALTHY_COLLAPSED = 0.55
HEALTH_THRESHOLD_DEGRADED_COLLAPSED = 0.48


def status_thresholds_for_scheme(weight_scheme: str | None) -> tuple[float, float]:
    """Return (healthy_min, degraded_min) for a health weight scheme.

    Batch CP: scheme-aware thresholds so window collapse does not force
    structural zero-healthy while ops is green (capture c328).
    """
    scheme = str(weight_scheme or "full_50_30_20")
    if scheme.startswith("collapsed") or "collapsed_recency" in scheme:
        return (
            HEALTH_THRESHOLD_HEALTHY_COLLAPSED,
            HEALTH_THRESHOLD_DEGRADED_COLLAPSED,
        )
    return (HEALTH_THRESHOLD_HEALTHY_FULL, HEALTH_THRESHOLD_DEGRADED_FULL)


def classify_health_status(
    health_score: float,
    *,
    weight_scheme: str | None = None,
) -> str:
    """Map a numeric health score to healthy/degraded/unhealthy."""
    healthy_min, degraded_min = status_thresholds_for_scheme(weight_scheme)
    if health_score >= healthy_min:
        return SignalHealthStatus.HEALTHY.value
    if health_score >= degraded_min:
        return SignalHealthStatus.DEGRADED.value
    return SignalHealthStatus.UNHEALTHY.value

@dataclass
class SignalPrediction:
    """A single signal prediction record."""
    timestamp: str
    source: str
    signal_value: float  # -1.0 to 1.0 (bearish to bullish)
    confidence: float  # 0.0 to 1.0
    predicted_direction: int  # -1 (down), 0 (neutral), 1 (up)
    metadata: Dict[str, Any]  # Source-specific data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "signal_value": self.signal_value,
            "confidence": self.confidence,
            "predicted_direction": self.predicted_direction,
            "metadata": json.dumps(self.metadata),
        }

@dataclass
class HealthScore:
    """Health score for a signal source."""
    source: str
    timestamp: str
    health_score: float  # 0.0 to 1.0
    accuracy_30d: float  # 30-day rolling accuracy
    accuracy_60d: float  # 60-day rolling accuracy
    accuracy_90d: float  # 90-day rolling accuracy
    decay_rate: float  # (acc_30d - acc_60d) / 30; negative = recent worse than mid
    predictions_count: int
    status: str  # healthy/degraded/unhealthy
    ic: Optional[float] = None  # Information Coefficient (Spearman ρ)
    ic_half_life_days: Optional[float] = None  # IC half-life in days (inf = stable)
    # Batch BU: multi-window honesty when 90d collapses onto 60d history
    window_collapse_90_60: bool = False
    weight_scheme: str = "full_50_30_20"  # or collapsed_recency_40_60
    weight_60d: float = 0.3
    weight_30d: float = 0.2
    weight_90d: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DecayAlert:
    """Alert when signal health drops significantly."""
    source: str
    alert_timestamp: str
    previous_health: float
    current_health: float
    drop_30d: float  # Percentage drop over 30 days
    severity: str  # warning/critical
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SignalHealthTracker:
    """
    Tracks signal predictions and calculates health scores.
    
    Key features:
    - Stores predictions in SQLite for historical analysis
    - Calculates 30/60/90-day rolling accuracy
    - Detects decay (health drop >20% in 30 days)
    - Provides health scores for ensemble weight adjustment
    """
    
    DECAY_THRESHOLD = 0.20  # 20% drop triggers alert
    HEALTH_FLOOR = 0.20  # Minimum weight multiplier
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_database()
        self.state = self._load_state()
    
    def _init_database(self):
        """Initialize signal_predictions table."""
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            # Signal predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    signal_value REAL,
                    confidence REAL,
                    predicted_direction INTEGER,
                    metadata TEXT,
                    actual_direction INTEGER,
                    accuracy_calculated INTEGER DEFAULT 0
                )
            """)
        
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_timestamp 
                ON signal_predictions(source, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON signal_predictions(timestamp)
            """)
        
            # Health scores history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_health_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    health_score REAL,
                    accuracy_30d REAL,
                    accuracy_60d REAL,
                    accuracy_90d REAL,
                    decay_rate REAL,
                    predictions_count INTEGER,
                    status TEXT
                )
            """)
        
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_source_timestamp 
                ON signal_health_scores(source, timestamp)
            """)
        
            conn.commit()

        logger.info("Signal health database initialized")
    
    def _load_state(self) -> Dict:
        """Load tracker state from disk."""
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                return json.load(f)
        return {
            "last_health_calculation": None,
            "decay_alerts": [],
            "version": "3.12.0"
        }
    
    def _save_state(self):
        """Save tracker state to disk."""
        self.state["last_health_calculation"] = datetime.now().isoformat()
        save_results_json(self.state, output_path=str(STATE_PATH))
    
    def log_prediction(self, prediction: SignalPrediction):
        """
        Log a new signal prediction.
        
        Call this from each signal source after generating signals.
        """
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            cursor.execute("""
                INSERT INTO signal_predictions 
                (timestamp, source, signal_value, confidence, predicted_direction, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction.timestamp,
                prediction.source,
                prediction.signal_value,
                prediction.confidence,
                prediction.predicted_direction,
                json.dumps(prediction.metadata)
            ))
        
            conn.commit()
    
    # Batch DB: deadband for continuous → discrete direction.
    # Legacy 0.2 was strict `>` and collides with arms clipped to ±0.2
    # (cross_asset_regime_arb): every row became predicted_direction=0 and
    # accuracy collapsed to the 0.5 neutral default — fake health vs true IC.
    DIRECTION_DEADBAND: float = 0.05

    # C1c: per-source deadband overrides.
    # The default 0.05 was tuned for arms clipped to ±0.2 (cross_asset_regime_arb).
    # Sources whose continuous signal is a gradual z-score / sentiment value
    # accumulate weak readings near zero that are noise, not directional calls.
    # Mapping those to ±1 destroys accuracy while IC (continuous) stays strong.
    # The override sets the noise floor per source so only meaningful readings
    # count as directional. Justified by IC-vs-accuracy gap (IC strong-positive,
    # accuracy below 0.5 at the default deadband).
    SOURCE_DEADBANDS: Dict[str, float] = {
        "cross_asset_rv": 0.12,  # gradual -current_z/ZSCORE_ENTRY; weak |z|<0.6 is noise
    }

    @classmethod
    def deadband_for(cls, source: str) -> float:
        """Return the direction deadband for a source (default 0.05)."""
        return float(cls.SOURCE_DEADBANDS.get(source, cls.DIRECTION_DEADBAND))

    @staticmethod
    def direction_from_signal_value(
        signal_value: float,
        deadband: float | None = None,
    ) -> int:
        """Map continuous signal in [-1, 1] to predicted direction {-1, 0, 1}.

        Uses inclusive bounds at ±deadband so clipped extremes (e.g. ±0.2)
        still count as directional. Default deadband 0.05 matches weak but
        non-zero ensemble readings common after soft-floor / conviction scale.
        """
        db = (
            float(SignalHealthTracker.DIRECTION_DEADBAND)
            if deadband is None
            else float(deadband)
        )
        try:
            v = float(signal_value)
        except (TypeError, ValueError):
            return 0
        if v >= db:
            return 1
        if v <= -db:
            return -1
        return 0

    def log_prediction_simple(
        self,
        source: str,
        signal_value: float,
        confidence: float,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Convenience method for logging predictions."""
        predicted = self.direction_from_signal_value(
            signal_value, deadband=self.deadband_for(source)
        )

        prediction = SignalPrediction(
            timestamp=timestamp or datetime.now().isoformat(),
            source=source,
            signal_value=signal_value,
            confidence=confidence,
            predicted_direction=predicted,
            metadata=metadata or {}
        )
        
        self.log_prediction(prediction)

    # Batch DG: min labeled polarity-stamped rows before trusting post-fix IC
    # (time-series cohort; cross-section research often wants 80+ — we gate ops at 10)
    POST_FIX_MIN_LABELED: int = 10
    # Forward label lag: SPY close-to-close next session after prediction date
    LABEL_LAG_SESSIONS: int = 1

    def post_fix_cohort_readiness(
        self,
        source: str,
        *,
        n_polarity_stamped: int,
        n_polarity_labeled: int,
        ic_polarity_cohort: Optional[float] = None,
        min_labeled: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch DG: min-sample + label-lag readiness for post-fix shadow IC.

        Does not force-wake or auto-invert. Thin cohorts stay ``not_ready``.
        """
        min_n = int(
            self.POST_FIX_MIN_LABELED if min_labeled is None else min_labeled
        )
        stamped = int(n_polarity_stamped or 0)
        labeled = int(n_polarity_labeled or 0)
        pending = max(0, stamped - labeled)
        deficit = max(0, min_n - labeled)
        ready = labeled >= min_n
        lag_note = (
            f"SPY forward label lag ≈ {self.LABEL_LAG_SESSIONS} session(s); "
            "same-day polarity stamps stay unlabeled until next close."
        )
        if ready:
            status = "cohort_ready_for_shadow_ic"
            hint = (
                f"Post-fix polarity cohort has ≥{min_n} labeled rows "
                f"(n_labeled={labeled}); report ic_polarity_cohort for ops — "
                "still no auto-invert; multi-horizon health reentry separate."
            )
        elif stamped == 0:
            status = "awaiting_provenance_stamps"
            hint = (
                "No polarity_policy metadata yet — ensure Batch DF vote logging "
                "is on the live path; shadow IC unavailable."
            )
        elif labeled == 0:
            status = "awaiting_label_lag"
            hint = (
                f"{stamped} polarity-stamped row(s), 0 labeled — {lag_note} "
                f"Need {min_n} labeled for min-sample IC gate."
            )
        else:
            status = "cohort_building"
            hint = (
                f"Polarity cohort building: labeled={labeled}/{min_n} "
                f"(stamped={stamped}, pending_labels={pending}, deficit={deficit}). "
                f"{lag_note}"
            )
        return {
            "source": source,
            "status": status,
            "ready": bool(ready),
            "min_labeled": min_n,
            "n_polarity_stamped": stamped,
            "n_polarity_labeled": labeled,
            "n_pending_labels": pending,
            "labeled_deficit": deficit,
            "label_lag_sessions": int(self.LABEL_LAG_SESSIONS),
            "ic_polarity_cohort": (
                None if ic_polarity_cohort is None else round(float(ic_polarity_cohort), 4)
            ),
            "auto_invert_policy": "disabled",
            "force_wake_policy": "disabled",
            "readiness_hint": hint,
        }

    def post_fix_provenance_readiness(
        self,
        source: str,
        *,
        n_provenance_stamped: int,
        n_provenance_labeled: int,
        min_labeled: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch DI: general provenance_batch cohort (alt_data etc., not polarity).

        Same min-sample + label-lag gates as polarity cohort; no auto-invert.
        """
        min_n = int(
            self.POST_FIX_MIN_LABELED if min_labeled is None else min_labeled
        )
        stamped = int(n_provenance_stamped or 0)
        labeled = int(n_provenance_labeled or 0)
        pending = max(0, stamped - labeled)
        deficit = max(0, min_n - labeled)
        ready = labeled >= min_n
        lag_note = (
            f"SPY forward label lag ≈ {self.LABEL_LAG_SESSIONS} session(s)."
        )
        if ready:
            status = "cohort_ready_for_shadow_ic"
            hint = (
                f"Provenance cohort ≥{min_n} labeled (n={labeled}) — "
                "shadow IC reportable; multi-horizon reentry still separate."
            )
        elif stamped == 0:
            status = "awaiting_provenance_stamps"
            hint = (
                "No provenance_batch metadata yet — Batch DF vote logging required."
            )
        elif labeled == 0:
            status = "awaiting_label_lag"
            hint = (
                f"{stamped} provenance-stamped row(s), 0 labeled — {lag_note} "
                f"Need {min_n} labeled for min-sample gate."
            )
        else:
            status = "cohort_building"
            hint = (
                f"Provenance cohort building: labeled={labeled}/{min_n} "
                f"(stamped={stamped}, pending={pending}, deficit={deficit}). "
                f"{lag_note}"
            )
        return {
            "source": source,
            "status": status,
            "ready": bool(ready),
            "min_labeled": min_n,
            "n_provenance_stamped": stamped,
            "n_provenance_labeled": labeled,
            "n_pending_labels": pending,
            "labeled_deficit": deficit,
            "label_lag_sessions": int(self.LABEL_LAG_SESSIONS),
            "auto_invert_policy": "disabled",
            "force_wake_policy": "disabled",
            "readiness_hint": hint,
            "cohort_kind": "provenance_batch",
        }

    def count_provenance_rows(
        self,
        source: str,
        *,
        lookback_days: int = 90,
        metadata_substring: str = "provenance_batch",
    ) -> Dict[str, Any]:
        """Batch DF/DG: provenance counts + post-fix cohort readiness.

        Used for post-fix shadow IC cohorts (e.g. polarity_policy stamped
        after Batch DC). No auto-invert / force-wake.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                """,
                (source, start_date, end_date),
            )
            n_all = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(*) FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND metadata IS NOT NULL
                  AND metadata != ''
                  AND metadata != '{}'
                  AND metadata LIKE ?
                """,
                (source, start_date, end_date, f"%{metadata_substring}%"),
            )
            n_prov = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(*) FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND metadata IS NOT NULL
                  AND metadata LIKE ?
                  AND actual_direction IS NOT NULL
                """,
                (source, start_date, end_date, f"%{metadata_substring}%"),
            )
            n_prov_labeled = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(*) FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND metadata LIKE '%polarity_policy%'
                """,
                (source, start_date, end_date),
            )
            n_polarity = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(*) FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND metadata LIKE '%polarity_policy%'
                  AND actual_direction IS NOT NULL
                """,
                (source, start_date, end_date),
            )
            n_polarity_labeled = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT signal_value, actual_direction FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND metadata LIKE '%polarity_policy%'
                  AND actual_direction IS NOT NULL
                  AND signal_value IS NOT NULL
                """,
                (source, start_date, end_date),
            )
            pairs = cursor.fetchall()
        ic_post = None
        min_n = int(self.POST_FIX_MIN_LABELED)
        if len(pairs) >= min_n:
            try:
                signals = [r[0] for r in pairs]
                actuals = [r[1] for r in pairs]
                ic_post = self._spearman_rank_correlation(signals, actuals)
            except Exception:  # noqa: BLE001
                ic_post = None
        readiness = self.post_fix_cohort_readiness(
            source,
            n_polarity_stamped=n_polarity,
            n_polarity_labeled=n_polarity_labeled,
            ic_polarity_cohort=None if ic_post is None else float(ic_post),
        )
        # Batch DI: general provenance cohort when polarity not applicable (e.g. alt_data)
        prov_readiness = self.post_fix_provenance_readiness(
            source,
            n_provenance_stamped=n_prov,
            n_provenance_labeled=n_prov_labeled,
        )
        # Prefer polarity readiness when any polarity stamps exist; else provenance
        primary = readiness if n_polarity > 0 else prov_readiness
        return {
            "source": source,
            "window_days": lookback_days,
            "n_rows": n_all,
            "n_with_provenance": n_prov,
            "n_provenance_labeled": n_prov_labeled,
            "n_polarity_stamped": n_polarity,
            "n_polarity_labeled": n_polarity_labeled,
            "ic_polarity_cohort": None if ic_post is None else round(float(ic_post), 4),
            "provenance_coverage": (
                round(n_prov / n_all, 4) if n_all else None
            ),
            "cohort_readiness": primary,
            "polarity_cohort_readiness": readiness,
            "provenance_cohort_readiness": prov_readiness,
            "policy": "shadow_ic_post_fix_no_auto_invert",
        }

    def repair_neutral_predicted_directions(
        self,
        source: Optional[str] = None,
        deadband: float | None = None,
    ) -> int:
        """Batch DB: reclassify rows with |signal| ≥ deadband but direction 0.

        Idempotent UPDATE — does not touch already-directional rows.
        Returns number of rows repaired.
        """
        db = (
            float(self.DIRECTION_DEADBAND)
            if deadband is None
            else float(deadband)
        )
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
            if source:
                cursor.execute(
                    """
                    UPDATE signal_predictions
                    SET predicted_direction = CASE
                        WHEN signal_value >= ? THEN 1
                        WHEN signal_value <= ? THEN -1
                        ELSE predicted_direction
                    END
                    WHERE source = ?
                      AND predicted_direction = 0
                      AND signal_value IS NOT NULL
                      AND (signal_value >= ? OR signal_value <= ?)
                    """,
                    (db, -db, source, db, -db),
                )
            else:
                cursor.execute(
                    """
                    UPDATE signal_predictions
                    SET predicted_direction = CASE
                        WHEN signal_value >= ? THEN 1
                        WHEN signal_value <= ? THEN -1
                        ELSE predicted_direction
                    END
                    WHERE predicted_direction = 0
                      AND signal_value IS NOT NULL
                      AND (signal_value >= ? OR signal_value <= ?)
                    """,
                    (db, -db, db, -db),
                )
            n = int(cursor.rowcount or 0)
            conn.commit()
        if n:
            logger.info(
                "Batch DB: repaired %d neutral predicted_direction rows "
                "(deadband=%.3f, source=%s)",
                n,
                db,
                source or "ALL",
            )
        return n
    
    def update_actual_directions(self, returns_data: Dict[str, float], date: str):
        """
        Update predictions with actual market direction.
        
        Args:
            returns_data: Dict mapping symbol to daily return (e.g., {'SPY': 0.012})
            date: Date string (YYYY-MM-DD) to update
        """
        # Use SPY as reference for market direction
        spy_return = returns_data.get('SPY', 0)
        actual_direction = 1 if spy_return > 0 else (-1 if spy_return < 0 else 0)
        
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            # Update all predictions for this date with actual direction
            cursor.execute("""
                UPDATE signal_predictions 
                SET actual_direction = ?, accuracy_calculated = 1
                WHERE date(timestamp) = date(?) AND actual_direction IS NULL
            """, (actual_direction, date))
        
            updated = cursor.rowcount
            conn.commit()
        
        logger.info("Updated %s predictions with actual direction for %s", updated, date)
        return updated

    def list_unresolved_prediction_dates(
        self,
        limit: int = 30,
        *,
        oldest_first: bool = False,
    ) -> List[str]:
        """Return distinct prediction calendar dates still missing actual_direction.

        Default is newest-first (IC freshness). Pass ``oldest_first=True`` for
        catch-up drain of the backlog tail. Bounded so health/cron cycles never
        scan the full multi-hundred-k pending backlog in one shot.
        """
        limit = max(1, int(limit))
        order = "ASC" if oldest_first else "DESC"
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                f"""
                SELECT DISTINCT date(timestamp) AS d
                FROM signal_predictions
                WHERE actual_direction IS NULL
                  AND timestamp IS NOT NULL
                  AND date(timestamp) IS NOT NULL
                ORDER BY d {order}
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]

    def _spy_forward_return(self, prediction_date: str) -> Optional[float]:
        """SPY close-to-close forward return for a prediction calendar date.

        Predictions often land on weekends/holidays (no SPY bar that day).
        Anchor = last SPY session with ``date <= prediction_date``; label with
        the next session's return (anchor → next). Returns None when the next
        bar does not yet exist (e.g. same-day / last-bar predictions).
        """
        try:
            with sqlite_connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Last available SPY close on or before the prediction date
                row0 = cursor.execute(
                    """
                    SELECT date(date) AS d, close FROM prices
                    WHERE symbol = 'SPY' AND date(date) <= date(?)
                    ORDER BY date(date) DESC
                    LIMIT 1
                    """,
                    (prediction_date,),
                ).fetchone()
                if not row0:
                    return None
                anchor_date = row0[0]
                p0 = float(row0[1])
                row1 = cursor.execute(
                    """
                    SELECT close FROM prices
                    WHERE symbol = 'SPY' AND date(date) > date(?)
                    ORDER BY date(date) ASC
                    LIMIT 1
                    """,
                    (anchor_date,),
                ).fetchone()
            if not row1:
                return None
            p1 = float(row1[0])
            if p0 <= 0:
                return None
            return (p1 / p0) - 1.0
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            logger.debug("SPY forward return unavailable for %s: %s", prediction_date, exc)
            return None

    def resolve_pending_labels(
        self,
        max_days: int = 30,
        *,
        oldest_first: bool = False,
    ) -> Dict[str, Any]:
        """Production label resolver: apply SPY forward returns to pending predictions.

        Bounded by ``max_days`` distinct unresolved dates so a cold start against
        a large backlog remains cheap per cycle.

        - Health path: ``oldest_first=False`` (newest first) keeps IC window fresh.
        - Catch-up path: ``oldest_first=True`` drains May→… backlog fairness.

        Returns summary: {dates_considered, dates_resolved, predictions_updated, skipped}.
        """
        max_days = max(1, int(max_days))
        dates = self.list_unresolved_prediction_dates(
            limit=max_days, oldest_first=oldest_first
        )
        predictions_updated = 0
        dates_resolved = 0
        skipped: List[str] = []

        for d in dates:
            fwd = self._spy_forward_return(d)
            if fwd is None:
                skipped.append(d)
                continue
            n = self.update_actual_directions({"SPY": fwd}, d)
            predictions_updated += int(n or 0)
            if n:
                dates_resolved += 1

        # Batch DI: if newest-first resolved nothing (label lag on calendar tail),
        # dual-pass drain oldest unresolved dates so backlog mid-window still moves.
        dual_pass = False
        dual_summary: Dict[str, Any] = {}
        if (
            not oldest_first
            and dates_resolved == 0
            and len(skipped) == len(dates)
            and len(dates) > 0
        ):
            dual_pass = True
            dual_summary = self.resolve_pending_labels(
                max_days=max_days, oldest_first=True
            )
            # avoid infinite recursion: dual call uses oldest_first=True
            predictions_updated += int(dual_summary.get("predictions_updated") or 0)
            dates_resolved += int(dual_summary.get("dates_resolved") or 0)
            skipped.extend(dual_summary.get("skipped_no_spy_return") or [])

        summary = {
            "dates_considered": len(dates) + (
                int(dual_summary.get("dates_considered") or 0) if dual_pass else 0
            ),
            "dates_resolved": dates_resolved,
            "predictions_updated": predictions_updated,
            "skipped_no_spy_return": skipped,
            "max_days": max_days,
            "oldest_first": bool(oldest_first),
            "dual_pass_oldest": dual_pass,
            "label_lag_note": (
                "SPY next-session required; newest dates often skip until market close"
            ),
        }
        logger.info(
            "resolve_pending_labels: considered=%s resolved_dates=%s predictions=%s skipped=%s oldest_first=%s dual=%s",
            summary["dates_considered"],
            summary["dates_resolved"],
            summary["predictions_updated"],
            len(skipped),
            oldest_first,
            dual_pass,
        )
        return summary
    
    @staticmethod
    def resolve_health_window_weights(
        counts: Dict[str, int],
        accuracies: Dict[str, float],
    ) -> Dict[str, Any]:
        """Choose multi-window weights with collapse honesty (Batch BU).

        Full scheme (independent 90d history): 50% 90d + 30% 60d + 20% 30d.

        When the 90d window collapses onto 60d (same labeled row count, or
        identical accuracy because there is no extra history), double-counting
        90d+60d masks recent decay. MLOps multi-window practice: compare recent
        vs long baselines without pretending a longer window exists; apply
        recency bias (60% 30d + 40% 60d).

        Returns weights + collapse flag + scheme name.
        """
        c90 = int(counts.get("90d") or 0)
        c60 = int(counts.get("60d") or 0)
        # Collapsed only when 90d has no extra labeled rows beyond 60d.
        # Do not use accuracy equality alone — full history can share the same
        # hit rate while still being a longer window (would false-collapse).
        collapsed = c90 > 0 and c60 > 0 and c90 == c60
        if collapsed:
            return {
                "window_collapse_90_60": True,
                "weight_scheme": "collapsed_recency_40_60",
                "weight_90d": 0.0,
                "weight_60d": 0.4,
                "weight_30d": 0.6,
            }
        return {
            "window_collapse_90_60": False,
            "weight_scheme": "full_50_30_20",
            "weight_90d": 0.5,
            "weight_60d": 0.3,
            "weight_30d": 0.2,
        }

    def calculate_health_score(
        self, 
        source: str,
        end_date: Optional[str] = None
    ) -> Optional[HealthScore]:
        """
        Calculate health score for a signal source.
        
        Health score formula (full independent history):
        - 50% weight on 90-day accuracy
        - 30% weight on 60-day accuracy  
        - 20% weight on 30-day accuracy

        Batch BU: when 90d window collapses onto 60d (same labeled count or
        identical accuracy), use recency-biased 40% 60d + 60% 30d so recent
        decay is not masked by double-counting the same mid window as "long".
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            # Get predictions with actual directions
            periods = {
                '30d': (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"),
                '60d': (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d"),
                '90d': (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d"),
            }
        
            accuracies = {}
            counts = {}

            # C1c: when a source has a deadband override, recompute
            # predicted_direction from signal_value so historical rows logged
            # before the override are scored consistently with new predictions.
            # Sources using the default deadband trust the stored column
            # (callers/tests may set predicted_direction independently).
            src_deadband = self.deadband_for(source)
            recompute_direction = src_deadband != self.DIRECTION_DEADBAND

            for period, start_date in periods.items():
                cursor.execute("""
                    SELECT signal_value, predicted_direction, actual_direction
                    FROM (
                        SELECT
                            signal_value,
                            predicted_direction,
                            actual_direction,
                            ROW_NUMBER() OVER (
                                PARTITION BY date(timestamp)
                                ORDER BY timestamp DESC, id DESC
                            ) AS daily_rank
                        FROM signal_predictions
                        WHERE source = ?
                          AND date(timestamp) BETWEEN date(?) AND date(?)
                          AND actual_direction IS NOT NULL
                    )
                    WHERE daily_rank = 1
                """, (source, start_date, end_date))

                rows = cursor.fetchall()

                if not rows:
                    accuracies[period] = 0.5  # Neutral if no data
                    counts[period] = 0
                    continue

                # Recompute direction with the per-source deadband when the
                # source has an override; otherwise trust the stored column.
                scored = []
                for sv, pred, actual in rows:
                    if recompute_direction and sv is not None:
                        pred = self.direction_from_signal_value(sv, deadband=src_deadband)
                    scored.append((pred, actual))

                # Calculate directional accuracy
                correct = sum(1 for pred, actual in scored if pred == actual and pred != 0)
                total = sum(1 for pred, actual in scored if pred != 0)  # Exclude neutral predictions
            
                if total > 0:
                    accuracies[period] = correct / total
                else:
                    accuracies[period] = 0.5
            
                counts[period] = len(rows)
        
        
        # Weighted health score
        if counts['90d'] < 10:  # Need minimum data
            ninety_day_count = counts['90d']
            logger.warning("Insufficient data for %s: only %d predictions", source, ninety_day_count)
            return None
        
        weights = self.resolve_health_window_weights(counts, accuracies)
        health = (
            accuracies['90d'] * float(weights["weight_90d"])
            + accuracies['60d'] * float(weights["weight_60d"])
            + accuracies['30d'] * float(weights["weight_30d"])
        )
        
        # Calculate decay rate (change per day over 30 days)
        # positive → recent better than mid; negative → recent worse (decay)
        decay_rate = (accuracies['30d'] - accuracies['60d']) / 30 if counts['60d'] > 0 else 0
        
        # Batch CP: scheme-aware status cutoffs (collapsed cannot hit 0.7)
        scheme = str(weights["weight_scheme"])
        status = classify_health_status(health, weight_scheme=scheme)
        
        return HealthScore(
            source=source,
            timestamp=end_date,
            health_score=round(health, 4),
            accuracy_30d=round(accuracies['30d'], 4),
            accuracy_60d=round(accuracies['60d'], 4),
            accuracy_90d=round(accuracies['90d'], 4),
            decay_rate=round(decay_rate, 6),
            predictions_count=counts['90d'],
            status=status,
            ic=self.compute_ic(source, end_date=end_date),
            ic_half_life_days=self.compute_ic_half_life(source, end_date=end_date),
            window_collapse_90_60=bool(weights["window_collapse_90_60"]),
            weight_scheme=scheme,
            weight_60d=float(weights["weight_60d"]),
            weight_30d=float(weights["weight_30d"]),
            weight_90d=float(weights["weight_90d"]),
        )
    
    def calculate_all_health_scores(
        self, 
        end_date: Optional[str] = None
    ) -> Dict[str, HealthScore]:
        """Calculate health scores for all signal sources."""
        scores = {}
        
        for source in SignalSource:
            score = self.calculate_health_score(source.value, end_date)
            if score:
                scores[source.value] = score
        
        return scores
    
    def save_health_scores(self, scores: Dict[str, HealthScore]):
        """Save health scores to database."""
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            for score in scores.values():
                cursor.execute("""
                    INSERT INTO signal_health_scores
                    (timestamp, source, health_score, accuracy_30d, accuracy_60d, 
                     accuracy_90d, decay_rate, predictions_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    score.timestamp,
                    score.source,
                    score.health_score,
                    score.accuracy_30d,
                    score.accuracy_60d,
                    score.accuracy_90d,
                    score.decay_rate,
                    score.predictions_count,
                    score.status
                ))
        
            conn.commit()
        self._save_state()
    
    def detect_decay_alerts(
        self,
        lookback_days: int = 30
    ) -> List[DecayAlert]:
        """
        Detect signals with significant health degradation.
        
        Returns alerts for signals where health dropped >20% over lookback period.
        """
        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
        
            alerts = []
        
            for source in SignalSource:
                # Get health score history for this source
                cursor.execute("""
                    SELECT timestamp, health_score
                    FROM signal_health_scores
                    WHERE source = ?
                      AND date(timestamp) >= date(
                          (SELECT MAX(timestamp) FROM signal_health_scores WHERE source = ?),
                          ?
                      )
                    ORDER BY timestamp ASC
                """, (source.value, source.value, f"-{lookback_days} days"))
            
                rows = cursor.fetchall()
            
                if len(rows) < 2:
                    continue
            
                # Calculate drop
                previous_health = rows[0][1]  # First record in period
                current_health = rows[-1][1]  # Most recent
                drop = (previous_health - current_health) / previous_health if previous_health > 0 else 0
            
                if drop >= self.DECAY_THRESHOLD:
                    severity = "critical" if drop >= 0.30 else "warning"
                
                    alert = DecayAlert(
                        source=source.value,
                        alert_timestamp=datetime.now().isoformat(),
                        previous_health=previous_health,
                        current_health=current_health,
                        drop_30d=round(drop, 4),
                        severity=severity,
                        message=f"{source.value}: Health dropped {drop:.1%} in {lookback_days}d "
                                f"({previous_health:.2f} -> {current_health:.2f})"
                    )
                
                    alerts.append(alert)

                    # Save to state (dedup: skip if same source + same health already recorded)
                    if "decay_alerts" not in self.state:
                        self.state["decay_alerts"] = []
                    existing = self.state["decay_alerts"]
                    is_duplicate = any(
                        a.get("source") == source.value
                        and abs(a.get("current_health", 0) - current_health) < 0.001
                        and abs(a.get("previous_health", 0) - previous_health) < 0.001
                        for a in existing[-20:]  # check last 20 for this source
                    )
                    if not is_duplicate:
                        self.state["decay_alerts"].append(alert.to_dict())
                    # Keep only last 100 alerts
                    self.state["decay_alerts"] = self.state["decay_alerts"][-100:]
        
        self._save_state()

        return alerts

    def detect_ic_alerts(
        self,
        lookback_days: int = 90,
        streak_threshold: int = 3,
        ic_ratio_floor: float = 0.3,
        ic_drawdown_threshold: float = 0.5,
    ) -> List[DecayAlert]:
        """Detect IC-based signal degradation alerts.

        Three alert types:
        1. Negative IC streak: signal has N consecutive negative IC windows
        2. Low IC ratio: |IC|/|IC_peak| below floor (signal losing predictive power)
        3. IC drawdown: IC dropped >threshold% from its peak

        Args:
            lookback_days: How far back to look for IC history.
            streak_threshold: Consecutive negative IC windows for streak alert.
            ic_ratio_floor: Minimum IC/|IC_peak| ratio (below = alert).
            ic_drawdown_threshold: Fraction of peak IC loss to trigger alert.

        Returns:
            List of DecayAlert instances for IC degradation.
        """
        alerts = []

        for source in SignalSource:
            # Compute current IC and recent IC history
            try:
                current_ic = self.compute_ic(source, lookback_days=lookback_days)
            except (ValueError, TypeError, RuntimeError, sqlite3.Error):
                continue

            if current_ic is None:
                continue

            # Get IC history from rolling windows

            # Compute IC at multiple lookback points to build history
            ic_history = []
            window_days = 30
            n_windows = min(lookback_days // window_days, 6)

            for i in range(n_windows):
                try:
                    window_end = (datetime.now() - timedelta(days=i * window_days)).strftime("%Y-%m-%d")
                    ic_val = self.compute_ic(source, lookback_days=window_days, end_date=window_end)
                    if ic_val is not None:
                        ic_history.append(ic_val)
                except (ValueError, TypeError, RuntimeError, sqlite3.Error):
                    continue

            if not ic_history:
                continue

            # Alert 1: Negative IC streak
            consecutive_neg = 0
            for ic_val in ic_history:
                if ic_val < 0:
                    consecutive_neg += 1
                else:
                    break  # Streak must be from most recent window

            if consecutive_neg >= streak_threshold:
                alerts.append(DecayAlert(
                    source=source.value,
                    alert_timestamp=datetime.now().isoformat(),
                    previous_health=0,  # Not health-based
                    current_health=0,
                    drop_30d=0,
                    severity="warning",
                    message=f"{source.value}: Negative IC streak ({consecutive_neg} consecutive "
                            f"windows, current IC={current_ic:.4f})",
                ))

            # Alert 2: Low IC ratio
            peak_ic = max(abs(ic) for ic in ic_history) if ic_history else 0
            if peak_ic > 0:
                ic_ratio = abs(current_ic) / peak_ic
                if ic_ratio < ic_ratio_floor and peak_ic > 0.02:
                    alerts.append(DecayAlert(
                        source=source.value,
                        alert_timestamp=datetime.now().isoformat(),
                        previous_health=peak_ic,
                        current_health=abs(current_ic),
                        drop_30d=round(1 - ic_ratio, 4),
                        severity="warning",
                        message=f"{source.value}: IC ratio {ic_ratio:.2f} below floor "
                                f"({ic_ratio_floor}), current IC={current_ic:.4f} vs peak={peak_ic:.4f}",
                    ))

            # Alert 3: IC drawdown from peak
            if peak_ic > 0.02:
                ic_drawdown = (peak_ic - current_ic) / peak_ic if peak_ic > 0 else 0
                if ic_drawdown > ic_drawdown_threshold:
                    severity = "critical" if ic_drawdown > 0.75 else "warning"
                    alerts.append(DecayAlert(
                        source=source.value,
                        alert_timestamp=datetime.now().isoformat(),
                        previous_health=peak_ic,
                        current_health=current_ic,
                        drop_30d=round(ic_drawdown, 4),
                        severity=severity,
                        message=f"{source.value}: IC drawdown {ic_drawdown:.1%} from peak "
                                f"({peak_ic:.4f} -> {current_ic:.4f})",
                    ))

        # Save IC alerts to state
        if alerts:
            if "ic_alerts" not in self.state:
                self.state["ic_alerts"] = []
            for alert in alerts:
                self.state["ic_alerts"].append(alert.to_dict())
            self.state["ic_alerts"] = self.state["ic_alerts"][-100:]
            self._save_state()

        return alerts
    
    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        min_weight_multiplier: float = 0.2,
        ic_bonus_threshold: float = 0.05,
        ic_penalty_threshold: float = 0.02,
        ic_weight_factor: float = 0.15,
        *,
        hard_zero_unhealthy: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate health-adjusted weights for ensemble voting.

        Formula:
          adjusted_weight = base_weight * health_multiplier * ic_multiplier

        health_multiplier:
          - status == unhealthy → 0.0 (hard quality gate; Batch BH)
          - else max(min_multiplier, health_score) soft floor for degraded/healthy
        ic_multiplier:
          - |IC| > ic_bonus_threshold: 1.0 + ic_weight_factor * |IC|
          - |IC| < ic_penalty_threshold: 1.0 - ic_weight_factor * (1 - |IC|/ic_penalty_threshold)
          - otherwise: 1.0 (neutral)

        When every arm is hard-zeroed, returns all zeros (do not reinflate toxic
        mass). Callers should freeze adaptive blend / fall back to champion.

        Args:
            base_weights: Dict mapping source to base weight (should sum to 1.0)
            min_weight_multiplier: Floor for weight adjustment (default 0.2)
            ic_bonus_threshold: IC above this gets a weight boost (default 0.05)
            ic_penalty_threshold: IC below this gets a weight penalty (default 0.02)
            ic_weight_factor: Magnitude of IC adjustment (default 0.15)
            hard_zero_unhealthy: When True, status=unhealthy forces multiplier 0

        Returns:
            Dict of adjusted weights (normalized to sum to 1.0, or all 0 if gated)
        """
        scores = self.calculate_all_health_scores()

        adjusted = {}
        slept: list[str] = []
        for source, base_weight in base_weights.items():
            score = scores.get(source)
            if score:
                status = str(getattr(score, "status", "") or "").lower()
                # Hard quality gate: unhealthy arms get zero mass (not soft floor).
                if hard_zero_unhealthy and status == SignalHealthStatus.UNHEALTHY.value:
                    health_mult = 0.0
                    slept.append(source)
                else:
                    health_mult = max(min_weight_multiplier, score.health_score)

                # IC multiplier
                ic_mult = 1.0
                if score.ic is not None and health_mult > 0:
                    abs_ic = abs(score.ic)
                    if abs_ic > ic_bonus_threshold:
                        ic_mult = 1.0 + ic_weight_factor * abs_ic
                    elif abs_ic < ic_penalty_threshold:
                        ic_mult = 1.0 - ic_weight_factor * (1 - abs_ic / ic_penalty_threshold)

                adjusted[source] = base_weight * health_mult * ic_mult
            else:
                # No health data - use neutral health (0.5)
                adjusted[source] = base_weight * 0.5

        if slept:
            logger.info(
                "Health gate slept %d unhealthy arm(s): %s",
                len(slept),
                ", ".join(slept),
            )

        # Normalize to sum to 1.0 — never reinflate when all hard-gated
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        else:
            adjusted = {k: 0.0 for k in base_weights}

        return adjusted

    # ------------------------------------------------------------------
    # Information Coefficient (IC) methods
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman_rank_correlation(x: List[float], y: List[float]) -> Optional[float]:
        """
        Compute Spearman rank correlation without scipy dependency.

        Spearman ρ = Pearson correlation of ranks.
        Returns None if fewer than 3 paired observations.
        """
        n = len(x)
        if n < 3 or len(y) != n:
            return None

        def _rank(vals: List[float]) -> List[float]:
            indexed = sorted(enumerate(vals), key=lambda t: t[1])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2  # Average of 1-based positions
                for k in range(i, j + 1):
                    ranks[indexed[k][0]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)

        mean_rx = sum(rx) / n
        mean_ry = sum(ry) / n

        cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
        std_rx = (sum((r - mean_rx) ** 2 for r in rx)) ** 0.5
        std_ry = (sum((r - mean_ry) ** 2 for r in ry)) ** 0.5

        if std_rx == 0 or std_ry == 0:
            return 0.0

        return cov / (std_rx * std_ry)

    def compute_ic(
        self, source: str, lookback_days: int = 90, end_date: Optional[str] = None
    ) -> Optional[float]:
        """
        Compute Information Coefficient (Spearman rank correlation) between
        signal values and actual forward returns for a given source.

        IC measures predictive power: |IC| > 0.05 is meaningful for most
        cross-sectional strategies; |IC| > 0.10 is strong.

        Returns None if insufficient data.
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT signal_value, actual_direction
                FROM (
                    SELECT
                        signal_value,
                        actual_direction,
                        timestamp,
                        ROW_NUMBER() OVER (
                            PARTITION BY date(timestamp)
                            ORDER BY timestamp DESC, id DESC
                        ) AS daily_rank
                    FROM signal_predictions
                    WHERE source = ?
                      AND date(timestamp) BETWEEN date(?) AND date(?)
                      AND actual_direction IS NOT NULL
                      AND signal_value IS NOT NULL
                )
                WHERE daily_rank = 1
                ORDER BY timestamp
                """,
                (source, start_date, end_date),
            )
            rows = cursor.fetchall()

        if len(rows) < 3:
            logger.debug("Insufficient data for IC: source=%s, rows=%d", source, len(rows))
            return None

        signals = [r[0] for r in rows]
        actuals = [r[1] for r in rows]
        return self._spearman_rank_correlation(signals, actuals)

    def compute_ic_half_life(
        self, source: str, end_date: Optional[str] = None, min_periods: int = 6
    ) -> Optional[float]:
        """
        Estimate IC half-life: the number of days over which IC decays to
        half its initial value.

        Method: compute rolling 30-day IC windows, fit exponential decay
        IC(t) = IC_0 * exp(-t * ln(2) / half_life), and return half_life
        in days.  Returns None if insufficient data.
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Compute IC over overlapping 30-day windows, stepped by 10 days
        window_days = 30
        step_days = 10
        ics: List[tuple] = []  # (center_offset_days, ic_value)

        offset = 0
        while True:
            window_end = end_dt - timedelta(days=offset)
            window_start = window_end - timedelta(days=window_days)
            if window_start < end_dt - timedelta(days=365):
                break

            ic = self.compute_ic(
                source,
                lookback_days=window_days,
                end_date=window_end.strftime("%Y-%m-%d"),
            )
            if ic is not None:
                ics.append((offset, ic))
            offset += step_days

        if len(ics) < min_periods:
            logger.debug(
                "Insufficient IC windows for half-life: source=%s, windows=%d",
                source,
                len(ics),
            )
            return None

        # Fit: IC(t) = IC_0 * exp(-k * t), half_life = ln(2) / k
        # Use log-linear regression: ln|IC(t)| = ln|IC_0| - k * t
        import math

        filtered = [(t, abs(ic)) for t, ic in ics if abs(ic) > 1e-9]
        if len(filtered) < min_periods:
            return None

        n = len(filtered)
        sum_t = sum(t for t, _ in filtered)
        sum_y = sum(math.log(v) for _, v in filtered)
        sum_tt = sum(t * t for t, _ in filtered)
        sum_ty = sum(t * math.log(v) for t, v in filtered)

        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            return None

        # slope = -k (decay rate per day)
        k = -(n * sum_ty - sum_t * sum_y) / denom
        if k <= 0:
            # No decay detected — IC is stable or increasing
            return float("inf")

        half_life = math.log(2) / k
        return round(half_life, 1)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report including IC metrics."""
        scores = self.calculate_all_health_scores()
        alerts = self.detect_decay_alerts()

        healthy_count = sum(1 for s in scores.values() if s.status == "healthy")
        degraded_count = sum(1 for s in scores.values() if s.status == "degraded")
        unhealthy_count = sum(1 for s in scores.values() if s.status == "unhealthy")

        # Compute IC for each tracked source
        ic_data = {}
        for source in scores:
            ic = self.compute_ic(source)
            half_life = self.compute_ic_half_life(source)
            ic_data[source] = {
                "ic": round(ic, 4) if ic is not None else None,
                "ic_half_life_days": half_life,
            }

        with sqlite_connect(self.db_path) as conn:
            cursor = conn.cursor()
            total_predictions = cursor.execute(
                "SELECT COUNT(*) FROM signal_predictions"
            ).fetchone()[0]
            resolved_predictions = cursor.execute(
                "SELECT COUNT(*) FROM signal_predictions WHERE actual_direction IS NOT NULL"
            ).fetchone()[0]
        pending_predictions = max(0, int(total_predictions) - int(resolved_predictions))

        if len(scores) == 0:
            overall_health = "insufficient_data" if pending_predictions else "unknown"
            status = "insufficient_data" if pending_predictions else "no_data"
        else:
            overall_health = "healthy" if healthy_count >= len(scores) * 0.6 else "degraded"
            status = overall_health

        collapsed_n = sum(
            1 for s in scores.values() if getattr(s, "window_collapse_90_60", False)
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total_tracked": len(scores),
                "total_predictions": int(total_predictions),
                "resolved_predictions": int(resolved_predictions),
                # Full historical unlabeled row count in signal_predictions
                "pending_predictions": pending_predictions,
                "pending_rows": pending_predictions,
                "pending_scope": "historical_db_unlabeled_rows",
                "pending_semantics": (
                    "pending_predictions/pending_rows = COUNT(signal_predictions) "
                    "WHERE actual_direction IS NULL (full history). "
                    "Not the same as signals.ic_decay.pending_predictions "
                    "(IC staged-date window)."
                ),
                # Batch BU: how many arms used collapsed multi-window weights
                "window_collapse_90_60_count": collapsed_n,
            },
            "scores": {s: scores[s].to_dict() for s in scores},
            "ic_metrics": ic_data,
            "alerts": [a.to_dict() for a in alerts],
            "overall_health": overall_health,
            "status": status,
            "label_horizon": "SPY actual direction resolved by update_actual_directions for each prediction date",
            "health_score_policy": {
                "full_scheme": "50% 90d + 30% 60d + 20% 30d",
                "collapsed_scheme": "40% 60d + 60% 30d (recency bias; no fake 90d)",
                "collapse_rule": "c90==c60 (no extra labeled history in 60→90)",
                "sampling_unit": "latest_prediction_per_source_calendar_date",
                "sampling_reason": (
                    "SPY forward labels are assigned per calendar date; one latest "
                    "source/date observation prevents cron/test run-frequency bias."
                ),
                "live_authoritative": False,
            },
        }

def backfill_predictions(
    db_path: Optional[Path] = None,
    start_date: str = "2024-01-01"
) -> int:
    """
    Backfill historical predictions from existing signals data.
    
    This populates the signal_predictions table from existing
    signal history for health score calculation.
    """
    tracker = SignalHealthTracker(db_path)
    
    # Load from regime_log as proxy for historical signals
    with sqlite_connect(tracker.db_path) as conn:
        cursor = conn.cursor()
    
        count = 0
    
        try:
            # Get historical regime classifications as HMM signal proxy
            cursor.execute("""
                SELECT date, regime, vix_level FROM regime_log
                WHERE date >= date(?) AND regime IS NOT NULL
                ORDER BY date
            """, (start_date,))
        
            rows = cursor.fetchall()
        
            for row in rows:
                date, regime, vix = row
            
                # Convert regime to signal value
                signal_map = {
                    'bull': 0.8,
                    'bear': -0.8,
                    'neutral': 0.0,
                    'high_vol': -0.3,
                    'crisis': -0.9
                }
            
                signal_value = signal_map.get(regime, 0.0)
            
                # Calculate actual direction from next day's SPY return
                cursor.execute("""
                    SELECT close FROM prices
                    WHERE symbol = 'SPY' AND date > date(?)
                    ORDER BY date LIMIT 2
                """, (date,))
            
                price_rows = cursor.fetchall()
                if len(price_rows) == 2:
                    p1, p2 = price_rows[0][0], price_rows[1][0]
                    ret = (p2 - p1) / p1 if p1 > 0 else 0
                    actual = 1 if ret > 0 else (-1 if ret < 0 else 0)
                
                    # Log prediction
                    prediction = SignalPrediction(
                        timestamp=date + "T00:00:00",
                        source="hmm",
                        signal_value=signal_value,
                        confidence=0.7 if regime in ['bull', 'bear'] else 0.5,
                        predicted_direction=SignalHealthTracker.direction_from_signal_value(
                            signal_value
                        ),
                        metadata={"regime": regime, "vix": vix}
                    )
                
                    # Insert with actual direction
                    cursor.execute("""
                        INSERT OR IGNORE INTO signal_predictions
                        (timestamp, source, signal_value, confidence, predicted_direction, metadata, actual_direction, accuracy_calculated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """, (
                        prediction.timestamp,
                        prediction.source,
                        prediction.signal_value,
                        prediction.confidence,
                        prediction.predicted_direction,
                        json.dumps(prediction.metadata),
                        actual
                    ))
                
                    if cursor.rowcount > 0:
                        count += 1
        
            conn.commit()
            logger.info("Backfilled %d historical predictions", count)
        
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError) as e:
            logger.error("Backfill error: %s", e)
    
    return count

# CLI interface
if __name__ == "__main__":
    from src.utils.log_config import configure_logging
    configure_logging()
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Health Tracker v3.12")
    parser.add_argument("--status", action="store_true", help="Show health status")
    parser.add_argument("--backfill", action="store_true", help="Backfill historical data")
    parser.add_argument("--calculate", action="store_true", help="Calculate and save health scores")
    parser.add_argument("--alerts", action="store_true", help="Check for decay alerts")
    parser.add_argument(
        "--resolve-labels",
        action="store_true",
        help="Resolve pending predictions with SPY forward returns (bounded)",
    )
    parser.add_argument(
        "--resolve-max-days",
        type=int,
        default=DEFAULT_RESOLVE_MAX_DAYS,
        help=f"Max unresolved dates per --resolve-labels run (default {DEFAULT_RESOLVE_MAX_DAYS})",
    )
    parser.add_argument(
        "--resolve-oldest-first",
        action="store_true",
        help="Catch-up mode: label oldest unresolved dates first (backlog drain)",
    )
    parser.add_argument("--source", type=str, help="Specific signal source")
    
    args = parser.parse_args()
    
    tracker = SignalHealthTracker()

    if args.resolve_labels:
        summary = tracker.resolve_pending_labels(
            max_days=args.resolve_max_days,
            oldest_first=bool(args.resolve_oldest_first),
        )
        logger.info("resolve_pending_labels: %s", json.dumps(summary))
        if not (args.backfill or args.calculate or args.alerts or args.status):
            # Resolve-only invocation: skip default status dump
            raise SystemExit(0)

    if args.backfill:
        count = backfill_predictions()
        logger.info("Backfilled %d predictions", count)
    
    elif args.calculate:
        if args.source:
            score = tracker.calculate_health_score(args.source)
            if score:
                tracker.save_health_scores({args.source: score})
                logger.info(json.dumps(score.to_dict(), indent=2))
            else:
                logger.info("No data available for %s", args.source)
        else:
            scores = tracker.calculate_all_health_scores()
            tracker.save_health_scores(scores)
            logger.info("Calculated health for %d sources", len(scores))
            for s, score in scores.items():
                logger.info("  %s: %.3f (%s)", s, score.health_score, score.status)
    
    elif args.alerts:
        alerts = tracker.detect_decay_alerts()
        if alerts:
            logger.info("Found %d decay alerts:", len(alerts))
            for alert in alerts:
                logger.info("  WARNING: %s", alert.message)
        else:
            logger.info("No decay alerts - all signals healthy")
    else:
        # Default to status
        report = tracker.get_health_report()
        logger.info("\n=== Signal Health Report ===")
        logger.info("Generated: %s", report['timestamp'])
        logger.info("\nSummary: %d healthy, %d degraded, %d unhealthy",
                    report['summary']['healthy'], report['summary']['degraded'],
                    report['summary']['unhealthy'])
        logger.info("\nOverall Status: %s", report['overall_health'].upper())
        
        logger.info("\nHealth Scores:")
        for source, score in report['scores'].items():
            status_text = "HEALTHY" if score['status'] == 'healthy' else (
                "DEGRADED" if score['status'] == 'degraded' else "UNHEALTHY")
            logger.info("  %s %-12s %.3f (30d: %.1f%%, 90d: %.1f%%)",
                        status_text, source, score['health_score'],
                        score['accuracy_30d'] * 100, score['accuracy_90d'] * 100)
        if report['alerts']:
            logger.info("\nDecay Alerts:")
            for alert in report['alerts']:
                logger.info("  %s: %s", alert['severity'].upper(), alert['message'])
