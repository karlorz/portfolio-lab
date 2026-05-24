#!/usr/bin/env python3
"""
Signal Health Decay Tracking - v3.12 Phase 1
Data infrastructure and health calculator for ensemble voter

Tracks rolling accuracy of signal sources to enable dynamic weight adjustment
when signals show degradation (health < 0.5 triggers weight reduction).

References:
- v3.12 spec: wiki/projects/portfolio-lab/work/2026-05-14-v312-signal-health-decay-tracking/
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DB_PATH = MARKET_DB
STATE_PATH = DATA_DIR / ".signal_health_state.json"

class SignalSource(Enum):
    """Signal sources tracked for health monitoring."""
    MULTI_SPEED_MOM = "multi_speed_momentum"
    CROSS_ASSET_RV = "cross_asset_rv"
    INTERNATIONAL_MOMENTUM = "international_momentum"
    ALTERNATIVE_DATA = "alternative_data"
    CROSS_ASSET_REGIME_ARB = "cross_asset_regime_arb"
    UNIFIED_OVERLAY = "unified_overlay"

class SignalHealthStatus(Enum):
    """Health status classification."""
    HEALTHY = "healthy"  # health >= 0.7
    DEGRADED = "degraded"  # 0.5 <= health < 0.7
    UNHEALTHY = "unhealthy"  # health < 0.5

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
    decay_rate: float  # Daily decay rate (negative = improving)
    predictions_count: int
    status: str  # healthy/degraded/unhealthy
    ic: Optional[float] = None  # Information Coefficient (Spearman ρ)
    ic_half_life_days: Optional[float] = None  # IC half-life in days (inf = stable)

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
        with open(STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=2)
    
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
    
    def log_prediction_simple(
        self,
        source: str,
        signal_value: float,
        confidence: float,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Convenience method for logging predictions."""
        # Determine predicted direction
        if signal_value > 0.2:
            predicted = 1
        elif signal_value < -0.2:
            predicted = -1
        else:
            predicted = 0
        
        prediction = SignalPrediction(
            timestamp=timestamp or datetime.now().isoformat(),
            source=source,
            signal_value=signal_value,
            confidence=confidence,
            predicted_direction=predicted,
            metadata=metadata or {}
        )
        
        self.log_prediction(prediction)
    
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
    
    def calculate_health_score(
        self, 
        source: str,
        end_date: Optional[str] = None
    ) -> Optional[HealthScore]:
        """
        Calculate health score for a signal source.
        
        Health score formula:
        - 50% weight on 90-day accuracy
        - 30% weight on 60-day accuracy  
        - 20% weight on 30-day accuracy
        - Decay penalty if health dropping >20% in 30 days
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
        
            for period, start_date in periods.items():
                cursor.execute("""
                    SELECT predicted_direction, actual_direction
                    FROM signal_predictions
                    WHERE source = ? 
                    AND date(timestamp) BETWEEN date(?) AND date(?)
                    AND actual_direction IS NOT NULL
                """, (source, start_date, end_date))
            
                rows = cursor.fetchall()
            
                if not rows:
                    accuracies[period] = 0.5  # Neutral if no data
                    counts[period] = 0
                    continue
            
                # Calculate directional accuracy
                correct = sum(1 for pred, actual in rows if pred == actual and pred != 0)
                total = sum(1 for pred, actual in rows if pred != 0)  # Exclude neutral predictions
            
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
        
        health = (
            accuracies['90d'] * 0.5 +
            accuracies['60d'] * 0.3 +
            accuracies['30d'] * 0.2
        )
        
        # Calculate decay rate (change per day over 30 days)
        decay_rate = (accuracies['30d'] - accuracies['60d']) / 30 if counts['60d'] > 0 else 0
        
        # Determine status
        if health >= 0.7:
            status = SignalHealthStatus.HEALTHY.value
        elif health >= 0.5:
            status = SignalHealthStatus.DEGRADED.value
        else:
            status = SignalHealthStatus.UNHEALTHY.value
        
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
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
                cursor.execute("""
                    SELECT timestamp, health_score
                    FROM signal_health_scores
                    WHERE source = ? AND date(timestamp) >= date(?)
                    ORDER BY timestamp ASC
                """, (source.value, start_date))
            
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
            except Exception:
                continue

            if current_ic is None:
                continue

            # Get IC history from rolling windows
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

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
                except Exception:
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
    ) -> Dict[str, float]:
        """
        Calculate health-adjusted weights for ensemble voting.

        Formula:
          adjusted_weight = base_weight * health_multiplier * ic_multiplier

        health_multiplier = max(min_multiplier, health_score)
        ic_multiplier:
          - |IC| > ic_bonus_threshold: 1.0 + ic_weight_factor * |IC|
          - |IC| < ic_penalty_threshold: 1.0 - ic_weight_factor * (1 - |IC|/ic_penalty_threshold)
          - otherwise: 1.0 (neutral)

        Args:
            base_weights: Dict mapping source to base weight (should sum to 1.0)
            min_weight_multiplier: Floor for weight adjustment (default 0.2)
            ic_bonus_threshold: IC above this gets a weight boost (default 0.05)
            ic_penalty_threshold: IC below this gets a weight penalty (default 0.02)
            ic_weight_factor: Magnitude of IC adjustment (default 0.15)

        Returns:
            Dict of adjusted weights (normalized to sum to 1.0)
        """
        scores = self.calculate_all_health_scores()

        adjusted = {}
        for source, base_weight in base_weights.items():
            score = scores.get(source)
            if score:
                # Health multiplier
                health_mult = max(min_weight_multiplier, score.health_score)

                # IC multiplier
                ic_mult = 1.0
                if score.ic is not None:
                    abs_ic = abs(score.ic)
                    if abs_ic > ic_bonus_threshold:
                        ic_mult = 1.0 + ic_weight_factor * abs_ic
                    elif abs_ic < ic_penalty_threshold:
                        ic_mult = 1.0 - ic_weight_factor * (1 - abs_ic / ic_penalty_threshold)

                adjusted[source] = base_weight * health_mult * ic_mult
            else:
                # No health data - use neutral health (0.5)
                adjusted[source] = base_weight * 0.5

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

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
                FROM signal_predictions
                WHERE source = ?
                  AND date(timestamp) BETWEEN date(?) AND date(?)
                  AND actual_direction IS NOT NULL
                  AND signal_value IS NOT NULL
                ORDER BY timestamp
                """,
                (source, start_date, end_date),
            )
            rows = cursor.fetchall()

        if len(rows) < 3:
            logger.info("Insufficient data for IC: source=%s, rows=%d", source, len(rows))
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
            logger.info(
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

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total_tracked": len(scores)
            },
            "scores": {s: scores[s].to_dict() for s in scores},
            "ic_metrics": ic_data,
            "alerts": [a.to_dict() for a in alerts],
            "overall_health": "healthy" if healthy_count >= len(scores) * 0.6 else "degraded"
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
                        predicted_direction=1 if signal_value > 0.2 else (-1 if signal_value < -0.2 else 0),
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
        
        except Exception as e:
            logger.error("Backfill error: %s", e)
    
    return count

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Health Tracker v3.12")
    parser.add_argument("--status", action="store_true", help="Show health status")
    parser.add_argument("--backfill", action="store_true", help="Backfill historical data")
    parser.add_argument("--calculate", action="store_true", help="Calculate and save health scores")
    parser.add_argument("--alerts", action="store_true", help="Check for decay alerts")
    parser.add_argument("--source", type=str, help="Specific signal source")
    
    args = parser.parse_args()
    
    tracker = SignalHealthTracker()
    
    if args.backfill:
        count = backfill_predictions()
        print(f"Backfilled {count} predictions")
    
    elif args.calculate:
        if args.source:
            score = tracker.calculate_health_score(args.source)
            if score:
                tracker.save_health_scores({args.source: score})
                print(json.dumps(score.to_dict(), indent=2))
            else:
                print(f"No data available for {args.source}")
        else:
            scores = tracker.calculate_all_health_scores()
            tracker.save_health_scores(scores)
            print(f"Calculated health for {len(scores)} sources")
            for s, score in scores.items():
                print(f"  {s}: {score.health_score:.3f} ({score.status})")
    
    elif args.alerts:
        alerts = tracker.detect_decay_alerts()
        if alerts:
            print(f"Found {len(alerts)} decay alerts:")
            for alert in alerts:
                print(f"  ⚠️ {alert.message}")
        else:
            print("No decay alerts - all signals healthy")
    
    else:
        # Default to status
        report = tracker.get_health_report()
        print("\n=== Signal Health Report ===")
        print(f"Generated: {report['timestamp']}")
        print(f"\nSummary: {report['summary']['healthy']} healthy, "
              f"{report['summary']['degraded']} degraded, "
              f"{report['summary']['unhealthy']} unhealthy")
        print(f"\nOverall Status: {report['overall_health'].upper()}")
        
        print("\nHealth Scores:")
        for source, score in report['scores'].items():
            status_icon = "🟢" if score['status'] == 'healthy' else ("🟡" if score['status'] == 'degraded' else "🔴")
            print(f"  {status_icon} {source:12s} {score['health_score']:.3f} "
                  f"(30d: {score['accuracy_30d']:.1%}, 90d: {score['accuracy_90d']:.1%})")
        
        if report['alerts']:
            print("\n⚠️ Decay Alerts:")
            for alert in report['alerts']:
                print(f"  {alert['severity'].upper()}: {alert['message']}")
