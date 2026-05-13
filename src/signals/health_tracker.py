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
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "market.db"
STATE_PATH = DATA_DIR / ".signal_health_state.json"


class SignalSource(Enum):
    """Signal sources tracked for health monitoring."""
    HMM = "hmm"  # HMM-LSTM Regime Detector
    CTA = "cta"  # CTA Trend Overlay (technical signals)
    MACRO_MOMENTUM = "macro_momentum"  # Macro/economic signals
    ALT_DATA = "alt_data"  # Alternative Data NLP
    FED_POLICY = "fed_policy"  # Fed Policy Overlay
    SENTIMENT = "sentiment"  # LLM Sentiment
    TAIL_HEDGE = "tail_hedge"  # Tail Risk Hedge
    VIX_SIGNAL = "vix"  # VIX-based signals
    DURATION = "duration"  # Duration/Yield Overlay


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
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
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
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
    
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
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update all predictions for this date with actual direction
        cursor.execute("""
            UPDATE signal_predictions 
            SET actual_direction = ?, accuracy_calculated = 1
            WHERE date(timestamp) = date(?) AND actual_direction IS NULL
        """, (actual_direction, date))
        
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {updated} predictions with actual direction for {date}")
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
        
        conn = sqlite3.connect(self.db_path)
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
        
        conn.close()
        
        # Weighted health score
        if counts['90d'] < 10:  # Need minimum data
            logger.warning(f"Insufficient data for {source}: only {counts['90d']} predictions")
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
            status=status
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
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
        self._save_state()
    
    def detect_decay_alerts(
        self,
        lookback_days: int = 30
    ) -> List[DecayAlert]:
        """
        Detect signals with significant health degradation.
        
        Returns alerts for signals where health dropped >20% over lookback period.
        """
        conn = sqlite3.connect(self.db_path)
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
                
                # Save to state
                if "decay_alerts" not in self.state:
                    self.state["decay_alerts"] = []
                self.state["decay_alerts"].append(alert.to_dict())
                # Keep only last 100 alerts
                self.state["decay_alerts"] = self.state["decay_alerts"][-100:]
        
        conn.close()
        self._save_state()
        
        return alerts
    
    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        min_weight_multiplier: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate health-adjusted weights for ensemble voting.
        
        Formula: adjusted_weight = base_weight * max(min_multiplier, health_score)
        
        Args:
            base_weights: Dict mapping source to base weight (should sum to 1.0)
            min_weight_multiplier: Floor for weight adjustment (default 0.2)
        
        Returns:
            Dict of adjusted weights (normalized to sum to 1.0)
        """
        scores = self.calculate_all_health_scores()
        
        adjusted = {}
        for source, base_weight in base_weights.items():
            score = scores.get(source)
            if score:
                multiplier = max(min_weight_multiplier, score.health_score)
                adjusted[source] = base_weight * multiplier
            else:
                # No health data - use neutral health (0.5)
                adjusted[source] = base_weight * 0.5
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        scores = self.calculate_all_health_scores()
        alerts = self.detect_decay_alerts()
        
        healthy_count = sum(1 for s in scores.values() if s.status == "healthy")
        degraded_count = sum(1 for s in scores.values() if s.status == "degraded")
        unhealthy_count = sum(1 for s in scores.values() if s.status == "unhealthy")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total_tracked": len(scores)
            },
            "scores": {s: scores[s].to_dict() for s in scores},
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
    conn = sqlite3.connect(tracker.db_path)
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
        logger.info(f"Backfilled {count} historical predictions")
        
    except Exception as e:
        logger.error(f"Backfill error: {e}")
    
    conn.close()
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
