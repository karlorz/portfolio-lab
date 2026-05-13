"""
Signal Health Monitor (v2.71)

Tracks signal quality and decay for ensemble voter components.
- Rolling correlation tracking (90-day window)
- Win rate monitoring
- Signal decay detection
- Automatic weight adjustment recommendations
"""

import os
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class SignalHealthMetrics:
    """Health metrics for a single signal source"""
    source: str
    timestamp: str
    
    # Correlation metrics (90-day rolling)
    prediction_correlation: float  # Correlation between signal and realized returns
    correlation_trend: str  # 'improving', 'stable', 'decaying'
    correlation_pvalue: float
    
    # Performance metrics
    win_rate_30d: float  # % of correct directional calls
    win_rate_90d: float
    win_rate_trend: str  # 'improving', 'stable', 'decaying'
    
    # Signal decay
    decay_rate: float  # Daily decay in predictive power
    half_life_days: float  # Days until correlation halves
    
    # Composite health score
    health_score: float  # 0-1 composite score
    health_status: str  # 'healthy', 'degraded', 'critical'
    
    # Recommendations
    recommended_action: str  # 'maintain', 'reduce_weight', 'increase_weight', 'disable'
    weight_adjustment: float  # Recommended weight multiplier


@dataclass
class EnsembleHealthReport:
    """Complete health report for all ensemble signals"""
    timestamp: str
    signals: Dict[str, SignalHealthMetrics]
    
    # Aggregate metrics
    overall_health: float  # Average health score
    consensus_degradation: bool  # Multiple signals degrading
    
    # Recommendations
    weight_adjustments: Dict[str, float]
    alerts: List[str]
    recommended_ensemble_weights: Dict[str, float]


class SignalHealthMonitor:
    """
    Monitors signal health for ensemble components.
    
    Tracks:
    - Rolling prediction accuracy (correlation with realized returns)
    - Win rates over 30d and 90d windows
    - Signal decay rates
    - Automatic weight adjustment recommendations
    """
    
    HEALTH_THRESHOLDS = {
        'healthy': 0.70,
        'degraded': 0.50,
        'critical': 0.30
    }
    
    CORRELATION_DECAY_THRESHOLD = 0.10  # 10% decline triggers alert
    WIN_RATE_DECAY_THRESHOLD = 0.15  # 15% decline triggers alert
    
    def __init__(self, db_path: str = "data/signal_health.db"):
        self.db_path = Path(project_root) / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for signal health tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signal predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_predictions (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction REAL,  -- Signal value (e.g., regime probability)
                direction INTEGER,  -- -1, 0, 1 for bear/neutral/bull
                realized_return_1d REAL,
                realized_return_5d REAL,
                realized_direction_1d INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, timestamp)
            )
        """)
        
        # Health metrics history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_correlation REAL,
                win_rate_30d REAL,
                win_rate_90d REAL,
                decay_rate REAL,
                health_score REAL,
                health_status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_prediction(
        self,
        source: str,
        timestamp: str,
        prediction: float,
        direction: int,
        realized_return_1d: Optional[float] = None,
        realized_return_5d: Optional[float] = None,
        realized_direction_1d: Optional[int] = None
    ):
        """Record a signal prediction for later health analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO signal_predictions 
            (source, timestamp, prediction, direction, realized_return_1d, 
             realized_return_5d, realized_direction_1d)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, timestamp, prediction, direction, realized_return_1d,
              realized_return_5d, realized_direction_1d))
        
        conn.commit()
        conn.close()
    
    def update_realized_returns(
        self,
        source: str,
        timestamp: str,
        realized_return_1d: float,
        realized_return_5d: float,
        realized_direction_1d: int
    ):
        """Update predictions with realized returns after the fact"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE signal_predictions 
            SET realized_return_1d = ?, realized_return_5d = ?, 
                realized_direction_1d = ?
            WHERE source = ? AND timestamp = ?
        """, (realized_return_1d, realized_return_5d, realized_direction_1d,
              source, timestamp))
        
        conn.commit()
        conn.close()
    
    def calculate_health_metrics(
        self,
        source: str,
        lookback_days: int = 90
    ) -> Optional[SignalHealthMetrics]:
        """Calculate health metrics for a signal source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions with realized returns
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        cursor.execute("""
            SELECT timestamp, prediction, direction, realized_return_1d, 
                   realized_return_5d, realized_direction_1d
            FROM signal_predictions
            WHERE source = ? AND timestamp >= ? AND timestamp <= ?
            AND realized_return_1d IS NOT NULL
            ORDER BY timestamp DESC
        """, (source, start_date.isoformat(), end_date.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 30:  # Need at least 30 observations
            return None
        
        # Extract data
        predictions = np.array([r[1] for r in rows])
        directions = np.array([r[2] for r in rows])
        realized_returns = np.array([r[3] for r in rows if r[3] is not None])
        realized_dirs = np.array([r[5] for r in rows if r[5] is not None])
        
        if len(realized_returns) < 30:
            return None
        
        # Calculate correlation (prediction vs realized returns)
        if len(predictions) == len(realized_returns):
            correlation = np.corrcoef(predictions[:len(realized_returns)], realized_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Calculate win rates
        win_rate_30d = self._calculate_win_rate(directions, realized_dirs, days=30)
        win_rate_90d = self._calculate_win_rate(directions, realized_dirs, days=90)
        
        # Determine trends
        correlation_trend = self._calculate_correlation_trend(source, correlation)
        win_rate_trend = self._calculate_win_rate_trend(source, win_rate_90d)
        
        # Calculate decay
        decay_rate, half_life = self._calculate_decay(source, correlation)
        
        # Composite health score
        health_score = self._calculate_health_score(
            correlation, win_rate_30d, win_rate_90d, decay_rate
        )
        
        # Health status
        if health_score >= self.HEALTH_THRESHOLDS['healthy']:
            health_status = 'healthy'
        elif health_score >= self.HEALTH_THRESHOLDS['degraded']:
            health_status = 'degraded'
        else:
            health_status = 'critical'
        
        # Recommendations
        recommended_action, weight_adjustment = self._get_recommendations(
            health_score, correlation_trend, decay_rate
        )
        
        return SignalHealthMetrics(
            source=source,
            timestamp=datetime.now().isoformat(),
            prediction_correlation=correlation,
            correlation_trend=correlation_trend,
            correlation_pvalue=0.05,  # Simplified
            win_rate_30d=win_rate_30d,
            win_rate_90d=win_rate_90d,
            win_rate_trend=win_rate_trend,
            decay_rate=decay_rate,
            half_life_days=half_life,
            health_score=health_score,
            health_status=health_status,
            recommended_action=recommended_action,
            weight_adjustment=weight_adjustment
        )
    
    def _calculate_win_rate(
        self,
        directions: np.ndarray,
        realized_dirs: np.ndarray,
        days: int
    ) -> float:
        """Calculate win rate for last N days"""
        if len(directions) < days or len(realized_dirs) < days:
            return 0.0
        
        recent_dirs = directions[:days]
        recent_realized = realized_dirs[:days]
        
        # Only count non-neutral predictions
        valid_mask = recent_dirs != 0
        if valid_mask.sum() == 0:
            return 0.0
        
        correct = (recent_dirs[valid_mask] == recent_realized[valid_mask]).sum()
        return correct / valid_mask.sum()
    
    def _calculate_correlation_trend(
        self,
        source: str,
        current_correlation: float
    ) -> str:
        """Determine if correlation is improving, stable, or decaying"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get historical correlations
        cursor.execute("""
            SELECT prediction_correlation
            FROM health_metrics
            WHERE source = ? AND timestamp >= date('now', '-30 days')
            ORDER BY timestamp DESC
            LIMIT 10
        """, (source,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 5:
            return 'stable'
        
        historical_corrs = [r[0] for r in rows if r[0] is not None]
        if len(historical_corrs) < 5:
            return 'stable'
        
        avg_historical = np.mean(historical_corrs)
        change = current_correlation - avg_historical
        
        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'decaying'
        return 'stable'
    
    def _calculate_win_rate_trend(
        self,
        source: str,
        current_win_rate: float
    ) -> str:
        """Determine if win rate is improving, stable, or decaying"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT win_rate_90d
            FROM health_metrics
            WHERE source = ? AND timestamp >= date('now', '-30 days')
            ORDER BY timestamp DESC
            LIMIT 10
        """, (source,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 5:
            return 'stable'
        
        historical_rates = [r[0] for r in rows if r[0] is not None]
        if len(historical_rates) < 5:
            return 'stable'
        
        avg_historical = np.mean(historical_rates)
        change = current_win_rate - avg_historical
        
        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'decaying'
        return 'stable'
    
    def _calculate_decay(
        self,
        source: str,
        current_correlation: float
    ) -> Tuple[float, float]:
        """Calculate signal decay rate and half-life"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get 60-day correlation history
        cursor.execute("""
            SELECT prediction_correlation, timestamp
            FROM health_metrics
            WHERE source = ? AND timestamp >= date('now', '-60 days')
            ORDER BY timestamp ASC
        """, (source,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 10:
            return 0.0, 999.0  # No decay detected
        
        correlations = [r[0] for r in rows if r[0] is not None]
        if len(correlations) < 10:
            return 0.0, 999.0
        
        # Linear regression to estimate decay
        x = np.arange(len(correlations))
        y = np.array(correlations)
        
        # Simple slope calculation
        slope = np.polyfit(x, y, 1)[0]
        decay_rate = slope  # Per observation decay
        
        # Calculate half-life (days until correlation halves)
        if current_correlation > 0 and decay_rate < 0:
            half_life = -current_correlation / (decay_rate * len(correlations) / 60)
        else:
            half_life = 999.0  # Stable or improving
        
        return decay_rate, max(half_life, 0)
    
    def _calculate_health_score(
        self,
        correlation: float,
        win_rate_30d: float,
        win_rate_90d: float,
        decay_rate: float
    ) -> float:
        """Calculate composite health score (0-1)"""
        # Correlation component (40%)
        corr_score = max(0, min(1, (correlation + 0.2) / 0.5))
        
        # Win rate component (40%)
        win_score = (win_rate_30d + win_rate_90d) / 2
        
        # Decay penalty (20%)
        decay_penalty = max(0, min(1, -decay_rate * 100))
        
        health_score = 0.4 * corr_score + 0.4 * win_score - 0.2 * decay_penalty
        return max(0, min(1, health_score))
    
    def _get_recommendations(
        self,
        health_score: float,
        correlation_trend: str,
        decay_rate: float
    ) -> Tuple[str, float]:
        """Get action recommendation and weight adjustment"""
        if health_score >= 0.80 and correlation_trend == 'improving':
            return 'increase_weight', 1.20
        elif health_score >= 0.70:
            return 'maintain', 1.0
        elif health_score >= 0.50 and decay_rate > -0.01:
            return 'maintain', 1.0
        elif health_score >= 0.30:
            return 'reduce_weight', 0.70
        else:
            return 'disable', 0.0
    
    def generate_health_report(
        self,
        sources: List[str]
    ) -> EnsembleHealthReport:
        """Generate complete health report for all signal sources"""
        signals = {}
        weight_adjustments = {}
        alerts = []
        
        for source in sources:
            metrics = self.calculate_health_metrics(source)
            if metrics:
                signals[source] = metrics
                weight_adjustments[source] = metrics.weight_adjustment
                
                # Generate alerts
                if metrics.health_status == 'critical':
                    alerts.append(f"CRITICAL: {source} health score {metrics.health_score:.2f}")
                elif metrics.correlation_trend == 'decaying' and metrics.health_score < 0.60:
                    alerts.append(f"WARNING: {source} correlation decaying")
                elif metrics.half_life_days < 30:
                    alerts.append(f"ALERT: {source} half-life only {metrics.half_life_days:.0f} days")
        
        # Overall health
        if signals:
            overall_health = np.mean([s.health_score for s in signals.values()])
        else:
            overall_health = 0.0
        
        # Consensus degradation check
        degraded_count = sum(1 for s in signals.values() if s.health_status == 'degraded')
        consensus_degradation = degraded_count >= len(signals) / 3
        
        if consensus_degradation:
            alerts.append("WARNING: Multiple signals showing degradation - consider risk-off")
        
        # Calculate recommended ensemble weights
        base_weights = self._get_base_weights()
        recommended_weights = self._adjust_weights(base_weights, weight_adjustments)
        
        return EnsembleHealthReport(
            timestamp=datetime.now().isoformat(),
            signals=signals,
            overall_health=overall_health,
            consensus_degradation=consensus_degradation,
            weight_adjustments=weight_adjustments,
            alerts=alerts,
            recommended_ensemble_weights=recommended_weights
        )
    
    def _get_base_weights(self) -> Dict[str, float]:
        """Get base ensemble weights from config"""
        config_path = Path(project_root) / "config" / "ensemble_weights.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return {
                    k: v['weight'] for k, v in config.get('weights', {}).items()
                }
        return {
            'hmm_regime': 0.35,
            'cta_trend': 0.22,
            'duration_yield': 0.13,
            'sector_momentum': 0.10,
            'alternative_data': 0.10,
            'cash_reserve': 0.10
        }
    
    def _adjust_weights(
        self,
        base_weights: Dict[str, float],
        adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust weights based on health recommendations"""
        adjusted = {}
        total = 0
        
        for source, base in base_weights.items():
            adj = adjustments.get(source, 1.0)
            adjusted[source] = base * adj
            total += adjusted[source]
        
        # Normalize to sum to 1
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def save_health_metrics(self, metrics: SignalHealthMetrics):
        """Save health metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO health_metrics
            (source, timestamp, prediction_correlation, win_rate_30d, win_rate_90d,
             decay_rate, health_score, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.source, metrics.timestamp, metrics.prediction_correlation,
            metrics.win_rate_30d, metrics.win_rate_90d, metrics.decay_rate,
            metrics.health_score, metrics.health_status
        ))
        
        conn.commit()
        conn.close()
    
    def get_health_history(
        self,
        source: str,
        days: int = 30
    ) -> List[Dict]:
        """Get historical health metrics for a source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, health_score, prediction_correlation, win_rate_90d
            FROM health_metrics
            WHERE source = ? AND timestamp >= date('now', ?)
            ORDER BY timestamp ASC
        """, (source, f'-{days} days'))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': r[0],
                'health_score': r[1],
                'correlation': r[2],
                'win_rate': r[3]
            }
            for r in rows
        ]


def cli():
    """CLI for signal health monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Signal Health Monitor v2.71')
    parser.add_argument('--report', action='store_true', help='Generate health report')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--source', type=str, help='Specific signal source to analyze')
    parser.add_argument('--history', action='store_true', help='Show health history')
    
    args = parser.parse_args()
    
    monitor = SignalHealthMonitor()
    
    sources = [
        'hmm_regime',
        'cta_trend', 
        'duration_yield',
        'sector_momentum',
        'alternative_data'
    ]
    
    if args.report:
        report = monitor.generate_health_report(sources)
        
        print("\n" + "="*60)
        print("ENSEMBLE SIGNAL HEALTH REPORT")
        print("="*60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Health: {report.overall_health:.2%}")
        print(f"Consensus Degradation: {report.consensus_degradation}")
        
        print("\n" + "-"*60)
        print("SIGNAL HEALTH DETAILS")
        print("-"*60)
        
        for source, metrics in report.signals.items():
            print(f"\n{source.upper()}:")
            print(f"  Health Score: {metrics.health_score:.2%} ({metrics.health_status})")
            print(f"  Correlation: {metrics.prediction_correlation:.3f} ({metrics.correlation_trend})")
            print(f"  Win Rate (30d/90d): {metrics.win_rate_30d:.1%} / {metrics.win_rate_90d:.1%}")
            print(f"  Decay Rate: {metrics.decay_rate:.4f} (half-life: {metrics.half_life_days:.0f} days)")
            print(f"  Recommendation: {metrics.recommended_action} (weight: {metrics.weight_adjustment:.2f}x)")
        
        if report.alerts:
            print("\n" + "!"*60)
            print("ALERTS")
            print("!"*60)
            for alert in report.alerts:
                print(f"  ⚠️  {alert}")
        
        print("\n" + "-"*60)
        print("RECOMMENDED WEIGHT ADJUSTMENTS")
        print("-"*60)
        for source, weight in report.recommended_ensemble_weights.items():
            base = report.weight_adjustments.get(source, 1.0)
            change = "↑" if base > 1.0 else "↓" if base < 1.0 else "="
            print(f"  {source}: {weight:.1%} {change}")
        
        print("\n" + "="*60)
    
    elif args.source and args.history:
        history = monitor.get_health_history(args.source, days=30)
        
        print(f"\nHealth History for {args.source} (Last 30 Days)")
        print("-"*60)
        
        for entry in history:
            print(f"  {entry['timestamp'][:10]}: "
                  f"health={entry['health_score']:.2%}, "
                  f"corr={entry['correlation']:.3f}, "
                  f"win={entry['win_rate']:.1%}")
    
    elif args.monitor:
        print("Starting continuous monitoring (Ctrl+C to exit)...")
        import time
        
        try:
            while True:
                report = monitor.generate_health_report(sources)
                print(f"\n[{datetime.now().isoformat()}] "
                      f"Overall Health: {report.overall_health:.1%}")
                
                if report.alerts:
                    for alert in report.alerts[:3]:  # Show top 3
                        print(f"  ⚠️  {alert}")
                
                time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
