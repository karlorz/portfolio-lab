#!/usr/bin/env python3
"""
Portfolio-Lab v2.81: Signal Health & Decay Monitoring Module

Implements signal health scoring and decay detection for the ensemble voter
to dynamically adjust signal weights based on real-world performance vs. 
backtest expectations.

Key Features:
- Rolling correlation tracking (prediction vs actual)
- Win rate monitoring (directional accuracy)
- Volatility regime detection
- Drawdown attribution
- Health-adjusted weight calculation
- Circuit breaker for degraded signals

Integration:
    from src.signals.health_monitor import SignalHealthMonitor
    
    monitor = SignalHealthMonitor()
    
    # Get health scores for all signal sources
    health = monitor.get_all_health_scores()
    
    # Calculate health-adjusted weights
    adjusted = monitor.calculate_adjusted_weights(base_weights)

CLI:
    python -m src.signals.health_monitor status
    python -m src.signals.health_monitor health --source tsmom
    python -m src.signals.health_monitor history --days 90
    python -m src.signals.health_monitor test-decay --source momentum

References:
    - spec: work/2026-05-13-v281-signal-health-monitoring/spec.md
    - compound: compound/trending-quant-strategies-2026-mid-may-update
"""

import json
import sqlite3
import argparse
import sys
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()
DB_PATH = DATA_DIR / "signals.db"
MARKET_DB_PATH = DATA_DIR / "market.db"
HEALTH_LOG_PATH = DATA_DIR / "signal_health_log.json"

# Health score thresholds
HEALTH_THRESHOLD_WARNING = 0.5   # Reduce weight by 50%
HEALTH_THRESHOLD_CRITICAL = 0.3  # Disable signal (circuit breaker)
HEALTH_THRESHOLD_RECOVERY = 0.6  # Begin restoring weight

# Rolling window periods (days)
HEALTH_WINDOWS = {
    'short': 30,
    'medium': 60,
    'long': 90
}

# Signal sources to monitor
MONITORED_SOURCES = [
    'hmm_regime',
    'tsmom',
    'fed_policy',
    'ai_agent',
    'momentum',
    'value',
    'macro',
    'quality',
    'sentiment',
    'vpin',  # v2.65
]

# Default base weights (from integrator.py)
DEFAULT_BASE_WEIGHTS = {
    'momentum': 0.20,
    'value': 0.15,
    'macro': 0.15,
    'quality': 0.10,
    'sentiment': 0.10,
    'ai_agent': 0.05,
    'tsmom': 0.10,
    'fed_policy': 0.10,
    'hmm_regime': 0.05,
    'vpin': 0.03,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class HealthMetrics:
    """Health metrics for a single signal source."""
    source: str
    timestamp: str
    
    # Rolling correlations (prediction vs actual)
    rolling_correlation_30d: Optional[float] = None
    rolling_correlation_60d: Optional[float] = None
    rolling_correlation_90d: Optional[float] = None
    
    # Win rate (directional accuracy)
    win_rate_30d: Optional[float] = None
    win_rate_60d: Optional[float] = None
    win_rate_90d: Optional[float] = None
    
    # Volatility regime
    current_volatility_regime: str = "neutral"  # low, neutral, high
    regime_stability: float = 1.0  # 0.0 to 1.0
    
    # Drawdown attribution
    drawdown_contribution_30d: float = 0.0
    max_drawdown_impact: float = 0.0
    
    # Sample counts
    samples_30d: int = 0
    samples_90d: int = 0
    
    # Metadata
    last_prediction_date: Optional[str] = None
    prediction_frequency: str = "daily"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthScore:
    """Composite health score for a signal source."""
    source: str
    timestamp: str
    
    # Component scores (0.0 to 1.0)
    correlation_score: float = 0.5
    accuracy_score: float = 0.5
    stability_score: float = 0.5
    
    # Composite score (weighted average)
    overall: float = 0.5
    
    # Status
    status: str = "healthy"  # healthy, degraded, critical, recovering
    
    # History
    trend: str = "stable"  # improving, stable, degrading
    days_in_current_state: int = 0
    
    # Action recommendation
    recommended_action: str = "maintain"  # maintain, reduce, disable, restore
    weight_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    """Complete health report for all signal sources."""
    timestamp: str
    scores: Dict[str, HealthScore] = field(default_factory=dict)
    metrics: Dict[str, HealthMetrics] = field(default_factory=dict)
    composite_health: float = 0.5
    
    # Alerts
    degraded_signals: List[str] = field(default_factory=list)
    critical_signals: List[str] = field(default_factory=list)
    recovering_signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'composite_health': round(self.composite_health, 4),
            'scores': {k: v.to_dict() for k, v in self.scores.items()},
            'alerts': {
                'degraded': self.degraded_signals,
                'critical': self.critical_signals,
                'recovering': self.recovering_signals,
            },
            'summary': {
                'total_monitored': len(self.scores),
                'healthy': sum(1 for s in self.scores.values() if s.status == 'healthy'),
                'degraded': len(self.degraded_signals),
                'critical': len(self.critical_signals),
            }
        }


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_health_database():
    """Initialize SQLite database for health tracking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Health scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            overall REAL,
            correlation_score REAL,
            accuracy_score REAL,
            stability_score REAL,
            status TEXT,
            trend TEXT,
            weight_multiplier REAL,
            UNIQUE(source, timestamp)
        )
    """)
    
    # Health metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            rolling_correlation_30d REAL,
            rolling_correlation_60d REAL,
            rolling_correlation_90d REAL,
            win_rate_30d REAL,
            win_rate_60d REAL,
            win_rate_90d REAL,
            current_volatility_regime TEXT,
            regime_stability REAL,
            drawdown_contribution_30d REAL,
            samples_30d INTEGER,
            UNIQUE(source, timestamp)
        )
    """)
    
    # Health history log (JSON for flexibility)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            report_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_health_source_ts 
        ON signal_health(source, timestamp)
    """)
    
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SignalHealthMonitor Class
# ---------------------------------------------------------------------------

class SignalHealthMonitor:
    """
    Signal health monitoring and decay detection system.
    
    Monitors signal sources for performance degradation and automatically
    adjusts weights to reduce exposure to decaying signals.
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_health_database()
        self._load_historical_scores()
    
    def _load_historical_scores(self):
        """Load previous health scores for trend calculation."""
        self._score_history = defaultdict(list)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load last 30 days of scores
        cursor.execute("""
            SELECT source, timestamp, overall, status
            FROM signal_health
            WHERE timestamp >= date('now', '-30 days')
            ORDER BY timestamp DESC
        """)
        
        for row in cursor.fetchall():
            source, ts, overall, status = row
            self._score_history[source].append({
                'timestamp': ts,
                'overall': overall,
                'status': status
            })
        
        conn.close()
    
    def calculate_rolling_correlation(
        self,
        source: str,
        ticker: str = "SPY",
        window_days: int = 30
    ) -> Optional[float]:
        """
        Calculate rolling correlation between signal predictions and actual returns.
        
        Args:
            source: Signal source name
            ticker: Asset ticker to analyze
            window_days: Rolling window size
        
        Returns:
            Pearson correlation coefficient or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get signal predictions
        cursor.execute("""
            SELECT timestamp, signal, ticker
            FROM signal_history
            WHERE source_name = ?
            AND ticker = ?
            AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (source, ticker, f'-{window_days} days', window_days))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 15:  # Need at least 15 samples
            return None
        
        # Fetch corresponding prices from market.db
        mkt_conn = sqlite3.connect(MARKET_DB_PATH)
        mkt_cursor = mkt_conn.cursor()
        
        predictions = []
        actuals = []
        
        for row in rows:
            ts, signal, _ = row
            # Get price at prediction time and next day
            mkt_cursor.execute("""
                SELECT date, close FROM prices 
                WHERE symbol = ? 
                AND date <= date(?)
                ORDER BY date DESC
                LIMIT 2
            """, (ticker, ts))
            
            price_rows = mkt_cursor.fetchall()
            if len(price_rows) >= 2:
                today_price = price_rows[0][1]
                prev_price = price_rows[1][1]
                if today_price and prev_price:
                    ret = (today_price / prev_price) - 1
                    predictions.append(signal)
                    actuals.append(ret)
        
        mkt_conn.close()
        
        if len(actuals) < 15:
            return None
        
        # Pearson correlation
        try:
            n = len(predictions)
            mean_p = statistics.mean(predictions)
            mean_a = statistics.mean(actuals)
            
            num = sum((p - mean_p) * (a - mean_a) 
                     for p, a in zip(predictions, actuals))
            den_p = sum((p - mean_p) ** 2 for p in predictions) ** 0.5
            den_a = sum((a - mean_a) ** 2 for a in actuals) ** 0.5
            
            if den_p == 0 or den_a == 0:
                return 0.0
            
            return num / (den_p * den_a)
        except:
            return None
    
    def calculate_win_rate(
        self,
        source: str,
        ticker: str = "SPY",
        window_days: int = 30
    ) -> Optional[float]:
        """
        Calculate directional accuracy (win rate) for signal predictions.
        
        Args:
            source: Signal source name
            ticker: Asset ticker
            window_days: Analysis window
        
        Returns:
            Win rate (0.0 to 1.0) or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, signal, ticker
            FROM signal_history
            WHERE source_name = ?
            AND ticker = ?
            AND timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (source, ticker, f'-{window_days} days', window_days + 1))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 20:
            return None
        
        # Fetch corresponding prices from market.db
        mkt_conn = sqlite3.connect(MARKET_DB_PATH)
        mkt_cursor = mkt_conn.cursor()
        
        predictions = []
        actuals = []
        
        for row in rows:
            ts, signal, _ = row
            mkt_cursor.execute("""
                SELECT date, close FROM prices 
                WHERE symbol = ? 
                AND date <= date(?)
                ORDER BY date DESC
                LIMIT 2
            """, (ticker, ts))
            
            price_rows = mkt_cursor.fetchall()
            if len(price_rows) >= 2:
                today_price = price_rows[0][1]
                prev_price = price_rows[1][1]
                if today_price and prev_price:
                    ret = (today_price / prev_price) - 1
                    predictions.append(signal)
                    actuals.append(ret)
        
        mkt_conn.close()
        
        # Check directional accuracy
        correct = 0
        total = 0
        
        for i in range(len(predictions)):
            if abs(predictions[i]) > 0.1:  # Only count confident predictions
                pred_direction = 1 if predictions[i] > 0 else -1
                actual_direction = 1 if actuals[i] > 0 else -1
                
                if pred_direction == actual_direction:
                    correct += 1
                total += 1
        
        if total < 10:
            return None
        
        return correct / total
    
    def detect_volatility_regime(self, ticker: str = "SPY") -> Tuple[str, float]:
        """
        Detect current volatility regime and stability.
        
        Returns:
            Tuple of (regime_name, stability_score)
        """
        conn = sqlite3.connect(MARKET_DB_PATH)
        cursor = conn.cursor()
        
        # Get price history
        cursor.execute("""
            SELECT date, close
            FROM prices
            WHERE symbol = ?
            AND date >= date('now', '-90 days')
            ORDER BY date DESC
            LIMIT 90
        """, (ticker,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 60:
            return "neutral", 0.5
        
        # Calculate daily returns and volatility
        closes = [r[1] for r in reversed(rows)]
        returns = [(closes[i] / closes[i-1]) - 1 for i in range(1, len(closes))]
        
        if len(returns) < 30:
            return "neutral", 0.5
        
        # Recent vs historical volatility
        recent_vol = statistics.stdev(returns[-20:]) * (252 ** 0.5)
        hist_vol = statistics.stdev(returns[:-20]) * (252 ** 0.5) if len(returns) > 40 else recent_vol
        
        # Regime classification
        if recent_vol < 0.10:
            regime = "low"
        elif recent_vol < 0.20:
            regime = "neutral"
        elif recent_vol < 0.30:
            regime = "high"
        else:
            regime = "extreme"
        
        # Stability: how consistent is the regime
        vol_changes = [abs(returns[i] - returns[i-1]) for i in range(-20, 0) if i > -len(returns)]
        if vol_changes:
            stability = max(0.0, 1.0 - statistics.mean(vol_changes) / recent_vol if recent_vol > 0 else 0.5)
        else:
            stability = 0.5
        
        return regime, min(1.0, stability)
    
    def calculate_drawdown_contribution(
        self,
        source: str,
        ticker: str = "SPY",
        window_days: int = 30
    ) -> float:
        """Calculate how much signal contributed to recent drawdowns."""
        # Placeholder - would need portfolio-level tracking
        # For now, return neutral value
        return 0.0
    
    def calculate_health_metrics(self, source: str) -> HealthMetrics:
        """Calculate all health metrics for a signal source."""
        now = datetime.now().isoformat()
        
        # Rolling correlations
        corr_30d = self.calculate_rolling_correlation(source, window_days=30)
        corr_60d = self.calculate_rolling_correlation(source, window_days=60)
        corr_90d = self.calculate_rolling_correlation(source, window_days=90)
        
        # Win rates
        win_30d = self.calculate_win_rate(source, window_days=30)
        win_60d = self.calculate_win_rate(source, window_days=60)
        win_90d = self.calculate_win_rate(source, window_days=90)
        
        # Volatility regime
        vol_regime, regime_stability = self.detect_volatility_regime()
        
        # Drawdown contribution
        dd_contrib = self.calculate_drawdown_contribution(source)
        
        return HealthMetrics(
            source=source,
            timestamp=now,
            rolling_correlation_30d=corr_30d,
            rolling_correlation_60d=corr_60d,
            rolling_correlation_90d=corr_90d,
            win_rate_30d=win_30d,
            win_rate_60d=win_60d,
            win_rate_90d=win_90d,
            current_volatility_regime=vol_regime,
            regime_stability=regime_stability,
            drawdown_contribution_30d=dd_contrib,
            samples_30d=30 if corr_30d else 0,
            samples_90d=90 if corr_90d else 0,
        )
    
    def calculate_health_score(self, source: str, metrics: HealthMetrics) -> HealthScore:
        """Calculate composite health score from metrics."""
        now = datetime.now().isoformat()
        
        # Correlation score (primary indicator)
        if metrics.rolling_correlation_30d is not None:
            # Scale correlation: 0 = 0.5, 0.3 = 0.7, 0.5 = 1.0
            corr_score = min(1.0, max(0.0, 0.5 + metrics.rolling_correlation_30d))
        elif metrics.rolling_correlation_60d is not None:
            corr_score = min(1.0, max(0.0, 0.5 + metrics.rolling_correlation_60d * 0.9))
        else:
            corr_score = 0.5  # Neutral if no data
        
        # Accuracy score from win rate
        if metrics.win_rate_30d is not None:
            # 50% = 0.5, 60% = 0.7, 70% = 0.9
            acc_score = min(1.0, max(0.0, metrics.win_rate_30d * 1.5 - 0.25))
        else:
            acc_score = 0.5
        
        # Stability score from regime
        stability_scores = {
            'low': 0.8,
            'neutral': 0.6,
            'high': 0.4,
            'extreme': 0.2
        }
        stab_score = stability_scores.get(metrics.current_volatility_regime, 0.5)
        stab_score = stab_score * 0.7 + metrics.regime_stability * 0.3
        
        # Composite score (weighted average)
        overall = (
            corr_score * 0.40 +
            acc_score * 0.35 +
            stab_score * 0.25
        )
        
        # Determine status
        if overall < HEALTH_THRESHOLD_CRITICAL:
            status = "critical"
            action = "disable"
            weight_mult = 0.0
        elif overall < HEALTH_THRESHOLD_WARNING:
            status = "degraded"
            action = "reduce"
            weight_mult = 0.5
        elif overall >= HEALTH_THRESHOLD_RECOVERY:
            status = "healthy"
            action = "maintain"
            weight_mult = 1.0
        else:
            status = "recovering"
            action = "restore"
            weight_mult = 0.75
        
        # Check historical trend
        trend = "stable"
        days_in_state = 0
        if source in self._score_history:
            history = self._score_history[source][:5]
            if len(history) >= 3:
                recent = [h['overall'] for h in history[:3]]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    trend = "improving"
                elif all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    trend = "degrading"
            days_in_state = len([h for h in history if h.get('status') == status])
        
        return HealthScore(
            source=source,
            timestamp=now,
            correlation_score=round(corr_score, 4),
            accuracy_score=round(acc_score, 4),
            stability_score=round(stab_score, 4),
            overall=round(overall, 4),
            status=status,
            trend=trend,
            days_in_current_state=days_in_state,
            recommended_action=action,
            weight_multiplier=weight_mult,
        )
    
    def get_health_score(self, source: str) -> Optional[HealthScore]:
        """Get health score for a single signal source."""
        if source not in MONITORED_SOURCES:
            return None
        
        metrics = self.calculate_health_metrics(source)
        score = self.calculate_health_score(source, metrics)
        
        # Store in database
        self._store_health(score, metrics)
        
        return score
    
    def get_all_health_scores(self) -> Dict[str, HealthScore]:
        """Get health scores for all monitored sources."""
        scores = {}
        for source in MONITORED_SOURCES:
            score = self.get_health_score(source)
            if score:
                scores[source] = score
        return scores
    
    def _store_health(self, score: HealthScore, metrics: HealthMetrics):
        """Store health score and metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store score
        cursor.execute("""
            INSERT OR REPLACE INTO signal_health 
            (source, timestamp, overall, correlation_score, accuracy_score, 
             stability_score, status, trend, weight_multiplier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            score.source, score.timestamp, score.overall,
            score.correlation_score, score.accuracy_score,
            score.stability_score, score.status, score.trend,
            score.weight_multiplier
        ))
        
        # Store metrics
        cursor.execute("""
            INSERT OR REPLACE INTO signal_health_metrics
            (source, timestamp, rolling_correlation_30d, rolling_correlation_60d,
             rolling_correlation_90d, win_rate_30d, win_rate_60d, win_rate_90d,
             current_volatility_regime, regime_stability, drawdown_contribution_30d, samples_30d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.source, metrics.timestamp,
            metrics.rolling_correlation_30d, metrics.rolling_correlation_60d,
            metrics.rolling_correlation_90d, metrics.win_rate_30d,
            metrics.win_rate_60d, metrics.win_rate_90d,
            metrics.current_volatility_regime, metrics.regime_stability,
            metrics.drawdown_contribution_30d, metrics.samples_30d
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_adjusted_weights(
        self,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate health-adjusted weights from base weights.
        
        Args:
            base_weights: Base weights dict. Uses DEFAULT_BASE_WEIGHTS if None.
        
        Returns:
            Health-adjusted weights (normalized to sum to 1.0)
        """
        if base_weights is None:
            base_weights = DEFAULT_BASE_WEIGHTS
        
        # Get current health scores
        health_scores = self.get_all_health_scores()
        
        # Calculate adjusted weights
        adjusted = {}
        for source, base_weight in base_weights.items():
            if source in health_scores:
                health = health_scores[source]
                adjusted[source] = base_weight * health.weight_multiplier
            else:
                adjusted[source] = base_weight
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def generate_health_report(self) -> HealthReport:
        """Generate complete health report for all sources."""
        now = datetime.now().isoformat()
        
        scores = self.get_all_health_scores()
        metrics = {source: self.calculate_health_metrics(source) 
                   for source in MONITORED_SOURCES}
        
        # Categorize signals
        degraded = [s for s, h in scores.items() if h.status == "degraded"]
        critical = [s for s, h in scores.items() if h.status == "critical"]
        recovering = [s for s, h in scores.items() if h.status == "recovering"]
        
        # Composite health
        if scores:
            composite = statistics.mean([h.overall for h in scores.values()])
        else:
            composite = 0.5
        
        report = HealthReport(
            timestamp=now,
            scores=scores,
            metrics=metrics,
            composite_health=round(composite, 4),
            degraded_signals=degraded,
            critical_signals=critical,
            recovering_signals=recovering,
        )
        
        # Save to history log
        self._save_health_report(report)
        
        return report
    
    def _save_health_report(self, report: HealthReport):
        """Save health report to JSON log."""
        HEALTH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing log
        history = []
        if HEALTH_LOG_PATH.exists():
            try:
                with open(HEALTH_LOG_PATH, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Append new report
        history.append(report.to_dict())
        
        # Keep last 90 days
        cutoff = (datetime.now() - timedelta(days=90)).isoformat()
        history = [h for h in history if h['timestamp'] > cutoff]
        
        # Save
        with open(HEALTH_LOG_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Also save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO health_history (timestamp, report_json)
            VALUES (?, ?)
        """, (report.timestamp, json.dumps(report.to_dict())))
        conn.commit()
        conn.close()
    
    def get_health_history(self, source: Optional[str] = None, days: int = 30) -> List[Dict]:
        """Get historical health scores."""
        if HEALTH_LOG_PATH.exists():
            try:
                with open(HEALTH_LOG_PATH, 'r') as f:
                    history = json.load(f)
                
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                filtered = [h for h in history if h['timestamp'] > cutoff]
                
                if source:
                    for h in filtered:
                        h['source_scores'] = {source: h.get('scores', {}).get(source)}
                
                return filtered
            except:
                return []
        return []


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Signal Health & Decay Monitoring (v2.81)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.signals.health_monitor status
    python -m src.signals.health_monitor health --source tsmom
    python -m src.signals.health_monitor weights
    python -m src.signals.health_monitor history --days 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    subparsers.add_parser('status', help='Show overall health status')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Show health for specific source')
    health_parser.add_argument('--source', type=str, help='Signal source name')
    
    # Weights command
    subparsers.add_parser('weights', help='Show health-adjusted weights')
    
    # History command
    hist_parser = subparsers.add_parser('history', help='Show health history')
    hist_parser.add_argument('--days', type=int, default=30, help='Days of history')
    hist_parser.add_argument('--source', type=str, help='Filter by source')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    monitor = SignalHealthMonitor()
    
    if args.command == 'status':
        report = monitor.generate_health_report()
        
        print("\n" + "=" * 60)
        print("SIGNAL HEALTH STATUS (v2.81)")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Composite Health: {report.composite_health:.2%}")
        print()
        print(f"Total Monitored: {len(report.scores)}")
        print(f"Healthy: {sum(1 for s in report.scores.values() if s.status == 'healthy')}")
        print(f"Degraded: {len(report.degraded_signals)}")
        print(f"Critical: {len(report.critical_signals)}")
        print(f"Recovering: {len(report.recovering_signals)}")
        
        if report.degraded_signals:
            print(f"\n⚠️  DEGRADED SIGNALS: {', '.join(report.degraded_signals)}")
        if report.critical_signals:
            print(f"\n🚫 CRITICAL SIGNALS (disabled): {', '.join(report.critical_signals)}")
        
        print("\n" + "-" * 60)
        print("INDIVIDUAL SCORES:")
        print("-" * 60)
        for source, score in sorted(report.scores.items(), key=lambda x: -x[1].overall):
            status_emoji = {
                'healthy': '✅',
                'recovering': '🔄',
                'degraded': '⚠️',
                'critical': '🚫'
            }.get(score.status, '?')
            print(f"{status_emoji} {source:15s} | Overall: {score.overall:.2f} | "
                  f"Corr: {score.correlation_score:.2f} | Acc: {score.accuracy_score:.2f} | "
                  f"{score.status.upper()} | x{score.weight_multiplier}")
        
        print("\n" + "=" * 60)
    
    elif args.command == 'health':
        if args.source:
            score = monitor.get_health_score(args.source)
            if score:
                print(f"\nHealth Score for {args.source}:")
                print(json.dumps(score.to_dict(), indent=2))
            else:
                print(f"Unknown source: {args.source}")
        else:
            scores = monitor.get_all_health_scores()
            print("\nAll Health Scores:")
            print(json.dumps({k: v.to_dict() for k, v in scores.items()}, indent=2))
    
    elif args.command == 'weights':
        base = DEFAULT_BASE_WEIGHTS
        adjusted = monitor.calculate_adjusted_weights(base)
        
        print("\n" + "=" * 60)
        print("HEALTH-ADJUSTED WEIGHTS")
        print("=" * 60)
        print(f"{'Source':<15} {'Base':>8} {'Adjusted':>10} {'Multiplier':>10}")
        print("-" * 60)
        for source in sorted(base.keys()):
            b = base.get(source, 0)
            a = adjusted.get(source, 0)
            m = a / b if b > 0 else 0
            print(f"{source:<15} {b:>7.2%} {a:>9.2%} {m:>9.1f}x")
        print("-" * 60)
        print(f"{'TOTAL':<15} {sum(base.values()):>7.0%} {sum(adjusted.values()):>9.0%}")
        print("=" * 60)
    
    elif args.command == 'history':
        history = monitor.get_health_history(args.source, args.days)
        print(f"\nHealth History (last {args.days} days):")
        print(f"Entries: {len(history)}")
        if history:
            print(json.dumps(history[-3:], indent=2))


if __name__ == '__main__':
    main()
