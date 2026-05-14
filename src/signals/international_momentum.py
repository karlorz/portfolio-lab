"""
International Equity Momentum Signal Generator
Generates momentum-based signals for EFA/EEM overlay strategy
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DB = Path("/root/projects/portfolio-lab/data/market.db")


class SignalType(Enum):
    """International momentum signal types"""
    NEUTRAL = "neutral"
    EFA_LEAD = "efa_lead"  # Developed markets outperforming
    EEM_LEAD = "eem_lead"  # Emerging markets outperforming
    

class ConfidenceLevel(Enum):
    """Signal confidence levels"""
    LOW = "low"      # < 0.5
    MEDIUM = "medium"  # 0.5 - 0.7
    HIGH = "high"    # > 0.7


@dataclass
class InternationalMomentumSignal:
    """Complete momentum signal with allocation recommendation"""
    timestamp: str
    signal_type: str
    confidence: float
    confidence_level: str
    
    # Momentum metrics
    efa_momentum_6m: float
    eem_momentum_6m: float
    spy_momentum_6m: float
    efa_vs_spy: float
    eem_vs_spy: float
    
    # Recommended allocation shifts
    spy_shift: float   # % to reduce SPY by (positive = reduce)
    efa_shift: float   # % to increase EFA by
    eem_shift: float   # % to increase EEM by
    
    # Risk controls
    max_allocation_efa: float  # 5% max
    max_allocation_eem: float  # 3% max
    holding_period_days: int   # 30 min hold
    
    # Metadata
    data_fresh: bool
    vix_filter_active: bool    # Disabled if VIX > 30
    correlation_override: bool  # If correlation > 0.95
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_active(self) -> bool:
        """Check if signal is actionable"""
        return (
            self.signal_type != SignalType.NEUTRAL.value and
            self.confidence >= 0.5 and
            self.data_fresh and
            not self.vix_filter_active and
            not self.correlation_override
        )
    
    def get_allocation_delta(self) -> Dict[str, float]:
        """Get allocation delta for ensemble integration"""
        if not self.is_active():
            return {'SPY': 0.0, 'EFA': 0.0, 'EEM': 0.0}
        
        return {
            'SPY': -self.spy_shift,
            'EFA': self.efa_shift,
            'EEM': self.eem_shift
        }


class InternationalMomentumGenerator:
    """Generates international equity momentum signals"""
    
    # Thresholds
    EFA_THRESHOLD = 0.05  # 5% outperformance required
    EEM_THRESHOLD = 0.08  # 8% outperformance required (higher vol)
    
    # Allocation limits
    MAX_EFA_ALLOCATION = 0.05  # 5% max
    MAX_EEM_ALLOCATION = 0.03  # 3% max
    MIN_HOLDING_DAYS = 30
    
    # Risk filters
    VIX_CUTOFF = 30.0
    CORRELATION_CUTOFF = 0.95
    
    def __init__(self, cache_db: Path = CACHE_DB):
        self.cache_db = cache_db
        self._init_signal_history()
    
    def _init_signal_history(self):
        """Initialize signal history table"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS international_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    efa_momentum_6m REAL,
                    eem_momentum_6m REAL,
                    spy_momentum_6m REAL,
                    allocation_delta_spy REAL,
                    allocation_delta_efa REAL,
                    allocation_delta_eem REAL,
                    is_active INTEGER,
                    data_fresh INTEGER
                )
            """)
            
            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_intl_sig_time 
                ON international_signals(timestamp)
            """)
            conn.commit()
    
    def _get_vix_level(self) -> float:
        """Get current VIX level from cache"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("""
                    SELECT value FROM market_data 
                    WHERE symbol = '^VIX' 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    return float(row[0])
        except Exception as e:
            logger.warning(f"Could not fetch VIX: {e}")
        return 20.0  # Default to normal level
    
    def _get_correlation(self) -> float:
        """Get 30-day EFA-SPY correlation"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                # This would need a correlation table
                # For now, return a placeholder
                cursor = conn.execute("""
                    SELECT correlation_30d FROM correlation_regime 
                    WHERE pair = 'EFA-SPY' 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    return float(row[0])
        except Exception as e:
            logger.warning(f"Could not fetch correlation: {e}")
        return 0.85  # Default normal correlation
    
    def _determine_signal_type(
        self, 
        efa_vs_spy: float, 
        eem_vs_spy: float
    ) -> Tuple[SignalType, float]:
        """Determine signal type and confidence"""
        
        # Check EFA lead
        if efa_vs_spy > self.EFA_THRESHOLD:
            confidence = min(efa_vs_spy / 0.10, 1.0)  # Max at 10% outperformance
            return SignalType.EFA_LEAD, confidence
        
        # Check EEM lead
        if eem_vs_spy > self.EEM_THRESHOLD:
            confidence = min(eem_vs_spy / 0.15, 1.0)  # Max at 15% outperformance
            return SignalType.EEM_LEAD, confidence
        
        # Neutral
        return SignalType.NEUTRAL, 0.0
    
    def _calculate_allocation_shifts(
        self, 
        signal_type: SignalType,
        confidence: float
    ) -> Tuple[float, float, float]:
        """Calculate allocation shifts based on signal"""
        
        if signal_type == SignalType.NEUTRAL:
            return 0.0, 0.0, 0.0
        
        # Scale by confidence
        if signal_type == SignalType.EFA_LEAD:
            shift = self.MAX_EFA_ALLOCATION * confidence
            return shift, shift, 0.0  # Reduce SPY, add EFA
        
        if signal_type == SignalType.EEM_LEAD:
            shift = self.MAX_EEM_ALLOCATION * confidence
            return shift, 0.0, shift  # Reduce SPY, add EEM
        
        return 0.0, 0.0, 0.0
    
    def generate_signal(self, data: Dict) -> InternationalMomentumSignal:
        """Generate momentum signal from fetched data"""
        
        # Extract metrics
        timestamp = data.get('timestamp', datetime.now().isoformat())
        relative = data.get('relative', {})
        data_fresh = data.get('data_fresh', False)
        
        efa_momentum = relative.get('efa_momentum_6m', 0.0)
        eem_momentum = relative.get('eem_momentum_6m', 0.0)
        spy_momentum = relative.get('spy_momentum_6m', 0.0)
        efa_vs_spy = relative.get('efa_vs_spy', 0.0)
        eem_vs_spy = relative.get('eem_vs_spy', 0.0)
        
        # Determine signal
        signal_type, confidence = self._determine_signal_type(efa_vs_spy, eem_vs_spy)
        
        # Calculate allocation shifts
        spy_shift, efa_shift, eem_shift = self._calculate_allocation_shifts(
            signal_type, confidence
        )
        
        # Risk filters
        vix = self._get_vix_level()
        correlation = self._get_correlation()
        
        vix_filter_active = vix > self.VIX_CUTOFF
        correlation_override = correlation > self.CORRELATION_CUTOFF
        
        # Determine confidence level
        if confidence < 0.5:
            confidence_level = ConfidenceLevel.LOW.value
        elif confidence < 0.7:
            confidence_level = ConfidenceLevel.MEDIUM.value
        else:
            confidence_level = ConfidenceLevel.HIGH.value
        
        signal = InternationalMomentumSignal(
            timestamp=timestamp,
            signal_type=signal_type.value,
            confidence=round(confidence, 2),
            confidence_level=confidence_level,
            efa_momentum_6m=round(efa_momentum, 4),
            eem_momentum_6m=round(eem_momentum, 4),
            spy_momentum_6m=round(spy_momentum, 4),
            efa_vs_spy=round(efa_vs_spy, 4),
            eem_vs_spy=round(eem_vs_spy, 4),
            spy_shift=round(spy_shift, 4),
            efa_shift=round(efa_shift, 4),
            eem_shift=round(eem_shift, 4),
            max_allocation_efa=self.MAX_EFA_ALLOCATION,
            max_allocation_eem=self.MAX_EEM_ALLOCATION,
            holding_period_days=self.MIN_HOLDING_DAYS,
            data_fresh=data_fresh,
            vix_filter_active=vix_filter_active,
            correlation_override=correlation_override
        )
        
        # Save to history
        self._save_signal(signal)
        
        return signal
    
    def _save_signal(self, signal: InternationalMomentumSignal):
        """Save signal to history database"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    INSERT INTO international_signals (
                        timestamp, signal_type, confidence,
                        efa_momentum_6m, eem_momentum_6m, spy_momentum_6m,
                        allocation_delta_spy, allocation_delta_efa, allocation_delta_eem,
                        is_active, data_fresh
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.timestamp,
                    signal.signal_type,
                    signal.confidence,
                    signal.efa_momentum_6m,
                    signal.eem_momentum_6m,
                    signal.spy_momentum_6m,
                    signal.spy_shift,
                    signal.efa_shift,
                    signal.eem_shift,
                    1 if signal.is_active() else 0,
                    1 if signal.data_fresh else 0
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def get_signal_history(self, days: int = 90) -> List[Dict]:
        """Get signal history for specified days"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM international_signals 
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            """, (f'-{days} days',))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def get_current_signal(self) -> Optional[InternationalMomentumSignal]:
        """Get most recent signal from database"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM international_signals 
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
            
            if not row:
                return None
            
            columns = [description[0] for description in cursor.description]
            data = dict(zip(columns, row))
            
            return InternationalMomentumSignal(
                timestamp=data['timestamp'],
                signal_type=data['signal_type'],
                confidence=data['confidence'],
                confidence_level='unknown',
                efa_momentum_6m=data['efa_momentum_6m'],
                eem_momentum_6m=data['eem_momentum_6m'],
                spy_momentum_6m=data['spy_momentum_6m'],
                efa_vs_spy=0.0,
                eem_vs_spy=0.0,
                spy_shift=data['allocation_delta_spy'],
                efa_shift=data['allocation_delta_efa'],
                eem_shift=data['allocation_delta_eem'],
                max_allocation_efa=0.05,
                max_allocation_eem=0.03,
                holding_period_days=30,
                data_fresh=bool(data['data_fresh']),
                vix_filter_active=False,
                correlation_override=False
            )
    
    def get_signal_statistics(self, days: int = 90) -> Dict:
        """Calculate signal statistics over period"""
        history = self.get_signal_history(days)
        
        if not history:
            return {'error': 'No signal history available'}
        
        # Calculate statistics
        total_signals = len(history)
        efa_signals = sum(1 for s in history if s['signal_type'] == 'efa_lead')
        eem_signals = sum(1 for s in history if s['signal_type'] == 'eem_lead')
        neutral_signals = sum(1 for s in history if s['signal_type'] == 'neutral')
        active_signals = sum(1 for s in history if s['is_active'])
        
        # Average confidence
        avg_confidence = sum(s['confidence'] for s in history) / total_signals if history else 0
        
        return {
            'period_days': days,
            'total_signals': total_signals,
            'efa_lead_count': efa_signals,
            'eem_lead_count': eem_signals,
            'neutral_count': neutral_signals,
            'active_count': active_signals,
            'activation_rate': round(active_signals / total_signals, 2) if total_signals > 0 else 0,
            'avg_confidence': round(avg_confidence, 2),
            'current_regime': history[0]['signal_type'] if history else 'unknown'
        }


def main():
    """CLI entry point"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='International Momentum Signal Generator')
    parser.add_argument('--generate', action='store_true', help='Generate signal from data')
    parser.add_argument('--history', type=int, metavar='DAYS', help='Show signal history')
    parser.add_argument('--stats', action='store_true', help='Show signal statistics')
    parser.add_argument('--current', action='store_true', help='Show current signal')
    parser.add_argument('--data-file', type=str, help='Path to international_momentum.json')
    
    args = parser.parse_args()
    
    generator = InternationalMomentumGenerator()
    
    if args.generate:
        if not args.data_file:
            # Try default location
            args.data_file = "/root/projects/portfolio-lab/data/international_momentum.json"
        
        try:
            with open(args.data_file, 'r') as f:
                data = json.load(f)
            
            signal = generator.generate_signal(data)
            print(json.dumps(signal.to_dict(), indent=2))
        except FileNotFoundError:
            print(f"Error: Data file not found: {args.data_file}", file=sys.stderr)
            print("Run data fetcher first: python -m src.data.international_fetcher --fetch --save", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error generating signal: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.history:
        history = generator.get_signal_history(args.history)
        print(json.dumps(history, indent=2, default=str))
    
    elif args.stats:
        stats = generator.get_signal_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.current:
        signal = generator.get_current_signal()
        if signal:
            print(json.dumps(signal.to_dict(), indent=2))
        else:
            print('{"error": "No signal found. Generate a signal first."}', file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
