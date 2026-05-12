"""
Bulk Volume Classification (BVC) VPIN Approximation
v2.65 Phase 1 - Market Microstructure & Flow Toxicity

Uses existing 1-minute bar data to approximate VPIN without tick data.
Based on: Easley, Lopez de Prado, O'Hara (2012) VPIN paper

BVC Method:
- buy_volume = volume * (close - low) / (high - low)
- sell_volume = volume - buy_volume
- vpin_approx = |buy_vol - sell_vol| / (buy_vol + sell_vol)

This provides a zero-cost approximation of order flow toxicity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3


@dataclass
class BVCBar:
    """Single bar with BVC classification"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    vpin_local: float  # Local VPIN for this bar


@dataclass
class VPINBucket:
    """Volume-synchronized VPIN bucket"""
    start_time: datetime
    end_time: datetime
    target_volume: float
    actual_volume: float
    bars: List[BVCBar]
    buy_volume: float
    sell_volume: float
    vpin: float
    complete: bool


@dataclass
class VPINSignal:
    """VPIN signal output for ensemble integration"""
    timestamp: datetime
    vpin: float
    vpin_ma: float  # Moving average
    vpin_std: float  # Rolling standard deviation
    z_score: float  # Normalized VPIN
    percentile: float  # Historical percentile
    regime: str  # 'low', 'normal', 'elevated', 'high'
    confidence: float
    
    # For ensemble integration
    toxicity_level: float  # 0-1 scale
    recommendation: str  # 'execute', 'delay', 'avoid'
    expected_cost_impact: float  # bps estimate


class BVCCalculator:
    """
    Bulk Volume Classification calculator
    
    Approximates buy/sell volume from OHLCV bars using:
    buy_volume = volume * (close - low) / (high - low)
    """
    
    def __init__(self):
        self.bars: List[BVCBar] = []
    
    def classify_bar(self, timestamp: datetime, o: float, h: float, 
                     l: float, c: float, v: float) -> BVCBar:
        """Classify a single bar using BVC"""
        if h == l:  # Avoid division by zero
            buy_volume = v * 0.5
        else:
            buy_volume = v * (c - l) / (h - l)
        
        sell_volume = v - buy_volume
        
        # Local VPIN for this bar
        if v > 0:
            vpin_local = abs(buy_volume - sell_volume) / v
        else:
            vpin_local = 0.0
        
        return BVCBar(
            timestamp=timestamp,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            vpin_local=vpin_local
        )
    
    def add_bar(self, bar: BVCBar) -> None:
        """Add bar to history"""
        self.bars.append(bar)
    
    def get_buy_sell_imbalance(self, window: int = 20) -> Tuple[float, float, float]:
        """
        Calculate buy/sell imbalance over window
        Returns: (buy_volume, sell_volume, imbalance_ratio)
        """
        if len(self.bars) < window:
            window = len(self.bars)
        
        recent = self.bars[-window:]
        total_buy = sum(b.buy_volume for b in recent)
        total_sell = sum(b.sell_volume for b in recent)
        total = total_buy + total_sell
        
        if total > 0:
            imbalance = abs(total_buy - total_sell) / total
        else:
            imbalance = 0.0
        
        return total_buy, total_sell, imbalance


class VPINEngine:
    """
    VPIN (Volume-Synchronized Probability of Informed Trading) Engine
    
    Implements volume-time buckets for VPIN calculation.
    Buckets are filled until target volume is reached.
    """
    
    def __init__(self, 
                 volume_bucket_size: float = 100000,  # Shares per bucket
                 vpin_window: int = 50,  # Number of buckets for VPIN
                 symbols: List[str] = None):
        self.volume_bucket_size = volume_bucket_size
        self.vpin_window = vpin_window
        self.symbols = symbols or ['SPY', 'QQQ', 'TLT', 'GLD']
        
        # Per-symbol state
        self.current_buckets: Dict[str, VPINBucket] = {}
        self.completed_buckets: Dict[str, List[VPINBucket]] = {
            s: [] for s in self.symbols
        }
        self.bvc_calcs: Dict[str, BVCCalculator] = {
            s: BVCCalculator() for s in self.symbols
        }
        
        # Historical VPIN for normalization
        self.vpin_history: Dict[str, List[float]] = {
            s: [] for s in self.symbols
        }
    
    def process_bar(self, symbol: str, timestamp: datetime,
                    o: float, h: float, l: float, c: float, 
                    v: float) -> Optional[VPINBucket]:
        """
        Process a new bar and update VPIN buckets
        Returns completed bucket if one finished
        """
        # Classify bar with BVC
        bvc = self.bvc_calcs[symbol]
        bar = bvc.classify_bar(timestamp, o, h, l, c, v)
        bvc.add_bar(bar)
        
        # Get or create current bucket
        if symbol not in self.current_buckets:
            self.current_buckets[symbol] = VPINBucket(
                start_time=timestamp,
                end_time=timestamp,
                target_volume=self.volume_bucket_size,
                actual_volume=0,
                bars=[],
                buy_volume=0,
                sell_volume=0,
                vpin=0,
                complete=False
            )
        
        bucket = self.current_buckets[symbol]
        bucket.bars.append(bar)
        bucket.actual_volume += v
        bucket.buy_volume += bar.buy_volume
        bucket.sell_volume += bar.sell_volume
        bucket.end_time = timestamp
        
        # Check if bucket is complete
        completed = None
        if bucket.actual_volume >= bucket.target_volume:
            bucket.complete = True
            
            # Calculate VPIN for this bucket
            if bucket.actual_volume > 0:
                bucket.vpin = abs(bucket.buy_volume - bucket.sell_volume) / bucket.actual_volume
            else:
                bucket.vpin = 0.0
            
            # Store completed bucket
            self.completed_buckets[symbol].append(bucket)
            completed = bucket
            
            # Trim history
            if len(self.completed_buckets[symbol]) > self.vpin_window * 2:
                self.completed_buckets[symbol] = self.completed_buckets[symbol][-self.vpin_window * 2:]
            
            # Start new bucket
            self.current_buckets[symbol] = VPINBucket(
                start_time=timestamp,
                end_time=timestamp,
                target_volume=self.volume_bucket_size,
                actual_volume=0,
                bars=[],
                buy_volume=0,
                sell_volume=0,
                vpin=0,
                complete=False
            )
        
        return completed
    
    def calculate_vpin(self, symbol: str) -> Optional[float]:
        """Calculate current VPIN from completed buckets"""
        buckets = self.completed_buckets[symbol]
        if len(buckets) < self.vpin_window:
            return None
        
        # VPIN is average of recent bucket VPINs
        recent = buckets[-self.vpin_window:]
        vpin = np.mean([b.vpin for b in recent])
        
        # Store in history
        self.vpin_history[symbol].append(vpin)
        if len(self.vpin_history[symbol]) > 500:
            self.vpin_history[symbol] = self.vpin_history[symbol][-500:]
        
        return vpin
    
    def get_signal(self, symbol: str) -> Optional[VPINSignal]:
        """Generate VPIN signal with full metrics"""
        vpin = self.calculate_vpin(symbol)
        if vpin is None:
            return None
        
        history = self.vpin_history[symbol]
        if len(history) < 50:
            return None
        
        # Calculate statistics
        vpin_ma = np.mean(history[-50:])
        vpin_std = np.std(history[-50:]) if len(history) >= 50 else 0.01
        
        # Z-score
        if vpin_std > 0:
            z_score = (vpin - vpin_ma) / vpin_std
        else:
            z_score = 0
        
        # Percentile
        percentile = sum(1 for v in history if v < vpin) / len(history)
        
        # Regime classification
        if percentile < 0.25:
            regime = 'low'
            confidence = 0.6
        elif percentile < 0.50:
            regime = 'normal'
            confidence = 0.7
        elif percentile < 0.75:
            regime = 'elevated'
            confidence = 0.7
        else:
            regime = 'high'
            confidence = 0.6
        
        # Toxicity level (0-1)
        toxicity_level = percentile
        
        # Recommendation for execution
        if percentile < 0.30:
            recommendation = 'execute'  # Low toxicity, good time to trade
            expected_cost = -3.0  # Save ~3bps
        elif percentile < 0.70:
            recommendation = 'delay'  # Moderate toxicity, wait if possible
            expected_cost = 0.0
        else:
            recommendation = 'avoid'  # High toxicity, avoid if possible
            expected_cost = 5.0  # Pay ~5bps more
        
        return VPINSignal(
            timestamp=datetime.now(),
            vpin=vpin,
            vpin_ma=vpin_ma,
            vpin_std=vpin_std,
            z_score=z_score,
            percentile=percentile,
            regime=regime,
            confidence=confidence,
            toxicity_level=toxicity_level,
            recommendation=recommendation,
            expected_cost_impact=expected_cost
        )


class RebalanceOptimizer:
    """
    Rebalancing timing optimizer based on VPIN signals
    
    Uses flow toxicity to recommend optimal execution windows.
    """
    
    def __init__(self, vpin_engine: VPINEngine, 
                 max_delay_minutes: int = 60):
        self.vpin_engine = vpin_engine
        self.max_delay_minutes = max_delay_minutes
        self.pending_rebalances: List[Dict] = []
    
    def should_execute_now(self, symbol: str = 'SPY') -> Tuple[bool, str, float]:
        """
        Determine if rebalancing should execute now or wait
        Returns: (execute_now, reason, expected_savings_bps)
        """
        signal = self.vpin_engine.get_signal(symbol)
        
        if signal is None:
            return True, "insufficient_data", 0.0
        
        if signal.recommendation == 'execute':
            return True, f"low_toxicity (vpin={signal.vpin:.3f}, p={signal.percentile:.2f})", \
                   abs(signal.expected_cost_impact)
        
        if signal.recommendation == 'avoid':
            return False, f"high_toxicity (vpin={signal.vpin:.3f}, p={signal.percentile:.2f})", \
                   abs(signal.expected_cost_impact)
        
        return True, f"moderate_toxicity (vpin={signal.vpin:.3f})", 0.0
    
    def get_execution_quality_report(self) -> Dict[str, Any]:
        """Generate execution quality metrics"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in self.vpin_engine.symbols:
            signal = self.vpin_engine.get_signal(symbol)
            if signal:
                report['symbols'][symbol] = {
                    'vpin': signal.vpin,
                    'regime': signal.regime,
                    'recommendation': signal.recommendation,
                    'expected_cost_bps': signal.expected_cost_impact,
                    'toxicity_level': signal.toxicity_level
                }
        
        return report


class VPINSignalAdapter:
    """
    Adapter to integrate VPIN signals into ensemble voter
    
    Maps VPIN signals to unified regime format for portfolio decisions.
    """
    
    # VPIN thresholds for risk-off triggers
    HIGH_VPIN_THRESHOLD = 0.75  # 75th percentile
    CRISIS_VPIN_THRESHOLD = 0.90  # 90th percentile
    
    def __init__(self, vpin_engine: VPINEngine):
        self.vpin_engine = vpin_engine
    
    def to_ensemble_signal(self, symbol: str = 'SPY') -> Dict[str, Any]:
        """Convert VPIN signal to ensemble-compatible format"""
        signal = self.vpin_engine.get_signal(symbol)
        
        if signal is None:
            return {
                'source': 'vpin',
                'regime': 'neutral',
                'probability': 0.5,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'raw_data': {'status': 'insufficient_data'}
            }
        
        # Map VPIN to regime
        if signal.percentile >= self.CRISIS_VPIN_THRESHOLD:
            regime = 'crisis'
            prob = 0.8
        elif signal.percentile >= self.HIGH_VPIN_THRESHOLD:
            regime = 'bear'  # High toxicity = risk-off
            prob = 0.7
        elif signal.percentile <= 0.25:
            regime = 'bull'  # Low toxicity = risk-on
            prob = 0.6
        else:
            regime = 'neutral'
            prob = 0.5
        
        return {
            'source': 'vpin',
            'regime': regime,
            'probability': prob,
            'confidence': signal.confidence,
            'timestamp': datetime.now().isoformat(),
            'raw_data': {
                'vpin': signal.vpin,
                'vpin_percentile': signal.percentile,
                'z_score': signal.z_score,
                'recommendation': signal.recommendation,
                'expected_cost_bps': signal.expected_cost_impact
            }
        }
    
    def get_rebalance_timing_signal(self) -> Dict[str, Any]:
        """Get signal specifically for rebalancing timing optimization"""
        optimizer = RebalanceOptimizer(self.vpin_engine)
        execute, reason, savings = optimizer.should_execute_now('SPY')
        
        return {
            'source': 'vpin_rebalance',
            'execute_now': execute,
            'reason': reason,
            'expected_savings_bps': savings,
            'timestamp': datetime.now().isoformat()
        }


def load_historical_bars(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    Load historical 1-minute bars from data source
    Placeholder - integrate with actual data pipeline
    """
    # This would connect to your existing data pipeline
    # For now, return empty DataFrame as placeholder
    return pd.DataFrame()


def backtest_vpin(symbols: List[str], days: int = 30) -> Dict[str, Any]:
    """
    Backtest VPIN calculation on historical data
    Returns performance metrics
    """
    engine = VPINEngine(symbols=symbols)
    results = {s: {'vpins': [], 'timestamps': []} for s in symbols}
    
    # Load and process historical bars
    for symbol in symbols:
        df = load_historical_bars(symbol, days)
        
        if len(df) == 0:
            continue
        
        for idx, row in df.iterrows():
            engine.process_bar(
                symbol=symbol,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                o=row.get('open', 0),
                h=row.get('high', 0),
                l=row.get('low', 0),
                c=row.get('close', 0),
                v=row.get('volume', 0)
            )
            
            vpin = engine.calculate_vpin(symbol)
            if vpin:
                results[symbol]['vpins'].append(vpin)
                results[symbol]['timestamps'].append(idx)
    
    # Calculate statistics
    stats = {}
    for symbol in symbols:
        vpins = results[symbol]['vpins']
        if len(vpins) > 0:
            stats[symbol] = {
                'mean': np.mean(vpins),
                'std': np.std(vpins),
                'min': np.min(vpins),
                'max': np.max(vpins),
                'buckets_completed': len(vpins)
            }
    
    return {'results': results, 'statistics': stats}


def cli():
    """Command-line interface for VPIN engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VPIN BVC Prototype')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'TLT', 'GLD'])
    parser.add_argument('--days', type=int, default=30)
    
    args = parser.parse_args()
    
    if args.backtest:
        print(f"Running VPIN backtest for {args.symbols} over {args.days} days...")
        results = backtest_vpin(args.symbols, args.days)
        
        print("\n=== VPIN Statistics ===")
        for symbol, stats in results['statistics'].items():
            print(f"\n{symbol}:")
            print(f"  Mean VPIN: {stats['mean']:.4f}")
            print(f"  Std VPIN: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Buckets: {stats['buckets_completed']}")
    
    elif args.status:
        engine = VPINEngine(symbols=args.symbols)
        adapter = VPINSignalAdapter(engine)
        optimizer = RebalanceOptimizer(engine)
        
        print("\n=== VPIN Status ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Symbols: {', '.join(args.symbols)}")
        
        print("\n--- Ensemble Signal ---")
        for symbol in args.symbols:
            signal = adapter.to_ensemble_signal(symbol)
            print(f"\n{symbol}:")
            print(f"  Regime: {signal['regime']}")
            print(f"  Probability: {signal['probability']:.2f}")
            print(f"  Confidence: {signal['confidence']:.2f}")
            raw = signal['raw_data']
            if 'vpin' in raw:
                print(f"  VPIN: {raw['vpin']:.4f}")
                print(f"  Percentile: {raw['vpin_percentile']:.2%}")
        
        print("\n--- Rebalance Timing ---")
        execute, reason, savings = optimizer.should_execute_now('SPY')
        print(f"Execute now: {execute}")
        print(f"Reason: {reason}")
        print(f"Expected savings: {savings:.1f} bps")
        
        print("\n--- Execution Quality Report ---")
        report = optimizer.get_execution_quality_report()
        for symbol, data in report['symbols'].items():
            print(f"\n{symbol}:")
            print(f"  VPIN: {data['vpin']:.4f}")
            print(f"  Regime: {data['regime']}")
            print(f"  Recommendation: {data['recommendation']}")
            print(f"  Expected cost: {data['expected_cost_bps']:.1f} bps")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
