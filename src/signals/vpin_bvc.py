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
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from cachetools import TTLCache
from src.paths import sqlite_connect


__all__ = ['BVCBar', 'VPINBucket', 'VPINSignal', 'BVCCalculator', 'VPINEngine', 'RebalanceOptimizer', 'VPINSignalAdapter', 'load_historical_bars', 'backtest_vpin']

logger = logging.getLogger(__name__)


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

    def to_signal_snapshot(self):
        """Convert to canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot

        # Map recommendation to directional value
        rec_map = {"execute": 0.2, "delay": -0.1, "avoid": -0.4}
        value = rec_map.get(self.recommendation, 0.0)
        is_active = self.confidence >= 0.3 and self.recommendation != "execute"

        return SignalSnapshot(
            source="vpin_bvc",
            timestamp=self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            value=value,
            confidence=self.confidence,
            asset_signals={},  # VPIN is execution-timing, not asset-allocation
            regime_fit="all",
            is_active=is_active,
            explanation=f"VPIN: regime={self.regime}, "
                        f"vpin={self.vpin:.4f}, "
                        f"z={self.z_score:.2f}, "
                        f"toxicity={self.toxicity_level:.2f}, "
                        f"rec={self.recommendation}",
            metadata={
                "vpin": self.vpin,
                "vpin_ma": self.vpin_ma,
                "z_score": self.z_score,
                "regime": self.regime,
                "toxicity_level": self.toxicity_level,
                "recommendation": self.recommendation,
            },
        )


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
    
    def get_signal_snapshot(self, tickers=None, date=None):
        """Generate a SignalSnapshot for ensemble voter consumption."""
        from src.signals.signal_snapshot import SignalSnapshot

        signal = self.vpin_engine.get_signal('SPY')
        if signal is None:
            return SignalSnapshot(
                source="vpin_bvc",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                regime_fit="all",
                is_active=False,
                explanation="VPIN: insufficient data",
            )
        return signal.to_signal_snapshot()

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


# TTL cache for OHLCV bar queries — avoids redundant SQLite hits per cron cycle
_BARS_CACHE: TTLCache = TTLCache(maxsize=64, ttl=300)  # 5-min TTL, up to 64 entries
_BARS_CACHE_LOCK = threading.Lock()


def load_historical_bars(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    Load historical OHLCV bars. Tries market.db first, falls back to Yahoo Finance.
    Uses daily bars — sufficient for portfolio-level VPIN estimation.
    Results are TTL-cached for 5 minutes to avoid redundant SQLite queries.
    """
    cache_key = f"{symbol}:{days}"
    with _BARS_CACHE_LOCK:
        if cache_key in _BARS_CACHE:
            return _BARS_CACHE[cache_key]

    # Try market.db first
    from src.paths import MARKET_DB
    db_path = MARKET_DB
    result_df = None
    if db_path.exists():
        with sqlite_connect(str(db_path)) as conn:
            try:
                df = pd.read_sql_query(
                    "SELECT date, open, high, low, close, volume, "
                    "price_semantics, is_adjusted_close_proxy FROM prices "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                    conn, params=(symbol, days),
                )
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    proxy_rows = (
                        df.get('is_adjusted_close_proxy', 0).fillna(0).astype(int).eq(1)
                        | df.get('price_semantics', '').fillna('').eq('adjusted_close_proxy_ohlc')
                    )
                    flat_proxy_rows = (
                        df['open'].eq(df['high'])
                        & df['high'].eq(df['low'])
                        & df['low'].eq(df['close'])
                        & df['volume'].fillna(0).eq(0)
                    )
                    if (proxy_rows | flat_proxy_rows).all():
                        logger.info(
                            "Skipping adjusted-close proxy OHLC rows for %s; falling back to Yahoo",
                            symbol,
                        )
                    else:
                        # Check if OHLC is populated
                        if df['open'].notna().sum() > len(df) * 0.5:
                            for col in ['open', 'high', 'low']:
                                df[col] = df[col].fillna(df['close'])
                            df = df.dropna(subset=['close', 'volume'])
                            df['volume'] = df['volume'].fillna(0)
                            result_df = df[['open', 'high', 'low', 'close', 'volume']]
            except (OSError, sqlite3.Error, KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Failed to fetch OHLCV for %s from DB: %s", symbol, e)

    # Fallback: fetch from Yahoo Finance v8 API
    if result_df is None:
        try:
            import requests
            from datetime import datetime as dt

            period2 = int(dt.now().timestamp())
            period1 = int((dt.now() - timedelta(days=days + 30)).timestamp())
            url = (
                f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
                f"?period1={period1}&period2={period2}&interval=1d"
            )
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]

            df = pd.DataFrame({
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
                'volume': quote.get('volume', []),
            }, index=pd.to_datetime(timestamps, unit='s'))

            df = df.dropna(subset=['close'])
            for col in ['open', 'high', 'low']:
                df[col] = df[col].fillna(df['close'])
            df['volume'] = df['volume'].fillna(0)

            result_df = df.tail(days)
        except (OSError, ConnectionError, TimeoutError, KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.warning("Failed to fetch OHLCV for %s from Yahoo Finance: %s", symbol, e)
            return pd.DataFrame()

    if result_df is None:
        return pd.DataFrame()

    # Cache the result
    with _BARS_CACHE_LOCK:
        _BARS_CACHE[cache_key] = result_df
    return result_df


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
        
        for row in df.itertuples():
            engine.process_bar(
                symbol=symbol,
                timestamp=row.Index if isinstance(row.Index, datetime) else datetime.now(),
                o=getattr(row, 'open', 0),
                h=getattr(row, 'high', 0),
                l=getattr(row, 'low', 0),
                c=getattr(row, 'close', 0),
                v=getattr(row, 'volume', 0)
            )

            vpin = engine.calculate_vpin(symbol)
            if vpin:
                results[symbol]['vpins'].append(vpin)
                results[symbol]['timestamps'].append(row.Index)
    
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
        logger.info(f"Running VPIN backtest for {args.symbols} over {args.days} days...")
        results = backtest_vpin(args.symbols, args.days)
        
        logger.info("\n=== VPIN Statistics ===")
        for symbol, stats in results['statistics'].items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  Mean VPIN: {stats['mean']:.4f}")
            logger.info(f"  Std VPIN: {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"  Buckets: {stats['buckets_completed']}")
    
    elif args.status:
        engine = VPINEngine(symbols=args.symbols)
        adapter = VPINSignalAdapter(engine)
        optimizer = RebalanceOptimizer(engine)
        
        logger.info("\n=== VPIN Status ===")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Symbols: {', '.join(args.symbols)}")
        
        logger.info("\n--- Ensemble Signal ---")
        for symbol in args.symbols:
            signal = adapter.to_ensemble_signal(symbol)
            logger.info(f"\n{symbol}:")
            logger.info(f"  Regime: {signal['regime']}")
            logger.info(f"  Probability: {signal['probability']:.2f}")
            logger.info(f"  Confidence: {signal['confidence']:.2f}")
            raw = signal['raw_data']
            if 'vpin' in raw:
                logger.info(f"  VPIN: {raw['vpin']:.4f}")
                logger.info(f"  Percentile: {raw['vpin_percentile']:.2%}")
        
        logger.info("\n--- Rebalance Timing ---")
        execute, reason, savings = optimizer.should_execute_now('SPY')
        logger.info(f"Execute now: {execute}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Expected savings: {savings:.1f} bps")
        
        logger.info("\n--- Execution Quality Report ---")
        report = optimizer.get_execution_quality_report()
        for symbol, data in report['symbols'].items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  VPIN: {data['vpin']:.4f}")
            logger.info(f"  Regime: {data['regime']}")
            logger.info(f"  Recommendation: {data['recommendation']}")
            logger.info(f"  Expected cost: {data['expected_cost_bps']:.1f} bps")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
