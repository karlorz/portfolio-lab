"""
Crypto Correlation Monitor v2.32
Implements monitoring infrastructure for BTC-SPY correlation tracking.
Based on deep research: crypto-diversification-analysis-v232

Strategy: MONITOR, DO NOT ALLOCATE (until correlation sustains <0.25)
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
import numpy as np

# Configuration
CRYPTO_SYMBOLS = {
    'BTC': 'IBIT',      # BlackRock Bitcoin ETF (preferred vehicle)
    'ETH': 'ETHA',      # BlackRock Ethereum ETF
}

BENCHMARK = 'SPY'
CORRELATION_WINDOW = 30  # 30-day rolling correlation
ALERT_THRESHOLD_LOW = 0.25  # Consider allocation if < 0.25 sustained
ALERT_THRESHOLD_HIGH = 0.50   # Reduce/stop if > 0.50 sustained

DATA_DIR = Path('/root/projects/portfolio-lab/data')
DB_PATH = DATA_DIR / 'crypto_correlation.db'
JSON_PATH = DATA_DIR / 'crypto_monitor.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('crypto_monitor')


@dataclass
class CorrelationMetrics:
    """Correlation metrics for crypto monitoring"""
    timestamp: str
    symbol: str
    benchmark: str
    correlation_30d: float
    correlation_60d: float
    correlation_90d: float
    btc_price: float
    spy_price: float
    btc_volatility_30d: float
    spy_volatility_30d: float
    regime: str  # 'low_corr', 'moderate', 'high_corr'
    allocation_signal: str  # 'monitor', 'consider', 'avoid'
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CryptoCorrelationMonitor:
    """
    Monitors BTC-SPY and ETH-SPY correlation for diversification assessment.
    
    Research finding: Crypto fails as diversifier when correlation > 0.50
    during crisis periods. Only consider allocation if correlation sustains < 0.25.
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for correlation history"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS correlation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                correlation_30d REAL,
                correlation_60d REAL,
                correlation_90d REAL,
                crypto_price REAL,
                benchmark_price REAL,
                crypto_vol_30d REAL,
                benchmark_vol_30d REAL,
                regime TEXT,
                allocation_signal TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON correlation_history(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol 
            ON correlation_history(symbol, benchmark)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    async def fetch_price_data(self, symbol: str) -> Optional[Tuple[List[float], List[str]]]:
        """Fetch historical price data from Yahoo Finance"""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': '1d',
            'range': '6mo',  # 6 months for correlation calculation
            'events': 'history'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Portfolio-Lab/2.32)'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to fetch {symbol}: HTTP {resp.status}")
                        return None
                    
                    data = await resp.json()
                    
                    if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                        logger.warning(f"No data returned for {symbol}")
                        return None
                    
                    result = data['chart']['result'][0]
                    prices = result['indicators']['adjclose'][0]['adjclose']
                    timestamps = result['timestamp']
                    dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
                    
                    return prices, dates
                    
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_correlation(self, prices1: List[float], prices2: List[float], window: int = 30) -> float:
        """Calculate rolling correlation between two price series"""
        if len(prices1) < window or len(prices2) < window:
            return 0.0
        
        # Calculate returns
        returns1 = np.diff(np.log(prices1[-window:]))
        returns2 = np.diff(np.log(prices2[-window:]))
        
        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0
        
        # Ensure equal length
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]
        
        # Calculate correlation
        if np.std(returns1) == 0 or np.std(returns2) == 0:
            return 0.0
        
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_volatility(self, prices: List[float], window: int = 30) -> float:
        """Calculate annualized volatility"""
        if len(prices) < window:
            return 0.0
        
        returns = np.diff(np.log(prices[-window:]))
        if len(returns) < 2:
            return 0.0
        
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return float(annual_vol) if not np.isnan(annual_vol) else 0.0
    
    def determine_regime(self, corr_30d: float, corr_60d: float, corr_90d: float) -> str:
        """Determine correlation regime based on multi-timeframe analysis"""
        avg_corr = np.mean([corr_30d, corr_60d, corr_90d])
        
        if avg_corr < 0.25:
            return 'low_corr'
        elif avg_corr < 0.50:
            return 'moderate'
        else:
            return 'high_corr'
    
    def allocation_signal(self, regime: str, corr_30d: float) -> str:
        """Generate allocation signal based on research criteria"""
        if regime == 'low_corr' and corr_30d < 0.25:
            return 'consider'  # Consider 1-2% allocation
        elif regime == 'high_corr' or corr_30d > 0.50:
            return 'avoid'  # High correlation - no allocation
        else:
            return 'monitor'  # Continue monitoring
    
    async def update_correlation_metrics(self) -> Optional[CorrelationMetrics]:
        """Update correlation metrics for BTC-SPY"""
        # Fetch BTC (via IBIT) and SPY data
        btc_data = await self.fetch_price_data(CRYPTO_SYMBOLS['BTC'])
        spy_data = await self.fetch_price_data(BENCHMARK)
        
        if not btc_data or not spy_data:
            logger.error("Failed to fetch price data")
            return None
        
        btc_prices, btc_dates = btc_data
        spy_prices, spy_dates = spy_data
        
        # Align dates (use common dates)
        common_dates = set(btc_dates) & set(spy_dates)
        if not common_dates:
            logger.error("No common trading dates found")
            return None
        
        # Get aligned price series
        aligned_btc = [p for d, p in zip(btc_dates, btc_prices) if d in common_dates]
        aligned_spy = [p for d, p in zip(spy_dates, spy_prices) if d in common_dates]
        
        if len(aligned_btc) < 90 or len(aligned_spy) < 90:
            logger.error(f"Insufficient aligned data: {len(aligned_btc)} days")
            return None
        
        # Calculate correlations
        corr_30d = self.calculate_correlation(aligned_btc, aligned_spy, 30)
        corr_60d = self.calculate_correlation(aligned_btc, aligned_spy, 60)
        corr_90d = self.calculate_correlation(aligned_btc, aligned_spy, 90)
        
        # Calculate volatilities
        btc_vol = self.calculate_volatility(aligned_btc, 30)
        spy_vol = self.calculate_volatility(aligned_spy, 30)
        
        # Determine regime
        regime = self.determine_regime(corr_30d, corr_60d, corr_90d)
        signal = self.allocation_signal(regime, corr_30d)
        
        metrics = CorrelationMetrics(
            timestamp=datetime.now().isoformat(),
            symbol=CRYPTO_SYMBOLS['BTC'],
            benchmark=BENCHMARK,
            correlation_30d=round(corr_30d, 4),
            correlation_60d=round(corr_60d, 4),
            correlation_90d=round(corr_90d, 4),
            btc_price=round(aligned_btc[-1], 2),
            spy_price=round(aligned_spy[-1], 2),
            btc_volatility_30d=round(btc_vol, 4),
            spy_volatility_30d=round(spy_vol, 4),
            regime=regime,
            allocation_signal=signal
        )
        
        return metrics
    
    def save_metrics(self, metrics: CorrelationMetrics):
        """Save metrics to database and JSON"""
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO correlation_history 
            (timestamp, symbol, benchmark, correlation_30d, correlation_60d, 
             correlation_90d, crypto_price, benchmark_price, crypto_vol_30d, 
             benchmark_vol_30d, regime, allocation_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.symbol, metrics.benchmark,
            metrics.correlation_30d, metrics.correlation_60d, metrics.correlation_90d,
            metrics.btc_price, metrics.spy_price,
            metrics.btc_volatility_30d, metrics.spy_volatility_30d,
            metrics.regime, metrics.allocation_signal
        ))
        
        conn.commit()
        conn.close()
        
        # Save to JSON for dashboard
        json_data = {
            'last_updated': metrics.timestamp,
            'metrics': metrics.to_dict(),
            'recommendation': {
                'current_signal': metrics.allocation_signal,
                'explanation': self._get_signal_explanation(metrics),
                'next_review': (datetime.now() + timedelta(days=7)).isoformat()
            }
        }
        
        with open(JSON_PATH, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Metrics saved: correlation_30d={metrics.correlation_30d:.3f}, regime={metrics.regime}")
    
    def _get_signal_explanation(self, metrics: CorrelationMetrics) -> str:
        """Get human-readable explanation of the signal"""
        if metrics.allocation_signal == 'consider':
            return (f"BTC-SPY correlation is low ({metrics.correlation_30d:.2f}). "
                   "Research suggests 1-2% allocation may provide diversification. "
                   "Monitor for 6+ months before implementation.")
        elif metrics.allocation_signal == 'avoid':
            return (f"BTC-SPY correlation is elevated ({metrics.correlation_30d:.2f}). "
                   "Crypto amplifies rather than hedges equity drawdowns at this level. "
                   "Wait for correlation to drop below 0.25 sustained.")
        else:
            return (f"BTC-SPY correlation is moderate ({metrics.correlation_30d:.2f}). "
                   "Continue monitoring. Consider allocation only if correlation "
                   "sustains below 0.25 for 6+ months.")
    
    def get_historical_summary(self, days: int = 90) -> Dict:
        """Get historical correlation summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, correlation_30d, regime, allocation_signal
            FROM correlation_history
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
        ''', (f'-{days} days',))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'error': 'No historical data available'}
        
        correlations = [r[1] for r in rows if r[1] is not None]
        regimes = [r[2] for r in rows]
        signals = [r[3] for r in rows]
        
        return {
            'period_days': days,
            'data_points': len(rows),
            'avg_correlation_30d': round(np.mean(correlations), 4) if correlations else 0,
            'min_correlation_30d': round(min(correlations), 4) if correlations else 0,
            'max_correlation_30d': round(max(correlations), 4) if correlations else 0,
            'regime_distribution': {
                'low_corr': regimes.count('low_corr'),
                'moderate': regimes.count('moderate'),
                'high_corr': regimes.count('high_corr')
            },
            'signal_distribution': {
                'monitor': signals.count('monitor'),
                'consider': signals.count('consider'),
                'avoid': signals.count('avoid')
            },
            'latest': rows[0][0] if rows else None
        }
    
    async def run_monitor_cycle(self):
        """Run a single monitoring cycle"""
        logger.info("Starting crypto correlation monitor cycle...")
        
        metrics = await self.update_correlation_metrics()
        if metrics:
            self.save_metrics(metrics)
            
            # Log summary
            summary = self.get_historical_summary(30)
            logger.info(f"Historical summary (30d): {summary}")
            
            return metrics
        else:
            logger.error("Failed to update metrics")
            return None


# CLI Interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Correlation Monitor v2.32')
    parser.add_argument('--update', action='store_true', help='Update correlation metrics')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--history', type=int, default=90, help='Days of history to show')
    parser.add_argument('--daemon', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=3600, help='Interval seconds for daemon')
    
    args = parser.parse_args()
    
    monitor = CryptoCorrelationMonitor()
    
    if args.update or (not args.status and not args.history):
        # Run update
        result = asyncio.run(monitor.run_monitor_cycle())
        if result:
            print(f"\n✓ Updated: BTC-SPY correlation = {result.correlation_30d:.3f}")
            print(f"  Regime: {result.regime}")
            print(f"  Signal: {result.allocation_signal}")
            print(f"  BTC 30d vol: {result.btc_volatility_30d:.1%}")
            print(f"  SPY 30d vol: {result.spy_volatility_30d:.1%}")
    
    if args.status:
        # Show status from JSON
        if JSON_PATH.exists():
            with open(JSON_PATH) as f:
                data = json.load(f)
            
            m = data['metrics']
            print("\n=== Crypto Correlation Monitor Status ===")
            print(f"Last Updated: {data['last_updated']}")
            print(f"\nBTC-SPY Correlations:")
            print(f"  30-day: {m['correlation_30d']:.3f}")
            print(f"  60-day: {m['correlation_60d']:.3f}")
            print(f"  90-day: {m['correlation_90d']:.3f}")
            print(f"\nVolatility (30-day annualized):")
            print(f"  BTC:  {m['btc_volatility_30d']:.1%}")
            print(f"  SPY:  {m['spy_volatility_30d']:.1%}")
            print(f"\nCurrent Regime: {m['regime']}")
            print(f"Signal: {m['allocation_signal']}")
            print(f"\nRecommendation:")
            print(f"  {data['recommendation']['explanation']}")
        else:
            print("No status data available. Run with --update first.")
    
    if args.history:
        summary = monitor.get_historical_summary(args.history)
        print(f"\n=== Historical Summary ({args.history} days) ===")
        print(json.dumps(summary, indent=2))
    
    if args.daemon:
        print(f"\nStarting daemon (interval: {args.interval}s)...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                asyncio.run(monitor.run_monitor_cycle())
                print(f"[{datetime.now().isoformat()}] Cycle complete, sleeping...")
                import time
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nDaemon stopped.")
