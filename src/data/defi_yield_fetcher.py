#!/usr/bin/env python3
"""
v2.95 DeFi Yield Curve Monitor
Passive monitoring infrastructure for DeFi yields (liquid staking, money markets)
to enable future opportunistic allocation when conditions become favorable.

From research: compound/defi-yield-curve-arbitrage-2026
Direct DeFi allocation not justified at current scale ($100K) due to:
- Risk-adjusted yields inferior to covered call alternatives
- High implementation complexity
- Correlation to SPY of 0.75-0.92 (no diversification benefit)
- Scale economics require $500K+ for positive risk-adjusted contribution

Monitoring enables rapid deployment when:
- DeFi yields exceed Treasuries by >2% sustained
- Portfolio reaches $500K+ scale
- Regulatory clarity improves
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from pathlib import Path
import argparse
import sys

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = ClientTimeout(total=30)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('defi_yield_monitor')

@dataclass
class YieldData:
    """Represents yield data for a DeFi protocol"""
    protocol: str
    asset: str
    yield_apy: float
    tvl_usd: float
    timestamp: str
    source: str
    
@dataclass
class YieldSpread:
    """Represents yield spread vs risk-free rate"""
    protocol: str
    asset: str
    defi_yield: float
    treasury_yield: float
    spread: float
    correlation_30d: Optional[float]
    signal: str  # monitor / consider / allocate
    timestamp: str

class DeFiYieldFetcher:
    """Fetches yield data from DeFi protocols"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_lido_steth(self) -> Optional[YieldData]:
        """Fetch Lido stETH staking yield"""
        try:
            # Lido APR API
            url = "https://stake.lido.fi/api/sma-steth-apr"
            async with self.session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return YieldData(
                        protocol="Lido",
                        asset="stETH",
                        yield_apy=float(data.get('data', {}).get('smaApr', 0)) / 100,
                        tvl_usd=0,  # Will fetch from DeFiLlama
                        timestamp=datetime.utcnow().isoformat(),
                        source="lido"
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch Lido data: {e}")
        return None
    
    async def fetch_jito_sol(self) -> Optional[YieldData]:
        """Fetch JitoSOL staking yield"""
        try:
            # JitoStake API
            url = "https://kobe.mainnet.jito.network/api/v1/stake_pools"
            async with self.session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pools = data.get('stake_pools', [])
                    if pools:
                        jito_pool = next((p for p in pools if 'jito' in p.get('pool_name', '').lower()), pools[0])
                        return YieldData(
                            protocol="Jito",
                            asset="JitoSOL",
                            yield_apy=float(jito_pool.get('apy', 0)) / 100,
                            tvl_usd=float(jito_pool.get('total_deposits', 0)),
                            timestamp=datetime.utcnow().isoformat(),
                            source="jito"
                        )
        except Exception as e:
            logger.warning(f"Failed to fetch Jito data: {e}")
        return None
    
    async def fetch_aave_usdc(self) -> Optional[YieldData]:
        """Fetch Aave V3 USDC lending yield (Ethereum mainnet)"""
        try:
            # Aave Protocol Data API
            url = "https://aave-api-v2.aave.com/data/markets-data"
            async with self.session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    markets = data.get('markets', [])
                    usdc_market = next((m for m in markets if m.get('symbol') == 'USDC' and 
                                       m.get('protocol', {}).get('name') == 'Aave V3'), None)
                    if usdc_market:
                        return YieldData(
                            protocol="Aave",
                            asset="USDC",
                            yield_apy=float(usdc_market.get('liquidityRate', 0)),
                            tvl_usd=float(usdc_market.get('totalLiquidityUSD', 0)),
                            timestamp=datetime.utcnow().isoformat(),
                            source="aave"
                        )
        except Exception as e:
            logger.warning(f"Failed to fetch Aave data: {e}")
        return None
    
    async def fetch_defillama_tvl(self, protocol: str) -> float:
        """Fetch TVL data from DeFiLlama"""
        try:
            slug_map = {
                "Lido": "lido",
                "Jito": "jito", 
                "Aave": "aave"
            }
            slug = slug_map.get(protocol)
            if not slug:
                return 0
            
            url = f"https://api.llama.fi/protocol/{slug}"
            async with self.session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tvl_data = data.get('tvl', [])
                    if tvl_data:
                        return float(tvl_data[-1].get('totalLiquidityUSD', 0))
        except Exception as e:
            logger.warning(f"Failed to fetch DeFiLlama TVL for {protocol}: {e}")
        return 0
    
    async def fetch_all_yields(self) -> List[YieldData]:
        """Fetch yields from all tracked protocols"""
        results = await asyncio.gather(
            self.fetch_lido_steth(),
            self.fetch_jito_sol(),
            self.fetch_aave_usdc()
        )
        
        yields = [r for r in results if r is not None]
        
        # Enrich with TVL data from DeFiLlama
        for y in yields:
            if y.tvl_usd == 0:
                y.tvl_usd = await self.fetch_defillama_tvl(y.protocol)
        
        return yields

class TreasuryYieldFetcher:
    """Fetches risk-free Treasury yields from FRED"""
    
    FRED_API_KEY = "YOUR_FRED_API_KEY"  # Set via environment variable
    
    async def fetch_3m_treasury(self) -> float:
        """Fetch 3-month Treasury bill yield (risk-free rate proxy)"""
        try:
            # Using FRED API for TB3MS
            api_key = self._get_api_key()
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&sort_order=desc&limit=1&api_key={api_key}&file_type=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        obs = data.get('observations', [])
                        if obs:
                            return float(obs[0].get('value', 0)) / 100
        except Exception as e:
            logger.warning(f"Failed to fetch Treasury yield: {e}")
        
        # Fallback: return approximate current rate
        return 0.0525  # 5.25% as fallback
    
    def _get_api_key(self) -> str:
        """Get FRED API key from environment"""
        import os
        key = os.environ.get('FRED_API_KEY', self.FRED_API_KEY)
        if key == "YOUR_FRED_API_KEY":
            logger.warning("FRED API key not set, using fallback yield")
        return key

class DeFiYieldDatabase:
    """SQLite database for storing yield history"""
    
    def __init__(self, db_path: str = "data/defi_yield_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS yields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    yield_apy REAL NOT NULL,
                    tvl_usd REAL,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spreads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    defi_yield REAL NOT NULL,
                    treasury_yield REAL NOT NULL,
                    spread REAL NOT NULL,
                    correlation_30d REAL,
                    signal TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_yields_protocol_ts 
                ON yields(protocol, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spreads_protocol_ts 
                ON spreads(protocol, timestamp)
            """)
            
            conn.commit()
    
    def store_yield(self, data: YieldData):
        """Store yield data point"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO yields (protocol, asset, yield_apy, tvl_usd, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (data.protocol, data.asset, data.yield_apy, data.tvl_usd, 
                  data.timestamp, data.source))
            conn.commit()
    
    def store_spread(self, data: YieldSpread):
        """Store spread analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO spreads (protocol, asset, defi_yield, treasury_yield, 
                                   spread, correlation_30d, signal, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (data.protocol, data.asset, data.defi_yield, data.treasury_yield,
                  data.spread, data.correlation_30d, data.signal, data.timestamp))
            conn.commit()
    
    def get_latest_yields(self, hours: int = 24) -> List[Dict]:
        """Get yields from last N hours"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT protocol, asset, yield_apy, tvl_usd, timestamp
                FROM yields
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_spread_history(self, protocol: str, days: int = 30) -> List[Dict]:
        """Get spread history for a protocol"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM spreads
                WHERE protocol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (protocol, cutoff))
            return [dict(row) for row in cursor.fetchall()]

class DeFiYieldMonitor:
    """Main monitor class orchestrating data fetching and analysis"""
    
    # Alert thresholds from research
    SPREAD_THRESHOLD_ALLOCATE = 0.02  # 2% above Treasuries
    SPREAD_THRESHOLD_CONSIDER = 0.01  # 1% above Treasuries
    CORRELATION_THRESHOLD = 0.60
    TVL_DECLINE_THRESHOLD = -0.20  # 20% decline
    
    def __init__(self, db_path: str = "data/defi_yield_history.db"):
        self.db = DeFiYieldDatabase(db_path)
        self.defi_fetcher = None
        self.treasury_fetcher = TreasuryYieldFetcher()
        self.output_path = Path("data/defi_monitor.json")
    
    async def update(self) -> Dict:
        """Fetch latest data and update database"""
        logger.info("Starting DeFi yield update...")
        
        async with DeFiYieldFetcher() as fetcher:
            # Fetch DeFi yields
            defi_yields = await fetcher.fetch_all_yields()
            
            # Fetch risk-free rate
            treasury_yield = await self.treasury_fetcher.fetch_3m_treasury()
            
            # Store yields
            for y in defi_yields:
                self.db.store_yield(y)
                logger.info(f"Stored: {y.protocol} {y.asset} @ {y.yield_apy*100:.2f}%")
            
            # Calculate spreads and signals
            spreads = []
            for y in defi_yields:
                spread_data = self._calculate_spread(y, treasury_yield)
                self.db.store_spread(spread_data)
                spreads.append(asdict(spread_data))
                logger.info(f"Spread: {y.protocol} = {spread_data.spread*100:.2f}% ({spread_data.signal})")
            
            # Generate status output
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "treasury_yield_3m": treasury_yield,
                "yields": [asdict(y) for y in defi_yields],
                "spreads": spreads,
                "alerts": self._check_alerts(defi_yields, spreads)
            }
            
            # Write to JSON for dashboard
            self._write_status(status)
            
            logger.info("DeFi yield update complete")
            return status
    
    def _calculate_spread(self, yield_data: YieldData, treasury_yield: float) -> YieldSpread:
        """Calculate yield spread and generate signal"""
        spread = yield_data.yield_apy - treasury_yield
        
        # Determine signal based on spread and research criteria
        if spread >= self.SPREAD_THRESHOLD_ALLOCATE:
            signal = "allocate"  # Threshold for active consideration
        elif spread >= self.SPREAD_THRESHOLD_CONSIDER:
            signal = "consider"
        else:
            signal = "monitor"
        
        # TODO: Add correlation data when available
        # For now, use None to indicate we need to fetch correlation
        
        return YieldSpread(
            protocol=yield_data.protocol,
            asset=yield_data.asset,
            defi_yield=yield_data.yield_apy,
            treasury_yield=treasury_yield,
            spread=spread,
            correlation_30d=None,  # To be implemented with price data correlation
            signal=signal,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _check_alerts(self, yields: List[YieldData], spreads: List[Dict]) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        for s in spreads:
            # High spread alert
            if s['spread'] >= self.SPREAD_THRESHOLD_ALLOCATE:
                alerts.append({
                    "type": "high_spread",
                    "protocol": s['protocol'],
                    "message": f"{s['protocol']} spread {s['spread']*100:.2f}% exceeds 2% threshold"
                })
            
            # TVL decline alert (would need historical comparison)
            # This is a placeholder for when we implement TVL tracking
        
        return alerts
    
    def _write_status(self, status: Dict):
        """Write status to JSON file for dashboard"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(status, f, indent=2)
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                return json.load(f)
        return {"error": "No status available - run update first"}
    
    def get_history(self, days: int = 30) -> Dict:
        """Get historical data summary"""
        protocols = ["Lido", "Jito", "Aave"]
        history = {}
        
        for protocol in protocols:
            spreads = self.db.get_spread_history(protocol, days)
            if spreads:
                avg_spread = sum(s['spread'] for s in spreads) / len(spreads)
                history[protocol] = {
                    "data_points": len(spreads),
                    "avg_spread_30d": avg_spread,
                    "latest_spread": spreads[0]['spread'] if spreads else None,
                    "signal": spreads[0]['signal'] if spreads else "monitor"
                }
        
        return {
            "period_days": days,
            "protocols": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_daemon(self, interval_seconds: int = 3600):
        """Run continuous monitoring daemon"""
        logger.info(f"Starting DeFi yield monitor daemon (interval: {interval_seconds}s)")
        
        while True:
            try:
                await self.update()
                logger.info(f"Update complete. Sleeping {interval_seconds}s...")
            except Exception as e:
                logger.error(f"Update failed: {e}")
            
            await asyncio.sleep(interval_seconds)

def main():
    parser = argparse.ArgumentParser(description='DeFi Yield Monitor v2.95')
    parser.add_argument('--update', action='store_true', help='Fetch and store latest yields')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--history', type=int, metavar='DAYS', help='Show history for N days')
    parser.add_argument('--daemon', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=3600, help='Daemon interval in seconds')
    
    args = parser.parse_args()
    
    monitor = DeFiYieldMonitor()
    
    if args.update:
        result = asyncio.run(monitor.update())
        print(json.dumps(result, indent=2))
    elif args.status:
        status = monitor.get_status()
        print(json.dumps(status, indent=2))
    elif args.history:
        history = monitor.get_history(args.history)
        print(json.dumps(history, indent=2))
    elif args.daemon:
        asyncio.run(monitor.run_daemon(args.interval))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
