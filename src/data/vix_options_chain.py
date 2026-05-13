"""
VIX Options Data Pipeline - Phase 1 Implementation
Fetches and stores VIX options chain data for tail hedge insurance overlay.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import aiohttp
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VIXOption:
    """Represents a single VIX option contract."""
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def premium(self) -> float:
        """Cost per contract (100x multiplier)."""
        return self.mid_price * 100


@dataclass
class VIXOptionsChain:
    """Complete options chain for a given date."""
    timestamp: str
    vix_spot: float
    vix_9day: Optional[float]  # Short-term VIX
    vix_3m: Optional[float]   # 3-month VIX
    term_structure: Dict[str, float]  # By expiration date
    calls: List[VIXOption]
    puts: List[VIXOption]
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'vix_spot': self.vix_spot,
            'vix_9day': self.vix_9day,
            'vix_3m': self.vix_3m,
            'term_structure': self.term_structure,
            'calls': [asdict(c) for c in self.calls],
            'puts': [asdict(p) for p in self.puts]
        }


class VIXDataPipeline:
    """
    VIX Options Data Pipeline
    
    Fetches VIX spot, futures, and options chain data.
    Stores historical data for backtesting and signal generation.
    """
    
    DB_PATH = Path("/root/projects/portfolio-lab/data/vix_options.db")
    DATA_DIR = Path("/root/projects/portfolio-lab/data/signals")
    
    # VIX futures tickers for term structure
    VIX_FUTURES = {
        'front_month': 'VIX=F',
        'second_month': 'VX=F',  # 2nd month
    }
    
    def __init__(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # VIX spot and term structure history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vix_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vix_spot REAL NOT NULL,
                vix_9day REAL,
                vix_3m REAL,
                front_month_future REAL,
                second_month_future REAL,
                contango REAL,  -- (f2-f1)/f1
                UNIQUE(timestamp)
            )
        """)
        
        # Options chain snapshot
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                expiration_date TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                bid REAL,
                ask REAL,
                last_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_vol REAL,
                delta REAL,
                gamma REAL,
                theta REAL
            )
        """)
        
        # Selected signals (30-delta calls for insurance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insurance_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                expiration_date TEXT NOT NULL,
                days_to_expiration INTEGER,
                strike REAL NOT NULL,
                delta REAL NOT NULL,
                premium REAL NOT NULL,
                breakeven_vix REAL NOT NULL,
                max_gain_scenario_40 REAL,  -- Gain if VIX hits 40
                max_gain_scenario_60 REAL,  -- Gain if VIX hits 60
                selected_for_trade BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Historical backtest data (for 2018-2024 analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_vix_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                event_name TEXT,
                vix_before REAL,
                vix_peak REAL,
                spy_drop_percent REAL,
                recovery_days INTEGER,
                max_option_gain REAL  -- Theoretical gain for 30d call
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("VIX database initialized")
    
    async def fetch_vix_spot(self) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Fetch current VIX spot price and short-term variants.
        
        Returns:
            (vix_spot, vix_9day, vix_3m)
        """
        try:
            # Primary VIX index
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            vix_spot = float(vix_data['Close'].iloc[-1])
            
            # VIX9D (9-day expected volatility)
            vix9d = None
            try:
                vix9d_ticker = yf.Ticker("^VIX9D")
                vix9d_data = vix9d_ticker.history(period="1d")
                vix9d = float(vix9d_data['Close'].iloc[-1])
            except Exception as e:
                logger.debug(f"VIX9D fetch failed: {e}")
            
            # VIX3M (3-month expected volatility)
            vix3m = None
            try:
                vix3m_ticker = yf.Ticker("^VIX3M")
                vix3m_data = vix3m_ticker.history(period="1d")
                vix3m = float(vix3m_data['Close'].iloc[-1])
            except Exception as e:
                logger.debug(f"VIX3M fetch failed: {e}")
            
            return vix_spot, vix9d, vix3m
            
        except Exception as e:
            logger.error(f"Failed to fetch VIX spot: {e}")
            raise
    
    async def fetch_vix_futures(self) -> Dict[str, Optional[float]]:
        """Fetch VIX futures prices for term structure analysis."""
        futures = {}
        
        for name, ticker in self.VIX_FUTURES.items():
            try:
                fut = yf.Ticker(ticker)
                data = fut.history(period="1d")
                futures[name] = float(data['Close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Failed to fetch {name} future: {e}")
                futures[name] = None
        
        return futures
    
    async def fetch_options_chain_yahoo(self) -> Optional[VIXOptionsChain]:
        """
        Fetch VIX options chain from Yahoo Finance.
        
        Note: VIX options are on VIX futures, not spot VIX.
        Returns options on front-month VIX futures.
        """
        try:
            # VIX options trade on futures
            ticker = yf.Ticker("^VIX")
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                logger.warning("No options expirations found for VIX")
                return None
            
            vix_spot, vix_9d, vix_3m = await self.fetch_vix_spot()
            futures = await self.fetch_vix_futures()
            
            all_calls = []
            all_puts = []
            term_structure = {}
            
            # Fetch up to 4 expiration months
            for exp_date in expirations[:4]:
                try:
                    chain = ticker.option_chain(exp_date)
                    
                    # Extract calls
                    for _, row in chain.calls.iterrows():
                        option = VIXOption(
                            strike=float(row['strike']),
                            expiration=exp_date,
                            option_type='call',
                            bid=float(row.get('bid', 0)),
                            ask=float(row.get('ask', 0)),
                            last_price=float(row.get('lastPrice', 0)),
                            volume=int(row.get('volume', 0)),
                            open_interest=int(row.get('openInterest', 0)),
                            implied_vol=float(row.get('impliedVolatility', 0)) * 100,
                            delta=float(row.get('delta', 0)) if 'delta' in row else None,
                            gamma=float(row.get('gamma', 0)) if 'gamma' in row else None,
                            theta=float(row.get('theta', 0)) if 'theta' in row else None
                        )
                        all_calls.append(option)
                    
                    # Calculate ATM implied vol for term structure
                    atm_calls = [c for c in all_calls[-len(chain.calls):] 
                                if abs(c.strike - vix_spot) < 2]
                    if atm_calls:
                        avg_iv = sum(c.implied_vol for c in atm_calls) / len(atm_calls)
                        term_structure[exp_date] = avg_iv
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch chain for {exp_date}: {e}")
            
            return VIXOptionsChain(
                timestamp=datetime.now().isoformat(),
                vix_spot=vix_spot,
                vix_9day=vix_9d,
                vix_3m=vix_3m,
                term_structure=term_structure,
                calls=all_calls,
                puts=all_puts
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch options chain: {e}")
            return None
    
    def calculate_delta_approx(self, strike: float, vix_spot: float, 
                               days_to_exp: int, implied_vol: float) -> float:
        """
        Approximate delta calculation for VIX calls.
        Simplified Black-Scholes-like calculation.
        """
        import math
        
        if days_to_exp <= 0 or implied_vol <= 0:
            return 0.0
        
        # Time to expiration in years
        t = days_to_exp / 365.0
        
        # Simplified delta approximation
        # Delta ≈ N(d1) for calls
        # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
        
        # For VIX options, we use log moneyness
        moneyness = math.log(vix_spot / strike)
        vol_term = implied_vol / 100.0 * math.sqrt(t)
        
        if vol_term == 0:
            return 1.0 if vix_spot > strike else 0.0
        
        d1 = moneyness / vol_term + 0.5 * vol_term
        
        # Approximate N(d1) using error function
        delta = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        
        return max(0.0, min(1.0, delta))
    
    async def select_insurance_candidates(self, chain: VIXOptionsChain) -> List[Dict]:
        """
        Select 30-delta call options suitable for tail hedge insurance.
        
        Criteria:
        - 30-delta calls (or closest available)
        - 45-75 DTE (avoid expiration week gamma risk)
        - Target strike: VIX spot + 3-5 points
        """
        candidates = []
        
        target_delta = 0.30
        vix_spot = chain.vix_spot
        
        # Filter calls by expiration (45-75 DTE)
        today = datetime.now().date()
        
        for call in chain.calls:
            try:
                exp_date = datetime.strptime(call.expiration, '%Y-%m-%d').date()
                days_to_exp = (exp_date - today).days
                
                # Skip if outside optimal DTE range
                if days_to_exp < 45 or days_to_exp > 75:
                    continue
                
                # Calculate or use provided delta
                delta = call.delta
                if delta is None:
                    delta = self.calculate_delta_approx(
                        call.strike, vix_spot, days_to_exp, call.implied_vol
                    )
                
                # Check if this is approximately 30-delta (20-40 range acceptable)
                if 0.20 <= delta <= 0.40:
                    # Calculate breakeven
                    breakeven = call.strike + call.mid_price
                    
                    # Calculate potential gains at various VIX levels
                    # Simplified: intrinsic value at expiration
                    gain_at_40 = max(0, 40 - call.strike) - call.mid_price
                    gain_at_60 = max(0, 60 - call.strike) - call.mid_price
                    
                    candidate = {
                        'timestamp': chain.timestamp,
                        'expiration_date': call.expiration,
                        'days_to_expiration': days_to_exp,
                        'strike': call.strike,
                        'delta': round(delta, 3),
                        'premium': round(call.premium, 2),
                        'breakeven_vix': round(breakeven, 2),
                        'mid_price': round(call.mid_price, 2),
                        'implied_vol': round(call.implied_vol, 1),
                        'volume': call.volume,
                        'open_interest': call.open_interest,
                        'max_gain_scenario_40': round(gain_at_40, 2) if gain_at_40 > 0 else 0,
                        'max_gain_scenario_60': round(gain_at_60, 2) if gain_at_60 > 0 else 0,
                        'delta_distance': abs(delta - target_delta)
                    }
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.debug(f"Error processing call option: {e}")
                continue
        
        # Sort by closest to 30-delta
        candidates.sort(key=lambda x: x['delta_distance'])
        
        return candidates[:5]  # Return top 5 candidates
    
    async def update(self):
        """
        Main update routine - fetch all data and store in database.
        """
        logger.info("Starting VIX data pipeline update...")
        
        # Fetch options chain
        chain = await self.fetch_options_chain_yahoo()
        
        if not chain:
            logger.error("Failed to fetch options chain")
            return False
        
        # Store in database
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # Store VIX spot and term structure
        futures = await self.fetch_vix_futures()
        front = futures.get('front_month')
        second = futures.get('second_month')
        contango = ((second - front) / front * 100) if front and second and front > 0 else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO vix_history 
            (timestamp, vix_spot, vix_9day, vix_3m, front_month_future, second_month_future, contango)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chain.timestamp,
            chain.vix_spot,
            chain.vix_9day,
            chain.vix_3m,
            front,
            second,
            contango
        ))
        
        # Store options chain
        for call in chain.calls:
            cursor.execute("""
                INSERT INTO options_chain 
                (timestamp, expiration_date, strike, option_type, bid, ask, last_price,
                 volume, open_interest, implied_vol, delta, gamma, theta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chain.timestamp, call.expiration, call.strike, call.option_type,
                call.bid, call.ask, call.last_price, call.volume, call.open_interest,
                call.implied_vol, call.delta, call.gamma, call.theta
            ))
        
        # Get and store insurance candidates
        candidates = await self.select_insurance_candidates(chain)
        for c in candidates:
            cursor.execute("""
                INSERT INTO insurance_candidates 
                (timestamp, expiration_date, days_to_expiration, strike, delta, premium,
                 breakeven_vix, max_gain_scenario_40, max_gain_scenario_60)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c['timestamp'], c['expiration_date'], c['days_to_expiration'],
                c['strike'], c['delta'], c['premium'], c['breakeven_vix'],
                c['max_gain_scenario_40'], c['max_gain_scenario_60']
            ))
        
        conn.commit()
        conn.close()
        
        # Export JSON for signal generator
        await self._export_json(chain, candidates)
        
        logger.info(f"VIX update complete - Spot: {chain.vix_spot:.2f}, Candidates: {len(candidates)}")
        return True
    
    async def _export_json(self, chain: VIXOptionsChain, candidates: List[Dict]):
        """Export current data to JSON for signal generator consumption."""
        output = {
            'timestamp': chain.timestamp,
            'vix_spot': chain.vix_spot,
            'vix_9day': chain.vix_9day,
            'vix_3m': chain.vix_3m,
            'term_structure': chain.term_structure,
            'insurance_candidates': candidates,
            'data_source': 'yahoo_finance',
            'update_frequency': 'daily'
        }
        
        output_path = self.DATA_DIR / 'vix_pipeline.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported VIX data to {output_path}")
    
    def get_latest_candidates(self) -> List[Dict]:
        """Retrieve latest insurance candidates from database."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM insurance_candidates 
            WHERE timestamp = (SELECT MAX(timestamp) FROM insurance_candidates)
            ORDER BY delta_distance ASC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_historical_context(self, days: int = 30) -> Dict:
        """Get recent VIX history for signal generation context."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM vix_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'available_history_days': 0}
        
        vix_values = [r['vix_spot'] for r in rows]
        
        return {
            'available_history_days': len(rows),
            'vix_current': vix_values[0],
            'vix_30d_avg': sum(vix_values) / len(vix_values),
            'vix_30d_min': min(vix_values),
            'vix_30d_max': max(vix_values),
            'vix_percentile': sum(1 for v in vix_values if v < vix_values[0]) / len(vix_values)
        }


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VIX Options Data Pipeline')
    parser.add_argument('--update', action='store_true', help='Fetch and store latest data')
    parser.add_argument('--candidates', action='store_true', help='Show insurance candidates')
    parser.add_argument('--status', action='store_true', help='Show current VIX status')
    
    args = parser.parse_args()
    
    pipeline = VIXDataPipeline()
    
    if args.update:
        success = await pipeline.update()
        exit(0 if success else 1)
    
    elif args.candidates:
        candidates = pipeline.get_latest_candidates()
        print("\n=== VIX Insurance Candidates ===\n")
        for c in candidates:
            print(f"Strike: ${c['strike']:.1f} | Exp: {c['expiration_date']} ({c['days_to_expiration']}d)")
            print(f"  Delta: {c['delta']:.2f} | Premium: ${c['premium']:.0f}")
            print(f"  Breakeven VIX: {c['breakeven_vix']:.1f} | IV: {c['implied_vol']:.1f}%")
            print(f"  Gain if VIX=40: ${c['max_gain_scenario_40']:.0f} | VIX=60: ${c['max_gain_scenario_60']:.0f}")
            print()
    
    elif args.status:
        spot, vix9d, vix3m = await pipeline.fetch_vix_spot()
        context = pipeline.get_historical_context()
        
        print("\n=== VIX Status ===\n")
        print(f"VIX Spot: {spot:.2f}")
        if vix9d:
            print(f"VIX 9-day: {vix9d:.2f}")
        if vix3m:
            print(f"VIX 3-month: {vix3m:.2f}")
        print(f"\n30-day avg: {context.get('vix_30d_avg', 'N/A')}")
        print(f"30-day range: {context.get('vix_30d_min', 'N/A')} - {context.get('vix_30d_max', 'N/A')}")
        
        # Insurance signal
        if spot < 16:
            print("\n🟢 CHEAP VOL - Full insurance allocation (1%)")
        elif spot < 20:
            print("\n🟡 FAIR VOL - Reduced insurance allocation (0.5%)")
        else:
            print("\n🔴 EXPENSIVE VOL - No insurance (VIX > 20)")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    asyncio.run(main())
