#!/usr/bin/env python3
"""
Portfolio-Lab v3.12 Phase 2: 0DTE Options Broker Integration

Options chain fetching and parsing for 0DTE yield enhancement strategy.
Integrates with Alpaca broker API for SPY 0DTE options data.

Usage:
    from src.broker.options_utils import OptionsChainFetcher, OptionQuote
    
    fetcher = OptionsChainFetcher()
    chain = await fetcher.fetch_0dte_chain("SPY")
    calls = chain.get_calls_by_delta(0.30)  # 30 delta calls
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from decimal import Decimal
import aiohttp
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OptionStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    EXERCISED = "exercised"
    ASSIGNED = "assigned"


@dataclass
class OptionQuote:
    """Represents a single option quote from the chain."""
    symbol: str                    # OCC option symbol (e.g., "SPY240516C00550000")
    underlying: str                # Underlying symbol
    option_type: OptionType
    strike: float
    expiration: date
    
    # Quote data
    bid: float
    ask: float
    last: float
    mark: float                     # Mid price
    
    # Greeks (if available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_vol: Optional[float] = None
    
    # Volume/OI
    volume: int = 0
    open_interest: int = 0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    bid_ask_spread_pct: float = field(init=False)
    
    def __post_init__(self):
        if self.mark > 0:
            self.bid_ask_spread_pct = (self.ask - self.bid) / self.mark * 100
        else:
            self.bid_ask_spread_pct = 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def is_liquid(self) -> bool:
        """Check if option meets liquidity criteria for trading."""
        return (
            self.volume >= 10 and
            self.open_interest >= 100 and
            self.bid_ask_spread_pct <= 5.0
        )
    
    @property
    def days_to_expiration(self) -> int:
        """Calculate DTE."""
        return (self.expiration - date.today()).days
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mark": self.mark,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "implied_vol": self.implied_vol,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "timestamp": self.timestamp.isoformat(),
            "bid_ask_spread_pct": self.bid_ask_spread_pct,
            "mid_price": self.mid_price,
            "is_liquid": self.is_liquid,
            "days_to_expiration": self.days_to_expiration,
        }


@dataclass
class OptionsChain:
    """Represents a full options chain for an underlying."""
    underlying: str
    quotes: List[OptionQuote] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def get_calls(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.option_type == OptionType.CALL]
    
    def get_puts(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.option_type == OptionType.PUT]
    
    def get_by_strike(self, strike: float) -> List[OptionQuote]:
        return [q for q in self.quotes if abs(q.strike - strike) < 0.01]
    
    def get_by_expiration(self, exp: date) -> List[OptionQuote]:
        return [q for q in self.quotes if q.expiration == exp]
    
    def get_0dte(self) -> List[OptionQuote]:
        """Get 0DTE options (expire today)."""
        today = date.today()
        return [q for q in self.quotes if q.expiration == today]
    
    def get_calls_by_delta(self, target_delta: float, tolerance: float = 0.05) -> List[OptionQuote]:
        """Get calls with delta near target."""
        return [
            q for q in self.get_calls()
            if q.delta is not None and abs(q.delta - target_delta) <= tolerance
        ]
    
    def get_liquid_calls(self, min_volume: int = 10, min_oi: int = 100, max_spread_pct: float = 5.0) -> List[OptionQuote]:
        """Filter for liquid call options."""
        return [
            q for q in self.get_calls()
            if q.volume >= min_volume 
            and q.open_interest >= min_oi
            and q.bid_ask_spread_pct <= max_spread_pct
        ]
    
    def find_optimal_call(self, target_delta: float = 0.30, max_spread_pct: float = 3.0) -> Optional[OptionQuote]:
        """Find best call option matching criteria."""
        candidates = self.get_liquid_calls(max_spread_pct=max_spread_pct)
        
        # Sort by delta proximity to target
        def delta_score(q: OptionQuote) -> float:
            if q.delta is None:
                return float('inf')
            return abs(q.delta - target_delta)
        
        candidates.sort(key=delta_score)
        return candidates[0] if candidates else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "underlying": self.underlying,
            "fetched_at": self.fetched_at.isoformat(),
            "quote_count": len(self.quotes),
            "call_count": len(self.get_calls()),
            "put_count": len(self.get_puts()),
            "quotes": [q.to_dict() for q in self.quotes],
        }


class OptionsChainFetcher:
    """
    Fetches options chain data from broker APIs.
    
    Supports Alpaca API for options data (if available) with fallback
    to simulation mode for testing and paper trading.
    """
    
    CACHE_TTL_SECONDS = 300  # 5 minute cache for options data
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper_mode = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self.cache_dir = Path("data/cache/options")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if we have live API access
        self.has_api_access = bool(self.api_key and self.secret_key)
        
        if not self.has_api_access:
            logger.warning("No Alpaca API credentials - running in simulation mode")
    
    async def fetch_0dte_chain(self, underlying: str = "SPY") -> OptionsChain:
        """
        Fetch 0DTE options chain for the underlying.
        
        Note: Alpaca's options API may not be available in all regions.
        This implementation uses simulation mode with realistic data
        for testing and paper trading.
        """
        if self.has_api_access:
            try:
                return await self._fetch_from_api(underlying)
            except Exception as e:
                logger.error(f"API fetch failed: {e}, falling back to simulation")
        
        return await self._generate_simulated_chain(underlying)
    
    async def _fetch_from_api(self, underlying: str) -> OptionsChain:
        """Fetch from Alpaca API (if available)."""
        # Alpaca options API endpoint
        base_url = "https://paper-api.alpaca.markets" if self.paper_mode else "https://api.alpaca.markets"
        
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        
        today = date.today()
        
        async with aiohttp.ClientSession() as session:
            # Get available expiration dates
            url = f"{base_url}/v2/options/snapshots/{underlying}"
            
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"API error: {resp.status}")
                
                data = await resp.json()
                
                quotes = []
                for option_data in data.get("options", []):
                    quote = self._parse_option_data(option_data, underlying)
                    if quote and quote.expiration == today:
                        quotes.append(quote)
                
                return OptionsChain(underlying=underlying, quotes=quotes)
    
    def _parse_option_data(self, data: Dict, underlying: str) -> Optional[OptionQuote]:
        """Parse API response into OptionQuote."""
        try:
            symbol = data.get("symbol", "")
            # Parse OCC symbol format: SPY240516C00550000
            # Underlying(3) + YY(2) + MM(2) + DD(2) + C/P(1) + Strike(8)
            
            if len(symbol) < 15:
                return None
            
            option_type = OptionType.CALL if symbol[15] == "C" else OptionType.PUT
            
            # Parse expiration
            year = int("20" + symbol[3:5])
            month = int(symbol[5:7])
            day = int(symbol[7:9])
            exp = date(year, month, day)
            
            # Parse strike (last 8 digits, divide by 1000)
            strike = int(symbol[16:]) / 1000
            
            quote_data = data.get("quote", {})
            
            return OptionQuote(
                symbol=symbol,
                underlying=underlying,
                option_type=option_type,
                strike=strike,
                expiration=exp,
                bid=float(quote_data.get("bid", 0)),
                ask=float(quote_data.get("ask", 0)),
                last=float(quote_data.get("last", 0)),
                mark=float(quote_data.get("mark", 0)),
                delta=quote_data.get("greeks", {}).get("delta"),
                gamma=quote_data.get("greeks", {}).get("gamma"),
                theta=quote_data.get("greeks", {}).get("theta"),
                vega=quote_data.get("greeks", {}).get("vega"),
                implied_vol=quote_data.get("implied_volatility"),
                volume=int(quote_data.get("volume", 0)),
                open_interest=int(quote_data.get("open_interest", 0)),
            )
        except Exception as e:
            logger.warning(f"Failed to parse option data: {e}")
            return None
    
    async def _generate_simulated_chain(self, underlying: str) -> OptionsChain:
        """
        Generate realistic simulated options chain for testing.
        
        Used when API is not available or for paper trading validation.
        """
        from src.data.price_fetcher import PriceFetcher
        
        # Get underlying price
        fetcher = PriceFetcher()
        spot_data = await fetcher.fetch_latest(underlying)
        spot = spot_data.get("price", 550.0)  # Default SPY price
        
        today = date.today()
        quotes = []
        
        # Generate strikes around current price (±5%)
        strike_step = 5.0  # $5 increments for SPY
        atm_strike = round(spot / strike_step) * strike_step
        strikes = [atm_strike + (i * strike_step) for i in range(-10, 11)]
        
        # Get VIX for volatility estimation
        try:
            vix_data = await fetcher.fetch_latest("VIX")
            vix = vix_data.get("price", 16.0)
        except:
            vix = 16.0
        
        # Time to expiration in years (0DTE = very small)
        tte = 1 / 365  # One day
        
        for strike in strikes:
            # Simulate realistic pricing using Black-Scholes approximation
            is_call = True
            price = self._simulate_option_price(spot, strike, vix/100, tte, is_call)
            
            # Estimate delta
            delta = self._estimate_delta(spot, strike, vix/100, tte, is_call)
            
            # Simulate liquidity (better near ATM)
            distance_from_atm = abs(strike - spot) / spot
            volume = max(10, int(100 * (1 - distance_from_atm)))
            oi = volume * 10
            
            # Simulate bid-ask spread (tighter near ATM)
            spread_pct = 1.0 + distance_from_atm * 10  # 1-5% spread
            bid = price * (1 - spread_pct/200)
            ask = price * (1 + spread_pct/200)
            
            quote = OptionQuote(
                symbol=f"{underlying}{today.strftime('%y%m%d')}C{int(strike*1000):08d}",
                underlying=underlying,
                option_type=OptionType.CALL,
                strike=strike,
                expiration=today,
                bid=bid,
                ask=ask,
                last=price,
                mark=(bid + ask) / 2,
                delta=delta,
                gamma=None,
                theta=None,
                vega=None,
                implied_vol=vix/100,
                volume=volume,
                open_interest=oi,
            )
            quotes.append(quote)
        
        return OptionsChain(underlying=underlying, quotes=quotes)
    
    def _simulate_option_price(self, spot: float, strike: float, vol: float, tte: float, is_call: bool) -> float:
        """Simple Black-Scholes approximation for simulation."""
        import math
        
        # Simplified: ATM options have premium ~ 0.4 * spot * vol * sqrt(tte)
        intrinsic = max(0, spot - strike) if is_call else max(0, strike - spot)
        time_value = spot * vol * math.sqrt(tte) * 0.4
        
        return intrinsic + time_value
    
    def _estimate_delta(self, spot: float, strike: float, vol: float, tte: float, is_call: bool) -> float:
        """Estimate delta using simplified formula."""
        import math
        
        # Simplified delta approximation
        d1 = (math.log(spot / strike) + (0.5 * vol**2) * tte) / (vol * math.sqrt(tte))
        
        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        delta = norm_cdf(d1) if is_call else norm_cdf(d1) - 1
        return round(delta, 3)
    
    def cache_chain(self, chain: OptionsChain):
        """Cache chain to SQLite for historical analysis."""
        cache_file = self.cache_dir / f"{chain.underlying}_options.db"
        
        conn = sqlite3.connect(str(cache_file))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                option_type TEXT NOT NULL,
                strike REAL NOT NULL,
                expiration TEXT NOT NULL,
                bid REAL,
                ask REAL,
                last REAL,
                delta REAL,
                volume INTEGER,
                open_interest INTEGER,
                fetched_at TEXT NOT NULL
            )
        """)
        
        for quote in chain.quotes:
            cursor.execute("""
                INSERT INTO options_chain 
                (symbol, option_type, strike, expiration, bid, ask, last, delta, volume, open_interest, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                quote.symbol,
                quote.option_type.value,
                quote.strike,
                quote.expiration.isoformat(),
                quote.bid,
                quote.ask,
                quote.last,
                quote.delta,
                quote.volume,
                quote.open_interest,
                chain.fetched_at.isoformat(),
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Cached {len(chain.quotes)} quotes to {cache_file}")


class OptionsChainCache:
    """Cache manager for options chain history."""
    
    def __init__(self, cache_dir: str = "data/cache/options"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical options data from cache."""
        cache_file = self.cache_dir / f"{symbol}_options.db"
        
        if not cache_file.exists():
            return []
        
        conn = sqlite3.connect(str(cache_file))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM options_chain 
            WHERE fetched_at >= datetime('now', '-{} days')
            ORDER BY fetched_at DESC
        """.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_avg_volume_by_strike(self, symbol: str, strike: float, days: int = 7) -> float:
        """Get average volume for a specific strike."""
        history = self.get_history(symbol, days)
        
        volumes = [
            row["volume"] for row in history 
            if abs(row["strike"] - strike) < 0.01
        ]
        
        return sum(volumes) / len(volumes) if volumes else 0.0


# Convenience functions for CLI usage
def fetch_chain_sync(underlying: str = "SPY") -> OptionsChain:
    """Synchronous wrapper for fetching options chain."""
    fetcher = OptionsChainFetcher()
    return asyncio.run(fetcher.fetch_0dte_chain(underlying))


def get_best_0dte_call(target_delta: float = 0.30) -> Optional[OptionQuote]:
    """Get best 0DTE call option near target delta."""
    chain = fetch_chain_sync("SPY")
    
    if not chain.get_0dte():
        logger.warning("No 0DTE options found")
        return None
    
    return chain.find_optimal_call(target_delta=target_delta)


if __name__ == "__main__":
    import asyncio
    
    async def test():
        fetcher = OptionsChainFetcher()
        chain = await fetcher.fetch_0dte_chain("SPY")
        
        print(f"Fetched {len(chain.quotes)} quotes for {chain.underlying}")
        print(f"0DTE calls: {len(chain.get_0dte())}")
        
        best = chain.find_optimal_call(target_delta=0.30)
        if best:
            print(f"\nBest 30-delta call:")
            print(f"  Strike: ${best.strike:.2f}")
            print(f"  Mark: ${best.mark:.2f}")
            print(f"  Delta: {best.delta:.3f}" if best.delta else "  Delta: N/A")
            print(f"  Volume: {best.volume}")
            print(f"  Spread: {best.bid_ask_spread_pct:.2f}%")
            print(f"  Liquid: {best.is_liquid}")
    
    asyncio.run(test())
