"""
Collar Options Bridge - v4.60/v4.80 Live Data Integration
Connects the collar signal module to broker options chain for real SPY options.

Fetches live SPY options chain to find optimal collar strikes with real market data.
Falls back to Black-Scholes estimates when options chain is unavailable.

Usage:
    python -m src.broker.collar_options_bridge fetch
    python -m src.broker.collar_options_bridge recommend
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

from .options_utils import OptionsChainFetcher, OptionsChain, OptionQuote, OptionType
from ..signals.collar_signal import (
    CollarSignalGenerator, CollarStrikes, CollarSignal,
    BlackScholesPricer, CollarRegime,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    LIVE = "live"          # Real options chain data
    SIMULATED = "simulated"  # Black-Scholes estimate
    CACHED = "cached"       # From local SQLite cache


@dataclass
class LiveCollarStrikes:
    """Collar strikes from live options chain."""
    source: str
    timestamp: str
    underlying_price: float
    vix_level: float
    days_to_expiry: int

    # Call
    call_symbol: str
    call_strike: float
    call_bid: float
    call_ask: float
    call_mark: float
    call_delta: Optional[float]
    call_volume: int
    call_oi: int

    # Put
    put_symbol: str
    put_strike: float
    put_bid: float
    put_ask: float
    put_mark: float
    put_delta: Optional[float]
    put_volume: int
    put_oi: int

    # Net
    net_premium: float      # call_mark - put_mark
    is_cashless: bool
    collar_cost_pct: float   # net premium / spot * 100

    # Quality
    call_liquid: bool
    put_liquid: bool
    bid_ask_spread_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


class CollarOptionsBridge:
    """
    Bridge between collar signal module and broker options chain.

    Workflow:
    1. Fetch SPY options chain from broker (or simulated fallback)
    2. Find optimal 30-delta call and 20-delta put
    3. Check if collar is cashless (net premium near zero)
    4. Return live collar strikes for execution
    """

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    OUTPUT_PATH = DATA_DIR / "signals" / "live_collar_strikes.json"

    # Target parameters
    CALL_DELTA_TARGET = 0.30
    PUT_DELTA_TARGET = -0.20
    TARGET_DAYS_TO_EXPIRY = 30  # Monthly collar
    CASHLESS_TOLERANCE_PCT = 0.15  # 0.15% of spot

    # Liquidity filters
    MIN_VOLUME = 10
    MIN_OI = 100
    MAX_SPREAD_PCT = 5.0

    def __init__(self):
        self._fetcher = OptionsChainFetcher()
        self._pricer = BlackScholesPricer()
        self._signal_gen = CollarSignalGenerator()
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "signals").mkdir(parents=True, exist_ok=True)

    async def fetch_optimal_collar(
        self,
        spot: Optional[float] = None,
        vix: Optional[float] = None,
        days_to_expiry: int = TARGET_DAYS_TO_EXPIRY,
    ) -> LiveCollarStrikes:
        """
        Fetch optimal collar strikes from live options chain.

        1. Try live options chain first
        2. Fall back to Black-Scholes estimate if unavailable
        """
        if spot is None:
            spot = 550.0
        if vix is None:
            vix = self._get_vix()

        # Try to fetch live chain, fall back if unavailable
        chain = None
        try:
            chain = await self._fetcher.fetch_0dte_chain("SPY")
        except (ModuleNotFoundError, ImportError, Exception) as e:
            logger.info(f"Options chain unavailable ({e}), using BS fallback")

        if chain is not None:
            if spot is None:
                spot = self._get_spot(chain)
            live_result = self._find_from_chain(chain, spot, vix, days_to_expiry)
            if live_result:
                return live_result

        # Fallback to Black-Scholes
        return await self._fallback_estimate(spot, vix, days_to_expiry)

    def _get_spot(self, chain: OptionsChain) -> float:
        """Infer spot price from options chain (ATM strike)."""
        # Use the mean of all strikes near the middle
        calls = chain.get_calls()
        if calls:
            strikes = sorted(set(q.strike for q in calls))
            if strikes:
                return strikes[len(strikes) // 2]
        return 550.0

    def _get_vix(self) -> float:
        vix_path = self.DATA_DIR / "vix_term_structure.json"
        if vix_path.exists():
            try:
                with open(vix_path) as f:
                    data = json.load(f)
                if data:
                    latest = max(data.keys())
                    return data[latest].get("vix_spot", 16.0)
            except Exception:
                pass
        return 16.0

    def _find_from_chain(
        self, chain: OptionsChain, spot: float, vix: float, dte: int
    ) -> Optional[LiveCollarStrikes]:
        """Find optimal collar from real options chain data."""
        calls = chain.get_calls()
        puts = chain.get_puts()

        if not calls or not puts:
            return None

        # Sort by delta proximity to target
        def delta_score(q: OptionQuote, target: float) -> float:
            if q.delta is None:
                return float("inf")
            return abs(q.delta - target)

        # Filter liquid
        liquid_calls = [q for q in calls if q.is_liquid]
        liquid_puts = [q for q in puts if q.is_liquid]

        if not liquid_calls or not liquid_puts:
            # Relax liquidity requirements
            liquid_calls = calls
            liquid_puts = puts

        # Find best call (target delta 0.30)
        liquid_calls.sort(key=lambda q: delta_score(q, self.CALL_DELTA_TARGET))
        best_call = liquid_calls[0] if liquid_calls else None

        # Find best put (target delta -0.20)
        liquid_puts.sort(key=lambda q: delta_score(q, self.PUT_DELTA_TARGET))
        best_put = liquid_puts[0] if liquid_puts else None

        if not best_call or not best_put:
            return None

        net = best_call.mark - best_put.mark
        cost_pct = net / spot * 100 if spot > 0 else 0
        is_cashless = abs(cost_pct) < self.CASHLESS_TOLERANCE_PCT

        spread_pct = max(best_call.bid_ask_spread_pct, best_put.bid_ask_spread_pct)

        return LiveCollarStrikes(
            source=DataSource.SIMULATED.value if not best_call.delta else DataSource.LIVE.value,
            timestamp=datetime.now().isoformat(),
            underlying_price=spot,
            vix_level=vix,
            days_to_expiry=dte,
            call_symbol=best_call.symbol,
            call_strike=best_call.strike,
            call_bid=best_call.bid,
            call_ask=best_call.ask,
            call_mark=best_call.mark,
            call_delta=best_call.delta,
            call_volume=best_call.volume,
            call_oi=best_call.open_interest,
            put_symbol=best_put.symbol,
            put_strike=best_put.strike,
            put_bid=best_put.bid,
            put_ask=best_put.ask,
            put_mark=best_put.mark,
            put_delta=best_put.delta,
            put_volume=best_put.volume,
            put_oi=best_put.open_interest,
            net_premium=round(net, 4),
            is_cashless=is_cashless,
            collar_cost_pct=round(cost_pct, 4),
            call_liquid=best_call.is_liquid,
            put_liquid=best_put.is_liquid,
            bid_ask_spread_pct=round(spread_pct, 2),
        )

    async def _fallback_estimate(
        self, spot: float, vix: float, dte: int
    ) -> LiveCollarStrikes:
        """Use Black-Scholes to estimate collar when chain unavailable."""
        strikes = self._signal_gen.calculate_strikes(spot, vix, dte)

        is_cashless = bool(strikes.is_cashless)
        cost_pct = float(strikes.collar_cost_pct)

        return LiveCollarStrikes(
            source=DataSource.SIMULATED.value,
            timestamp=datetime.now().isoformat(),
            underlying_price=spot,
            vix_level=vix,
            days_to_expiry=dte,
            call_symbol=f"SPY{datetime.now().strftime('%y%m%d')}C{int(strikes.call_strike*1000):08d}",
            call_strike=strikes.call_strike,
            call_bid=strikes.call_premium * 0.98,
            call_ask=strikes.call_premium * 1.02,
            call_mark=strikes.call_premium,
            call_delta=strikes.call_delta,
            call_volume=100,
            call_oi=1000,
            put_symbol=f"SPY{datetime.now().strftime('%y%m%d')}P{int(strikes.put_strike*1000):08d}",
            put_strike=strikes.put_strike,
            put_bid=strikes.put_premium * 0.98,
            put_ask=strikes.put_premium * 1.02,
            put_mark=strikes.put_premium,
            put_delta=strikes.put_delta,
            put_volume=100,
            put_oi=1000,
            net_premium=round(strikes.net_premium, 4),
            is_cashless=is_cashless,
            collar_cost_pct=round(cost_pct, 4),
            call_liquid=True,
            put_liquid=True,
            bid_ask_spread_pct=2.0,
        )

    def save_strikes(self, strikes: LiveCollarStrikes):
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(strikes.to_dict(), f, indent=2, default=str)

    def compare_with_signal(self, strikes: LiveCollarStrikes) -> Dict:
        """Compare live strikes with Black-Scholes signal estimate."""
        signal = self._signal_gen.generate_signal(
            spot=strikes.underlying_price, vix=strikes.vix_level
        )

        return {
            "live_call_strike": strikes.call_strike,
            "bs_call_strike": signal.call_strike,
            "call_diff_pct": round(
                (strikes.call_strike / signal.call_strike - 1) * 100, 2
            ) if signal.call_strike > 0 else 0,
            "live_put_strike": strikes.put_strike,
            "bs_put_strike": signal.put_strike,
            "put_diff_pct": round(
                (strikes.put_strike / signal.put_strike - 1) * 100, 2
            ) if signal.put_strike > 0 else 0,
            "live_net_premium": strikes.net_premium,
            "bs_net_premium": signal.strikes.net_premium,
            "source": strikes.source,
        }


def fetch_collar_sync() -> LiveCollarStrikes:
    """Synchronous wrapper for collar options fetch."""
    bridge = CollarOptionsBridge()
    return asyncio.run(bridge.fetch_optimal_collar())


def main():
    import sys

    bridge = CollarOptionsBridge()
    strikes = asyncio.run(bridge.fetch_optimal_collar())

    print("=" * 60)
    print("COLLAR OPTIONS BRIDGE v4.80")
    print("=" * 60)
    print(f"Source: {strikes.source}")
    print(f"Timestamp: {strikes.timestamp}")
    print(f"SPY: ${strikes.underlying_price:.2f}")
    print(f"VIX: {strikes.vix_level:.1f}")
    print(f"DTE: {strikes.days_to_expiry}")
    print()
    print("CALL (Short):")
    print(f"  Symbol: {strikes.call_symbol}")
    print(f"  Strike: ${strikes.call_strike:.2f}")
    print(f"  Bid/Ask: ${strikes.call_bid:.2f} / ${strikes.call_ask:.2f}")
    print(f"  Mark: ${strikes.call_mark:.2f}")
    if strikes.call_delta:
        print(f"  Delta: {strikes.call_delta:.3f}")
    print(f"  Volume/OI: {strikes.call_volume}/{strikes.call_oi}")
    print(f"  Liquid: {strikes.call_liquid}")
    print()
    print("PUT (Long):")
    print(f"  Symbol: {strikes.put_symbol}")
    print(f"  Strike: ${strikes.put_strike:.2f}")
    print(f"  Bid/Ask: ${strikes.put_bid:.2f} / ${strikes.put_ask:.2f}")
    print(f"  Mark: ${strikes.put_mark:.2f}")
    if strikes.put_delta:
        print(f"  Delta: {strikes.put_delta:.3f}")
    print(f"  Volume/OI: {strikes.put_volume}/{strikes.put_oi}")
    print(f"  Liquid: {strikes.put_liquid}")
    print()
    print(f"Net Premium: ${strikes.net_premium:.2f}")
    print(f"Cashless: {strikes.is_cashless}")
    print(f"Collar Cost: {strikes.collar_cost_pct:.2f}%")
    print(f"Bid-Ask Spread: {strikes.bid_ask_spread_pct:.2f}%")
    print("=" * 60)

    if "--save" in sys.argv:
        bridge.save_strikes(strikes)

    if "--compare" in sys.argv:
        print()
        comparison = bridge.compare_with_signal(strikes)
        print("Live vs Black-Scholes Comparison:")
        print(f"  Call: ${comparison['live_call_strike']:.2f} vs "
              f"${comparison['bs_call_strike']:.2f} "
              f"({comparison['call_diff_pct']:+.1f}%)")
        print(f"  Put:  ${comparison['live_put_strike']:.2f} vs "
              f"${comparison['bs_put_strike']:.2f} "
              f"({comparison['put_diff_pct']:+.1f}%)")
        print(f"  Net Premium: ${comparison['live_net_premium']:.2f} vs "
              f"${comparison['bs_net_premium']:.2f}")
        print(f"  Source: {comparison['source']}")


if __name__ == "__main__":
    main()
