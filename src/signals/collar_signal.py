"""
Collared Signal Generator - v4.60 Implementation
Cashless collar (OTM call + OTM put) on SPY allocation for drawdown protection.

Target: Reduce max drawdown from -26.2% to <= -20% while maintaining >=90% of returns.
Core mechanic: Write OTM call at delta ~0.30, buy OTM put at delta ~-0.20.
Monthly roll cycle, VIX-aware strike widening.

Usage:
    python -m src.signals.collar_signal generate
    python -m src.signals.collar_signal status
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollarState(Enum):
    ACTIVE = "active"           # Collar in place
    UNHEDGED = "unhedged"       # No collar (market closed, data missing)
    WIDE = "wide"               # Strikes widened for high VIX
    NARROW = "narrow"           # Tight strikes for low VIX
    ROLLING = "rolling"         # Mid-roll window


class CollarRegime(Enum):
    NORMAL = "normal"                  # VIX 10-20, standard strikes
    ELEVATED = "elevated"              # VIX 20-30, wider strikes
    STRESS = "stress"                  # VIX 30-40, very wide
    CRISIS = "crisis"                  # VIX >40, collar disabled (cost prohibitive)


@dataclass
class CollarStrikes:
    """Call and put strikes for a cashless collar."""
    underlying_price: float
    call_strike: float
    put_strike: float

    # Premiums
    call_premium: float
    put_premium: float
    net_premium: float       # call_premium - put_premium (target ~0)

    # Greeks
    call_delta: float
    put_delta: float

    # Context
    vix_level: float
    regime: str
    days_to_expiry: int

    # Quality
    is_cashless: bool         # |net_premium| < tolerance
    collar_cost_pct: float    # net cost as % of underlying

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollarSignal:
    """Complete collar signal output."""
    timestamp: str
    signal_state: str

    # Strikes
    call_strike: float
    put_strike: float
    underlying_price: float

    # Expected P&L impact
    expected_monthly_yield: float   # Net credit if positive
    max_upside_pct: float           # Cap from call strike
    max_downside_pct: float         # Floor from put strike

    # Collar details
    vix_level: float
    regime: str
    strikes: CollarStrikes

    # Portfolio integration
    collar_notional_pct: float  # % of SPY allocation to collar
    spy_shift: float             # Effective equity shift
    confidence: float

    is_valid: bool
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strikes"] = self.strikes.to_dict()
        return d


class BlackScholesPricer:
    """
    Black-Scholes option pricing for European options.

    Used to estimate call/put premiums for collar strike selection.
    No ML dependencies — uses scipy.stats.norm if available, else math.erf.
    """

    def __init__(self):
        self._have_scipy = False
        try:
            from scipy.stats import norm
            self._norm = norm
            self._have_scipy = True
        except ImportError:
            self._norm = None

    def _norm_cdf(self, x: float) -> float:
        if self._have_scipy:
            return self._norm.cdf(x)
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        if self._have_scipy:
            return self._norm.pdf(x)
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    def price_option(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,  # years
        rate: float,
        vol: float,
        is_call: bool,
    ) -> Dict[str, float]:
        """Price a European option using Black-Scholes."""
        if time_to_expiry <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
            return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * time_to_expiry) / (
            vol * math.sqrt(time_to_expiry)
        )
        d2 = d1 - vol * math.sqrt(time_to_expiry)

        if is_call:
            price = spot * self._norm_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * self._norm_cdf(d2)
            delta = self._norm_cdf(d1)
            theta = (
                -spot * self._norm_pdf(d1) * vol / (2 * math.sqrt(time_to_expiry))
                - rate * strike * math.exp(-rate * time_to_expiry) * self._norm_cdf(d2)
            ) / 365
        else:
            price = strike * math.exp(-rate * time_to_expiry) * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)
            delta = self._norm_cdf(d1) - 1
            theta = (
                -spot * self._norm_pdf(d1) * vol / (2 * math.sqrt(time_to_expiry))
                + rate * strike * math.exp(-rate * time_to_expiry) * self._norm_cdf(-d2)
            ) / 365

        gamma = self._norm_pdf(d1) / (spot * vol * math.sqrt(time_to_expiry))
        vega = spot * self._norm_pdf(d1) * math.sqrt(time_to_expiry) / 100

        return {"price": round(price, 4), "delta": round(delta, 4),
                "gamma": round(gamma, 6), "theta": round(theta, 4), "vega": round(vega, 4)}

    def find_strike_by_delta(
        self,
        spot: float,
        target_delta: float,
        time_to_expiry: float,
        rate: float,
        vol: float,
        is_call: bool,
    ) -> float:
        """Binary search to find strike with target delta.

        For both calls and puts, delta is monotonically decreasing with strike:
        - Calls: delta 1→0 as strike increases
        - Puts:  delta 0→-1 as strike increases
        So both use: delta > target → need higher strike (lo=mid).
        """
        lo = spot * 0.5
        hi = spot * 1.5
        for _ in range(50):
            mid = (lo + hi) / 2
            result = self.price_option(spot, mid, time_to_expiry, rate, vol, is_call)
            delta = result["delta"]
            if abs(delta - target_delta) < 0.001:
                return round(mid, 2)
            # For both calls and puts: delta decreases as strike increases.
            # If our delta is too high (less negative for puts), need higher strike.
            if delta > target_delta:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 2)


class CollarSignalGenerator:
    """
    Generates cashless collar signals for SPY allocation.

    Strategy:
    - Write OTM call at Δ≈0.30 (cap upside at ~2-3% monthly)
    - Buy OTM put  at Δ≈-0.20 (floor at ~-3% monthly)
    - Net premium target: near zero (cashless)
    - VIX-aware strike widening: wider in high vol
    - Monthly roll at expiry
    """

    # Default parameters
    CALL_DELTA_TARGET = 0.30
    PUT_DELTA_TARGET = -0.20

    # Strike widening by regime
    WIDE_FACTOR = {CollarRegime.NORMAL: 1.0, CollarRegime.ELEVATED: 1.3,
                   CollarRegime.STRESS: 1.6}

    # VIX thresholds for regime
    VIX_ELEVATED = 20.0
    VIX_STRESS = 30.0
    VIX_CRISIS = 40.0

    # Collar parameters
    DEFAULT_DAYS_TO_EXPIRY = 30
    RISK_FREE_RATE = 0.045     # ~4.5% current
    CASHLESS_TOLERANCE = 0.15  # |net premium| / spot < 0.15% considered cashless

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    OUTPUT_PATH = DATA_DIR / "signals" / "collar_signal.json"

    def __init__(self):
        self.pricer = BlackScholesPricer()
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "signals").mkdir(parents=True, exist_ok=True)

    def classify_regime(self, vix: float) -> CollarRegime:
        if vix >= self.VIX_CRISIS:
            return CollarRegime.CRISIS
        elif vix >= self.VIX_STRESS:
            return CollarRegime.STRESS
        elif vix >= self.VIX_ELEVATED:
            return CollarRegime.ELEVATED
        return CollarRegime.NORMAL

    def calculate_strikes(
        self,
        spot: float,
        vix: float,
        days_to_expiry: int = DEFAULT_DAYS_TO_EXPIRY,
    ) -> CollarStrikes:
        """Calculate optimal collar strikes for cashless or near-cashless setup."""
        regime = self.classify_regime(vix)
        tte = days_to_expiry / 365.0  # time to expiry in years
        rate = self.RISK_FREE_RATE

        if regime == CollarRegime.CRISIS:
            # Cost prohibitive — disable collar
            return CollarStrikes(
                underlying_price=spot, call_strike=spot * 1.10, put_strike=spot * 0.90,
                call_premium=0.0, put_premium=0.0, net_premium=0.0,
                call_delta=0.0, put_delta=0.0, vix_level=vix, regime=regime.value,
                days_to_expiry=days_to_expiry, is_cashless=False, collar_cost_pct=1.0,
            )

        wide = self.WIDE_FACTOR.get(regime, 1.0)

        # Adjust target deltas for regime
        call_delta_target = self.CALL_DELTA_TARGET / wide   # e.g., 0.30/1.3 = 0.23 in elevated
        put_delta_target = self.PUT_DELTA_TARGET * wide     # e.g., -0.20*1.3 = -0.26 in elevated

        # Implied vol from VIX + spread
        call_vol = (vix / 100) * 1.05   # OTM calls slightly higher IV
        put_vol = (vix / 100) * 1.10    # OTM puts have vol skew premium

        # Find strikes by delta
        call_strike = self.pricer.find_strike_by_delta(
            spot, call_delta_target, tte, rate, call_vol, is_call=True
        )
        put_strike = self.pricer.find_strike_by_delta(
            spot, put_delta_target, tte, rate, put_vol, is_call=False
        )

        # Price both legs
        call_result = self.pricer.price_option(spot, call_strike, tte, rate, call_vol, is_call=True)
        put_result = self.pricer.price_option(spot, put_strike, tte, rate, put_vol, is_call=False)

        call_premium = call_result["price"]
        put_premium = put_result["price"]
        net = call_premium - put_premium
        cost_pct = net / spot * 100 if spot > 0 else 0

        is_cashless = abs(net) / spot * 100 < self.CASHLESS_TOLERANCE if spot > 0 else False

        return CollarStrikes(
            underlying_price=spot, call_strike=call_strike, put_strike=put_strike,
            call_premium=call_premium, put_premium=put_premium, net_premium=round(net, 4),
            call_delta=call_result["delta"], put_delta=put_result["delta"],
            vix_level=vix, regime=regime.value, days_to_expiry=days_to_expiry,
            is_cashless=is_cashless, collar_cost_pct=round(cost_pct, 4),
        )

    def generate_signal(
        self,
        spot: Optional[float] = None,
        vix: Optional[float] = None,
        days_to_expiry: int = DEFAULT_DAYS_TO_EXPIRY,
    ) -> CollarSignal:
        """Generate complete collar signal."""
        if spot is None:
            spot = self._fetch_spot_price()
        if vix is None:
            vix = self._fetch_vix_level()

        if spot <= 0:
            return CollarSignal(
                timestamp=datetime.now().isoformat(),
                signal_state="error", call_strike=0, put_strike=0,
                underlying_price=0, expected_monthly_yield=0,
                max_upside_pct=0, max_downside_pct=0, vix_level=vix,
                regime="unknown",
                strikes=CollarStrikes(spot, 0, 0, 0, 0, 0, 0, 0, vix, "unknown", days_to_expiry, False, 0),
                collar_notional_pct=0, spy_shift=0, confidence=0,
                is_valid=False, reason="Invalid spot price",
            )

        regime = self.classify_regime(vix)
        strikes = self.calculate_strikes(spot, vix, days_to_expiry)

        # Monthly yield from collar (annualized)
        monthly_yield = (strikes.net_premium / spot) * 12 if spot > 0 else 0

        # Upside cap and downside floor
        max_upside = (strikes.call_strike / spot - 1) * 100 if spot > 0 else 0
        max_downside = (1 - strikes.put_strike / spot) * 100 if spot > 0 else 0

        if regime == CollarRegime.CRISIS:
            state = CollarState.UNHEDGED
            confidence = 0.0
            reason = "Collar disabled: VIX crisis level, cost prohibitive"
        elif strikes.is_cashless:
            state = CollarState.ACTIVE
            confidence = 85.0
            reason = f"Cashless collar active: call {strikes.call_strike}, put {strikes.put_strike}"
        else:
            state = CollarState.ACTIVE
            confidence = 60.0
            net_str = "credit" if strikes.net_premium > 0 else "debit"
            reason = (
                f"Near-cashless collar: {net_str} ${abs(strikes.net_premium):.2f}, "
                f"call {strikes.call_strike}, put {strikes.put_strike}"
            )

        return CollarSignal(
            timestamp=datetime.now().isoformat(),
            signal_state=state.value,
            call_strike=strikes.call_strike,
            put_strike=strikes.put_strike,
            underlying_price=spot,
            expected_monthly_yield=round(monthly_yield * 100, 2),
            max_upside_pct=round(max_upside, 2),
            max_downside_pct=round(max_downside, 2),
            vix_level=vix,
            regime=regime.value,
            strikes=strikes,
            collar_notional_pct=0.46,  # 46% SPY allocation
            spy_shift=round(-strikes.net_premium / spot * 100, 4) if spot > 0 else 0,
            confidence=confidence,
            is_valid=regime != CollarRegime.CRISIS,
            reason=reason,
        )

    def _fetch_spot_price(self) -> float:
        """Fetch current SPY price from market data."""
        db_path = self.DATA_DIR / "market.db"
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol='SPY' ORDER BY date DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    return float(row[0])
            except Exception:
                pass
        return 550.0  # fallback

    def _fetch_vix_level(self) -> float:
        """Fetch current VIX level."""
        db_path = self.DATA_DIR / "market.db"
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol='VIX' ORDER BY date DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    return float(row[0])
            except Exception:
                pass

        # Try alternative data sources
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

    def save_signal(self, signal: CollarSignal):
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(signal.to_dict(), f, indent=2)
        logger.info(f"Collar signal saved to {self.OUTPUT_PATH}")


def generate_collar_signal(
    spot: Optional[float] = None,
    vix: Optional[float] = None,
) -> CollarSignal:
    """Convenience function for collar signal generation."""
    generator = CollarSignalGenerator()
    return generator.generate_signal(spot=spot, vix=vix)


def main():
    """CLI entry point."""
    import sys
    generator = CollarSignalGenerator()
    signal = generator.generate_signal()

    print("=" * 60)
    print("CASHLESS COLLAR SIGNAL GENERATOR v4.60")
    print("=" * 60)
    print(f"Timestamp: {signal.timestamp}")
    print(f"State: {signal.signal_state}")
    print(f"Regime: {signal.regime}")
    print()
    print(f"SPY Price: ${signal.underlying_price:.2f}")
    print(f"VIX: {signal.vix_level:.1f}")
    print()
    print("Collar Strikes:")
    print(f"  Call (short): ${signal.call_strike:.2f} "
          f"(+{signal.max_upside_pct:.1f}% cap)")
    print(f"  Put  (long):  ${signal.put_strike:.2f} "
          f"(-{signal.max_downside_pct:.1f}% floor)")
    print()
    print(f"Net Premium: ${signal.strikes.net_premium:.2f} per share")
    print(f"Annualized Yield: {signal.expected_monthly_yield:.2f}%")
    print(f"Cashless: {signal.strikes.is_cashless}")
    print()
    print(f"Notional: {signal.collar_notional_pct:.0%} of portfolio")
    print(f"Confidence: {signal.confidence:.0f}%")
    print(f"Valid: {signal.is_valid}")
    print(f"Reason: {signal.reason}")
    print("=" * 60)

    if "--save" in sys.argv:
        generator.save_signal(signal)


if __name__ == "__main__":
    main()
