"""
Credit spread data fetcher for LQD/HYG/AGG signals.

Fetches corporate bond ETF data to calculate credit risk spreads
as macro regime indicators for the ensemble voter.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────

CACHE_TTL_HOURS = 4
CACHE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "credit_cache.db"

SYMBOLS = {
    "LQD": "iShares iBoxx $ Invmt Grade Corp Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corp Bond ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
}

# Analysis windows
RETURN_WINDOW_DAYS = 30
ZSCORE_WINDOW_DAYS = 90
VOLATILITY_WINDOW_DAYS = 30

# Signal thresholds (from spec)
RISK_OFF_THRESHOLD = -0.02  # HYG underperforms LQD by 2%
RISK_ON_THRESHOLD = 0.02   # HYG outperforms LQD by 2%
PERSISTENCE_DAYS = 5


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CreditMetrics:
    """Container for credit spread metrics."""
    symbol: str
    timestamp: str
    price: float
    return_30d: float
    return_90d: float
    volatility_30d: float


@dataclass(frozen=True)
class CreditSpread:
    """LQD/HYG spread analysis."""
    timestamp: str
    lqd_return_30d: float
    hyg_return_30d: float
    agg_return_30d: float
    spread_absolute: float  # HYG - LQD
    spread_zscore: float
    trend_direction: str  # widening/tightening/stable
    volatility_regime: str  # low/medium/high
    persistence_days: int
    signal: str  # risk_on/risk_off/neutral
    confidence: float


@dataclass(frozen=True)
class CreditData:
    """Complete credit market snapshot."""
    timestamp: str
    metrics: dict[str, CreditMetrics]
    spread: CreditSpread
    is_fresh: bool


# ── Database ──────────────────────────────────────────────────────────────

class CreditCache:
    """SQLite cache for credit data with TTL."""

    def __init__(self, db_path: Path = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_metrics (
                    symbol TEXT PRIMARY KEY,
                    timestamp TEXT,
                    price REAL,
                    return_30d REAL,
                    return_90d REAL,
                    volatility_30d REAL,
                    cached_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_spread (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    timestamp TEXT,
                    lqd_return_30d REAL,
                    hyg_return_30d REAL,
                    agg_return_30d REAL,
                    spread_absolute REAL,
                    spread_zscore REAL,
                    trend_direction TEXT,
                    volatility_regime TEXT,
                    persistence_days INTEGER,
                    signal TEXT,
                    confidence REAL,
                    cached_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spread_history (
                    date TEXT PRIMARY KEY,
                    spread_absolute REAL,
                    lqd_return_30d REAL,
                    hyg_return_30d REAL
                )
            """)
            conn.commit()

    def is_fresh(self) -> bool:
        """Check if cached data is within TTL."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT cached_at FROM credit_spread WHERE id = 1"
            )
            row = cursor.fetchone()
            if not row:
                return False
            cached_at = datetime.fromisoformat(row[0])
            return datetime.now() - cached_at < timedelta(hours=CACHE_TTL_HOURS)

    def get_metrics(self, symbol: str) -> Optional[CreditMetrics]:
        """Retrieve cached metrics for symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT timestamp, price, return_30d, return_90d, volatility_30d
                   FROM credit_metrics WHERE symbol = ?""",
                (symbol,)
            )
            row = cursor.fetchone()
            if row:
                return CreditMetrics(
                    symbol=symbol,
                    timestamp=row[0],
                    price=row[1],
                    return_30d=row[2],
                    return_90d=row[3],
                    volatility_30d=row[4]
                )
        return None

    def get_spread(self) -> Optional[CreditSpread]:
        """Retrieve cached spread data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT timestamp, lqd_return_30d, hyg_return_30d, agg_return_30d,
                   spread_absolute, spread_zscore, trend_direction,
                   volatility_regime, persistence_days, signal, confidence
                   FROM credit_spread WHERE id = 1"""
            )
            row = cursor.fetchone()
            if row:
                return CreditSpread(
                    timestamp=row[0],
                    lqd_return_30d=row[1],
                    hyg_return_30d=row[2],
                    agg_return_30d=row[3],
                    spread_absolute=row[4],
                    spread_zscore=row[5],
                    trend_direction=row[6],
                    volatility_regime=row[7],
                    persistence_days=row[8],
                    signal=row[9],
                    confidence=row[10]
                )
        return None

    def save_metrics(self, metrics: CreditMetrics):
        """Save metrics to cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO credit_metrics
                   (symbol, timestamp, price, return_30d, return_90d, volatility_30d, cached_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (metrics.symbol, metrics.timestamp, metrics.price,
                 metrics.return_30d, metrics.return_90d, metrics.volatility_30d,
                 datetime.now().isoformat())
            )
            conn.commit()

    def save_spread(self, spread: CreditSpread):
        """Save spread analysis to cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO credit_spread
                   (id, timestamp, lqd_return_30d, hyg_return_30d, agg_return_30d,
                   spread_absolute, spread_zscore, trend_direction, volatility_regime,
                   persistence_days, signal, confidence, cached_at)
                   VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (spread.timestamp, spread.lqd_return_30d, spread.hyg_return_30d,
                 spread.agg_return_30d, spread.spread_absolute, spread.spread_zscore,
                 spread.trend_direction, spread.volatility_regime,
                 spread.persistence_days, spread.signal, spread.confidence,
                 datetime.now().isoformat())
            )
            # Also save to history
            conn.execute(
                """INSERT OR REPLACE INTO spread_history
                   (date, spread_absolute, lqd_return_30d, hyg_return_30d)
                   VALUES (?, ?, ?, ?)""",
                (spread.timestamp[:10], spread.spread_absolute,
                 spread.lqd_return_30d, spread.hyg_return_30d)
            )
            conn.commit()

    def get_history(self, days: int = 90) -> list[dict]:
        """Get spread history for trend analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT date, spread_absolute, lqd_return_30d, hyg_return_30d
                   FROM spread_history
                   WHERE date >= date(?, '-' || ? || ' days')
                   ORDER BY date""",
                (datetime.now().isoformat(), days)
            )
            return [
                {
                    "date": row[0],
                    "spread": row[1],
                    "lqd_return": row[2],
                    "hyg_return": row[3]
                }
                for row in cursor.fetchall()
            ]


# ── Data Fetching ──────────────────────────────────────────────────────────

class CreditFetcher:
    """Fetches and analyzes credit market data."""

    def __init__(self, cache: Optional[CreditCache] = None):
        self.cache = cache or CreditCache()

    def fetch_symbol_data(self, symbol: str, period: str = "6mo") -> Optional[tuple]:
        """Fetch price and calculate returns for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, auto_adjust=True)
            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])

            # Calculate returns
            returns = hist['Close'].pct_change().dropna()

            # 30-day return (approx 21 trading days)
            if len(hist) >= 21:
                return_30d = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1)
            else:
                return_30d = float(returns.sum())

            # 90-day return (approx 63 trading days)
            if len(hist) >= 63:
                return_90d = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-64] - 1)
            else:
                return_90d = float(returns.sum())

            # 30-day volatility (annualized)
            if len(returns) >= 21:
                vol_30d = float(returns.tail(21).std() * np.sqrt(252))
            else:
                vol_30d = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

            return current_price, return_30d, return_90d, vol_30d

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def calculate_spread(
        self,
        lqd_data: tuple,
        hyg_data: tuple,
        agg_data: tuple,
        history: list[dict]
    ) -> CreditSpread:
        """Calculate credit spread metrics and generate signal."""
        lqd_price, lqd_ret_30d, lqd_ret_90d, lqd_vol = lqd_data
        hyg_price, hyg_ret_30d, hyg_ret_90d, hyg_vol = hyg_data
        agg_price, agg_ret_30d, agg_ret_90d, agg_vol = agg_data

        # Spread calculation: HYG - LQD (high yield minus investment grade)
        spread_absolute = hyg_ret_30d - lqd_ret_30d

        # Z-score calculation vs 90-day history
        if history and len(history) >= 30:
            spreads = [h["spread"] for h in history if h["spread"] is not None]
            if len(spreads) >= 30:
                mean_spread = np.mean(spreads[-90:])
                std_spread = np.std(spreads[-90:])
                if std_spread > 0:
                    spread_zscore = (spread_absolute - mean_spread) / std_spread
                else:
                    spread_zscore = 0.0
            else:
                spread_zscore = 0.0
        else:
            spread_zscore = 0.0

        # Trend direction
        if len(history) >= 5:
            recent_spreads = [h["spread"] for h in history[-5:] if h["spread"] is not None]
            if len(recent_spreads) >= 3:
                if spread_absolute > recent_spreads[0] + 0.005:
                    trend_direction = "widening"
                elif spread_absolute < recent_spreads[0] - 0.005:
                    trend_direction = "tightening"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

        # Volatility regime based on spread volatility
        if history and len(history) >= 20:
            spreads = [h["spread"] for h in history[-20:] if h["spread"] is not None]
            if len(spreads) >= 20:
                spread_vol = np.std(spreads)
                if spread_vol < 0.005:
                    volatility_regime = "low"
                elif spread_vol < 0.015:
                    volatility_regime = "medium"
                else:
                    volatility_regime = "high"
            else:
                volatility_regime = "low"
        else:
            volatility_regime = "low"

        # Persistence (how long current regime has held)
        persistence_days = 0
        if history:
            current_signal = "neutral"
            if spread_absolute < RISK_OFF_THRESHOLD:
                current_signal = "risk_off"
            elif spread_absolute > RISK_ON_THRESHOLD:
                current_signal = "risk_on"

            for h in reversed(history):
                h_spread = h.get("spread", 0)
                h_signal = "neutral"
                if h_spread < RISK_OFF_THRESHOLD:
                    h_signal = "risk_off"
                elif h_spread > RISK_ON_THRESHOLD:
                    h_signal = "risk_on"

                if h_signal == current_signal:
                    persistence_days += 1
                else:
                    break

        # Signal generation
        if spread_absolute < RISK_OFF_THRESHOLD:
            signal = "risk_off"
            confidence = min(abs(spread_absolute) / 0.04, 1.0)
        elif spread_absolute > RISK_ON_THRESHOLD:
            signal = "risk_on"
            confidence = min(spread_absolute / 0.04, 1.0)
        else:
            signal = "neutral"
            confidence = 0.0

        return CreditSpread(
            timestamp=datetime.now().isoformat(),
            lqd_return_30d=lqd_ret_30d,
            hyg_return_30d=hyg_ret_30d,
            agg_return_30d=agg_ret_30d,
            spread_absolute=spread_absolute,
            spread_zscore=spread_zscore,
            trend_direction=trend_direction,
            volatility_regime=volatility_regime,
            persistence_days=persistence_days,
            signal=signal,
            confidence=round(confidence, 4)
        )

    def fetch_all(self, force_refresh: bool = False) -> CreditData:
        """Fetch complete credit data snapshot."""
        # Check cache first
        if not force_refresh and self.cache.is_fresh():
            cached_spread = self.cache.get_spread()
            if cached_spread:
                metrics = {}
                for symbol in SYMBOLS:
                    m = self.cache.get_metrics(symbol)
                    if m:
                        metrics[symbol] = m
                if len(metrics) == len(SYMBOLS):
                    return CreditData(
                        timestamp=cached_spread.timestamp,
                        metrics=metrics,
                        spread=cached_spread,
                        is_fresh=True
                    )

        # Fetch fresh data
        print("Fetching fresh credit data from Yahoo Finance...")
        metrics = {}
        data = {}

        for symbol in SYMBOLS:
            result = self.fetch_symbol_data(symbol)
            if result:
                price, ret_30d, ret_90d, vol = result
                metrics[symbol] = CreditMetrics(
                    symbol=symbol,
                    timestamp=datetime.now().isoformat(),
                    price=price,
                    return_30d=ret_30d,
                    return_90d=ret_90d,
                    volatility_30d=vol
                )
                data[symbol] = result

        if len(data) < 3:
            raise RuntimeError(f"Failed to fetch data for all symbols. Got {len(data)}/3")

        # Get history for spread calculation
        history = self.cache.get_history(days=90)

        # Calculate spread
        spread = self.calculate_spread(
            data["LQD"], data["HYG"], data["AGG"], history
        )

        # Save to cache
        for m in metrics.values():
            self.cache.save_metrics(m)
        self.cache.save_spread(spread)

        return CreditData(
            timestamp=spread.timestamp,
            metrics=metrics,
            spread=spread,
            is_fresh=True
        )


# ── CLI Interface ─────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Credit spread data fetcher for LQD/HYG/AGG"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch fresh data from Yahoo Finance"
    )
    parser.add_argument(
        "--signal", action="store_true",
        help="Display current signal analysis"
    )
    parser.add_argument(
        "--history", type=int, default=0,
        help="Show last N days of spread history"
    )
    parser.add_argument(
        "--export", type=str,
        help="Export data to JSON file"
    )
    parser.add_argument(
        "--cache-path", type=str,
        help="Override default cache database path"
    )

    args = parser.parse_args()

    cache_path = Path(args.cache_path) if args.cache_path else None
    cache = CreditCache(cache_path) if cache_path else CreditCache()
    fetcher = CreditFetcher(cache)

    try:
        data = fetcher.fetch_all(force_refresh=args.fetch)

        if args.signal or not (args.history or args.export):
            # Display current signal
            s = data.spread
            print(f"\n{'='*60}")
            print(f"Credit Spread Monitor — {data.timestamp[:19]}")
            print(f"{'='*60}")
            print(f"\nReturns (30-day):")
            print(f"  LQD (Investment Grade): {s.lqd_return_30d:+.2%}")
            print(f"  HYG (High Yield):       {s.hyg_return_30d:+.2%}")
            print(f"  AGG (Aggregate):        {s.agg_return_30d:+.2%}")
            print(f"\nSpread Analysis:")
            print(f"  Absolute (HYG - LQD):   {s.spread_absolute:+.2%}")
            print(f"  Z-Score (90d):          {s.spread_zscore:+.2f}")
            print(f"  Trend Direction:        {s.trend_direction}")
            print(f"  Volatility Regime:      {s.volatility_regime}")
            print(f"  Persistence:            {s.persistence_days} days")
            print(f"\nSignal:")
            signal_emoji = {"risk_on": "🟢", "risk_off": "🔴", "neutral": "⚪"}
            print(f"  Regime:                 {signal_emoji.get(s.signal, '⚪')} {s.signal.upper()}")
            print(f"  Confidence:             {s.confidence:.1%}")

            # Recommendation
            if s.signal == "risk_off" and s.confidence > 0.5 and s.persistence_days >= 5:
                print(f"\n  → Recommendation: Reduce equity exposure, add duration/quality")
            elif s.signal == "risk_on" and s.confidence > 0.5 and s.persistence_days >= 5:
                print(f"\n  → Recommendation: Increase equity exposure, reduce duration")
            else:
                print(f"\n  → Recommendation: Hold current allocation")

            print(f"\n{'='*60}")

        if args.history > 0:
            print(f"\nSpread History (last {args.history} days):")
            print(f"{'Date':<12} {'Spread':>10} {'LQD':>10} {'HYG':>10}")
            print("-" * 45)
            history = cache.get_history(days=args.history)
            for h in history[-args.history:]:
                print(f"{h['date']:<12} {h['spread']:>+9.2%} {h['lqd_return']:>+9.2%} {h['hyg_return']:>+9.2%}")

        if args.export:
            export_data = {
                "timestamp": data.timestamp,
                "metrics": {
                    k: asdict(v) for k, v in data.metrics.items()
                },
                "spread": asdict(data.spread),
                "is_fresh": data.is_fresh
            }
            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"\nExported to: {args.export}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
