"""
Credit Spread Monitor — Portfolio-Lab v3.14
Fetches LQD/HYG/AGG data and computes credit spread signals
for macro regime detection in the ensemble voter.

LQD = iShares iBoxx Investment Grade Corporate Bond ETF
HYG = iShares iBoxx High Yield Corporate Bond ETF
AGG = iShares Core US Aggregate Bond ETF (broad bond benchmark)
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "market.db"
CACHE_TTL_HOURS = 4

SPREAD_WIDENING_THRESHOLD = 2.0
SPREAD_TIGHTENING_THRESHOLD = -2.0
ZSCORE_WINDOW_DAYS = 90
ZSCORE_ALERT_THRESHOLD = 2.0


@dataclass(frozen=True)
class CreditMetrics:
    """Credit spread metrics for a single snapshot."""
    timestamp: str
    lqd_price: float
    hyg_price: float
    agg_price: float
    lqd_return_30d: float
    hyg_return_30d: float
    agg_return_30d: float
    spread_absolute: float
    spread_zscore: float
    trend_direction: str
    volatility_regime: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CreditSignal:
    """Credit-based macro regime signal."""
    timestamp: str
    spread_absolute: float
    spread_zscore: float
    trend_direction: str
    signal: str
    confidence: float
    equity_shift_pct: float
    rationale: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CreditFetcher:
    """Fetches LQD/HYG/AGG data and computes credit spread signals."""

    SYMBOLS = ["LQD", "HYG", "AGG"]

    def __init__(self, cache_db: Path = None):
        self.cache_db = cache_db or CACHE_DB_PATH
        self._init_cache()

    def _init_cache(self):
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS credit_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        data TEXT,
                        spread_absolute REAL,
                        trend_direction TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init credit cache: {e}")

    def _fetch_price(self, symbol: str) -> float:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        try:
            import requests
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if "chart" in data and data["chart"]["result"]:
                return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
        except Exception:
            pass
        return self._get_latest_cached_price(symbol)

    def _get_latest_cached_price(self, symbol: str) -> float:
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                    (symbol,),
                )
                row = cursor.fetchone()
                if row:
                    return float(row[0])
        except Exception:
            pass
        return 0.0

    def _fetch_30d_return(self, symbol: str) -> float:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if len(hist) >= 2:
                start = float(hist["Close"].iloc[0])
                end = float(hist["Close"].iloc[-1])
                if start > 0:
                    return (end - start) / start * 100
        except Exception:
            pass
        return 0.0

    def _compute_spread(self, lqd_return: float, hyg_return: float) -> float:
        return hyg_return - lqd_return

    def _compute_zscore(self, spread: float) -> float:
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cutoff = (datetime.now() - timedelta(days=ZSCORE_WINDOW_DAYS)).isoformat()
                cursor = conn.execute(
                    "SELECT spread_absolute FROM credit_cache WHERE created_at >= ?",
                    (cutoff,),
                )
                historical = [row[0] for row in cursor.fetchall() if row[0] is not None]
                if len(historical) < 10:
                    return spread / 2.0
                mean = sum(historical) / len(historical)
                variance = sum((s - mean) ** 2 for s in historical) / len(historical)
                std = variance ** 0.5
                if std < 0.01:
                    return 0.0
                return (spread - mean) / std
        except Exception:
            return spread / 2.0

    def _classify_trend(self, spread: float) -> str:
        if spread > SPREAD_WIDENING_THRESHOLD:
            return "tightening"
        elif spread < SPREAD_TIGHTENING_THRESHOLD:
            return "widening"
        return "stable"

    def _classify_volatility(self, hyg_return: float, agg_return: float) -> str:
        dispersion = abs(hyg_return - agg_return)
        if dispersion > 5.0:
            return "high"
        elif dispersion > 2.0:
            return "medium"
        return "low"

    def _get_cached(self) -> Optional[CreditMetrics]:
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT data FROM credit_cache ORDER BY id DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    cached_ts = datetime.fromisoformat(data.get("timestamp", ""))
                    if datetime.now() - cached_ts < timedelta(hours=CACHE_TTL_HOURS):
                        return CreditMetrics(**data)
        except Exception:
            pass
        return None

    def _save_cache(self, metrics: CreditMetrics):
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT INTO credit_cache (timestamp, data, spread_absolute, trend_direction) VALUES (?, ?, ?, ?)",
                    (metrics.timestamp, json.dumps(metrics.to_dict()), metrics.spread_absolute, metrics.trend_direction),
                )
                conn.execute("DELETE FROM credit_cache WHERE created_at < date('now', '-180 days')")
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def fetch_metrics(self, use_cache: bool = True) -> CreditMetrics:
        if use_cache:
            cached = self._get_cached()
            if cached:
                return cached
        lqd_price = self._fetch_price("LQD")
        hyg_price = self._fetch_price("HYG")
        agg_price = self._fetch_price("AGG")
        lqd_ret = self._fetch_30d_return("LQD")
        hyg_ret = self._fetch_30d_return("HYG")
        agg_ret = self._fetch_30d_return("AGG")
        spread = self._compute_spread(lqd_ret, hyg_ret)
        zscore = self._compute_zscore(spread)
        trend = self._classify_trend(spread)
        vol = self._classify_volatility(hyg_ret, agg_ret)
        metrics = CreditMetrics(
            timestamp=datetime.now().isoformat(),
            lqd_price=lqd_price, hyg_price=hyg_price, agg_price=agg_price,
            lqd_return_30d=round(lqd_ret, 2), hyg_return_30d=round(hyg_ret, 2),
            agg_return_30d=round(agg_ret, 2),
            spread_absolute=round(spread, 2), spread_zscore=round(zscore, 4),
            trend_direction=trend, volatility_regime=vol,
        )
        self._save_cache(metrics)
        return metrics

    def get_signal(self) -> CreditSignal:
        metrics = self.fetch_metrics()
        if abs(metrics.spread_zscore) > ZSCORE_ALERT_THRESHOLD:
            if metrics.trend_direction == "widening":
                signal, shift, conf = "risk_off", -3.0, min(0.9, abs(metrics.spread_zscore) / 3.0)
                rationale = f"Credit spreads widening (z={metrics.spread_zscore:+.2f}). Flight to quality."
            elif metrics.trend_direction == "tightening":
                signal, shift, conf = "risk_on", 3.0, min(0.9, abs(metrics.spread_zscore) / 3.0)
                rationale = f"Credit spreads tightening (z={metrics.spread_zscore:+.2f}). Risk appetite improving."
            else:
                signal, shift, conf = "neutral", 0.0, 0.5
                rationale = "Stable credit spreads, no regime signal."
        elif abs(metrics.spread_absolute) > 1.0:
            if metrics.trend_direction == "widening":
                signal, shift, conf = "risk_off", -2.0, 0.6
                rationale = f"Moderate spread widening ({metrics.spread_absolute:+.1f}%)."
            elif metrics.trend_direction == "tightening":
                signal, shift, conf = "risk_on", 2.0, 0.6
                rationale = f"Moderate spread tightening ({metrics.spread_absolute:+.1f}%)."
            else:
                signal, shift, conf = "neutral", 0.0, 0.5
                rationale = "Stable credit spreads."
        else:
            signal, shift, conf = "neutral", 0.0, 0.5
            rationale = "Credit spreads within normal range."
        return CreditSignal(
            timestamp=metrics.timestamp,
            spread_absolute=metrics.spread_absolute,
            spread_zscore=metrics.spread_zscore,
            trend_direction=metrics.trend_direction,
            signal=signal, confidence=round(conf, 4),
            equity_shift_pct=shift, rationale=rationale,
        )

    def get_history(self, days: int = 30) -> List[Dict]:
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT data FROM credit_cache WHERE created_at >= date('now', ?) ORDER BY created_at DESC",
                    (f"-{days} days",),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"History fetch failed: {e}")
            return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Spread Monitor")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--history", type=int)
    args = parser.parse_args()
    fetcher = CreditFetcher()
    if args.fetch or (not args.signal and not args.history):
        m = fetcher.fetch_metrics()
        print(f"LQD: ${m.lqd_price:.2f} (30d: {m.lqd_return_30d:+.1f}%)")
        print(f"HYG: ${m.hyg_price:.2f} (30d: {m.hyg_return_30d:+.1f}%)")
        print(f"AGG: ${m.agg_price:.2f} (30d: {m.agg_return_30d:+.1f}%)")
        print(f"Spread (HYG-LQD): {m.spread_absolute:+.2f}% (z={m.spread_zscore:+.2f})")
        print(f"Trend: {m.trend_direction} | Vol: {m.volatility_regime}")
    if args.signal:
        sig = fetcher.get_signal()
        print(f"Signal: {sig.signal} | Confidence: {sig.confidence:.1%} | Shift: {sig.equity_shift_pct:+.1f}%")
        print(f"Rationale: {sig.rationale}")
    if args.history:
        hist = fetcher.get_history(args.history)
        print(f"Last {len(hist)} records")
