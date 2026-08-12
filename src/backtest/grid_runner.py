"""Shared grid-search helpers for backtest research scripts (A5 phase 1, Item 32).

Consolidates the price-loading / portfolio-simulation / metric helpers that
were duplicated across ``gold_allocation_sweep.py`` and
``duration_yield_backtest.py``. Semantics are ported faithfully from the
pilot implementations — NO behavior change (A4 output-equality verified
against pre-migration captures, float tolerance ``pytest.approx(rel=1e-9)``).

Surface:
- ``load_prices()`` — prices.json via ``price_cache.get_prices`` (both
  pilots consume prices.json; market.db sqlite loading is phase-2 territory)
- ``prices_to_frame(data)`` — dict-of-symbols → long DataFrame (duration's
  loader semantics; lowercase symbol columns)
- ``simulate_portfolio(prices, dates, weights)`` — gold's portfolio
  simulation math (cagr/vol/sharpe/max_dd/yearly returns)
- ``calculate_sharpe`` / ``calculate_max_drawdown`` / ``calculate_cagr`` —
  duration's metric math (the reference implementation)
- ``run_grid_and_save(result, out_path, experiment_manifest=None)`` —
  ``save_results_json`` wrapper

ML-safe: no ML imports. Research-only: no cron/live path.
"""

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.backtest.metrics import save_results_json
from src.paths import RISK_FREE_RATE, sqlite_connect

logger = logging.getLogger(__name__)


def load_prices_market_db(
    cache_db: Path,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, float]]:
    """Load daily close prices from market.db, indexed by date.

    Ported verbatim from ``BehavioralSentimentBacktest._load_prices``
    (behavioral_sentiment_backtest.py:70-90). Returns
    ``{symbol: {date_str: close}}`` — the date-indexed shape, NOT the
    prices.json symbol → [{d, p}] shape (A5 phase 2, Item 37).
    """
    prices: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
    try:
        with sqlite_connect(cache_db) as conn:
            placeholders = ",".join("?" for _ in symbols)
            cursor = conn.execute(
                f"""SELECT symbol, date, close FROM prices
                    WHERE symbol IN ({placeholders})
                    AND date >= ? AND date <= ?
                    ORDER BY date""",
                (*symbols, start_date, end_date),
            )
            for symbol, date_str, close in cursor.fetchall():
                if close is not None and close > 0:
                    prices[symbol][date_str] = float(close)
    except (OSError, sqlite3.Error, KeyError, ValueError, TypeError) as e:
        logger.error("Failed to load prices: %s", e)
    return prices


def load_prices() -> Dict[str, List[Dict[str, Any]]]:
    """Load the raw prices.json payload (symbol → [{d, p}, ...]).

    Deferred import keeps the ``src.data.price_cache.get_prices`` patch
    seam live for tests (both pilots patch it via the fixture).
    """
    from src.data.price_cache import get_prices
    return get_prices()


def load_prices_numpy() -> Dict[str, np.ndarray]:
    """Load prices.json as ``{symbol: np.ndarray}`` of close prices.

    Ported verbatim from ``regime_allocation_backtest.load_prices`` /
    ``combined_regime_alloc_vol_target.load_prices`` (A5 phase 3, Item 38):
    symbol → float array of ``p`` values; records that are not
    list-of-dicts are skipped. NO DataFrame round-trip (float-path drift).
    """
    from src.paths import PRICES_JSON

    with open(PRICES_JSON) as f:
        raw = json.load(f)
    prices: Dict[str, np.ndarray] = {}
    for symbol, records in raw.items():
        if isinstance(records, list) and len(records) > 0 and isinstance(records[0], dict):
            prices[symbol] = np.array([r["p"] for r in records], dtype=float)
    return prices


def load_prices_dates_prices() -> Dict[str, Dict[str, list]]:
    """Load prices.json as ``{symbol: {"dates": [...], "prices": [...]}}``.

    Ported verbatim from ``walk_forward_champion._load_prices``
    (A5 phase 3, Item 38): SPY/GLD/TLT only, per-symbol dates/prices lists.
    """
    from src.paths import PRICES_JSON

    with open(PRICES_JSON) as f:
        raw = json.load(f)

    result: Dict[str, Dict[str, list]] = {}
    for sym in ["SPY", "GLD", "TLT"]:
        entries = raw.get(sym, [])
        if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
            result[sym] = {
                "dates": [e["d"] for e in entries],
                "prices": [e["p"] for e in entries],
            }
    return result


def prices_to_frame(data: Dict[str, List[Dict[str, Any]]]) -> Any:
    """Convert the raw prices dict to a DataFrame (dates as rows).

    Ported verbatim from ``duration_yield_backtest.load_price_data``:
    symbols become lowercase columns; sorted by date.
    """
    import pandas as pd

    all_dates = set()
    for symbol, entries in data.items():
        for entry in entries:
            all_dates.add(entry["d"])

    dates = sorted(all_dates)
    records = []

    for date in dates:
        record = {"date": date}
        for symbol, entries in data.items():
            # Find price for this date
            price = None
            for entry in entries:
                if entry["d"] == date:
                    price = entry["p"]
                    break
            if price is not None:
                record[symbol.lower()] = price
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def simulate_portfolio(
    prices: Dict[str, List[float]],
    dates: List[str],
    weights: Dict[str, float],
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """Simulate a portfolio from aligned price series.

    Ported verbatim from ``GoldAllocationSweep._simulate_portfolio``
    (gold_allocation_sweep.py:123-167): returns
    (cagr, vol, sharpe, max_dd, year_returns) with the pilot's rounding.
    """
    def _returns(series: List[float]) -> List[float]:
        return [(series[i] / series[i - 1] - 1) for i in range(1, len(series))]

    spy_rets = _returns(prices["SPY"])
    gld_rets = _returns(prices["GLD"])
    tlt_rets = _returns(prices["TLT"])
    ief_rets = _returns(prices.get("IEF", prices["TLT"]))

    n = min(len(spy_rets), len(gld_rets), len(tlt_rets), len(ief_rets))

    values = [1.0]
    peak = 1.0
    daily_rets = []
    yearly_rets: Dict[str, List[float]] = {}

    for i in range(n):
        ret = (
            weights.get("spy", 0) * spy_rets[i] +
            weights.get("gld", 0) * gld_rets[i] +
            weights.get("tlt", 0) * tlt_rets[i] +
            weights.get("ief", 0) * ief_rets[i]
        )
        values.append(values[-1] * (1 + ret))
        daily_rets.append(ret)
        peak = max(peak, values[-1])

        # Track yearly returns
        year = dates[i + 1][:4] if i + 1 < len(dates) else "unknown"
        if year not in yearly_rets:
            yearly_rets[year] = []
        yearly_rets[year].append(ret)

    cagr = np.mean(daily_rets) * 252 * 100
    vol = np.std(daily_rets) * math.sqrt(252) * 100
    sharpe = cagr / vol if vol > 0 else 0
    max_dd = min((v / peak - 1) * 100 for v in values) if values else 0

    # Year-level returns
    year_total = {}
    for y, rets in yearly_rets.items():
        year_total[y] = (np.prod([1 + r for r in rets]) - 1) * 100

    return round(cagr, 2), round(vol, 2), round(sharpe, 3), round(max_dd, 2), year_total


def calculate_sharpe(returns: Any, risk_free_rate: float = RISK_FREE_RATE / 100) -> float:
    """Annualized Sharpe ratio (duration_yield_backtest.py reference math)."""
    if len(returns) < 30:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0

    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_max_drawdown(returns: Any) -> float:
    """Maximum drawdown (duration_yield_backtest.py reference math)."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_cagr(returns: Any) -> float:
    """Annualized return (duration_yield_backtest.py reference math)."""
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod()
    years = len(returns) / 252

    if years < 0.1:
        return 0.0

    cagr = (total_return ** (1 / years)) - 1
    return cagr


def run_grid_and_save(
    result: Any,
    out_path: Path,
    experiment_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a grid result dict via ``save_results_json``."""
    save_results_json(
        result,
        output_path=str(out_path),
        experiment_manifest=experiment_manifest,
    )
    logger.info("Grid results saved to %s", out_path)
