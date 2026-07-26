"""Deterministic price history for tests that run without operator data."""

from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

from tests.fixtures.compact_prices import CompactPricePayload

# Covers the production market-data universe plus aliases used by Python-only
# overlay tests. This is intentionally test-owned rather than live authority.
_SYNTHETIC_SYMBOLS = (
    "AGG",
    "BTC",
    "DBC",
    "EFA",
    "ETH",
    "FXA",
    "FXB",
    "FXC",
    "FXE",
    "FXF",
    "FXY",
    "GLD",
    "IEF",
    "MTUM",
    "QQQ",
    "QUAL",
    "SHY",
    "SPY",
    "TLT",
    "TMF",
    "UBT",
    "UDN",
    "USMV",
    "UUP",
    "VBR",
    "VIX",
    "VLUE",
    "VTI",
    "VXUS",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "^VIX",
    "^VIX3M",
)
_BOND_SYMBOLS = frozenset({"AGG", "IEF", "SHY", "TLT", "TMF", "UBT"})

SYNTHETIC_PRICE_START = date(2005, 1, 3)
SYNTHETIC_PRICE_END = date(2026, 7, 24)
_RNG_SEED = 20260726


def _business_dates(start: date, end: date) -> list[date]:
    dates: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _crisis_adjustment(day: date, *, bond: bool) -> float:
    """Add modest stress behavior without introducing invalid price jumps."""
    if day.year == 2008 and day.month in {9, 10}:
        return 0.0015 if bond else -0.0025
    if day.year == 2020 and day.month == 3:
        return 0.001 if bond else -0.002
    if day.year == 2022 and bond:
        return -0.00045
    return 0.0


def build_synthetic_prices() -> CompactPricePayload:
    """Build varied, positive daily prices spanning every backtest window."""
    dates = _business_dates(SYNTHETIC_PRICE_START, SYNTHETIC_PRICE_END)
    dated_rows = [(day, day.isoformat()) for day in dates]
    common_rng = random.Random(_RNG_SEED)
    common_shocks = [common_rng.gauss(0.0, 0.0052) for _ in dates]
    payload: CompactPricePayload = {}

    for symbol_index, symbol in enumerate(_SYNTHETIC_SYMBOLS):
        if symbol in {"VIX", "^VIX", "^VIX3M"}:
            offset = 1.5 if symbol == "^VIX3M" else 0.0
            payload[symbol] = [
                {
                    "d": day_text,
                    "p": round(
                        19.0
                        + offset
                        + 4.0 * math.sin(i / 31.0)
                        + 1.5 * math.cos(i / 11.0),
                        6,
                    ),
                }
                for i, (_, day_text) in enumerate(dated_rows)
            ]
            continue

        price = 50.0 + symbol_index * 3.0
        rows = []
        bond = symbol in _BOND_SYMBOLS
        symbol_rng = random.Random(1000 + symbol_index)
        for i, (day, day_text) in enumerate(dated_rows):
            market_shock = common_shocks[i] * (-0.25 if bond else 1.0)
            idiosyncratic = symbol_rng.gauss(0.0, 0.0047)
            daily_return = (
                (0.0003 if bond else 0.00055) + market_shock + idiosyncratic
            )
            daily_return += _crisis_adjustment(day, bond=bond)
            price *= 1.0 + daily_return
            rows.append({"d": day_text, "p": round(price, 6)})
        payload[symbol] = rows

    return payload


def write_synthetic_prices(destination: Path) -> None:
    """Write the deterministic fixture to an isolated test data directory."""
    with destination.open("w", encoding="utf-8") as output:
        json.dump(build_synthetic_prices(), output, separators=(",", ":"))
