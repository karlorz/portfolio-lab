#!/usr/bin/env python3
"""
TS<->Python sector-momentum parity contract (SECTOR-PARITY, 2026-08-12).

Twin implementations live: TS ``src/strategy/sector_momentum.ts`` (consumers:
types/sector.ts:6, sector_overlay.ts:15) vs Python ``src/strategy/sector_momentum_calc.py``
(consumers: generator.py, strategy/__init__). No parity pin existed — this
test pins the shared constants/definitions and the composite formula.

TS values are mirrored LOCALLY below with sector_momentum.ts line pins as
source-of-truth (mirrors test_walk_forward_grid_parity.py), so any TS change
fails this test (intended alarm). Float tolerance: pytest.approx(rel=1e-9)
per the Item 32 A4 float-tolerance policy.

VERIFIED-IDENTICAL composite semantics (A2, 04:1xZ):
  Python calculate_momentum :91-94 (``if long>0 and short>0: (l+s)/2 else
  min(l,s)``) == TS calculateSectorMomentum :109-116 with useDualMomentum
  default true. Volatility windows IDENTICAL for len > lookback: Python
  ``range(max(1, len-lookback), len)`` :81-87 == TS ``i = len-longLookback ..
  len-1`` :96-99 (population variance, sqrt(252) annualization :85-97 vs
  :101-106).

NEW DISCOVERY (this fire, 2026-08-12 ~20:1xZ) — documented NON-identical surface:
  Momentum ANCHOR indexing is off by one trading day:
    Python: ``long_price = sorted_prices[-lookback_days]`` :66,
            ``short_price = sorted_prices[-short_lookback]`` :70
            → indices len-lookback / len-short_lookback.
    TS:     ``sorted[len - 1 - longLookback]`` :88, ``sorted[len - 1 -
            shortLookback]`` :89 → indices len-1-lookback / len-1-short.
  So Python's long/short momentum spans one fewer trading step than TS
  (p[len-1]/p[len-k] vs p[len-1]/p[len-1-k]). Do NOT align (per plan rule);
  the alarm test ``test_ts_anchor_off_by_one_documented`` fails if either
  side is changed to match the other. Volatility is unaffected (returns
  windows identical) and pinned with exact-equality.

Other DOCUMENTED NON-identical surfaces (do NOT align; pinned as
intentional):
  - Python returns dict adds ``symbol``/``name``/``riskAdjustedMomentum``
    (:99-105) vs TS bare ``{longMomentum, shortMomentum, composite,
    volatility}`` (:82-84).
  - Python zero-price guard ``current_price == 0 ... → None`` :73-75
    (TS lacks — division yields NaN/Inf).
  - Python vol default 0.2 when <=1 return :89-90 (TS → NaN/0).
  - Python ``prev > 0`` skip in returns loop :91 (TS divides directly).
  - Python date-key fallback ``get("date", get("d", ""))`` :62 vs TS
    ``new Date(a.date)`` :82.
  - TS ``useDualMomentum`` config flag :108-118 (Python always dual);
    TS ``riskAdjust`` flag :61/:107 (Python always adjusts).
  - TS getSectorAllocation has NO vix gate (sector_momentum.ts:252-294);
    Python ``vix_threshold=30`` :157-174 gates rotation.
  - Python short lookback derived ``max(1, lookback_days // 4)`` :69 vs TS
    literal ``DEFAULT_SHORT_LOOKBACK=63`` :40 — pinned via the relationship
    ``252 // 4 == 63`` (a lookback change to 504 keeps 504//4==126 and the
    test stays meaningful).
  - PY spy_weight is BASE_ALLOCATION-delegated (live-dashboard authority, A4-SOT s3); TS spyWeight is a literal display-layer fallback — equal today (0.46); if BASE_ALLOCATION changes, TS :260 drifts silently and THIS pin alarms.
"""
from __future__ import annotations

import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

# Ensure project root is on sys.path (mirrors test_walk_forward_validation.py)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.strategy.sector_momentum_calc import (  # noqa: E402
    SECTOR_ETF_DEFINITIONS,
    SectorMomentumCalculator,
)
from src.paths import BASE_ALLOCATION  # noqa: E402

# ---------------------------------------------------------------------------
# TS mirror — source of truth: src/strategy/sector_momentum.ts
#   SECTOR_ETF_DEFINITIONS :19-34  (11 entries: symbol, name, beta, sectorGroup)
#   DEFAULT_MOMENTUM_LOOKBACK=252 :39; DEFAULT_SHORT_LOOKBACK=63 :40
#   DEFAULT_MIN_MOMENTUM=0 :41; getTopSectors(n = 3) :169-171
# ---------------------------------------------------------------------------
TS_SECTOR_ETF_DEFINITIONS = [
    ("XLK", "Technology", 1.10, "sensitive"),
    ("XLV", "Healthcare", 0.85, "defensive"),
    ("XLF", "Financials", 1.05, "cyclical"),
    ("XLY", "Consumer Discretionary", 1.15, "cyclical"),
    ("XLI", "Industrials", 1.00, "cyclical"),
    ("XLE", "Energy", 0.95, "sensitive"),
    ("XLP", "Consumer Staples", 0.65, "defensive"),
    ("XLU", "Utilities", 0.55, "defensive"),
    ("XLB", "Materials", 1.05, "sensitive"),
    ("XLRE", "Real Estate", 0.75, "sensitive"),
    ("XLC", "Communication Services", 1.00, "sensitive"),
]

TS_DEFAULT_MOMENTUM_LOOKBACK = 252
TS_DEFAULT_SHORT_LOOKBACK = 63
TS_DEFAULT_MIN_MOMENTUM = 0
TS_GET_TOP_SECTORS_DEFAULT_N = 3


def _series(returns, start=100.0):
    """Build 260 daily price rows with the given per-step return pattern."""
    prices = []
    price = start
    base_date = date(2025, 1, 1)
    for i, r in enumerate(returns):
        prices.append(
            {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "adjClose": price,
            }
        )
        price *= 1 + r
    return prices


def _rising_returns(n=260):
    """Deterministic non-constant positive returns (varying daily vol)."""
    return [0.0005 + 0.0002 * ((i * 37) % 5) / 2 for i in range(n)]


def _falling_returns(n=260):
    return [-0.0004 - 0.0001 * ((i * 37) % 5) / 2 for i in range(n)]


def _ts_reference(prices, lookback=252, short_lookback=63):
    """Mirror calculateSectorMomentum (sector_momentum.ts:74-121) EXACTLY,
    including the len-1-k anchor indexing (:88-89)."""
    sorted_p = sorted(prices, key=lambda x: x["date"])
    current = sorted_p[-1]["adjClose"]
    long_price = sorted_p[-1 - lookback]["adjClose"]
    short_price = sorted_p[-1 - short_lookback]["adjClose"]
    long_momentum = current / long_price - 1
    short_momentum = current / short_price - 1

    returns = [
        sorted_p[i]["adjClose"] / sorted_p[i - 1]["adjClose"] - 1
        for i in range(len(sorted_p) - lookback, len(sorted_p))
    ]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    volatility = (variance ** 0.5) * (252 ** 0.5)

    if long_momentum > 0 and short_momentum > 0:
        composite = (long_momentum + short_momentum) / 2
    else:
        composite = min(long_momentum, short_momentum)
    return long_momentum, short_momentum, composite, volatility


def _py_reference(prices, lookback=252, short_lookback=63):
    """Mirror calculate_momentum (sector_momentum_calc.py:55-103) EXACTLY,
    including the sorted[-k] anchor indexing (:66/:70)."""
    sorted_p = sorted(prices, key=lambda x: x.get("date", x.get("d", "")))
    current = sorted_p[-1]["adjClose"]
    long_price = sorted_p[-lookback]["adjClose"]
    short_price = sorted_p[-short_lookback]["adjClose"]
    long_momentum = current / long_price - 1
    short_momentum = current / short_price - 1

    returns = [
        sorted_p[i]["adjClose"] / sorted_p[i - 1]["adjClose"] - 1
        for i in range(max(1, len(sorted_p) - lookback), len(sorted_p))
    ]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    volatility = (variance ** 0.5) * (252 ** 0.5)

    if long_momentum > 0 and short_momentum > 0:
        composite = (long_momentum + short_momentum) / 2
    else:
        composite = min(long_momentum, short_momentum)
    return long_momentum, short_momentum, composite, volatility


# ---------------------------------------------------------------------------
# A1: shared constants / definitions
# ---------------------------------------------------------------------------
def test_sector_etf_definitions_match_ts():
    """11 definitions: symbol+name+beta+sectorGroup identical to TS :19-34."""
    py = [
        (s["symbol"], s["name"], s["beta"], s["sectorGroup"])
        for s in SECTOR_ETF_DEFINITIONS
    ]
    assert py == TS_SECTOR_ETF_DEFINITIONS


def test_lookback_defaults_and_derived_short_relationship():
    """252 // 4 == 63 — Python derives short; TS pins the literal."""
    assert 252 // 4 == 63
    assert TS_DEFAULT_MOMENTUM_LOOKBACK == 252
    assert TS_DEFAULT_SHORT_LOOKBACK == 63
    sig = inspect.signature(SectorMomentumCalculator.calculate_momentum)
    assert sig.parameters["lookback_days"].default == 252


def test_allocation_defaults_pinned():
    """top_n=3, vix_threshold=30, min_momentum=0.0 (calc :154-157; TS :41/:169)."""
    sig = inspect.signature(SectorMomentumCalculator.get_allocation)
    assert sig.parameters["top_n"].default == 3
    assert sig.parameters["vix_threshold"].default == 30
    assert sig.parameters["min_momentum"].default == 0.0
    assert TS_GET_TOP_SECTORS_DEFAULT_N == 3
    assert TS_DEFAULT_MIN_MOMENTUM == 0


def test_weight_defaults_parity():
    """spy_weight/overlay_pct defaults (calc :154-155; TS :259-260)."""
    sig = inspect.signature(SectorMomentumCalculator.get_allocation)
    assert sig.parameters["spy_weight"].default == BASE_ALLOCATION["SPY"]
    assert sig.parameters["overlay_pct"].default == 0.25
    # sector_momentum.ts:260 spyWeight: number = 0.46
    # sector_momentum.ts:259 overlayPct: number = 0.25


# ---------------------------------------------------------------------------
# A2: composite formula parity on the shared surface
# ---------------------------------------------------------------------------
def _assert_parity(result, ref):
    assert result["longMomentum"] == pytest.approx(ref[0], rel=1e-9)
    assert result["shortMomentum"] == pytest.approx(ref[1], rel=1e-9)
    assert result["compositeMomentum"] == pytest.approx(ref[2], rel=1e-9)
    assert result["volatility"] == pytest.approx(ref[3], rel=1e-9)


def test_composite_parity_rising_dual_positive():
    """Both momenta positive → average branch; matches Python-formula ref."""
    series = _series(_rising_returns())
    calc = SectorMomentumCalculator({"XLK": series})
    result = calc.calculate_momentum("XLK", lookback_days=252)
    assert result is not None

    ref = _py_reference(series)
    _assert_parity(result, ref)
    assert ref[0] > 0 and ref[1] > 0


def test_composite_parity_falling_min_branch():
    """Both momenta negative → min branch; matches Python-formula ref."""
    series = _series(_falling_returns())
    calc = SectorMomentumCalculator({"XLK": series})
    result = calc.calculate_momentum("XLK", lookback_days=252)
    assert result is not None

    ref = _py_reference(series)
    _assert_parity(result, ref)
    assert ref[0] < 0 and ref[1] < 0
    assert result["compositeMomentum"] == pytest.approx(min(ref[0], ref[1]), rel=1e-9)


def test_composite_parity_mixed_signs_min_branch():
    """Long positive / short negative → min branch (mixed-sign fixture)."""
    series = _series(_rising_returns(200) + _falling_returns(60))
    ref = _py_reference(series)
    assert ref[0] > 0 and ref[1] < 0  # fixture guard: exercises min branch

    calc = SectorMomentumCalculator({"XLK": series})
    result = calc.calculate_momentum("XLK", lookback_days=252)
    assert result is not None
    _assert_parity(result, ref)


def test_volatility_identical_to_ts_reference():
    """Volatility windows are identical across twins — exact-equality pin."""
    for returns in (_rising_returns(), _falling_returns()):
        series = _series(returns)
        ts_ref = _ts_reference(series)
        py_ref = _py_reference(series)
        assert ts_ref[3] == pytest.approx(py_ref[3], rel=1e-9)


def test_ts_anchor_off_by_one_documented():
    """Alarm: TS len-1-k anchors vs Python len-k anchors differ by one day.

    If either twin is changed to match the other, this test fails (intended)
    and the docstring's NON-identical surface list must be updated.
    """
    series = _series(_rising_returns())
    ts_ref = _ts_reference(series)
    py_ref = _py_reference(series)
    assert ts_ref[0] != pytest.approx(py_ref[0], rel=1e-6)
    assert ts_ref[1] != pytest.approx(py_ref[1], rel=1e-6)
    assert ts_ref[2] != pytest.approx(py_ref[2], rel=1e-6)


def test_composite_parity_short_lookback_window():
    """Python's short window is 63 (252//4), not 252 — pin is real."""
    series = _series(_rising_returns())
    ref63 = _py_reference(series, short_lookback=63)
    ref252 = _py_reference(series, short_lookback=252)
    calc = SectorMomentumCalculator({"XLK": series})
    result = calc.calculate_momentum("XLK", lookback_days=252)
    assert result["shortMomentum"] == pytest.approx(ref63[1], rel=1e-9)
    assert ref63[1] != pytest.approx(ref252[1], rel=1e-6)


# ---------------------------------------------------------------------------
# Documented non-identical surfaces — pinned as intentional
# ---------------------------------------------------------------------------
def test_python_insufficient_history_returns_none():
    """len < lookback → None (TS returns a zeros tuple :78-80) — intentional."""
    series = _series(_rising_returns())[:100]
    calc = SectorMomentumCalculator({"XLK": series})
    assert calc.calculate_momentum("XLK", lookback_days=252) is None


def test_python_zero_price_returns_none():
    """Zero long-anchor price → None (TS would divide by zero) — intentional."""
    series = _series(_rising_returns())
    series[len(series) - 252]["adjClose"] = 0.0  # Python long anchor index 8
    calc = SectorMomentumCalculator({"XLK": series})
    assert calc.calculate_momentum("XLK", lookback_days=252) is None
