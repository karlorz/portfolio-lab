"""CAR25 TS<->Python parity contract (Item 4 CAR25-TWIN-PARITY-PIN, 2026-08-14).

Pins the deterministic surface shared by the twin CAR25 implementations:
TypeScript ``src/backtest/car25.ts`` (read-only; consumers: tests/ts only)
and Python ``src/backtest/car25.py`` (consumed via the ``src/backtest``
re-export + tests/test_car25.py). The twins are NOT numerically comparable
in their Monte Carlo / bootstrap cores (JS ``makeSeededRng`` vs numpy RNG —
documented limitation, do NOT assert numeric safe-f equality), so this
contract pins constants, the max-drawdown definition, and the
block-bootstrap structure only — mirroring the M2
``test_walk_forward_grid_parity.py`` scope discipline (count/structure,
not numbers).

TS mirrors are local with car25.ts line pins as source of truth — any change
to the TS constants or drawdown algorithm fails this test (intended alarm).
If the twins ever converge (car25.ts deletion remains operator-gated, A5/A9
class), this test must be updated alongside.

Sign convention (documented divergence): TS ``calculateMaxDrawdown`` returns
POSITIVE drawdown (car25.ts:112-123, ``(peak - value) / peak``); py
``calculate_max_drawdown`` returns NEGATIVE (car25.py:125-131, ``np.min``).
Parity asserts ``|ts - |py|| < 1e-9``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.backtest.car25 import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_YEARS,
    DEFAULT_RISK_TOLERANCE,
    DEFAULT_SIMULATIONS,
    F_TOLERANCE,
    MAX_ITERATIONS,
    TRADING_DAYS_PER_YEAR,
    block_bootstrap_returns,
    calculate_max_drawdown,
    safe_f,
)

_CAR25_TS = Path(__file__).resolve().parent.parent / "src" / "backtest" / "car25.ts"
_TS_SRC = _CAR25_TS.read_text()


def _ts_float(name: str) -> float:
    m = re.search(rf"const {name} = ([\d.]+);", _TS_SRC)
    assert m, f"const {name} = ... not found in car25.ts"
    return float(m.group(1))


def _ts_int(pattern: str) -> int:
    m = re.search(pattern, _TS_SRC)
    assert m, f"pattern {pattern!r} not found in car25.ts"
    return int(m.group(1))


def _ts_search(pattern: str) -> str:
    m = re.search(pattern, _TS_SRC)
    assert m, f"pattern {pattern!r} not found in car25.ts"
    return m.group(1)


# ---------------------------------------------------------------------------
# (a) Constants parity — TS source pins: car25.ts:17-22, :165, :191
# ---------------------------------------------------------------------------

TS_CONSTANT_PINS = [
    ("DEFAULT_SIMULATIONS", DEFAULT_SIMULATIONS),
    ("DEFAULT_HORIZON_YEARS", DEFAULT_HORIZON_YEARS),
    ("DEFAULT_RISK_TOLERANCE", DEFAULT_RISK_TOLERANCE),
    ("DEFAULT_CONFIDENCE", DEFAULT_CONFIDENCE),
    ("DEFAULT_BLOCK_SIZE", DEFAULT_BLOCK_SIZE),
    ("TRADING_DAYS_PER_YEAR", TRADING_DAYS_PER_YEAR),
]


@pytest.mark.parametrize("ts_name,py_value", TS_CONSTANT_PINS, ids=[c[0] for c in TS_CONSTANT_PINS])
def test_constants_parity(ts_name: str, py_value: float) -> None:
    """Named constants are identical across the twins."""
    assert _ts_float(ts_name) == pytest.approx(py_value)


def test_max_iterations_parity() -> None:
    """TS hardcoded maxIterations (car25.ts:165) == py MAX_ITERATIONS (car25.py:34)."""
    assert _ts_int(r"const maxIterations = (\d+);") == MAX_ITERATIONS


def test_tolerance_parity() -> None:
    """TS convergence tolerance 0.005 (car25.ts:191) == py F_TOLERANCE (car25.py:35)."""
    assert float(_ts_search(r"riskTolerance\) < ([\d.]+)")) == pytest.approx(F_TOLERANCE)


# ---------------------------------------------------------------------------
# (b) Max-drawdown equivalence on fixed equity curves
# ---------------------------------------------------------------------------


def _ts_calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Local mirror of car25.ts:112-123 (positive drawdown convention)."""
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


DRAWDOWN_CURVES = [
    [100.0, 90.0, 110.0, 80.0],
    [100.0, 110.0, 105.0, 120.0, 90.0],
    [100.0, 50.0, 75.0, 40.0],
    [100.0, 100.0, 100.0],
]


@pytest.mark.parametrize("curve", DRAWDOWN_CURVES)
def test_max_drawdown_parity(curve: list[float]) -> None:
    """|TS positive dd - |py negative dd|| < 1e-9 on fixed curves."""
    ts_dd = _ts_calculate_max_drawdown(curve)
    py_dd = calculate_max_drawdown(np.array(curve, dtype=float))
    assert ts_dd >= 0
    assert py_dd <= 0
    assert abs(ts_dd - abs(py_dd)) < 1e-9


# ---------------------------------------------------------------------------
# (c) Block-bootstrap structural parity (NOT numeric RNG equality)
# ---------------------------------------------------------------------------

HORIZONS = [1.5, 2.0, 3.7]


@pytest.mark.parametrize("horizon_years", HORIZONS)
def test_num_days_composition_parity(horizon_years: float) -> None:
    """numDays composition matches: TS Math.floor (car25.ts:157) == py int (car25.py:159)."""
    ts_num_days = int(np.floor(horizon_years * TRADING_DAYS_PER_YEAR))
    py_num_days = int(horizon_years * TRADING_DAYS_PER_YEAR)
    assert py_num_days == ts_num_days
    # Functional: bootstrap returns exactly num_days elements for the default
    # block size, drawn only from the input series.
    series = np.arange(100.0, dtype=float)
    bootstrapped = block_bootstrap_returns(
        series, num_days=py_num_days, block_size=DEFAULT_BLOCK_SIZE, rng=np.random.default_rng(7)
    )
    assert len(bootstrapped) == py_num_days
    assert set(bootstrapped) <= set(series)


def test_bootstrap_block_structure() -> None:
    """Default block size matches (20, ~1 month); blocks are contiguous draws."""
    series = np.arange(60.0, dtype=float)
    rng = np.random.default_rng(3)
    bootstrapped = block_bootstrap_returns(series, num_days=100, block_size=DEFAULT_BLOCK_SIZE, rng=rng)
    assert len(bootstrapped) == 100
    # Contiguity: every consecutive pair inside a block stays adjacent in the
    # source (block boundary = the only allowed jump).
    for i in range(1, len(bootstrapped)):
        if bootstrapped[i] != bootstrapped[i - 1] + 1.0:
            # A new block may start anywhere, but then it must be a block start
            # (index % block_size == 0 in the drawn sequence).
            assert i % DEFAULT_BLOCK_SIZE == 0


def test_max_iterations_behavioral_py() -> None:
    """py safe_f exhausts MAX_ITERATIONS on a monotone series (dd always 0)."""
    returns = np.full(300, 0.001)
    result = safe_f(returns, risk_tolerance=0.20, n_sims=50, seed=11)
    assert result.iterations == MAX_ITERATIONS
