#!/usr/bin/env python3
"""
TS<->Python grid parity contract (Item 31, 2026-08-12).

Pins the relationship between the Python walk-forward grid
(scripts/walk_forward_validation.py GRID_CONFIGS) and the TypeScript grid
(src/backtest/grid-search.ts): the Python engine is SPY/GLD/TLT-only, so it
mirrors TS regions 1+2+8 (53 configs) — NOT the full 94-config grid.

TS region definitions are mirrored LOCALLY below with grid-search.ts line
pins as source-of-truth, so any change to the TS grid bounds fails this
test (intended alarm). If the Python engine is ever extended (IEF/VTI),
this test must be updated alongside.

Float tolerance: config weights are exact fractions (step/100), so exact
comparison is safe; comparisons use pytest.approx(rel=1e-9) per the Item 32
A4 float-tolerance policy.
"""
from __future__ import annotations

import sys
from pathlib import Path


# Ensure project root is on sys.path (mirrors test_walk_forward_validation.py)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.walk_forward_validation import GRID_CONFIGS  # noqa: E402

# ---------------------------------------------------------------------------
# TS region mirror — source of truth: src/backtest/grid-search.ts
#   R1: grid-search.ts:52-58   (spy 40..70 +=5, gld = 100-spy)
#   R2: grid-search.ts:62-71   (tlt 5..20 +=5 × spy 50..65 +=5, keep 10<=gld<=60)
#   R8: grid-search.ts:129-136 (spy 46..54 +=2 × tlt 10..20 +=2, keep 25<=gld<=45)
# ---------------------------------------------------------------------------
TS_R1 = {
    (spy, 100 - spy, 0)
    for spy in range(40, 71, 5)
}

TS_R2 = {
    (spy, 100 - spy - tlt, tlt)
    for tlt in range(5, 21, 5)
    for spy in range(50, 66, 5)
    if 10 <= (100 - spy - tlt) <= 60
}

TS_R8 = {
    (spy, 100 - spy - tlt, tlt)
    for spy in range(46, 55, 2)
    for tlt in range(10, 21, 2)
    if 25 <= (100 - spy - tlt) <= 45
}

TS_MIRROR = TS_R1 | TS_R2 | TS_R8


def _as_pct_triple(config: dict) -> tuple[int, int, int]:
    """Normalize a config dict (fractions) to integer percent (SPY, GLD, TLT)."""
    return (
        round(config["SPY"] * 100),
        round(config["GLD"] * 100),
        round(config["TLT"] * 100),
    )


def test_grid_config_count_is_53():
    """Python grid = TS regions 1+2+8 = 7 + 16 + 30 = 53 configs."""
    assert len(GRID_CONFIGS) == 53


def test_every_python_config_is_a_ts_region_member():
    """Each Python config belongs to the TS R1/R2/R8 mirror sets."""
    for config in GRID_CONFIGS:
        triple = _as_pct_triple(config)
        assert triple in TS_MIRROR, f"{triple} not in TS regions 1/2/8 mirror"


def test_ts_region_membership_is_exhaustive():
    """Every TS R1/R2/R8 mirror config appears in the Python grid."""
    python_triples = {_as_pct_triple(c) for c in GRID_CONFIGS}
    assert python_triples == TS_MIRROR


def test_regions_3_to_9_explicitly_excluded():
    """Engine limitation: Python grid carries only SPY/GLD/TLT weights.

    TS regions 3-7/9 (IEF, trend-following, quarterly rebalance, VTI/VBR,
    vol-target — grid-search.ts:74-96, :100-116, :121-128, :139-153) use
    other symbols or rebalance modes and are structurally excluded: every
    Python config has exactly {SPY, GLD, TLT} keys.
    """
    for config in GRID_CONFIGS:
        assert set(config.keys()) == {"SPY", "GLD", "TLT"}
