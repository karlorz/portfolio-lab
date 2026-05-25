#!/usr/bin/env python3
"""
Centralized path resolution for portfolio-lab.

All modules should import paths from here instead of hardcoding
absolute or tilde-home paths. Derived from __file__ so the project
works regardless of where it's cloned.

Usage:
    from src.paths import DATA_DIR, MARKET_DB, PRICES_JSON
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

# Repository root (3 levels up from this file: paths.py -> src/ -> repo_root/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Primary data directories
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = PROJECT_ROOT / "public" / "data"
# Common database paths
MARKET_DB = DATA_DIR / "market.db"

# Common data files
PRICES_JSON = PUBLIC_DATA_DIR / "prices.json"
SIGNALS_JSON = PUBLIC_DATA_DIR / "signals.json"
HISTORICAL_JSON = PUBLIC_DATA_DIR / "historical.json"
YIELDS_JSON = PUBLIC_DATA_DIR / "yields.json"

# Subdirectories
SIGNALS_DIR = DATA_DIR / "signals"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
FACTORS_DIR = DATA_DIR / "factors"
CACHE_DIR = DATA_DIR / "cache"
OPTIONS_CACHE_DIR = DATA_DIR / "cache" / "options"
LOCK_DIR = Path(os.environ.get("LOCK_DIR", os.path.join(tempfile.gettempdir(), "portfolio-lab-locks")))
LLM_COSTS_DIR = DATA_DIR / "llm_costs"
ATTRIBUTION_DIR = DATA_DIR / "attribution"

# External directories (user home-based)
HOME = Path.home()
WIKI_DIR = HOME / "wiki"
WORK_DIR = HOME / "projects" / "portfolio-lab" / "work"

# ── Champion allocation ──────────────────────────────────────────────
# Grid-search winner: SPY/GLD/TLT 46/38/16, Sharpe 0.79 (2005-2026)
# Single source of truth — import this instead of repeating the dict.
BASE_ALLOCATION: Dict[str, float] = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

# ── Regime overrides ────────────────────────────────────────────────
# Core strategy regime-based allocation overrides.
# Single source of truth — import from evaluator.py and generator.py.
REGIME_OVERRIDES: Dict[str, Optional[Dict[str, float]]] = {
    "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},  # Risk-off
    "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},  # Defensive
    "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},  # Risk-on
    "normal": None,  # Use BASE_ALLOCATION (46/38/16)
}


def sqlite_connect(db_path: Union[str, Path], **kwargs) -> sqlite3.Connection:
    """Open a SQLite connection with WAL journal mode enabled.

    WAL mode allows concurrent reads during writes, preventing 'database
    is locked' errors under cron-heavy workloads. The PRAGMA is idempotent
    — calling it on an already-WAL database is a no-op.

    Usage::

        from src.paths import sqlite_connect
        with sqlite_connect(MARKET_DB) as conn:
            conn.execute("SELECT ...")

    Args:
        db_path: Path to the SQLite database file.
        **kwargs: Additional keyword arguments forwarded to sqlite3.connect().

    Returns:
        sqlite3.Connection with WAL mode enabled.
    """
    conn = sqlite3.connect(str(db_path), **kwargs)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
