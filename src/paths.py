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

# External directories (user home-based, configurable via env vars)
HOME = Path.home()
WIKI_DIR = Path(os.environ.get("WIKI_DIR", str(HOME / "wiki")))
WORK_DIR = Path(os.environ.get("WORK_DIR", str(HOME / "projects" / "portfolio-lab" / "work")))

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

# ── Shared strategy parameters ────────────────────────────────────────
# Used by tsmom_overlay.py and multi_speed_momentum.py.
# Single source of truth — import from here instead of repeating.
VOL_TARGET: float = float(os.environ.get("VOL_TARGET", "0.15"))       # 15% target volatility
MAX_DEVIATION: float = float(os.environ.get("MAX_DEVIATION", "0.10")) # ±10% max allocation drift
MIN_WEIGHT: float = float(os.environ.get("MIN_WEIGHT", "0.05"))       # Minimum 5% per asset
REBALANCE_FREQ: int = int(os.environ.get("REBALANCE_FREQ", "21"))     # Monthly rebalancing

# ── VIX regime thresholds ─────────────────────────────────────────────
# Used by evaluator.py get_current_regime() and generator.py.
# Single source of truth — import from here instead of hardcoding.
VIX_CRISIS_THRESHOLD: float = float(os.environ.get("VIX_CRISIS_THRESHOLD", "25.0"))
VIX_VOL_SPIKE_THRESHOLD: float = float(os.environ.get("VIX_VOL_SPIKE_THRESHOLD", "20.0"))
VIX_LOW_VOL_THRESHOLD: float = float(os.environ.get("VIX_LOW_VOL_THRESHOLD", "15.0"))

# ── Risk-Free Rate ─────────────────────────────────────────────────────
# Single source of truth for Sharpe ratio computation.
# Default is 4.5% (current ~1yr Treasury yield as of May 2026).
# Override via RISK_FREE_RATE env var (e.g., "5.0" for 5%).
RISK_FREE_RATE: float = float(os.environ.get("RISK_FREE_RATE", "4.5"))

# ── Ensemble Voter Regime Thresholds ──────────────────────────────────
# Used by ensemble_voter.py EnsembleVoter class for regime detection.
# Single source of truth — import from here instead of hardcoding.
ENSEMBLE_CRISIS_VOL_THRESHOLD: float = float(os.environ.get("ENSEMBLE_CRISIS_VOL_THRESHOLD", "0.30"))
ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD: float = float(os.environ.get("ENSEMBLE_CRISIS_DRAWDOWN_THRESHOLD", "-0.10"))
ENSEMBLE_HIGH_VOL_VOL_THRESHOLD: float = float(os.environ.get("ENSEMBLE_HIGH_VOL_VOL_THRESHOLD", "0.20"))
ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD: float = float(os.environ.get("ENSEMBLE_HIGH_VOL_DRAWDOWN_THRESHOLD", "-0.05"))
ENSEMBLE_LOW_VOL_VOL_THRESHOLD: float = float(os.environ.get("ENSEMBLE_LOW_VOL_VOL_THRESHOLD", "0.12"))
ENSEMBLE_LOW_VOL_MOM_THRESHOLD: float = float(os.environ.get("ENSEMBLE_LOW_VOL_MOM_THRESHOLD", "0.01"))
ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD: float = float(os.environ.get("ENSEMBLE_RECOVERY_DRAWDOWN_THRESHOLD", "-0.03"))
ENSEMBLE_RECOVERY_MOM_THRESHOLD: float = float(os.environ.get("ENSEMBLE_RECOVERY_MOM_THRESHOLD", "0.02"))

# ── Ensemble Voter Consensus ─────────────────────────────────────────
# Fraction of weighted signals that must agree for action.
ENSEMBLE_CONSENSUS_THRESHOLD: float = float(os.environ.get("ENSEMBLE_CONSENSUS_THRESHOLD", str(2/3)))


def sqlite_connect(db_path: Union[str, Path], **kwargs) -> sqlite3.Connection:
    """Open a SQLite connection with WAL journal mode enabled.

    WAL mode allows concurrent reads during writes, preventing 'database
    is locked' errors under cron-heavy workloads. The PRAGMA is idempotent
    — calling it on an already-WAL database is a no-op.

    Usage::

        from src.paths import sqlite_connect
        with sqlite_connect(MARKET_DB) as conn:
            conn.execute("SELECT ...")

    Note: ``Connection.__exit__`` commits/rollbacks but does NOT close the
    connection.  For guaranteed close, use ``with closing(sqlite_connect(...))``
    or call ``conn.close()`` explicitly.

    Args:
        db_path: Path to the SQLite database file.
        **kwargs: Additional keyword arguments forwarded to sqlite3.connect().

    Returns:
        sqlite3.Connection with WAL mode enabled.
    """
    conn = sqlite3.connect(str(db_path), **kwargs)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
