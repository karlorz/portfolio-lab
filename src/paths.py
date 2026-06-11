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
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

# Repository root (3 levels up from this file: paths.py -> src/ -> repo_root/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Primary data directories
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = Path(os.environ.get("PUBLIC_DATA_DIR", str(PROJECT_ROOT / "public" / "data"))).expanduser()
# Common database paths
MARKET_DB = DATA_DIR / "market.db"
TASKER_DB = DATA_DIR / "tasker.db"

# Common data files
PRICES_JSON = PUBLIC_DATA_DIR / "prices.json"
SIGNALS_JSON = PUBLIC_DATA_DIR / "signals.json"
HISTORICAL_JSON = PUBLIC_DATA_DIR / "historical.json"
YIELDS_JSON = PUBLIC_DATA_DIR / "yields.json"
PUBLIC_TASKER_STATUS_JSON = PUBLIC_DATA_DIR / "tasker_status.json"
TASKER_STATUS_JSON = DATA_DIR / "tasker_status.json"

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


def _parse_skillwiki_path_output(stdout: str) -> Optional[Path]:
    """Parse `skillwiki path` output from either JSON or plain-text CLIs."""
    text = stdout.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return Path(text.splitlines()[0]).expanduser()

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and data.get("path"):
            return Path(str(data["path"])).expanduser()
        if payload.get("path"):
            return Path(str(payload["path"])).expanduser()

    return None


def _looks_like_skillwiki_vault(path: Path) -> bool:
    """Return true only for an existing SkillWiki-style vault root."""
    return path.is_dir() and (path / "SCHEMA.md").is_file() and (path / "projects").is_dir()


def _default_skillwiki_vault_path() -> Path:
    """Return the non-validating fallback path used only for import-time constants."""
    return Path(os.environ.get("WIKI_DIR", str(HOME / "wiki"))).expanduser()


def resolve_skillwiki_vault() -> Path:
    """Resolve the SkillWiki vault root without silently creating wrong paths.

    Precedence:
    1. WIKI_DIR environment override.
    2. `skillwiki path` with a short timeout.
    3. Validated ~/wiki fallback, requiring SCHEMA.md and projects/.
    """
    env_wiki_dir = os.environ.get("WIKI_DIR")
    if env_wiki_dir:
        return Path(env_wiki_dir).expanduser()

    try:
        result = subprocess.run(
            ["skillwiki", "path"],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        result = None

    if result is not None and result.returncode == 0:
        resolved = _parse_skillwiki_path_output(result.stdout)
        if resolved is not None:
            return resolved

    home_wiki = HOME / "wiki"
    if _looks_like_skillwiki_vault(home_wiki):
        return home_wiki

    raise RuntimeError("SkillWiki vault could not be resolved. Set WIKI_DIR or configure `skillwiki path`.")


def _resolve_skillwiki_vault_for_import() -> Path:
    """Best-effort SkillWiki path for module constants.

    Many modules import src.paths only for repo-local constants such as DATA_DIR
    or MARKET_DB. Keep those imports usable on hosts without SkillWiki, while
    explicit vault writers still call require_project_wiki_dir() before writes.
    """
    try:
        return resolve_skillwiki_vault()
    except RuntimeError:
        return _default_skillwiki_vault_path()


def require_skillwiki_vault() -> Path:
    """Resolve and validate the active SkillWiki vault before write operations."""
    vault = resolve_skillwiki_vault()
    if not _looks_like_skillwiki_vault(vault):
        raise RuntimeError(f"Resolved SkillWiki path is not a valid vault root: {vault}")
    return vault


def require_project_wiki_dir() -> Path:
    """Return the canonical portfolio-lab project workspace in the active vault."""
    return require_skillwiki_vault() / "projects" / "portfolio-lab"


WIKI_DIR = _resolve_skillwiki_vault_for_import()
PROJECT_WIKI_DIR = WIKI_DIR / "projects" / "portfolio-lab"
PROJECT_WORK_DIR = PROJECT_WIKI_DIR / "work"
WORK_DIR = Path(os.environ.get("WORK_DIR", str(PROJECT_WORK_DIR))).expanduser()

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

# ── Regime-Conditional Volatility Targeting ───────────────────────────
# Scaling exponents control how aggressively target_vol / realized_vol
# translates into leverage by regime. 1.0 preserves the legacy linear rule.
REGIME_VOL_SCALING_EXPONENTS: Dict[str, float] = {
    "CRISIS": float(os.environ.get("REGIME_VOL_SCALING_CRISIS", "0.5")),
    "HIGH_VOL": float(os.environ.get("REGIME_VOL_SCALING_HIGH_VOL", "0.75")),
    "NORMAL": float(os.environ.get("REGIME_VOL_SCALING_NORMAL", "1.0")),
    "LOW_VOL": float(os.environ.get("REGIME_VOL_SCALING_LOW_VOL", "0.5")),
    "RECOVERY": float(os.environ.get("REGIME_VOL_SCALING_RECOVERY", "0.8")),
}

# Adaptive realized-volatility windows used after regime classification.
REGIME_VOL_LOOKBACKS: Dict[str, int] = {
    "CRISIS": int(os.environ.get("REGIME_VOL_LOOKBACK_CRISIS", "252")),
    "HIGH_VOL": int(os.environ.get("REGIME_VOL_LOOKBACK_HIGH_VOL", "126")),
    "NORMAL": int(os.environ.get("REGIME_VOL_LOOKBACK_NORMAL", "63")),
    "LOW_VOL": int(os.environ.get("REGIME_VOL_LOOKBACK_LOW_VOL", "20")),
    "RECOVERY": int(os.environ.get("REGIME_VOL_LOOKBACK_RECOVERY", "63")),
}

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
