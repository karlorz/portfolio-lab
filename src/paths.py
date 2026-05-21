#!/usr/bin/env python3
"""
Centralized path resolution for portfolio-lab.

All modules should import paths from here instead of hardcoding
absolute or tilde-home paths. Derived from __file__ so the project
works regardless of where it's cloned.

Usage:
    from src.paths import DATA_DIR, MARKET_DB, PRICES_JSON
"""

from pathlib import Path

# Repository root (3 levels up from this file: paths.py -> src/ -> repo_root/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Primary data directories
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DATA_DIR = PROJECT_ROOT / "public" / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

# Common database paths
MARKET_DB = DATA_DIR / "market.db"
VIX_OPTIONS_DB = DATA_DIR / "vix_options.db"
CRYPTO_DB = DATA_DIR / "crypto_allocation.db"
ENSEMBLE_DB = DATA_DIR / "ensemble_signals.db"

# Common data files
PRICES_JSON = PUBLIC_DATA_DIR / "prices.json"
HISTORICAL_JSON = PUBLIC_DATA_DIR / "historical.json"
YIELDS_JSON = PUBLIC_DATA_DIR / "yields.json"

# Subdirectories
SIGNALS_DIR = DATA_DIR / "signals"
POSITIONS_DIR = DATA_DIR / "positions"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
FEATURES_DIR = DATA_DIR / "features"
LLM_COSTS_DIR = DATA_DIR / "llm_costs"
ATTRIBUTION_DIR = DATA_DIR / "attribution"
