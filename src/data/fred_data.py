"""
FRED-MD Macroeconomic Data Fetcher for Portfolio-Lab v2.70

Fetches and caches FRED-MD macro series for regime detection.
Uses fredapi (MIT license, free FRED API key required for live data).

Designed for Phase 1 of v970: Data Infrastructure
Regime mapping follows arXiv 2503.11499 two-stage k-means approach.

Usage:
    from src.data.fred_data import FredMdFetcher
    fetcher = FredMdFetcher(api_key="YOUR_KEY")
    data = fetcher.get_regime_indicators()  # Returns DataFrame of all series
"""

import json
import os
import sqlite3
import time
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from src.paths import MARKET_DB, DATA_DIR, sqlite_connect

logger = logging.getLogger(__name__)

# ── FRED-MD Series Definitions ──────────────────────────────────────────
# 120+ monthly series across 8 categories from McCracken & Ng (2016)
# Mapped to project's 5-regime system: CRISIS/HIGH_VOL/NORMAL/RECOVERY/LOW_VOL

# Default series by regime type
DEFAULT_FRED_SERIES: Dict[str, List[str]] = {
    "crisis": [
        "RECPROUSM156N",  # US Recession Probability
        "INDPRO",         # Industrial Production Index
        "CLAIMS",         # Initial Claims
        "BAASPREAD",      # BAA Corporate Bond Spread
    ],
    "high_vol": [
        "VIXCLS",         # CBOE Volatility Index (VIX)
        "BAASPREAD",      # BAA Corporate Bond Spread
        "OILPRICEx",      # Crude Oil Price
    ],
    "inflation": [
        "CPIAUCSL",       # CPI All Urban Consumers
        "CPILFESL",       # CPI Less Food & Energy
        "FEDFUNDS",       # Federal Funds Rate
        "TBSPR",          # 10Y-3M Treasury Spread
    ],
    "recovery": [
        "INDPRO",         # Industrial Production Index
        "PAYEMS",         # All Employees: Total Nonfarm
        "RRSFS",          # Real Retail and Food Services Sales
        "NAPMI",          # ISM Manufacturing PMI
    ],
    "low_vol": [
        "NAPMI",          # ISM Manufacturing PMI
        "T10Y2Y",         # 10Y-2Y Treasury Spread
        "FEDFUNDS",       # Federal Funds Rate
        "DTWEXBGS",       # Trade Weighted US Dollar Index
    ],
}

# All unique series across all regimes
ALL_FRED_SERIES: List[str] = sorted(set(
    s for series_list in DEFAULT_FRED_SERIES.values() for s in series_list
))

# Series metadata
SERIES_METADATA: Dict[str, Dict[str, Any]] = {
    "RECPROUSM156N": {"name": "Recession Probability", "freq": "monthly", "units": "percent"},
    "INDPRO": {"name": "Industrial Production Index", "freq": "monthly", "units": "index 2017=100"},
    "CLAIMS": {"name": "Initial Claims", "freq": "weekly", "units": "thousands"},
    "BAASPREAD": {"name": "BAA Corporate Bond Spread", "freq": "monthly", "units": "percent"},
    "VIXCLS": {"name": "CBOE VIX Index", "freq": "monthly_avg", "units": "index"},
    "OILPRICEx": {"name": "Crude Oil Price", "freq": "monthly", "units": "dollars/barrel"},
    "CPIAUCSL": {"name": "CPI All Items", "freq": "monthly", "units": "index 1982-84=100"},
    "CPILFESL": {"name": "CPI Core", "freq": "monthly", "units": "index 1982-84=100"},
    "FEDFUNDS": {"name": "Federal Funds Rate", "freq": "monthly", "units": "percent"},
    "TBSPR": {"name": "10Y-3M Treasury Spread", "freq": "monthly", "units": "percent"},
    "PAYEMS": {"name": "Nonfarm Payrolls", "freq": "monthly", "units": "thousands"},
    "RRSFS": {"name": "Real Retail Sales", "freq": "monthly", "units": "millions $"},
    "NAPMI": {"name": "ISM Manufacturing PMI", "freq": "monthly", "units": "index"},
    "T10Y2Y": {"name": "10Y-2Y Treasury Spread", "freq": "monthly", "units": "percent"},
    "DTWEXBGS": {"name": "Trade Weighted USD Index", "freq": "monthly", "units": "index"},
}

# Cache configuration
FRED_CACHE_TABLE = "fred_cache"
FRED_CACHE_TTL_HOURS = 24  # Monthly series can be cached for a day

# Regime detection thresholds (from arXiv 2503.11499 analysis)
# These will be refined in Phase 2
REGIME_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "recession_prob": {"crisis": 30.0, "high_vol": 15.0},
    "baa_spread": {"crisis": 3.5, "high_vol": 2.5},
    "vix": {"high_vol": 25.0, "normal": 20.0},
    "inflation_yoy": {"high": 4.0, "moderate": 2.5},
    "fed_rate": {"tight": 4.0, "normal": 2.0},
    "pmi": {"expansion": 50.0, "recovery": 45.0},
}

FRED_AVAILABLE = False
try:
    from fredapi import Fred as _Fred
    from src.utils.rate_limiter import rate_limited, retry_on_api_error
    FRED_AVAILABLE = True
except ImportError:
    _Fred = None  # type: ignore
    logger.warning("fredapi not installed. FRED-MD fetcher disabled.")


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FredSeriesObservation:
    """Single observation for a FRED series."""
    series_id: str
    date: str
    value: float
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class FredSignal:
    """Regime signal derived from FRED-MD data."""
    timestamp: str
    regime: str
    confidence: float
    indicators: Dict[str, float]
    recession_probability: float
    inflation_pressure: float
    monetary_stance: str  # "tight", "neutral", "accommodative"
    manufacturing_health: float  # PMI-based (0-100)
    credit_conditions: str  # "tight", "normal", "loose"


# ── Cache Helpers ───────────────────────────────────────────────────────

def _init_cache_table() -> None:
    """Create FRED cache table if it doesn't exist."""
    try:
        with sqlite_connect(MARKET_DB) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {FRED_CACHE_TABLE} (
                    series_id TEXT PRIMARY KEY,
                    json_data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    except Exception as e:
        logger.error(f"Failed to init FRED cache table: {e}")


def _get_cached_series(series_id: str) -> Optional[pd.Series]:
    """Retrieve cached series data if fresh."""
    try:
        with sqlite_connect(MARKET_DB) as conn:
            row = conn.execute(
                f"SELECT json_data, fetched_at FROM {FRED_CACHE_TABLE} WHERE series_id = ?",
                (series_id,),
            ).fetchone()
            if row is None:
                return None
            json_data, fetched_at = row
            # Check TTL
            fetched_dt = datetime.fromisoformat(fetched_at)
            age_hours = (datetime.now(timezone.utc) - fetched_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if age_hours > FRED_CACHE_TTL_HOURS:
                return None
            data_dict = json.loads(json_data)
            return pd.Series(data_dict)
    except Exception as e:
        logger.warning(f"Cache read error for {series_id}: {e}")
        return None


def _set_cached_series(series_id: str, series: pd.Series) -> None:
    """Store series data in cache."""
    try:
        data_dict = series.to_dict()
        # Convert Timestamps to strings for JSON serialization
        clean_dict = {}
        for k, v in data_dict.items():
            if hasattr(k, 'isoformat'):
                k = str(k)
            if hasattr(v, 'isoformat'):
                v = str(v)
            if pd.isna(v):
                v = None
            clean_dict[str(k)] = v
        json_data = json.dumps(clean_dict)
        with sqlite_connect(MARKET_DB) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {FRED_CACHE_TABLE} (series_id, json_data) VALUES (?, ?)",
                (series_id, json_data),
            )
    except Exception as e:
        logger.warning(f"Failed to cache {series_id}: {e}")


def _parse_fred_cache_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_fred_md_cache_health(
    db_path: str | Path = MARKET_DB,
    *,
    now: datetime | None = None,
    ttl_hours: int = FRED_CACHE_TTL_HOURS,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return a read-only FRED-MD cache health summary.

    Status values are intentionally operator-facing:
    - ``ok``: cache has rows and latest fetch is within TTL
    - ``stale``: cache has rows but latest fetch is older than TTL
    - ``empty``: cache table exists but has no rows
    - ``unavailable``: table/database cannot be read
    """
    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    resolved_now = resolved_now.astimezone(timezone.utc)
    api_key_configured = bool(api_key if api_key is not None else os.environ.get("FRED_API_KEY"))
    base = {
        "status": "unavailable",
        "row_count": 0,
        "latest_fetched_at": None,
        "age_hours": None,
        "ttl_hours": ttl_hours,
        "fredapi_available": FRED_AVAILABLE,
        "api_key_configured": api_key_configured,
        "source_mode": "unavailable",
        "reason": None,
    }

    try:
        with sqlite_connect(str(db_path)) as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (FRED_CACHE_TABLE,),
            ).fetchone()
            if table is None:
                return {**base, "reason": "missing_table"}

            row = conn.execute(
                f"SELECT COUNT(*) AS row_count, MAX(fetched_at) AS latest_fetched_at FROM {FRED_CACHE_TABLE}",
            ).fetchone()
    except sqlite3.Error as exc:
        return {**base, "reason": f"sqlite_error: {exc}"}

    row_count = int(row[0] or 0) if row else 0
    latest_raw = row[1] if row else None
    if row_count == 0:
        return {
            **base,
            "status": "empty",
            "source_mode": "unavailable",
            "reason": "empty_cache",
        }

    latest = _parse_fred_cache_timestamp(latest_raw)
    if latest is None:
        return {
            **base,
            "row_count": row_count,
            "latest_fetched_at": latest_raw,
            "reason": "invalid_latest_fetched_at",
        }

    age_hours = (resolved_now - latest).total_seconds() / 3600
    status = "ok" if age_hours <= ttl_hours else "stale"
    return {
        **base,
        "status": status,
        "row_count": row_count,
        "latest_fetched_at": latest.isoformat(),
        "age_hours": round(age_hours, 2),
        "source_mode": "cached" if status == "ok" else "stale_cached",
        "reason": None if status == "ok" else "cache_stale",
    }


# ── FRED-MD Fetcher ─────────────────────────────────────────────────────

class FredMdFetcher:
    """
    FRED-MD Macroeconomic Data Fetcher and Regime Indicator Generator.

    Fetches, caches, and processes FRED-MD macro series for regime detection.

    Args:
        api_key: FRED API key (free from fred.stlouisfed.org)
        use_cache: Whether to use SQLite cache (default: True)
    """

    def __init__(self, api_key: str = "", use_cache: bool = True):
        self._api_key = api_key
        self.use_cache = use_cache
        self._fred_client = None
        self._initialized = False
        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize cache infrastructure."""
        if self.use_cache:
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                _init_cache_table()
            except Exception as e:
                logger.warning(f"Cache init failed (continuing without cache): {e}")
                self.use_cache = False

    @property
    def fred_client(self):
        """Lazy-initialized FRED API client."""
        if self._fred_client is None:
            if not self._api_key:
                raise ValueError(
                    "FRED API key required. Set via FredMdFetcher(api_key=...) "
                    "or FRED_API_KEY env var."
                )
            if _Fred is None:
                raise ImportError("fredapi not installed. Run: uv add fredapi")
            self._fred_client = _Fred(api_key=self._api_key)
        return self._fred_client

    def get_series(self, series_id: str, cache_ok: bool = True) -> pd.Series:
        """
        Fetch a FRED series, using cache if available.

        Returns a pandas Series indexed by date with values.
        Returns empty Series on failure.
        """
        # Try cache first
        if cache_ok and self.use_cache:
            cached = _get_cached_series(series_id)
            if cached is not None and len(cached) > 0:
                logger.debug(f"Cache hit for {series_id}")
                return cached

        # Fetch from FRED API
        try:
            data = self.fred_client.get_series(series_id)
            if data is not None and len(data) > 0:
                if self.use_cache:
                    _set_cached_series(series_id, data)
                logger.info(f"Fetched {series_id}: {len(data)} observations")
                return data
            else:
                logger.warning(f"Empty data for {series_id}")
                return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.Series(dtype=float)

    def get_all_series(self, series_list: Optional[List[str]] = None,
                       cache_ok: bool = True) -> pd.DataFrame:
        """
        Fetch all specified FRED series into a single DataFrame.

        Args:
            series_list: List of series IDs (default: ALL_FRED_SERIES)
            cache_ok: Whether to check/use cache

        Returns:
            DataFrame with dates as index, series IDs as columns
        """
        series_list = series_list or ALL_FRED_SERIES
        result = pd.DataFrame()

        for sid in series_list:
            series = self.get_series(sid, cache_ok=cache_ok)
            if len(series) > 0:
                result[sid] = series

        return result

    def get_regime_indicators(self, cache_ok: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch indicators grouped by regime type.

        Returns dict mapping regime types to DataFrames of their indicators.
        """
        indicators: Dict[str, pd.DataFrame] = {}
        for regime, series_list in DEFAULT_FRED_SERIES.items():
            df = self.get_all_series(series_list, cache_ok=cache_ok)
            if not df.empty:
                indicators[regime] = df
        return indicators

    def compute_recession_probability(self, cache_ok: bool = True) -> Optional[float]:
        """
        Compute current recession probability from RECPROUSM156N.

        Returns None if data unavailable.
        """
        recprob = self.get_series("RECPROUSM156N", cache_ok=cache_ok)
        if len(recprob) == 0:
            return None
        return float(recprob.iloc[-1])

    def compute_inflation_pressure(self, cache_ok: bool = True) -> Optional[float]:
        """
        Compute trailing 12-month CPI inflation rate.

        Returns None if data unavailable.
        """
        cpi = self.get_series("CPIAUCSL", cache_ok=cache_ok)
        if len(cpi) < 13:
            return None
        # Year-over-year change
        latest = float(cpi.iloc[-1])
        year_ago = float(cpi.iloc[-13])
        return ((latest - year_ago) / year_ago) * 100.0

    def compute_pmi_health(self, cache_ok: bool = True) -> Optional[float]:
        """
        Get latest ISM Manufacturing PMI reading.

        PMI > 50 = expansion, < 50 = contraction.
        """
        pmi = self.get_series("NAPMI", cache_ok=cache_ok)
        if len(pmi) == 0:
            return None
        return float(pmi.iloc[-1])

    def compute_monetary_stance(self, cache_ok: bool = True) -> str:
        """
        Determine monetary policy stance from Fed Funds rate.

        Returns "tight", "neutral", or "accommodative".
        """
        fed_rate_series = self.get_series("FEDFUNDS", cache_ok=cache_ok)
        if len(fed_rate_series) == 0:
            return "unknown"
        rate = float(fed_rate_series.iloc[-1])
        if rate >= 4.0:
            return "tight"
        elif rate >= 2.0:
            return "neutral"
        else:
            return "accommodative"

    def compute_credit_conditions(self, cache_ok: bool = True) -> str:
        """
        Determine credit conditions from BAA spread.

        Returns "tight", "normal", or "loose".
        """
        spread = self.get_series("BAASPREAD", cache_ok=cache_ok)
        if len(spread) == 0:
            return "unknown"
        val = float(spread.iloc[-1])
        if val >= 3.5:
            return "tight"
        elif val >= 2.0:
            return "normal"
        else:
            return "loose"

    def compute_regime_signal(self, cache_ok: bool = True) -> FredSignal:
        """
        Compute composite regime signal from all FRED-MD indicators.

        This is a preliminary implementation (Phase 1).
        Phase 2 will implement the full two-stage k-means algorithm
        from arXiv 2503.11499.

        Returns:
            FredSignal with regime classification and indicators
        """
        # Fetch all indicators
        recprob = self.compute_recession_probability(cache_ok=cache_ok)
        inflation = self.compute_inflation_pressure(cache_ok=cache_ok)
        pmi = self.compute_pmi_health(cache_ok=cache_ok)
        monet_stance = self.compute_monetary_stance(cache_ok=cache_ok)
        credit = self.compute_credit_conditions(cache_ok=cache_ok)

        # Build indicators dict
        indicators = {}
        if recprob is not None:
            indicators["recession_probability"] = recprob
        if inflation is not None:
            indicators["inflation_yoy"] = inflation
        if pmi is not None:
            indicators["pmi"] = pmi

        # Preliminary regime classification (will be replaced in Phase 2)
        regime = "NORMAL"
        confidence = 0.5

        if recprob is not None and recprob > REGIME_THRESHOLDS["recession_prob"]["crisis"]:
            regime = "CRISIS"
            confidence = 0.7
        elif credit == "tight" and (pmi is None or pmi < 45):
            regime = "CRISIS"
            confidence = 0.6
        elif recprob is not None and recprob > REGIME_THRESHOLDS["recession_prob"]["high_vol"]:
            regime = "HIGH_VOL"
            confidence = 0.6
        elif credit == "tight":
            regime = "HIGH_VOL"
            confidence = 0.5
        elif pmi is not None and pmi > 55 and credit == "loose":
            regime = "LOW_VOL"
            confidence = 0.6
        elif pmi is not None and pmi > REGIME_THRESHOLDS["pmi"]["expansion"]:
            # Expansion — check if it's recovery-like
            # Recovery typically has strong PMI growth from low base
            regime = "RECOVERY" if recprob is not None and recprob > 10 else "NORMAL"
            confidence = 0.5

        return FredSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            recession_probability=recprob or 0.0,
            inflation_pressure=inflation or 0.0,
            monetary_stance=monet_stance,
            manufacturing_health=pmi or 50.0,
            credit_conditions=credit,
        )


def get_fred_signal(fetcher: Optional[FredMdFetcher] = None,
                    api_key: str = "") -> FredSignal:
    """
    Convenience function: get current FRED-MD regime signal.

    Creates fetcher if not provided. Falls back to empty signal on failure.

    Args:
        fetcher: Existing FredMdFetcher instance (optional)
        api_key: FRED API key (used if no fetcher provided)

    Returns:
        FredSignal with regime classification
    """
    try:
        if fetcher is None:
            fetcher = FredMdFetcher(api_key=api_key, use_cache=True)
        return fetcher.compute_regime_signal()
    except Exception as e:
        logger.error(f"Failed to compute FRED-MD regime signal: {e}")
        return FredSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime="UNKNOWN",
            confidence=0.0,
            indicators={},
            recession_probability=0.0,
            inflation_pressure=0.0,
            monetary_stance="unknown",
            manufacturing_health=50.0,
            credit_conditions="unknown",
        )
