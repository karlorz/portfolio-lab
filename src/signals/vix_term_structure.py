"""
VIX Term Structure Signal Generator - v4.50 Implementation
Generates tactical overlay signals based on VIX/VIX3M/VIX6M term structure slope.

Target: +0.03 to +0.04 Sharpe improvement through drawdown avoidance.
Based on research: VIX term structure slope predicts equity returns better than absolute VIX level.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from src.paths import DATA_DIR, SIGNALS_DIR, MARKET_DB, sqlite_connect
from src.backtest.metrics import save_results_json
import numpy as np


__all__ = ['VIXRegime', 'VIXSignalState', 'VIXTermStructureSignal', 'VIXTermStructureCalculator', 'VIXTermStructureSignalGenerator']

logger = logging.getLogger(__name__)


class VIXRegime(Enum):
    """VIX term structure regime classification."""
    EXTREME_CONTANGO = "extreme_contango"      # VIX3M/VIX > 1.15 (complacency)
    CONTANGO = "contango"                       # VIX3M/VIX 1.0-1.15 (normal)
    FLAT = "flat"                               # VIX3M/VIX 0.95-1.0 (neutral)
    BACKWARDATION = "backwardation"             # VIX3M/VIX 0.8-0.95 (caution)
    EXTREME_BACKWARDATION = "extreme_backwardation"  # VIX3M/VIX < 0.8 (crisis)


class VIXSignalState(Enum):
    """Signal states for portfolio overlay."""
    RISK_ON = 1         # Increase equity exposure
    NEUTRAL = 0         # Maintain baseline
    RISK_OFF = -1       # Reduce equity, add defensive


@dataclass
class VIXTermStructureSignal:
    """Complete VIX term structure signal with tactical recommendation."""
    timestamp: str
    signal_state: str  # risk_on, neutral, risk_off
    signal_value: float  # -1.0 to +1.0
    
    # Raw inputs
    vix_spot: float
    vix3m: Optional[float]
    vix6m: Optional[float]
    slope_vix3m_vix: float  # VIX3M / VIX ratio
    
    # Regime classification
    regime: str
    regime_strength: float  # 0-1
    
    # Composite components
    slope_signal: float  # -1 to +1
    roll_yield_signal: float
    vix_zscore_signal: float
    curve_shape_signal: float
    
    # Portfolio overlay recommendation
    spy_shift: float  # Percentage point shift (-0.10 to +0.05)
    gld_shift: float
    tlt_shift: float
    
    # Confidence and constraints
    confidence: float  # 0-100%
    is_valid: bool
    reason: str
    
    def to_dict(self) -> dict:
        return asdict(self)

    def _confidence_fraction(self) -> float:
        """Normalize percent-style VIX confidence for typed ensemble consumers."""
        confidence = float(self.confidence)
        if confidence > 1.0:
            confidence /= 100.0
        return max(0.0, min(1.0, confidence))

    def to_signal_snapshot(self):
        """Convert to canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot

        return SignalSnapshot(
            source="vix_term_structure",
            timestamp=self.timestamp,
            value=self.signal_value,
            confidence=self._confidence_fraction(),
            asset_signals={
                "SPY": self.spy_shift,
                "GLD": self.gld_shift,
                "TLT": self.tlt_shift,
            },
            regime_fit="all",
            is_active=self.is_valid,
            explanation=f"VIX TS: {self.signal_state}, "
                        f"regime={self.regime}({self.regime_strength:.2f}), "
                        f"VIX={self.vix_spot:.1f}, "
                        f"slope={self.slope_vix3m_vix:.3f}",
            metadata={
                "signal_state": self.signal_state,
                "regime": self.regime,
                "regime_strength": self.regime_strength,
                "vix_spot": self.vix_spot,
                "slope_signal": self.slope_signal,
                "roll_yield_signal": self.roll_yield_signal,
                **(
                    getattr(self, "_freshness", None)
                    if isinstance(getattr(self, "_freshness", None), dict)
                    else {}
                ),
            },
        )


class VIXTermStructureCalculator:
    """
    Calculates VIX term structure slope and generates tactical signals.
    
    Key insight: VIX3M/VIX ratio predicts equity returns better than spot VIX.
    - Backwardation (VIX > VIX3M): Risk-off, reduce equity
    - Contango (VIX < VIX3M): Risk-on or neutral depending on steepness
    """
    
    # Signal thresholds based on research
    EXTREME_CONTANGO_THRESHOLD = 1.15   # Complacency warning
    CONTANGO_THRESHOLD = 1.00           # Normal market
    FLAT_UPPER = 1.00
    FLAT_LOWER = 0.95
    BACKWARDATION_THRESHOLD = 0.80      # Risk-off warning
    
    # VIX level context
    VIX_CHEAP = 16.0
    VIX_FAIR = 20.0
    VIX_EXPENSIVE = 25.0
    
    def __init__(self, history_days: int = 252):
        self.history_days = history_days
        self.vix_history: List[Tuple[str, float]] = []
    
    def add_vix_reading(self, date: str, vix: float):
        """Add VIX reading to history for Z-score calculation."""
        self.vix_history.append((date, vix))
        if len(self.vix_history) > self.history_days:
            self.vix_history.pop(0)
    
    def calculate_slope_signal(self, vix: float, vix3m: float) -> float:
        """
        Map VIX3M/VIX ratio to [-1, +1] signal.
        
        < 0.85: Extreme backwardation (risk-off) -> -1
        0.85-1.0: Backwardation (caution) -> -0.5 to 0
        1.0-1.15: Normal contango -> 0 to +0.5
        > 1.15: Steep contango (complacency) -> +0.5 to +1
        """
        if vix <= 0 or vix3m <= 0:
            return 0.0
        
        slope = vix3m / vix
        
        if slope < 0.85:
            return -1.0
        elif slope < 1.0:
            # Linear interpolation from -1.0 to -0.5
            return -1.0 + (slope - 0.85) / 0.15 * 0.5
        elif slope < 1.15:
            # Linear interpolation from 0 to +0.5
            return (slope - 1.0) / 0.15 * 0.5
        else:
            # Cap at +1.0 for extreme contango
            return min(0.5 + (slope - 1.15) / 0.15 * 0.5, 1.0)
    
    def calculate_roll_yield_signal(self, vix: float, vix3m: float) -> float:
        """
        Roll yield signal: (VIX3M - VIX) / VIX3M normalized to [-1, 1].
        Positive = contango (futures > spot), negative = backwardation.
        """
        if vix3m <= 0:
            return 0.0
        
        roll_yield = (vix3m - vix) / vix3m
        # Normalize: typical range -0.2 to +0.2
        return max(-1.0, min(1.0, roll_yield * 5))
    
    def calculate_vix_zscore_signal(self, vix: float) -> float:
        """
        VIX Z-score relative to 1-year history, mapped to [-1, 1].
        High VIX = negative signal (risk-off), low VIX = positive (risk-on).
        """
        if len(self.vix_history) < 60:  # Need at least 60 days
            return 0.0
        
        vix_values = [v for _, v in self.vix_history]
        mean_vix = np.mean(vix_values)
        std_vix = np.std(vix_values)
        
        if std_vix == 0:
            return 0.0
        
        zscore = (vix - mean_vix) / std_vix
        # Invert: high VIX = risk-off (-1), low VIX = risk-on (+1)
        # Typical Z-score range: -2 to +2
        signal = -max(-1.0, min(1.0, float(zscore) / 2))
        return signal
    
    def calculate_curve_shape_signal(self, vix3m: float, vix6m: Optional[float]) -> float:
        """
        Curve shape using VIX6M/VIX3M if available.
        Steepening = risk building, flattening = normalization.
        """
        if vix6m is None or vix3m <= 0:
            return 0.0
        
        curve_shape = vix6m / vix3m
        # Normalize around 1.0, typical range 0.9 to 1.1
        return max(-1.0, min(1.0, (curve_shape - 1.0) * 10))
    
    def classify_regime(self, slope: float) -> Tuple[VIXRegime, float]:
        """Classify VIX term structure regime and return strength."""
        if slope >= self.EXTREME_CONTANGO_THRESHOLD:
            strength = min(1.0, (slope - 1.15) / 0.15 + 0.5)
            return VIXRegime.EXTREME_CONTANGO, strength
        elif slope >= self.CONTANGO_THRESHOLD:
            strength = (slope - 1.0) / 0.15
            return VIXRegime.CONTANGO, strength
        elif slope >= self.FLAT_LOWER:
            strength = (1.0 - slope) / 0.05
            return VIXRegime.FLAT, strength
        elif slope >= self.BACKWARDATION_THRESHOLD:
            strength = (0.95 - slope) / 0.15
            return VIXRegime.BACKWARDATION, strength
        else:
            strength = min(1.0, (0.80 - slope) / 0.10 + 0.5)
            return VIXRegime.EXTREME_BACKWARDATION, strength
    
    def calculate_composite_signal(
        self,
        vix: float,
        vix3m: Optional[float],
        vix6m: Optional[float],
        date: str
    ) -> Dict:
        """
        Calculate composite signal using weighted components.
        
        Weights based on research:
        - Slope: 40% (primary predictor)
        - Roll yield: 25% (carry signal)
        - VIX Z-score: 20% (absolute vol context)
        - Curve shape: 15% (confirmation)
        """
        if vix3m is None or vix3m <= 0:
            logger.warning("[%s] VIX3M unavailable, using VIX spot proxy", date)
            # Use VIX level as fallback
            if vix < self.VIX_CHEAP:
                slope_signal = 0.5  # Complacent
            elif vix < self.VIX_FAIR:
                slope_signal = 0.0  # Normal
            elif vix < self.VIX_EXPENSIVE:
                slope_signal = -0.3  # Elevated
            else:
                slope_signal = -0.8  # Stress
            vix3m = vix * (1.1 if slope_signal > 0 else 0.9)
        
        # Calculate individual signals
        slope_signal = self.calculate_slope_signal(vix, vix3m)
        roll_signal = self.calculate_roll_yield_signal(vix, vix3m)
        zscore_signal = self.calculate_vix_zscore_signal(vix)
        curve_signal = self.calculate_curve_shape_signal(vix3m, vix6m)
        
        # Weighted composite
        composite = (
            0.40 * slope_signal +
            0.25 * roll_signal +
            0.20 * zscore_signal +
            0.15 * curve_signal
        )
        
        # Bound to [-1, 1]
        composite = max(-1.0, min(1.0, composite))
        
        return {
            "composite": composite,
            "slope_signal": slope_signal,
            "roll_yield_signal": roll_signal,
            "vix_zscore_signal": zscore_signal,
            "curve_shape_signal": curve_signal,
            "slope": vix3m / vix if vix > 0 else 1.0
        }
    
    def get_allocation_shifts(self, signal: float) -> Dict[str, float]:
        """
        Map composite signal to allocation shifts.
        
        Signal ranges and shifts:
        +0.7 to +1.0 (Complacent): SPY +5%, GLD -3%, TLT -2%
        +0.3 to +0.7 (Normal): No change
        -0.3 to +0.3 (Neutral): No change
        -0.7 to -0.3 (Caution): SPY -5%, GLD +3%, TLT +2%
        -1.0 to -0.7 (Risk-Off): SPY -10%, GLD +5%, TLT +5%
        """
        if signal >= 0.7:
            return {"spy": 0.05, "gld": -0.03, "tlt": -0.02}
        elif signal >= 0.3:
            return {"spy": 0.02, "gld": -0.01, "tlt": -0.01}
        elif signal >= -0.3:
            return {"spy": 0.0, "gld": 0.0, "tlt": 0.0}
        elif signal >= -0.7:
            return {"spy": -0.05, "gld": 0.03, "tlt": 0.02}
        else:
            return {"spy": -0.10, "gld": 0.05, "tlt": 0.05}


class VIXTermStructureSignalGenerator:
    """
    Main signal generator for VIX term structure tactical overlay.
    
    Fetches data, calculates signals, and generates portfolio recommendations.
    Prefers market.db levels when the JSON history file is stale.
    """
    
    DATA_DIR = DATA_DIR
    VIX_DATA_PATH = DATA_DIR / 'vix_term_structure.json'
    OUTPUT_PATH = SIGNALS_DIR / 'vix_term_structure_signal.json'
    # If file latest date lags market.db by more than this many days, prefer DB.
    FILE_STALE_DAYS = 3
    
    def __init__(self, data_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        self.DATA_DIR = Path(data_dir) if data_dir is not None else DATA_DIR
        self.VIX_DATA_PATH = self.DATA_DIR / 'vix_term_structure.json'
        self.OUTPUT_PATH = SIGNALS_DIR / 'vix_term_structure_signal.json'
        self.db_path = Path(db_path) if db_path is not None else MARKET_DB
        self.calculator = VIXTermStructureCalculator()
        self._last_levels_meta: Dict[str, Any] = {}
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure output directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_iso_date(value: str) -> Optional[date]:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    def load_vix_data(self) -> Dict:
        """Load VIX term structure data from storage.

        Batch BV: strip ``_meta`` / non-date keys so provenance never sorts
        as the latest calendar day (``max(keys)`` would prefer ``_meta``).
        """
        if not self.VIX_DATA_PATH.exists():
            logger.warning("VIX data file not found: %s", self.VIX_DATA_PATH)
            return {}
        
        try:
            with open(self.VIX_DATA_PATH, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return {
                k: v
                for k, v in data.items()
                if isinstance(v, dict)
                and not str(k).startswith("_")
                and str(k) not in {"meta", "schema"}
                and len(str(k)) >= 10
                and str(k)[4:5] == "-"
            }
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("Error loading VIX data: %s", e)
            return {}

    def _file_latest_as_of(self, data: Dict) -> Optional[str]:
        if not data:
            return None
        try:
            dates = [
                k
                for k in data.keys()
                if not str(k).startswith("_")
                and len(str(k)) >= 10
                and str(k)[4:5] == "-"
            ]
            return max(dates) if dates else None
        except ValueError:
            return None

    def fetch_levels_from_market_db(self) -> Optional[Dict[str, Any]]:
        """Read latest ^VIX / ^VIX3M closes from market.db."""
        if not self.db_path.exists():
            return None
        try:
            with sqlite_connect(self.db_path) as conn:
                def latest(symbol: str) -> Optional[Tuple[str, float]]:
                    row = conn.execute(
                        "SELECT date, close FROM prices WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 1",
                        (symbol,),
                    ).fetchone()
                    if not row or row[0] is None or row[1] is None:
                        return None
                    return str(row[0])[:10], float(row[1])

                vix_row = latest("^VIX") or latest("VIX")
                vix3m_row = latest("^VIX3M") or latest("VIX3M")
                if vix_row is None and vix3m_row is None:
                    return None

                # Prefer the freshest available as_of across symbols
                as_of_candidates = [r[0] for r in (vix_row, vix3m_row) if r]
                as_of = max(as_of_candidates)
                vix_spot = vix_row[1] if vix_row else None
                vix3m = vix3m_row[1] if vix3m_row else None
                # If spot missing but VIX3M present, leave spot None for caller fallback
                return {
                    "date": as_of,
                    "vix_spot": vix_spot,
                    "front_month": vix3m,
                    "third_month": None,
                    "source": "market.db",
                    "as_of": as_of,
                }
        except (OSError, sqlite3.Error, TypeError, ValueError) as e:
            logger.warning("market.db VIX fetch failed: %s", e)
            return None

    def _file_is_stale_vs_db(
        self,
        file_as_of: Optional[str],
        db_as_of: Optional[str],
    ) -> bool:
        if db_as_of is None:
            return False
        if file_as_of is None:
            return True
        f_d = self._parse_iso_date(file_as_of)
        d_d = self._parse_iso_date(db_as_of)
        if f_d is None or d_d is None:
            return d_d is not None and f_d is None
        return (d_d - f_d).days > self.FILE_STALE_DAYS

    def resolve_current_levels(
        self,
        historical_data: Dict,
        requested_date: Optional[str] = None,
        persist_refresh: bool = True,
    ) -> Tuple[Optional[str], Optional[Dict], Dict[str, Any]]:
        """Choose levels from JSON history and/or market.db by freshness.

        Returns (as_of_date, levels_dict, meta).

        Explicit ``requested_date`` present in the JSON history always uses that
        row (backtests / historical generation). Freshness vs market.db applies
        only to live/latest resolution (``requested_date is None``).
        """
        file_as_of = self._file_latest_as_of(historical_data)
        db_levels = self.fetch_levels_from_market_db()
        db_as_of = db_levels.get("as_of") if db_levels else None
        prefer_db = self._file_is_stale_vs_db(file_as_of, db_as_of)

        # Explicit historical date: always prefer file row when present
        if requested_date is not None and requested_date in historical_data:
            row = dict(historical_data[requested_date])
            meta = {
                "source": "vix_term_structure.json",
                "as_of": requested_date,
                "file_as_of": file_as_of,
                "db_as_of": db_as_of,
            }
            return requested_date, row, meta

        # Explicit date missing from file: fall through to live/latest resolution
        # (legacy generate_signal behavior when date not in history).

        if prefer_db and db_levels is not None:
            # Merge file last spot if DB lacks ^VIX
            file_last = historical_data.get(file_as_of, {}) if file_as_of else {}
            vix_spot = db_levels.get("vix_spot")
            if vix_spot is None:
                vix_spot = file_last.get("vix_spot")
            # Require a real spot for a usable slope; do not invent 0.0
            if vix_spot is None or float(vix_spot) <= 0:
                return None, None, {
                    "source": "none",
                    "as_of": None,
                    "file_as_of": file_as_of,
                    "db_as_of": db_as_of,
                    "reason": "market.db missing usable ^VIX spot",
                }
            levels = {
                "date": db_levels["as_of"],
                "vix_spot": vix_spot,
                "front_month": db_levels.get("front_month"),
                "third_month": db_levels.get("third_month") or file_last.get("third_month"),
                "source": "market.db",
                "as_of": db_levels["as_of"],
            }
            meta = {
                "source": "market.db",
                "as_of": db_levels["as_of"],
                "file_as_of": file_as_of,
                "db_as_of": db_as_of,
                "refreshed_from_db": True,
            }
            if persist_refresh:
                self._persist_file_row(historical_data, levels)
            return db_levels["as_of"], levels, meta

        # File path (fresh enough or no DB) — live/latest
        if file_as_of and file_as_of in historical_data:
            row = dict(historical_data[file_as_of])
            meta = {
                "source": "vix_term_structure.json",
                "as_of": file_as_of,
                "file_as_of": file_as_of,
                "db_as_of": db_as_of,
            }
            return file_as_of, row, meta

        if db_levels is not None:
            meta = {
                "source": "market.db",
                "as_of": db_levels["as_of"],
                "file_as_of": file_as_of,
                "db_as_of": db_as_of,
            }
            return db_levels["as_of"], db_levels, meta

        return None, None, {
            "source": "none",
            "as_of": None,
            "file_as_of": file_as_of,
            "db_as_of": db_as_of,
        }

    def _persist_file_row(self, historical_data: Dict, levels: Dict) -> None:
        """Write/refresh a JSON history row so the file does not stay frozen.

        Only writes under ``self.DATA_DIR`` (never invents paths like
        ``/nonexistent/...`` from tests).
        """
        as_of = levels.get("as_of") or levels.get("date")
        if not as_of:
            return
        try:
            target = Path(self.VIX_DATA_PATH)
            data_root = Path(self.DATA_DIR).resolve()
            # Allow write only when target lives under DATA_DIR
            try:
                target.resolve().relative_to(data_root)
            except (ValueError, OSError):
                # If path does not exist yet, check parent prefix without resolve
                if data_root not in target.parents and target.parent != data_root:
                    logger.debug(
                        "Skip VIX file refresh outside DATA_DIR: %s (DATA_DIR=%s)",
                        target,
                        data_root,
                    )
                    return
        except (OSError, RuntimeError):
            return

        # Persist with derived contango fields so VIXDataManager / from_dict
        # never sees sparse market.db proxy rows (Batch BE sticky-kill class).
        raw_row = {
            "date": as_of,
            "vix_spot": levels.get("vix_spot"),
            "front_month": levels.get("front_month"),
            "third_month": levels.get("third_month"),
            "source": levels.get("source", "market.db"),
            "as_of": as_of,
            "refreshed_at": datetime.now().isoformat(),
        }
        try:
            from src.data.vix_futures import VIXTermStructure as _VTS

            # Hydrate required dataclass fields (second_month, contango_*, is_contango)
            if raw_row.get("vix_spot") is not None and raw_row.get("front_month") is not None:
                ts = _VTS.from_dict(raw_row)
                row = ts.to_dict()
                row["source"] = raw_row.get("source", "market.db")
                row["as_of"] = as_of
                row["refreshed_at"] = raw_row["refreshed_at"]
            else:
                row = raw_row
        except (TypeError, ValueError, KeyError) as e:
            logger.debug("VIX row hydrate skipped: %s", e)
            row = raw_row
        try:
            # Batch CM: never write the meta-stripped in-memory view from
            # load_vix_data() — that drops _meta and can orphan the full history
            # when the caller only held a partial dict. Re-read disk, merge row.
            on_disk: Dict = {}
            if self.VIX_DATA_PATH.exists():
                try:
                    raw = json.loads(self.VIX_DATA_PATH.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        on_disk = raw
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    on_disk = {}
            # Prefer full disk history; fall back to caller's date rows only
            if not on_disk:
                on_disk = {
                    k: v
                    for k, v in historical_data.items()
                    if isinstance(v, dict) and not str(k).startswith("_")
                }
            on_disk[as_of] = row
            # Keep/refresh light meta so schema rebuilds remain honest
            meta = on_disk.get("_meta") if isinstance(on_disk.get("_meta"), dict) else {}
            date_keys = [
                k
                for k, v in on_disk.items()
                if isinstance(v, dict)
                and not str(k).startswith("_")
                and len(str(k)) >= 10
                and str(k)[4:5] == "-"
            ]
            meta = {
                **meta,
                "schema": meta.get("schema") or "vix_term_structure/v1",
                "last_row_refresh_at": row.get("refreshed_at"),
                "last_row_as_of": as_of,
                "n_dates": len(date_keys),
                "date_min": min(date_keys) if date_keys else None,
                "date_max": max(date_keys) if date_keys else None,
                "live_authoritative": False,
            }
            on_disk["_meta"] = meta
            self.VIX_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            save_results_json(on_disk, output_path=str(self.VIX_DATA_PATH))
            logger.info(
                "Refreshed VIX term-structure file row as_of=%s from market.db (n_dates=%d)",
                as_of,
                len(date_keys),
            )
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to refresh vix_term_structure.json: %s", e)
    
    def fetch_current_vix(self) -> Optional[Dict]:
        """
        Fetch current VIX levels: market.db when fresher, else JSON history.
        """
        data = self.load_vix_data()
        _as_of, levels, meta = self.resolve_current_levels(data, requested_date=None)
        self._last_levels_meta = meta
        return levels
    
    def generate_signal(self, date: Optional[str] = None) -> VIXTermStructureSignal:
        """Generate complete VIX term structure signal."""
        requested = date  # None means "live/latest"
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Load historical data for context
        historical_data = self.load_vix_data()

        as_of, current, meta = self.resolve_current_levels(
            historical_data,
            requested_date=requested if requested is not None else None,
            persist_refresh=(requested is None),
        )
        self._last_levels_meta = meta

        # When live request and DB/file latest differs from "today", use as_of
        if requested is None and as_of:
            date = as_of

        # Build VIX history for Z-score (file history + optional current)
        for d in sorted(historical_data.keys())[-252:]:
            vix = historical_data[d].get('vix_spot', 0)
            if vix and vix > 0:
                self.calculator.add_vix_reading(d, float(vix))
        if current and current.get("vix_spot"):
            try:
                self.calculator.add_vix_reading(date, float(current["vix_spot"]))
            except (TypeError, ValueError):
                pass
        
        if current is None:
            return VIXTermStructureSignal(
                timestamp=datetime.now().isoformat(),
                signal_state="neutral",
                signal_value=0.0,
                vix_spot=0.0,
                vix3m=None,
                vix6m=None,
                slope_vix3m_vix=1.0,
                regime="unknown",
                regime_strength=0.0,
                slope_signal=0.0,
                roll_yield_signal=0.0,
                vix_zscore_signal=0.0,
                curve_shape_signal=0.0,
                spy_shift=0.0,
                gld_shift=0.0,
                tlt_shift=0.0,
                confidence=0.0,
                is_valid=False,
                reason=(
                    "No VIX data available "
                    f"(source={meta.get('source')}, file_as_of={meta.get('file_as_of')}, "
                    f"db_as_of={meta.get('db_as_of')})"
                ),
            )
        
        try:
            vix = float(current.get("vix_spot", 0) or 0)
        except (TypeError, ValueError):
            # Preserve prior TypeError surface for non-numeric spot in callers that expect it
            vix = current.get("vix_spot")
            if not isinstance(vix, (int, float)):
                raise TypeError(f"vix_spot must be numeric, got {type(vix)!r}")
            vix = float(vix)
        vix3m = current.get('front_month')  # Using front month as proxy for VIX3M
        vix6m = current.get('third_month')  # Third month as VIX6M proxy
        
        # Calculate composite signal
        components = self.calculator.calculate_composite_signal(
            vix=vix,
            vix3m=vix3m,
            vix6m=vix6m,
            date=date
        )
        
        # Classify regime
        regime, strength = self.calculator.classify_regime(components['slope'])
        
        # Map to signal state
        composite = components['composite']
        if composite > 0.5:
            signal_state = VIXSignalState.RISK_ON
        elif composite < -0.5:
            signal_state = VIXSignalState.RISK_OFF
        else:
            signal_state = VIXSignalState.NEUTRAL
        
        # Get allocation shifts
        shifts = self.calculator.get_allocation_shifts(composite)
        
        # Calculate confidence based on data quality
        confidence = 50.0  # Base confidence
        if vix3m is not None:
            confidence += 30.0
        if vix6m is not None:
            confidence += 10.0
        if len(self.calculator.vix_history) >= 60:
            confidence += 10.0
        if meta.get("source") == "market.db":
            confidence = min(100.0, confidence + 5.0)
        
        source = meta.get("source", "unknown")
        as_of_meta = meta.get("as_of") or date
        reason = (
            f"VIX={vix:.2f}, Slope={components['slope']:.3f}, Regime={regime.value}, "
            f"as_of={as_of_meta}, source={source}"
        )
        
        signal = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state=signal_state.name,
            signal_value=composite,
            vix_spot=vix,
            vix3m=vix3m,
            vix6m=vix6m,
            slope_vix3m_vix=components['slope'],
            regime=regime.value,
            regime_strength=strength,
            slope_signal=components['slope_signal'],
            roll_yield_signal=components['roll_yield_signal'],
            vix_zscore_signal=components['vix_zscore_signal'],
            curve_shape_signal=components['curve_shape_signal'],
            spy_shift=shifts['spy'],
            gld_shift=shifts['gld'],
            tlt_shift=shifts['tlt'],
            confidence=confidence,
            is_valid=True,
            reason=reason,
        )
        # Attach freshness meta for snapshot consumers (not a dataclass field)
        signal._freshness = {  # type: ignore[attr-defined]
            "as_of": as_of_meta,
            "source": source,
            "file_as_of": meta.get("file_as_of"),
            "db_as_of": meta.get("db_as_of"),
        }
        return signal
    
    def get_signal_snapshot(self, tickers=None, date=None):
        """Generate a SignalSnapshot for ensemble voter consumption."""
        signal = self.generate_signal()
        return signal.to_signal_snapshot()

    def save_signal(self, signal: VIXTermStructureSignal):
        """Save signal to disk."""
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            save_results_json(signal.to_dict(), output_path=str(self.OUTPUT_PATH))
            logger.info("Saved VIX signal to %s", self.OUTPUT_PATH)
        except (OSError, KeyError, ValueError, TypeError) as e:
            logger.error("Error saving signal: %s", e)

    def get_signal_history(self, days: int = 30) -> List[VIXTermStructureSignal]:
        """Generate signals for historical dates."""
        historical_data = self.load_vix_data()
        signals = []
        
        dates = sorted(historical_data.keys())[-days:]
        
        for date in dates:
            signal = self.generate_signal(date)
            if signal.is_valid:
                signals.append(signal)
        
        return signals


def main():
    """CLI entry point for signal generation."""
    generator = VIXTermStructureSignalGenerator()
    signal = generator.generate_signal()
    
    logger.info("=" * 60)
    logger.info("VIX TERM STRUCTURE SIGNAL GENERATOR v4.50")
    logger.info("=" * 60)
    logger.info("Timestamp: %s", signal.timestamp)
    logger.info("Signal State: %s", signal.signal_state)
    logger.info("Signal Value: %.3f", signal.signal_value)
    logger.info("")
    logger.info("VIX Spot: %.2f", signal.vix_spot)
    logger.info("VIX3M: %s", signal.vix3m)
    logger.info("VIX6M: %s", signal.vix6m)
    logger.info("Slope (VIX3M/VIX): %.3f", signal.slope_vix3m_vix)
    logger.info("")
    logger.info("Regime: %s", signal.regime)
    logger.info("Regime Strength: %.2f", signal.regime_strength)
    logger.info("")
    logger.info("Component Signals:")
    logger.info("  Slope Signal: %.3f", signal.slope_signal)
    logger.info("  Roll Yield: %.3f", signal.roll_yield_signal)
    logger.info("  VIX Z-Score: %.3f", signal.vix_zscore_signal)
    logger.info("  Curve Shape: %.3f", signal.curve_shape_signal)
    logger.info("")
    logger.info("Portfolio Shifts:")
    logger.info("  SPY: %+.1f%%", signal.spy_shift * 100)
    logger.info("  GLD: %+.1f%%", signal.gld_shift * 100)
    logger.info("  TLT: %+.1f%%", signal.tlt_shift * 100)
    logger.info("")
    logger.info("Confidence: %.0f%%", signal.confidence)
    logger.info("Valid: %s", signal.is_valid)
    logger.info("Reason: %s", signal.reason)
    logger.info("=" * 60)
    
    # Save signal
    generator.save_signal(signal)
    
    return signal


if __name__ == '__main__':
    from src.utils.log_config import configure_logging
    configure_logging()
    main()
