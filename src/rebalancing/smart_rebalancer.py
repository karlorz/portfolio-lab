"""
Smart Rebalancing Controller — v2.90
Combines drift-based triggers, VPIN microstructure timing, and intraday
seasonality optimization to minimize transaction costs while maintaining
portfolio tracking accuracy.

Integrates:
- v2.24 Drift-based rebalancing (±10% threshold)
- v2.65 VPIN microstructure timing
- v2.71 Intraday seasonality execution

Target: 40%+ cost reduction vs calendar rebalancing.
"""

import json
import logging
import os
import tempfile
import yaml
from src.paths import BASE_ALLOCATION, DATA_DIR
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
from zoneinfo import ZoneInfo

# Durable controller state (YTD costs + last_rebalance) across process restarts.
SMART_REBALANCE_STATE_FILENAME = "smart_rebalance_state.json"

# Optimal timing window is defined in America/New_York wall clock (not host local).
_ET = ZoneInfo("America/New_York")


class RebalanceDecision(Enum):
    EXECUTE = "execute"
    DEFER_TOXICITY = "defer_toxicity"
    DEFER_TIMING = "defer_timing"
    DEFER_BUDGET = "defer_budget"
    SKIP_LOW_DRIFT = "skip_low_drift"
    OVERRIDE_EMERGENCY = "override_emergency"


class UrgencyLevel(Enum):
    LOW = "low"           # Drift 10-12%, can wait
    MODERATE = "moderate" # Drift 12-15%
    HIGH = "high"         # Drift 15-20%
    EMERGENCY = "emergency"  # Drift > 20%, override all


@dataclass
class PortfolioSnapshot:
    """Current portfolio state for drift calculation."""
    holdings: Dict[str, float]      # symbol -> current market value
    targets: Dict[str, float]       # symbol -> target allocation (0-1)
    total_value: float
    timestamp: datetime


@dataclass
class MarketConditions:
    """Current market microstructure conditions."""
    vpin: float                     # Volume-Synchronized Probability of Informed Trading (0-1)
    vix: Optional[float] = None
    spread_bps: Optional[Dict[str, float]] = None  # symbol -> spread in bps
    timestamp: Optional[datetime] = None


@dataclass
class RebalanceDecisionResult:
    """Output of the should_rebalance decision."""
    decision: RebalanceDecision
    urgency: UrgencyLevel
    max_drift: float
    drift_details: Dict[str, float]  # symbol -> drift %
    vpin: float
    estimated_cost_bps: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudgetTracker:
    """Tracks YTD rebalancing costs against annual budget."""
    annual_limit_pct: float = 0.005     # 0.5% default
    warning_threshold_pct: float = 0.004  # Alert at 80%
    ytd_costs: List[Dict] = field(default_factory=list)
    # Calendar year for YTD view (Batch DY); None → datetime.now().year
    ytd_year: Optional[int] = None
    # Batch DZ: single-trade cap; None = disabled (unit tests / raw tracker).
    # Controller wires safety.max_single_trade_cost_bps (default 15).
    max_single_trade_cost_bps: Optional[float] = None
    # Outliers kept for audit but excluded from YTD budget sum
    quarantined_costs: List[Dict] = field(default_factory=list)

    @staticmethod
    def _entry_year(date_val: Any) -> Optional[int]:
        text = str(date_val or "").strip()
        if len(text) >= 4 and text[:4].isdigit():
            try:
                return int(text[:4])
            except ValueError:
                return None
        return None

    @staticmethod
    def _entry_day_key(date_val: Any) -> str:
        """Normalize date to YYYY-MM-DD for dedupe (strip time suffix)."""
        text = str(date_val or "").strip()
        if not text:
            return ""
        # ISO with T → date part; bare date ok
        if "T" in text:
            return text.split("T", 1)[0][:10]
        return text[:10]

    @staticmethod
    def _symbols_key(symbols: Any) -> Tuple[str, ...]:
        if not symbols:
            return tuple()
        try:
            return tuple(sorted(str(s) for s in symbols))
        except TypeError:
            return tuple()

    def _active_year(self) -> int:
        if self.ytd_year is not None:
            return int(self.ytd_year)
        return datetime.now().year

    def _costs_in_year(self, year: Optional[int] = None) -> List[Dict]:
        y = int(year) if year is not None else self._active_year()
        out: List[Dict] = []
        for c in self.ytd_costs:
            if not isinstance(c, dict):
                continue
            cy = self._entry_year(c.get("date"))
            # Missing year → include (legacy / fail-open into current window)
            if cy is None or cy == y:
                out.append(c)
        return out

    @property
    def ytd_total_bps(self) -> float:
        # Batch DY: year-scoped sum (calendar YTD view)
        return sum(float(c.get("cost_bps", 0) or 0) for c in self._costs_in_year())

    @property
    def ytd_total_pct(self) -> float:
        return self.ytd_total_bps / 10000

    @property
    def remaining_budget_pct(self) -> float:
        return max(0, self.annual_limit_pct - self.ytd_total_pct)

    def _is_outlier_bps(self, bps: float) -> bool:
        cap = self.max_single_trade_cost_bps
        if cap is None:
            return False
        try:
            return float(bps) > float(cap) + 1e-12
        except (TypeError, ValueError):
            return False

    def _quarantine_key(self, bps: float, day: str, syms: Any) -> Tuple:
        return (round(float(bps), 4), day, self._symbols_key(syms))

    def _already_quarantined(self, key: Tuple) -> bool:
        for existing in self.quarantined_costs:
            if not isinstance(existing, dict):
                continue
            try:
                ek = self._quarantine_key(
                    float(existing.get("cost_bps", 0) or 0),
                    self._entry_day_key(existing.get("date")),
                    existing.get("symbols"),
                )
            except (TypeError, ValueError):
                continue
            if ek == key:
                return True
        return False

    def add_cost(self, cost_bps: float, date: str, symbols: List[str]):
        """Append a cost row with composite-key idempotency (Batch DY/DZ).

        Exact duplicate of (rounded cost_bps, calendar day, sorted symbols)
        is skipped so re-runs / double record_rebalance do not inflate YTD.
        Rows above max_single_trade_cost_bps are quarantined (audit only).
        """
        try:
            bps = float(cost_bps)
        except (TypeError, ValueError):
            return
        day = self._entry_day_key(date)
        syms = list(symbols or [])
        key = (round(bps, 4), day, self._symbols_key(syms))

        if self._is_outlier_bps(bps):
            if not self._already_quarantined(key):
                self.quarantined_costs.append(
                    {
                        "cost_bps": bps,
                        "date": date,
                        "symbols": syms,
                        "reason": "above_max_single_trade_cost_bps",
                        "cap_bps": self.max_single_trade_cost_bps,
                    }
                )
            return

        for existing in self.ytd_costs:
            if not isinstance(existing, dict):
                continue
            try:
                ek = (
                    round(float(existing.get("cost_bps", 0) or 0), 4),
                    self._entry_day_key(existing.get("date")),
                    self._symbols_key(existing.get("symbols")),
                )
            except (TypeError, ValueError):
                continue
            if ek == key:
                return  # idempotent no-op
        self.ytd_costs.append({
            'cost_bps': bps,
            'date': date,
            'symbols': syms,
        })

    def sanitize_ledger(
        self,
        *,
        as_of_year: Optional[int] = None,
        drop_prior_years_from_storage: bool = False,
        drop_zero_cost: bool = True,
    ) -> Dict[str, Any]:
        """Dedupe, zero-drop, and quarantine single-trade outliers (DY/DZ).

        Composite key: (cost_bps rounded 4dp, calendar day, sorted symbols).
        Outliers above max_single_trade_cost_bps move to quarantined_costs
        (audit trail) and are excluded from ytd_total_bps.
        """
        year = int(as_of_year) if as_of_year is not None else self._active_year()
        before = list(self.ytd_costs)
        before_count = len(before)
        dropped_zero = 0
        dropped_duplicate = 0
        dropped_prior_year = 0
        quarantined_outlier = 0
        quarantined_bps = 0.0
        seen: set = set()
        cleaned: List[Dict] = []
        # Preserve prior quarantines; re-scan may re-add from ytd_costs
        prior_q = list(self.quarantined_costs)

        for entry in before:
            if not isinstance(entry, dict):
                continue
            try:
                bps = float(entry.get("cost_bps", 0) or 0)
            except (TypeError, ValueError):
                continue
            day = self._entry_day_key(entry.get("date"))
            syms = list(entry.get("symbols") or [])
            cy = self._entry_year(entry.get("date"))

            if drop_prior_years_from_storage and cy is not None and cy < year:
                dropped_prior_year += 1
                continue
            if drop_zero_cost and abs(bps) < 1e-12:
                dropped_zero += 1
                continue

            key = (round(bps, 4), day, self._symbols_key(syms))
            if self._is_outlier_bps(bps):
                if not self._already_quarantined(key):
                    self.quarantined_costs.append(
                        {
                            "cost_bps": bps,
                            "date": entry.get("date") if entry.get("date") is not None else day,
                            "symbols": syms,
                            "reason": "above_max_single_trade_cost_bps",
                            "cap_bps": self.max_single_trade_cost_bps,
                        }
                    )
                quarantined_outlier += 1
                quarantined_bps += bps
                continue

            if key in seen:
                dropped_duplicate += 1
                continue
            seen.add(key)
            cleaned.append(
                {
                    "cost_bps": bps,
                    "date": entry.get("date") if entry.get("date") is not None else day,
                    "symbols": syms,
                }
            )

        # Dedupe quarantine list itself (same composite key)
        q_seen: set = set()
        q_clean: List[Dict] = []
        for entry in self.quarantined_costs:
            if not isinstance(entry, dict):
                continue
            try:
                bps = float(entry.get("cost_bps", 0) or 0)
            except (TypeError, ValueError):
                continue
            qk = self._quarantine_key(
                bps,
                self._entry_day_key(entry.get("date")),
                entry.get("symbols"),
            )
            if qk in q_seen:
                continue
            q_seen.add(qk)
            q_clean.append(entry)
        self.quarantined_costs = q_clean

        self.ytd_costs = cleaned
        after_count = len(cleaned)
        changed = (
            dropped_zero > 0
            or dropped_duplicate > 0
            or dropped_prior_year > 0
            or quarantined_outlier > 0
            or after_count != before_count
            or len(self.quarantined_costs) != len(prior_q)
        )
        return {
            "changed": changed,
            "before_count": before_count,
            "after_count": after_count,
            "kept": after_count,
            "dropped_zero": dropped_zero,
            "dropped_duplicate": dropped_duplicate,
            "dropped_prior_year": dropped_prior_year,
            "quarantined_outlier": quarantined_outlier,
            "quarantined_bps": round(quarantined_bps, 4),
            "quarantined_total": len(self.quarantined_costs),
            "max_single_trade_cost_bps": self.max_single_trade_cost_bps,
            "as_of_year": year,
            "ytd_total_bps": self.ytd_total_bps,
        }

    def is_over_budget(self) -> bool:
        return self.ytd_total_pct >= self.annual_limit_pct

    def is_warning(self) -> bool:
        return self.ytd_total_pct >= self.warning_threshold_pct


from src.costs.etf_cost_table import ETF_COST_BPS, DEFAULT_COST_BPS as _DEFAULT_COST_BPS

logger = logging.getLogger(__name__)


__all__ = ['RebalanceDecision', 'UrgencyLevel', 'PortfolioSnapshot', 'MarketConditions', 'RebalanceDecisionResult', 'CostBudgetTracker', 'SmartRebalancingController', 'create_sample_portfolio']

class SmartRebalancingController:
    """
    Unified rebalancing controller combining drift triggers, VPIN timing,
    and intraday seasonality optimization.
    """

    # Per-ETF one-way transaction costs — delegates to centralized cost table.
    ETF_TRANSACTION_COSTS_BPS: Dict[str, float] = dict(ETF_COST_BPS)
    DEFAULT_COST_BPS: float = _DEFAULT_COST_BPS

    DEFAULT_CONFIG = {
        'drift_threshold': 0.10,
        'drift_threshold_by_regime': {
            'low_vol': 0.15,   # Calm market: tolerate more drift
            'normal': 0.10,    # Default
            'high_vol': 0.07,  # Volatile: rebalance sooner
            'crisis': 0.05,    # Crisis: tight tracking
        },
        'urgency_levels': {
            'emergency': 0.95,   # Drift > 20%
            'high': 0.70,        # Drift 15-20%
            'moderate': 0.50,    # Drift 12-15%
            'low': 0.30,         # Drift 10-12%
        },
        'vpin': {
            'threshold': 0.50,
            'default': 0.30,     # Default when VPIN not available
        },
        'timing': {
            'optimal_start': 11,  # 11:00 ET
            'optimal_end': 14,    # 14:00 ET
            'low_urgency_can_wait': True,
        },
        'cost_budget': {
            'annual_limit': 0.005,
            'warning_threshold': 0.004,
        },
        'fallback': {
            'deferral_max_hours': 4,
            'force_if_drift_exceeds': 0.25,
        },
        'safety': {
            'max_deferral_hours': 4,
            'max_single_trade_cost_bps': 15,
            'max_annual_cost_pct': 0.006,
            'min_drift_override': 0.08,
        },
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        state_path: Optional[str | Path] = None,
        data_dir: Optional[str | Path] = None,
        load_state: bool = True,
    ):
        self.config = self._load_config(config_path)
        safety = self.config.get("safety") if isinstance(self.config.get("safety"), dict) else {}
        max_single = safety.get("max_single_trade_cost_bps", 15)
        try:
            max_single_f = float(max_single) if max_single is not None else 15.0
        except (TypeError, ValueError):
            max_single_f = 15.0
        self.cost_tracker = CostBudgetTracker(
            annual_limit_pct=self.config['cost_budget']['annual_limit'],
            warning_threshold_pct=self.config['cost_budget']['warning_threshold'],
            max_single_trade_cost_bps=max_single_f,
        )
        self.deferred_until: Optional[datetime] = None
        self.last_rebalance: Optional[datetime] = None
        # Batch DX: provenance when last_rebalance was advanced from order events
        self.last_rebalance_clock_source: Optional[str] = None
        self.last_rebalance_reconciled: bool = False
        self.last_rebalance_reconciled_from: Optional[str] = None
        # Batch DY: ledger sanitize meta
        self.ledger_sanitized: bool = False
        self.ledger_sanitize_report: Optional[Dict[str, Any]] = None
        self.data_dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        if state_path is not None:
            self.state_path = Path(state_path)
        else:
            self.state_path = self.data_dir / SMART_REBALANCE_STATE_FILENAME
        if load_state:
            self.load_state()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load config from YAML file or use defaults."""
        config = self.DEFAULT_CONFIG.copy()
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                if user_config and 'smart_rebalancing' in user_config:
                    self._deep_merge(config, user_config['smart_rebalancing'])
        return config

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base dict."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def calculate_drift(self, portfolio: PortfolioSnapshot) -> Tuple[float, Dict[str, float]]:
        """
        Calculate maximum drift and per-asset drift details.
        Drift = |current_allocation - target_allocation| / target_allocation
        """
        drift_details = {}
        max_drift = 0.0

        for symbol, target_alloc in portfolio.targets.items():
            current_value = portfolio.holdings.get(symbol, 0)
            current_alloc = current_value / portfolio.total_value if portfolio.total_value > 0 else 0
            drift = abs(current_alloc - target_alloc) / target_alloc if target_alloc > 0 else 0
            drift_details[symbol] = round(drift, 4)
            max_drift = max(max_drift, drift)

        return round(max_drift, 4), drift_details

    def calculate_urgency(self, max_drift: float) -> UrgencyLevel:
        """Map drift level to urgency classification."""
        if max_drift > 0.20:
            return UrgencyLevel.EMERGENCY
        elif max_drift > 0.15:
            return UrgencyLevel.HIGH
        elif max_drift > 0.12:
            return UrgencyLevel.MODERATE
        else:
            return UrgencyLevel.LOW

    def estimate_cost_bps(self, vpin: float, in_optimal_window: bool) -> float:
        """
        Estimate execution cost in basis points.
        Cost = Base_Spread × VPIN_Multiplier × Time_Multiplier + Fixed
        """
        base_spread = 0.0003   # 3 bps
        fixed = 0.0002         # 2 bps

        # VPIN multiplier: higher VPIN = higher cost
        vpin_mult = max(1.0, 1.0 + (vpin - 0.30) * 2.0)
        vpin_mult = min(vpin_mult, 2.0)

        # Time multiplier: outside optimal window = higher cost (ET wall clock)
        if in_optimal_window:
            time_mult = 1.0
        else:
            hour = self._et_now().hour
            if hour < 10:
                time_mult = 1.25   # Opening volatility
            elif hour >= 15.5:
                time_mult = 1.15   # Closing auction
            else:
                time_mult = 1.05   # Mid-morning / afternoon

        cost = base_spread * vpin_mult * time_mult + fixed
        return round(cost * 10000, 2)  # Convert to bps

    def estimate_per_symbol_cost_bps(
        self, symbol: str, vpin: float, in_optimal_window: bool
    ) -> float:
        """
        Estimate execution cost for a specific ETF symbol.

        Uses per-ETF base cost from ETF_TRANSACTION_COSTS_BPS, then applies
        VPIN and timing multipliers (same formula as estimate_cost_bps but
        with symbol-specific base instead of flat 3 bps).
        """
        base_bps = self.ETF_TRANSACTION_COSTS_BPS.get(
            symbol, self.DEFAULT_COST_BPS
        )
        base_spread = base_bps / 10000  # Convert bps → decimal

        # VPIN multiplier (same logic as estimate_cost_bps)
        vpin_mult = max(1.0, 1.0 + (vpin - 0.30) * 2.0)
        vpin_mult = min(vpin_mult, 2.0)

        # Time multiplier (same logic as estimate_cost_bps; ET wall clock)
        if in_optimal_window:
            time_mult = 1.0
        else:
            hour = self._et_now().hour
            if hour < 10:
                time_mult = 1.25
            elif hour >= 15.5:
                time_mult = 1.15
            else:
                time_mult = 1.05

        cost_decimal = base_spread * vpin_mult * time_mult
        return round(cost_decimal * 10000, 2)  # Convert to bps

    def estimate_total_cost_bps(
        self, drift_details: Dict[str, float], vpin: float, in_optimal_window: bool
    ) -> float:
        """
        Estimate total rebalancing cost using per-ETF transaction costs.

        For each symbol with non-zero drift, computes the per-symbol cost
        and sums them. Falls back to flat estimate_cost_bps if no drift
        details available.
        """
        if not drift_details:
            return self.estimate_cost_bps(vpin, in_optimal_window)

        total = 0.0
        for symbol, drift in drift_details.items():
            if drift > 0:
                symbol_cost = self.estimate_per_symbol_cost_bps(
                    symbol, vpin, in_optimal_window
                )
                # Scale cost by drift magnitude (larger drift = larger trade)
                total += symbol_cost * drift
        return round(total, 2)

    @staticmethod
    def _et_now(now: Optional[datetime] = None) -> datetime:
        """Resolve clock to America/New_York wall time for timing gates.

        - ``now is None`` → current time in ET (not host-local).
        - aware datetimes → converted to ET.
        - naive datetimes → treated as ET wall clock (documented contract).
        """
        if now is None:
            return datetime.now(_ET)
        if now.tzinfo is None:
            return now.replace(tzinfo=_ET)
        return now.astimezone(_ET)

    def _in_optimal_window(self, now: Optional[datetime] = None) -> bool:
        """Check if current time is in optimal execution window (11am-2pm ET)."""
        et_now = self._et_now(now)
        hour = et_now.hour
        start = self.config['timing']['optimal_start']
        end = self.config['timing']['optimal_end']
        return start <= hour < end

    def should_rebalance(
        self,
        portfolio: PortfolioSnapshot,
        market: MarketConditions,
        now: Optional[datetime] = None,
        regime: Optional[str] = None,
    ) -> RebalanceDecisionResult:
        """
        Core decision engine: should we rebalance now, defer, or skip?

        Decision flow:
        1. Check drift threshold (skip if below, regime-adaptive)
        2. Calculate urgency from drift
        3. Check VPIN toxicity (defer if high and not urgent)
        4. Check timing window (defer if low urgency and outside window)
        5. Check cost budget (defer if over budget)
        6. Safety overrides (force if drift > 25%)

        Args:
            regime: Optional market regime ('low_vol', 'normal', 'high_vol',
                     'crisis'). When provided, overrides drift_threshold with
                     the regime-specific value from drift_threshold_by_regime.
        """
        now = self._et_now(now)

        # Resolve regime-adaptive drift threshold
        regime_thresholds = self.config.get('drift_threshold_by_regime', {})
        if regime and regime in regime_thresholds:
            drift_threshold = regime_thresholds[regime]
        else:
            drift_threshold = self.config['drift_threshold']

        # Step 1: Drift check
        max_drift, drift_details = self.calculate_drift(portfolio)
        self.config['safety']['min_drift_override']

        if max_drift < drift_threshold:
            return RebalanceDecisionResult(
                decision=RebalanceDecision.SKIP_LOW_DRIFT,
                urgency=UrgencyLevel.LOW,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=market.vpin,
                estimated_cost_bps=0.0,
                reason=f"drift_below_threshold ({max_drift:.1%} < {drift_threshold:.1%})",
            )

        # Step 2: Urgency
        urgency = self.calculate_urgency(max_drift)
        vpin = market.vpin if market.vpin is not None else self.config['vpin']['default']

        # Step 3: Safety override — force if drift > 25%
        force_threshold = self.config['fallback']['force_if_drift_exceeds']
        if max_drift > force_threshold:
            cost = self.estimate_total_cost_bps(drift_details, vpin, self._in_optimal_window(now))
            return RebalanceDecisionResult(
                decision=RebalanceDecision.OVERRIDE_EMERGENCY,
                urgency=UrgencyLevel.EMERGENCY,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=cost,
                reason=f"emergency_override (drift {max_drift:.1%} > {force_threshold:.1%})",
            )

        # Step 4: VPIN toxicity check
        vpin_threshold = self.config['vpin']['threshold']
        if vpin > vpin_threshold and urgency != UrgencyLevel.EMERGENCY:
            self.consecutive_deferrals = getattr(self, 'consecutive_deferrals', 0) + 1
            max_deferrals = self.config['fallback']['deferral_max_hours']  # Reused as max deferral count
            if self.consecutive_deferrals > max_deferrals:
                # Max deferral count exceeded — force execution
                self.consecutive_deferrals = 0
                cost = self.estimate_total_cost_bps(drift_details, vpin, self._in_optimal_window(now))
                return RebalanceDecisionResult(
                    decision=RebalanceDecision.EXECUTE,
                    urgency=urgency,
                    max_drift=max_drift,
                    drift_details=drift_details,
                    vpin=vpin,
                    estimated_cost_bps=cost,
                    reason=f"max_deferral_exceeded (VPIN={vpin:.2f}, deferred {self.consecutive_deferrals}x)",
                )
            return RebalanceDecisionResult(
                decision=RebalanceDecision.DEFER_TOXICITY,
                urgency=urgency,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=0.0,
                reason=f"high_toxicity_defer (VPIN={vpin:.2f} > {vpin_threshold}, defer #{self.consecutive_deferrals})",
            )
        else:
            self.consecutive_deferrals = 0

        # Step 5: Timing window check
        in_window = self._in_optimal_window(now)
        if (urgency == UrgencyLevel.LOW
                and not in_window
                and self.config['timing']['low_urgency_can_wait']):
            return RebalanceDecisionResult(
                decision=RebalanceDecision.DEFER_TIMING,
                urgency=urgency,
                max_drift=max_drift,
                drift_details=drift_details,
                vpin=vpin,
                estimated_cost_bps=0.0,
                reason=f"wait_for_optimal_window (next: {self.config['timing']['optimal_start']}:00 ET)",
            )

        # Step 6: Cost budget check
        if self.cost_tracker.is_over_budget():
            if urgency != UrgencyLevel.EMERGENCY:
                return RebalanceDecisionResult(
                    decision=RebalanceDecision.DEFER_BUDGET,
                    urgency=urgency,
                    max_drift=max_drift,
                    drift_details=drift_details,
                    vpin=vpin,
                    estimated_cost_bps=0.0,
                    reason=f"cost_budget_exceeded (YTD: {self.cost_tracker.ytd_total_bps:.1f} bps)",
                )

        # All checks passed — execute
        cost = self.estimate_total_cost_bps(drift_details, vpin, in_window)
        per_symbol_costs = {
            sym: self.estimate_per_symbol_cost_bps(sym, vpin, in_window) * drift
            for sym, drift in drift_details.items()
            if drift > 0
        }
        return RebalanceDecisionResult(
            decision=RebalanceDecision.EXECUTE,
            urgency=urgency,
            max_drift=max_drift,
            drift_details=drift_details,
            vpin=vpin,
            estimated_cost_bps=cost,
            reason="execute",
            metadata={
                'in_optimal_window': in_window,
                'ytd_cost_bps': self.cost_tracker.ytd_total_bps,
                'remaining_budget_pct': self.cost_tracker.remaining_budget_pct,
                'remaining_budget_ratio': self.cost_tracker.remaining_budget_pct,
                'per_symbol_cost_bps': per_symbol_costs,
            },
        )

    def record_rebalance(self, cost_bps: float, date: str, symbols: List[str]):
        """Record a completed rebalance for budget tracking and persist state."""
        self.cost_tracker.add_cost(cost_bps, date, symbols)
        self.last_rebalance = datetime.fromisoformat(date) if 'T' in date else datetime.strptime(date, '%Y-%m-%d')
        self.deferred_until = None
        # Explicit record path is authoritative (not a lag reconcile)
        self.last_rebalance_clock_source = "record_rebalance"
        self.last_rebalance_reconciled = False
        self.last_rebalance_reconciled_from = None
        self.save_state()

    @staticmethod
    def _parse_event_clock(value: Any) -> Optional[datetime]:
        """Parse order-event / ISO timestamps as timezone-aware UTC."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            if "T" in text:
                dt = datetime.fromisoformat(text)
            else:
                dt = datetime.strptime(text[:10], "%Y-%m-%d")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def reconcile_last_rebalance_from_event(
        self,
        event_ts: Any,
        *,
        source: str = "order_event_timestamp",
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Advance ``last_rebalance`` from order-event time when controller lags.

        Event-sourcing practice (Batch DX / DW dual-clock): treat fill /
        rebalance_health ``last_execution_at`` as authoritative for the
        business clock. Does **not** invent YTD cost rows — budget tracking
        still requires ``record_rebalance`` / explicit cost ingress.
        """
        event_dt = self._parse_event_clock(event_ts)
        if event_dt is None:
            return {
                "reconciled": False,
                "advanced": False,
                "reason": "invalid_event_ts",
                "event_ts": str(event_ts) if event_ts is not None else None,
            }

        prior = self._as_utc(self.last_rebalance)
        # Advance when missing or strictly behind event clock (1s tolerance)
        if prior is not None and (event_dt - prior).total_seconds() <= 1.0:
            return {
                "reconciled": False,
                "advanced": False,
                "reason": "controller_not_behind",
                "controller_last_rebalance": prior.isoformat(),
                "event_ts": event_dt.isoformat(),
                "source": source,
            }

        prior_iso = prior.isoformat() if prior else None
        self.last_rebalance = event_dt
        self.last_rebalance_clock_source = source
        self.last_rebalance_reconciled = True
        self.last_rebalance_reconciled_from = prior_iso
        if persist:
            self.save_state()
        return {
            "reconciled": True,
            "advanced": True,
            "reason": "advanced_from_order_event",
            "controller_last_rebalance_before": prior_iso,
            "controller_last_rebalance": event_dt.isoformat(),
            "event_ts": event_dt.isoformat(),
            "source": source,
        }

    def reconcile_from_rebalance_health(
        self,
        rebalance_health: Optional[Dict[str, Any]] = None,
        *,
        health_path: Optional[str | Path] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Read ``rebalance_health`` next_rebalance last_execution and reconcile."""
        payload = rebalance_health
        if payload is None:
            path = Path(health_path) if health_path is not None else (
                self.data_dir / "rebalance_health.json"
            )
            if not path.exists():
                return {
                    "reconciled": False,
                    "advanced": False,
                    "reason": "rebalance_health_missing",
                }
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("rebalance_health load for reconcile failed: %s", exc)
                return {
                    "reconciled": False,
                    "advanced": False,
                    "reason": "rebalance_health_unreadable",
                }
        if not isinstance(payload, dict):
            return {
                "reconciled": False,
                "advanced": False,
                "reason": "rebalance_health_invalid",
            }
        next_reb = payload.get("next_rebalance")
        if not isinstance(next_reb, dict):
            return {
                "reconciled": False,
                "advanced": False,
                "reason": "next_rebalance_missing",
            }
        event_ts = next_reb.get("last_execution_at")
        source = (
            str(next_reb.get("last_execution_clock") or "order_event_timestamp")
        )
        return self.reconcile_last_rebalance_from_event(
            event_ts, source=source, persist=persist
        )

    def state_to_dict(self) -> Dict[str, Any]:
        """Serialize durable controller fields for disk."""
        return {
            "schema_version": "smart-rebalance-state/v1",
            "ytd_costs": list(self.cost_tracker.ytd_costs),
            "last_rebalance": (
                self.last_rebalance.isoformat() if self.last_rebalance else None
            ),
            "last_rebalance_clock_source": self.last_rebalance_clock_source,
            "last_rebalance_reconciled": bool(self.last_rebalance_reconciled),
            "last_rebalance_reconciled_from": self.last_rebalance_reconciled_from,
            "ledger_sanitized": bool(self.ledger_sanitized),
            "ledger_sanitize_report": self.ledger_sanitize_report,
            "quarantined_costs": list(self.cost_tracker.quarantined_costs),
            "max_single_trade_cost_bps": self.cost_tracker.max_single_trade_cost_bps,
            "deferred_until": (
                self.deferred_until.isoformat() if self.deferred_until else None
            ),
            "updated_at": datetime.now().isoformat(),
        }

    def load_state(self, path: Optional[str | Path] = None) -> bool:
        """Load YTD costs and last_rebalance from JSON. Returns True if loaded."""
        state_file = Path(path) if path is not None else self.state_path
        if not state_file.exists():
            return False
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("smart_rebalance state load failed (%s): %s", state_file, exc)
            return False

        if not isinstance(data, dict):
            return False

        costs = data.get("ytd_costs")
        if isinstance(costs, list):
            # Keep only well-formed entries
            cleaned = []
            for entry in costs:
                if not isinstance(entry, dict):
                    continue
                try:
                    cleaned.append(
                        {
                            "cost_bps": float(entry.get("cost_bps", 0)),
                            "date": str(entry.get("date", "")),
                            "symbols": list(entry.get("symbols") or []),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            self.cost_tracker.ytd_costs = cleaned

        # Batch DZ: restore prior quarantine audit trail
        q_costs = data.get("quarantined_costs")
        if isinstance(q_costs, list):
            q_cleaned = []
            for entry in q_costs:
                if not isinstance(entry, dict):
                    continue
                try:
                    q_cleaned.append(
                        {
                            "cost_bps": float(entry.get("cost_bps", 0)),
                            "date": str(entry.get("date", "")),
                            "symbols": list(entry.get("symbols") or []),
                            "reason": str(
                                entry.get("reason")
                                or "above_max_single_trade_cost_bps"
                            ),
                            "cap_bps": entry.get("cap_bps"),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            self.cost_tracker.quarantined_costs = q_cleaned

        lr = data.get("last_rebalance")
        if lr:
            try:
                self.last_rebalance = (
                    datetime.fromisoformat(lr)
                    if "T" in str(lr)
                    else datetime.strptime(str(lr)[:10], "%Y-%m-%d")
                )
            except (TypeError, ValueError):
                self.last_rebalance = None

        src = data.get("last_rebalance_clock_source")
        self.last_rebalance_clock_source = str(src) if src else None
        self.last_rebalance_reconciled = bool(data.get("last_rebalance_reconciled"))
        rf = data.get("last_rebalance_reconciled_from")
        self.last_rebalance_reconciled_from = str(rf) if rf else None
        self.ledger_sanitized = bool(data.get("ledger_sanitized"))
        rep = data.get("ledger_sanitize_report")
        self.ledger_sanitize_report = rep if isinstance(rep, dict) else None

        du = data.get("deferred_until")
        if du:
            try:
                self.deferred_until = datetime.fromisoformat(str(du))
            except (TypeError, ValueError):
                self.deferred_until = None

        # Batch DY/DZ: dedupe + quarantine outliers on load; persist if changed
        try:
            report = self.cost_tracker.sanitize_ledger()
            if report.get("changed"):
                self.ledger_sanitized = True
                self.ledger_sanitize_report = report
                self.save_state()
            elif data.get("ledger_sanitized"):
                self.ledger_sanitized = True
        except Exception as exc:  # noqa: BLE001 — never fail load on sanitize
            logger.warning("cost ledger sanitize on load skipped: %s", exc)

        return True

    def save_state(self, path: Optional[str | Path] = None) -> bool:
        """Atomically persist controller state to JSON. Returns True on success."""
        state_file = Path(path) if path is not None else self.state_path
        payload = self.state_to_dict()
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".smart_rebalance_state.",
                suffix=".tmp",
                dir=str(state_file.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    json.dump(payload, tmp, indent=2)
                    tmp.write("\n")
                os.replace(tmp_name, state_file)
            except Exception as write_exc:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                # Surface unexpected errors; serialize/IO failures stay soft
                if isinstance(write_exc, (OSError, TypeError, ValueError, AttributeError)):
                    raise write_exc
                raise
            return True
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("smart_rebalance state save failed (%s): %s", state_file, exc)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status for dashboard/monitoring."""
        return {
            'ytd_cost_bps': self.cost_tracker.ytd_total_bps,
            'ytd_cost_pct': round(self.cost_tracker.ytd_total_pct * 100, 3),
            'remaining_budget_pct': round(self.cost_tracker.remaining_budget_pct * 100, 3),
            'remaining_budget_ratio': round(self.cost_tracker.remaining_budget_pct, 6),
            # Unit honesty: pct is percent-of-portfolio (0.5 means 0.5%),
            # ratio is portfolio fraction (0.005 means 0.5%).
            'remaining_budget_pct_unit': 'percent_of_portfolio',
            'remaining_budget_ratio_unit': 'portfolio_fraction',
            'is_over_budget': self.cost_tracker.is_over_budget(),
            'is_warning': self.cost_tracker.is_warning(),
            'ytd_cost_entries': len(self.cost_tracker._costs_in_year()),
            'ledger_sanitized': bool(self.ledger_sanitized),
            'ledger_sanitize_report': self.ledger_sanitize_report,
            'max_single_trade_cost_bps': self.cost_tracker.max_single_trade_cost_bps,
            'ytd_outlier_quarantined_count': len(self.cost_tracker.quarantined_costs),
            'ytd_outlier_quarantined_bps': round(
                sum(
                    float(c.get("cost_bps", 0) or 0)
                    for c in self.cost_tracker.quarantined_costs
                    if isinstance(c, dict)
                ),
                4,
            ),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'last_rebalance_clock_source': self.last_rebalance_clock_source,
            'last_rebalance_reconciled': bool(self.last_rebalance_reconciled),
            'last_rebalance_reconciled_from': self.last_rebalance_reconciled_from,
            'deferred_until': self.deferred_until.isoformat() if self.deferred_until else None,
            'config': {
                'drift_threshold': self.config['drift_threshold'],
                'vpin_threshold': self.config['vpin']['threshold'],
                'optimal_window': f"{self.config['timing']['optimal_start']}:00-{self.config['timing']['optimal_end']}:00 ET",
                'annual_cost_limit': f"{self.config['cost_budget']['annual_limit'] * 100:.1f}%",
                'max_single_trade_cost_bps': self.cost_tracker.max_single_trade_cost_bps,
            },
        }


def create_sample_portfolio() -> PortfolioSnapshot:
    """Create a sample portfolio for testing."""
    return PortfolioSnapshot(
        holdings={
            'SPY': 46000,
            'GLD': 38000,
            'TLT': 16000,
        },
        targets=dict(BASE_ALLOCATION),
        total_value=100000,
        timestamp=datetime.now(),
    )


def demo():
    """Demonstrate the smart rebalancing controller (no durable state I/O)."""
    controller = SmartRebalancingController(load_state=False)
    # Avoid writing into shared DATA_DIR during CLI demos
    controller.state_path = Path(os.devnull)

    # Scenario 1: No drift — skip
    portfolio = create_sample_portfolio()
    market = MarketConditions(vpin=0.30)
    result = controller.should_rebalance(portfolio, market)
    logger.info("Scenario 1 (no drift): %s — %s", result.decision.value, result.reason)

    # Scenario 2: 12% drift, low VPIN, in window — execute
    portfolio.holdings['SPY'] = 52000
    portfolio.holdings['GLD'] = 33000
    portfolio.holdings['TLT'] = 15000
    now = datetime(2026, 5, 13, 12, 0)  # Noon ET
    result = controller.should_rebalance(portfolio, market, now=now)
    logger.info("Scenario 2 (12%% drift, noon): %s — %s", result.decision.value, result.reason)
    logger.info("  Urgency: %s, Cost: %.1f bps", result.urgency.value, result.estimated_cost_bps)

    # Scenario 3: 12% drift, high VPIN — defer
    market_high_vpin = MarketConditions(vpin=0.60)
    result = controller.should_rebalance(portfolio, market_high_vpin, now=now)
    logger.info("Scenario 3 (12%% drift, VPIN=0.60): %s — %s", result.decision.value, result.reason)

    # Scenario 4: 12% drift, outside optimal window — defer
    morning = datetime(2026, 5, 13, 9, 30)  # Market open
    result = controller.should_rebalance(portfolio, market, now=morning)
    logger.info("Scenario 4 (12%% drift, 9:30am): %s — %s", result.decision.value, result.reason)

    # Scenario 5: 26% drift — emergency override
    portfolio.holdings['SPY'] = 60000
    portfolio.holdings['GLD'] = 28000
    portfolio.holdings['TLT'] = 12000
    result = controller.should_rebalance(portfolio, market_high_vpin, now=morning)
    logger.info("Scenario 5 (26%% drift, high VPIN, morning): %s — %s", result.decision.value, result.reason)

    # Print status
    logger.info("\nController Status: %s", json.dumps(controller.get_status(), indent=2))


if __name__ == '__main__':
    demo()
