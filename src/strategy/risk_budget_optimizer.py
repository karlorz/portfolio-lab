"""
Portfolio-Lab v6.04: Factor Risk Budgeting & Scenario Analyzer

Bridges v6.03 Risk Factor Decomposition with v6.01 Regime-Constrained Optimizer:

1. Loads factor risk decomposition (v6.03) to understand current risk profile
2. Computes gaps between current factor risk contributions and target budgets
3. Runs stress scenarios (equity crash, rate spike, gold rally, stagflation)
4. Adjusts optimizer constraints to meet factor budgets
5. Risk budget signal generation (not integrated into EnsembleVoter)

No ML dependencies — pure numpy + scipy.

Usage:
    python -m src.strategy.risk_budget_optimizer gaps
    python -m src.strategy.risk_budget_optimizer scenario --name equity_crash
    python -m src.strategy.risk_budget_optimizer all-scenarios
    python -m src.strategy.risk_budget_optimizer optimize
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.paths import BASE_ALLOCATION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = DATA_DIR / "risk_budget_state.json"
PRICES_PATH = PROJECT_ROOT / "public" / "data" / "prices.json"

# ── Asset Universe ──────────────────────────────────────────────────────────

ASSETS = ["SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH"]

# Hard bounds (matching regime_optimizer.py)
HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "SPY": (0.36, 0.56),
    "GLD": (0.28, 0.48),
    "TLT": (0.06, 0.26),
    "IEF": (0.00, 0.10),
    "SHY": (0.00, 0.10),
    "BTC": (0.00, 0.04),
    "ETH": (0.00, 0.03),
}

# ── Default Factor Risk Budgets ─────────────────────────────────────────────
# These represent reasonable target ranges for the 5-factor risk decomposition.
# Factor contributions are % of TOTAL portfolio variance (systematic + idiosyncratic).
#
# Rationale:
#   - Equity Beta: 25-45% (balanced multi-asset portfolio)
#   - Duration: 3-15% (bond sleeve dampens equity, not a primary risk driver)
#   - Gold Beta: 8-20% (strategic diversifier)
#   - Crypto Beta: 0-8% (small tactical allocation)
#   - FX Beta: 15-30% (EFA exposure from international)
#   - Idiosyncratic: 10-30% (diversification benefit)

DEFAULT_FACTOR_BUDGETS: Dict[str, Dict[str, float]] = {
    "equity": {"min": 25.0, "max": 45.0},
    "duration": {"min": 3.0, "max": 15.0},
    "gold": {"min": 8.0, "max": 20.0},
    "crypto": {"min": 0.0, "max": 8.0},
    "fx": {"min": 10.0, "max": 30.0},
    "idiosyncratic": {"min": 10.0, "max": 40.0},
}

# Regime-adjusted factor budget multipliers
# During crisis: tighten equity, loosen duration/gold
REGIME_BUDGET_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "low_vol": {"equity": 1.0, "duration": 1.0, "gold": 1.0, "crypto": 1.5, "fx": 1.0},
    "normal": {"equity": 1.0, "duration": 1.0, "gold": 1.0, "crypto": 1.0, "fx": 1.0},
    "high_vol": {"equity": 0.8, "duration": 1.2, "gold": 1.2, "crypto": 0.5, "fx": 0.8},
    "crisis": {"equity": 0.6, "duration": 1.5, "gold": 1.3, "crypto": 0.0, "fx": 0.7},
    "recovery": {"equity": 1.1, "duration": 0.9, "gold": 1.0, "crypto": 1.3, "fx": 1.0},
    "unknown": {"equity": 1.0, "duration": 1.0, "gold": 1.0, "crypto": 1.0, "fx": 1.0},
}

# ── Pre-built Scenarios ─────────────────────────────────────────────────────
# Each scenario defines factor shocks (multiplicative return impact)
# and correlation regime (how cross-factor correlations behave).

SCENARIOS: Dict[str, Dict] = {
    "equity_crash": {
        "name": "Equity Crash",
        "description": "S&P 500 drops 20%, gold rallies on safe-haven, bonds rally",
        "shocks": {"equity": -0.20, "duration": 0.05, "gold": 0.10, "fx": -0.03, "crypto": -0.15},
        "correlation_regime": "crisis",
    },
    "rate_spike": {
        "name": "Rate Spike",
        "description": "Bonds sell off (TLT -5%), equities dip, gold mixed",
        "shocks": {"equity": -0.05, "duration": -0.05, "gold": 0.02, "fx": -0.02, "crypto": -0.05},
        "correlation_regime": "high_vol",
    },
    "gold_rally": {
        "name": "Gold Rally",
        "description": "Gold +15% on inflation fears, USD weakens, equities dip slightly",
        "shocks": {"equity": -0.02, "duration": -0.02, "gold": 0.15, "fx": -0.03, "crypto": 0.05},
        "correlation_regime": "normal",
    },
    "stagflation": {
        "name": "Stagflation",
        "description": "Equities -10%, bonds -5%, gold +10% (classic 1970s regime)",
        "shocks": {"equity": -0.10, "duration": -0.05, "gold": 0.10, "fx": -0.05, "crypto": -0.10},
        "correlation_regime": "crisis",
    },
    "recession": {
        "name": "Recession",
        "description": "Equities -15%, bonds rally (flight to quality), gold stable, rates fall",
        "shocks": {"equity": -0.15, "duration": 0.06, "gold": 0.03, "fx": -0.05, "crypto": -0.20},
        "correlation_regime": "high_vol",
    },
    "inflation_spike": {
        "name": "Inflation Spike",
        "description": "Equities -8%, bonds -8%, gold +12%, USD weakens",
        "shocks": {"equity": -0.08, "duration": -0.08, "gold": 0.12, "fx": -0.04, "crypto": 0.08},
        "correlation_regime": "high_vol",
    },
}

# Correlation matrices for scenario regimes (SPY/GLD/TLT)
# These override the normal correlations when running scenario analysis
SCENARIO_CORRELATIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    "normal": {
        "SPY": {"SPY": 1.0, "GLD": 0.15, "TLT": -0.15},
        "GLD": {"SPY": 0.15, "GLD": 1.0, "TLT": 0.20},
        "TLT": {"SPY": -0.15, "GLD": 0.20, "TLT": 1.0},
    },
    "high_vol": {
        "SPY": {"SPY": 1.0, "GLD": 0.05, "TLT": -0.05},
        "GLD": {"SPY": 0.05, "GLD": 1.0, "TLT": 0.10},
        "TLT": {"SPY": -0.05, "GLD": 0.10, "TLT": 1.0},
    },
    "crisis": {
        "SPY": {"SPY": 1.0, "GLD": 0.40, "TLT": 0.10},
        "GLD": {"SPY": 0.40, "GLD": 1.0, "TLT": 0.25},
        "TLT": {"SPY": 0.10, "GLD": 0.25, "TLT": 1.0},
    },
}

# ── Regime-Dependent Covariance (from regime_optimizer.py) ──────────────────
# For budget-constrained optimization, we need annualized covariances
REGIME_COVARIANCES: Dict[str, Dict[str, Dict[str, float]]] = {
    "normal": {
        "SPY": {"SPY": 0.0220, "GLD": 0.0040, "TLT": -0.0020},
        "GLD": {"SPY": 0.0040, "GLD": 0.0160, "TLT": 0.0050},
        "TLT": {"SPY": -0.0020, "GLD": 0.0050, "TLT": 0.0360},
    },
    "high_vol": {
        "SPY": {"SPY": 0.0380, "GLD": 0.0010, "TLT": -0.0010},
        "GLD": {"SPY": 0.0010, "GLD": 0.0200, "TLT": 0.0060},
        "TLT": {"SPY": -0.0010, "GLD": 0.0060, "TLT": 0.0450},
    },
    "crisis": {
        "SPY": {"SPY": 0.0640, "GLD": 0.0280, "TLT": 0.0100},
        "GLD": {"SPY": 0.0280, "GLD": 0.0350, "TLT": 0.0150},
        "TLT": {"SPY": 0.0100, "GLD": 0.0150, "TLT": 0.0600},
    },
}

FACTOR_NAMES: Dict[str, str] = {
    "equity": "Equity Beta",
    "duration": "Duration",
    "gold": "Gold Beta",
    "crypto": "Crypto Beta",
    "fx": "FX Beta",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RiskBudgetGap:
    """Difference between current factor risk contribution and target budget."""

    factor: str
    current_pct: float
    target_min: float
    target_max: float
    gap_min: float  # positive means below min (needs more exposure)
    gap_max: float  # positive means above max (needs less exposure)
    breached: bool  # True if outside target range


@dataclass
class ScenarioResult:
    """Result of a factor stress scenario analysis."""

    scenario_name: str
    description: str
    weights: Dict[str, float]
    portfolio_return_impact: float
    factor_contributions: Dict[str, float]
    var_95_impact: float
    cvar_95_impact: float
    budget_violations: List[str]
    risk_budget_gaps: Dict[str, RiskBudgetGap]
    correlation_regime: str


@dataclass
class BudgetOptimizationResult:
    """Result from budget-constrained portfolio optimization."""

    timestamp: str
    method: str
    regime: str
    original_weights: Dict[str, float]
    optimized_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    factor_contributions_before: Dict[str, float]
    factor_contributions_after: Dict[str, float]
    budget_gaps_before: Dict[str, RiskBudgetGap]
    budget_gaps_after: Dict[str, RiskBudgetGap]
    constraints_satisfied: bool
    all_budgets_met: bool
    portfolio_vol_before: float
    portfolio_vol_after: float


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _load_risk_decomposition() -> Optional[Dict]:
    """Load risk decomposition from v6.03 output (cached state or live).

    Returns:
        PortfolioRiskDecomposition as dict, or None if unavailable.
    """
    # Try loading from risk_decomposition state cache
    rd_state = DATA_DIR / "risk_decomposition_state.json"
    if rd_state.exists():
        try:
            data = json.loads(rd_state.read_text())
            # The state may contain a list or dict with a 'results' or 'last_decomposition' key
            if "last_decomposition" in data:
                return data["last_decomposition"]
            if "results" in data and data["results"]:
                return data["results"][-1] if isinstance(data["results"], list) else data["results"]
            return data
        except Exception as e:
            logger.warning(f"Failed to load risk decomposition state: {e}")

    # Fallback: try running live decomposition
    try:
        from src.monitor.risk_decomposition import decompose_portfolio

        result = decompose_portfolio(weights=BASE_ALLOCATION)
        return result.to_dict()
    except Exception as e:
        logger.warning(f"Failed to run live risk decomposition: {e}")

    return None


def _load_regime_state() -> Dict:
    """Load current regime from v5.73 ML-Light Regime Predictor state."""
    regime_path = DATA_DIR / "regime_classifier_state.json"
    if regime_path.exists():
        try:
            state = json.loads(regime_path.read_text())
            reading = state.get("last_reading", {})
            reading_regime = reading.get("regime")
            # Graceful fallback: if last reading is "unknown" (e.g. data fetch
            # failure), use the more stable root-level current_regime instead.
            if reading_regime is None or reading_regime == "unknown":
                reading_regime = state.get("current_regime", "normal")
            return {
                "regime": reading_regime,
                "confidence": reading.get("confidence", 0.7),
                "previous_regime": state.get("previous_regime"),
                "regime_start_date": state.get("regime_start_date"),
            }
        except Exception as e:
            logger.warning(f"Failed to load regime state: {e}")
    return {"regime": "normal", "confidence": 0.7, "previous_regime": None}


def _load_prices() -> Optional[Dict]:
    """Load price data from pipeline JSON."""
    candidates = [
        PRICES_PATH,
        DATA_DIR / "prices.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (IOError, OSError, json.JSONDecodeError):
                logger.warning("Failed to load prices from %s, trying next candidate", path)
                continue
    return None


def _get_annualized_vol(prices: Dict, symbol: str) -> float:
    """Get annualized volatility for a symbol from price data."""
    if symbol not in prices:
        return 0.15  # fallback volatility guess
    try:
        closes = np.array([p["p"] for p in prices[symbol]], dtype=np.float64)
        if len(closes) < 20:
            return 0.15
        log_rets = np.diff(np.log(closes))
        return float(np.std(log_rets) * np.sqrt(252))
    except (ValueError, ZeroDivisionError):
        logger.warning("Failed to compute annualized volatility for %s, defaulting to 0.15", symbol)
        return 0.15


# ---------------------------------------------------------------------------
# RiskBudgetOptimizer
# ---------------------------------------------------------------------------


class RiskBudgetOptimizer:
    """Factor risk budgeting and scenario analyzer.

    Loads factor risk decomposition, computes budget gaps, runs stress
    scenarios, and optimizes portfolio weights to meet factor risk budgets.

    Integrates with:
    - v6.03 RiskDecomposer (risk factor betas + contributions)
    - v6.01 RegimeOptimizer (weight constraints)
    - Standalone signal generation (not integrated into EnsembleVoter)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        factor_budgets: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.weights = weights or dict(BASE_ALLOCATION)
        self.factor_budgets = factor_budgets or dict(DEFAULT_FACTOR_BUDGETS)
        self.regime_state = _load_regime_state()
        self.current_regime = self.regime_state.get("regime", "normal")
        self._risk_data: Optional[Dict] = None
        self._cached_contributions: Optional[Dict[str, float]] = None
        self._cached_total_vol: Optional[float] = None
        self._cached_systematic_pct: Optional[float] = None
        self._cached_idiosyncratic_pct: Optional[float] = None

    # ── Data Loading ────────────────────────────────────────────────────────

    def load_risk_decomposition(self) -> bool:
        """Load and cache the latest factor risk decomposition.

        Returns:
            True if decomposition loaded successfully.
        """
        data = _load_risk_decomposition()
        if data is None:
            logger.warning("No risk decomposition available")
            return False

        # Normalize: cached decomposition may be the full dict or a nested structure
        self._risk_data = data

        # Extract factor contributions
        contribs = data.get("factor_contributions", {})
        self._cached_contributions = {}
        for factor_key in ["equity", "duration", "gold", "crypto", "fx"]:
            self._cached_contributions[factor_key] = contribs.get(factor_key, 0.0)

        self._cached_total_vol = data.get("total_portfolio_volatility", 0.0)
        self._cached_systematic_pct = data.get("systematic_pct", 0.0)
        self._cached_idiosyncratic_pct = data.get("idiosyncratic_pct", 0.0)

        # Store asset weights from decomposition if available
        dec_weights = data.get("portfolio_weights", {})
        if dec_weights:
            self.weights = dec_weights

        logger.info(
            f"Risk decomposition loaded: "
            f"vol={self._cached_total_vol:.2%}, "
            f"systematic={self._cached_systematic_pct:.1f}%, "
            f"{len(self._cached_contributions)} factors"
        )
        return True

    def factor_contributions(self) -> Dict[str, float]:
        """Get current factor risk contributions (%)."""
        if self._cached_contributions is None:
            self.load_risk_decomposition()
        return self._cached_contributions or {f: 0.0 for f in DEFAULT_FACTOR_BUDGETS}

    def total_portfolio_volatility(self) -> float:
        """Get current annualized portfolio volatility."""
        return self._cached_total_vol or 0.10

    def systematic_pct(self) -> float:
        """Get % of risk explained by systematic factors."""
        return self._cached_systematic_pct or 0.60

    def idiosyncratic_pct(self) -> float:
        """Get % of risk from idiosyncratic noise."""
        return self._cached_idiosyncratic_pct or 0.40

    # ── Budget Management ──────────────────────────────────────────────────

    def set_target_factor_budgets(self, budgets: Dict[str, Dict[str, float]]):
        """Set target risk budget ranges for factors.

        Args:
            budgets: Dict mapping factor_key -> {"min": float, "max": float}
                     where min/max are % of total portfolio variance (0-100).
                     Example: {"equity": {"min": 25, "max": 45}}
        """
        for factor, bounds in budgets.items():
            if factor not in self.factor_budgets:
                logger.warning(f"Unknown factor '{factor}', skipping budget")
                continue
            bmin = bounds.get("min", self.factor_budgets[factor]["min"])
            bmax = bounds.get("max", self.factor_budgets[factor]["max"])
            if bmin < 0 or bmax > 100 or bmin > bmax:
                logger.warning(
                    f"Invalid budget for '{factor}': min={bmin}, max={bmax}, using defaults"
                )
                continue
            self.factor_budgets[factor] = {"min": bmin, "max": bmax}

        logger.info(f"Updated factor budgets: {len(budgets)} factors modified")

    def get_regime_adjusted_budgets(self) -> Dict[str, Dict[str, float]]:
        """Get factor budgets adjusted by current regime state.

        During crisis/volatile regimes, equity budget shrinks and
        duration/gold budgets expand as natural hedges.
        """
        multipliers = REGIME_BUDGET_MULTIPLIERS.get(
            self.current_regime, REGIME_BUDGET_MULTIPLIERS["normal"]
        )
        adjusted = {}
        for factor, bounds in self.factor_budgets.items():
            mult = multipliers.get(factor, 1.0)
            if factor == "idiosyncratic":
                adjusted[factor] = dict(bounds)
                continue
            # Scale both min and max by multiplier
            adjusted[factor] = {
                "min": max(0.0, bounds["min"] * mult),
                "max": min(100.0, bounds["max"] * mult),
            }
            # Ensure min <= max after adjustment
            if adjusted[factor]["min"] > adjusted[factor]["max"]:
                adjusted[factor]["min"] = adjusted[factor]["max"] * 0.8
        return adjusted

    # ── Budget Gap Analysis ─────────────────────────────────────────────────

    def compute_risk_budget_gaps(
        self, contributions: Optional[Dict[str, float]] = None
    ) -> Dict[str, RiskBudgetGap]:
        """Compute gaps between current factor contributions and target budgets.

        Args:
            contributions: Factor contributions dict (auto-loads if None).
                          Values are % of total portfolio variance (e.g., 34.0 = 34%).

        Returns:
            Dict mapping factor_key -> RiskBudgetGap
        """
        if contributions is None:
            contributions = self.factor_contributions()
        if not contributions:
            return {}

        budgets = self.get_regime_adjusted_budgets()
        gaps: Dict[str, RiskBudgetGap] = {}

        for factor in budgets:
            current_raw = contributions.get(factor, 0.0)
            # Ensure we work in percentage units (34.0 = 34%)
            if current_raw < 1.0 and current_raw > 0.0 and budgets[factor]["min"] >= 3:
                # Raw value looks like a decimal (0.34), treat as percentage * 100
                current = current_raw * 100.0
            else:
                current = current_raw

            target_min = budgets[factor]["min"]  # Already percentage
            target_max = budgets[factor]["max"]  # Already percentage

            gap_min = max(0.0, target_min - current)
            gap_max = max(0.0, current - target_max)
            breached = current < target_min or current > target_max

            gaps[factor] = RiskBudgetGap(
                factor=factor,
                current_pct=round(current, 2),
                target_min=round(target_min, 2),
                target_max=round(target_max, 2),
                gap_min=round(gap_min, 2),
                gap_max=round(gap_max, 2),
                breached=breached,
            )

        # Add idiosyncratic budget check
        idio_current = self.idiosyncratic_pct()  # Already in % (e.g., 22.4)
        idio_min = budgets.get("idiosyncratic", {}).get("min", 10)  # Already %
        idio_max = budgets.get("idiosyncratic", {}).get("max", 40)  # Already %
        idio_breached = idio_current < idio_min or idio_current > idio_max
        gaps["idiosyncratic"] = RiskBudgetGap(
            factor="idiosyncratic",
            current_pct=round(idio_current, 2),
            target_min=round(idio_min, 2),
            target_max=round(idio_max, 2),
            gap_min=round(max(0.0, idio_min - idio_current), 2),
            gap_max=round(max(0.0, idio_current - idio_max), 2),
            breached=idio_breached,
        )

        return gaps

    def budget_summary_string(self, gaps: Optional[Dict[str, RiskBudgetGap]] = None) -> str:
        """Generate human-readable budget gap report."""
        if gaps is None:
            gaps = self.compute_risk_budget_gaps()

        lines = [
            f"Factor Risk Budget Report (regime: {self.current_regime.upper()})",
            f"  Portfolio Vol: {self.total_portfolio_volatility():.2%} ann.",
            f"  Systematic Risk: {self.systematic_pct():.1f}%",
            f"  Idiosyncratic: {self.idiosyncratic_pct():.1f}%",
            "",
            f"  {'Factor':20s} {'Current':>8s} {'Min':>8s} {'Max':>8s} {'Gap↓':>8s} {'Gap↑':>8s} {'Status':>10s}",
            f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}",
        ]
        for fname, gap in gaps.items():
            label = FACTOR_NAMES.get(fname, fname)
            status = "BREACHED" if gap.breached else "OK"
            lines.append(
                f"  {label:20s} {gap.current_pct:>7.1f}% "
                f"{gap.target_min:>7.1f}% {gap.target_max:>7.1f}% "
                f"{gap.gap_min:>7.1f}% {gap.gap_max:>7.1f}% "
                f"{status:>10s}"
            )

        breached_count = sum(1 for g in gaps.values() if g.breached)
        lines.append("")
        lines.append(f"  Factors breaching budget: {breached_count}/{len(gaps)}")
        return "\n".join(lines)

    # ── Scenario Analysis ──────────────────────────────────────────────────

    def run_scenario(self, scenario_name: str) -> Optional[ScenarioResult]:
        """Run a pre-built or custom stress scenario.

        Args:
            scenario_name: Name of scenario in SCENARIOS dict.
                          (e.g., 'equity_crash', 'rate_spike')

        Returns:
            ScenarioResult with factor-level breakdown, or None if scenario unknown.
        """
        if scenario_name not in SCENARIOS:
            logger.warning(f"Unknown scenario '{scenario_name}'")
            return None

        scenario = SCENARIOS[scenario_name]
        shocks = scenario["shocks"]
        corr_regime = scenario["correlation_regime"]

        # Decompose the portfolio to understand factor betas
        if self._cached_contributions is None:
            self.load_risk_decomposition()

        # Factor contributions (current)
        factor_contribs = self.factor_contributions()

        # Compute portfolio return impact from factor shocks
        # Portfolio return = Σ(w_i * Σ(β_if * shock_f))
        # Simplified: use factor contributions as proxy for sensitivity
        port_impact = 0.0
        factor_impacts: Dict[str, float] = {}
        for fkey, shock in shocks.items():
            if fkey not in factor_contribs:
                continue
            # Weight factor contribution by shock magnitude
            fcontrib = factor_contribs.get(fkey, 0.0)
            factor_impact = fcontrib * shock
            factor_impacts[fkey] = round(factor_impact * 100, 2)  # in %
            port_impact += factor_impact

        # Compute scenario-adjusted volatility
        scenario_cov = SCENARIO_CORRELATIONS.get(corr_regime, SCENARIO_CORRELATIONS["normal"])
        assets = ["SPY", "GLD", "TLT"]
        base_vols = {}
        prices = _load_prices()
        if prices:
            for a in assets:
                base_vols[a] = _get_annualized_vol(prices, a)

        # Build scenario covariance matrix
        n = len(assets)
        cov_scenario = np.zeros((n, n))
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                corr = scenario_cov.get(a1, {}).get(a2, 1.0 if a1 == a2 else 0.0)
                v1 = base_vols.get(a1, 0.15)
                v2 = base_vols.get(a2, 0.15)
                cov_scenario[i, j] = corr * v1 * v2

        w_vec = np.array([self.weights.get(a, 0.0) for a in assets])
        scenario_vol = max(float(np.sqrt(w_vec @ cov_scenario @ w_vec)), 0.01)

        # VaR 95% (single-tailed normal approximation)
        var_95 = port_impact + scenario_vol * 1.645
        cvar_95 = port_impact + scenario_vol * 2.063  # Approximate CVaR for normal

        # Check budget violations under scenario
        gaps = self.compute_risk_budget_gaps()

        # Add scenario-specific budget adjustments
        budget_violations = []
        for fkey, shock in shocks.items():
            if shock < 0 and fkey in gaps:
                gap = gaps[fkey]
                if gap.breached:
                    budget_violations.append(
                        f"{FACTOR_NAMES.get(fkey, fkey)} already at {gap.current_pct:.1f}% "
                        f"(target {gap.target_min:.1f}-{gap.target_max:.1f}%)"
                    )

        return ScenarioResult(
            scenario_name=scenario_name,
            description=scenario["description"],
            weights=dict(self.weights),
            portfolio_return_impact=round(port_impact * 100, 2),
            factor_contributions=factor_impacts,
            var_95_impact=round(var_95 * 100, 2),
            cvar_95_impact=round(cvar_95 * 100, 2),
            budget_violations=budget_violations,
            risk_budget_gaps=gaps,
            correlation_regime=corr_regime,
        )

    def run_all_scenarios(self) -> Dict[str, ScenarioResult]:
        """Run all pre-built scenarios."""
        results = {}
        for sname in SCENARIOS:
            result = self.run_scenario(sname)
            if result is not None:
                results[sname] = result
        return results

    # ── Optimization ─────────────────────────────────────────────────────────

    def optimize_with_budget(
        self,
        target_budgets: Optional[Dict[str, Dict[str, float]]] = None,
        method: str = "risk_parity",
    ) -> Optional[BudgetOptimizationResult]:
        """Optimize portfolio weights to meet factor risk budgets.

        Uses a constrained search: tries weight adjustments within hard bounds
        to bring factor risk contributions within target ranges.

        Args:
            target_budgets: Optional override budgets (uses current if None).
            method: Optimization approach ('risk_parity', 'min_vol', 'max_sharpe')

        Returns:
            BudgetOptimizationResult with before/after comparison.
        """
        if target_budgets is not None:
            self.set_target_factor_budgets(target_budgets)

        if self._cached_contributions is None:
            self.load_risk_decomposition()

        # Current state
        contribs_before = self.factor_contributions()
        gaps_before = self.compute_risk_budget_gaps(contribs_before)
        vol_before = self.total_portfolio_volatility()

        # ── Optimization via RegimeOptimizer ──────────────────────────
        # Try regime optimizer first, then adjust to meet budgets
        try:
            from src.strategy.regime_optimizer import RegimeConstrainedOptimizer

            optimizer = RegimeConstrainedOptimizer()
            result = optimizer.optimize(method=method)
            optimized_weights = result.weights

            # Extract core 3-asset weights for budget adjustment
            core_optimized = {
                k: optimized_weights.get(k, 0.0) for k in ["SPY", "GLD", "TLT"]
            }

            # If we got a result, re-decompose with optimized weights
            adjusted_weights = self._adjust_weights_to_budgets(
                dict(core_optimized), gaps_before
            )

        except Exception as e:
            logger.warning(f"Regime optimizer failed ({e}), using budget-only adjustment")
            adjusted_weights = self._adjust_weights_to_budgets(
                dict(self.weights), gaps_before
            )

        # Normalize
        total_w = sum(adjusted_weights.values())
        if total_w > 0:
            adjusted_weights = {k: v / total_w for k, v in adjusted_weights.items()}

        # Compute estimated contributions after adjustment
        # (proportional shift based on weight change)
        contribs_after = dict(contribs_before)
        for sym, w_change in adjusted_weights.items():
            old_w = self.weights.get(sym, 0.0)
            if old_w > 0 and contribs_before:
                ratio = adjusted_weights.get(sym, old_w) / old_w if old_w > 0 else 1.0
                # Equity contribution scales with SPY weight
                if sym == "SPY" and "equity" in contribs_after:
                    contribs_after["equity"] = contribs_before.get("equity", 0.0) * ratio
                if sym == "TLT" and "duration" in contribs_after:
                    contribs_after["duration"] = contribs_before.get("duration", 0.0) * ratio
                if sym == "GLD" and "gold" in contribs_after:
                    contribs_after["gold"] = contribs_before.get("gold", 0.0) * ratio

        # Clamp contributions to reasonable range
        for k in contribs_after:
            contribs_after[k] = max(0.0, min(1.0, contribs_after.get(k, 0.0)))

        # Re-normalize
        total_c = sum(contribs_after.values())
        if total_c > 0:
            contribs_after = {k: v / total_c for k, v in contribs_after.items()}

        gaps_after = self.compute_risk_budget_gaps(contribs_after)

        # Check which budgets are met
        all_met = not any(g.breached for g in gaps_after.values())

        # Compute expected vol after adjustment
        vol_after = self._estimate_vol_adjustment(adjusted_weights)

        # Weight changes
        weight_changes = {}
        for sym in set(list(self.weights.keys()) + list(adjusted_weights.keys())):
            old = self.weights.get(sym, 0.0)
            new = adjusted_weights.get(sym, 0.0)
            if abs(old - new) > 0.0001:
                weight_changes[sym] = round(new - old, 4)

        result = BudgetOptimizationResult(
            timestamp=datetime.now().isoformat(),
            method=method,
            regime=self.current_regime,
            original_weights=dict(self.weights),
            optimized_weights=adjusted_weights,
            weight_changes=weight_changes,
            factor_contributions_before=contribs_before,
            factor_contributions_after=contribs_after,
            budget_gaps_before=gaps_before,
            budget_gaps_after=gaps_after,
            constraints_satisfied=self._check_constraints(adjusted_weights),
            all_budgets_met=all_met,
            portfolio_vol_before=vol_before,
            portfolio_vol_after=vol_after,
        )

        # Update state
        self.weights = adjusted_weights
        self._save_state(result)
        return result

    def _adjust_weights_to_budgets(
        self,
        current_weights: Dict[str, float],
        gaps: Dict[str, RiskBudgetGap],
    ) -> Dict[str, float]:
        """Simple heuristic weight adjustment to reduce budget breaches.

        This is a lightweight adjustment that moves weights in the right direction.
        Full convex optimization with factor budget constraints requires the
        RegimeOptimizer with custom constraints.
        """
        adjusted = dict(current_weights)
        epsilon = 0.01  # Minimum adjustment step

        for fname, gap in gaps.items():
            if not gap.breached:
                continue

            if fname == "equity":
                # Excess equity risk: reduce SPY, increase GLD/TLT
                if gap.gap_max > 0:
                    reduction = min(gap.gap_max / 100.0, 0.05)  # Max 5% reduction
                    adjusted["SPY"] = max(HARD_BOUNDS["SPY"][0], adjusted.get("SPY", 0.46) - reduction)
                    # Reallocate to GLD/TLT proportionally
                    gld_inc = reduction * 0.6
                    tlt_inc = reduction * 0.4
                    adjusted["GLD"] = min(HARD_BOUNDS["GLD"][1], adjusted.get("GLD", 0.38) + gld_inc)
                    adjusted["TLT"] = min(HARD_BOUNDS["TLT"][1], adjusted.get("TLT", 0.16) + tlt_inc)
                # Too little equity risk: increase SPY
                elif gap.gap_min > 0:
                    increase = min(gap.gap_min / 100.0, 0.03)
                    adjusted["SPY"] = min(HARD_BOUNDS["SPY"][1], adjusted.get("SPY", 0.46) + increase)
                    # Reduce GLD/TLT proportionally
                    adjusted["GLD"] = max(HARD_BOUNDS["GLD"][0], adjusted.get("GLD", 0.38) - increase * 0.6)
                    adjusted["TLT"] = max(HARD_BOUNDS["TLT"][0], adjusted.get("TLT", 0.16) - increase * 0.4)

            elif fname == "duration":
                # Too much duration risk: reduce TLT, increase IEF/SHY
                if gap.gap_max > 0:
                    reduction = min(gap.gap_max / 100.0, 0.04)
                    adjusted["TLT"] = max(HARD_BOUNDS["TLT"][0], adjusted.get("TLT", 0.16) - reduction)
                    adjusted["IEF"] = min(HARD_BOUNDS["IEF"][1], adjusted.get("IEF", 0.0) + reduction * 0.5)
                    adjusted["SHY"] = min(HARD_BOUNDS["SHY"][1], adjusted.get("SHY", 0.0) + reduction * 0.5)
                # Too little duration: increase TLT
                elif gap.gap_min > 0:
                    increase = min(gap.gap_min / 100.0, 0.03)
                    adjusted["TLT"] = min(HARD_BOUNDS["TLT"][1], adjusted.get("TLT", 0.16) + increase)
                    adjusted["IEF"] = max(0.0, adjusted.get("IEF", 0.0) - increase * 0.5)
                    adjusted["SHY"] = max(0.0, adjusted.get("SHY", 0.0) - increase * 0.3)

            elif fname == "gold":
                if gap.gap_max > 0:
                    reduction = min(gap.gap_max / 100.0, 0.04)
                    adjusted["GLD"] = max(HARD_BOUNDS["GLD"][0], adjusted.get("GLD", 0.38) - reduction)
                    adjusted["SPY"] = min(HARD_BOUNDS["SPY"][1], adjusted.get("SPY", 0.46) + reduction * 0.5)
                    adjusted["TLT"] = min(HARD_BOUNDS["TLT"][1], adjusted.get("TLT", 0.16) + reduction * 0.5)
                elif gap.gap_min > 0:
                    increase = min(gap.gap_min / 100.0, 0.03)
                    adjusted["GLD"] = min(HARD_BOUNDS["GLD"][1], adjusted.get("GLD", 0.38) + increase)
                    adjusted["SPY"] = max(HARD_BOUNDS["SPY"][0], adjusted.get("SPY", 0.46) - increase * 0.5)
                    adjusted["TLT"] = max(HARD_BOUNDS["TLT"][0], adjusted.get("TLT", 0.16) - increase * 0.5)

        return adjusted

    def _estimate_vol_adjustment(self, weights: Dict[str, float]) -> float:
        """Estimate portfolio volatility after weight adjustment."""
        cov = REGIME_COVARIANCES.get(self.current_regime, REGIME_COVARIANCES["normal"])
        assets = ["SPY", "GLD", "TLT"]
        w = np.array([weights.get(a, 0.0) for a in assets])
        n = len(assets)
        cov_mat = np.zeros((n, n))
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                cov_mat[i, j] = cov.get(a1, {}).get(a2, 0.0)
        var = float(w @ cov_mat @ w)
        return float(np.sqrt(max(var, 1e-10)))

    def _check_constraints(self, weights: Dict[str, float]) -> bool:
        """Check if weights satisfy all hard bounds."""
        ok = True
        for asset, w in weights.items():
            if asset in HARD_BOUNDS:
                lo, hi = HARD_BOUNDS[asset]
                if w < lo - 0.001 or w > hi + 0.001:
                    ok = False
        return ok

    # ── Integration with RegimeOptimizer ───────────────────────────────────

    def adjust_optimizer_constraints(
        self, budget_gaps: Optional[Dict[str, RiskBudgetGap]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Generate adjusted hard bounds for the RegimeOptimizer based on budget gaps.

        When a factor budget is breached, tighten the corresponding asset bounds.

        Args:
            budget_gaps: Pre-computed gaps (auto-computes if None).

        Returns:
            Dict mapping asset -> (min, max) adjusted bounds.
        """
        if budget_gaps is None:
            budget_gaps = self.compute_risk_budget_gaps()

        adjusted_bounds = dict(HARD_BOUNDS)

        for fname, gap in budget_gaps.items():
            if not gap.breached:
                continue

            if fname == "equity" and gap.gap_max > 0:
                # Too much equity risk: tighten SPY upper bound
                current_upper = adjusted_bounds.get("SPY", (0.36, 0.56))[1]
                new_upper = max(HARD_BOUNDS["SPY"][0], current_upper - gap.gap_max / 100.0)
                adjusted_bounds["SPY"] = (HARD_BOUNDS["SPY"][0], new_upper)

            elif fname == "equity" and gap.gap_min > 0:
                # Too little equity risk: raise SPY lower bound
                current_lower = adjusted_bounds.get("SPY", (0.36, 0.56))[0]
                new_lower = min(HARD_BOUNDS["SPY"][1], current_lower + gap.gap_min / 100.0)
                adjusted_bounds["SPY"] = (new_lower, HARD_BOUNDS["SPY"][1])

            elif fname == "duration" and gap.gap_max > 0:
                # Too much duration risk: tighten TLT upper bound
                current_upper = adjusted_bounds.get("TLT", (0.06, 0.26))[1]
                new_upper = max(HARD_BOUNDS["TLT"][0], current_upper - gap.gap_max / 100.0)
                adjusted_bounds["TLT"] = (HARD_BOUNDS["TLT"][0], new_upper)

            elif fname == "gold" and gap.gap_max > 0:
                # Too much gold risk: tighten GLD upper bound
                current_upper = adjusted_bounds.get("GLD", (0.28, 0.48))[1]
                new_upper = max(HARD_BOUNDS["GLD"][0], current_upper - gap.gap_max / 100.0)
                adjusted_bounds["GLD"] = (HARD_BOUNDS["GLD"][0], new_upper)

        return adjusted_bounds

    # ── Signal Generation for EnsembleVoter ────────────────────────────────

    def to_signal_value(self) -> float:
        """Generate a -1 to +1 signal value based on risk budget status.

        - Negative: factor budgets are breached (risk-off)
        - Positive: factor budgets are within range (risk-on)
        - Magnitude reflects severity of breaches

        This generates a risk budget signal (not integrated into EnsembleVoter).
        """
        gaps = self.compute_risk_budget_gaps()
        if not gaps:
            return 0.0

        # Count breaches and their severity
        breach_score = 0.0
        total_factors = len(gaps)

        for gap in gaps.values():
            if gap.breached:
                # Each breached factor contributes -0.2 to -1.0 depending on gap size
                gap_magnitude = max(gap.gap_max, gap.gap_min)
                breach_score -= min(1.0, gap_magnitude / 15.0)  # Each 15% gap = full -1

        # Clamp to [-1, 0] — budget constraints are asymmetric (breaches are negative)
        # If no breaches, small positive signal
        if breach_score == 0.0:
            return 0.05  # Slight positive: budgets in check

        return max(-1.0, breach_score)

    def scenario_summary_string(self, result: ScenarioResult) -> str:
        """Format a scenario result for human consumption."""
        lines = [
            f"Scenario: {result.scenario_name}",
            f"  Description: {result.description}",
            f"  Correlation Regime: {result.correlation_regime}",
            "",
            f"  Portfolio Return Impact: {result.portfolio_return_impact:+.2f}%",
            f"  VaR (95%): {result.var_95_impact:+.2f}%",
            f"  CVaR (95%): {result.cvar_95_impact:+.2f}%",
            "",
            "  Factor Contributions to P&L:",
        ]
        for fkey, impact in sorted(result.factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {FACTOR_NAMES.get(fkey, fkey):20s} {impact:+.2f}%")
        if result.budget_violations:
            lines.append("")
            lines.append("  ⚠ Pre-existing Budget Violations Amplified:")
            for v in result.budget_violations:
                lines.append(f"    - {v}")
        return "\n".join(lines)

    def optimize_summary_string(self, result: BudgetOptimizationResult) -> str:
        """Format optimization result for human consumption."""
        lines = [
            f"Budget-Constrained Optimization (method: {result.method}, regime: {result.regime})",
            f"  {'Metric':30s} {'Before':>10s} {'After':>10s}",
            f"  {'-'*30} {'-'*10} {'-'*10}",
            f"  {'Portfolio Vol (ann.)':30s} {result.portfolio_vol_before:>9.2%} {result.portfolio_vol_after:>9.2%}",
            f"  {'All budgets met':30s} {'No':>10s} {str(result.all_budgets_met):>10s}",
            "",
            "  Weight Changes:",
        ]
        for sym, change in result.weight_changes.items():
            lines.append(f"    {sym:6s}: {change:+.2%}")
        lines.append("")
        lines.append("  Budget Gaps: Before vs After")
        lines.append(f"  {'Factor':20s} {'Before':>10s} {'After':>10s}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*10}")
        for fname in result.budget_gaps_before:
            bb = result.budget_gaps_before.get(fname)
            ba = result.budget_gaps_after.get(fname)
            if bb:
                b_status = "⚠" if bb.breached else "✓"
                a_status = "⚠" if (ba and ba.breached) else "✓"
                lines.append(
                    f"  {FACTOR_NAMES.get(fname, fname):20s} "
                    f"{bb.current_pct:>7.1f}%{b_status} "
                    f"{(ba.current_pct if ba else 0):>7.1f}%{a_status}"
                )
        lines.append("")
        return "\n".join(lines)

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save_state(self, result: BudgetOptimizationResult):
        """Persist optimizer state to disk."""
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "timestamp": result.timestamp,
                "regime": self.current_regime,
                "weights": self.weights,
                "regime_weights": result.optimized_weights,
                "all_budgets_met": result.all_budgets_met,
                "portfolio_vol_before": result.portfolio_vol_before,
                "portfolio_vol_after": result.portfolio_vol_after,
            }
            STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def load_state(self) -> bool:
        """Load persisted optimizer state."""
        if not STATE_PATH.exists():
            return False
        try:
            state = json.loads(STATE_PATH.read_text())
            self.weights = state.get("weights", self.weights)
            self.current_regime = state.get("regime", self.current_regime)
            logger.info(f"Loaded risk budget optimizer state from {STATE_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


def create_risk_budget_signal() -> float:
    """One-shot risk budget signal (standalone, not integrated into EnsembleVoter).

    Returns:
        Signal value (-1 to +1) indicating risk budget health.
    """
    try:
        optimizer = RiskBudgetOptimizer()
        optimizer.load_risk_decomposition()
        return optimizer.to_signal_value()
    except Exception as e:
        logger.warning(f"Risk budget signal error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for risk budget optimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="v6.04 Factor Risk Budgeting & Scenario Analyzer"
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # gaps
    p_gaps = sub.add_parser("gaps", help="Show current budget gaps")
    p_gaps.add_argument(
        "--weights",
        default=None,
        help="Portfolio weights as slash-separated (default: 46/38/16)",
    )

    # scenario
    p_scen = sub.add_parser("scenario", help="Run a stress scenario")
    p_scen.add_argument(
        "--name",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Scenario name",
    )

    # all-scenarios
    sub.add_parser("all-scenarios", help="Run all pre-built scenarios")

    # optimize
    p_opt = sub.add_parser("optimize", help="Run budget-constrained optimization")
    p_opt.add_argument(
        "--method",
        default="risk_parity",
        choices=["min_vol", "max_sharpe", "risk_parity"],
        help="Optimization method (default: risk_parity)",
    )

    # signal
    sub.add_parser("signal", help="Generate risk budget signal value")

    # status
    sub.add_parser("status", help="Show saved state")

    args = parser.parse_args()

    # Parse weights if provided
    weights = None
    if hasattr(args, "weights") and args.weights:
        parts = [float(x) for x in args.weights.split("/")]
        symbols = ["SPY", "GLD", "TLT"]
        weights = dict(zip(symbols, [p / 100.0 for p in parts]))

    optimizer = RiskBudgetOptimizer(weights=weights)
    optimizer.load_risk_decomposition()

    try:
        if args.command == "gaps":
            gaps = optimizer.compute_risk_budget_gaps()
            print(optimizer.budget_summary_string(gaps))

        elif args.command == "scenario":
            result = optimizer.run_scenario(args.name)
            if result:
                print(optimizer.scenario_summary_string(result))
            else:
                print(f"Unknown scenario: {args.name}")
                sys.exit(1)

        elif args.command == "all-scenarios":
            results = optimizer.run_all_scenarios()
            for sname, result in results.items():
                print(optimizer.scenario_summary_string(result))
                print()

        elif args.command == "optimize":
            result = optimizer.optimize_with_budget(method=args.method)
            if result:
                print(optimizer.optimize_summary_string(result))
            else:
                print("Optimization failed")
                sys.exit(1)

        elif args.command == "signal":
            signal = optimizer.to_signal_value()
            print(f"RISK_BUDGET signal: {signal:+.4f}")

        elif args.command == "status":
            if STATE_PATH.exists():
                print(json.dumps(json.loads(STATE_PATH.read_text()), indent=2))
            else:
                print("No state saved. Run 'optimize' first.")

        else:
            # Default: show gaps
            gaps = optimizer.compute_risk_budget_gaps()
            print(optimizer.budget_summary_string(gaps))
            print()
            print("Run --help for available commands.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
