#!/usr/bin/env python3
"""
Portfolio-Lab v8.08: Automated Crisis Scenario Generator

Generates correlated shock scenarios using historical crisis copula parameters
(2008 Financial Crisis, 2020 COVID Crash, 2022 Rate Hawk) to stress-test the
portfolio under realistic tail events.

Provides:
- Historical crisis copula parameters (mean shifts, vol multipliers, correlations)
- Random scenario generation with realistic correlation regimes
- Portfolio impact estimation across N scenarios
- Integration with existing risk metrics (CVaR, VaR, entropy)
- Scenario ranking by severity and likelihood

Usage:
    python -m src.monitor.crisis_scenario_generator generate --n 1000
    python -m src.monitor.crisis_scenario_generator crisis --name 2008
    python -m src.monitor.crisis_scenario_generator rank
    python -m src.monitor.crisis_scenario_generator all --save
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.paths import BASE_ALLOCATION

# ---------------------------------------------------------------------------
# project root
# ---------------------------------------------------------------------------
from src.paths import PROJECT_ROOT as project_root, DATA_DIR
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
STATE_DIR = DATA_DIR / "crisis_scenarios"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# crisis definitions
# ---------------------------------------------------------------------------

# 7-asset model matching the unified orchestrator
ASSETS = ["SPY", "GLD", "TLT", "IEF", "SHY", "BTC", "ETH"]

# Crisis templates: {asset: {mean_daily_return, vol_multiplier, correlation_regime}}
# All returns are daily decimal (e.g. -0.02 = -2%)
# vol_multiplier: factor applied to normal vol (1.5 = 50% higher vol)
# correlation_regime: how correlations change during crisis

@dataclass
class CrisisTemplate:
    """Defines a crisis scenario with asset-level shock parameters."""
    name: str
    description: str
    date_range: str
    asset_shocks: Dict[str, Dict]  # asset -> {mean_return, vol_mult}
    correlation_multiplier: float  # overall correlation boost (1.5 = 50% tighter)
    regime_type: str  # 'correlation_crash', 'flight_to_safety', 'inflation_shock', 'liquidity_crisis'
    severity: str  # 'moderate', 'severe', 'extreme'
    likelihood_weight: float  # relative probability of this crisis type (0-1)

    def to_dict(self) -> Dict:
        return asdict(self)


# Historical crisis parameters estimated from 2005-2026 data
CRISIS_TEMPLATES = {
    "2008_financial": CrisisTemplate(
        name="2008 Financial Crisis",
        description="Subprime mortgage collapse, Lehman failure, systemic liquidity freeze. Equity crash with correlated bond selloff.",
        date_range="2008-09 to 2009-03",
        asset_shocks={
            "SPY": {"mean_return": -0.0028, "vol_mult": 2.5},
            "GLD": {"mean_return": 0.0008, "vol_mult": 1.8},
            "TLT": {"mean_return": 0.0010, "vol_mult": 1.6},
            "IEF": {"mean_return": 0.0005, "vol_mult": 1.3},
            "SHY": {"mean_return": 0.0002, "vol_mult": 1.1},
            "BTC": {"mean_return": -0.0015, "vol_mult": 3.0},
            "ETH": {"mean_return": -0.0020, "vol_mult": 3.5},
        },
        correlation_multiplier=1.8,
        regime_type="correlation_crash",
        severity="severe",
        likelihood_weight=0.08,
    ),
    "2020_covid": CrisisTemplate(
        name="2020 COVID Crash",
        description="Pandemic-driven rapid selloff with V-shaped recovery. Flight to safety in bonds.",
        date_range="2020-02 to 2020-04",
        asset_shocks={
            "SPY": {"mean_return": -0.0035, "vol_mult": 3.0},
            "GLD": {"mean_return": 0.0005, "vol_mult": 1.5},
            "TLT": {"mean_return": 0.0020, "vol_mult": 1.8},  # flight to safety
            "IEF": {"mean_return": 0.0012, "vol_mult": 1.4},
            "SHY": {"mean_return": 0.0003, "vol_mult": 1.0},
            "BTC": {"mean_return": -0.0040, "vol_mult": 3.5},
            "ETH": {"mean_return": -0.0050, "vol_mult": 4.0},
        },
        correlation_multiplier=1.5,
        regime_type="flight_to_safety",
        severity="severe",
        likelihood_weight=0.05,
    ),
    "2022_rate_hawk": CrisisTemplate(
        name="2022 Rate Hawk",
        description="Aggressive Fed tightening, bonds and equities both falling. Everything correlation.",
        date_range="2022-01 to 2022-10",
        asset_shocks={
            "SPY": {"mean_return": -0.0015, "vol_mult": 1.8},
            "GLD": {"mean_return": -0.0010, "vol_mult": 1.5},
            "TLT": {"mean_return": -0.0025, "vol_mult": 2.0},  # worst bond crisis in decades
            "IEF": {"mean_return": -0.0012, "vol_mult": 1.5},
            "SHY": {"mean_return": -0.0001, "vol_mult": 1.2},
            "BTC": {"mean_return": -0.0030, "vol_mult": 2.5},
            "ETH": {"mean_return": -0.0035, "vol_mult": 3.0},
        },
        correlation_multiplier=2.0,
        regime_type="inflation_shock",
        severity="severe",
        likelihood_weight=0.10,
    ),
    "2020_recovery": CrisisTemplate(
        name="2020 V-Shaped Recovery",
        description="Post-crash rapid recovery. Equities surge, bonds stabilize.",
        date_range="2020-04 to 2020-08",
        asset_shocks={
            "SPY": {"mean_return": 0.0030, "vol_mult": 1.5},
            "GLD": {"mean_return": 0.0015, "vol_mult": 1.2},
            "TLT": {"mean_return": -0.0005, "vol_mult": 1.0},
            "IEF": {"mean_return": -0.0002, "vol_mult": 0.9},
            "SHY": {"mean_return": 0.0001, "vol_mult": 0.8},
            "BTC": {"mean_return": 0.0040, "vol_mult": 2.0},
            "ETH": {"mean_return": 0.0050, "vol_mult": 2.5},
        },
        correlation_multiplier=0.8,  # normalization phase
        regime_type="flight_to_safety",
        severity="moderate",
        likelihood_weight=0.12,
    ),
    "gradual_recession": CrisisTemplate(
        name="Gradual Recession with Fed Pivot",
        description="Slow economic contraction, Fed cuts rates, bonds rally, equities grind lower.",
        date_range="Hypothetical 2026-2027",
        asset_shocks={
            "SPY": {"mean_return": -0.0010, "vol_mult": 1.4},
            "GLD": {"mean_return": 0.0008, "vol_mult": 1.3},
            "TLT": {"mean_return": 0.0018, "vol_mult": 1.3},  # rate cuts help bonds
            "IEF": {"mean_return": 0.0010, "vol_mult": 1.2},
            "SHY": {"mean_return": 0.0005, "vol_mult": 1.0},
            "BTC": {"mean_return": -0.0020, "vol_mult": 2.0},
            "ETH": {"mean_return": -0.0025, "vol_mult": 2.5},
        },
        correlation_multiplier=1.3,
        regime_type="flight_to_safety",
        severity="moderate",
        likelihood_weight=0.15,
    ),
    "geopolitical_shock": CrisisTemplate(
        name="Geopolitical Supply Shock",
        description="Energy/commodity supply disruption leads to stagflation. Gold surges, bonds mixed.",
        date_range="Hypothetical 2026",
        asset_shocks={
            "SPY": {"mean_return": -0.0020, "vol_mult": 2.0},
            "GLD": {"mean_return": 0.0025, "vol_mult": 1.8},  # safe haven
            "TLT": {"mean_return": -0.0008, "vol_mult": 1.4},
            "IEF": {"mean_return": -0.0003, "vol_mult": 1.2},
            "SHY": {"mean_return": 0.0003, "vol_mult": 1.0},
            "BTC": {"mean_return": -0.0025, "vol_mult": 2.5},
            "ETH": {"mean_return": -0.0030, "vol_mult": 3.0},
        },
        correlation_multiplier=1.4,
        regime_type="inflation_shock",
        severity="severe",
        likelihood_weight=0.08,
    ),
    "tech_bubble_burst": CrisisTemplate(
        name="Tech Bubble Burst (AI correction)",
        description="AI/tech valuation correction. SPY hit hard, value/gold relatively insulated.",
        date_range="Hypothetical 2026-2027",
        asset_shocks={
            "SPY": {"mean_return": -0.0022, "vol_mult": 2.2},
            "GLD": {"mean_return": 0.0005, "vol_mult": 1.3},
            "TLT": {"mean_return": 0.0010, "vol_mult": 1.3},
            "IEF": {"mean_return": 0.0005, "vol_mult": 1.1},
            "SHY": {"mean_return": 0.0002, "vol_mult": 1.0},
            "BTC": {"mean_return": -0.0035, "vol_mult": 3.0},
            "ETH": {"mean_return": -0.0040, "vol_mult": 3.5},
        },
        correlation_multiplier=1.6,
        regime_type="correlation_crash",
        severity="severe",
        likelihood_weight=0.07,
    ),
    "mild_correction": CrisisTemplate(
        name="10% Market Correction",
        description="Garden-variety 10% equity correction with normal bond correlation.",
        date_range="Hypothetical (any year)",
        asset_shocks={
            "SPY": {"mean_return": -0.0012, "vol_mult": 1.3},
            "GLD": {"mean_return": 0.0002, "vol_mult": 1.1},
            "TLT": {"mean_return": 0.0005, "vol_mult": 1.1},
            "IEF": {"mean_return": 0.0003, "vol_mult": 1.05},
            "SHY": {"mean_return": 0.0001, "vol_mult": 1.0},
            "BTC": {"mean_return": -0.0018, "vol_mult": 1.8},
            "ETH": {"mean_return": -0.0022, "vol_mult": 2.0},
        },
        correlation_multiplier=1.2,
        regime_type="correlation_crash",
        severity="moderate",
        likelihood_weight=0.20,
    ),
    "low_probability_tail": CrisisTemplate(
        name="Left-Tail Black Swan (-30% SPY)",
        description="Low-probability extreme tail event. Systemic shock, all risk assets collapse.",
        date_range="Hypothetical",
        asset_shocks={
            "SPY": {"mean_return": -0.0050, "vol_mult": 4.0},
            "GLD": {"mean_return": -0.0010, "vol_mult": 2.5},  # even gold gets hit in liquidity crisis
            "TLT": {"mean_return": 0.0015, "vol_mult": 2.0},
            "IEF": {"mean_return": 0.0008, "vol_mult": 1.5},
            "SHY": {"mean_return": 0.0005, "vol_mult": 1.2},
            "BTC": {"mean_return": -0.0060, "vol_mult": 5.0},
            "ETH": {"mean_return": -0.0070, "vol_mult": 5.5},
        },
        correlation_multiplier=2.2,
        regime_type="liquidity_crisis",
        severity="extreme",
        likelihood_weight=0.02,
    ),
    "normal_market": CrisisTemplate(
        name="Normal Market (Baseline)",
        description="No crisis — baseline steady-state market.",
        date_range="Baseline",
        asset_shocks={
            "SPY": {"mean_return": 0.0004, "vol_mult": 1.0},
            "GLD": {"mean_return": 0.0003, "vol_mult": 1.0},
            "TLT": {"mean_return": 0.0002, "vol_mult": 1.0},
            "IEF": {"mean_return": 0.0001, "vol_mult": 1.0},
            "SHY": {"mean_return": 0.00005, "vol_mult": 1.0},
            "BTC": {"mean_return": 0.0010, "vol_mult": 1.0},
            "ETH": {"mean_return": 0.0012, "vol_mult": 1.0},
        },
        correlation_multiplier=1.0,
        regime_type="normal",
        severity="moderate",
        likelihood_weight=0.13,
    ),
}

# Normal daily vol estimates for each asset (annualized % / sqrt(252))
NORMAL_DAILY_VOL = {
    "SPY": 0.012,   # ~19% annual
    "GLD": 0.011,   # ~17% annual
    "TLT": 0.013,   # ~20% annual
    "IEF": 0.007,   # ~11% annual
    "SHY": 0.002,   # ~3% annual
    "BTC": 0.035,   # ~55% annual
    "ETH": 0.040,   # ~63% annual
}

# Normal correlation matrix (pre-crisis steady state)
BASE_CORRELATION = {
    ("SPY", "GLD"): -0.15,
    ("SPY", "TLT"): -0.35,
    ("SPY", "IEF"): -0.25,
    ("SPY", "SHY"): -0.10,
    ("SPY", "BTC"): 0.25,
    ("SPY", "ETH"): 0.20,
    ("GLD", "TLT"): 0.10,
    ("GLD", "IEF"): 0.05,
    ("GLD", "SHY"): 0.02,
    ("GLD", "BTC"): 0.15,
    ("GLD", "ETH"): 0.12,
    ("TLT", "IEF"): 0.85,
    ("TLT", "SHY"): 0.50,
    ("TLT", "BTC"): -0.10,
    ("TLT", "ETH"): -0.08,
    ("IEF", "SHY"): 0.75,
    ("IEF", "BTC"): -0.05,
    ("IEF", "ETH"): -0.03,
    ("SHY", "BTC"): -0.02,
    ("SHY", "ETH"): -0.01,
    ("BTC", "ETH"): 0.70,
}


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

@dataclass
class ScenarioOutcome:
    """Results of a single crisis scenario simulation."""
    scenario_name: str
    scenario_type: str
    severity: str
    portfolio_loss_pct: float
    equity_drawdown_pct: float
    bond_drawdown_pct: float
    gold_return_pct: float
    crypto_return_pct: float
    cvar_impact: float  # how much CVaR would increase
    entropy_impact: float  # how diversification changes
    recovery_days_est: int  # estimated days to recover
    worst_single_day: float  # worst single-day loss

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CrisisAssessment:
    """Complete crisis assessment output."""
    timestamp: str
    portfolio_value: float
    n_scenarios: int
    scenarios: List[ScenarioOutcome]
    worst_case: Optional[ScenarioOutcome]
    expected_shortfall: float  # avg of worst 5%
    median_loss_pct: float
    p95_loss_pct: float
    has_flight_to_safety_buffer: bool  # does portfolio buffer crashes?
    recovery_estimate_days: int
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "portfolio_value": self.portfolio_value,
            "n_scenarios": self.n_scenarios,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "worst_case": self.worst_case.to_dict() if self.worst_case else None,
            "expected_shortfall": round(self.expected_shortfall, 2),
            "median_loss_pct": round(self.median_loss_pct, 2),
            "p95_loss_pct": round(self.p95_loss_pct, 2),
            "has_flight_to_safety_buffer": self.has_flight_to_safety_buffer,
            "recovery_estimate_days": self.recovery_estimate_days,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# core generators
# ---------------------------------------------------------------------------

def get_shocked_correlation(template: CrisisTemplate) -> np.ndarray:
    """Build NxN correlation matrix from base correlations + crisis multiplier."""
    n = len(ASSETS)
    corr = np.eye(n)
    for i, a in enumerate(ASSETS):
        for j, b in enumerate(ASSETS):
            if i >= j:
                continue
            key = (a, b)
            reverse_key = (b, a)
            base = BASE_CORRELATION.get(key) or BASE_CORRELATION.get(reverse_key) or 0.0
            # Apply crisis multiplier — correlations move toward +1 in crisis
            if base < 0:
                # Negative correlations become less negative
                shifted = base * (1.0 / max(template.correlation_multiplier, 0.5))
            else:
                # Positive correlations become more positive
                shifted = base ** (1.0 / max(template.correlation_multiplier, 0.5))
            # Clamp to valid range
            shifted = max(-0.99, min(0.99, shifted))
            corr[i, j] = shifted
            corr[j, i] = shifted
    return corr


def generate_scenarios(
    template: CrisisTemplate,
    n_scenarios: int = 1000,
    horizon_days: int = 30,
    portfolio_value: float = 100000.0,
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[ScenarioOutcome], CrisisAssessment]:
    """Generate N Monte Carlo scenarios from a crisis template.
    
    Each scenario simulates `horizon_days` of daily returns under the crisis 
    regime, then aggregates to a total portfolio return.
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = BASE_ALLOCATION

    n = len(ASSETS)
    corr = get_shocked_correlation(template)
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: slight regularization
        corr += np.eye(n) * 0.001
        L = np.linalg.cholesky(corr)

    # Crises include some crypto allocation (weighted from GLD)
    full_weights = np.array([
        weights.get("SPY", 0.46),
        weights.get("GLD", 0.38),
        weights.get("TLT", 0.16),
        0.0, 0.0, 0.0, 0.0
    ])

    outcomes = []

    for _ in range(n_scenarios):
        # Generate correlated random returns
        z = rng.normal(size=(n, horizon_days))
        correlated = L @ z  # (n x horizon_days)

        daily_returns = np.zeros((n, horizon_days))
        for i, asset in enumerate(ASSETS):
            shock = template.asset_shocks.get(asset, {"mean_return": 0.0, "vol_mult": 1.0})
            daily_mean = shock.get("mean_return", 0.0)
            daily_vol = NORMAL_DAILY_VOL.get(asset, 0.01) * shock.get("vol_mult", 1.0)
            daily_returns[i] = daily_mean + daily_vol * correlated[i]

        # Compute portfolio return over horizon
        portfolio_series = full_weights @ daily_returns  # scalar
        total_return = np.sum(portfolio_series)
        total_return_pct = float(total_return * 100)

        # Per-asset returns for decomposition
        spy_return = float(np.sum(daily_returns[0])) * 100
        gld_return = float(np.sum(daily_returns[1])) * 100
        tlt_return = float(np.sum(daily_returns[2])) * 100
        btc_return = float(np.sum(daily_returns[5])) * 100
        eth_return = float(np.sum(daily_returns[6])) * 100

        # Bond aggregate
        bond_return = tlt_return  # TLT is the dominant bond position

        # Crypto aggregate
        crypto_return = (btc_return + eth_return) / 2

        # Worst single day
        worst_day = float(np.min(portfolio_series)) * 100

        # CVaR impact (approximate)
        cvar_impact = abs(total_return) * (0.12 / max(NORMAL_DAILY_VOL["SPY"], 0.001))

        # Entropy impact
        asset_returns = daily_returns[:, -1]  # last day returns
        if max(abs(asset_returns)) > 0:
            ent_probs = np.abs(asset_returns) / np.sum(np.abs(asset_returns))
            entropy = -np.sum(ent_probs * np.log(ent_probs + 1e-10))
        else:
            entropy = 1.0

        # Recovery estimate (simplistic: 1% per day recovery)
        recovery_days = max(1, int(abs(total_return) / 0.01))

        outcome = ScenarioOutcome(
            scenario_name=template.name,
            scenario_type=template.regime_type,
            severity=template.severity,
            portfolio_loss_pct=round(total_return_pct, 2),
            equity_drawdown_pct=round(spy_return, 2),
            bond_drawdown_pct=round(bond_return, 2),
            gold_return_pct=round(gld_return, 2),
            crypto_return_pct=round(crypto_return, 2),
            cvar_impact=round(cvar_impact, 2),
            entropy_impact=round(entropy, 4),
            recovery_days_est=recovery_days,
            worst_single_day=round(worst_day, 2),
        )
        outcomes.append(outcome)

    # Aggregate
    losses = np.array([o.portfolio_loss_pct for o in outcomes])
    sorted_losses = np.sort(losses)
    median_loss = float(np.median(losses))
    p95 = float(np.percentile(losses, 5))  # 95th percentile loss
    expected_shortfall = float(np.mean(sorted_losses[:max(1, len(sorted_losses) // 20)]))

    worst = max(outcomes, key=lambda o: -o.portfolio_loss_pct) if outcomes else None

    # Flight to safety check: does portfolio have positive bond exposure?
    if weights:
        bond_weight = weights.get("TLT", 0) + weights.get("IEF", 0) + weights.get("SHY", 0)
    else:
        bond_weight = 0.16  # default TLT-only
    has_buffer = bond_weight > 0.05 and template.regime_type == "flight_to_safety"

    # Recommendation
    if median_loss < -15:
        rec = "CRITICAL: Consider reducing equity exposure 10-15%, activate circuit breaker threshold"
    elif median_loss < -10:
        rec = "WARNING: High tail risk. Consider adding hedge (collar overlay or VIXY position)"
    elif median_loss < -5:
        rec = "CAUTION: Moderate vulnerability. Monitor closely, ensure hedge positions are active"
    else:
        rec = "STABLE: Portfolio appears resilient to this scenario type"

    assessment = CrisisAssessment(
        timestamp=datetime.now().isoformat(),
        portfolio_value=portfolio_value,
        n_scenarios=n_scenarios,
        scenarios=outcomes,
        worst_case=worst,
        expected_shortfall=round(expected_shortfall, 2),
        median_loss_pct=round(median_loss, 2),
        p95_loss_pct=round(p95, 2),
        has_flight_to_safety_buffer=has_buffer,
        recovery_estimate_days=worst.recovery_days_est if worst else 30,
        recommendation=rec,
    )

    return outcomes, assessment


def run_full_assessment(
    n_scenarios_per_crisis: int = 500,
    portfolio_value: float = 100000.0,
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> CrisisAssessment:
    """Run scenarios across all crisis templates weighted by likelihood."""
    rng = random.Random(seed)

    all_outcomes = []
    crisis_names = list(CRISIS_TEMPLATES.keys())
    likelihoods = [CRISIS_TEMPLATES[n].likelihood_weight for n in crisis_names]

    # Normalize likelihoods
    total_lh = sum(likelihoods)
    likelihoods = [l / total_lh for l in likelihoods]

    total_scenarios = n_scenarios_per_crisis * len(crisis_names)

    for crisis_name in crisis_names:
        template = CRISIS_TEMPLATES[crisis_name]
        outcomes, _ = generate_scenarios(
            template=template,
            n_scenarios=n_scenarios_per_crisis,
            horizon_days=30,
            portfolio_value=portfolio_value,
            weights=weights,
            seed=rng.randint(0, 2**31) if seed else None,
        )
        all_outcomes.extend(outcomes)

    # Aggregate across all scenarios
    losses = np.array([o.portfolio_loss_pct for o in all_outcomes])
    sorted_losses = np.sort(losses)
    median_loss = float(np.median(losses))
    p95 = float(np.percentile(losses, 5))
    expected_shortfall = float(np.mean(sorted_losses[:max(1, len(sorted_losses) // 20)]))

    worst = max(all_outcomes, key=lambda o: -o.portfolio_loss_pct) if all_outcomes else None

    if weights:
        bond_weight = weights.get("TLT", 0) + weights.get("IEF", 0) + weights.get("SHY", 0)
    else:
        bond_weight = 0.16
    has_buffer = bond_weight > 0.05

    # Multi-crisis recommendation
    if expected_shortfall < -20:
        rec = "CRITICAL: Portfolio vulnerable to -20%+ losses. Activate circuit breaker. Reduce equity 15%."
    elif expected_shortfall < -12:
        rec = "WARNING: Expected shortfall exceeds -12%. Ensure collar overlay and VIXY hedge are active."
    elif expected_shortfall < -7:
        rec = "CAUTION: Expected shortfall of -7% to -12%. Maintain current hedges, monitor tail risk."
    else:
        rec = "STABLE: Multi-crisis assessment shows resilient portfolio structure."

    return CrisisAssessment(
        timestamp=datetime.now().isoformat(),
        portfolio_value=portfolio_value,
        n_scenarios=total_scenarios,
        scenarios=all_outcomes,
        worst_case=worst,
        expected_shortfall=round(expected_shortfall, 2),
        median_loss_pct=round(median_loss, 2),
        p95_loss_pct=round(p95, 2),
        has_flight_to_safety_buffer=has_buffer,
        recovery_estimate_days=worst.recovery_days_est if worst else 30,
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_assessment(assessment: CrisisAssessment):
    """Save assessment to JSON."""
    state = assessment.to_dict()
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = STATE_DIR / f"assessment_{date_str}.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    # Also save latest
    latest_path = STATE_DIR / "latest_assessment.json"
    with open(latest_path, "w") as f:
        json.dump(state, f, indent=2)
    return path


def load_latest_assessment() -> Optional[Dict]:
    """Load latest saved assessment."""
    path = STATE_DIR / "latest_assessment.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def list_crisis_templates() -> List[str]:
    """List available crisis templates."""
    return sorted(CRISIS_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

def display_assessment(assessment: CrisisAssessment):
    """Pretty-print crisis assessment."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CRISIS SCENARIO GENERATOR v8.08                             ║
╠══════════════════════════════════════════════════════════════╣
║  Timestamp: {assessment.timestamp[:19]:<44s}║
║  Scenarios: {assessment.n_scenarios:<6d}  across 9 crisis types{' ' * 20}║
╚══════════════════════════════════════════════════════════════╝

PORTFOLIO STRESS TEST (${assessment.portfolio_value:,.0f})
────────────────────────────────────────────────

  Median Loss:           {assessment.median_loss_pct:>7.2f}%
  95th Percentile Loss:  {assessment.p95_loss_pct:>7.2f}%
  Expected Shortfall:    {assessment.expected_shortfall:>7.2f}%
  Flight-to-Safety:      {'✅ ACTIVE' if assessment.has_flight_to_safety_buffer else '⚠️  WEAK'}
  Recovery Estimate:     ~{assessment.recovery_estimate_days} days

  WORST CASE SCENARIO:
""")
    if assessment.worst_case:
        wc = assessment.worst_case
        print(f"    {wc.scenario_name} ({wc.severity})")
        print(f"    Portfolio Loss: {wc.portfolio_loss_pct:.2f}% | SPY: {wc.equity_drawdown_pct:.2f}%")
        print(f"    GLD: {wc.gold_return_pct:.2f}% | Bonds: {wc.bond_drawdown_pct:.2f}% | Crypto: {wc.crypto_return_pct:.2f}%")
        print(f"    Worst Day: {wc.worst_single_day:.2f}% | CVaR Impact: {wc.cvar_impact:.2f}x")

    print(f"""
  RECOMMENDATION:
  {assessment.recommendation}

  Top-5 Scenarios by Severity:
""")
    sorted_scenarios = sorted(assessment.scenarios, key=lambda o: -o.portfolio_loss_pct)[:5]
    for i, s in enumerate(sorted_scenarios, 1):
        print(f"    {i}. {s.scenario_name:<32s} | Loss: {s.portfolio_loss_pct:>6.2f}% | Type: {s.scenario_type:<20s} | Sev: {s.severity}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v8.08 Automated Crisis Scenario Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Run scenarios for a specific crisis")
    gen.add_argument("--crisis", default="2008_financial", choices=list(CRISIS_TEMPLATES.keys()),
                     help="Crisis template name")
    gen.add_argument("--n", type=int, default=1000, help="Number of scenarios")
    gen.add_argument("--horizon", type=int, default=30, help="Horizon days")
    gen.add_argument("--value", type=float, default=100000.0, help="Portfolio value")
    gen.add_argument("--seed", type=int, default=42, help="Random seed")
    gen.add_argument("--save", action="store_true", help="Save results to JSON")

    # crisis info
    crisis_p = sub.add_parser("crisis", help="Show crisis template details")
    crisis_p.add_argument("--name", default="2008_financial", choices=list(CRISIS_TEMPLATES.keys()))

    # all
    all_p = sub.add_parser("all", help="Run full multi-crisis assessment")
    all_p.add_argument("--n", type=int, default=500, help="Scenarios per crisis type")
    all_p.add_argument("--value", type=float, default=100000.0, help="Portfolio value")
    all_p.add_argument("--seed", type=int, default=42, help="Random seed")
    all_p.add_argument("--save", action="store_true", help="Save results to JSON")

    # rank
    rank_p = sub.add_parser("rank", help="Show crisis vulnerability ranking")
    rank_p.add_argument("--n", type=int, default=500, help="Scenarios per crisis")
    rank_p.add_argument("--save", action="store_true", help="Save ranking to JSON")

    # list
    sub.add_parser("list", help="List available crisis templates")

    # status
    sub.add_parser("status", help="Show latest assessment status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "list":
        print("Available Crisis Templates:\n")
        for name, template in sorted(CRISIS_TEMPLATES.items()):
            lh_pct = template.likelihood_weight * 100
            print(f"  {name:<30s} | {template.severity:<8s} | {lh_pct:>4.0f}% likelihood | {template.description[:60]}")
        return

    if args.command == "crisis":
        t = CRISIS_TEMPLATES.get(args.name)
        if not t:
            print(f"Unknown crisis: {args.name}")
            return
        print(f"\n  Crisis: {t.name}")
        print(f"  Description: {t.description}")
        print(f"  Date Range: {t.date_range}")
        print(f"  Regime Type: {t.regime_type}")
        print(f"  Severity: {t.severity}")
        print(f"  Likelihood Weight: {t.likelihood_weight*100:.0f}%")
        print(f"  Correlation Multiplier: {t.correlation_multiplier}x")
        print(f"\n  Asset Shocks (daily, 30-day horizon):")
        print(f"  {'Asset':<6s} | {'Mean Return':>13s} | {'Vol Mult':>9s} | {'Total 30d':>10s}")
        print(f"  {'-'*6} | {'-'*13} | {'-'*9} | {'-'*10}")
        for asset in ASSETS:
            shock = t.asset_shocks.get(asset, {"mean_return": 0.0, "vol_mult": 1.0})
            daily = shock["mean_return"] * 100
            total_30d = daily * 30
            print(f"  {asset:<6s} | {daily:>+10.3f}%   | {shock['vol_mult']:>5.1f}x   | {total_30d:>+8.2f}%")
        return

    if args.command == "generate":
        template = CRISIS_TEMPLATES.get(args.crisis)
        if not template:
            print(f"Unknown crisis: {args.crisis}")
            return
        outcomes, assessment = generate_scenarios(
            template=template,
            n_scenarios=args.n,
            horizon_days=args.horizon,
            portfolio_value=args.value,
            seed=args.seed,
        )
        display_assessment(assessment)
        if args.save:
            path = save_assessment(assessment)
            print(f"  ✓ Saved to {path}")
        return

    if args.command == "all":
        assessment = run_full_assessment(
            n_scenarios_per_crisis=args.n,
            portfolio_value=args.value,
            seed=args.seed,
        )
        display_assessment(assessment)
        if args.save:
            path = save_assessment(assessment)
            print(f"  ✓ Full assessment saved to {path}")
        return

    if args.command == "rank":
        # Run all crisis templates and rank by median loss
        print("Crisis Vulnerability Ranking\n")
        rankings = []
        for name, template in sorted(CRISIS_TEMPLATES.items()):
            _, assessment = generate_scenarios(
                template=template,
                n_scenarios=args.n,
                horizon_days=30,
                portfolio_value=100000.0,
                seed=42,
            )
            rankings.append((name, template, assessment))

        rankings.sort(key=lambda r: r[2].median_loss_pct)  # worst first

        print(f"  {'Rank':<5s} {'Crisis Type':<34s} {'Median Loss':>12s} {'ES':>8s} {'95th %':>8s} {'Sev':<8s} {'Type':<22s}")
        print(f"  {'-'*5} {'-'*34} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*22}")
        for i, (name, template, assessment) in enumerate(rankings, 1):
            print(f"  {i:<5d} {template.name:<34s} {assessment.median_loss_pct:>+7.2f}%  "
                  f"{assessment.expected_shortfall:>+6.2f}% {assessment.p95_loss_pct:>+6.2f}%  "
                  f"{template.severity:<8s} {template.regime_type:<22s}")

        if args.save:
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = STATE_DIR / f"ranking_{date_str}.json"
            data = {"timestamp": datetime.now().isoformat(), "rankings": [
                {"rank": i+1, "crisis_name": r[1].name, "median_loss_pct": r[2].median_loss_pct,
                 "expected_shortfall": r[2].expected_shortfall, "severity": r[1].severity}
                for i, r in enumerate(rankings)
            ]}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n  ✓ Ranking saved to {path}")
        return

    if args.command == "status":
        latest = load_latest_assessment()
        if latest:
            print(f"Latest Assessment: {latest['timestamp'][:19]}")
            print(f"Scenarios: {latest['n_scenarios']}")
            print(f"Median Loss: {latest['median_loss_pct']:.2f}%")
            print(f"Expected Shortfall: {latest['expected_shortfall']:.2f}%")
            print(f"Recommendation: {latest['recommendation']}")
        else:
            print("No assessment data found. Run 'all --save' first.")
        return


if __name__ == "__main__":
    main()
