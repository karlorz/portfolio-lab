"""
Unified Overlay Orchestrator - v9.15 Implementation
Combines all tactical overlays into a single portfolio recommendation with
VIX-gated activation, overlay conflict reduction, and explicit thresholds.

v9.15 Changes:
- VIX-gated overlay activation (collar active VIX<30, VIXY active VIX>20)
- Limit concurrent premium-cost overlays to max 2
- Collar/VIXY complement: VIX<20→collar, VIX>30→VIXY
- Crypto: momentum threshold check (>0% 6m)
- Live VIX detection instead of hardcoded defaults

Overlays integrated:
- v4.60 Cashless Collar (equity drawdown protection)
- v7.04 VIXY Hedge Sizing (volatility hedge)
- v3.50 Calendar Seasonality (execution timing)
- v4.70 Crypto Tactical (uncorrelated alpha)
- v4.80 Bond Duration Rotation (fixed-income optimization)

Conflict resolution: additive shift model with hard constraints.
Each overlay proposes allocation deltas; orchestrator validates and combines.

Usage:
    python -m src.strategy.unified_orchestrator recommend
    python -m src.strategy.unified_orchestrator status
    python -m src.strategy.unified_orchestrator backtest
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.paths import DATA_DIR, SIGNALS_DIR, MARKET_DB
from src.signals.calendar_seasonality import get_calendar_modifier
from src.signals.bond_duration_signal import generate_bond_duration_signal
from src.signals.collar_signal import generate_collar_signal
from src.signals.crypto_momentum import generate_crypto_signal

try:
    from src.strategy.vixy_hedge_sizing import VIXYHedgeSizer
    _HAS_VIXY = True
except ImportError:
    _HAS_VIXY = False
    VIXYHedgeSizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverlayStatus(Enum):
    ACTIVE = "active"
    SUPPRESSED = "suppressed"    # Conflict resolution lowered weight
    DISABLED = "disabled"        # Hard constraints disabled overlay


@dataclass
class OverlayContribution:
    """Individual overlay contribution to the unified portfolio."""
    name: str
    version: str
    status: str  # active, suppressed, disabled
    weight: float  # 0.0-1.0 in unified model

    # Allocation deltas (percentage points of total portfolio)
    spy_delta: float
    gld_delta: float
    tlt_delta: float
    ief_delta: float
    shy_delta: float
    btc_delta: float
    eth_delta: float

    # Risk contribution
    vol_impact: float      # Estimated volatility contribution
    sharpe_contribution: float  # Estimated Sharpe contribution

    # Meta
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnifiedRecommendation:
    """Complete unified portfolio recommendation."""
    timestamp: str

    # Baseline (46/38/16)
    baseline_spy: float
    baseline_gld: float
    baseline_tlt: float

    # Recommended weights
    spy: float
    gld: float
    tlt: float
    ief: float
    shy: float
    btc: float
    eth: float

    # Overlay contributions
    contributions: List[OverlayContribution]

    # Risk metrics
    total_spy_delta: float
    total_vol_impact: float
    estimated_sharpe: float

    # Conflicts
    conflict_count: int
    conflicts_resolved: List[str]

    # Calendar timing
    calendar_modifier: float
    execution_recommendation: str

    # Meta
    confidence: float
    recommendation: str
    is_actionable: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contributions"] = [c.to_dict() for c in self.contributions]
        return d


class UnifiedOrchestrator:
    """
    Unified overlay orchestrator.

    Collects signals from all configured overlays, resolves conflicts,
    and produces a single portfolio recommendation with hard constraints.

    Hard Constraints:
    - Total equity (SPY): 36-56% (baseline 46% ± 10pp)
    - Total gold (GLD): 28-48% (baseline 38% ± 10pp)
    - Total bonds (TLT+IEF+SHY): 6-26% (baseline 16% ± 10pp)
    - Total crypto (BTC+ETH): 0-5%
    - All weights sum to 100%
    - No negative weights
    """

    BASELINE = {"spy": 0.46, "gld": 0.38, "tlt": 0.16, "ief": 0.0,
                 "shy": 0.0, "btc": 0.0, "eth": 0.0}

    # Hard bounds
    BOUNDS = {
        "spy": (0.36, 0.56),
        "gld": (0.28, 0.48),
        "tlt": (0.0, 0.20),
        "ief": (0.0, 0.15),
        "shy": (0.0, 0.15),
        "btc": (0.0, 0.03),
        "eth": (0.0, 0.02),
    }

    # v9.15: Overlay weights — vixy added, collar & vixy share hedge budget
    OVERLAY_WEIGHTS = {
        "collar": 0.20,
        "crypto": 0.15,
        "bond_duration": 0.25,
        "calendar": 0.10,
        "vixy": 0.05,
        # Remaining 25% is base passive allocation
    }

    # v9.15: VIX thresholds for overlay activation
    VIX_COLLAR_MAX = 30.0         # Collar disabled above VIX 30 (cost prohibitive)
    VIX_VIXY_MIN = 20.0           # VIXY hedge starts at VIX 20
    VIX_CRISIS = 40.0             # Collar freeze threshold

    STATE_FILE = DATA_DIR / "unified_orchestrator_state.json"

    def __init__(self):
        self._state = self._load_state()
        self._ensure_dirs()

    def _ensure_dirs(self):
        sig_dir = SIGNALS_DIR
        sig_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict:
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state file {self.STATE_FILE}: {e}")
        return {"last_unified": None, "conflict_history": []}

    def _save_state(self):
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)

    def _fetch_vix_level(self) -> float:
        """Fetch VIX level from market DB for overlay gating."""
        db_path = MARKET_DB
        if not db_path.exists():
            return 16.0  # default fallback (normal VIX)
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT close FROM prices WHERE symbol='^VIX' ORDER BY date DESC LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()
            if row and len(row) > 0 and row[0] > 0:
                return float(row[0])
        except Exception as e:
            logger.debug(f"Could not fetch VIX from DB: {e}")
        return 16.0  # fallback

    def collect_overlay_contributions(self) -> List[OverlayContribution]:
        """Collect signals from all active overlays."""
        contributions = []
        vix_level = self._fetch_vix_level()

        # ── Premium-cost overlay conflict reduction (v9.15) ────────
        # Determine which hedges are appropriate based on VIX level
        collar_active = vix_level < self.VIX_COLLAR_MAX
        collar_crisis = vix_level >= self.VIX_CRISIS
        vixy_active = vix_level >= self.VIX_VIXY_MIN

        # If VIX is moderate (20-30), both can be active in reduced capacity
        # If VIX < 20: collar preferred (cheaper), VIXY suppressed
        # If VIX > 30: VIXY preferred, collar stressed/disabled
        # If VIX > 40: collar crisis-disabled, VIXY fully active
        if vix_level < self.VIX_VIXY_MIN:
            vixy_status = "suppressed"
            collar_status = "active"
        elif vix_level >= self.VIX_CRISIS:
            vixy_status = "active"
            collar_status = "disabled"
        elif vix_level >= self.VIX_COLLAR_MAX:
            vixy_status = "active"
            collar_status = "suppressed"  # expensive but not crisis-level
        else:
            # VIX 20-30: both can be active in reduced capacity
            vixy_status = "active"
            collar_status = "active"

        # Count premium-cost active overlays (max 2)
        premium_cost_active = sum(1 for s in [collar_status, vixy_status] if s == "active")
        if premium_cost_active > 2:
            # Safety: should never happen, but if so, suppress VIXY
            vixy_status = "suppressed"

        # 1. Collar Overlay (v4.60)
        try:
            # v9.15: Use live data instead of hardcoded spot/vix
            collar = generate_collar_signal(spot=None, vix=None)
            if collar.is_valid and collar_status != "disabled":
                spy_shift = -(collar.strikes.net_premium / collar.underlying_price) if collar.underlying_price > 0 else 0
                contributions.append(OverlayContribution(
                    name="collar", version="v4.60",
                    status=collar_status,
                    weight=self.OVERLAY_WEIGHTS["collar"],
                    spy_delta=round(spy_shift, 4) if abs(spy_shift) < 0.05 else 0.0,
                    gld_delta=0.01 if collar.confidence > 70 else 0.0,
                    tlt_delta=0.0, ief_delta=0.0, shy_delta=0.0,
                    btc_delta=0.0, eth_delta=0.0,
                    vol_impact=-0.005, sharpe_contribution=0.03,
                    confidence=collar.confidence,
                    reason=f"Collar: {collar.regime} (VIX={vix_level:.1f}), "
                           f"{'cashless' if collar.strikes.is_cashless else 'debit'}",
                ))
        except Exception as e:
            logger.warning(f"Collar overlay unavailable: {e}")

        # 2. VIXY Hedge Overlay (v7.04) — v9.15: integrated into orchestrator
        if _HAS_VIXY and vixy_status != "disabled":
            try:
                assert VIXYHedgeSizer is not None  # _HAS_VIXY guarantees this
                sizer = VIXYHedgeSizer()
                vixy_signal = sizer.get_signal(vix_level=vix_level)
                if vixy_signal and vixy_signal.allocation_pct > 0:
                    alloc_frac = vixy_signal.allocation_pct / 100.0
                    # VIXY hedge is funded from SPY allocation
                    spy_shift_vixy = -alloc_frac
                    contributions.append(OverlayContribution(
                        name="vixy", version="v7.04",
                        status=vixy_status,
                        weight=self.OVERLAY_WEIGHTS["vixy"],
                        spy_delta=round(spy_shift_vixy, 4),
                        gld_delta=0.0, tlt_delta=0.0,
                        ief_delta=0.0, shy_delta=0.0,
                        btc_delta=0.0, eth_delta=0.0,
                        vol_impact=alloc_frac * 0.5,  # VIXY adds vol
                        sharpe_contribution=0.01 if vixy_signal.hedge_efficiency > 1.0 else 0.0,
                        confidence=vixy_signal.confidence * 100,
                        reason=f"VIXY: {vixy_signal.regime} ({vixy_signal.allocation_pct:.1f}%), "
                               f"efficiency {vixy_signal.hedge_efficiency:.2f}x",
                    ))
            except Exception as e:
                logger.warning(f"VIXY overlay unavailable: {e}")

        # 3. Crypto Tactical (v4.70) — v9.15: momentum threshold check
        try:
            crypto = generate_crypto_signal()
            if crypto.is_valid:
                # v9.15: Only activate when 6m momentum > 0%
                btc_mom = getattr(crypto.btc_signal, 'momentum_6m', 0)
                crypto_has_momentum = btc_mom > 0.0 if btc_mom is not None else False
                crypto_status = "active" if (crypto.confidence > 50 and crypto_has_momentum) else "suppressed"

                contributions.append(OverlayContribution(
                    name="crypto", version="v4.70",
                    status=crypto_status,
                    weight=self.OVERLAY_WEIGHTS["crypto"],
                    spy_delta=0.0,
                    gld_delta=round(-crypto.composite_weight, 4),
                    tlt_delta=0.0, ief_delta=0.0, shy_delta=0.0,
                    btc_delta=round(crypto.btc_signal.target_weight * crypto.composite_weight, 4),
                    eth_delta=round(crypto.eth_signal.target_weight * crypto.composite_weight, 4),
                    vol_impact=0.003 if crypto.composite_weight > 0 else 0.0,
                    sharpe_contribution=0.02 if crypto.composite_weight > 0 else 0.0,
                    confidence=crypto.confidence,
                    reason=f"Crypto: {crypto.signal_state}, {crypto.composite_weight:.1%} weight"
                           + (f", 6m mom={btc_mom:+.1%}" if btc_mom is not None else ""),
                ))
        except Exception as e:
            logger.warning(f"Crypto overlay unavailable: {e}")

        # 4. Bond Duration Rotation (v4.80)
        try:
            bond = generate_bond_duration_signal()
            if bond.is_valid:
                base_tlt = self.BASELINE["tlt"]
                contributions.append(OverlayContribution(
                    name="bond_duration", version="v4.80",
                    status="active" if bond.confidence > 50 else "suppressed",
                    weight=self.OVERLAY_WEIGHTS["bond_duration"],
                    spy_delta=0.0, gld_delta=0.0,
                    tlt_delta=round(
                        bond.tlt_weight * base_tlt - base_tlt, 4
                    ),
                    ief_delta=round(bond.ief_weight * base_tlt, 4),
                    shy_delta=round(bond.shy_weight * base_tlt, 4),
                    btc_delta=0.0, eth_delta=0.0,
                    vol_impact=-0.003 if bond.position == "short" else 0.0,
                    sharpe_contribution=0.025,
                    confidence=bond.confidence,
                    reason=f"Bond: {bond.position} ({bond.curve_regime}/{bond.rate_direction})",
                ))
        except Exception as e:
            logger.warning(f"Bond duration overlay unavailable: {e}")

        # 5. Calendar Seasonality (v3.50) — execution timing only
        try:
            mod = get_calendar_modifier()
            contributions.append(OverlayContribution(
                name="calendar", version="v3.50",
                status="active",
                weight=self.OVERLAY_WEIGHTS["calendar"],
                spy_delta=0.0, gld_delta=0.0, tlt_delta=0.0,
                ief_delta=0.0, shy_delta=0.0,
                btc_delta=0.0, eth_delta=0.0,
                vol_impact=0.0,
                sharpe_contribution=0.015 if mod < 0.85 else 0.005,
                confidence=85.0,
                reason=f"Calendar: {mod:.2f}x urgency modifier",
            ))
        except Exception as e:
            logger.warning(f"Calendar overlay unavailable: {e}")

        return contributions

    def resolve_conflicts(self, contributions: List[OverlayContribution]) -> Tuple[
        Dict[str, float], List[str]
    ]:
        """
        Resolve conflicts between overlay contributions.

        Strategy: weighted sum with hard bounds enforcement.
        - Each overlay proposes deltas proportional to its weight
        - Sum weighted deltas for each asset
        - Clip to hard bounds
        - Rebalance to sum to 1.0
        """
        deltas = {k: 0.0 for k in self.BASELINE}
        conflicts = []

        # Weighted sum of contributions
        for contrib in contributions:
            if contrib.status == "disabled":
                continue
            w = contrib.weight if contrib.status == "active" else contrib.weight * 0.5

            for asset in deltas:
                contrib_delta = getattr(contrib, f"{asset}_delta", 0.0)
                deltas[asset] += contrib_delta * w

        # Check for conflicting equity signals
        spy_deltas = [
            getattr(c, "spy_delta", 0) for c in contributions
            if c.status != "disabled"
        ]
        if spy_deltas:
            positive = sum(d for d in spy_deltas if d > 0)
            negative = sum(d for d in spy_deltas if d < 0)
            if positive > 0 and negative < 0:
                conflicts.append(
                    f"SPY conflict: +{positive:.1%} vs {negative:.1%} — "
                    f"net {positive + negative:.1%}"
                )

        # Check for gold vs crypto funding
        gld_deltas = [
            getattr(c, "gld_delta", 0) for c in contributions
            if c.status != "disabled"
        ]
        if sum(d for d in gld_deltas if d < 0) < -0.05:
            conflicts.append(
                f"GLD reduction exceeds 5% ({sum(d for d in gld_deltas if d < 0):.1%})"
            )

        # Apply to baseline
        weights = dict(self.BASELINE)
        for asset in weights:
            weights[asset] += deltas[asset]

        # Enforce hard bounds
        for asset, (lo, hi) in self.BOUNDS.items():
            if weights[asset] < lo:
                conflicts.append(f"{asset.upper()} below floor ({weights[asset]:.1%} < {lo:.0%})")
                weights[asset] = lo
            elif weights[asset] > hi:
                conflicts.append(f"{asset.upper()} above ceiling ({weights[asset]:.1%} > {hi:.0%})")
                weights[asset] = hi

        # Normalize to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            for k in weights:
                weights[k] /= total

        return weights, conflicts

    def recommend(self) -> UnifiedRecommendation:
        """Generate unified portfolio recommendation."""
        contributions = self.collect_overlay_contributions()
        weights, conflicts = self.resolve_conflicts(contributions)

        # Calendar modifier for execution timing
        cal_mod = 1.0
        for c in contributions:
            if c.name == "calendar":
                cal_mod = float(c.reason.split(":")[1].split("x")[0].strip()) \
                    if ":" in c.reason else 1.0
                break

        # Execution recommendation
        if cal_mod < 0.60:
            exec_rec = "wait — strong seasonal headwind"
        elif cal_mod < 0.80:
            exec_rec = "delay — moderate seasonal effect"
        else:
            exec_rec = "proceed — normal conditions"

        # Total SPY delta
        spy_delta = weights["spy"] - self.BASELINE["spy"]

        # Confidence weighted by active overlay count
        active_count = sum(1 for c in contributions if c.status == "active")
        avg_confidence = (
            sum(c.confidence for c in contributions) / len(contributions)
            if contributions else 50.0
        )

        # Estimated Sharpe
        estimated_sharpe = 0.79 + sum(
            c.sharpe_contribution for c in contributions
            if c.status != "disabled"
        )

        # v9.15: Include VIX info in recommendation
        vix_level = self._fetch_vix_level()

        return UnifiedRecommendation(
            timestamp=datetime.now().isoformat(),
            baseline_spy=self.BASELINE["spy"],
            baseline_gld=self.BASELINE["gld"],
            baseline_tlt=self.BASELINE["tlt"],
            spy=round(weights["spy"], 4),
            gld=round(weights["gld"], 4),
            tlt=round(weights["tlt"], 4),
            ief=round(weights["ief"], 4),
            shy=round(weights["shy"], 4),
            btc=round(weights["btc"], 4),
            eth=round(weights["eth"], 4),
            contributions=contributions,
            total_spy_delta=round(spy_delta, 4),
            total_vol_impact=round(
                sum(c.vol_impact for c in contributions if c.status != "disabled"), 4
            ),
            estimated_sharpe=round(estimated_sharpe, 3),
            conflict_count=len(conflicts),
            conflicts_resolved=conflicts,
            calendar_modifier=round(cal_mod, 2),
            execution_recommendation=exec_rec,
            confidence=round(avg_confidence, 1),
            recommendation=(
                f"Unified: {active_count}/{len(contributions)} overlays active, "
                f"VIX={vix_level:.1f}, "
                f"SPY {weights['spy']:.1%}, GLD {weights['gld']:.1%}, "
                f"Bonds {weights['tlt']+weights['ief']+weights['shy']:.1%}, "
                f"Crypto {weights['btc']+weights['eth']:.1%}, "
                f"{len(conflicts)} conflict(s)"
            ),
            is_actionable=len(conflicts) == 0,
        )

    def save_recommendation(self, rec: UnifiedRecommendation):
        out = self.STATE_FILE.parent / "signals" / "unified_recommendation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec.to_dict(), f, indent=2)


def get_unified_recommendation() -> UnifiedRecommendation:
    """Convenience function."""
    orch = UnifiedOrchestrator()
    return orch.recommend()


def main():
    import sys
    orch = UnifiedOrchestrator()
    rec = orch.recommend()

    print("=" * 60)
    print("UNIFIED OVERLAY ORCHESTRATOR v9.15")
    print("=" * 60)
    print(f"Timestamp: {rec.timestamp}")
    print()
    print("Portfolio Allocation:")
    print(f"  Baseline: SPY {rec.baseline_spy:.0%} / "
          f"GLD {rec.baseline_gld:.0%} / TLT {rec.baseline_tlt:.0%}")
    print(f"  Unified:  SPY {rec.spy:.1%} / GLD {rec.gld:.1%} / "
          f"TLT {rec.tlt:.1%} / IEF {rec.ief:.1%} / SHY {rec.shy:.1%} / "
          f"BTC {rec.btc:.1%} / ETH {rec.eth:.1%}")
    print(f"  SPY Delta: {rec.total_spy_delta:+.2%}")
    print()
    print("Active Overlays:")
    for c in rec.contributions:
        if c.status != "disabled":
            flag = "✓" if c.status == "active" else "~"
            print(f"  {flag} {c.name} (v{c.version}): {c.reason}")
    print()
    print(f"Conflicts: {rec.conflict_count}")
    for conflict in rec.conflicts_resolved:
        print(f"  ⚠ {conflict}")
    print()
    print(f"Calendar: {rec.calendar_modifier:.2f}x → {rec.execution_recommendation}")
    print(f"Estimated Sharpe: {rec.estimated_sharpe:.3f}")
    print(f"Confidence: {rec.confidence:.0f}%")
    print(f"Actionable: {rec.is_actionable}")
    print()
    print(f"Recommendation: {rec.recommendation}")
    print("=" * 60)

    if "--save" in sys.argv:
        orch.save_recommendation(rec)


if __name__ == "__main__":
    main()
