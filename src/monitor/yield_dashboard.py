"""
Portfolio-Lab v7.05: DeFi/CeFi Yield Comparison Dashboard

Compares portfolio yields across asset classes:
- Staking yields (ETH ~3-5%) vs bond yields (TLT/IEF/SHY) vs money market
- Real yield comparison (after inflation)
- Yield curve of portfolio opportunities
- Integration: feeds into bond duration rotation decision

Usage:
    python -m src.monitor.yield_dashboard status    # Current yield dashboard
    python -m src.monitor.yield_dashboard compare   # Yield comparison table
    python -m src.monitor.yield_dashboard summary   # One-line summary for EnsembleVoter
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from src.paths import DATA_DIR
STATE_FILE = DATA_DIR / "yield_dashboard_state.json"


# --- Dataclasses ---

@dataclass
class YieldSource:
    """Single yield observation from one source."""
    name: str                         # e.g., "ETH Staking", "TLT", "IEF", "SHY", "Money Market"
    asset_type: str                   # "staking", "bond", "money_market"
    yield_nominal: float              # Nominal yield (decimal, e.g. 0.045 = 4.5%)
    yield_real: float                 # Real yield after inflation (decimal)
    duration_years: Optional[float] = None  # Bond duration, or None for non-bonds
    source: str = "estimated"         # "estimated", "live", "fallback"
    confidence: float = 0.7           # 0-1
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class YieldComparison:
    """Comparison of yields across portfolio opportunities."""
    timestamp: str
    risk_free_rate: float             # 10Y treasury proxy (decimal)
    cpi_rate: float                   # Latest CPI / inflation (decimal)
    sources: List[YieldSource]
    best_nominal: Tuple[str, float]   # (name, yield) — highest nominal
    best_real: Tuple[str, float]      # (name, yield) — highest real
    staking_premium_bps: float        # ETH staking - risk_free_rate (bps)
    bond_premium_bps: float           # Best bond - risk_free_rate (bps)
    recommendation: str               # Where to allocate yield-seeking capital

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "risk_free_rate": self.risk_free_rate,
            "cpi_rate": self.cpi_rate,
            "sources": [asdict(s) for s in self.sources],
            "best_nominal": {"name": self.best_nominal[0], "yield": self.best_nominal[1]},
            "best_real": {"name": self.best_real[0], "yield": self.best_real[1]},
            "staking_premium_bps": self.staking_premium_bps,
            "bond_premium_bps": self.bond_premium_bps,
            "recommendation": self.recommendation,
        }


@dataclass
class YieldDashboardState:
    """Persistent state for yield dashboard."""
    last_comparison: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)
    max_history: int = 30


# --- Main Dashboard Class ---

class YieldDashboard:
    """
    DeFi/CeFi Yield Comparison Dashboard.

    Aggregates yield data from:
    - crypto_staking module (ETH staking yields)
    - bond_duration_signal module (TLT/IEF/SHY yields)
    - Fed overlay (risk-free rate)
    - External CPI estimate

    Provides:
    - Nominal and real yield comparison
    - Staking premium over risk-free rate
    - Yield-seeking capital recommendation
    - Integration signal for EnsembleVoter (3% weight)
    """

    # Default yields for fallback when modules unavailable
    DEFAULT_BOND_YIELDS = {
        "TLT": {"yield": 0.044, "duration": 16.0},    # Long duration
        "IEF": {"yield": 0.042, "duration": 7.0},     # Intermediate
        "SHY": {"yield": 0.043, "duration": 2.0},     # Short duration
    }

    # Money market / cash proxy
    DEFAULT_MM_YIELD = 0.045  # ~4.5% money market

    # Inflation / CPI estimate
    DEFAULT_CPI_RATE = 0.030  # ~3.0%

    # Risk-free rate proxy (10Y yield)
    DEFAULT_RISK_FREE_RATE = 0.043  # ~4.3%

    # Thresholds
    STAKING_PREMIUM_THRESHOLD_BPS = 100  # 100bps premium = attractive
    SIGNAL_WEIGHT = 0.03  # 3% in EnsembleVoter

    def __init__(self):
        self._state = self._load_state()
        self._risk_free_rate = self.DEFAULT_RISK_FREE_RATE
        self._cpi_rate = self.DEFAULT_CPI_RATE

        # Try to import optional modules
        self._staking_model = None
        self._bond_signal_module = None
        self._fed_overlay_module = None

    def _load_state(self) -> Dict:
        """Load persistent state."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load yield dashboard state: {e}")
        return {"last_comparison": None, "history": [], "max_history": 30}

    def _save_state(self):
        """Persist current state."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)

    def _get_fed_rate(self) -> Optional[float]:
        """Try to get Fed funds rate from fed_overlay module."""
        try:
            from src.signals.fed_policy_overlay import FedPolicyOverlay
            if self._fed_overlay_module is None:
                self._fed_overlay_module = FedPolicyOverlay()
            regime = self._fed_overlay_module.detect_regime()
            if regime is not None:
                return regime.fed_funds_rate / 100.0  # Fed returns as percentage
        except (ImportError, AttributeError, Exception):
            pass
        return None

    def _get_staking_yield(self) -> Optional[float]:
        """Try to get ETH staking yield from crypto_staking module."""
        try:
            from src.strategy.crypto_staking import ETHStakingModel
            if self._staking_model is None:
                self._staking_model = ETHStakingModel()
            metrics = self._staking_model.estimate_yield()
            return metrics.annual_yield
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Staking module unavailable: {e}")
            return None

    def _get_bond_yields(self) -> Dict[str, Dict]:
        """Try to get bond yields from bond_duration_signal module.

        Returns dict of {name: {"yield": float, "duration": float}}.
        """
        try:
            from src.signals.bond_duration_signal import BondDurationSignalGenerator
            gen = BondDurationSignalGenerator()
            signal = gen.generate_signal()
            return {
                "TLT": {"yield": signal.yield_10y / 100.0, "duration": 16.0},
                "IEF": {"yield": max(signal.yield_10y / 100.0 * 0.95, 0.03), "duration": 7.0},
                "SHY": {"yield": signal.yield_2y / 100.0, "duration": 2.0},
            }
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Bond duration signal unavailable: {e}")
            return dict(self.DEFAULT_BOND_YIELDS)

    def gather_yields(self) -> YieldComparison:
        """
        Gather yields from all available sources and produce comparison.

        Returns:
            YieldComparison with all sources and recommendations
        """
        now = datetime.now()
        cpi = self._cpi_rate

        # Try to get Fed rate for more accurate risk-free rate
        fed_rate = self._get_fed_rate()
        if fed_rate is not None:
            rfr = fed_rate  # Already in decimal form
        else:
            rfr = self.DEFAULT_RISK_FREE_RATE

        self._risk_free_rate = rfr

        sources = []

        # 1. ETH Staking yield
        staking_yield = self._get_staking_yield()
        if staking_yield is not None:
            sources.append(YieldSource(
                name="ETH Staking",
                asset_type="staking",
                yield_nominal=staking_yield,
                yield_real=staking_yield - cpi,
                duration_years=None,
                source="estimated",
                confidence=0.7,
            ))
        else:
            # Fallback
            sources.append(YieldSource(
                name="ETH Staking",
                asset_type="staking",
                yield_nominal=0.035,
                yield_real=0.035 - cpi,
                duration_years=None,
                source="fallback",
                confidence=0.5,
            ))

        # 2. Bond yields
        bond_yields = self._get_bond_yields()
        for name, data in bond_yields.items():
            y = data["yield"]
            dur = data.get("duration")
            sources.append(YieldSource(
                name=name,
                asset_type="bond",
                yield_nominal=y,
                yield_real=y - cpi,
                duration_years=dur,
                source="estimated",
                confidence=0.8,
            ))

        # 3. Money Market
        sources.append(YieldSource(
            name="Money Market",
            asset_type="money_market",
            yield_nominal=self.DEFAULT_MM_YIELD,
            yield_real=self.DEFAULT_MM_YIELD - cpi,
            duration_years=0.0,
            source="fallback",
            confidence=0.6,
        ))

        # 4. Risk-Free Rate (reference)
        sources.append(YieldSource(
            name="10Y Treasury (RFR)",
            asset_type="bond",
            yield_nominal=rfr,
            yield_real=rfr - cpi,
            duration_years=10.0,
            source="estimated",
            confidence=0.8,
        ))

        # Find best yields
        non_rfr_sources = [s for s in sources if "RFR" not in s.name]
        best_nominal = max(non_rfr_sources, key=lambda s: s.yield_nominal)
        best_real = max(non_rfr_sources, key=lambda s: s.yield_real)

        # Staking yield for premium calculation
        staking_source = next((s for s in sources if s.name == "ETH Staking"), None)
        staking_yield_val = staking_source.yield_nominal if staking_source else 0.035

        staking_premium_bps = (staking_yield_val - rfr) * 10000

        # Best bond premium
        bond_sources = [s for s in non_rfr_sources if s.asset_type == "bond" and "RFR" not in s.name]
        best_bond_yield = max(bond_sources, key=lambda s: s.yield_nominal).yield_nominal if bond_sources else rfr
        bond_premium_bps = (best_bond_yield - rfr) * 10000

        # Generate recommendation
        recommendation = self._generate_recommendation(
            staking_premium_bps, staking_source, best_nominal, sources
        )

        comparison = YieldComparison(
            timestamp=now.isoformat(),
            risk_free_rate=rfr,
            cpi_rate=cpi,
            sources=sources,
            best_nominal=(best_nominal.name, best_nominal.yield_nominal),
            best_real=(best_real.name, best_real.yield_real),
            staking_premium_bps=staking_premium_bps,
            bond_premium_bps=bond_premium_bps,
            recommendation=recommendation,
        )

        # Save to state
        self._state["last_comparison"] = comparison.to_dict()
        history = self._state.get("history", [])
        history.insert(0, comparison.to_dict())
        max_h = self._state.get("max_history", 30)
        self._state["history"] = history[:max_h]
        self._save_state()

        return comparison

    def _generate_recommendation(
        self,
        staking_premium_bps: float,
        staking_source: Optional[YieldSource],
        best_nominal: YieldSource,
        sources: List[YieldSource],
    ) -> str:
        """Generate yield-seeking capital recommendation based on comparison."""
        parts = []

        # Staking analysis
        if staking_premium_bps > self.STAKING_PREMIUM_THRESHOLD_BPS:
            parts.append(f"ETH staking premium +{staking_premium_bps:.0f}bps > threshold "
                         f"({self.STAKING_PREMIUM_THRESHOLD_BPS:.0f}bps) — staking attractive")
        elif staking_premium_bps > 0:
            parts.append(f"ETH staking premium +{staking_premium_bps:.0f}bps — modest carry")
        else:
            parts.append(f"ETH staking at {staking_premium_bps:.0f}bps vs RFR — not compelling")

        # Best yield source
        parts.append(f"Best nominal yield: {best_nominal.name} @ {best_nominal.yield_nominal*100:.1f}%")

        # Bond vs staking comparison
        bond_avg = np.mean([s.yield_nominal for s in sources
                           if s.asset_type == "bond" and "RFR" not in s.name])
        if staking_source and staking_source.yield_nominal > bond_avg:
            parts.append("Staking yields exceed average bond yields — consider increasing crypto allocation")
        else:
            parts.append("Bond yields competitive with staking — maintain fixed-income allocation")

        # Real yield check
        positive_real = [s for s in sources if s.yield_real > 0 and "RFR" not in s.name]
        if len(positive_real) >= len([s for s in sources if "RFR" not in s.name]) // 2:
            parts.append("Most asset classes offer positive real yields — favorable for yield-seekers")
        else:
            parts.append("Negative real yields across most assets — inflation erodes nominal returns")

        return " | ".join(parts)

    def get_ensemble_signal(self) -> float:
        """
        Get signal for EnsembleVoter integration (3% weight).

        Returns:
            float: Signal value -1.0 to +1.0
                   +1.0 = Strong yield opportunity (increase yield-seeking allocation)
                   -1.0 = Poor yield environment (reduce duration/exposure)
                   0.0  = Neutral
        """
        comparison = self.gather_yields()
        signal = 0.0

        # Staking premium signal
        if comparison.staking_premium_bps > self.STAKING_PREMIUM_THRESHOLD_BPS:
            signal += 0.3  # Attractive staking premium
        elif comparison.staking_premium_bps < 0:
            signal -= 0.2  # Staking below RFR — not compelling

        # Bond premium signal
        if comparison.bond_premium_bps > 50:
            signal += 0.2  # Bonds offering meaningful premium over RFR
        elif comparison.bond_premium_bps < -20:
            signal -= 0.1  # Bond yields compressed

        # Real yield environment
        positive_real_count = sum(
            1 for s in comparison.sources
            if s.yield_real > 0 and "RFR" not in s.name
        )
        total_non_rfr = sum(1 for s in comparison.sources if "RFR" not in s.name)
        if total_non_rfr > 0:
            real_yield_ratio = positive_real_count / total_non_rfr
            if real_yield_ratio > 0.6:
                signal += 0.2  # Generally good real yield environment
            elif real_yield_ratio < 0.3:
                signal -= 0.2  # Poor real yield environment

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, signal))

    def display_status(self) -> str:
        """Return formatted status string."""
        comparison = self.gather_yields()

        lines = []
        lines.append("=" * 60)
        lines.append("   v7.05 DeFi/CeFi Yield Comparison Dashboard")
        lines.append(f"   {comparison.timestamp[:19]}")
        lines.append("=" * 60)
        lines.append(f"  Risk-Free Rate:  {comparison.risk_free_rate*100:.2f}%")
        lines.append(f"  CPI / Inflation: {comparison.cpi_rate*100:.2f}%")
        lines.append("-" * 60)
        lines.append(f"  {'Source':<20} {'Nominal':>8} {'Real':>8} {'Dur':>6} {'Conf':>6}")
        lines.append(f"  {'-'*18:20} {'-'*8:8} {'-'*8:8} {'-'*4:6} {'-'*4:6}")
        for s in comparison.sources:
            dur_str = f"{s.duration_years:.0f}y" if s.duration_years is not None else "N/A"
            lines.append(
                f"  {s.name:<20} {s.yield_nominal*100:>7.2f}% {s.yield_real*100:>7.2f}% "
                f"{dur_str:>6} {s.confidence*100:>5.0f}%"
            )
        lines.append("-" * 60)
        lines.append(f"  Best Nominal: {comparison.best_nominal[0]} @ {comparison.best_nominal[1]*100:.2f}%")
        lines.append(f"  Best Real:    {comparison.best_real[0]} @ {comparison.best_real[1]*100:.2f}%")
        lines.append(f"  Staking Premium: {comparison.staking_premium_bps:+.0f} bps vs RFR")
        lines.append(f"  Bond Premium:    {comparison.bond_premium_bps:+.0f} bps vs RFR")
        lines.append("-" * 60)
        lines.append(f"  Ensemble Signal: {self.get_ensemble_signal():+.2f} (weight: {self.SIGNAL_WEIGHT:.0%})")
        lines.append("=" * 60)
        lines.append(f"  RECOMMENDATION:")
        lines.append(f"  {comparison.recommendation}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def display_compare(self) -> str:
        """Return formatted comparison table."""
        comparison = self.gather_yields()

        lines = []
        lines.append("Yield Comparison — " + comparison.timestamp[:19])
        lines.append(f"  RFR={comparison.risk_free_rate*100:.2f}% CPI={comparison.cpi_rate*100:.2f}%")
        lines.append("")
        lines.append(f"  {'Asset':<20} {'Nominal':>8} {'Real':>8} {'Premium(bps)':>14}")
        lines.append(f"  {'-'*18:20} {'-'*8:8} {'-'*8:8} {'-'*12:14}")
        for s in comparison.sources:
            premium_bps = (s.yield_nominal - comparison.risk_free_rate) * 10000
            lines.append(
                f"  {s.name:<20} {s.yield_nominal*100:>7.2f}% {s.yield_real*100:>7.2f}% "
                f"{premium_bps:>+10.0f} bps"
            )
        lines.append("")
        lines.append(f"  Best nominal: {comparison.best_nominal[0]} ({comparison.best_nominal[1]*100:.2f}%)")
        lines.append(f"  Best real:    {comparison.best_real[0]} ({comparison.best_real[1]*100:.2f}%)")
        lines.append(f"  Signal: {self.get_ensemble_signal():+.2f}")

        return "\n".join(lines)


# --- CLI Entry Points ---

def _get_dashboard() -> YieldDashboard:
    return YieldDashboard()


def cmd_status():
    """Print yield dashboard status."""
    db = _get_dashboard()
    print(db.display_status())


def cmd_compare():
    """Print yield comparison table."""
    db = _get_dashboard()
    print(db.display_compare())


def cmd_summary():
    """Print one-line summary for EnsembleVoter integration."""
    db = _get_dashboard()
    signal = db.get_ensemble_signal()
    comparison = db._state.get("last_comparison")
    if comparison:
        print(f"[v7.05] signal={signal:+.2f} | "
              f"best={comparison['best_nominal']['name']} "
              f"@{comparison['best_nominal']['yield']*100:.2f}% | "
              f"staking_p={comparison['staking_premium_bps']:+.0f}bps | "
              f"weight={YieldDashboard.SIGNAL_WEIGHT:.0%}")
    else:
        print(f"[v7.05] signal={signal:+.2f} | no data yet")


def cmd_signal():
    """Print ensemble signal value only (for scripting)."""
    db = _get_dashboard()
    signal = db.get_ensemble_signal()
    print(f"{signal:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        cmd_status()
    else:
        cmd = sys.argv[1]
        if cmd == "status":
            cmd_status()
        elif cmd == "compare":
            cmd_compare()
        elif cmd == "summary":
            cmd_summary()
        elif cmd == "signal":
            cmd_signal()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python -m src.monitor.yield_dashboard [status|compare|summary|signal]")
