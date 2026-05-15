"""
Overlay Dashboard Data Generator - v4.91
Collects signals from all tactical overlays and generates dashboard-ready JSON.

Feeds the frontend with:
- Collar status: strikes, premium, regime, VIX level
- Crypto allocation: BTC/ETH weight, momentum, vol regime
- Bond duration: TLT/IEF/SHY split, curve regime
- Calendar: urgency modifier, active windows, next window
- Kurtosis: regime, KER, strategy routing
- Unified: composite portfolio recommendation

Usage:
    python -m src.dashboard.overlay_dashboard generate
    python -m src.dashboard.overlay_dashboard status
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OverlayDashboardData:
    """Complete overlay dashboard data ready for frontend consumption."""
    timestamp: str
    generated_at: str

    # Collar overlay (v4.60)
    collar: Dict[str, Any]

    # Crypto tactical (v4.70)
    crypto: Dict[str, Any]

    # Bond duration rotation (v4.80)
    bond_duration: Dict[str, Any]

    # Calendar seasonality (v3.50)
    calendar: Dict[str, Any]

    # Kurtosis regime (v4.91)
    kurtosis: Dict[str, Any]

    # Mean reversion (v4.81)
    mean_reversion: Dict[str, Any]

    # Unified orchestrator (v4.90)
    unified: Dict[str, Any]

    # Summary
    active_overlays: int
    total_overlays: int
    portfolio_risk: str  # low, moderate, elevated, high
    alerts: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class OverlayDashboardGenerator:
    """
    Generates dashboard JSON for all tactical overlays.

    Collects live signals from each overlay module, formats them
    for frontend consumption, and saves to a single JSON file.
    """

    OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "dashboard" / "overlay_dashboard.json"

    def __init__(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _get_collar_data(self) -> Dict[str, Any]:
        """Collect collar overlay data."""
        try:
            from src.signals.collar_signal import generate_collar_signal
            signal = generate_collar_signal(spot=550.0, vix=16.0)
            return {
                "active": signal.is_valid,
                "regime": signal.regime,
                "call_strike": signal.call_strike,
                "put_strike": signal.put_strike,
                "net_premium": signal.strikes.net_premium,
                "is_cashless": signal.strikes.is_cashless,
                "max_upside_pct": signal.max_upside_pct,
                "max_downside_pct": signal.max_downside_pct,
                "vix_level": signal.vix_level,
                "confidence": signal.confidence,
                "status_text": f"Collar: {signal.regime}, "
                               f"call ${signal.call_strike:.0f}, "
                               f"put ${signal.put_strike:.0f}",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_crypto_data(self) -> Dict[str, Any]:
        """Collect crypto tactical data."""
        try:
            from src.signals.crypto_momentum import generate_crypto_signal
            signal = generate_crypto_signal()
            return {
                "active": signal.is_valid,
                "btc_weight": signal.btc_signal.target_weight,
                "eth_weight": signal.eth_signal.target_weight,
                "total_crypto": signal.composite_weight,
                "btc_momentum_6m": signal.btc_signal.momentum_6m,
                "eth_momentum_6m": signal.eth_signal.momentum_6m,
                "btc_vol_regime": signal.btc_signal.vol_regime,
                "eth_vol_regime": signal.eth_signal.vol_regime,
                "confidence": signal.confidence,
                "status_text": f"Crypto: {signal.composite_weight:.1%}, "
                               f"BTC {signal.btc_signal.momentum_6m:+.1%} 6m",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_bond_duration_data(self) -> Dict[str, Any]:
        """Collect bond duration rotation data."""
        try:
            from src.signals.bond_duration_signal import generate_bond_duration_signal
            signal = generate_bond_duration_signal()
            return {
                "active": signal.is_valid,
                "yield_10y": signal.yield_10y,
                "yield_2y": signal.yield_2y,
                "spread": signal.spread_10y2y,
                "curve_regime": signal.curve_regime,
                "rate_direction": signal.rate_direction,
                "tlt_weight": signal.tlt_weight,
                "ief_weight": signal.ief_weight,
                "shy_weight": signal.shy_weight,
                "effective_duration": signal.effective_duration,
                "position": signal.position,
                "confidence": signal.confidence,
                "status_text": f"Bonds: {signal.position} "
                               f"({signal.curve_regime}/{signal.rate_direction}), "
                               f"dur {signal.effective_duration:.0f}yr",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_calendar_data(self) -> Dict[str, Any]:
        """Collect calendar seasonality data."""
        try:
            from src.signals.calendar_seasonality import check_calendar
            signal = check_calendar()
            return {
                "active": signal.is_trading_day,
                "modifier": signal.urgency_modifier,
                "active_windows": signal.active_windows,
                "next_window": signal.next_window,
                "days_to_next": signal.days_to_next_window,
                "recommendation": signal.recommendation,
                "effect": signal.effect,
                "status_text": f"Calendar: {signal.urgency_modifier:.2f}x, "
                               f"{len(signal.active_windows)} windows active",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_kurtosis_data(self) -> Dict[str, Any]:
        """Collect kurtosis regime data."""
        try:
            from src.regime.kurtosis_regime import detect_kurtosis_regime
            signal = detect_kurtosis_regime()
            return {
                "active": True,
                "kurtosis_20d": signal.kurtosis_20d,
                "kurtosis_60d": signal.kurtosis_60d,
                "ker_ratio": signal.ker_ratio,
                "regime": signal.regime,
                "transitioning": signal.is_transitioning,
                "strategy_preference": signal.strategy_preference,
                "tsom_weight": signal.tsom_weight,
                "mr_weight": signal.mr_weight,
                "fat_tail_risk": signal.fat_tail_risk,
                "status_text": f"Kurtosis: {signal.regime} "
                               f"(k={signal.kurtosis_60d:.1f}, KER={signal.ker_ratio:.2f})",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_mean_reversion_data(self) -> Dict[str, Any]:
        """Collect mean reversion data."""
        try:
            from src.strategy.mean_reversion_overlay import get_mean_reversion_status
            status = get_mean_reversion_status()
            return {
                "active": status.get("active", False),
                "allocation_pct": status.get("allocation_pct", 0),
                "vix_level": status.get("vix_level", 0),
                "vix_regime": status.get("vix_regime", "N/A"),
                "rationale": status.get("rationale", ""),
                "status_text": f"MR: {status.get('allocation_pct', 0):.1f}% alloc, "
                               f"VIX={status.get('vix_level', 0):.1f}",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _get_unified_data(self) -> Dict[str, Any]:
        """Collect unified orchestrator data."""
        try:
            from src.strategy.unified_orchestrator import get_unified_recommendation
            rec = get_unified_recommendation()
            return {
                "active": True,
                "spy": rec.spy,
                "gld": rec.gld,
                "tlt": rec.tlt,
                "ief": rec.ief,
                "shy": rec.shy,
                "btc": rec.btc,
                "eth": rec.eth,
                "estimated_sharpe": rec.estimated_sharpe,
                "conflict_count": rec.conflict_count,
                "calendar_modifier": rec.calendar_modifier,
                "execution_rec": rec.execution_recommendation,
                "status_text": f"Unified: SPY {rec.spy:.1%}, GLD {rec.gld:.1%}, "
                               f"Sharpe est {rec.estimated_sharpe:.3f}",
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

    def _assess_portfolio_risk(self, data: Dict) -> Tuple[str, List[str]]:
        """Assess overall portfolio risk level and generate alerts."""
        alerts = []
        risk_score = 0

        # Collar risk
        collar = data.get("collar", {})
        if collar.get("vix_level", 0) > 30:
            alerts.append(f"VIX elevated ({collar['vix_level']:.0f}) — collar active")
            risk_score += 2
        elif collar.get("vix_level", 0) > 25:
            risk_score += 1

        # Crypto risk
        crypto = data.get("crypto", {})
        btc_vol = crypto.get("btc_vol_regime", "")
        if btc_vol == "extreme":
            alerts.append("BTC vol extreme — crypto position exited")
            risk_score += 2
        elif btc_vol == "high":
            risk_score += 1

        # Kurtosis risk
        kurt = data.get("kurtosis", {})
        if kurt.get("fat_tail_risk", 0) > 0.7:
            alerts.append(f"Fat tail risk elevated ({kurt['fat_tail_risk']:.1%})")
            risk_score += 2

        # Bond risk
        bond = data.get("bond_duration", {})
        if bond.get("curve_regime") == "inverted":
            alerts.append("Yield curve inverted — defensive bond posture")
            risk_score += 1

        # Unified conflicts
        unified = data.get("unified", {})
        if unified.get("conflict_count", 0) > 0:
            alerts.append(f"{unified['conflict_count']} overlay conflict(s) detected")
            risk_score += unified["conflict_count"]

        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "elevated"
        elif risk_score >= 1:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return risk_level, alerts

    def generate(self) -> OverlayDashboardData:
        """Generate complete dashboard data."""
        data = {
            "collar": self._get_collar_data(),
            "crypto": self._get_crypto_data(),
            "bond_duration": self._get_bond_duration_data(),
            "calendar": self._get_calendar_data(),
            "kurtosis": self._get_kurtosis_data(),
            "mean_reversion": self._get_mean_reversion_data(),
            "unified": self._get_unified_data(),
        }

        risk_level, alerts = self._assess_portfolio_risk(data)

        active = sum(1 for v in data.values() if v.get("active"))
        total = len(data)

        return OverlayDashboardData(
            timestamp=datetime.now().isoformat(),
            generated_at=datetime.now().isoformat(),
            collar=data["collar"],
            crypto=data["crypto"],
            bond_duration=data["bond_duration"],
            calendar=data["calendar"],
            kurtosis=data["kurtosis"],
            mean_reversion=data["mean_reversion"],
            unified=data["unified"],
            active_overlays=active,
            total_overlays=total,
            portfolio_risk=risk_level,
            alerts=alerts,
        )

    def save(self, dashboard: OverlayDashboardData):
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(dashboard.to_dict(), f, indent=2, default=str)
        logger.info(f"Dashboard saved to {self.OUTPUT_PATH}")


def generate_overlay_dashboard() -> OverlayDashboardData:
    """Convenience function."""
    gen = OverlayDashboardGenerator()
    return gen.generate()


def main():
    import sys
    gen = OverlayDashboardGenerator()
    dashboard = gen.generate()

    print("=" * 60)
    print("OVERLAY DASHBOARD v4.91")
    print("=" * 60)
    print(f"Generated: {dashboard.generated_at}")
    print(f"Active: {dashboard.active_overlays}/{dashboard.total_overlays}")
    print(f"Risk Level: {dashboard.portfolio_risk.upper()}")
    print()

    for name, data in [
        ("Collar", dashboard.collar),
        ("Crypto", dashboard.crypto),
        ("Bond Duration", dashboard.bond_duration),
        ("Calendar", dashboard.calendar),
        ("Kurtosis", dashboard.kurtosis),
        ("Mean Reversion", dashboard.mean_reversion),
        ("Unified", dashboard.unified),
    ]:
        status_text = data.get("status_text", data.get("error", "N/A"))
        flag = "✓" if data.get("active") else "✗"
        print(f"  {flag} {name:<18} {status_text}")

    print()
    if dashboard.alerts:
        print("Alerts:")
        for alert in dashboard.alerts:
            print(f"  ⚠ {alert}")
    else:
        print("No alerts — all systems normal")
    print("=" * 60)

    if "--save" in sys.argv:
        gen.save(dashboard)
        print(f"Saved to {gen.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
