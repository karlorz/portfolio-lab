"""Map signal keys to producer jobs for recovery under sustained unavailability."""

from __future__ import annotations

from typing import Any

# signal_key -> {job, make_target, module, intentional_when_ml_off}
SIGNAL_OWNERSHIP: dict[str, dict[str, Any]] = {
    "ensemble_voting": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "recovery": "make dashboard  # or make ops-regen",
    },
    "alternative_data": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals.alternative_data_signal",
        "recovery": "uv run python -m src.signals.alternative_data_signal",
    },
    "behavioral_sentiment": {
        "job": "portfolio-lab-research",
        "make_target": "research",
        "module": "src.signals.behavioral_sentiment",
        "intentional_when_ml_off": True,
        "recovery": "PORTFOLIO_LAB_ENABLE_ML=1 make research  # or accept advisory-only",
    },
    "garch_cvar": {
        "job": "portfolio-lab-garch-risk",
        "make_target": "garch-risk",
        "module": "src.monitor.garch_cvar",
        "recovery": "make garch-risk",
    },
    "smart_rebalance": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "recovery": "make dashboard",
    },
    "calendar_seasonality": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals.calendar_seasonality",
        "recovery": "make overlay-signals",
    },
    "crypto_allocation": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals.crypto_momentum",
        "recovery": "make overlay-signals",
    },
    "factor_rotation": {
        # Produced in dashboard generator via FactorMomentumEngine (not overlay job).
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.strategy.factor_rotation",
        "recovery": "make dashboard",
    },
    "stacking_ensemble": {
        "job": "portfolio-lab-research",
        "make_target": "research",
        "module": "src.signals.stacking_integrator",
        "intentional_when_ml_off": True,
        "recovery": "PORTFOLIO_LAB_ENABLE_ML=1 make research  # ML-gated",
    },
    "convexity_harvest": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals",
        "recovery": "make overlay-signals",
    },
    "llm_sentiment": {
        "job": "portfolio-lab-research",
        "make_target": "research",
        "module": "src.llm",
        "recovery": "make research",
    },
    "sector_rotation": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals",
        "recovery": "make overlay-signals",
    },
    "kurtosis_regime": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.regime.kurtosis_regime",
        "recovery": "make overlay-signals",
    },
    "volatility_parity": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals",
        "recovery": "make overlay-signals",
    },
    "collar": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals.collar_signal",
        "recovery": "make overlay-signals",
    },
    "bond_momentum": {
        "job": "portfolio-lab-overlay-signals",
        "make_target": "overlay-signals",
        "module": "src.signals.bond_duration_signal",
        "recovery": "make overlay-signals",
    },
    "risk_decomposition": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "recovery": "make dashboard",
    },
    "rebalance_health": {
        "job": "portfolio-lab-rebalance-health",
        "make_target": "rebalance-health",
        "module": "src.monitor",
        "recovery": "make rebalance-health",
    },
    "two_stage_regime": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        # Depends on FRED-MD series; lab without key cannot produce this.
        "intentional_when_fred_unconfigured": True,
        "recovery": "Set FRED_API_KEY then make data && make dashboard",
    },
    "bocd_regime": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "recovery": "make dashboard",
    },
    "regime_transition": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "intentional_when_fred_unconfigured": True,
        "recovery": "Set FRED_API_KEY then make data && make dashboard",
    },
    "hedge_selector": {
        "job": "portfolio-lab-dashboard",
        "make_target": "dashboard",
        "module": "src.dashboard.generator",
        "recovery": "make dashboard",
    },
    "fred_macro": {
        "job": "portfolio-lab-data",
        "make_target": "data",
        "module": "src.signals",
        # Lab mode without FRED_API_KEY is expected; do not block all-fresh PASS.
        "intentional_when_fred_unconfigured": True,
        "recovery": "Set FRED_API_KEY then make data && make dashboard",
    },
}


def annotate_unavailable_signals(
    unavailable: list[str] | None,
    *,
    ml_enabled: bool = False,
    fred_configured: bool | None = None,
) -> list[dict[str, Any]]:
    """Return ownership rows for unavailable signal keys."""
    if fred_configured is None:
        import os

        fred_configured = bool(os.environ.get("FRED_API_KEY", "").strip())
    rows: list[dict[str, Any]] = []
    for name in unavailable or []:
        key = str(name)
        owner = SIGNAL_OWNERSHIP.get(key, {})
        intentional_ml = bool(owner.get("intentional_when_ml_off")) and not ml_enabled
        intentional_fred = (
            bool(owner.get("intentional_when_fred_unconfigured")) and not fred_configured
        )
        intentional = intentional_ml or intentional_fred
        rows.append(
            {
                "signal": key,
                "job": owner.get("job") or "unknown",
                "make_target": owner.get("make_target") or "unknown",
                "module": owner.get("module"),
                "recovery": owner.get("recovery")
                or "make ops-regen  # inspect producer then dashboard",
                "intentional_when_ml_off": intentional_ml,
                "intentional_when_fred_unconfigured": intentional_fred,
                "intentional_lab_gap": intentional,
            }
        )
    return rows


def recovery_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate jobs to re-run for recovery (excluding intentional lab gaps)."""
    actionable = [
        r
        for r in rows
        if not (r.get("intentional_lab_gap") or r.get("intentional_when_ml_off"))
    ]
    jobs = sorted({str(r.get("job")) for r in actionable if r.get("job") != "unknown"})
    targets = sorted(
        {str(r.get("make_target")) for r in actionable if r.get("make_target") != "unknown"}
    )
    intentional_count = len(rows) - len(actionable)
    return {
        "actionable_unavailable_count": len(actionable),
        "intentional_ml_off_count": sum(
            1 for r in rows if r.get("intentional_when_ml_off")
        ),
        "intentional_lab_gap_count": intentional_count,
        "jobs_to_rerun": jobs,
        "make_targets": targets,
        "suggested_commands": [
            f"make {t}" if t != "unknown" else "make ops-regen" for t in targets
        ]
        or ["make overlay-signals", "make ops-regen"],
        "note": (
            "Do not auto-clear kill_switch; re-run producers then make ops-regen, "
            "then re-evaluate after signals are producer-fresh."
        ),
    }
