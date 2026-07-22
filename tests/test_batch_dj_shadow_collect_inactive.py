"""Batch DJ: collect zero-weight arms for provenance; inactive_signal shadow."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator
from src.signals.signal_source import SignalSource
from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime
from src.strategy.signal_aggregator import SignalAggregator


def test_active_sources_includes_zero_weight_msm() -> None:
    """Soft-delete MSM (weight 0) must still be in collector roster."""
    agg = SignalAggregator(load_price_data=lambda: None, regime_weights=REGIME_WEIGHTS)
    active = agg.active_sources_for(Regime.NORMAL)
    assert active is not None
    assert SignalSource.MULTI_SPEED_MOM in active
    # should_skip false for zero-weight configured arm
    assert agg.should_skip(SignalSource.MULTI_SPEED_MOM, active, Regime.NORMAL) is False


def test_should_skip_unknown_source() -> None:
    agg = SignalAggregator(load_price_data=lambda: None, regime_weights=REGIME_WEIGHTS)
    active = agg.active_sources_for(Regime.NORMAL)
    # empty active set would skip everything — use subset without MSM
    subset = {s for s in (active or set()) if s != SignalSource.MULTI_SPEED_MOM}
    assert agg.should_skip(SignalSource.MULTI_SPEED_MOM, subset, Regime.NORMAL) is True


def test_inactive_intl_shadow_health_pass_activation_hold() -> None:
    """Live shape: intl multi-horizon IC ok but RS neutral — dual hold."""
    metrics = {
        "status": "degraded",
        "health_score": 0.52,
        "ic": 0.17,
        "ic_30d": 0.14,
        "ic_60d": 0.17,
        "ic_90d": 0.17,
        "reentry": {
            "reentry_eligible": True,
            "reentry_eps": 0.02,
            "reentry_blocked_reason": None,
            "horizons": {"ic_30d": 0.14, "ic_60d": 0.17, "ic_90d": 0.17},
        },
        "reentry_eligible": True,
    }
    act = DashboardGenerator._international_activation_disclosure(
        explanation=(
            "Intl Momentum: neutral, conf=low, EFA/SPY=-3.01pp, "
            "EEM/SPY=-9.75pp, VIX_filter=False"
        ),
        value=0.0,
        confidence=0.0,
    )
    shadow = DashboardGenerator._inactive_signal_shadow_checklist(
        "international_momentum", metrics, act
    )
    assert shadow["health_gates_pass"] is True
    assert shadow["activation_cleared"] is False
    assert shadow["force_activate"] is False
    assert shadow["policy"] == "inactive_signal_shadow_no_force_activate"
    assert any("efa" in g or "neutral" in g for g in shadow["activation_gaps"])
    assert "threshold" in shadow["shadow_hint"].lower() or "inactive" in shadow[
        "shadow_hint"
    ].lower()


def test_inactive_row_attaches_shadow() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {
                "source": "international_momentum",
                "weight": 0.0,
                "is_active": False,
                "value": 0.0,
                "confidence": 0.0,
                "inactive_explanation": (
                    "Intl Momentum: neutral, conf=low, EFA/SPY=-3.01pp, "
                    "EEM/SPY=-9.75pp, VIX_filter=False"
                ),
            }
        ],
        health_gate_slept={},
        health_metrics={
            "international_momentum": {
                "status": "degraded",
                "health_score": 0.52,
                "ic": 0.17,
                "ic_30d": 0.14,
                "ic_60d": 0.17,
                "ic_90d": 0.17,
                "reentry": {
                    "reentry_eligible": True,
                    "reentry_eps": 0.02,
                    "reentry_blocked_reason": None,
                    "horizons": {
                        "ic_30d": 0.14,
                        "ic_60d": 0.17,
                        "ic_90d": 0.17,
                    },
                },
                "reentry_eligible": True,
            }
        },
    )
    intl = next(r for r in statuses if r["source"] == "international_momentum")
    assert intl["status"] == "inactive_signal"
    assert "shadow" in intl
    assert intl.get("health_gates_pass") is True
    assert intl.get("force_activate") is False
    assert "shadow:" in intl["reason"]
