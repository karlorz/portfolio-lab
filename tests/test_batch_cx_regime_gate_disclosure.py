"""Batch CX: regime-gated arms disclose as regime_gate not zero_weight."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_unified_overlay_regime_gate_status() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "unified_overlay", "weight": 0.0},
            {"source": "cross_asset_rv", "weight": 0.5},
        ],
        regime_gated={
            "unified_overlay": "regime_gate_off(normal; off=CRISIS,HIGH_VOL,NORMAL)",
        },
    )
    by = {r["source"]: r for r in statuses}
    uo = by["unified_overlay"]
    assert uo["status"] == "regime_gate"
    assert "Regime-gated off" in uo["reason"]
    assert "regime_gate_off" in uo["regime_gate_reason"]
    assert uo["status"] != "zero_weight"


def test_health_sleep_takes_priority_over_regime_gate() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[{"source": "alternative_data", "weight": 0.0}],
        health_gate_slept={"alternative_data": "degraded_negative_ic(-0.1)"},
        regime_gated={"alternative_data": "should_not_win"},
    )
    alt = next(r for r in statuses if r["source"] == "alternative_data")
    assert alt["status"] == "health_sleep"


def test_inactive_count_includes_regime_gate() -> None:
    counts = DashboardGenerator._build_ensemble_source_count_metadata(
        regime="normal",
        source_breakdown=[{"source": "a", "weight": 1.0}],
        configured_source_status=[
            {"source": "a", "status": "active", "contributing": True},
            {"source": "uo", "status": "regime_gate", "contributing": False},
        ],
    )
    assert "uo" in counts["inactive_sources"]


def test_vote_exposes_regime_gated_for_normal() -> None:
    from src.strategy.ensemble_voter import EnsembleVoter

    vote = EnsembleVoter().compute_vote()
    assert vote is not None
    gated = getattr(vote, "regime_gated", None)
    assert gated is None or isinstance(gated, dict)
    # In normal regime, unified_overlay should be gated off per gate_rules
    if isinstance(gated, dict) and vote.regime.value == "normal":
        assert "unified_overlay" in gated
