"""Batch CW: health-gate sleep reasons on vote + configured_source_status."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_configured_status_health_sleep_from_map() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "alternative_data", "weight": 0.0},
            {"source": "cross_asset_rv", "weight": 0.5},
            {"source": "vix_term_structure", "weight": 0.0},
        ],
        health_gate_slept={
            "alternative_data": "degraded_negative_ic(-0.071)",
            "vix_term_structure": "unhealthy",
        },
    )
    by = {r["source"]: r for r in statuses}
    alt = by["alternative_data"]
    assert alt["status"] == "health_sleep"
    assert "degraded_negative_ic" in alt["reason"]
    assert alt["health_sleep_reason"] == "degraded_negative_ic(-0.071)"
    assert alt["collected"] is True
    assert alt["contributing"] is False

    vix = by["vix_term_structure"]
    assert vix["status"] == "health_sleep"
    assert vix["health_sleep_reason"] == "unhealthy"

    carv = by["cross_asset_rv"]
    assert carv["status"] == "active"


def test_inactive_count_includes_health_sleep() -> None:
    configured = [
        {"source": "a", "status": "active", "contributing": True},
        {"source": "b", "status": "health_sleep", "contributing": False},
    ]
    counts = DashboardGenerator._build_ensemble_source_count_metadata(
        regime="normal",
        source_breakdown=[{"source": "a", "weight": 1.0}],
        configured_source_status=configured,
    )
    assert "b" in counts["inactive_sources"]


def test_vote_carries_sleep_map_after_health_gate() -> None:
    """Smoke: compute_vote attaches health_gate_slept when gate runs."""
    from src.strategy.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter()
    vote = voter.compute_vote()
    assert vote is not None
    # Field present (may be empty if no hard sleeps in this environment)
    slept = getattr(vote, "health_gate_slept", None)
    assert slept is None or isinstance(slept, dict)
    assert hasattr(vote, "health_gate_freeze")
