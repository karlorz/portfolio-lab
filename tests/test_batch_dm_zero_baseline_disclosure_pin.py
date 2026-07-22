"""Batch DM: soft-delete (cfg_w=0) never discloses as active/contributing."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_msm_leaked_vote_weight_discloses_zero_baseline_not_active() -> None:
    """Even if source_breakdown has positive weight (pre-DK leak), disclosure pins."""
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {
                "source": "multi_speed_momentum",
                "weight": 0.053,  # leaked vote mass
                "is_active": True,
                "value": -0.33,
                "confidence": 0.67,
            },
            {"source": "cross_asset_rv", "weight": 0.947},
        ],
        health_gate_slept={},
    )
    msm = next(r for r in statuses if r["source"] == "multi_speed_momentum")
    assert msm["configured_weight"] == 0.0
    assert msm["status"] == "zero_baseline"
    assert msm["contributing"] is False
    assert msm["active"] is False
    assert float(msm["effective_weight"]) == 0.0
    assert float(msm.get("active_weight") or 0.0) == 0.0
    assert msm["collected"] is True  # Batch DJ: still collected for provenance
    assert "shadow" in msm
    assert msm.get("shadow_reenable_ready") is False
    assert "soft-delete" in msm["reason"].lower() or "baseline" in msm["reason"].lower()


def test_active_weights_rollup_excludes_soft_delete_leak() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "multi_speed_momentum", "weight": 0.1},
            {"source": "cross_asset_rv", "weight": 0.9},
        ],
    )
    rollup = DashboardGenerator._ensemble_active_weights_rollup(statuses)
    assert "multi_speed_momentum" not in rollup["active_weights"] or rollup[
        "active_weights"
    ].get("multi_speed_momentum", 0) == 0
    # Only contributing positive-cfg arms
    for name, aw in (rollup.get("active_weights") or {}).items():
        if name == "multi_speed_momentum":
            assert aw == 0
        else:
            assert aw > 0
    assert abs(rollup["active_weights_sum"] - 1.0) < 0.02 or rollup[
        "active_weights_sum"
    ] == 0


def test_zero_baseline_uncollected_still_zero_baseline() -> None:
    """CU regression: uncollected soft-delete remains zero_baseline."""
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[{"source": "cross_asset_rv", "weight": 1.0}],
    )
    msm = next(r for r in statuses if r["source"] == "multi_speed_momentum")
    assert msm["status"] == "zero_baseline"
    assert msm["collected"] is False
