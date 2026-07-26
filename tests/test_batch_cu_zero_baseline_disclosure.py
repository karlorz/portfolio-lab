"""Batch CU: zero-baseline ensemble arms disclose as zero_baseline not missing."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_multi_speed_zero_baseline_not_missing() -> None:
    """multi_speed_momentum has configured weight 0.0 all regimes — not missing."""
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "cross_asset_rv", "weight": 0.5},
            {"source": "multi_timeframe_fusion", "weight": 0.5},
        ],
    )
    by = {r["source"]: r for r in statuses}
    msm = by.get("multi_speed_momentum")
    assert msm is not None
    assert msm["configured_weight"] == 0.0
    assert msm["status"] == "zero_baseline"
    assert msm["collected"] is False
    assert msm["contributing"] is False
    assert "baseline weight is 0" in msm["reason"].lower() or "0" in msm["reason"]
    # Must not look like a fetch failure
    assert msm["status"] != "missing"


def test_positive_weight_uncollected_still_missing() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        # Only partial breakdown — international has positive weight, not collected
        source_breakdown=[{"source": "cross_asset_rv", "weight": 1.0}],
    )
    by = {r["source"]: r for r in statuses}
    intl = by.get("international_momentum")
    if intl is not None and float(intl.get("configured_weight") or 0) > 0:
        assert intl["status"] in {"missing", "stale", "unavailable"}
        assert intl["status"] != "zero_baseline"


def test_inactive_count_includes_zero_baseline() -> None:
    configured_status = [
        {"source": "cross_asset_rv", "status": "active", "contributing": True},
        {
            "source": "multi_speed_momentum",
            "status": "zero_baseline",
            "contributing": False,
        },
    ]
    counts = DashboardGenerator._build_ensemble_source_count_metadata(
        regime="normal",
        source_breakdown=[{"source": "cross_asset_rv", "weight": 1.0}],
        configured_source_status=configured_status,
    )
    assert "multi_speed_momentum" in counts["inactive_sources"]
    assert counts["inactive_source_count"] >= 1
