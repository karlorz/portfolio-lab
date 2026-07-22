"""Batch DD: zero_baseline soft-delete shadow re-enable disclosure."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_msm_shadow_health_pass_but_not_portfolio_ready() -> None:
    """Live shape: multi_speed healthy + multi-horizon IC ok; still soft-deleted."""
    metrics = {
        "status": "healthy",
        "health_score": 0.5538,
        "ic": 0.0262,
        "ic_30d": 0.0371,
        "ic_60d": 0.0262,
        "ic_90d": 0.0262,
        "reentry_eligible": True,
        "reentry": {
            "reentry_eligible": True,
            "reentry_eps": 0.02,
            "reentry_blocked_reason": None,
            "horizons": {"ic_30d": 0.0371, "ic_60d": 0.0262, "ic_90d": 0.0262},
        },
    }
    shadow = DashboardGenerator._zero_baseline_shadow_checklist(
        "multi_speed_momentum", metrics
    )
    assert shadow["health_gates_pass"] is True
    assert shadow["portfolio_gates_pass"] is False
    assert shadow["shadow_reenable_ready"] is False
    assert shadow["policy"] == "soft_delete_shadow_no_auto_reenable"
    assert "net_negative" in shadow["soft_delete_reason"] or "soft-delete" in shadow[
        "soft_delete_reason"
    ].lower()
    # Batch DH: ADR evidence may replace generic hint with net-negative ΔSharpe
    hint = shadow["shadow_hint"].lower()
    assert (
        "do not auto-reenable" in hint
        or "soft-delete" in hint
        or "net-negative" in hint
        or "adr" in hint
    )


def test_shadow_blocked_when_ic_negative() -> None:
    metrics = {
        "status": "healthy",
        "health_score": 0.6,
        "ic": -0.05,
        "ic_30d": -0.05,
        "ic_60d": -0.05,
        "ic_90d": -0.05,
    }
    shadow = DashboardGenerator._zero_baseline_shadow_checklist("multi_speed_momentum", metrics)
    assert shadow["health_gates_pass"] is False
    assert shadow["reentry_eligible"] is False
    assert shadow["shadow_reenable_ready"] is False


def test_zero_baseline_row_attaches_shadow() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[],  # collector skips zero-weight arms
        health_gate_slept={},
        health_metrics={
            "multi_speed_momentum": {
                "status": "healthy",
                "health_score": 0.5538,
                "ic": 0.0262,
                "ic_30d": 0.0371,
                "ic_60d": 0.0262,
                "ic_90d": 0.0262,
                "reentry": {
                    "reentry_eligible": True,
                    "reentry_eps": 0.02,
                    "reentry_blocked_reason": None,
                    "horizons": {
                        "ic_30d": 0.0371,
                        "ic_60d": 0.0262,
                        "ic_90d": 0.0262,
                    },
                },
                "reentry_eligible": True,
            }
        },
    )
    msm = next(r for r in statuses if r["source"] == "multi_speed_momentum")
    assert msm["status"] == "zero_baseline"
    assert "shadow" in msm
    assert msm["shadow_reenable_ready"] is False
    assert msm.get("health_gates_pass") is True
    assert "shadow:" in msm["reason"]
    assert "Soft-delete" in msm["reason"] or "soft-delete" in msm["reason"].lower()
