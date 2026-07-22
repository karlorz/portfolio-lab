"""Batch DA: multi-horizon IC reentry hysteresis (disclosure; no force-wake)."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator


def test_reentry_blocked_when_any_horizon_negative() -> None:
    # Live shape: alt_data IC30>0 but IC60/90 negative — must NOT reenter
    r = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.0456,
        ic_60d=-0.0709,
        ic_90d=-0.0709,
    )
    assert r["reentry_eligible"] is False
    assert r["horizons_all_positive"] is False
    assert r["reentry_blocked_reason"]
    assert "negative" in r["reentry_blocked_reason"]
    assert r["policy"] == "multi_horizon_hysteresis_no_force_wake"


def test_reentry_blocked_deep_negative_regime_arb() -> None:
    r = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=-0.191,
        ic_60d=-0.299,
        ic_90d=-0.299,
    )
    assert r["reentry_eligible"] is False
    assert "negative_ic_horizon" in (r["reentry_blocked_reason"] or "")


def test_reentry_blocked_below_hysteresis_eps() -> None:
    # All non-negative but below eps=0.02 — hysteresis gap
    r = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.01,
        ic_60d=0.005,
        ic_90d=0.0,
        reentry_eps=0.02,
    )
    assert r["reentry_eligible"] is False
    assert r["horizons_all_positive"] is False  # 0.0 is not > 0
    assert "below_reentry_eps" in (r["reentry_blocked_reason"] or "") or (
        "negative" in (r["reentry_blocked_reason"] or "")
    )


def test_reentry_eligible_all_horizons_above_eps() -> None:
    r = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.05,
        ic_60d=0.04,
        ic_90d=0.03,
        reentry_eps=0.02,
    )
    assert r["reentry_eligible"] is True
    assert r["horizons_all_positive"] is True
    assert r["horizons_all_above_eps"] is True
    assert r["reentry_blocked_reason"] is None


def test_reentry_blocked_missing_horizon() -> None:
    r = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.05,
        ic_60d=None,
        ic_90d=0.04,
    )
    assert r["reentry_eligible"] is False
    assert "insufficient" in (r["reentry_blocked_reason"] or "")


def test_recovery_hint_short_horizon_bounce() -> None:
    reentry = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.05,
        ic_60d=-0.07,
        ic_90d=-0.07,
    )
    hint = DashboardGenerator._health_recovery_hint(
        status="degraded",
        ic=-0.07,
        acc30=0.43,
        acc60=0.63,
        health_score=0.51,
        half_life=14.0,
        reentry=reentry,
    )
    assert "hysteresis" in hint.lower() or "force-wake" in hint.lower()
    assert "short-horizon" in hint.lower() or "multi-horizon" in hint.lower()


def test_recovery_hint_eligible_path() -> None:
    reentry = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.08,
        ic_60d=0.06,
        ic_90d=0.05,
    )
    hint = DashboardGenerator._health_recovery_hint(
        status="degraded",
        ic=0.05,
        acc30=0.55,
        acc60=0.55,
        health_score=0.55,
        half_life=40.0,
        reentry=reentry,
    )
    assert "eligible" in hint.lower()
    assert "force" in hint.lower()


def test_health_sleep_row_attaches_reentry() -> None:
    reentry = DashboardGenerator._evaluate_ic_reentry(
        ic_30d=0.0456,
        ic_60d=-0.0709,
        ic_90d=-0.0709,
    )
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {"source": "alternative_data", "weight": 0.0, "is_active": True},
        ],
        health_gate_slept={"alternative_data": "degraded_negative_ic(-0.071)"},
        health_metrics={
            "alternative_data": {
                "status": "degraded",
                "health_score": 0.5148,
                "ic": -0.0709,
                "ic_30d": 0.0456,
                "ic_60d": -0.0709,
                "ic_90d": -0.0709,
                "accuracy_30d": 0.4353,
                "accuracy_60d": 0.634,
                "ic_half_life_days": 13.7,
                "window_collapse_90_60": True,
                "reentry": reentry,
                "reentry_eligible": False,
                "recovery_hint": "Short-horizon IC bounce only.",
            }
        },
    )
    alt = next(r for r in statuses if r["source"] == "alternative_data")
    assert alt["status"] == "health_sleep"
    assert alt.get("reentry_eligible") is False
    assert "reentry" in alt
    assert alt["reentry"]["reentry_eligible"] is False
