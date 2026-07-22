"""Batch BO residual honesty: ops SH re-fold + bandit identical-reward skip."""

from __future__ import annotations

from src.monitor.health_check import apply_ops_monitor_to_dashboard_health
from src.strategy.ensemble_voter import EnsembleVoter


def test_ops_merge_refolds_signal_health_after_kill_clear_demote(tmp_path):
    """Sticky kill→healthy demote must not leave 0/N SH as system healthy."""
    health = {
        "system_status": "warning",  # was elevated by kill
        "kill_switch": {"enabled": True, "level": "warning"},
        "open_incidents": {"open_count": 1, "incidents": [{"id": "x"}]},
        "signal_health": {
            "status": "degraded",
            "overall_health": "degraded",
            "summary": {
                "healthy": 0,
                "degraded": 7,
                "unhealthy": 2,
                "total_tracked": 9,
            },
        },
    }
    # Empty ops report with clear kill on disk
    (tmp_path / "kill_switch.json").write_text(
        '{"enabled": false, "level": null}', encoding="utf-8"
    )
    (tmp_path / "incidents.json").write_text(
        '{"open_count": 0, "incidents": []}', encoding="utf-8"
    )
    ops = {
        "status": "ok",
        "timestamp": "2026-07-21T06:00:00+00:00",
        "checks": {},
    }
    out = apply_ops_monitor_to_dashboard_health(
        health, ops, data_dir=tmp_path, public_dir=tmp_path
    )
    # Kill cleared → would set healthy without SH fold; with fold → degraded
    assert out["system_status"] == "degraded"
    assert out["kill_switch"].get("enabled") is False


def test_bandit_skips_identical_portfolio_reward_broadcast(tmp_path):
    voter = EnsembleVoter(data_path=tmp_path)
    summary = voter.apply_daily_bandit_rewards(0.01, regime_name="NORMAL", persist=True)
    assert summary["skipped"] is True
    assert summary["reason"] == "identical_portfolio_reward_all_arms"
    assert voter.bandit_observations == 0

    # Single-arm still updates (use voting arm; soft-delete MSM skipped by Batch DL)
    one = voter.apply_daily_bandit_rewards(
        0.01,
        regime_name="NORMAL",
        sources=["cross_asset_rv"],
        persist=True,
    )
    assert one["skipped"] is False
    assert one["updates"] == 1
    assert voter.bandit_observations == 1
