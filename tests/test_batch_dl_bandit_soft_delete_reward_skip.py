"""Batch DL: skip bandit reward updates for static soft-delete arms."""

from __future__ import annotations

from pathlib import Path

from src.strategy.ensemble_voter import EnsembleVoter, SignalSource


def test_soft_delete_arm_skipped_by_default(tmp_path: Path) -> None:
    voter = EnsembleVoter(data_path=tmp_path)
    # Explicit single-arm MSM must still be filtered (soft-delete = non-voting)
    summary = voter.apply_daily_bandit_rewards(
        0.01,
        regime_name="NORMAL",
        sources=["multi_speed_momentum"],
        persist=True,
    )
    assert summary["skipped"] is True
    assert summary["reason"] == "all_arms_soft_delete_or_empty"
    assert "multi_speed_momentum" in (summary.get("soft_delete_excluded") or [])
    assert voter.bandit_observations == 0


def test_soft_delete_filtered_from_attribution_rewards(tmp_path: Path) -> None:
    voter = EnsembleVoter(data_path=tmp_path)
    summary = voter.apply_daily_bandit_rewards(
        0.0,
        regime_name="NORMAL",
        source_rewards={
            "multi_speed_momentum": 0.002,
            "cross_asset_rv": 0.001,
            "alternative_data": -0.0005,
        },
        persist=True,
    )
    assert summary["skipped"] is False
    assert summary["updates"] == 2  # MSM excluded
    assert "multi_speed_momentum" not in (summary.get("arms_updated") or [])
    assert set(summary.get("arms_updated") or []) == {
        "cross_asset_rv",
        "alternative_data",
    }
    assert "multi_speed_momentum" in (summary.get("soft_delete_excluded") or [])
    assert voter.bandit_observations == 2


def test_include_soft_delete_arms_opt_in(tmp_path: Path) -> None:
    """Shadow-learning escape hatch: explicit opt-in still updates MSM."""
    voter = EnsembleVoter(data_path=tmp_path)
    summary = voter.apply_daily_bandit_rewards(
        0.01,
        regime_name="NORMAL",
        sources=["multi_speed_momentum"],
        include_soft_delete_arms=True,
        persist=True,
    )
    assert summary["skipped"] is False
    assert summary["updates"] == 1
    assert voter.bandit_observations == 1


def test_static_zero_helpers_list_msm() -> None:
    zeros = EnsembleVoter._static_zero_baseline_sources("NORMAL")
    assert SignalSource.MULTI_SPEED_MOM in zeros
    names = EnsembleVoter._soft_delete_source_names("NORMAL")
    assert "multi_speed_momentum" in names
