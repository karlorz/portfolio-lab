"""Batch BQ: per-arm bandit rewards from attribution (credit assignment)."""

from __future__ import annotations

import json
from pathlib import Path

from src.strategy.ensemble_voter import EnsembleVoter


def _write_attr(tmp_path: Path, sources: dict) -> Path:
    attr_dir = tmp_path / "attribution"
    attr_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": "2026-07-21T00:00:00+00:00",
        "analysis_days": 60,
        "sources": sources,
        "live_authoritative": False,
    }
    path = attr_dir / "latest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_attribution_source_rewards_maps_bps_to_decimal(tmp_path):
    _write_attr(
        tmp_path,
        {
            "multi_speed_momentum": {"avg_return_bps": 10.0},
            "cross_asset_rv": {"avg_return_bps": -5.0},
            "unified_overlay": {"avg_return_bps": 0.0},
        },
    )
    rewards = EnsembleVoter.load_attribution_source_rewards(tmp_path)
    assert rewards is not None
    assert abs(rewards["multi_speed_momentum"] - 0.001) < 1e-12
    assert abs(rewards["cross_asset_rv"] - (-0.0005)) < 1e-12
    assert abs(rewards["unified_overlay"] - 0.0) < 1e-12


def test_load_attribution_rejects_identical_zero_spread(tmp_path):
    _write_attr(
        tmp_path,
        {
            "a": {"avg_return_bps": 1.0},
            "b": {"avg_return_bps": 1.0},
        },
    )
    assert EnsembleVoter.load_attribution_source_rewards(tmp_path) is None


def test_multi_arm_attribution_rewards_update_and_persist(tmp_path):
    """Differentiated per-arm rewards bypass BO identical-broadcast skip.

    Batch DL: soft-delete MSM is excluded from training (updates=2 not 3).
    """
    voter = EnsembleVoter(data_path=tmp_path)
    src_rewards = {
        "multi_speed_momentum": 0.001,  # soft-delete — skipped by default
        "cross_asset_rv": -0.0005,
        "unified_overlay": 0.0002,
    }
    summary = voter.apply_daily_bandit_rewards(
        0.01,  # portfolio scalar still logged; not broadcast
        regime_name="NORMAL",
        sources=list(src_rewards.keys()),
        source_rewards=src_rewards,
        persist=True,
    )
    assert summary["skipped"] is False
    assert summary["reward_mode"] == "attribution_source_rewards"
    assert summary["updates"] == 2  # MSM soft-delete excluded (Batch DL)
    assert "multi_speed_momentum" in (summary.get("soft_delete_excluded") or [])
    assert "multi_speed_momentum" not in (summary.get("arms_updated") or [])
    assert summary["bandit_days"] == 1
    assert summary["live_authoritative"] is False
    assert summary["reward_spread"] > 0
    assert voter.bandit_observations == 2
    # Persist reloads
    voter2 = EnsembleVoter(data_path=tmp_path)
    assert voter2.bandit_observations == 2
    assert voter2.bandit_days == 1


def test_bo_identical_skip_still_holds_without_source_rewards(tmp_path):
    voter = EnsembleVoter(data_path=tmp_path)
    summary = voter.apply_daily_bandit_rewards(0.01, regime_name="NORMAL", persist=True)
    assert summary["skipped"] is True
    assert summary["reason"] == "identical_portfolio_reward_all_arms"
    assert voter.bandit_observations == 0


def test_identical_attribution_map_skipped(tmp_path):
    voter = EnsembleVoter(data_path=tmp_path)
    same = {"a": 0.001, "b": 0.001}
    summary = voter.apply_daily_bandit_rewards(
        0.01,
        sources=list(same.keys()),
        source_rewards=same,
        persist=True,
    )
    assert summary["skipped"] is True
    assert summary["reason"] == "identical_attribution_rewards_all_arms"
    assert voter.bandit_observations == 0


def test_attribution_rewards_noise_floor_filters_arms(tmp_path):
    voter = EnsembleVoter(data_path=tmp_path)
    src_rewards = {
        "multi_speed_momentum": 1e-9,  # below default floor 1e-6
        "cross_asset_rv": 0.001,
    }
    summary = voter.apply_daily_bandit_rewards(
        0.0,
        sources=list(src_rewards.keys()),
        source_rewards=src_rewards,
        persist=True,
    )
    assert summary["skipped"] is False
    assert summary["updates"] == 1
    assert summary["arms_updated"] == ["cross_asset_rv"]
