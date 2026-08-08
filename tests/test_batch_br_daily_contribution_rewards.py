"""Batch BR (B1): daily per-source contribution rewards for bandit credit assignment.

Prefer single-day signal × portfolio return credit over windowed attribution
avg_return_bps. Differentiated per-arm rewards remain advisory.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.strategy.ensemble_voter import EnsembleVoter


def test_contribution_reward_decimal_directional_and_neutral():
    # Directional: |value| > 0.05 → ret * |value|
    assert abs(
        EnsembleVoter.contribution_reward_decimal(0.01, value=0.5, weight=0.1) - 0.005
    ) < 1e-12
    # Neutral: ret * weight * 2
    assert abs(
        EnsembleVoter.contribution_reward_decimal(-0.002, value=0.01, weight=0.2)
        - (-0.0008)
    ) < 1e-12


def test_compute_daily_contribution_rewards_identifying():
    signals = [
        {"source": "multi_speed_momentum", "value": 0.4, "weight": 0.2},
        {"source": "cross_asset_rv", "value": -0.3, "weight": 0.15},
        {"source": "unified_overlay", "value": 0.0, "weight": 0.25},  # neutral path
    ]
    rewards = EnsembleVoter.compute_daily_contribution_rewards(
        signals, daily_return=0.01
    )
    assert rewards is not None
    assert abs(rewards["multi_speed_momentum"] - 0.004) < 1e-12
    assert abs(rewards["cross_asset_rv"] - 0.003) < 1e-12  # abs(value)
    assert abs(rewards["unified_overlay"] - 0.005) < 1e-12  # 0.01 * 0.25 * 2
    assert max(rewards.values()) - min(rewards.values()) > 0


def test_compute_daily_contribution_rejects_zero_spread():
    signals = [
        {"source": "a", "value": 0.5, "weight": 0.1},
        {"source": "b", "value": 0.5, "weight": 0.1},
    ]
    assert (
        EnsembleVoter.compute_daily_contribution_rewards(signals, daily_return=0.01)
        is None
    )


def test_load_daily_contribution_from_ensemble_db(tmp_path: Path):
    """Hermetic: seed ensemble_signals.db + daily_pnl for one identifying day.

    Dates are relative to today so the fixture never drifts outside the
    loader's recency window (cutoff = now - lookback_days).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    db = tmp_path / "ensemble_signals.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE source_readings (
            timestamp TEXT, source TEXT, value REAL, confidence REAL,
            weight REAL, regime_fit REAL, explanation TEXT
        )
        """
    )
    rows = [
        (f"{today}T16:00:00", "multi_speed_momentum", 0.4, 0.8, 0.2, 1.0, ""),
        (f"{today}T16:00:00", "cross_asset_rv", -0.2, 0.7, 0.15, 1.0, ""),
        (f"{today}T16:00:00", "unified_overlay", 0.1, 0.6, 0.25, 1.0, ""),
        # older day should not win
        (f"{yesterday}T16:00:00", "multi_speed_momentum", 0.9, 0.8, 0.2, 1.0, ""),
        (f"{yesterday}T16:00:00", "cross_asset_rv", 0.9, 0.7, 0.15, 1.0, ""),
    ]
    conn.executemany(
        "INSERT INTO source_readings VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    pnl = tmp_path / "daily_pnl_latest.json"
    pnl.write_text(
        json.dumps({"date": today, "daily_return": 0.01}),
        encoding="utf-8",
    )
    # Also jsonl so PerformanceAttribution paper returns path works
    (tmp_path / "daily_pnl.jsonl").write_text(
        json.dumps({"date": today, "daily_return": 0.01}) + "\n",
        encoding="utf-8",
    )

    out = EnsembleVoter.load_daily_contribution_source_rewards(tmp_path)
    assert out is not None
    rewards, meta = out
    assert meta["reward_mode"] == "daily_contribution_source_rewards"
    assert meta["as_of_date"] == today
    assert meta["live_authoritative"] is False
    assert len(rewards) >= 2
    assert rewards["multi_speed_momentum"] > rewards["cross_asset_rv"]


def test_load_preferred_source_rewards_prefers_daily_over_windowed(tmp_path: Path):
    """Daily contribution wins over windowed attribution when both exist."""
    # Windowed attribution (would map to different rewards)
    attr = tmp_path / "attribution"
    attr.mkdir()
    (attr / "latest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "multi_speed_momentum": {"avg_return_bps": 100.0},
                    "cross_asset_rv": {"avg_return_bps": -50.0},
                }
            }
        ),
        encoding="utf-8",
    )

    db = tmp_path / "ensemble_signals.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE source_readings (
            timestamp TEXT, source TEXT, value REAL, confidence REAL,
            weight REAL, regime_fit REAL, explanation TEXT
        )
        """
    )
    today = datetime.now().strftime("%Y-%m-%d")
    conn.executemany(
        "INSERT INTO source_readings VALUES (?,?,?,?,?,?,?)",
        [
            (f"{today}T12:00:00", "multi_speed_momentum", 0.5, 0.8, 0.2, 1.0, ""),
            (f"{today}T12:00:00", "cross_asset_rv", 0.1, 0.7, 0.15, 1.0, ""),
        ],
    )
    conn.commit()
    conn.close()
    (tmp_path / "daily_pnl.jsonl").write_text(
        json.dumps({"date": today, "daily_return": -0.002}) + "\n",
        encoding="utf-8",
    )

    rewards, mode = EnsembleVoter.load_preferred_source_rewards(tmp_path)
    assert mode == "daily_contribution_source_rewards"
    assert rewards is not None
    # daily: ret * |value| → negative portfolio return * positive values
    assert rewards["multi_speed_momentum"] < 0
    # Windowed would have been +0.01 and -0.005 — ensure we did not use that
    assert abs(rewards["multi_speed_momentum"] - 0.01) > 1e-6


def test_apply_daily_reward_mode_tag(tmp_path: Path):
    voter = EnsembleVoter(data_path=tmp_path)
    src = {"a": 0.001, "b": -0.0005}
    summary = voter.apply_daily_bandit_rewards(
        -0.002,
        sources=list(src.keys()),
        source_rewards=src,
        reward_mode="daily_contribution_source_rewards",
        persist=True,
    )
    assert summary["skipped"] is False
    assert summary["reward_mode"] == "daily_contribution_source_rewards"
    assert summary["live_authoritative"] is False
    assert summary["updates"] == 2


def test_preferred_falls_back_to_windowed_attribution(tmp_path: Path):
    attr = tmp_path / "attribution"
    attr.mkdir()
    (attr / "latest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "multi_speed_momentum": {"avg_return_bps": 10.0},
                    "cross_asset_rv": {"avg_return_bps": -5.0},
                }
            }
        ),
        encoding="utf-8",
    )
    # No ensemble DB → daily path fails
    rewards, mode = EnsembleVoter.load_preferred_source_rewards(tmp_path)
    assert mode == "attribution_source_rewards"
    assert abs(rewards["multi_speed_momentum"] - 0.001) < 1e-12
