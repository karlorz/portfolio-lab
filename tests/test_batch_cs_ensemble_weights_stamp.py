"""Batch CS: ensemble_weights.json freshness stamp + voter skips _meta."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.strategy.adaptive_ensemble_weights import stamp_ensemble_weights_freshness


def test_stamp_refreshes_mtime_and_meta(tmp_path: Path) -> None:
    ew = tmp_path / "ensemble_weights.json"
    payload = {
        "normal": {"cross_asset_rv": 0.5, "multi_timeframe_fusion": 0.5},
        "high_vol": {"cross_asset_rv": 0.6, "multi_timeframe_fusion": 0.4},
    }
    ew.write_text(json.dumps(payload), encoding="utf-8")
    old = time.time() - 46 * 86400
    os.utime(ew, (old, old))
    age_before = (time.time() - ew.stat().st_mtime) / 86400.0
    assert age_before > 40

    result = stamp_ensemble_weights_freshness(ew, reason="test_stamp")
    assert result["ok"] is True
    age_after = (time.time() - ew.stat().st_mtime) / 86400.0
    assert age_after < 1.0

    disk = json.loads(ew.read_text(encoding="utf-8"))
    assert disk["normal"]["cross_asset_rv"] == 0.5
    assert disk["high_vol"]["multi_timeframe_fusion"] == 0.4
    meta = disk["_meta"]
    assert meta["stamp_reason"] == "test_stamp"
    assert meta["content_identity"] == "unchanged"
    assert meta["regime_count"] == 2
    assert "last_freshness_stamp" in meta
    assert disk["artifact_id"] == "ensemble_weights.json"
    assert disk["runtime_provenance"]["artifact_id"] == "ensemble_weights.json"


def test_stamp_missing_file(tmp_path: Path) -> None:
    result = stamp_ensemble_weights_freshness(tmp_path / "nope.json")
    assert result["ok"] is False
    assert result["error"] == "missing"


def test_voter_skips_meta_without_breaking(tmp_path: Path, monkeypatch) -> None:
    from src.strategy import ensemble_voter as ev

    ew = tmp_path / "ensemble_weights.json"
    ew.write_text(
        json.dumps(
            {
                "_meta": {"schema": "ensemble-weights/v1"},
                "normal": {
                    "cross_asset_rv": 0.5,
                    "multi_timeframe_fusion": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENSEMBLE_WEIGHTS_FILE", str(ew))
    # Incomplete regimes fall back / warn but must not treat _meta as regime
    weights = ev._load_regime_weights()
    # Either loaded partial or full fallback — must not KeyError
    assert weights is not None
    # If file partially loaded, normal present
    from src.strategy.ensemble_voter import Regime

    if Regime.NORMAL in weights:
        assert len(weights[Regime.NORMAL]) >= 1


def test_quality_disclosure_not_stale_after_stamp(tmp_path: Path) -> None:
    from src.dashboard.signal_health_section import attach_signal_quality_disclosure

    ew = tmp_path / "ensemble_weights.json"
    ew.write_text(
        json.dumps({"normal": {"a": 1.0}}),
        encoding="utf-8",
    )
    # age then stamp
    old = time.time() - 40 * 86400
    os.utime(ew, (old, old))
    stamp_ensemble_weights_freshness(ew, reason="test")

    report = {
        "summary": {
            "healthy": 3,
            "degraded": 5,
            "unhealthy": 1,
            "total_tracked": 9,
        },
        "scores": {},
    }
    out = attach_signal_quality_disclosure(report, data_dir=tmp_path)
    freeze = out["quality_disclosure"]["ensemble_weight_freeze"]
    assert freeze["weight_file_stale"] is False
    assert freeze["weight_freeze_active"] is False
    assert freeze["ensemble_weights_age_days"] < 1.0
