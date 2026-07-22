"""Batch DV: ML feature staleness projected onto compact health."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.health_check import refresh_signals_health_kill_fields


def test_partial_health_refresh_reprojects_ml_staleness(
    tmp_path: Path, monkeypatch
) -> None:
    public = tmp_path / "public"
    private = tmp_path / "data"
    public.mkdir()
    private.mkdir()
    signals = {
        "generated_at": "2026-07-22T05:00:00+00:00",
        "generator_git_sha": "deadbeef",
        "generator_git_sha_status": "full_generate",
        "ensemble_voting": {
            "active_weights": {"cross_asset_rv": 0.5, "google_trends": 0.5},
            "n_eff": 2.0,
            "per_signal_active_weight_cap": 0.50,
            "ensemble_concentration_ok": True,
            "max_active_weight": 0.5,
        },
        "ml_signals": {
            "available": True,
            "feature_as_of": "2026-05-08",
            "feature_freshness_status": "stale",
            "feature_staleness_days": 75,
            "prediction_source_mode": "stale_features",
            "execution_role": {
                "role": "advisory_non_routed",
                "live_authoritative": False,
            },
        },
        "health": {"status": "ok"},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)

    report = {
        "status": "ok",
        "system_status": "ok",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0, "status": "ok"},
    }
    refresh_signals_health_kill_fields(
        report, public_dir=public, data_dir=private
    )

    out = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h = out.get("health") or {}
    assert h.get("ml_features_stale") is True
    assert h.get("ml_feature_freshness_status") == "stale"
    assert h.get("ml_feature_staleness_days") == 75
    assert h.get("ml_feature_as_of") == "2026-05-08"
    assert h.get("ml_live_authoritative") is False
    assert h.get("status") == "warning"
