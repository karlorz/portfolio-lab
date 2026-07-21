"""Batch CM: SH quality disclosure when 0/N healthy + VTS row persist keeps history."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.signal_health_section import attach_signal_quality_disclosure


def test_quality_disclosure_zero_healthy_badge_and_freeze(tmp_path):
    ew = tmp_path / "ensemble_weights.json"
    ew.write_text(json.dumps({"normal": {"a": 0.5}}), encoding="utf-8")
    # age the file
    import os
    import time

    old = time.time() - 40 * 86400
    os.utime(ew, (old, old))

    report = {
        "overall_health": "degraded",
        "status": "degraded",
        "summary": {
            "healthy": 0,
            "degraded": 8,
            "unhealthy": 1,
            "total_tracked": 9,
        },
        "scores": {
            "vix_term_structure": {
                "status": "unhealthy",
                "health_score": 0.46,
                "accuracy_30d": 0.44,
                "ic": 0.05,
            },
            "multi_speed_momentum": {
                "status": "degraded",
                "health_score": 0.55,
                "accuracy_30d": 0.42,
                "ic": 0.02,
            },
        },
    }
    out = attach_signal_quality_disclosure(report, data_dir=tmp_path)
    qd = out["quality_disclosure"]
    assert qd["zero_healthy_sources"] is True
    assert qd["badge"] == "0/9 healthy sources"
    assert qd["severity"] == "degraded"
    assert out["summary"]["quality_badge"] == "0/9 healthy sources"
    assert out["summary"]["ensemble_weight_freeze_active"] is True
    freeze = qd["ensemble_weight_freeze"]
    assert freeze["weight_freeze_active"] is True
    assert freeze["weight_file_stale"] is True
    assert freeze["ensemble_weights_age_days"] > 30
    assert qd["worst_sleeves"][0]["source"] == "vix_term_structure"


def test_quality_disclosure_not_zero_when_some_healthy(tmp_path):
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
    assert out["quality_disclosure"]["zero_healthy_sources"] is False
    assert out["summary"]["quality_badge"] == "3/9 healthy sources"


def test_batch_cq_stale_file_not_freeze_when_healthy_gt_zero(tmp_path):
    """Batch CQ: age>7d is advisory stale only; freeze requires healthy==0."""
    import os
    import time

    ew = tmp_path / "ensemble_weights.json"
    ew.write_text(json.dumps({"normal": {"a": 0.5}}), encoding="utf-8")
    old = time.time() - 46 * 86400
    os.utime(ew, (old, old))

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
    assert freeze["weight_file_stale"] is True
    assert freeze["weight_freeze_active"] is False
    assert freeze["freeze_reason"] is None
    assert freeze["ensemble_weights_age_days"] > 40
    assert "stale_note" in freeze
    assert out["summary"].get("ensemble_weight_freeze_active") is not True
    assert out["summary"].get("ensemble_weights_file_stale") is True
    assert out["summary"].get("ensemble_weights_age_days") > 40


def test_vts_persist_file_row_preserves_history(tmp_path, monkeypatch):
    from src.signals.vix_term_structure import VIXTermStructureSignalGenerator

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vts_path = data_dir / "vix_term_structure.json"
    # Full-ish history with meta
    history = {
        "_meta": {"schema": "vix_term_structure/v1", "n_dates": 2},
        "2026-05-12": {"date": "2026-05-12", "vix_spot": 20.0, "front_month": 21.0},
        "2026-07-20": {"date": "2026-07-20", "vix_spot": 18.0, "front_month": 19.0},
    }
    vts_path.write_text(json.dumps(history), encoding="utf-8")

    gen = VIXTermStructureSignalGenerator.__new__(VIXTermStructureSignalGenerator)
    gen.DATA_DIR = data_dir
    gen.VIX_DATA_PATH = vts_path
    gen.db_path = data_dir / "market.db"

    # Caller only has stripped view (as load_vix_data would)
    stripped = {
        k: v
        for k, v in history.items()
        if not str(k).startswith("_")
    }
    gen._persist_file_row(
        stripped,
        {
            "as_of": "2026-07-21",
            "vix_spot": 17.0,
            "front_month": 19.5,
            "third_month": 20.0,
            "source": "market.db",
        },
    )
    disk = json.loads(vts_path.read_text(encoding="utf-8"))
    assert "2026-05-12" in disk
    assert "2026-07-20" in disk
    assert "2026-07-21" in disk
    assert "_meta" in disk
    assert disk["_meta"]["n_dates"] == 3
