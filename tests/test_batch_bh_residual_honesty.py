"""Batch BH residual honesty: unhealthy ensemble hard-zero + compact health elevate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_apply_health_weights_hard_zeros_unhealthy():
    """Batch DU: unknown IC hard-zeros; unhealthy + IC>=min soft-floors."""
    from src.strategy.ensemble_voter import EnsembleVoter
    from src.signals.signal_source import SignalSource

    voter = EnsembleVoter.__new__(EnsembleVoter)

    healthy = SimpleNamespace(
        health_score=0.9, status="healthy", ic=0.1, predictions_count=20
    )
    # unknown IC → hard sleep (fail-closed)
    unhealthy_unknown = SimpleNamespace(
        health_score=0.43, status="unhealthy", ic=None, predictions_count=20
    )
    # IC >= ENSEMBLE_UNHEALTHY_MIN_IC (0.08) → soft floor (Batch DU; not weak 0.05)
    unhealthy_pos_ic = SimpleNamespace(
        health_score=0.43, status="unhealthy", ic=0.12, predictions_count=20
    )
    degraded = SimpleNamespace(
        health_score=0.55, status="degraded", ic=0.0, predictions_count=20
    )

    scores = {
        SignalSource.MULTI_SPEED_MOM.value: healthy,
        SignalSource.VIX_TERM_STRUCTURE.value: unhealthy_unknown,
        SignalSource.UNIFIED_OVERLAY.value: unhealthy_pos_ic,
        SignalSource.ALTERNATIVE_DATA.value: degraded,
    }

    mock_tracker = MagicMock()
    mock_tracker.calculate_all_health_scores.return_value = scores

    base = {
        SignalSource.MULTI_SPEED_MOM: 0.25,
        SignalSource.VIX_TERM_STRUCTURE: 0.25,
        SignalSource.UNIFIED_OVERLAY: 0.25,
        SignalSource.ALTERNATIVE_DATA: 0.25,
    }

    with patch(
        "src.signals.health_tracker.SignalHealthTracker",
        return_value=mock_tracker,
    ):
        out = voter._apply_health_weights(base)

    assert out[SignalSource.VIX_TERM_STRUCTURE] == 0.0
    assert out[SignalSource.UNIFIED_OVERLAY] > 0.0  # soft floor, not hard zero
    assert out[SignalSource.MULTI_SPEED_MOM] > 0
    assert out[SignalSource.ALTERNATIVE_DATA] > 0
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert getattr(voter, "_health_gate_freeze", False) is False
    assert "vix_term_structure" in getattr(voter, "_health_gate_slept", [])
    assert "unified_overlay" not in getattr(voter, "_health_gate_slept", [])


def test_apply_health_weights_freeze_when_all_unhealthy():
    from src.strategy.ensemble_voter import EnsembleVoter
    from src.signals.signal_source import SignalSource

    voter = EnsembleVoter.__new__(EnsembleVoter)
    scores = {
        SignalSource.VIX_TERM_STRUCTURE.value: SimpleNamespace(
            health_score=0.4, status="unhealthy", ic=None, predictions_count=20
        ),
        SignalSource.UNIFIED_OVERLAY.value: SimpleNamespace(
            health_score=0.3, status="unhealthy", ic=None, predictions_count=20
        ),
    }
    mock_tracker = MagicMock()
    mock_tracker.calculate_all_health_scores.return_value = scores
    base = {
        SignalSource.VIX_TERM_STRUCTURE: 0.5,
        SignalSource.UNIFIED_OVERLAY: 0.5,
    }
    with patch(
        "src.signals.health_tracker.SignalHealthTracker",
        return_value=mock_tracker,
    ):
        out = voter._apply_health_weights(base)
    assert all(v == 0.0 for v in out.values())
    assert voter._health_gate_freeze is True


def test_elevate_compact_health_status_max_severity():
    from src.dashboard.cron_scheduler_section import _elevate_compact_health_status

    h = _elevate_compact_health_status(
        {
            "status": "healthy",
            "scheduler_status": "degraded",
            "failed_cron_jobs": 0,
        }
    )
    assert h["status"] == "degraded"
    assert h.get("status_elevated_from") == "healthy"

    h2 = _elevate_compact_health_status(
        {
            "status": "healthy",
            "failed_cron_jobs": 1,
            "scheduler_status": "ok",
        }
    )
    assert h2["status"] != "healthy"


def test_elevate_compact_keeps_signal_health_status_on_quality_plane():
    """SH degraded remains compact disclosure without demoting ops status."""
    from src.dashboard.cron_scheduler_section import _elevate_compact_health_status

    h = _elevate_compact_health_status(
        {
            "status": "healthy",
            "signal_health_status": "degraded",
            "signal_health_healthy": 0,
            "signal_health_total_tracked": 9,
            "scheduler_status": "ok",
            "failed_cron_jobs": 0,
        }
    )
    assert h["status"] == "healthy"
    assert h["signal_health_status"] == "degraded"
    assert h.get("status_elevated_from") is None


def test_elevate_compact_keeps_zero_healthy_counts_on_quality_plane():
    """0/N remains compact quality disclosure without demoting ops status."""
    from src.dashboard.cron_scheduler_section import _elevate_compact_health_status

    h = _elevate_compact_health_status(
        {
            "status": "healthy",
            "signal_health_healthy": 0,
            "signal_health_total_tracked": 8,
            "scheduler_status": "ok",
            "failed_cron_jobs": 0,
        }
    )
    assert h["status"] == "healthy"
    assert h["signal_health_healthy"] == 0
    assert h["signal_health_total_tracked"] == 8
    assert h.get("status_elevate_reason") is None


def test_elevate_compact_never_promotes_worse_status():
    """Max-severity only — unhealthy stays unhealthy when SH is merely degraded."""
    from src.dashboard.cron_scheduler_section import _elevate_compact_health_status

    h = _elevate_compact_health_status(
        {
            "status": "unhealthy",
            "signal_health_status": "degraded",
            "scheduler_status": "ok",
            "failed_cron_jobs": 0,
        }
    )
    assert h["status"] == "unhealthy"
    assert "status_elevated_from" not in h


def test_cron_refresh_refuses_compact_job_count_collapse(tmp_path, monkeypatch):
    from src.dashboard.cron_scheduler_section import refresh_public_health_cron_section
    import src.paths as paths

    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public"
    data_dir.mkdir()
    public_dir.mkdir()
    # Poison short inventory (2 jobs) while compact previously had 16
    (data_dir / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-data",
                        "status": "success",
                        "last_run": "2026-07-21T04:06:00+00:00",
                        "enabled": True,
                        "state": "scheduled",
                    },
                    {
                        "name": "portfolio-lab-health",
                        "status": "success",
                        "last_run": "2026-07-21T04:00:00+00:00",
                        "enabled": True,
                        "state": "scheduled",
                    },
                ],
            }
        )
    )
    (public_dir / "health.json").write_text(
        json.dumps(
            {
                "system_status": "ok",
                "cron_jobs": [{"name": f"job-{i}", "status": "ok"} for i in range(16)],
                "scheduler_status": {"status": "ok", "backends": {}},
                "data_freshness": {},
            }
        )
    )
    (public_dir / "signals.json").write_text(
        json.dumps(
            {
                "health": {
                    "status": "healthy",
                    "cron_job_count": 16,
                    "failed_cron_jobs": 0,
                    "scheduler_status": "ok",
                }
            }
        )
    )

    old_public, old_data = paths.PUBLIC_DATA_DIR, paths.DATA_DIR
    paths.PUBLIC_DATA_DIR = public_dir
    paths.DATA_DIR = data_dir
    try:
        assert refresh_public_health_cron_section(
            public_health_path=public_dir / "health.json",
            cron_status_file=data_dir / "cron_status.json",
        )
    finally:
        paths.PUBLIC_DATA_DIR = old_public
        paths.DATA_DIR = old_data

    signals = json.loads((public_dir / "signals.json").read_text())
    health = signals["health"]
    assert health["cron_job_count"] == 16
    assert health.get("cron_job_count_drop_refused", {}).get("attempted") == 2
