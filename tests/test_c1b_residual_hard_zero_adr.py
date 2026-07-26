"""C1b residual: ADR-006 advisory hard-zero policy contract."""

from __future__ import annotations

from types import SimpleNamespace

from src.dashboard.signal_health_section import attach_signal_quality_disclosure
from src.strategy.ensemble_voter import EnsembleVoter, SignalSource


def _report() -> dict:
    return {
        "summary": {
            "healthy": 1,
            "degraded": 5,
            "unhealthy": 3,
            "total_tracked": 9,
        },
        "scores": {},
    }


def test_quality_disclosure_cites_human_approved_advisory_adr(tmp_path) -> None:
    out = attach_signal_quality_disclosure(_report(), data_dir=tmp_path)

    policy = out["quality_disclosure"]["hard_zero_policy"]
    assert policy["decision"] == "ADR-006"
    assert policy["human_approved"] is True
    assert policy["mode"] == "advisory_only"
    assert policy["live_authoritative"] is False
    assert policy["min_labeled_daily_cohorts"] == 20
    assert policy["unhealthy_min_ic"] == 0.08
    assert policy["shadow_collection"] is True


def test_negative_ic_below_minimum_cohort_stays_shadow_soft_floor(
    tmp_path, monkeypatch
) -> None:
    scores = {
        "alternative_data": SimpleNamespace(
            status="degraded",
            health_score=0.40,
            ic=-0.20,
            predictions_count=19,
        ),
        "vix_term_structure": SimpleNamespace(
            status="healthy",
            health_score=0.70,
            ic=0.20,
            predictions_count=40,
        ),
    }

    class _Tracker:
        def calculate_all_health_scores(self):
            return scores

    monkeypatch.setattr("src.signals.health_tracker.SignalHealthTracker", _Tracker)
    voter = EnsembleVoter(data_path=tmp_path)
    result = voter._apply_health_weights(
        {
            SignalSource.ALTERNATIVE_DATA: 0.5,
            SignalSource.VIX_TERM_STRUCTURE: 0.5,
        }
    )

    assert result[SignalSource.ALTERNATIVE_DATA] > 0.0
    assert "alternative_data" not in voter._health_gate_slept
    assert "insufficient_cohorts" in voter._health_gate_soft_floor["alternative_data"]


def test_negative_ic_at_minimum_cohort_hard_sleeps(tmp_path, monkeypatch) -> None:
    scores = {
        "alternative_data": SimpleNamespace(
            status="degraded",
            health_score=0.40,
            ic=-0.20,
            predictions_count=20,
        ),
        "vix_term_structure": SimpleNamespace(
            status="healthy",
            health_score=0.70,
            ic=0.20,
            predictions_count=40,
        ),
    }

    class _Tracker:
        def calculate_all_health_scores(self):
            return scores

    monkeypatch.setattr("src.signals.health_tracker.SignalHealthTracker", _Tracker)
    voter = EnsembleVoter(data_path=tmp_path)
    result = voter._apply_health_weights(
        {
            SignalSource.ALTERNATIVE_DATA: 0.5,
            SignalSource.VIX_TERM_STRUCTURE: 0.5,
        }
    )

    assert result[SignalSource.ALTERNATIVE_DATA] == 0.0
    assert "alternative_data" in voter._health_gate_slept
    assert voter._health_gate_sleep_reasons["alternative_data"].startswith(
        "degraded_negative_ic"
    )


def test_missing_cohort_evidence_cannot_hard_sleep(tmp_path, monkeypatch) -> None:
    scores = {
        "alternative_data": SimpleNamespace(
            status="degraded",
            health_score=0.40,
            ic=-0.20,
        ),
        "vix_term_structure": SimpleNamespace(
            status="healthy",
            health_score=0.70,
            ic=0.20,
            predictions_count=40,
        ),
    }

    class _Tracker:
        def calculate_all_health_scores(self):
            return scores

    monkeypatch.setattr("src.signals.health_tracker.SignalHealthTracker", _Tracker)
    voter = EnsembleVoter(data_path=tmp_path)
    result = voter._apply_health_weights(
        {
            SignalSource.ALTERNATIVE_DATA: 0.5,
            SignalSource.VIX_TERM_STRUCTURE: 0.5,
        }
    )

    assert result[SignalSource.ALTERNATIVE_DATA] > 0.0
    assert "alternative_data" not in voter._health_gate_slept
    assert "insufficient_cohorts(0<20" in voter._health_gate_soft_floor[
        "alternative_data"
    ]
