"""C1b: signal-health scores use one end-of-day cohort row per source/date."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.signals.health_tracker import SignalHealthTracker


def _stamp_direction(
    tracker: SignalHealthTracker,
    *,
    source: str,
    day: datetime,
    early_value: float,
    final_value: float,
    actual_return: float,
    early_repeats: int = 25,
) -> None:
    """Write noisy intraday repeats followed by the intended closing snapshot."""
    day_text = day.strftime("%Y-%m-%d")
    for minute in range(early_repeats):
        tracker.log_prediction_simple(
            source=source,
            signal_value=early_value,
            confidence=0.8,
            timestamp=f"{day_text}T09:{minute:02d}:00",
            metadata={"run": "intraday"},
        )
    tracker.log_prediction_simple(
        source=source,
        signal_value=final_value,
        confidence=0.8,
        timestamp=f"{day_text}T23:59:00",
        metadata={"run": "closing"},
    )
    tracker.update_actual_directions({"SPY": actual_return}, day_text)


def test_health_score_counts_latest_prediction_per_source_date(tmp_path) -> None:
    tracker = SignalHealthTracker(tmp_path / "health.db")
    end = datetime(2026, 7, 25)

    # The repeated early rows are wrong. The final daily snapshot is correct.
    # Run-frequency weighting would report near-zero health; date-cohort scoring
    # must report 15 correct observations, not 390 pseudo-independent trials.
    for offset in range(15):
        _stamp_direction(
            tracker,
            source="vix_term_structure",
            day=end - timedelta(days=offset * 5),
            early_value=-0.5,
            final_value=0.5,
            actual_return=0.01,
        )

    result = tracker.calculate_health_score(
        "vix_term_structure", end_date="2026-07-25"
    )

    assert result is not None
    assert result.predictions_count == 15
    assert result.accuracy_30d == pytest.approx(1.0)
    assert result.accuracy_60d == pytest.approx(1.0)
    assert result.accuracy_90d == pytest.approx(1.0)
    assert result.health_score == pytest.approx(1.0)


def test_ic_uses_latest_prediction_per_source_date(tmp_path) -> None:
    tracker = SignalHealthTracker(tmp_path / "health.db")
    end = datetime(2026, 7, 25)

    # Closing snapshots have the correct polarity; repeated early rows have the
    # opposite sign. IC should measure the six independent date cohorts.
    for offset in range(6):
        actual_return = -0.01 if offset < 3 else 0.01
        final_value = -0.8 + offset * 0.3
        _stamp_direction(
            tracker,
            source="vix_term_structure",
            day=end - timedelta(days=offset * 7),
            early_value=-final_value,
            final_value=final_value,
            actual_return=actual_return,
        )

    ic = tracker.compute_ic("vix_term_structure", end_date="2026-07-25")

    assert ic is not None
    assert ic > 0.8


def test_health_report_discloses_calendar_date_sampling(tmp_path) -> None:
    tracker = SignalHealthTracker(tmp_path / "health.db")

    policy = tracker.get_health_report()["health_score_policy"]

    assert policy["sampling_unit"] == "latest_prediction_per_source_calendar_date"
    assert "run-frequency" in policy["sampling_reason"]
