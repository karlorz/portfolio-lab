"""C1c: per-source direction deadband fixes IC-vs-accuracy scoring gap.

The default DIRECTION_DEADBAND (0.05) was tuned for arms clipped to ±0.2
(cross_asset_regime_arb). Sources whose continuous signal is a gradual
z-score / sentiment value accumulate weak readings near zero that are noise.
Mapping those to ±1 destroys accuracy while IC (continuous) stays strong.

C1c introduces a per-source deadband override (SOURCE_DEADBANDS) so the
noise floor is set per source. Health queries recompute predicted_direction
from signal_value using the source deadband so historical rows logged before
the override are scored consistently with new predictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.signals.health_tracker import SignalHealthTracker


def _stamp_labeled(
    tracker: SignalHealthTracker,
    *,
    source: str,
    day: datetime,
    signal_value: float,
    actual_direction: int,
) -> None:
    """Write one labeled prediction for a source/date."""
    day_text = day.strftime("%Y-%m-%d")
    tracker.log_prediction_simple(
        source=source,
        signal_value=signal_value,
        confidence=0.8,
        timestamp=f"{day_text}T23:59:00",
        metadata={"run": "closing"},
    )
    tracker.update_actual_directions(
        {"SPY": 0.01 if actual_direction == 1 else -0.01}, day_text
    )


def test_default_deadband_unchanged_for_unregistered_sources() -> None:
    """Sources not in SOURCE_DEADBANDS keep the 0.05 default."""
    assert SignalHealthTracker.deadband_for("vix_term_structure") == pytest.approx(0.05)
    assert SignalHealthTracker.deadband_for("multi_speed_momentum") == pytest.approx(0.05)
    assert SignalHealthTracker.deadband_for("unknown_source") == pytest.approx(0.05)


def test_cross_asset_rv_uses_wider_deadband() -> None:
    """cross_asset_rv has a wider noise floor (gradual z-score signal)."""
    assert SignalHealthTracker.deadband_for("cross_asset_rv") > 0.05
    assert SignalHealthTracker.deadband_for("cross_asset_rv") >= 0.10


def test_log_prediction_simple_uses_source_deadband(tmp_path) -> None:
    """New predictions are discretized with the per-source deadband."""
    tracker = SignalHealthTracker(tmp_path / "health.db")
    db = SignalHealthTracker.deadband_for("cross_asset_rv")
    # A weak reading inside the cross_asset_rv deadband must be neutral
    weak = db * 0.5
    pred = tracker.direction_from_signal_value(
        weak, deadband=tracker.deadband_for("cross_asset_rv")
    )
    assert pred == 0


def test_health_score_recomputes_direction_with_source_deadband(tmp_path) -> None:
    """Historical rows are re-scored with the per-source deadband.

    cross_asset_rv readings at |signal| < deadband are noise. At the default
    0.05 deadband they map to ±1 and drag accuracy below 0.5 even though the
    strong readings (|signal| > deadband) hit. The per-source deadband must
    filter the weak readings to neutral so accuracy reflects only meaningful
    calls.
    """
    tracker = SignalHealthTracker(tmp_path / "health.db")
    end = datetime(2026, 7, 25)
    db = SignalHealthTracker.deadband_for("cross_asset_rv")

    # 20 days: 10 weak-noise readings (|signal| < db, should be neutral)
    # mixed actuals so they would drag accuracy to ~0.5 if scored as directional.
    # 10 strong readings (|signal| > db) that all hit.
    for offset in range(20):
        day = end - timedelta(days=offset)
        if offset < 10:
            # Weak noise reading near zero; actual direction alternates
            weak = db * 0.4
            actual = 1 if offset % 2 == 0 else -1
            _stamp_labeled(
                tracker, source="cross_asset_rv", day=day,
                signal_value=weak, actual_direction=actual,
            )
        else:
            # Strong reading that matches actual
            strong = db * 2.0
            _stamp_labeled(
                tracker, source="cross_asset_rv", day=day,
                signal_value=strong, actual_direction=1,
            )

    result = tracker.calculate_health_score("cross_asset_rv", end_date="2026-07-25")
    assert result is not None
    # Strong readings all hit -> accuracy must be 1.0 (weak readings are neutral
    # and excluded). If the deadband were 0.05, the weak readings would score
    # as directional and accuracy would drop toward 0.75.
    assert result.accuracy_90d == pytest.approx(1.0)


def test_vix_term_structure_healthy_regression_guard(tmp_path) -> None:
    """C1c guard: vix_term_structure (healthy) must not regress under deadband changes.

    The per-source deadband registry must not include vix_term_structure so its
    0.05 default - and its healthy status under the collapsed 0.55 bar - is
    preserved.
    """
    tracker = SignalHealthTracker(tmp_path / "health.db")
    end = datetime(2026, 7, 25)

    # 15 correct daily cohorts with strong readings (well above any deadband)
    for offset in range(15):
        _stamp_labeled(
            tracker, source="vix_term_structure",
            day=end - timedelta(days=offset * 5),
            signal_value=0.5, actual_direction=1,
        )

    result = tracker.calculate_health_score("vix_term_structure", end_date="2026-07-25")
    assert result is not None
    assert result.accuracy_90d == pytest.approx(1.0)
    assert result.health_score == pytest.approx(1.0)
