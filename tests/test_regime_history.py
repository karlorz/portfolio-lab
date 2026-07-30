"""Contract tests for the canonical daily regime-history loader."""

from __future__ import annotations

import json

from src.regime.regime_history import load_daily_regime_history


def _write_rows(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_daily_loader_sorts_collapses_normalizes_and_counts(tmp_path) -> None:
    path = tmp_path / "regime_log.json"
    _write_rows(
        path,
        [
            {"regime": "crisis", "detected_at": "2026-07-21T23:00:00Z"},
            {"regime": "normal", "detected_at": "2026-07-20T01:00:00+00:00", "vix_level": 18.5},
            {"regime": "high_vol", "detected_at": "2026-07-20T22:00:00Z", "vix": 21.2},
            {"regime": "recovery", "detected_at": "2026-07-22T00:30:00Z"},
        ],
    )

    result = load_daily_regime_history(path)

    assert result.labels == ["HIGH_VOL", "CRISIS", "RECOVERY"]
    assert result.records == [
        {"d": "2026-07-20", "r": "high_vol", "v": 21.2},
        {"d": "2026-07-21", "r": "crisis", "v": None},
        {"d": "2026-07-22", "r": "recovery", "v": None},
    ]
    assert result.metadata["raw_row_count"] == 4
    assert result.metadata["valid_row_count"] == 4
    assert result.metadata["history_len"] == 3
    assert result.metadata["observed_transition_count"] == 2
    assert result.metadata["observation_unit"] == "utc_day_last_observation"


def test_daily_loader_skips_bad_rows_and_discloses_quality(tmp_path) -> None:
    path = tmp_path / "regime_log.json"
    _write_rows(
        path,
        [
            "{not json",
            {"regime": "NORMAL"},
            {"regime": "NOT_A_REGIME", "detected_at": "2026-07-20T01:00:00Z"},
            {"regime": "NORMAL", "detected_at": "2026-07-20T01:00:00Z"},
        ],
    )

    result = load_daily_regime_history(path)

    assert result.labels == ["NORMAL"]
    assert result.metadata["malformed_row_count"] == 2
    assert result.metadata["unknown_regime_count"] == 1
    assert result.metadata["evidence_quality"] == "insufficient"


def test_nine_days_without_switches_are_prior_dominated(tmp_path) -> None:
    path = tmp_path / "regime_log.json"
    _write_rows(
        path,
        [
            {"regime": "NORMAL", "detected_at": f"2026-07-{day:02d}T12:00:00Z"}
            for day in range(20, 29)
        ],
    )

    result = load_daily_regime_history(path)

    assert result.metadata["history_len"] == 9
    assert result.metadata["observed_transition_count"] == 0
    assert result.metadata["evidence_quality"] == "prior_dominated"


def test_missing_file_returns_honest_empty_result(tmp_path) -> None:
    result = load_daily_regime_history(tmp_path / "missing.json")

    assert result.labels == []
    assert result.records == []
    assert result.metadata["history_len"] == 0
    assert result.metadata["evidence_quality"] == "insufficient"
    assert result.metadata["history_source"].endswith("missing.json")
