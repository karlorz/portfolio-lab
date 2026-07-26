"""Tests for public data size-budget metadata."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.public_data_index import build_public_data_index


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _entries_by_filename(index: dict) -> dict[str, dict]:
    return {entry["filename"]: entry for entry in index["entries"]}


def test_public_data_size_budget_classifies_size_and_row_count(tmp_path: Path) -> None:
    from src.dashboard.public_data_size_budget import measure_public_data_size_budget

    within = _write_json(tmp_path / "within.json", {"rows": [{"id": 1}]})
    warning = _write_json(tmp_path / "warning.json", [{"id": idx} for idx in range(7)])
    oversized = _write_json(tmp_path / "oversized.json", [{"payload": "x" * 80} for _ in range(8)])

    within_budget = measure_public_data_size_budget(within, warning_bytes=256, max_bytes=512, warning_rows=5, max_rows=10)
    warning_budget = measure_public_data_size_budget(warning, warning_bytes=512, max_bytes=1024, warning_rows=5, max_rows=10)
    oversized_budget = measure_public_data_size_budget(
        oversized,
        warning_bytes=256,
        max_bytes=512,
        warning_rows=5,
        max_rows=20,
    )

    assert within_budget["status"] == "within_budget"
    assert within_budget["row_count"] == 1
    assert within_budget["requires_downsampling"] is False
    assert within_budget["requires_pagination"] is False

    assert warning_budget["status"] == "warning"
    assert warning_budget["row_count"] == 7
    assert warning_budget["requires_downsampling"] is True
    assert warning_budget["requires_pagination"] is False

    assert oversized_budget["status"] == "oversized"
    assert oversized_budget["estimated_parse_ms"] > 0
    assert oversized_budget["requires_downsampling"] is True
    assert oversized_budget["requires_pagination"] is True


def test_public_data_size_budget_surfaces_validation_truncation_metadata(tmp_path: Path) -> None:
    from src.dashboard.public_data_size_budget import measure_public_data_size_budget

    validation_report = _write_json(
        tmp_path / "labs_validation.json",
        {
            "schema_version": "labs-validation/v1",
            "generated_at": "2026-06-08T00:00:00+00:00",
            "results": [
                {
                    "path": "public/data/labs_scorecards.json[0]",
                    "artifact_type": "scorecard",
                    "schema_version": "labs-scorecard/v1",
                    "valid": False,
                    "errors": ["$.status: unsupported status 'ship'"],
                }
            ],
            "truncation": {
                "max_results": 1,
                "max_errors_per_result": 2,
                "total_result_count": 10,
                "returned_result_count": 1,
                "omitted_result_count": 9,
                "omitted_error_count": 27,
            },
        },
    )

    budget = measure_public_data_size_budget(validation_report)

    assert budget["row_count"] == 1
    assert budget["truncated"] is True
    assert budget["total_row_count"] == 10
    assert budget["omitted_row_count"] == 9
    assert budget["omitted_error_count"] == 27


def test_public_data_index_embeds_labs_size_budget_metadata(tmp_path: Path) -> None:
    registry = _write_json(
        tmp_path / "labs_registry.json",
        {
            "schema_version": "labs-registry/v1",
            "generated_at": "2026-06-08T00:00:00",
            "experiments": [
                {
                    "experiment_id": f"experiment-{idx}",
                    "artifact_path": f"data/backtest_results/experiment-{idx}.json",
                    "status": "validated",
                    "provenance_status": "present",
                    "metrics": {"sharpe": 0.9 + idx / 100},
                    "baseline_deltas": {"sharpe": 0.01},
                }
                for idx in range(3)
            ],
        },
    )
    scorecards = _write_json(
        tmp_path / "labs_scorecards.json",
        [
            {
                "schema_version": "labs-scorecard/v1",
                "experiment_id": f"experiment-{idx}",
                "generated_at": "2026-06-08T00:00:00",
                "status": "promote",
                "provenance_status": "present",
                "metrics": {"sharpe": 0.9 + idx / 100},
                "baseline_deltas": {"sharpe": 0.01},
            }
            for idx in range(2)
        ],
    )

    index = build_public_data_index([registry, scorecards], public_dir=tmp_path, generated_at="2026-06-08T00:00:00")

    entries = _entries_by_filename(index)
    registry_budget = entries["labs_registry.json"]["size_budget"]
    scorecard_budget = entries["labs_scorecards.json"]["size_budget"]
    assert registry_budget["schema_version"] == "public-data-size-budget/v1"
    assert registry_budget["row_count"] == 3
    assert registry_budget["status"] == "within_budget"
    assert registry_budget["render_strategy"] == "direct"
    assert registry_budget["requires_downsampling"] is False

    assert scorecard_budget["row_count"] == 2
    assert scorecard_budget["status"] == "within_budget"
    assert scorecard_budget["requires_pagination"] is False


def test_public_data_index_discovers_heavy_market_data_files_with_fetch_strategy(tmp_path: Path) -> None:
    prices = _write_json(
        tmp_path / "prices.json",
        {
            "SPY": [{"d": f"2026-01-{(idx % 28) + 1:02d}", "p": 100 + idx} for idx in range(1001)],
            "GLD": [{"d": "2026-01-01", "p": 200}],
        },
    )

    index = build_public_data_index([], public_dir=tmp_path, generated_at="2026-06-08T00:00:00")

    entries = _entries_by_filename(index)
    price_entry = entries["prices.json"]
    assert price_entry["category"] == "market_data"
    assert price_entry["schema_version"] == "prices/full-v1"
    assert price_entry["status"] == "present"
    assert price_entry["size_bytes"] == prices.stat().st_size
    assert price_entry["size_budget"]["row_count"] == 1002
    assert price_entry["size_budget"]["status"] == "oversized"
    assert price_entry["size_budget"]["render_strategy"] == "paginate"
