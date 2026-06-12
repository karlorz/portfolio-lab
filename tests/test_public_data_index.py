"""Tests for the public data index manifest contract."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.dashboard.generator import DashboardGenerator
from src.dashboard.public_data_index import build_public_data_index


def _create_market_db(db_path: Path, days: int = 30) -> None:
    """Create a minimal market database for dashboard generation."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE prices (
            symbol TEXT,
            date TEXT,
            close REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regime_log (
            date TEXT,
            regime TEXT,
            vix_level REAL,
            detected_at TEXT
        )
        """
    )
    base_date = datetime.now()
    for symbol in ("SPY", "GLD", "TLT", "QQQ"):
        for offset in range(days):
            date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
            close = round(500 + np.random.normal(0, 2.0), 2)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", (symbol, date, close))
    conn.commit()
    conn.close()


def _make_generator(tmp_path: Path) -> DashboardGenerator:
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    return gen


def _run_generator(tmp_path: Path) -> dict:
    gen = _make_generator(tmp_path)
    with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            gen.run()
    with open(tmp_path / "index.json") as f:
        return json.load(f)


def _entries_by_filename(index: dict) -> dict[str, dict]:
    return {entry["filename"]: entry for entry in index["entries"]}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _registry_payload(row_count: int) -> dict:
    return {
        "schema_version": "labs-registry/v1",
        "generated_at": "2026-06-08T00:00:00+00:00",
        "experiments": [
            {
                "experiment_id": f"experiment-{idx}",
                "artifact_path": f"data/backtest_results/experiment-{idx}.json",
                "status": "validated",
                "provenance_status": "present",
                "metrics": {"sharpe": 0.9 + idx / 10_000},
                "baseline_deltas": {"sharpe": 0.01},
            }
            for idx in range(row_count)
        ],
    }


def _validation_report_payload(row_count: int) -> dict:
    return {
        "schema_version": "labs-validation/v1",
        "generated_at": "2026-06-08T00:00:00+00:00",
        "results": [
            {
                "path": f"public/data/labs_registry.json[{idx}]",
                "artifact_type": "registry",
                "schema_version": "labs-registry/v1",
                "valid": True,
                "errors": [],
                "experiment_id": f"experiment-{idx}",
                "artifact_path": f"data/backtest_results/experiment-{idx}.json",
            }
            for idx in range(row_count)
        ],
    }


def test_public_index_keeps_files_list_and_adds_typed_entries(tmp_path: Path) -> None:
    index = _run_generator(tmp_path)

    assert index["schema_version"] == "public-data-index/v1"
    assert "dashboard.json" in index["files"]
    assert "generated_at" in index

    entries = _entries_by_filename(index)
    dashboard_entry = entries["dashboard.json"]
    assert dashboard_entry["category"] == "dashboard"
    assert dashboard_entry["schema_version"] == "dashboard/v1"
    assert dashboard_entry["status"] == "present"
    assert dashboard_entry["validation_status"] == "not_applicable"
    assert dashboard_entry["size_bytes"] > 0
    assert len(dashboard_entry["sha256"]) == 64
    assert dashboard_entry["generated_at"]


def test_public_index_reuses_cached_hash_for_unchanged_files(tmp_path: Path) -> None:
    artifact = _write_json(tmp_path / "dashboard.json", {"generated_at": "2026-06-08T00:00:00"})
    cache_path = tmp_path / ".public_data_index_hash_cache.json"

    first = build_public_data_index(
        [artifact],
        public_dir=tmp_path,
        generated_at="2026-06-08T00:00:00",
        hash_cache_path=cache_path,
    )
    first_hash = _entries_by_filename(first)["dashboard.json"]["sha256"]

    with patch("src.dashboard.public_data_index._sha256_file", side_effect=AssertionError("cache miss")):
        second = build_public_data_index(
            [artifact],
            public_dir=tmp_path,
            generated_at="2026-06-08T00:00:00",
            hash_cache_path=cache_path,
        )

    assert cache_path.exists()
    assert _entries_by_filename(second)["dashboard.json"]["sha256"] == first_hash


def test_public_index_recomputes_cached_hash_when_file_size_changes(tmp_path: Path) -> None:
    artifact = _write_json(tmp_path / "dashboard.json", {"generated_at": "2026-06-08T00:00:00"})
    cache_path = tmp_path / ".public_data_index_hash_cache.json"
    first = build_public_data_index(
        [artifact],
        public_dir=tmp_path,
        generated_at="2026-06-08T00:00:00",
        hash_cache_path=cache_path,
    )
    first_hash = _entries_by_filename(first)["dashboard.json"]["sha256"]
    artifact.write_text(json.dumps({"generated_at": "2026-06-08T00:00:00", "rows": [{"id": 1}]}))

    with patch(
        "src.dashboard.public_data_index._sha256_file",
        wraps=__import__(
            "src.dashboard.public_data_index",
            fromlist=["_sha256_file"],
        )._sha256_file,
    ) as sha_spy:
        second = build_public_data_index(
            [artifact],
            public_dir=tmp_path,
            generated_at="2026-06-08T00:00:00",
            hash_cache_path=cache_path,
        )

    second_hash = _entries_by_filename(second)["dashboard.json"]["sha256"]
    assert sha_spy.call_count == 1
    assert second_hash != first_hash
    assert len(second_hash) == 64


def test_public_index_hash_cache_preserves_index_output_semantics(tmp_path: Path) -> None:
    artifact = _write_json(
        tmp_path / "dashboard.json",
        {"generated_at": "2026-06-08T00:00:00", "paper_portfolio": []},
    )
    cache_path = tmp_path / ".public_data_index_hash_cache.json"

    cached = build_public_data_index(
        [artifact],
        public_dir=tmp_path,
        generated_at="2026-06-08T00:00:00",
        hash_cache_path=cache_path,
    )
    uncached = build_public_data_index(
        [artifact],
        public_dir=tmp_path,
        generated_at="2026-06-08T00:00:00",
        use_hash_cache=False,
    )

    assert cached == uncached


def test_public_index_adds_market_source_metadata_from_manifest(tmp_path: Path) -> None:
    prices = _write_json(tmp_path / "prices.json", {"SPY": [{"d": "2026-06-10", "p": 600.0}]})
    _write_json(
        tmp_path / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "artifacts": [
                {
                    "artifact": "prices.json",
                    "provider": "Yahoo Finance",
                    "feed": "chart/v8",
                    "source_mode": "live",
                    "status": "success",
                    "fetched_at": "2026-06-11T00:00:00+00:00",
                    "latest_observation": "2026-06-10",
                    "row_count": 1,
                    "data_quality": {
                        "artifact": "data_quality.json",
                        "schema_version": "price-data-quality/v1",
                        "generated_at": "2026-06-11T00:00:00+00:00",
                        "status": "ok",
                        "issue_counts": {
                            "duplicate_dates": 0,
                            "empty_symbols": 0,
                            "extreme_returns": 0,
                            "internal_gaps": 0,
                            "invalid_dates": 0,
                            "invalid_prices": 0,
                            "missing_required_keys": 0,
                            "non_monotonic_rows": 0,
                            "non_object_records": 0,
                            "split_like_returns": 0,
                            "stale_latest_dates": 0,
                            "total": 0,
                        },
                    },
                }
            ],
        },
    )

    index = build_public_data_index([prices], public_dir=tmp_path, generated_at="2026-06-11T00:00:00+00:00")

    entries = _entries_by_filename(index)
    assert entries["prices.json"]["source_manifest_path"] == "source_manifest.json"
    assert entries["prices.json"]["source_metadata"] == {
        "provider": "Yahoo Finance",
        "feed": "chart/v8",
        "source_mode": "live",
        "status": "success",
        "fetched_at": "2026-06-11T00:00:00+00:00",
        "latest_observation": "2026-06-10",
        "row_count": 1,
        "data_quality": {
            "artifact": "data_quality.json",
            "schema_version": "price-data-quality/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "status": "ok",
            "issue_counts": {
                "duplicate_dates": 0,
                "empty_symbols": 0,
                "extreme_returns": 0,
                "internal_gaps": 0,
                "invalid_dates": 0,
                "invalid_prices": 0,
                "missing_required_keys": 0,
                "non_monotonic_rows": 0,
                "non_object_records": 0,
                "split_like_returns": 0,
                "stale_latest_dates": 0,
                "total": 0,
            },
        },
    }
    assert entries["source_manifest.json"]["schema_version"] == "market-data-source-manifest/v1"
    assert entries["source_manifest.json"]["category"] == "market_data"


def test_public_index_adds_top_level_source_manifest_identity(tmp_path: Path) -> None:
    prices = _write_json(tmp_path / "prices.json", {"SPY": [{"d": "2026-06-10", "p": 600.0}]})
    _write_json(
        tmp_path / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "artifacts": [{"artifact": "prices.json", "provider": "Yahoo Finance", "status": "success"}],
        },
    )

    index = build_public_data_index([prices], public_dir=tmp_path, generated_at="2026-06-11T00:00:00+00:00")
    entries = _entries_by_filename(index)

    assert index["source_manifest"] == {
        "path": "source_manifest.json",
        "schema_version": "market-data-source-manifest/v1",
        "generated_at": "2026-06-11T00:00:00+00:00",
        "sha256": entries["source_manifest.json"]["sha256"],
    }


def test_public_index_source_manifest_identity_hash_changes_with_manifest_content(tmp_path: Path) -> None:
    prices = _write_json(tmp_path / "prices.json", {"SPY": [{"d": "2026-06-10", "p": 600.0}]})
    manifest = tmp_path / "source_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "artifacts": [{"artifact": "prices.json", "provider": "Yahoo Finance", "status": "success"}],
        },
    )
    first = build_public_data_index(
        [prices],
        public_dir=tmp_path,
        generated_at="2026-06-11T00:00:00+00:00",
        use_hash_cache=False,
    )

    _write_json(
        manifest,
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "artifacts": [
                {
                    "artifact": "prices.json",
                    "provider": "Licensed Feed",
                    "status": "success",
                }
            ],
        },
    )
    second = build_public_data_index(
        [prices],
        public_dir=tmp_path,
        generated_at="2026-06-11T00:00:00+00:00",
        use_hash_cache=False,
    )

    assert first["source_manifest"]["sha256"] != second["source_manifest"]["sha256"]


def test_public_index_marks_project_generated_artifacts_public_safe(tmp_path: Path) -> None:
    dashboard = _write_json(tmp_path / "dashboard.json", {"generated_at": "2026-06-11T00:00:00+00:00"})

    index = build_public_data_index([dashboard], public_dir=tmp_path, generated_at="2026-06-11T00:00:00+00:00")

    entry = _entries_by_filename(index)["dashboard.json"]
    assert entry["redistribution_mode"] == "public_summary"
    assert entry["license_scope"] == "project_generated"
    assert entry["public_safe"] is True
    assert "licensing_notes" in entry


def test_public_index_applies_restricted_provider_artifact_policy_without_secrets(tmp_path: Path) -> None:
    prices = _write_json(tmp_path / "prices.json", {"SPY": [{"d": "2026-06-10", "p": 600.0}]})
    _write_json(
        tmp_path / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-06-11T00:00:00+00:00",
            "artifacts": [
                {
                    "artifact": "prices.json",
                    "provider": "Licensed Fixture",
                    "feed": "eod",
                    "source_mode": "live",
                    "status": "success",
                    "fetched_at": "2026-06-11T00:00:00+00:00",
                    "latest_observation": "2026-06-10",
                    "row_count": 1,
                    "redistribution_mode": "restricted",
                    "license_scope": "licensed_provider",
                    "public_safe": False,
                    "provider_account_id": "acct-secret",
                    "authorization_header": "Bearer secret-token",
                    "signed_url": "https://provider.example/download?sig=secret",
                }
            ],
        },
    )

    index = build_public_data_index([prices], public_dir=tmp_path, generated_at="2026-06-11T00:00:00+00:00")

    entry = _entries_by_filename(index)["prices.json"]
    assert entry["redistribution_mode"] == "restricted"
    assert entry["license_scope"] == "licensed_provider"
    assert entry["public_safe"] is False
    assert entry["source_metadata"]["provider"] == "Licensed Fixture"
    serialized_entry = json.dumps(entry)
    assert "acct-secret" not in serialized_entry
    assert "secret-token" not in serialized_entry
    assert "sig=secret" not in serialized_entry


def test_public_index_keeps_source_metadata_optional_for_non_market_entries(tmp_path: Path) -> None:
    dashboard = _write_json(tmp_path / "dashboard.json", {"generated_at": "2026-06-11T00:00:00+00:00"})

    index = build_public_data_index([dashboard], public_dir=tmp_path, generated_at="2026-06-11T00:00:00+00:00")

    entry = _entries_by_filename(index)["dashboard.json"]
    assert "source_manifest_path" not in entry
    assert "source_metadata" not in entry


def test_public_index_generates_labs_registry_page_shards(tmp_path: Path) -> None:
    registry = _write_json(tmp_path / "labs_registry.json", _registry_payload(1001))

    index = build_public_data_index([registry], public_dir=tmp_path, generated_at="2026-06-08T00:00:00")

    entry = _entries_by_filename(index)["labs_registry.json"]
    assert entry["size_budget"]["render_strategy"] == "paginate"
    assert entry["pagination"] == {
        "total_rows": 1001,
        "page_size": 1000,
        "page_count": 2,
        "pages": [
            {"page": 1, "path": "labs_registry.page-1.json", "row_count": 1000},
            {"page": 2, "path": "labs_registry.page-2.json", "row_count": 1},
        ],
    }
    assert "labs_registry.page-1.json" not in index["files"]
    assert "labs_registry.page-2.json" not in index["files"]

    page_1 = json.loads((tmp_path / "labs_registry.page-1.json").read_text())
    page_2 = json.loads((tmp_path / "labs_registry.page-2.json").read_text())
    full_registry = json.loads(registry.read_text())

    assert len(page_1["experiments"]) == 1000
    assert page_1["experiments"][0]["experiment_id"] == "experiment-0"
    assert len(page_2["experiments"]) == 1
    assert page_2["experiments"][0]["experiment_id"] == "experiment-1000"
    assert len(full_registry["experiments"]) == 1001


def test_public_index_generates_labs_validation_page_shards(tmp_path: Path) -> None:
    validation = _write_json(tmp_path / "labs_validation.json", _validation_report_payload(1001))

    index = build_public_data_index([validation], public_dir=tmp_path, generated_at="2026-06-08T00:00:00")

    entry = _entries_by_filename(index)["labs_validation.json"]
    assert entry["size_budget"]["render_strategy"] == "paginate"
    assert entry["pagination"]["total_rows"] == 1001
    assert entry["pagination"]["page_size"] == 1000
    assert entry["pagination"]["page_count"] == 2
    assert entry["pagination"]["pages"] == [
        {"page": 1, "path": "labs_validation.page-1.json", "row_count": 1000},
        {"page": 2, "path": "labs_validation.page-2.json", "row_count": 1},
    ]

    page_2 = json.loads((tmp_path / "labs_validation.page-2.json").read_text())
    assert page_2["schema_version"] == "labs-validation/v1"
    assert page_2["generated_at"] == "2026-06-08T00:00:00+00:00"
    assert len(page_2["results"]) == 1
    assert page_2["results"][0]["path"] == "public/data/labs_registry.json[1000]"


def test_public_index_removes_obsolete_labs_page_shards_when_row_count_shrinks(tmp_path: Path) -> None:
    registry = _write_json(tmp_path / "labs_registry.json", _registry_payload(2001))

    initial = build_public_data_index([registry], public_dir=tmp_path, generated_at="2026-06-08T00:00:00")
    assert _entries_by_filename(initial)["labs_registry.json"]["pagination"]["page_count"] == 3
    assert (tmp_path / "labs_registry.page-3.json").exists()

    registry.write_text(json.dumps(_registry_payload(1001)))
    updated = build_public_data_index([registry], public_dir=tmp_path, generated_at="2026-06-08T00:05:00")

    entry = _entries_by_filename(updated)["labs_registry.json"]
    assert entry["pagination"]["page_count"] == 2
    assert [page["path"] for page in entry["pagination"]["pages"]] == [
        "labs_registry.page-1.json",
        "labs_registry.page-2.json",
    ]
    assert (tmp_path / "labs_registry.page-1.json").exists()
    assert (tmp_path / "labs_registry.page-2.json").exists()
    assert (tmp_path / "labs_registry.page-3.json").exists() is False


def test_public_index_represents_missing_optional_labs_files(tmp_path: Path) -> None:
    index = _run_generator(tmp_path)

    entries = _entries_by_filename(index)
    labs_registry = entries["labs_registry.json"]
    assert labs_registry["category"] == "labs"
    assert labs_registry["schema_version"] == "labs-registry/v1"
    assert labs_registry["status"] == "missing"
    assert labs_registry["validation_status"] == "missing"
    assert labs_registry["size_bytes"] is None
    assert labs_registry["sha256"] is None
    assert "labs_registry.json" not in index["files"]


def test_dashboard_generation_publishes_labs_scorecards_from_registry_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "backtest_results" / "scorecard_candidate_results.json",
        {
            "experiment_id": "scorecard-candidate",
            "sharpe_ratio": 1.04,
            "dsr": 0.97,
            "baseline_sharpe": 0.96,
            "max_drawdown": -18.0,
        },
    )

    index = _run_generator(tmp_path)

    entries = _entries_by_filename(index)
    scorecards = entries["labs_scorecards.json"]
    assert scorecards["status"] == "present"
    assert scorecards["schema_version"] == "labs-scorecard/v1"
    assert scorecards["validation_status"] == "valid"
    assert scorecards["size_budget"]["render_strategy"] == "direct"
    assert "labs_scorecards.json" in index["files"]

    payload = json.loads((tmp_path / "labs_scorecards.json").read_text())
    assert payload[0]["experiment_id"] == "scorecard-candidate"
    assert payload[0]["status"] == "watch"


def test_dashboard_generation_publishes_labs_replays_from_explicit_targets(tmp_path: Path) -> None:
    marker = tmp_path / "unsafe-replay-ran.txt"
    _write_json(
        tmp_path / "labs_replay_targets.json",
        [
            {
                "experiment_id": "dashboard-unsafe-replay",
                "artifact_path": "data/dashboard-unsafe-replay.json",
                "status": "candidate",
                "provenance_status": "sidecar",
                "metrics": {"sharpe": 0.95},
                "baseline_deltas": {},
                "command": f"python -c \"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\"",
                "replay_safe": False,
                "fetches_market_data": False,
            }
        ],
    )

    index = _run_generator(tmp_path)

    entries = _entries_by_filename(index)
    replays = entries["labs_replays.json"]
    assert marker.exists() is False
    assert replays["status"] == "present"
    assert replays["schema_version"] == "labs-replay/v1"
    assert replays["validation_status"] == "valid"
    assert replays["size_budget"]["render_strategy"] == "direct"
    assert "labs_replays.json" in index["files"]

    payload = json.loads((tmp_path / "labs_replays.json").read_text())
    assert payload[0]["experiment_id"] == "dashboard-unsafe-replay"
    assert payload[0]["status"] == "warning"
    assert payload[0]["failure_reason"] == "safety_skip"


def test_public_index_marks_invalid_labs_artifact_without_breaking_run(tmp_path: Path) -> None:
    invalid_registry = {
        "schema_version": "labs-registry/v1",
        "generated_at": "2026-06-08T00:00:00",
        "experiments": [
            {
                "experiment_id": "bad-registry-row",
                "artifact_path": "public/data/bad.json",
                "status": "validated",
                "provenance_status": "present",
                "baseline_deltas": {},
            }
        ],
    }
    (tmp_path / "labs_registry.json").write_text(json.dumps(invalid_registry))

    index = _run_generator(tmp_path)

    labs_registry = _entries_by_filename(index)["labs_registry.json"]
    assert labs_registry["status"] == "present"
    assert labs_registry["validation_status"] == "invalid"
    assert labs_registry["size_bytes"] > 0
    assert len(labs_registry["sha256"]) == 64
    assert any("$.experiments[0].metrics" in error for error in labs_registry["validation_errors"])


def test_public_index_discovers_and_validates_static_experiment_diff_artifacts(tmp_path: Path) -> None:
    diff_artifact = _write_json(
        tmp_path / "experiment_diff.json",
        {
            "schema_version": "experiment-diff/v1",
            "generated_at": "2026-06-08T12:00:00+00:00",
            "left": {
                "label": "champion",
                "experiment_id": "champion",
                "artifact_path": "data/champion.json",
                "artifact_type": "registry_row",
            },
            "right": {
                "label": "challenger",
                "experiment_id": "challenger",
                "artifact_path": "data/challenger.json",
                "artifact_type": "registry_row",
            },
            "metric_deltas": {
                "sharpe": {
                    "left": 0.95,
                    "right": 0.99,
                    "delta": 0.04,
                },
            },
            "missing_metrics": [],
            "config_diffs": {},
            "provenance": {
                "left": "present",
                "right": "stale",
                "changed": True,
            },
        },
    )

    index = build_public_data_index([], public_dir=tmp_path, generated_at="2026-06-08T12:00:00+00:00")

    entry = _entries_by_filename(index)["experiment_diff.json"]
    assert "experiment_diff.json" in index["files"]
    assert entry["category"] == "labs"
    assert entry["schema_version"] == "experiment-diff/v1"
    assert entry["status"] == "present"
    assert entry["validation_status"] == "valid"
    assert entry["validation_errors"] == []
    assert entry["size_bytes"] == diff_artifact.stat().st_size
    assert entry["size_budget"]["render_strategy"] == "direct"


def test_public_index_marks_invalid_static_experiment_diff_as_diagnosable(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "experiment_diff_invalid.json",
        {
            "schema_version": "experiment-diff/v1",
            "generated_at": "2026-06-08T12:00:00+00:00",
            "left": {"label": "champion", "artifact_type": "registry_row"},
            "right": {"label": "challenger", "artifact_type": "registry_row"},
            "metric_deltas": {
                "sharpe": {
                    "left": 0.95,
                    "right": 0.99,
                },
            },
            "missing_metrics": [],
            "config_diffs": {},
            "provenance": {
                "left": "present",
                "right": "stale",
                "changed": True,
            },
        },
    )

    index = build_public_data_index([], public_dir=tmp_path, generated_at="2026-06-08T12:00:00+00:00")

    entry = _entries_by_filename(index)["experiment_diff_invalid.json"]
    assert entry["category"] == "labs"
    assert entry["schema_version"] == "experiment-diff/v1"
    assert entry["status"] == "present"
    assert entry["validation_status"] == "invalid"
    assert any("$.metric_deltas.sharpe.delta" in error for error in entry["validation_errors"])
    assert entry["size_budget"]["render_strategy"] == "direct"


def test_dashboard_run_publishes_labs_registry_when_experiment_artifacts_exist(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "backtest_results"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "walk_forward_report.json").write_text(
        json.dumps(
            {
                "sharpe_ratio": 1.23,
                "cagr": 9.8,
                "max_drawdown": -19.2,
                "baseline_sharpe": 0.95,
            }
        )
    )

    index = _run_generator(tmp_path)

    entries = _entries_by_filename(index)
    labs_registry = entries["labs_registry.json"]
    assert labs_registry["status"] == "present"
    assert labs_registry["validation_status"] == "valid"
    assert "labs_registry.json" in index["files"]
    labs_validation = entries["labs_validation.json"]
    assert labs_validation["status"] == "present"
    assert labs_validation["validation_status"] == "valid"
    assert "labs_validation.json" in index["files"]

    registry = json.loads((tmp_path / "labs_registry.json").read_text())
    rows = {row["experiment_id"]: row for row in registry["experiments"]}
    assert rows["artifact:walk_forward_report"]["artifact_path"] == "backtest_results/walk_forward_report.json"
    assert rows["artifact:walk_forward_report"]["metrics"]["sharpe"] == 1.23
    validation_report = json.loads((tmp_path / "labs_validation.json").read_text())
    assert validation_report["schema_version"] == "labs-validation/v1"
    assert validation_report["results"][0]["path"] == "labs_registry.json"
    assert validation_report["results"][0]["valid"] is True
