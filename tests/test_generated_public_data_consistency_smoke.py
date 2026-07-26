"""Offline smoke tests for generated public data artifact consistency."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from scripts.check_public_data_consistency import check_public_data_consistency


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_consistent_public_data_set(app_dir: Path, *, source_hash: str | None = None) -> None:
    source_generated_at = "2026-06-12T09:05:25.028Z"
    index_generated_at = "2026-06-12T09:06:00+00:00"
    public_data = app_dir / "public" / "data"
    dist_data = app_dir / "dist" / "data"

    _write_json(
        public_data / "source_manifest.json",
        {
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": source_generated_at,
            "artifacts": [{"artifact": "prices.json", "provider": "Yahoo Finance", "status": "success"}],
        },
    )
    actual_source_hash = _sha256(public_data / "source_manifest.json")
    _write_json(
        public_data / "index.json",
        {
            "schema_version": "public-data-index/v1",
            "generated_at": index_generated_at,
            "source_manifest": {
                "path": "source_manifest.json",
                "schema_version": "market-data-source-manifest/v1",
                "generated_at": source_generated_at,
                "sha256": source_hash or actual_source_hash,
            },
            "entries": [
                {
                    "filename": "source_manifest.json",
                    "path": "source_manifest.json",
                    "status": "present",
                    "generated_at": source_generated_at,
                    "sha256": source_hash or actual_source_hash,
                }
            ],
        },
    )
    _write_json(public_data / "health.json", {"status": "ok", "generated_at": index_generated_at})
    dist_data.mkdir(parents=True, exist_ok=True)
    for filename in ("source_manifest.json", "index.json", "health.json"):
        shutil.copyfile(public_data / filename, dist_data / filename)


def test_generated_public_data_consistency_smoke_accepts_coherent_artifact_set(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is True, result.errors


def test_generated_public_data_consistency_smoke_rejects_source_manifest_hash_drift(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path, source_hash="0" * 64)

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any(
        "public/data/index.json source_manifest.sha256 does not match public/data/source_manifest.json" in error
        for error in result.errors
    )


def test_generated_public_data_consistency_smoke_rejects_missing_required_artifact(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    (tmp_path / "public" / "data" / "health.json").unlink()

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert "public/data/health.json is missing" in result.errors


def test_generated_public_data_consistency_rejects_critical_health_without_health_slo_alert(
    tmp_path: Path,
) -> None:
    """Critical health/SLO must not be invisible from alerts.json."""
    from src.dashboard.health_slo_alerts import HEALTH_SLO_ALERT_TYPE

    _write_consistent_public_data_set(tmp_path)
    public_data = tmp_path / "public" / "data"
    _write_json(
        public_data / "health.json",
        {
            "system_status": "critical",
            "generated_at": "2026-06-12T09:06:00+00:00",
            "data_pipeline_slo": {
                "status": "critical",
                "top_dimension": "alpaca_feed_entitlement",
            },
        },
    )
    # alerts present but missing health_slo projection
    _write_json(
        public_data / "alerts.json",
        {
            "alerts": [
                {
                    "level": "error",
                    "type": "kill_switch",
                    "title": "Kill Switch",
                    "message": "test",
                }
            ],
            "count": 1,
        },
    )
    # index alerts.json so unmanaged-json check does not fire first
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "filename": "alerts.json",
            "path": "alerts.json",
            "status": "present",
            "generated_at": index.get("generated_at"),
        }
    )
    _write_json(index_path, index)
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(public_data / "index.json", tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any(
        f"type={HEALTH_SLO_ALERT_TYPE!r}" in error or HEALTH_SLO_ALERT_TYPE in error
        for error in result.errors
    )


def test_generated_public_data_consistency_accepts_critical_health_with_health_slo_alert(
    tmp_path: Path,
) -> None:
    from src.dashboard.health_slo_alerts import HEALTH_SLO_ALERT_TYPE

    _write_consistent_public_data_set(tmp_path)
    public_data = tmp_path / "public" / "data"
    _write_json(
        public_data / "health.json",
        {
            "system_status": "critical",
            "generated_at": "2026-06-12T09:06:00+00:00",
            "data_pipeline_slo": {
                "status": "critical",
                "top_dimension": "alpaca_feed_entitlement",
            },
        },
    )
    _write_json(
        public_data / "alerts.json",
        {
            "alerts": [
                {
                    "level": "error",
                    "type": HEALTH_SLO_ALERT_TYPE,
                    "title": "Critical Health/SLO: alpaca_feed_entitlement",
                    "message": "Critical health/SLO: alpaca_feed_entitlement (missing_entitlement)",
                    "top_dimension": "alpaca_feed_entitlement",
                    "requires_action": True,
                }
            ],
            "count": 1,
        },
    )
    index_path = public_data / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "filename": "alerts.json",
            "path": "alerts.json",
            "status": "present",
            "generated_at": index.get("generated_at"),
        }
    )
    _write_json(index_path, index)
    shutil.copyfile(public_data / "health.json", tmp_path / "dist" / "data" / "health.json")
    shutil.copyfile(public_data / "index.json", tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is True, result.errors


def test_generated_public_data_consistency_smoke_rejects_present_index_entry_with_missing_path(
    tmp_path: Path,
) -> None:
    _write_consistent_public_data_set(tmp_path)
    index_path = tmp_path / "public" / "data" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "filename": "overlay_dashboard.json",
            "path": "overlay_dashboard.json",
            "status": "present",
        }
    )
    _write_json(index_path, index)
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert (
        "public/data/index.json entry overlay_dashboard.json is marked present "
        "but public/data/overlay_dashboard.json is missing"
    ) in result.errors


def test_generated_public_data_consistency_smoke_rejects_manifest_referenced_quality_report_missing_from_index(
    tmp_path: Path,
) -> None:
    _write_consistent_public_data_set(tmp_path)
    source_path = tmp_path / "public" / "data" / "source_manifest.json"
    index_path = tmp_path / "public" / "data" / "index.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest["artifacts"][0]["data_quality"] = {
        "artifact": "data_quality.json",
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-06-12T09:05:25.028Z",
        "status": "ok",
    }
    _write_json(source_path, source_manifest)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["source_manifest"]["sha256"] = _sha256(source_path)
    _write_json(index_path, index)
    shutil.copyfile(source_path, tmp_path / "dist" / "data" / "source_manifest.json")
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert (
        "public/data/source_manifest.json references data_quality.json "
        "but public/data/index.json has no entry for it"
    ) in result.errors


def test_generated_public_data_consistency_smoke_rejects_unmanaged_public_json(
    tmp_path: Path,
) -> None:
    _write_consistent_public_data_set(tmp_path)
    _write_json(
        tmp_path / "public" / "data" / "duration-sweep-results.json",
        {"schema_version": "duration-sweep-results/v1", "results": []},
    )

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert (
        "public/data/duration-sweep-results.json exists but is absent from public/data/index.json"
    ) in result.errors


def test_generated_public_data_consistency_smoke_rejects_stale_market_db_vix3m(
    tmp_path: Path,
) -> None:
    _write_consistent_public_data_set(tmp_path)
    prices_path = tmp_path / "public" / "data" / "prices.json"
    _write_json(
        prices_path,
        {
            "^VIX3M": [
                {"d": "2026-06-26", "p": 20.13},
                {"d": "2026-07-02", "p": 19.04},
            ]
        },
    )
    index_path = tmp_path / "public" / "data" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "filename": "prices.json",
            "path": "prices.json",
            "status": "present",
            "sha256": _sha256(prices_path),
        }
    )
    _write_json(index_path, index)
    shutil.copyfile(index_path, tmp_path / "dist" / "data" / "index.json")
    db_path = tmp_path / "data" / "market.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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
        conn.execute("INSERT INTO prices VALUES ('^VIX3M', '2026-06-26', 20.13)")

    result = check_public_data_consistency(tmp_path, env={}, allow_repo_public_data=True)

    assert result.ok is False
    assert any("^VIX3M market.db latest date 2026-06-26 lags prices.json 2026-07-02 by 6d" in error for error in result.errors)
    assert any("src.data.market_db_sync" in error for error in result.errors)


def _write_prices(path: Path, payload: dict) -> None:
    _write_json(path, payload)


def _load_data_quality_cli():
    script_path = PROJECT_ROOT / "scripts" / "check_public_data_quality.py"
    assert script_path.exists(), "scripts/check_public_data_quality.py is missing"
    spec = importlib.util.spec_from_file_location("check_public_data_quality", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_data_quality_cli_accepts_clean_app_dir_prices(tmp_path: Path, capsys) -> None:
    data_quality_cli = _load_data_quality_cli()
    _write_prices(
        tmp_path / "public" / "data" / "prices.json",
        {
            "SPY": [{"d": "2026-06-11", "p": 100.0}, {"d": "2026-06-12", "p": 101.0}],
            "GLD": [{"d": "2026-06-11", "p": 200.0}, {"d": "2026-06-12", "p": 201.0}],
        },
    )

    exit_code = data_quality_cli.main([
        "--app-dir",
        str(tmp_path),
        "--reference-date",
        "2026-06-12",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "price data quality check passed" in captured.out
    assert captured.err == ""


def test_public_data_quality_cli_rejects_duplicate_dates_and_writes_json_report(
    tmp_path: Path,
    capsys,
) -> None:
    data_quality_cli = _load_data_quality_cli()
    prices_path = tmp_path / "prices.json"
    report_path = tmp_path / "quality-report.json"
    _write_prices(
        prices_path,
        {
            "SPY": [
                {"d": "2026-06-12", "p": 100.0},
                {"d": "2026-06-12", "p": 101.0},
            ],
        },
    )

    exit_code = data_quality_cli.main([
        "--prices",
        str(prices_path),
        "--reference-date",
        "2026-06-12",
        "--json-report",
        str(report_path),
    ])

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert "duplicate_dates" in captured.err
    assert report["status"] == "fail"
    assert report["issue_counts"]["duplicate_dates"] == 1


def test_public_data_quality_cli_rejects_stale_symbols(tmp_path: Path, capsys) -> None:
    data_quality_cli = _load_data_quality_cli()
    prices_path = tmp_path / "prices.json"
    _write_prices(
        prices_path,
        {
            "SPY": [{"d": "2026-06-12", "p": 100.0}],
            "GLD": [{"d": "2026-06-01", "p": 200.0}],
        },
    )

    exit_code = data_quality_cli.main([
        "--prices",
        str(prices_path),
        "--reference-date",
        "2026-06-12",
        "--max-latest-lag-days",
        "3",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stale_latest_dates" in captured.err
    assert "GLD" in captured.err


def test_public_data_quality_cli_rejects_extreme_returns(tmp_path: Path, capsys) -> None:
    data_quality_cli = _load_data_quality_cli()
    prices_path = tmp_path / "prices.json"
    _write_prices(
        prices_path,
        {
            "SPY": [
                {"d": "2026-06-11", "p": 100.0},
                {"d": "2026-06-12", "p": 250.0},
            ],
        },
    )

    exit_code = data_quality_cli.main([
        "--prices",
        str(prices_path),
        "--reference-date",
        "2026-06-12",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "extreme_returns" in captured.err
    assert "SPY" in captured.err
