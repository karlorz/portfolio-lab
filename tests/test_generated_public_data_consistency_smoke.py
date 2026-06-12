"""Offline smoke tests for generated public data artifact consistency."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
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

    result = check_public_data_consistency(tmp_path)

    assert result.ok is True, result.errors


def test_generated_public_data_consistency_smoke_rejects_source_manifest_hash_drift(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path, source_hash="0" * 64)

    result = check_public_data_consistency(tmp_path)

    assert result.ok is False
    assert any(
        "public/data/index.json source_manifest.sha256 does not match public/data/source_manifest.json" in error
        for error in result.errors
    )


def test_generated_public_data_consistency_smoke_rejects_missing_required_artifact(tmp_path: Path) -> None:
    _write_consistent_public_data_set(tmp_path)
    (tmp_path / "public" / "data" / "health.json").unlink()

    result = check_public_data_consistency(tmp_path)

    assert result.ok is False
    assert "public/data/health.json is missing" in result.errors


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
