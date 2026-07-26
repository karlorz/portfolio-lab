"""Tests for data_pipeline_slo_section builder."""

from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import patch

from src.dashboard.data_pipeline_slo_section import (
    build_data_pipeline_slo_section,
    data_pipeline_slo_unavailable_payload,
)


def test_data_pipeline_slo_unavailable_payload_shape() -> None:
    payload = data_pipeline_slo_unavailable_payload(ImportError("boom"))
    assert payload["status"] == "warning"
    assert payload["top_dimension"] == "unknown"
    assert payload["schema_version"] == "data-pipeline-slo/v1"
    assert "boom" in payload["error"]


def test_build_data_pipeline_slo_section_success(tmp_path: Path) -> None:
    expected = {"schema_version": "data-pipeline-slo/v1", "status": "ok", "top_dimension": "scheduler"}
    with patch("src.monitor.data_pipeline_slo.load_rebalance_health", return_value={}) as rebal:
        with patch("src.monitor.data_pipeline_slo.load_source_manifest", return_value={}) as sm:
            with patch("src.monitor.data_pipeline_slo.load_public_index", return_value={}) as pi:
                with patch("src.monitor.data_pipeline_slo.load_signal_staleness", return_value={}) as ss:
                    with patch("src.monitor.data_pipeline_slo.build_data_pipeline_slo", return_value=expected) as build:
                        out = build_data_pipeline_slo_section(
                            health_data={"data_freshness": {}},
                            public_dir=tmp_path,
                            data_dir=tmp_path,
                        )
    assert out is expected
    rebal.assert_called_once_with(tmp_path)
    sm.assert_called_once_with(tmp_path)
    pi.assert_called_once_with(tmp_path)
    ss.assert_called_once_with(tmp_path)
    build.assert_called_once()
    # rebalance_health-derived kwargs flow through
    _, kwargs = build.call_args
    assert kwargs["alpaca_feed_entitlement"] is None
    assert kwargs["market_data_consistency"] is None
    assert kwargs["data_dir"] == tmp_path


def test_build_data_pipeline_slo_section_failure(tmp_path: Path) -> None:
    with patch("src.monitor.data_pipeline_slo.load_rebalance_health", side_effect=ImportError("no module")):
        out = build_data_pipeline_slo_section(health_data={}, public_dir=tmp_path)
    assert out["status"] == "warning"
    assert out["top_dimension"] == "unknown"
    assert "no module" in out["error"]


def test_build_data_pipeline_slo_section_log_error_invoked(tmp_path: Path) -> None:
    calls: list[tuple[str, Exception]] = []

    def log_error(name: str, exc: Exception) -> None:
        calls.append((name, exc))

    with patch("src.monitor.data_pipeline_slo.load_rebalance_health", side_effect=OSError("io fail")):
        out = build_data_pipeline_slo_section(
            health_data={}, public_dir=tmp_path, log_error=log_error
        )
    assert out["status"] == "warning"
    assert len(calls) == 1
    assert calls[0][0] == "data_pipeline_slo"
    assert isinstance(calls[0][1], OSError)


def test_build_data_pipeline_slo_section_uses_current_data_quality_artifact(tmp_path: Path) -> None:
    """Current data_quality.json dominates stale embedded source-manifest summaries."""
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "source_mode": "live",
                "status": "success",
                "symbols": ["SPY", "GLD"],
                "data_quality": {
                    "artifact": "data_quality.json",
                    "schema_version": "price-data-quality/v1",
                    "generated_at": "2026-06-01T00:00:00Z",
                    "status": "fail",
                    "issue_counts": {"stale_latest_dates": 2, "total": 2},
                },
            }
        ]
    }), encoding="utf-8")
    (tmp_path / "data_quality.json").write_text(json.dumps({
        "artifact": "data_quality.json",
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-06-16T12:00:00Z",
        "status": "ok",
        "issue_counts": {"stale_latest_dates": 0, "total": 0},
    }), encoding="utf-8")

    out = build_data_pipeline_slo_section(
        health_data={
            "cron_jobs": [],
            "scheduler_status": {"status": "ok", "backends": {}},
            "data_freshness": {},
        },
        public_dir=tmp_path,
    )

    data_quality = out["dimensions"]["data_quality"]
    assert data_quality["status"] == "ok"
    assert data_quality["quality_status"] == "ok"
    assert data_quality["generated_at"] == "2026-06-16T12:00:00Z"
    assert data_quality["issue_counts"]["stale_latest_dates"] == 0


def test_build_data_pipeline_slo_section_accepts_current_overall_status_shape(tmp_path: Path) -> None:
    """Standalone data_quality.json uses overall_status in generated artifacts."""
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "source_mode": "live",
                "status": "success",
                "symbols": ["SPY", "GLD"],
            }
        ]
    }), encoding="utf-8")
    (tmp_path / "data_quality.json").write_text(json.dumps({
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-07-06T11:05:20.437Z",
        "overall_status": "warn",
        "issue_counts": {"split_like_returns": 4, "stale_latest_dates": 0, "total": 4},
    }), encoding="utf-8")

    out = build_data_pipeline_slo_section(
        health_data={
            "cron_jobs": [],
            "scheduler_status": {"status": "ok", "backends": {}},
            "data_freshness": {},
        },
        public_dir=tmp_path,
    )

    data_quality = out["dimensions"]["data_quality"]
    # split-like-only overall_status=warn is advisory (not a pipeline fail).
    assert data_quality["status"] == "ok"
    assert data_quality["quality_status"] == "warn"
    assert data_quality["generated_at"] == "2026-07-06T11:05:20.437Z"
    assert data_quality["issue_counts"]["split_like_returns"] == 4
