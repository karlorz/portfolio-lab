"""Unit tests for src.monitor.repo_public_mirror_lag (Item Q47).

Tests cover:
- load_mirror_script_module
- summarize_repo_public_mirror_lag (clean match, missing dest, divergent files, error handling)
- ops_dimensions_block_lag_heal
- rederive_ops_status_for_lag_heal
- apply_lag_summary_to_health_doc (top-level, nested, with/without status elevation)
- is_ephemeral_restamp_path
- restamp_mirror_lag_on_health_documents (paths, skipped, error handling, nested-only signals)
- resolve_mirror_lag_for_consumer (max(live, stamp) honesty, thresholds)
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.monitor import repo_public_mirror_lag as rpml


def test_load_mirror_script_module() -> None:
    mod = rpml.load_mirror_script_module()
    assert hasattr(mod, "lag_sli_report")
    assert hasattr(mod, "lag_report")


def test_is_ephemeral_restamp_path(tmp_path: Path) -> None:
    assert rpml.is_ephemeral_restamp_path("/tmp/plab-pytest-public.abc123/data") is True
    assert rpml.is_ephemeral_restamp_path("/tmp/pytest-of-root/test0") is True
    assert rpml.is_ephemeral_restamp_path(tmp_path / "fixture.json") is True
    assert rpml.is_ephemeral_restamp_path("/var/www/portfolio-lab/data/health.json") is False
    assert rpml.is_ephemeral_restamp_path(None) is False
    assert rpml.is_ephemeral_restamp_path("") is False


def test_resolve_mirror_lag_for_consumer() -> None:
    # Live worse than stamp
    res1 = rpml.resolve_mirror_lag_for_consumer(
        stamp={"lagging_count": 0, "total": 10},
        live={"lagging_count": 5, "total": 10, "lagging_paths": ["a.json"]},
        warn_threshold=1,
        critical_threshold=10,
    )
    assert res1["lagging_count"] == 5
    assert res1["repo_public_mirror_lag_status"] == "lagging"
    assert res1["source_of_truth"] == "live"
    assert res1["lagging_paths"] == ["a.json"]

    # Stamp worse than live (max policy)
    res2 = rpml.resolve_mirror_lag_for_consumer(
        stamp={"lagging_count": 12, "total": 10, "lagging_paths": ["b.json"]},
        live={"lagging_count": 0, "total": 10},
        warn_threshold=1,
        critical_threshold=10,
    )
    assert res2["lagging_count"] == 12
    assert res2["repo_public_mirror_lag_status"] == "critical"
    assert res2["source_of_truth"] == "stamp"
    assert res2["lagging_paths"] == ["b.json"]

    # Both clean
    res3 = rpml.resolve_mirror_lag_for_consumer(
        stamp={"lagging_count": 0, "total": 5},
        live={"lagging_count": 0, "total": 5},
    )
    assert res3["lagging_count"] == 0
    assert res3["repo_public_mirror_lag_status"] == "ok"

    # Both empty / 0 total
    res4 = rpml.resolve_mirror_lag_for_consumer(
        stamp={},
        live={},
    )
    assert res4["lagging_count"] == 0
    assert res4["repo_public_mirror_lag_status"] == "unknown"


def test_summarize_repo_public_mirror_lag(tmp_path: Path) -> None:
    src_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    dest_dir.mkdir()

    # Empty dirs
    summary0 = rpml.summarize_repo_public_mirror_lag(source_root=src_dir, dest_root=dest_dir)
    assert summary0["ok"] is True
    assert summary0["lagging_count"] == 0
    assert summary0["total"] == 0

    # Matching files
    (src_dir / "prices.json").write_text('{"SPY": 100}')
    (dest_dir / "prices.json").write_text('{"SPY": 100}')
    summary1 = rpml.summarize_repo_public_mirror_lag(source_root=src_dir, dest_root=dest_dir)
    assert summary1["ok"] is True
    assert summary1["lagging_count"] == 0
    assert summary1["total"] == 1

    # Divergent files
    (src_dir / "prices.json").write_text('{"SPY": 105}')
    summary2 = rpml.summarize_repo_public_mirror_lag(source_root=src_dir, dest_root=dest_dir)
    assert summary2["ok"] is True
    assert summary2["lagging_count"] == 1
    assert "prices.json" in summary2["lagging_paths"]


def test_ops_dimensions_block_lag_heal() -> None:
    # Clean doc does not block
    clean_doc = {
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": "ok",
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lag_status": "ok",
        "repo_public_mirror_lagging_count": 0,
    }
    assert rpml.ops_dimensions_block_lag_heal(clean_doc) is False

    # Kill switch enabled blocks
    assert rpml.ops_dimensions_block_lag_heal({"kill_switch": {"enabled": True}}) is True
    assert rpml.ops_dimensions_block_lag_heal({"checks": {"kill_switch": {"enabled": True}}}) is True

    # Open incidents block
    assert rpml.ops_dimensions_block_lag_heal({"open_incidents": {"open_count": 1}}) is True
    assert rpml.ops_dimensions_block_lag_heal({"checks": {"open_incidents": {"open_count": 2}}}) is True

    # Degraded scheduler blocks
    assert rpml.ops_dimensions_block_lag_heal({"scheduler_status": "failed"}) is True

    # Degraded SLO blocks
    assert rpml.ops_dimensions_block_lag_heal({"data_pipeline_slo": {"status": "degraded"}}) is True

    # Active lag blocks
    assert rpml.ops_dimensions_block_lag_heal({"repo_public_mirror_lagging_count": 3}) is True
    assert rpml.ops_dimensions_block_lag_heal({"repo_public_mirror_lag_status": "lagging"}) is True


def test_rederive_ops_status_for_lag_heal() -> None:
    # Blocked by kill switch -> no updates
    doc_blocked = {"kill_switch": {"enabled": True}, "status": "critical"}
    assert rpml.rederive_ops_status_for_lag_heal(doc_blocked) == {}

    # Monitor schema clean
    doc_monitor = {
        "status": "warning",
        "ops_health_status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": "ok",
    }
    updates = rpml.rederive_ops_status_for_lag_heal(doc_monitor)
    assert updates["status"] == "ok"
    assert updates["ops_health_status"] == "ok"

    # Dashboard schema clean
    doc_dashboard = {
        "system_status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": "ok",
        "failed_cron_jobs": 0,
    }
    updates_dash = rpml.rederive_ops_status_for_lag_heal(doc_dashboard)
    assert updates_dash["system_status"] == "healthy"


def test_apply_lag_summary_to_health_doc() -> None:
    doc = {
        "status": "warning",
        "system_status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
    }
    lag_summary = {
        "lagging_count": 0,
        "total": 10,
        "lagging_paths": [],
        "source": "/var/www/data",
        "dest": "/root/projects/public/data",
        "ok": True,
    }

    # With elevate_status = True and 0 lag -> demotes to ok / healthy
    res = rpml.apply_lag_summary_to_health_doc(doc, lag_summary, elevate_status=True)
    assert res["repo_public_mirror_lagging_count"] == 0
    assert res["repo_public_mirror_lag"]["lagging_count"] == 0
    assert res["mirror_lag_source_of_truth"] == "live"
    assert res["mirror_lag_live_lagging_count"] == 0
    assert res["status"] == "ok"
    assert res["system_status"] == "healthy"

    # With elevate_status = False -> status untouched
    doc2 = {"status": "warning", "system_status": "warning"}
    res2 = rpml.apply_lag_summary_to_health_doc(doc2, lag_summary, elevate_status=False)
    assert res2["status"] == "warning"


def test_restamp_mirror_lag_on_health_documents(tmp_path: Path) -> None:
    # 1. Top-level health.json in tmp
    health_file = tmp_path / "health.json"
    health_file.write_text(
        json.dumps({
            "status": "ok",
            "repo_public_mirror_lagging_count": 5,
            "repo_public_mirror_lag_status": "lagging",
            "kill_switch": {"enabled": False},
            "open_incidents": {"open_count": 0},
        })
    )

    # 2. Nested-only signals.json in tmp
    signals_file = tmp_path / "signals.json"
    signals_file.write_text(
        json.dumps({
            "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "health": {
                "repo_public_mirror_lagging_count": 5,
                "repo_public_mirror_lag_status": "lagging",
            },
        })
    )

    lag_summary = {
        "lagging_count": 0,
        "total": 12,
        "lagging_paths": [],
        "source": str(tmp_path / "src"),
        "dest": str(tmp_path / "dst"),
        "ok": True,
    }

    result = rpml.restamp_mirror_lag_on_health_documents(
        paths=[health_file, signals_file],
        lag_summary=lag_summary,
    )

    assert "health.json" in result["restamped"]
    assert "signals.json" in result["restamped"]
    assert len(result["errors"]) == 0

    # Verify health_file was updated
    h_data = json.loads(health_file.read_text())
    assert h_data["repo_public_mirror_lagging_count"] == 0
    assert "mirror_lag_restamped_at" in h_data

    # Verify signals_file nested health was updated while target_allocations was preserved
    s_data = json.loads(signals_file.read_text())
    assert s_data["health"]["repo_public_mirror_lagging_count"] == 0
    assert s_data["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    assert "mirror_lag_restamped_at" in s_data["health"]
