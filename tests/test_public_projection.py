"""Public logical-reference and fan-out disclosure contracts."""

from __future__ import annotations

import json
from pathlib import Path

from src.paths import PROJECT_ROOT


PROJECT_DATA_DIR = PROJECT_ROOT / "data"


def test_public_projection_rewrites_known_diagnostic_planes_without_business_drift() -> None:
    from src.dashboard.public_projection import (
        find_public_internal_paths,
        project_public_paths,
    )

    private = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "history_source": str(PROJECT_DATA_DIR / "regime_log.json"),
        "scheduler_source": "/root/.hermes/cron/jobs.json",
        "log_path": str(PROJECT_DATA_DIR / "tasker_logs" / "run-abc.log"),
        "provenance_completeness": {
            "private_path": str(PROJECT_DATA_DIR / "signals.json"),
            "public_path": "/var/www/portfolio-lab/data/signals.json",
        },
    }

    public = project_public_paths(private)

    assert public["target_allocations"] == private["target_allocations"]
    assert public["history_source"] == "data/regime_log.json"
    assert public["scheduler_source"] == "scheduler/hermes/cron/jobs.json"
    assert public["log_path"] == "tasker/logs/run-abc.log"
    assert public["provenance_completeness"] == {
        "private_path": "data/signals.json",
        "public_path": "data/signals.json",
    }
    assert find_public_internal_paths(public) == []


def test_public_business_values_ignore_only_approved_metadata() -> None:
    from src.dashboard.public_projection import (
        public_business_values_equal,
        project_public_business_values,
    )

    private = {
        "status": "ok",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "timestamp": "2026-07-31T12:00:00+00:00",
        "runtime_provenance": {"plane": "private", "generator_git_sha": "private"},
        "provenance_completeness": {
            "private_path": str(PROJECT_DATA_DIR / "signals.json"),
            "public_path": "/var/www/portfolio-lab/data/signals.json",
        },
    }
    public = {
        "status": "ok",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "timestamp": "2026-07-31T12:01:00+00:00",
        "runtime_provenance": {"plane": "public", "generator_git_sha": "public"},
        "provenance_completeness": {
            "private_path": "data/signals.json",
            "public_path": "data/signals.json",
        },
    }

    assert public_business_values_equal(private, public)
    assert project_public_business_values(private) == {
        "status": "ok",
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    }

    public["target_allocations"]["SPY"] = 0.45
    assert not public_business_values_equal(private, public)


def test_public_path_gate_finds_nested_and_key_paths() -> None:
    from src.dashboard.public_projection import find_public_internal_paths

    payload = {
        f"files.{PROJECT_DATA_DIR / 'dashboard.json'}": {
            "path": "/var/www/portfolio-lab/data/dashboard.json",
        }
    }

    offenders = find_public_internal_paths(payload)

    assert len(offenders) == 2
    assert any("dashboard.json" in value for _, value in offenders)


def test_public_fanout_projects_only_public_body(tmp_path: Path, monkeypatch) -> None:
    from src.monitor.signal_authority import write_json_multi_dest

    monkeypatch.setenv("PORTFOLIO_LAB_FORCE_PUBLIC_PROJECTION", "1")
    private_path = tmp_path / "private" / "health.json"
    public_path = tmp_path / "public" / "health.json"
    payload = {
        "status": "healthy",
        "source": str(PROJECT_DATA_DIR / "cron_status.json"),
    }

    result = write_json_multi_dest(
        payload,
        private_path=private_path,
        public_path=public_path,
        soft_mirror_repo=False,
    )

    assert result.wrote_private and result.wrote_public
    private = json.loads(private_path.read_text())
    public = json.loads(public_path.read_text())
    assert private["source"] == str(PROJECT_DATA_DIR / "cron_status.json")
    assert public["source"] == "data/cron_status.json"
    assert private["status"] == public["status"]


def test_public_mirror_expands_indexed_nested_artifacts(tmp_path: Path) -> None:
    from scripts.mirror_repo_public_data import mirror_repo_public_data

    source = tmp_path / "live"
    destination = tmp_path / "repo"
    (source / "attribution").mkdir(parents=True)
    (source / "attribution" / "latest.json").write_text(
        json.dumps({"artifact": "latest"}), encoding="utf-8"
    )
    (source / "attribution" / "attribution_2026-07-31.json").write_text(
        json.dumps({"artifact": "dated"}), encoding="utf-8"
    )

    report = mirror_repo_public_data(
        source_root=source,
        dest_root=destination,
        files=("attribution/*.json",),
        restamp_health_lag=False,
    )

    assert report.errors == []
    assert set(report.copied) == {
        "attribution/latest.json",
        "attribution/attribution_2026-07-31.json",
    }
    assert (destination / "attribution" / "latest.json").is_file()
    assert (destination / "attribution" / "attribution_2026-07-31.json").is_file()


def test_public_mirror_lag_restamp_keeps_public_paths_logical(
    tmp_path: Path, monkeypatch
) -> None:
    from src.monitor.repo_public_mirror_lag import (
        restamp_mirror_lag_on_health_documents,
    )

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(public))
    health = public / "health.json"
    health.write_text(
        json.dumps(
            {
                "status": "ok",
                "repo_public_mirror_lagging_count": 0,
                "repo_public_mirror_total": 1,
                "repo_public_mirror_source": "/var/www/portfolio-lab/data",
                "repo_public_mirror_dest": "/root/projects/portfolio-lab/public/data",
                "repo_public_mirror_lag": {
                    "source": "/var/www/portfolio-lab/data",
                    "dest": "/root/projects/portfolio-lab/public/data",
                },
            }
        ),
        encoding="utf-8",
    )

    result = restamp_mirror_lag_on_health_documents(
        paths=[health],
        lag_summary={
            "lagging_count": 0,
            "total": 1,
            "lagging_paths": [],
            "source": "/var/www/portfolio-lab/data",
            "dest": "/root/projects/portfolio-lab/public/data",
        },
    )

    assert result["errors"] == []
    written = json.loads(health.read_text())
    assert written["repo_public_mirror_source"] == "data/."
    assert written["repo_public_mirror_dest"] == "data/."
    assert written["repo_public_mirror_lag"]["source"] == "data/."


def test_decision_registry_public_projection_preserves_private_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    from src.monitor.decision_registry import (
        DecisionRegistry,
        ExperimentRecord,
        publish_decision_registry_json,
    )
    from src.dashboard.public_projection import find_public_internal_paths

    monkeypatch.setenv("PORTFOLIO_LAB_FORCE_PUBLIC_PROJECTION", "1")
    private = tmp_path / "private"
    public = tmp_path / "public"
    registry = DecisionRegistry(db_path=tmp_path / "registry.db")
    registry.record_experiment(
        ExperimentRecord(
            experiment_id="projection-regression",
            timestamp_utc="2026-07-31T00:00:00+00:00",
            name="projection regression",
            artifacts={
                "output_path": str(PROJECT_DATA_DIR / ".promote_to_live")
            },
        )
    )

    publish_decision_registry_json(
        public_dir=public,
        private_dir=private,
        registry=registry,
    )

    public_payload = json.loads((public / "decision_registry.json").read_text())
    private_payload = json.loads((private / "decision_registry.json").read_text())
    public_path = public_payload["recent_experiments"][0]["artifacts"]["output_path"]
    private_path = private_payload["recent_experiments"][0]["artifacts"]["output_path"]
    assert public_path == "data/.promote_to_live"
    assert private_path == str(PROJECT_DATA_DIR / ".promote_to_live")
    assert find_public_internal_paths(public_payload) == []
