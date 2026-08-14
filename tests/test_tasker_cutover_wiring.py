import json
from pathlib import Path
from unittest.mock import patch


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_docker_entrypoint_starts_tasker_by_default_with_cron_fallback():
    source = _read("scripts/docker-entrypoint.sh")

    assert "TASKER_ENTRYPOINT_MODE" in source
    assert "TASKER_ENTRYPOINT_MODE:-tasker" in source
    assert "/app/scripts/python_runtime.sh -m src.tasker.service" in source
    assert "exec cron -f" in source
    assert "TASKER_ENTRYPOINT_MODE=cron" in source


def test_dockerfile_defaults_to_tasker_and_healthchecks_tasker_api():
    source = _read("Dockerfile")

    assert "ENV CRON_BACKEND=tasker" in source
    assert "http://127.0.0.1:8000/api/tasker/status" in source
    assert "pgrep -x cron" not in source


def test_compose_runs_tasker_backend_and_healthchecks_tasker_api():
    source = _read("docker-compose.yml")

    assert "CRON_BACKEND=tasker" in source
    assert "http://localhost:8000/api/tasker/status" in source
    assert "data/signals/signals.json" not in source


def test_cron_compat_recognizes_tasker_backend():
    with patch.dict("os.environ", {"CRON_BACKEND": "tasker"}, clear=True):
        import importlib
        import src.cron_compat as cc

        importlib.reload(cc)
        assert cc.BACKEND == "tasker"
        assert cc.IS_TASKER is True
        assert cc.IS_HERMES is False
        assert cc.IS_CRONTAB is False
        assert cc.IS_MANUAL is False


def test_tasker_owned_cron_status_is_normalized_as_tasker_backend(tmp_path):
    from src.monitor.hermes_cron import load_local_cron_jobs

    status_path = tmp_path / "cron_status.json"
    status_path.write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-health",
                        "status": "success",
                        "backend": "tasker",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    jobs, backend = load_local_cron_jobs(status_path)

    assert backend["backend"] == "tasker"
    assert backend["status"] == "ok"
    assert jobs[0]["backend"] == "tasker"


def test_cron_normalization_truncates_unbounded_log_fields():
    from src.monitor.hermes_cron import CRON_FIELD_PREVIEW_CHARS, normalize_cron_job

    oversized = "x" * (CRON_FIELD_PREVIEW_CHARS + 100)
    job = {
        "name": "portfolio-lab-health",
        "last_status": "failed",
        "error": oversized,
        "stdout": oversized,
        "stderr": oversized,
    }

    normalized = normalize_cron_job(job, backend="tasker", source="unit-test", index=0)

    for field in ("error", "stdout", "stderr"):
        assert len(normalized[field]) <= CRON_FIELD_PREVIEW_CHARS
        assert normalized[f"{field}_truncated"] is True
        assert normalized[f"{field}_original_length"] == len(oversized)


def test_tasker_health_ignores_hermes_by_default_when_tasker_active(tmp_path, monkeypatch):
    from src.monitor.health_check import _check_data_freshness

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
    # _check_data_freshness resolves DATA_DIR/PUBLIC_DATA_DIR in
    # health_freshness_cb (post HEALTH-CHECK-SPLIT); the hub PUBLIC_DATA_DIR
    # binding is not read on this path.
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", tmp_path)
    monkeypatch.setenv("CRON_BACKEND", "tasker")
    monkeypatch.delenv("TASKER_INCLUDE_HERMES_AUDIT", raising=False)

    (tmp_path / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [{"name": "portfolio-lab-health", "status": "success", "backend": "tasker"}],
            }
        ),
        encoding="utf-8",
    )
    hermes_jobs = tmp_path / "hermes_jobs.json"
    hermes_jobs.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-hermes-job",
                        "name": "portfolio-lab-health",
                        "last_status": "error",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

    freshness = _check_data_freshness()

    assert freshness["cron"]["status"] == "ok"
    assert freshness["cron"]["failed_jobs"] == 0
    assert set(freshness["cron"]["backends"]) == {"tasker"}


def test_tasker_health_can_include_hermes_audit_when_explicitly_enabled(tmp_path, monkeypatch):
    from src.monitor.health_check import _check_data_freshness

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", tmp_path)
    # _check_data_freshness resolves DATA_DIR/PUBLIC_DATA_DIR in
    # health_freshness_cb (post HEALTH-CHECK-SPLIT); the hub PUBLIC_DATA_DIR
    # binding is not read on this path.
    monkeypatch.setattr("src.monitor.health_freshness_cb.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.monitor.health_freshness_cb.PUBLIC_DATA_DIR", tmp_path)
    monkeypatch.setenv("CRON_BACKEND", "tasker")
    monkeypatch.setenv("TASKER_INCLUDE_HERMES_AUDIT", "1")

    (tmp_path / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [{"name": "portfolio-lab-health", "status": "success", "backend": "tasker"}],
            }
        ),
        encoding="utf-8",
    )
    hermes_jobs = tmp_path / "hermes_jobs.json"
    hermes_jobs.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-hermes-job",
                        # Non-self job: portfolio-lab-health errors are excluded
                        # from rollup so sticky health mirrors do not degrade forever.
                        "name": "portfolio-lab-dashboard",
                        "last_status": "error",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

    freshness = _check_data_freshness()

    assert freshness["cron"]["status"] == "degraded"
    assert set(freshness["cron"]["backends"]) == {"tasker", "hermes"}
