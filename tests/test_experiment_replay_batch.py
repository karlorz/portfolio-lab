"""Tests for Labs replay batch reporting."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.research.experiment_artifact_validator import LABS_REPLAY_SCHEMA_VERSION, validate_artifact
from src.research.experiment_replay_batch import main, publish_labs_replays, run_replay_batch

EXPECTED_METRICS = {
    "sharpe": 0.95,
    "cagr_pct": 10.4,
    "max_drawdown_pct": -25.0,
}


def _write_fixture_experiment(
    tmp_path: Path,
    metrics: dict[str, float],
    *,
    marker: Path | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    script_path = tmp_path / f"fixture_experiment_{len(list(tmp_path.glob('fixture_experiment_*.py')))}.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import sys",
                f"marker = {str(marker)!r}",
                "if marker:",
                "    Path(marker).write_text('ran')",
                *(extra_lines or []),
                f"print(json.dumps({{'metrics': {json.dumps(metrics)}}}))",
            ]
        )
    )
    return f"{sys.executable} {script_path}"


def _artifact(tmp_path: Path, name: str = "saved_experiment.json") -> Path:
    artifact_path = tmp_path / name
    artifact_path.write_text(json.dumps({"metrics": EXPECTED_METRICS}))
    return artifact_path


def _target(tmp_path: Path, experiment_id: str, command: str, **overrides) -> dict:
    target = {
        "experiment_id": experiment_id,
        "artifact_path": str(_artifact(tmp_path, f"{experiment_id}.json")),
        "status": "candidate",
        "provenance_status": "present",
        "metrics": EXPECTED_METRICS,
        "baseline_deltas": {},
        "command": command,
        "replay_safe": True,
        "fetches_market_data": False,
        "metric_tolerances": {
            "sharpe": 0.01,
            "cagr_pct": 0.1,
            "max_drawdown_pct": 0.1,
        },
    }
    target.update(overrides)
    return target


def _registry(path: Path, rows: list[dict]) -> Path:
    registry_path = path / "labs_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "labs-registry/v1",
                "generated_at": "2026-06-08T12:00:00+00:00",
                "experiments": rows,
            }
        )
    )
    return registry_path


def _registry_at(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "labs-registry/v1",
                "generated_at": "2026-06-08T12:00:00+00:00",
                "experiments": rows,
            }
        )
    )
    return path


def test_run_replay_batch_writes_pass_fail_and_skipped_rows_without_unsafe_commands(tmp_path: Path) -> None:
    marker = tmp_path / "unsafe-ran.txt"
    registry_path = _registry(
        tmp_path,
        [
            _target(tmp_path, "pass-target", _write_fixture_experiment(tmp_path, EXPECTED_METRICS)),
            _target(
                tmp_path,
                "fail-target",
                _write_fixture_experiment(tmp_path, {**EXPECTED_METRICS, "sharpe": 0.80}),
            ),
            _target(
                tmp_path,
                "unsafe-target",
                _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=marker),
                replay_safe=False,
            ),
        ],
    )
    output_path = tmp_path / "public" / "data" / "labs_replays.json"

    result = run_replay_batch([registry_path], output_path=output_path, project_root=tmp_path)

    assert output_path.exists()
    assert marker.exists() is False
    assert result.counts == {"passed": 1, "failed": 1, "warning": 1}
    rows = json.loads(output_path.read_text())
    statuses = {row["experiment_id"]: row["status"] for row in rows}
    assert statuses == {
        "pass-target": "passed",
        "fail-target": "failed",
        "unsafe-target": "warning",
    }
    assert rows[2]["metrics"]["skipped_count"] == 1
    assert rows[2]["metrics"]["safety_skip_count"] == 1
    for row in rows:
        assert validate_artifact(row).valid


def test_publish_labs_replays_noops_without_explicit_targets_or_precomputed_report(tmp_path: Path) -> None:
    output_path = publish_labs_replays(data_dir=tmp_path / "data", public_dir=tmp_path / "public" / "data")

    assert output_path is None
    assert (tmp_path / "public" / "data" / "labs_replays.json").exists() is False


def test_publish_labs_replays_copies_valid_precomputed_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    public_dir = tmp_path / "public" / "data"
    precomputed = data_dir / "labs_replays.json"
    precomputed.parent.mkdir(parents=True, exist_ok=True)
    precomputed.write_text(
        json.dumps(
            [
                {
                    "schema_version": "labs-replay/v1",
                    "experiment_id": "precomputed-replay",
                    "generated_at": "2026-06-09T00:00:00+00:00",
                    "artifact_path": "data/precomputed-replay.json",
                    "status": "passed",
                    "provenance_status": "sidecar",
                    "passed": True,
                    "metrics": {"metrics_compared": 3},
                    "baseline_deltas": {},
                }
            ]
        )
    )

    output_path = publish_labs_replays(data_dir=data_dir, public_dir=public_dir)

    assert output_path == public_dir / "labs_replays.json"
    rows = json.loads(output_path.read_text())
    assert rows[0]["experiment_id"] == "precomputed-replay"
    assert validate_artifact(rows[0]).valid


def test_run_replay_batch_adds_safety_skip_diagnostics(tmp_path: Path) -> None:
    marker = tmp_path / "unsafe-ran.txt"
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "unsafe-target",
                _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=marker),
                replay_safe=False,
            ),
        ],
    )

    result = run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    row = result.rows[0]
    assert marker.exists() is False
    assert row["status"] == "warning"
    assert row["failure_reason"] == "safety_skip"
    assert row["error_type"] == "ReplaySafetyError"
    assert "unsafe-target is not marked replay_safe" in row["error_message"]
    assert validate_artifact(row).valid


def test_run_replay_batch_adds_timeout_diagnostics(tmp_path: Path) -> None:
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "timeout-target",
                _write_fixture_experiment(
                    tmp_path,
                    EXPECTED_METRICS,
                    extra_lines=["import time", "time.sleep(1.0)"],
                ),
            ),
        ],
    )

    result = run_replay_batch(
        [registry_path],
        output_path=tmp_path / "labs_replays.json",
        project_root=tmp_path,
        timeout_seconds=0.01,
    )

    row = result.rows[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "timeout"
    assert row["error_type"] == "TimeoutExpired"
    assert "timed out" in row["error_message"]
    assert validate_artifact(row).valid


def test_run_replay_batch_adds_validation_failure_diagnostics(tmp_path: Path) -> None:
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "metric-mismatch",
                _write_fixture_experiment(tmp_path, {**EXPECTED_METRICS, "sharpe": 0.80}),
            ),
        ],
    )

    result = run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    row = result.rows[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "validation_failure"
    assert row["error_type"] == "ReplayValidationFailure"
    assert "sharpe" in row["error_message"]
    assert validate_artifact(row).valid


def test_run_replay_batch_adds_command_failure_diagnostics(tmp_path: Path) -> None:
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "command-failure",
                _write_fixture_experiment(
                    tmp_path,
                    EXPECTED_METRICS,
                    extra_lines=["sys.stderr.write('password=supersecret should stay private')", "raise SystemExit(7)"],
                ),
            ),
        ],
    )

    result = run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    row = result.rows[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "command_failure"
    assert row["error_type"] == "ReplayCommandFailure"
    assert row["error_message"] == "replay command exited with return code 7"
    assert "supersecret" not in json.dumps(row)
    assert validate_artifact(row).valid


def test_run_replay_batch_adds_sanitized_unexpected_exception_diagnostics(tmp_path: Path) -> None:
    registry_path = _registry(
        tmp_path,
        [
            {
                "experiment_id": "unexpected-error",
                "artifact_path": str(_artifact(tmp_path, "unexpected-error.json")),
                "status": "candidate",
                "provenance_status": "present",
                "metrics": EXPECTED_METRICS,
                "baseline_deltas": {},
                "replay_safe": True,
                "fetches_market_data": False,
            }
        ],
    )

    result = run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    row = result.rows[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "unexpected_error"
    assert row["error_type"] == "ValueError"
    assert row["error_message"] == "target.command is required"
    assert validate_artifact(row).valid


def test_run_replay_batch_defaults_to_sequential_execution(tmp_path: Path) -> None:
    first_done = tmp_path / "first-done.txt"
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "first-sequential-target",
                _write_fixture_experiment(
                    tmp_path,
                    EXPECTED_METRICS,
                    extra_lines=[
                        "import time",
                        "time.sleep(0.1)",
                        f"Path({str(first_done)!r}).write_text('done')",
                    ],
                ),
            ),
            _target(
                tmp_path,
                "second-sequential-target",
                _write_fixture_experiment(
                    tmp_path,
                    EXPECTED_METRICS,
                    extra_lines=[
                        f"first_done = Path({str(first_done)!r})",
                        "if not first_done.exists():",
                        "    raise SystemExit(9)",
                    ],
                ),
            ),
        ],
    )

    result = run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    assert [row["experiment_id"] for row in result.rows] == [
        "first-sequential-target",
        "second-sequential-target",
    ]
    assert result.counts == {"passed": 2, "failed": 0, "warning": 0}


def test_run_replay_batch_runs_safe_targets_concurrently_in_input_order(tmp_path: Path) -> None:
    delays = {
        "slow-first": 0.6,
        "fast-second": 0.2,
        "fast-third": 0.2,
        "fast-fourth": 0.2,
    }
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                experiment_id,
                _write_fixture_experiment(
                    tmp_path,
                    EXPECTED_METRICS,
                    extra_lines=["import time", f"time.sleep({delay})"],
                ),
            )
            for experiment_id, delay in delays.items()
        ],
    )

    start = time.perf_counter()
    result = run_replay_batch(
        [registry_path],
        output_path=tmp_path / "labs_replays.json",
        project_root=tmp_path,
        concurrency=2,
    )
    elapsed = time.perf_counter() - start

    assert [row["experiment_id"] for row in result.rows] == list(delays)
    assert result.counts == {"passed": 4, "failed": 0, "warning": 0}
    assert elapsed < 1.05
    assert all(validate_artifact(row).valid for row in result.rows)


def test_run_replay_batch_concurrency_does_not_spawn_unsafe_targets(tmp_path: Path) -> None:
    unsafe_marker = tmp_path / "unsafe-ran.txt"
    registry_path = _registry(
        tmp_path,
        [
            _target(
                tmp_path,
                "safe-target",
                _write_fixture_experiment(tmp_path, EXPECTED_METRICS),
            ),
            _target(
                tmp_path,
                "unsafe-target",
                _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=unsafe_marker),
                replay_safe=False,
            ),
        ],
    )

    result = run_replay_batch(
        [registry_path],
        output_path=tmp_path / "labs_replays.json",
        project_root=tmp_path,
        concurrency=2,
    )

    assert unsafe_marker.exists() is False
    assert [row["experiment_id"] for row in result.rows] == ["safe-target", "unsafe-target"]
    assert result.counts == {"passed": 1, "failed": 0, "warning": 1}
    assert result.rows[1]["failure_reason"] == "safety_skip"


def test_run_replay_batch_rejects_non_positive_concurrency(tmp_path: Path) -> None:
    registry_path = _registry(
        tmp_path,
        [_target(tmp_path, "cli-pass", _write_fixture_experiment(tmp_path, EXPECTED_METRICS))],
    )

    try:
        run_replay_batch([registry_path], output_path=tmp_path / "labs_replays.json", concurrency=0)
    except ValueError as exc:
        assert "concurrency must be a positive integer" in str(exc)
    else:
        raise AssertionError("expected concurrency=0 to fail")


def test_labs_replay_validator_accepts_optional_diagnostics() -> None:
    result = validate_artifact(
        {
            "schema_version": LABS_REPLAY_SCHEMA_VERSION,
            "experiment_id": "diagnostic-target",
            "generated_at": "2026-06-08T12:00:00+00:00",
            "artifact_path": "data/diagnostic-target.json",
            "status": "failed",
            "provenance_status": "present",
            "passed": False,
            "command": "python script.py",
            "duration_seconds": 0.0,
            "metric_deltas": {},
            "metrics": {"error_count": 1},
            "baseline_deltas": {},
            "failure_reason": "unexpected_error",
            "error_type": "ValueError",
            "error_message": "target.command is required",
        }
    )

    assert result.valid, result.error_messages()


def test_run_replay_batch_accepts_provenance_manifest_input(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path, "manifest_saved_experiment.json")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "experiment-manifest/v1",
                "experiment_id": "manifest-target",
                "generated_at": "2026-06-08T12:00:00+00:00",
                "source_artifact_path": str(artifact_path),
                "command": _write_fixture_experiment(tmp_path, EXPECTED_METRICS),
                "module": "tests.fixture",
                "git": {},
                "config_snapshot": {},
                "environment": {},
                "input_file_hashes": {},
                "freeze_manifest": {"config": {}, "file_hashes": {}, "file_count": 0},
                "replay_safe": True,
                "provenance_status": "sidecar",
            }
        )
    )

    result = run_replay_batch([manifest_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    assert result.counts == {"passed": 1, "failed": 0, "warning": 0}
    row = json.loads((tmp_path / "labs_replays.json").read_text())[0]
    assert row["experiment_id"] == "manifest-target"
    assert row["status"] == "passed"
    assert validate_artifact(row).valid


def test_run_replay_batch_deduplicates_targets_across_registry_files_with_first_wins(tmp_path: Path) -> None:
    first_marker = tmp_path / "first-ran.txt"
    duplicate_marker = tmp_path / "duplicate-ran.txt"
    first_registry = _registry_at(
        tmp_path / "first_registry.json",
        [_target(tmp_path, "duplicate-target", _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=first_marker))],
    )
    duplicate_registry = _registry_at(
        tmp_path / "duplicate_registry.json",
        [
            _target(
                tmp_path,
                "duplicate-target",
                _write_fixture_experiment(tmp_path, {**EXPECTED_METRICS, "sharpe": 0.10}, marker=duplicate_marker),
            )
        ],
    )

    result = run_replay_batch(
        [first_registry, duplicate_registry],
        output_path=tmp_path / "labs_replays.json",
        project_root=tmp_path,
    )

    rows = json.loads((tmp_path / "labs_replays.json").read_text())
    assert [row["experiment_id"] for row in rows] == ["duplicate-target"]
    assert rows[0]["status"] == "passed"
    assert first_marker.exists() is True
    assert duplicate_marker.exists() is False
    assert result.duplicate_targets == (
        {
            "experiment_id": "duplicate-target",
            "retained_source": str(first_registry),
            "duplicate_source": str(duplicate_registry),
            "retained_artifact_path": str(tmp_path / "duplicate-target.json"),
            "duplicate_artifact_path": str(tmp_path / "duplicate-target.json"),
        },
    )
    assert result.as_dict()["duplicate_targets"] == [result.duplicate_targets[0]]


def test_run_replay_batch_deduplicates_registry_and_manifest_inputs(tmp_path: Path) -> None:
    registry_marker = tmp_path / "registry-ran.txt"
    manifest_marker = tmp_path / "manifest-ran.txt"
    artifact_path = _artifact(tmp_path, "registry-target.json")
    registry_path = _registry_at(
        tmp_path / "registry.json",
        [
            _target(
                tmp_path,
                "mixed-target",
                _write_fixture_experiment(tmp_path, EXPECTED_METRICS, marker=registry_marker),
                artifact_path=str(artifact_path),
            )
        ],
    )
    manifest_path = tmp_path / "mixed-target.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "experiment-manifest/v1",
                "experiment_id": "mixed-target",
                "generated_at": "2026-06-08T12:00:00+00:00",
                "source_artifact_path": str(artifact_path),
                "command": _write_fixture_experiment(tmp_path, {**EXPECTED_METRICS, "sharpe": 0.10}, marker=manifest_marker),
                "module": "tests.fixture",
                "git": {},
                "config_snapshot": {},
                "environment": {},
                "input_file_hashes": {},
                "freeze_manifest": {"config": {}, "file_hashes": {}, "file_count": 0},
                "replay_safe": True,
                "provenance_status": "sidecar",
            }
        )
    )

    result = run_replay_batch([registry_path, manifest_path], output_path=tmp_path / "labs_replays.json", project_root=tmp_path)

    rows = json.loads((tmp_path / "labs_replays.json").read_text())
    assert [row["experiment_id"] for row in rows] == ["mixed-target"]
    assert rows[0]["status"] == "passed"
    assert registry_marker.exists() is True
    assert manifest_marker.exists() is False
    assert result.duplicate_targets[0]["retained_source"] == str(registry_path)
    assert result.duplicate_targets[0]["duplicate_source"] == str(manifest_path)


def test_replay_batch_cli_smoke_writes_report_from_fixture_registry(tmp_path: Path, capsys) -> None:
    registry_path = _registry(
        tmp_path,
        [_target(tmp_path, "cli-pass", _write_fixture_experiment(tmp_path, EXPECTED_METRICS))],
    )
    output_path = tmp_path / "labs_replays.json"

    exit_code = main([
        "--output",
        str(output_path),
        "--project-root",
        str(tmp_path),
        str(registry_path),
    ])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["output_path"] == str(output_path)
    assert summary["counts"] == {"passed": 1, "failed": 0, "warning": 0}
    assert json.loads(output_path.read_text())[0]["experiment_id"] == "cli-pass"


def test_replay_batch_cli_accepts_concurrency_flag(tmp_path: Path, capsys) -> None:
    registry_path = _registry(
        tmp_path,
        [_target(tmp_path, "cli-concurrent-pass", _write_fixture_experiment(tmp_path, EXPECTED_METRICS))],
    )
    output_path = tmp_path / "labs_replays.json"

    exit_code = main([
        "--output",
        str(output_path),
        "--project-root",
        str(tmp_path),
        "--concurrency",
        "2",
        str(registry_path),
    ])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["output_path"] == str(output_path)
    assert summary["counts"] == {"passed": 1, "failed": 0, "warning": 0}
