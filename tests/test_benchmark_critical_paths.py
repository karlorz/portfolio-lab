"""Tests for the opt-in critical-path performance benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"


EXPECTED_BENCHMARKS = {
    "price_loading",
    "ensemble_compute_vote",
    "signal_correlation_matrix",
    "combined_regime_backtest_fixture",
    "dashboard_health_generation",
}


def test_benchmark_cli_writes_structured_json(tmp_path, monkeypatch) -> None:
    """The benchmark entry point should write a structured latest-run JSON file."""
    monkeypatch.delenv("PORTFOLIO_LAB_ENABLE_ML", raising=False)
    from scripts import benchmark_critical_paths as benchmark

    output_path = tmp_path / "critical_paths_latest.json"
    baseline_path = tmp_path / "critical_paths_baseline.json"

    exit_code = benchmark.main(
        [
            "--output",
            str(output_path),
            "--baseline",
            str(baseline_path),
            "--runs",
            "1",
            "--warmup",
            "0",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "ok"
    assert payload["regression_threshold"] == 0.25
    assert payload["environment"]["ml_enabled"] == "0"

    benchmarks = payload["benchmarks"]
    assert {case["name"] for case in benchmarks} == EXPECTED_BENCHMARKS
    for case in benchmarks:
        assert case["duration_ms"] >= 0
        assert case["median_ms"] >= 0
        assert case["mean_ms"] >= 0
        assert case["runs"] == 1
        assert case["budget_status"] == "no_baseline"


def test_update_baseline_writes_reusable_budget_file(tmp_path) -> None:
    """Baseline refresh mode should persist benchmark medians for later checks."""
    from scripts import benchmark_critical_paths as benchmark

    output_path = tmp_path / "critical_paths_latest.json"
    baseline_path = tmp_path / "critical_paths_baseline.json"

    exit_code = benchmark.main(
        [
            "--output",
            str(output_path),
            "--baseline",
            str(baseline_path),
            "--update-baseline",
            "--runs",
            "1",
            "--warmup",
            "0",
        ]
    )

    assert exit_code == 0
    baseline = json.loads(baseline_path.read_text())
    assert baseline["benchmarks"].keys() == EXPECTED_BENCHMARKS
    for case in baseline["benchmarks"].values():
        assert case["baseline_ms"] >= 0


def test_fail_on_regression_returns_nonzero_for_clear_budget_breach(
    tmp_path,
) -> None:
    """Opt-in budget checks should fail when every case clearly exceeds baseline."""
    from scripts import benchmark_critical_paths as benchmark

    output_path = tmp_path / "critical_paths_latest.json"
    baseline_path = tmp_path / "critical_paths_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    name: {"baseline_ms": 0.0001}
                    for name in EXPECTED_BENCHMARKS
                }
            }
        )
    )

    exit_code = benchmark.main(
        [
            "--output",
            str(output_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-regression",
            "--runs",
            "1",
            "--warmup",
            "0",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "failed_budget"
    assert all(case["budget_status"] == "failed_budget" for case in payload["benchmarks"])


def test_makefile_has_opt_in_perf_target_and_baseline_help() -> None:
    """Performance checks should be opt-in and documented in Makefile help."""
    text = MAKEFILE.read_text()

    assert ".PHONY: perf" in text
    assert "make perf" in text
    assert "PERF_UPDATE_BASELINE=1 make perf" in text
    assert "benchmark_critical_paths.py" in text
