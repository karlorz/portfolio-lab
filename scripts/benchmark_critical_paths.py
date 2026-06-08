#!/usr/bin/env python3
"""Opt-in critical-path performance benchmark harness.

The harness intentionally avoids pytest-benchmark or other new dependencies.
It measures a small set of stable local code paths with time.perf_counter(),
writes structured JSON under data/perf by default, and compares medians against
an optional baseline with a tolerant regression threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PORTFOLIO_LAB_ENABLE_ML", "0")

from src.paths import DATA_DIR  # noqa: E402


DEFAULT_PERF_DIR = DATA_DIR / "perf"
DEFAULT_OUTPUT = DEFAULT_PERF_DIR / "critical_paths_latest.json"
DEFAULT_BASELINE = DEFAULT_PERF_DIR / "critical_paths_baseline.json"
DEFAULT_REGRESSION_THRESHOLD = 0.25


@dataclass(frozen=True)
class BenchmarkCase:
    """Single benchmark case and callable action."""

    name: str
    description: str
    action: Callable[[], Any]


@dataclass(frozen=True)
class BenchmarkSuite:
    """Benchmark cases plus cleanup callbacks for fixture resources."""

    cases: list[BenchmarkCase]
    cleanup: list[Callable[[], None]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_baselines(path: Path) -> dict[str, float]:
    """Load baseline medians keyed by benchmark name.

    Supports the baseline file written by this harness and latest-run JSON files
    for convenience when users seed a baseline from an existing result.
    """
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    benchmarks = payload.get("benchmarks", {})
    if isinstance(benchmarks, dict):
        baselines: dict[str, float] = {}
        for name, case in benchmarks.items():
            if not isinstance(case, dict):
                continue
            value = case.get("baseline_ms", case.get("median_ms"))
            if isinstance(value, (int, float)):
                baselines[str(name)] = float(value)
        return baselines

    if isinstance(benchmarks, list):
        baselines = {}
        for case in benchmarks:
            if not isinstance(case, dict):
                continue
            name = case.get("name")
            value = case.get("baseline_ms", case.get("median_ms"))
            if isinstance(name, str) and isinstance(value, (int, float)):
                baselines[name] = float(value)
        return baselines

    return {}


def _compare_to_baseline(
    median_ms: float,
    baseline_ms: float | None,
    threshold: float,
) -> tuple[str, float | None]:
    if baseline_ms is None:
        return "no_baseline", None
    if baseline_ms <= 0:
        return "invalid_baseline", None

    ratio = median_ms / baseline_ms
    if ratio > 1.0 + threshold:
        return "failed_budget", ratio
    return "ok", ratio


def _run_case(
    case: BenchmarkCase,
    runs: int,
    warmup: int,
    baselines: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    for _ in range(warmup):
        case.action()

    durations_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        case.action()
        elapsed_ms = max((time.perf_counter() - start) * 1000, 0.000001)
        durations_ms.append(elapsed_ms)

    median_ms = float(statistics.median(durations_ms))
    baseline_ms = baselines.get(case.name)
    budget_status, ratio = _compare_to_baseline(
        median_ms=median_ms,
        baseline_ms=baseline_ms,
        threshold=threshold,
    )

    return {
        "name": case.name,
        "description": case.description,
        "runs": runs,
        "duration_ms": round(median_ms, 6),
        "median_ms": round(median_ms, 6),
        "mean_ms": round(float(statistics.fmean(durations_ms)), 6),
        "min_ms": round(min(durations_ms), 6),
        "max_ms": round(max(durations_ms), 6),
        "baseline_ms": round(baseline_ms, 6) if baseline_ms is not None else None,
        "ratio": round(ratio, 6) if ratio is not None else None,
        "budget_status": budget_status,
    }


def _make_ic_data() -> dict[str, list[list[float]]]:
    return {
        "alternative_data": [[0.01 * i, 0.001 * i] for i in range(1, 31)],
        "international_momentum": [[0.012 * i, 0.001 * i] for i in range(1, 31)],
        "cross_asset_rv": [[0.04 - 0.001 * i, -0.0005 * i] for i in range(1, 31)],
        "__staged__": [["ignored", 0.0]],
    }


def _make_price_fixture(length: int = 500) -> dict[str, Any]:
    import numpy as np

    days = np.arange(length, dtype=float)
    return {
        "SPY": 100.0 * (1.0 + 0.0007 * days + 0.015 * np.sin(days / 19.0)),
        "GLD": 100.0 * (1.0 + 0.00035 * days + 0.010 * np.cos(days / 23.0)),
        "TLT": 100.0 * (1.0 + 0.00015 * days + 0.012 * np.sin(days / 29.0)),
    }


def _make_signal_readings() -> dict[Any, Any]:
    from src.signals.signal_source import SignalSource
    from src.strategy.ensemble_voter import SignalReading

    timestamp = "2026-06-08T00:00:00+00:00"
    return {
        SignalSource.CROSS_ASSET_RV: SignalReading(
            source=SignalSource.CROSS_ASSET_RV,
            timestamp=timestamp,
            value=0.35,
            confidence=0.80,
            weight=0.12,
            regime_fit="normal",
            asset_signals={"SPY": 0.20, "GLD": 0.10, "TLT": -0.05},
        ),
        SignalSource.ALTERNATIVE_DATA: SignalReading(
            source=SignalSource.ALTERNATIVE_DATA,
            timestamp=timestamp,
            value=0.20,
            confidence=0.70,
            weight=0.22,
            regime_fit="normal",
            asset_signals={"SPY": 0.25, "GLD": 0.00, "TLT": -0.05},
        ),
        SignalSource.INTERNATIONAL_MOMENTUM: SignalReading(
            source=SignalSource.INTERNATIONAL_MOMENTUM,
            timestamp=timestamp,
            value=0.10,
            confidence=0.65,
            weight=0.22,
            regime_fit="normal",
            asset_signals={"SPY": 0.15, "GLD": -0.05, "TLT": 0.00},
        ),
        SignalSource.UNIFIED_OVERLAY: SignalReading(
            source=SignalSource.UNIFIED_OVERLAY,
            timestamp=timestamp,
            value=-0.05,
            confidence=0.60,
            weight=0.17,
            regime_fit="normal",
            asset_signals={"SPY": -0.05, "GLD": 0.10, "TLT": 0.15},
        ),
        SignalSource.MULTI_TIMEFRAME_FUSION: SignalReading(
            source=SignalSource.MULTI_TIMEFRAME_FUSION,
            timestamp=timestamp,
            value=0.15,
            confidence=0.75,
            weight=0.10,
            regime_fit="normal",
            asset_signals={"SPY": 0.20, "GLD": 0.05, "TLT": 0.00},
        ),
        SignalSource.GOOGLE_TRENDS: SignalReading(
            source=SignalSource.GOOGLE_TRENDS,
            timestamp=timestamp,
            value=0.05,
            confidence=0.55,
            weight=0.05,
            regime_fit="normal",
            asset_signals={"SPY": 0.05, "GLD": 0.05, "TLT": 0.05},
        ),
    }


def _create_dashboard_fixture(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
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

    base_date = datetime.now() - timedelta(days=20)
    for symbol, base_price in {"SPY": 500.0, "GLD": 220.0, "TLT": 90.0, "QQQ": 450.0}.items():
        for offset in range(20):
            day = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                (symbol, day, round(base_price + offset * 0.25, 2)),
            )
    conn.commit()
    return conn


def build_benchmark_suite(runtime_dir: Path) -> BenchmarkSuite:
    """Build benchmark cases with isolated runtime fixtures."""
    runtime_dir.mkdir(parents=True, exist_ok=True)

    from src.backtest.combined_regime_alloc_vol_target import (
        REGIME_VOL_TARGETS,
        backtest_strategy,
    )
    from src.dashboard import generator as dashboard_module
    from src.dashboard.generator import DashboardGenerator
    from src.data.price_cache import get_prices_df, invalidate_price_cache
    from src.strategy.regime_allocation import DEFAULT_ALLOCATION, REGIME_ALLOCATIONS
    from src.strategy.ensemble_voter import (
        EnsembleVoter,
        Regime,
        compute_signal_correlation_matrix,
    )

    voter = EnsembleVoter(data_path=runtime_dir / "ensemble")
    readings = _make_signal_readings()
    ic_data = _make_ic_data()
    prices = _make_price_fixture()

    dashboard_dir = runtime_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "cron_status.json").write_text('{"jobs": []}\n')
    dashboard_conn = _create_dashboard_fixture(dashboard_dir / "market.db")
    dashboard = DashboardGenerator.__new__(DashboardGenerator)
    dashboard.conn = dashboard_conn
    dashboard.conn.row_factory = sqlite3.Row

    def benchmark_price_loading() -> int:
        invalidate_price_cache()
        return len(get_prices_df(symbols=["SPY", "GLD", "TLT"]))

    def benchmark_compute_vote() -> str:
        vote = voter.compute_vote(
            readings=readings,
            regime=Regime.NORMAL,
            regime_confidence=0.90,
        )
        return vote.action

    def benchmark_correlation_matrix() -> int:
        matrix = compute_signal_correlation_matrix(ic_data=ic_data)
        return len(matrix["correlation_penalties"])

    def benchmark_combined_backtest_fixture() -> float:
        row = backtest_strategy(
            prices=prices,
            label="Benchmark Combined Fixture",
            allocation_map=REGIME_ALLOCATIONS,
            default_alloc=DEFAULT_ALLOCATION,
            vol_target_map=REGIME_VOL_TARGETS,
            apply_vol_target=True,
        )
        return row.sharpe

    def benchmark_dashboard_health_generation() -> str:
        old_data_dir = dashboard_module.DATA_DIR
        old_public_dir = dashboard_module.PUBLIC_DIR
        dashboard_module.DATA_DIR = dashboard_dir
        dashboard_module.PUBLIC_DIR = dashboard_dir / "public"
        try:
            return str(dashboard.generate_health_json())
        finally:
            dashboard_module.DATA_DIR = old_data_dir
            dashboard_module.PUBLIC_DIR = old_public_dir

    return BenchmarkSuite(
        cases=[
            BenchmarkCase(
                name="price_loading",
                description="Load SPY/GLD/TLT close prices through get_prices_df().",
                action=benchmark_price_loading,
            ),
            BenchmarkCase(
                name="ensemble_compute_vote",
                description="Compute an ensemble vote from synthetic active signal readings.",
                action=benchmark_compute_vote,
            ),
            BenchmarkCase(
                name="signal_correlation_matrix",
                description="Compute pairwise IC prediction correlations from synthetic IC data.",
                action=benchmark_correlation_matrix,
            ),
            BenchmarkCase(
                name="combined_regime_backtest_fixture",
                description="Run the combined regime allocation and vol-target fixture backtest.",
                action=benchmark_combined_backtest_fixture,
            ),
            BenchmarkCase(
                name="dashboard_health_generation",
                description="Generate dashboard health JSON from an isolated SQLite fixture.",
                action=benchmark_dashboard_health_generation,
            ),
        ],
        cleanup=[dashboard.close],
    )


def run_benchmarks(
    output_path: Path,
    baseline_path: Path,
    runtime_dir: Path,
    runs: int,
    warmup: int,
    regression_threshold: float,
    update_baseline: bool,
) -> dict[str, Any]:
    baselines = _load_baselines(baseline_path)
    suite = build_benchmark_suite(runtime_dir)
    try:
        results = [
            _run_case(
                case=case,
                runs=runs,
                warmup=warmup,
                baselines=baselines,
                threshold=regression_threshold,
            )
            for case in suite.cases
        ]
    finally:
        for cleanup in suite.cleanup:
            cleanup()

    failed = any(case["budget_status"] == "failed_budget" for case in results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed_budget" if failed else "ok",
        "runs": runs,
        "warmup": warmup,
        "regression_threshold": regression_threshold,
        "output_path": str(output_path),
        "baseline_path": str(baseline_path),
        "environment": {
            "ml_enabled": os.environ.get("PORTFOLIO_LAB_ENABLE_ML", ""),
            "python": sys.version.split()[0],
        },
        "benchmarks": results,
    }
    _write_json(output_path, payload)

    if update_baseline:
        baseline_payload = {
            "generated_at": payload["generated_at"],
            "source_output": str(output_path),
            "regression_threshold": regression_threshold,
            "benchmarks": {
                case["name"]: {
                    "baseline_ms": case["median_ms"],
                    "description": case["description"],
                }
                for case in results
            },
        }
        _write_json(baseline_path, baseline_payload)

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark portfolio-lab critical paths and compare optional budgets.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Directory for isolated benchmark fixture state.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help="Allowed slowdown ratio above baseline, default 0.25 for 25%%.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Refresh the baseline JSON from this run.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero when any benchmark exceeds its budget.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.regression_threshold < 0:
        raise ValueError("--regression-threshold must be >= 0")

    output_path = args.output
    baseline_path = args.baseline
    runtime_dir = args.runtime_dir or output_path.parent / ".critical_paths_runtime"

    payload = run_benchmarks(
        output_path=output_path,
        baseline_path=baseline_path,
        runtime_dir=runtime_dir,
        runs=args.runs,
        warmup=args.warmup,
        regression_threshold=args.regression_threshold,
        update_baseline=args.update_baseline,
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_regression and not args.update_baseline and payload["status"] == "failed_budget":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
