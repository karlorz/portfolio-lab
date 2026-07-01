"""Tests for runtime decision replay + experiment registry (SQLite)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitor.decision_registry import (
    DECISION_REGISTRY_SCHEMA_VERSION,
    DecisionRecord,
    DecisionRegistry,
    ExperimentRecord,
    build_decision_registry_snapshot,
    evaluate_promotion_candidate,
    publish_decision_registry_json,
    record_backtest_experiment,
    record_dashboard_cycle_decision,
    record_evaluator_cycle_decision,
    sync_labs_registry_experiments,
)


def _sample_decision(decision_id: str = "dec-1") -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        timestamp_utc="2026-07-01T12:00:00+00:00",
        run_id="run-test",
        action="hold",
        reason="drift_below_threshold",
        current_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
        target_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        regime="NORMAL",
        regime_confidence=0.82,
        signal_weights={"multi_speed_momentum": 0.4, "cross_asset_rv": 0.2},
        signal_votes={"multi_speed_momentum": 0.6},
        gates_triggered=["smart_rebalance_hold"],
    )


def test_decision_record_validation() -> None:
    record = _sample_decision()
    assert record.action == "hold"
    assert record.target_weights["SPY"] == 0.46


def test_experiment_record_validation() -> None:
    exp = ExperimentRecord(
        experiment_id="wf-1",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="walk-forward",
        metrics={"sharpe": 1.1},
        benchmark_metrics={"sharpe": 0.95},
    )
    assert exp.promotion_status == "candidate"


def test_sqlite_init_and_crud(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    reg = DecisionRegistry(db_path=db)

    dec_id = reg.record_decision(_sample_decision())
    assert dec_id == "dec-1"
    loaded = reg.get_decision("dec-1")
    assert loaded is not None
    assert loaded.regime == "NORMAL"

    exp = ExperimentRecord(
        experiment_id="exp-a",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="exp-a",
        metrics={"sharpe": 1.0},
    )
    reg.record_experiment(exp)
    listed = reg.list_experiments(status="candidate")
    assert len(listed) == 1
    assert listed[0].experiment_id == "exp-a"

    recent = reg.list_recent_decisions(limit=5)
    assert len(recent) == 1


def test_replay_decision_output(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    reg.record_decision(_sample_decision())

    replay = reg.replay_decision("dec-1")
    assert replay["found"] is True
    assert replay["replay"]["action"] == "hold"
    assert "SPY" in replay["replay"]["weight_delta"]

    missing = reg.replay_decision("missing")
    assert missing["found"] is False


def test_promotion_gate_pass_and_fail() -> None:
    passing = ExperimentRecord(
        experiment_id="good",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="good",
        metrics={"sharpe": 1.0, "max_drawdown_pct": -20.0, "turnover": 0.5},
        benchmark_metrics={"sharpe": 0.95, "max_drawdown_pct": -22.0},
    )
    result = evaluate_promotion_candidate(passing)
    assert result["pass"] is True
    assert result["recommended_status"] == "promoted"

    failing = ExperimentRecord(
        experiment_id="bad",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="bad",
        metrics={"sharpe": 0.96, "max_drawdown_pct": -30.0},
        benchmark_metrics={"sharpe": 0.95, "max_drawdown_pct": -22.0},
    )
    result_fail = evaluate_promotion_candidate(failing)
    assert result_fail["pass"] is False
    assert result_fail["failures"]


def test_dashboard_cycle_recording(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    signals = {
        "ensemble_voting": {
            "action": "rebalance",
            "regime": "NORMAL",
            "regime_confidence": 0.9,
            "confidence": 0.7,
            "weighted_consensus": 0.55,
            "source_breakdown": [
                {"source": "msm", "weight": 0.5, "strength": 0.4},
            ],
        },
        "smart_rebalance": {
            "decision": "defer",
            "reason": "vpin_high",
            "should_execute": False,
            "max_drift": 0.03,
            "estimated_cost_bps": 12.0,
        },
        "staleness": {"stale_signals": ["fred_macro"]},
    }
    context = {
        "target_alloc": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "total_value": 100_000.0,
        "positions": [
            {"symbol": "SPY", "value": 50_000},
            {"symbol": "GLD", "value": 30_000},
            {"symbol": "TLT", "value": 20_000},
        ],
    }
    recorded = record_dashboard_cycle_decision(signals, context=context, registry=reg)
    assert recorded is not None
    decisions = reg.list_recent_decisions()
    assert decisions[0].action == "defer"
    assert "signal_staleness" in decisions[0].gates_triggered


def test_publish_decision_registry_json(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    reg.record_decision(_sample_decision())
    public = tmp_path / "public" / "data"
    path = publish_decision_registry_json(public_dir=public, registry=reg)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == DECISION_REGISTRY_SCHEMA_VERSION
    assert payload["counts"]["decisions"] == 1
    assert payload["replay_summaries"]


def test_sync_labs_registry_experiments(tmp_path: Path) -> None:
    labs_path = tmp_path / "labs_registry.json"
    labs_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-01T12:00:00+00:00",
                "experiments": [
                    {
                        "experiment_id": "walk-forward",
                        "status": "candidate",
                        "artifact_path": "data/walk_forward_report.json",
                        "metrics": {"sharpe": 1.12},
                    }
                ],
            }
        )
    )
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    count = sync_labs_registry_experiments(labs_path, registry=reg)
    assert count == 1
    exps = reg.list_experiments()
    assert exps[0].experiment_id == "walk-forward"
    assert exps[0].metrics["sharpe"] == pytest.approx(1.12)


def test_build_snapshot_limits(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    for i in range(3):
        reg.record_decision(_sample_decision(f"dec-{i}"))
    snap = build_decision_registry_snapshot(reg, decision_limit=2)
    assert snap["counts"]["decisions"] == 2


def test_evaluator_cycle_recording(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    recorded = record_evaluator_cycle_decision(
        mode="paper",
        regime="NORMAL",
        target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        prices={"SPY": 500.0, "GLD": 200.0, "TLT": 90.0},
        portfolio_value=100_000.0,
        current_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
        orders=[{"symbol": "SPY", "side": "sell"}],
        registry=reg,
    )
    assert recorded is not None
    row = reg.list_recent_decisions()[0]
    assert row.action == "rebalance"
    assert row.extras.get("source") == "evaluator"


def test_evaluator_kill_switch_recording(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    record_evaluator_cycle_decision(
        mode="paper",
        regime="CRISIS",
        target_alloc={"SPY": 0.35},
        prices={"SPY": 400.0},
        portfolio_value=50_000.0,
        current_weights={"SPY": 0.6},
        orders=None,
        kill_reason="drawdown_20pct",
        registry=reg,
    )
    row = reg.list_recent_decisions()[0]
    assert row.action == "halt"
    assert "kill_switch" in row.gates_triggered


def test_record_backtest_experiment(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    exp_id = record_backtest_experiment(
        {"sharpe_ratio": 1.05, "max_drawdown": -22.0, "baseline_sharpe": 0.95},
        experiment_id="wf-test-1",
        output_path=tmp_path / "out.json",
        registry=reg,
    )
    assert exp_id == "wf-test-1"
    listed = reg.list_experiments()
    assert listed[0].metrics["sharpe"] == pytest.approx(1.05)
    assert listed[0].benchmark_metrics.get("sharpe") == pytest.approx(0.95)