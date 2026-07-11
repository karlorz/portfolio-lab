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


def _sample_shadow_decision(
    decision_id: str = "shadow-1",
    *,
    divergence_metrics: dict[str, float] | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        timestamp_utc="2026-07-01T12:05:00+00:00",
        run_id="dashboard-shadow-run",
        decision_role="shadow",
        decision_source="dashboard_cycle",
        controller_id="ensemble_voter_shadow",
        live_decision_id="live-1",
        baseline_controller_id="regime_allocation_live",
        benchmark_window={"label": "dashboard_cycle", "observations": 1},
        divergence_metrics=divergence_metrics
        or {"max_weight_delta": 0.04, "action_mismatch": 1.0},
        promotion_review_status="pending_review",
        action="rebalance",
        reason="shadow_candidate_recommended_rebalance",
    )


def test_decision_record_validation() -> None:
    record = _sample_decision()
    assert record.action == "hold"
    assert record.target_weights["SPY"] == 0.46


def test_shadow_decision_requires_controller_and_live_linkage() -> None:
    record = _sample_shadow_decision()

    assert record.controller_id == "ensemble_voter_shadow"
    assert record.live_decision_id == "live-1"
    assert record.promotion_review_status == "pending_review"

    with pytest.raises(ValueError, match="controller_id"):
        DecisionRecord(
            decision_id="shadow-missing-controller",
            timestamp_utc="2026-07-01T12:05:00+00:00",
            run_id="dashboard-shadow-run",
            decision_role="shadow",
            decision_source="dashboard_cycle",
            live_decision_id="live-1",
            action="hold",
        )

    with pytest.raises(ValueError, match="live_decision_id"):
        DecisionRecord(
            decision_id="shadow-missing-live-link",
            timestamp_utc="2026-07-01T12:05:00+00:00",
            run_id="dashboard-shadow-run",
            decision_role="shadow",
            decision_source="dashboard_cycle",
            controller_id="ensemble_voter_shadow",
            action="hold",
        )


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


def test_replay_decision_includes_shadow_evidence_fields(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    reg.record_decision(_sample_decision("live-1"))
    reg.record_decision(_sample_shadow_decision())

    replay = reg.replay_decision("shadow-1")

    assert replay["found"] is True
    assert replay["replay"]["decision_role"] == "shadow"
    assert replay["replay"]["controller_id"] == "ensemble_voter_shadow"
    assert replay["replay"]["live_decision_id"] == "live-1"
    assert replay["replay"]["baseline_controller_id"] == "regime_allocation_live"
    assert replay["replay"]["benchmark_window"] == {
        "label": "dashboard_cycle",
        "observations": 1,
    }
    assert replay["replay"]["divergence_metrics"] == {
        "max_weight_delta": 0.04,
        "action_mismatch": 1.0,
    }
    assert replay["replay"]["promotion_review_status"] == "pending_review"


def test_promotion_gate_pass_and_fail() -> None:
    passing = ExperimentRecord(
        experiment_id="good",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="good",
        metrics={"sharpe": 1.0, "max_drawdown_pct": -20.0, "turnover": 0.5},
        benchmark_metrics={"sharpe": 0.95, "max_drawdown_pct": -22.0},
        artifacts={"provenance_status": "sidecar"},
    )
    result = evaluate_promotion_candidate(passing)
    assert result["pass"] is True
    assert result["recommended_status"] == "promoted"
    assert result["metric_gate_status"] == "promoted"

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


def test_promotion_gate_requires_clean_provenance_for_governance_ready_promotion() -> None:
    missing_provenance = ExperimentRecord(
        experiment_id="missing-provenance",
        timestamp_utc="2026-07-01T12:00:00+00:00",
        name="missing-provenance",
        metrics={"sharpe": 1.05, "max_drawdown_pct": -20.0, "turnover": 0.5},
        benchmark_metrics={"sharpe": 0.95, "max_drawdown_pct": -22.0},
        promotion_status="candidate",
        artifacts={"provenance_status": "missing"},
    )

    result = evaluate_promotion_candidate(missing_provenance)

    assert result["metric_gate_pass"] is True
    assert result["metric_gate_status"] == "promoted"
    assert result["recommended_status"] == "candidate"
    assert result["pass"] is False
    assert result["failures"] == ["provenance_missing"]


def test_promotion_gate_respects_rejected_mapping_status() -> None:
    result = evaluate_promotion_candidate(
        {
            "experiment_id": "rejected-row",
            "promotion_status": "rejected",
            "metrics": {"sharpe": 1.05, "max_drawdown_pct": -20.0},
            "benchmark_metrics": {"sharpe": 0.95, "max_drawdown_pct": -22.0},
            "artifacts": {"provenance_status": "sidecar"},
        }
    )

    assert result["metric_gate_status"] == "promoted"
    assert result["recommended_status"] == "rejected"
    assert result["pass"] is False
    assert result["failures"] == ["registry_status_rejected"]


def test_promotion_gate_fails_closed_for_empty_metrics() -> None:
    result = evaluate_promotion_candidate(
        ExperimentRecord(
            experiment_id="empty-metrics",
            timestamp_utc="2026-07-01T12:00:00+00:00",
            name="empty-metrics",
            metrics={},
            benchmark_metrics={"sharpe": 0.95},
            artifacts={"provenance_status": "sidecar"},
        )
    )

    assert result["metric_gate_status"] == "candidate"
    assert result["metric_gate_pass"] is False
    assert result["recommended_status"] != "promoted"
    assert result["pass"] is False
    assert "missing_metrics" in result["failures"]


def test_promotion_gate_fails_closed_without_benchmark_inputs() -> None:
    result = evaluate_promotion_candidate(
        ExperimentRecord(
            experiment_id="missing-benchmark",
            timestamp_utc="2026-07-01T12:00:00+00:00",
            name="missing-benchmark",
            metrics={"sharpe": 1.25},
            benchmark_metrics={},
            artifacts={"provenance_status": "sidecar"},
        )
    )

    assert result["metric_gate_status"] == "candidate"
    assert result["metric_gate_pass"] is False
    assert result["pass"] is False
    assert "missing_benchmark_metrics" in result["failures"]


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
    assert decisions[0].decision_role == "live_executed"
    assert decisions[0].decision_source == "dashboard_cycle"
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
    assert payload["projection_freshness"]["status"] == "current"
    assert payload["projection_freshness"]["ledger_head"]["decision_id"] == "dec-1"
    assert payload["projection_freshness"]["projection_head"]["decision_id"] == "dec-1"
    assert payload["projection_freshness"]["lag_decision_count"] == 0


def test_evaluator_decision_refreshes_public_projection_after_existing_publish(
    tmp_path: Path,
) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    public = tmp_path / "public" / "data"
    reg.record_decision(_sample_decision("dashboard-old"))
    publish_decision_registry_json(public_dir=public, registry=reg)

    recorded = record_evaluator_cycle_decision(
        mode="paper",
        regime="NORMAL",
        target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        prices={"SPY": 500.0, "GLD": 200.0, "TLT": 90.0},
        portfolio_value=100_000.0,
        current_weights={"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
        orders=None,
        registry=reg,
        public_dir=public,
    )

    payload = json.loads((public / "decision_registry.json").read_text())

    assert recorded is not None
    assert payload["recent_decisions"][0]["decision_id"] == recorded
    assert payload["recent_decisions"][0]["decision_source"] == "evaluator_cycle"
    assert payload["projection_freshness"]["status"] == "current"
    assert payload["projection_freshness"]["ledger_head"]["decision_id"] == recorded
    assert payload["projection_freshness"]["projection_head"]["decision_id"] == recorded


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


def test_sync_labs_registry_blocks_fixture_and_temp_path_rows(tmp_path: Path) -> None:
    labs_path = tmp_path / "labs_registry.json"
    labs_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-01T12:00:00+00:00",
                "experiments": [
                    {
                        "experiment_id": "bad-registry-row",
                        "status": "candidate",
                        "artifact_path": "/tmp/pytest-of-root/pytest-99/bad.json",
                        "metrics": {},
                    }
                ],
            }
        )
    )
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")

    count = sync_labs_registry_experiments(labs_path, registry=reg)
    snap = build_decision_registry_snapshot(reg)
    exp = snap["recent_experiments"][0]
    promo = snap["promotion_evaluations"][0]

    assert count == 1
    assert exp["experiment_id"] == "bad-registry-row"
    assert exp["promotion_status"] == "rejected"
    assert exp["artifacts"]["registry_status"] == "rejected"
    assert exp["artifacts"]["provenance_status"] == "invalid"
    assert exp["artifacts"]["artifact_path"] is None
    assert exp["artifacts"]["temp_path_redacted"] is True
    assert "synthetic_or_test_only" in exp["tags"]
    assert "/tmp/pytest-of-root" not in json.dumps(snap)
    assert promo["metric_gate_status"] == "candidate"
    assert promo["recommended_status"] == "rejected"
    assert promo["pass"] is False


def test_public_snapshot_sanitizes_existing_fixture_and_temp_experiments(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    reg.record_experiment(
        ExperimentRecord(
            experiment_id="bad-registry-row",
            timestamp_utc="2026-07-01T12:00:00+00:00",
            name="bad-registry-row",
            metrics={},
            promotion_status="candidate",
            artifacts={
                "artifact_path": "/tmp/pytest-of-root/pytest-42/bad.json",
                "output_path": "/tmp/pytest-of-root/pytest-42/out.json",
                "provenance_status": "sidecar",
            },
        )
    )

    snap = build_decision_registry_snapshot(reg)
    exp = snap["recent_experiments"][0]
    promo = snap["promotion_evaluations"][0]

    assert "/tmp/pytest-of-root" not in json.dumps(snap)
    assert exp["promotion_status"] == "rejected"
    assert exp["artifacts"]["artifact_path"] is None
    assert exp["artifacts"]["output_path"] is None
    assert exp["artifacts"]["temp_path_redacted"] is True
    assert "synthetic_or_test_only" in exp["tags"]
    assert promo["recommended_status"] == "rejected"
    assert promo["pass"] is False


def test_build_snapshot_limits(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    for i in range(3):
        reg.record_decision(_sample_decision(f"dec-{i}"))
    snap = build_decision_registry_snapshot(reg, decision_limit=2)
    assert snap["counts"]["decisions"] == 2


def test_snapshot_publishes_shadow_evidence_summary(tmp_path: Path) -> None:
    reg = DecisionRegistry(db_path=tmp_path / "registry.db")
    reg.record_decision(_sample_decision("live-1"))
    reg.record_decision(_sample_shadow_decision(divergence_metrics={"max_weight_delta": 0.04}))

    snap = build_decision_registry_snapshot(reg)

    assert snap["shadow_evidence"] == {
        "shadow_decision_count": 1,
        "linked_shadow_decision_count": 1,
        "controllers": ["ensemble_voter_shadow"],
        "promotion_review_status_counts": {"pending_review": 1},
        "latest_shadow_decisions": [
            {
                "decision_id": "shadow-1",
                "controller_id": "ensemble_voter_shadow",
                "live_decision_id": "live-1",
                "baseline_controller_id": "regime_allocation_live",
                "promotion_review_status": "pending_review",
                "divergence_metrics": {"max_weight_delta": 0.04},
                "benchmark_window": {"label": "dashboard_cycle", "observations": 1},
            }
        ],
    }


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
    assert row.decision_role == "live_executed"
    assert row.decision_source == "evaluator_cycle"
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
