"""Dashboard delivery includes decision_registry.json when generator publishes it."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitor.decision_registry import (
    DECISION_REGISTRY_JSON,
    DECISION_REGISTRY_SCHEMA_VERSION,
    publish_decision_registry_json,
)


def test_decision_registry_json_contract(tmp_path: Path) -> None:
    from src.monitor.decision_registry import DecisionRecord, DecisionRegistry

    public = tmp_path / "public" / "data"
    reg = DecisionRegistry(db_path=tmp_path / "decision_registry.db")
    reg.record_decision(
        DecisionRecord(
            decision_id="contract-dec-1",
            timestamp_utc="2026-07-01T12:00:00+00:00",
            run_id="contract-run",
            action="hold",
            reason="test",
            target_weights={"SPY": 0.46},
        )
    )
    path = publish_decision_registry_json(public_dir=public, registry=reg)
    assert path.name == DECISION_REGISTRY_JSON
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == DECISION_REGISTRY_SCHEMA_VERSION
    assert payload["counts"]["decisions"] == 1
    assert payload["recent_decisions"][0]["decision_id"] == "contract-dec-1"
    assert payload["replay_summaries"]
    assert payload["projection_freshness"]["status"] == "current"
    assert payload["projection_freshness"]["ledger_head"]["decision_id"] == "contract-dec-1"
    assert payload["projection_freshness"]["projection_head"]["decision_id"] == "contract-dec-1"
    assert payload["projection_freshness"]["lag_decision_count"] == 0


def test_decision_registry_json_publishes_metric_gate_separately_from_governance_status(
    tmp_path: Path,
) -> None:
    from src.monitor.decision_registry import (
        DecisionRegistry,
        ExperimentRecord,
    )

    public = tmp_path / "public" / "data"
    reg = DecisionRegistry(db_path=tmp_path / "decision_registry.db")
    reg.record_experiment(
        ExperimentRecord(
            experiment_id="artifact:watch-row",
            timestamp_utc="2026-07-01T12:00:00+00:00",
            name="artifact:watch-row",
            metrics={"sharpe": 1.05},
            benchmark_metrics={"sharpe": 0.95},
            promotion_status="candidate",
            artifacts={"provenance_status": "missing"},
            tags=["labs_registry"],
        )
    )

    path = publish_decision_registry_json(public_dir=public, registry=reg)
    payload = json.loads(path.read_text())
    promo = payload["promotion_evaluations"][0]

    assert promo["recommended_status"] == "candidate"
    assert promo["pass"] is False
    assert promo["metric_gate_status"] == "promoted"
    assert promo["metric_gate_pass"] is True
    assert promo["failures"] == ["provenance_missing"]
    assert promo["semantic_disclosure"] == {
        "state": "governance_blocked",
        "recommendation_type": "metric_gate",
        "governance_status": "candidate",
        "provenance_status": "missing",
        "metric_gate_status": "promoted",
        "reasons": ["provenance_missing"],
    }


def test_decision_registry_json_discloses_live_role_and_source(
    tmp_path: Path,
) -> None:
    from src.monitor.decision_registry import (
        DecisionRegistry,
        record_dashboard_cycle_decision,
    )

    public = tmp_path / "public" / "data"
    reg = DecisionRegistry(db_path=tmp_path / "decision_registry.db")
    recorded = record_dashboard_cycle_decision(
        {
            "ensemble_voting": {"action": "hold", "source_breakdown": []},
            "smart_rebalance": {"decision": "hold", "reason": "drift_below_threshold"},
        },
        context={},
        registry=reg,
        run_id="dashboard-test-run",
    )
    assert recorded is not None

    path = publish_decision_registry_json(public_dir=public, registry=reg)
    payload = json.loads(path.read_text())
    decision = payload["recent_decisions"][0]

    assert decision["decision_role"] == "live_executed"
    assert decision["decision_source"] == "dashboard_cycle"


def test_decision_registry_json_evaluates_all_recent_experiment_rows(
    tmp_path: Path,
) -> None:
    from src.monitor.decision_registry import (
        DecisionRegistry,
        ExperimentRecord,
    )

    public = tmp_path / "public" / "data"
    reg = DecisionRegistry(db_path=tmp_path / "decision_registry.db")
    for idx in range(12):
        reg.record_experiment(
            ExperimentRecord(
                experiment_id=f"experiment-{idx:02d}",
                timestamp_utc=f"2026-07-01T12:{59 - idx:02d}:00+00:00",
                name=f"experiment-{idx:02d}",
                metrics={"sharpe": 1.0 + idx / 100},
            )
        )

    path = publish_decision_registry_json(public_dir=public, registry=reg)
    payload = json.loads(path.read_text())

    assert len(payload["recent_experiments"]) == 12
    assert len(payload["promotion_evaluations"]) == 12
    assert payload["promotion_coverage"] == {
        "recent_experiment_count": 12,
        "evaluated_count": 12,
        "unmatched_count": 0,
        "unmatched_experiment_ids": [],
        "disclosure": "complete_promotion_evaluation_coverage",
    }
    assert "promotion_disclosure" not in payload["recent_experiments"][10]
