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
    from src.monitor.decision_registry import DecisionRecord, DecisionRegistry, ExperimentRecord

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
    for i in range(11):
        reg.record_experiment(
            ExperimentRecord(
                experiment_id=f"contract-exp-{i}",
                timestamp_utc=f"2026-07-{i + 1:02d}T12:00:00+00:00",
                name=f"contract experiment {i}",
                metrics={"sharpe": 1.0 + (i / 100)},
                benchmark_metrics={"sharpe": 0.95},
            )
        )
    path = publish_decision_registry_json(public_dir=public, registry=reg)
    assert path.name == DECISION_REGISTRY_JSON
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == DECISION_REGISTRY_SCHEMA_VERSION
    assert payload["counts"]["decisions"] == 1
    assert payload["recent_decisions"][0]["decision_id"] == "contract-dec-1"
    recent_ids = {row["experiment_id"] for row in payload["recent_experiments"]}
    promotion_ids = {row["experiment_id"] for row in payload["promotion_evaluations"]}
    assert len(recent_ids) == len(promotion_ids) == 11
    assert promotion_ids == recent_ids
    assert payload["promotion_coverage"]["disclosure"] == "complete_promotion_evaluation_coverage"
    assert payload["promotion_coverage"]["unmatched_experiment_ids"] == []
    assert payload["replay_summaries"]
