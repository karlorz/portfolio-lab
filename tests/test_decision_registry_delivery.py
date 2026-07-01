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