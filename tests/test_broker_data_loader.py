"""Tests for BrokerDataLoader."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.broker_data_loader import BrokerDataLoader, empty_broker_payload


def test_empty_broker_payload_shape() -> None:
    b = empty_broker_payload()
    assert b["connected"] is False
    assert b["kill_switch"] is False


def test_load_from_position_sync_tail(tmp_path: Path) -> None:
    sync = tmp_path / "position_sync.jsonl"
    sync.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-01T12:00:00Z",
                "broker_positions": [{"symbol": "SPY", "qty": 10}],
                "drift": [{"symbol": "SPY", "drift_pct": 0.02}],
            }
        )
        + "\n"
    )
    loader = BrokerDataLoader(data_dir=tmp_path)
    broker = loader.load()
    assert broker["connected"] is True
    assert broker["positions"][0]["symbol"] == "SPY"
    assert broker["drift"][0]["drift_pct"] == 0.02


def test_kill_switch_enabled(tmp_path: Path) -> None:
    (tmp_path / "kill_switch.json").write_text(json.dumps({
        "enabled": True,
        "level": "halt",
        "reason": "unresolved_incident:signal_staleness",
        "source": "incident_lifecycle",
        "incident_id": "incident-123",
    }))
    broker = BrokerDataLoader(data_dir=tmp_path).load()
    assert broker["kill_switch"] is True
    assert broker["kill_switch_level"] == "halt"
    assert broker["kill_switch_source"] == "incident_lifecycle"
    assert broker["kill_switch_reason"] == "unresolved_incident:signal_staleness"
    assert broker["kill_switch_incident_id"] == "incident-123"
