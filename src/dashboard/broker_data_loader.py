"""Broker position sync, orders, and kill-switch state for dashboard signals."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

from src.paths import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["BrokerDataLoader", "empty_broker_payload"]


def empty_broker_payload() -> dict[str, Any]:
    return {
        "connected": False,
        "positions": [],
        "drift": [],
        "recent_orders": [],
        "last_sync": None,
        "kill_switch": False,
        "kill_switch_level": None,
        "kill_switch_source": None,
        "kill_switch_reason": None,
        "kill_switch_incident_id": None,
    }


class BrokerDataLoader:
    """Load broker JSONL logs and kill_switch.json for dashboard broker section."""

    def __init__(self, data_dir=DATA_DIR) -> None:
        self.data_dir = data_dir

    def load(self) -> dict[str, Any]:
        broker = empty_broker_payload()

        sync_log = self.data_dir / "position_sync.jsonl"
        if sync_log.exists():
            try:
                with open(sync_log) as f:
                    tail = deque(f, maxlen=1)
                if tail:
                    last = json.loads(tail[0])
                    broker["connected"] = True
                    broker["last_sync"] = last.get("timestamp")
                    broker["positions"] = last.get("broker_positions", [])
                    broker["drift"] = last.get("drift", [])
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load position sync log: %s", e)

        orders_log = self.data_dir / "broker_orders.jsonl"
        if orders_log.exists():
            try:
                with open(orders_log) as f:
                    recent = []
                    for line in deque(f, maxlen=10):
                        if line.strip():
                            recent.append(json.loads(line))
                    broker["recent_orders"] = list(reversed(recent))
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load broker orders log: %s", e)

        kill_file = self.data_dir / "kill_switch.json"
        if kill_file.exists():
            try:
                with open(kill_file) as f:
                    ks = json.load(f)
                broker["kill_switch"] = ks.get("enabled", False)
                broker["kill_switch_level"] = ks.get("level")
                broker["kill_switch_source"] = ks.get("source")
                broker["kill_switch_reason"] = ks.get("reason")
                broker["kill_switch_incident_id"] = ks.get("incident_id")
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning("Failed to load kill switch state: %s", e)

        return broker
