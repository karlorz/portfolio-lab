"""Opt-in loader for broker-readonly-gateway daily-brief plugin.

Default off. Reads a redacted snapshot file via the plugin; does not open
IBKR/OpenD sockets or import the live execution broker.
"""

from __future__ import annotations

import logging
import os
from typing import Any

ENABLE_ENV = "PORTFOLIO_LAB_ENABLE_BROKER_SNAPSHOT"
ENTRY_POINT_GROUP = "portfolio_lab.plugins"
ENTRY_POINT_NAME = "broker_snapshot"

logger = logging.getLogger(__name__)


def broker_snapshot_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "0") == "1"


def _discover_plugin() -> Any | None:
    from importlib.metadata import entry_points

    selected = entry_points(group=ENTRY_POINT_GROUP, name=ENTRY_POINT_NAME)
    if not selected:
        return None
    loaded = next(iter(selected)).load()
    if isinstance(loaded, type):
        return loaded()
    return loaded


def load_broker_snapshot_section(
    *,
    plugin: Any | None = None,
    snapshot_path: str | None = None,
):
    from src.monitor.daily_brief import BriefSection

    if not broker_snapshot_enabled():
        return None
    try:
        loaded = plugin if plugin is not None else _discover_plugin()
    except Exception:
        logger.warning("broker snapshot plugin discovery failed", exc_info=True)
        return None
    if loaded is None:
        return None
    if getattr(loaded, "read_only", None) is not True:
        logger.warning("refusing broker snapshot plugin without read_only=True")
        return None
    try:
        payload = loaded.brief_section(snapshot_path)
    except Exception:
        logger.warning("broker snapshot plugin brief_section failed", exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    return BriefSection(
        name="broker_snapshot",
        title=str(payload.get("title") or "Broker snapshot"),
        severity=str(payload.get("severity") or "normal"),
        data_text=str(payload.get("text") or ""),
    )
