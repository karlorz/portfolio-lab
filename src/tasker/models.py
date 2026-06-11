"""Tasker data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_SUCCESS = "success"
RUN_ERROR = "error"
RUN_TIMEOUT = "timeout"
RUN_CANCELLED = "cancelled"

TERMINAL_RUN_STATUSES = {RUN_SUCCESS, RUN_ERROR, RUN_TIMEOUT, RUN_CANCELLED}


@dataclass(frozen=True)
class TaskDefinition:
    """Checked-in task definition loaded from config/tasker.yaml."""

    id: str
    label: str
    command: list[str]
    schedule: str | None = None
    enabled: bool = True
    manual_only: bool = False
    timeout_seconds: int = 300
    description: str | None = None
    auto_retry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "command": list(self.command),
            "schedule": self.schedule,
            "enabled": self.enabled,
            "manual_only": self.manual_only,
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
            "auto_retry": self.auto_retry,
        }
