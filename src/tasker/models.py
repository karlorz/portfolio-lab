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
# Intentional no-op (e.g. eval EXIT_BLOCKED=2 under kill authority) — not a failure.
RUN_BLOCKED = "blocked"

TERMINAL_RUN_STATUSES = {
    RUN_SUCCESS,
    RUN_ERROR,
    RUN_TIMEOUT,
    RUN_CANCELLED,
    RUN_BLOCKED,
}

# Process exit codes that map to intentional block (not RUN_ERROR) **only** for
# tasks listed in INTENTIONAL_BLOCK_TASK_IDS. GNU Make also exits 2 on recipe
# parse/recipe failure ("missing separator") — those must be RUN_ERROR, not
# greenwashed as intentional blocked (Batch CE / c363).
EXIT_CODE_BLOCKED = 2

# Named terminal causes (Task 3A). Planned interruptions (service restart /
# operator cancellation) are not ordinary failures: they never increment
# consecutive_failures. Unplanned orphan/timeout/error rows carry no cause or
# an explicit unplanned cause and keep counting.
TERMINATION_CAUSE_SERVICE_RESTART = "service_restart"
TERMINATION_CAUSE_OPERATOR_CANCELLED = "operator_cancelled"
TERMINATION_CAUSE_UNPLANNED = "unplanned"
PLANNED_TERMINATION_CAUSES = frozenset(
    {
        TERMINATION_CAUSE_SERVICE_RESTART,
        TERMINATION_CAUSE_OPERATOR_CANCELLED,
    }
)


# Tasks whose command legitimately returns EXIT_BLOCKED=2 as control-loop skip.
# Expand only when a Makefile target documents `exit 2` → STATUS=blocked.
INTENTIONAL_BLOCK_TASK_IDS = frozenset(
    {
        "portfolio-lab-eval",
    }
)


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
