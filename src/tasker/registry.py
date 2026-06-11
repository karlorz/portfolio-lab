"""YAML task registry and lightweight cron matching."""

from __future__ import annotations

import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import yaml

from src.paths import PROJECT_ROOT
from src.tasker.models import TaskDefinition

DEFAULT_TASKER_CONFIG = PROJECT_ROOT / "config" / "tasker.yaml"


class TaskRegistry:
    """In-memory registry of allowed tasker jobs."""

    def __init__(self, tasks: Iterable[TaskDefinition], validate_commands: bool = True):
        self.tasks = list(tasks)
        self._by_id = {task.id: task for task in self.tasks}
        if len(self._by_id) != len(self.tasks):
            raise ValueError("Task IDs must be unique")
        if validate_commands:
            for task in self.tasks:
                _validate_command(task.command)

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    def due_tasks(self, now: datetime) -> list[TaskDefinition]:
        now = _normalize_datetime(now)
        return [
            task
            for task in self.tasks
            if task.enabled and not task.manual_only and task.schedule and _cron_matches(task.schedule, now)
        ]

    def next_run_after(self, task_id: str, after: datetime) -> datetime | None:
        task = self.get(task_id)
        if not task.enabled or task.manual_only or not task.schedule:
            return None
        cursor = _normalize_datetime(after).replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = cursor + timedelta(days=370)
        while cursor <= limit:
            if _cron_matches(task.schedule, cursor):
                return cursor
            cursor += timedelta(minutes=1)
        raise ValueError(f"No next run found within one year for {task_id}: {task.schedule}")


def load_task_registry(path: str | Path = DEFAULT_TASKER_CONFIG) -> TaskRegistry:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("tasker config must contain a tasks list")
    return TaskRegistry(_parse_task(item) for item in raw_tasks)


def _parse_task(item: object) -> TaskDefinition:
    if not isinstance(item, dict):
        raise ValueError("task entries must be mappings")
    task_id = str(item["id"])
    command = _parse_command(item["command"])
    return TaskDefinition(
        id=task_id,
        label=str(item.get("label") or task_id),
        command=command,
        schedule=str(item["schedule"]) if item.get("schedule") else None,
        enabled=bool(item.get("enabled", True)),
        manual_only=bool(item.get("manual_only", False)),
        timeout_seconds=int(item.get("timeout_seconds", 300)),
        description=str(item["description"]) if item.get("description") is not None else None,
        auto_retry=bool(item.get("auto_retry", False)),
    )


def _parse_command(raw: object) -> list[str]:
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list):
        return [str(part) for part in raw]
    raise ValueError("task command must be a string or list")


def _validate_command(command: list[str]) -> None:
    if len(command) != 2 or command[0] != "make":
        raise ValueError("Only make targets are allowed in tasker registry commands")


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _cron_matches(expr: str, value: datetime) -> bool:
    minute, hour, day, month, weekday = expr.split()
    return (
        _field_matches(minute, value.minute, 0, 59)
        and _field_matches(hour, value.hour, 0, 23)
        and _field_matches(day, value.day, 1, 31)
        and _field_matches(month, value.month, 1, 12)
        and _field_matches(weekday, (value.weekday() + 1) % 7, 0, 6)
    )


def _field_matches(field: str, value: int, minimum: int, maximum: int) -> bool:
    if field == "*":
        return True
    allowed: set[int] = set()
    for part in field.split(","):
        if part.startswith("*/"):
            step = int(part[2:])
            allowed.update(range(minimum, maximum + 1, step))
        elif "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            allowed.update(range(start, end + 1))
        else:
            allowed.add(int(part))
    return value in allowed
