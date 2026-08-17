"""Unit tests for src.tasker.models (Item Q52).

Tests cover:
- Status constants (RUN_PENDING, RUN_RUNNING, RUN_SUCCESS, RUN_ERROR, RUN_TIMEOUT, RUN_CANCELLED, RUN_BLOCKED)
- TERMINAL_RUN_STATUSES set membership
- EXIT_CODE_BLOCKED and INTENTIONAL_BLOCK_TASK_IDS
- Termination cause constants and PLANNED_TERMINATION_CAUSES frozenset
- TaskDefinition immutability (frozen=True)
- TaskDefinition default field values
- TaskDefinition.to_dict serialization
"""

from dataclasses import FrozenInstanceError

import pytest

from src.tasker import models


def test_status_constants() -> None:
    assert models.RUN_PENDING == "pending"
    assert models.RUN_RUNNING == "running"
    assert models.RUN_SUCCESS == "success"
    assert models.RUN_ERROR == "error"
    assert models.RUN_TIMEOUT == "timeout"
    assert models.RUN_CANCELLED == "cancelled"
    assert models.RUN_BLOCKED == "blocked"


def test_terminal_run_statuses() -> None:
    expected = {
        models.RUN_SUCCESS,
        models.RUN_ERROR,
        models.RUN_TIMEOUT,
        models.RUN_CANCELLED,
        models.RUN_BLOCKED,
    }
    assert models.TERMINAL_RUN_STATUSES == expected
    assert models.RUN_PENDING not in models.TERMINAL_RUN_STATUSES
    assert models.RUN_RUNNING not in models.TERMINAL_RUN_STATUSES


def test_exit_code_blocked_and_intentional_block_tasks() -> None:
    assert models.EXIT_CODE_BLOCKED == 2
    assert isinstance(models.INTENTIONAL_BLOCK_TASK_IDS, frozenset)
    assert "portfolio-lab-eval" in models.INTENTIONAL_BLOCK_TASK_IDS


def test_termination_causes() -> None:
    assert models.TERMINATION_CAUSE_SERVICE_RESTART == "service_restart"
    assert models.TERMINATION_CAUSE_OPERATOR_CANCELLED == "operator_cancelled"
    assert models.TERMINATION_CAUSE_UNPLANNED == "unplanned"
    assert isinstance(models.PLANNED_TERMINATION_CAUSES, frozenset)
    assert models.TERMINATION_CAUSE_SERVICE_RESTART in models.PLANNED_TERMINATION_CAUSES
    assert models.TERMINATION_CAUSE_OPERATOR_CANCELLED in models.PLANNED_TERMINATION_CAUSES
    assert models.TERMINATION_CAUSE_UNPLANNED not in models.PLANNED_TERMINATION_CAUSES


def test_task_definition_defaults() -> None:
    task = models.TaskDefinition(
        id="test-task",
        label="Test Task",
        command=["make", "test"],
    )
    assert task.id == "test-task"
    assert task.label == "Test Task"
    assert task.command == ["make", "test"]
    assert task.schedule is None
    assert task.enabled is True
    assert task.manual_only is False
    assert task.timeout_seconds == 300
    assert task.description is None
    assert task.auto_retry is False


def test_task_definition_custom_fields() -> None:
    task = models.TaskDefinition(
        id="custom-task",
        label="Custom Task",
        command=["python3", "script.py"],
        schedule="0 * * * *",
        enabled=False,
        manual_only=True,
        timeout_seconds=600,
        description="A scheduled custom task",
        auto_retry=True,
    )
    assert task.schedule == "0 * * * *"
    assert task.enabled is False
    assert task.manual_only is True
    assert task.timeout_seconds == 600
    assert task.description == "A scheduled custom task"
    assert task.auto_retry is True


def test_task_definition_immutability() -> None:
    task = models.TaskDefinition(
        id="frozen-task",
        label="Frozen Task",
        command=["echo", "1"],
    )
    with pytest.raises(FrozenInstanceError):
        task.enabled = False  # type: ignore


def test_task_definition_to_dict() -> None:
    task = models.TaskDefinition(
        id="dict-task",
        label="Dict Task",
        command=["echo", "hello"],
        schedule="*/5 * * * *",
        enabled=True,
        manual_only=False,
        timeout_seconds=120,
        description="Dict representation test",
        auto_retry=True,
    )
    d = task.to_dict()
    assert d == {
        "id": "dict-task",
        "label": "Dict Task",
        "command": ["echo", "hello"],
        "schedule": "*/5 * * * *",
        "enabled": True,
        "manual_only": False,
        "timeout_seconds": 120,
        "description": "Dict representation test",
        "auto_retry": True,
    }
    # Mutating the returned command list does not mutate the original
    d["command"].append("world")
    assert task.command == ["echo", "hello"]
