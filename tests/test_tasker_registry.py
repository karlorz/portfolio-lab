from datetime import datetime, timezone

import pytest

from src.tasker.registry import load_task_registry


def test_default_registry_loads_expected_portfolio_lab_tasks():
    registry = load_task_registry()

    expected_ids = {
        "portfolio-lab-health",
        "portfolio-lab-data",
        "portfolio-lab-dashboard",
        "portfolio-lab-eval",
        "portfolio-lab-research",
        "portfolio-lab-wiki-sync",
        "portfolio-lab-overlay-signals",
        "portfolio-lab-overlay-dashboard",
        "portfolio-lab-attribution",
        "portfolio-lab-unified-dashboard",
        "portfolio-lab-position-sync",
        "portfolio-lab-garch-risk",
        "portfolio-lab-daily-pnl",
        "portfolio-lab-build",
        "portfolio-lab-autonomous-agent",
    }

    assert {task.id for task in registry.tasks} == expected_ids
    assert registry.get("portfolio-lab-health").command == ["make", "health"]
    assert registry.get("portfolio-lab-health").schedule == "0,30 * * * *"
    assert registry.get("portfolio-lab-build").enabled is False
    assert registry.get("portfolio-lab-build").manual_only is True
    assert registry.get("portfolio-lab-autonomous-agent").enabled is False
    assert registry.get("portfolio-lab-autonomous-agent").manual_only is True


def test_registry_rejects_non_make_commands(tmp_path):
    config_path = tmp_path / "tasker.yaml"
    config_path.write_text(
        """
tasks:
  - id: unsafe
    label: Unsafe
    command: "python -c 'print(1)'"
    schedule: "* * * * *"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only make targets"):
        load_task_registry(config_path)


def test_registry_calculates_next_run_for_supported_cron_expressions():
    registry = load_task_registry()
    after = datetime(2026, 6, 10, 10, 4, tzinfo=timezone.utc)

    assert registry.next_run_after("portfolio-lab-data", after).isoformat() == "2026-06-10T10:05:00+00:00"
    assert registry.next_run_after("portfolio-lab-health", after).isoformat() == "2026-06-10T10:30:00+00:00"
    assert registry.next_run_after("portfolio-lab-eval", after).isoformat() == "2026-06-10T10:20:00+00:00"
    assert registry.next_run_after("portfolio-lab-build", after) is None


def test_registry_reports_due_tasks_without_manual_disabled_entries():
    registry = load_task_registry()
    now = datetime(2026, 6, 10, 10, 30, tzinfo=timezone.utc)

    due_ids = {task.id for task in registry.due_tasks(now)}

    assert "portfolio-lab-health" in due_ids
    assert "portfolio-lab-build" not in due_ids
    assert "portfolio-lab-autonomous-agent" not in due_ids
