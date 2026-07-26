from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

from src.tasker.registry import load_task_registry
from tests.makefile_helpers import makefile_recipe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )


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
        "portfolio-lab-prune-logs",
        "portfolio-lab-build",
        "portfolio-lab-autonomous-agent",
        "portfolio-lab-prod-ideas",
        "portfolio-lab-fetch-trends",
        "portfolio-lab-daily-brief",
    }

    assert {task.id for task in registry.tasks} == expected_ids
    assert registry.get("portfolio-lab-health").command == ["make", "health"]
    assert registry.get("portfolio-lab-health").schedule == "0,30 * * * *"
    assert registry.get("portfolio-lab-build").enabled is False
    assert registry.get("portfolio-lab-build").manual_only is True
    assert registry.get("portfolio-lab-autonomous-agent").enabled is False
    assert registry.get("portfolio-lab-autonomous-agent").manual_only is True


def test_make_health_target_runs_system_health_check_module():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile_recipe(makefile, "health")

    assert "src.monitor.health_check" in recipe
    assert "src.monitor.rebalance_health" not in recipe


def test_make_rebalance_health_target_runs_rebalance_health_exporter():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile_recipe(makefile, "rebalance-health")

    assert "src.monitor.rebalance_health" in recipe
    assert "portfolio-lab-rebalance-health" in recipe


def test_make_daily_brief_target_reports_cron_status():
    """daily-brief must follow the dual-mode cron_status reporting contract.

    Previously the target was a bare stub that saved the JSON but never wrote
    cron_status, so the brief froze at 2026-07-04 while ops stayed green.
    The cron dual-mode wire (Makefile + crontab + cron_compat + tasker.yaml)
    requires the standard START/EXIT/STATUS/DUR/CRON_UPDATE recipe shape.
    """
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile_recipe(makefile, "daily-brief")

    assert "src.monitor.daily_brief" in recipe
    assert "--save" in recipe
    assert "portfolio-lab-daily-brief" in recipe
    assert "CRON_UPDATE" in recipe
    assert "STATUS" in recipe


def test_daily_brief_tasker_entry_uses_hourly_25_schedule():
    """tasker SoT schedule for daily-brief: :25 hourly (after dashboard :15)."""
    registry = load_task_registry()
    task = registry.get("portfolio-lab-daily-brief")

    assert task.enabled is True
    assert task.manual_only is False
    assert task.schedule == "25 * * * *"
    assert task.command == ["make", "daily-brief"]
    assert task.timeout_seconds > 0


def test_make_help_describes_system_and_rebalance_health_targets():
    result = _run_make("help")

    assert result.returncode == 0, result.stderr
    assert "make health" in result.stdout
    assert "health.json" in result.stdout
    assert "make rebalance-health" in result.stdout
    assert "rebalance_health.json" in result.stdout


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
