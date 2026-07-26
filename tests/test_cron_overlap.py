"""Tests for live Hermes/system-crontab/tasker overlap detection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAP_SCRIPT = PROJECT_ROOT / "scripts" / "detect_cron_overlap.py"


def _load_overlap_module():
    spec = importlib.util.spec_from_file_location("detect_cron_overlap", OVERLAP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detects_overlap_between_active_crontab_and_hermes_jobs() -> None:
    """Active crontab jobs that also appear in Hermes should be reported."""
    overlap = _load_overlap_module()
    crontab_text = """
# commented entries are fallback documentation only
# 7,37 * * * * CRON_BACKEND=crontab make -C /root/projects/portfolio-lab dashboard
5 * * * * CRON_BACKEND=crontab make -C /root/projects/portfolio-lab data
10,25,40,55 * * * * CRON_BACKEND=crontab make -C /root/projects/portfolio-lab unified-dashboard
"""
    hermes_text = """
  3aed5ba90876 [active]
    Name:      portfolio-lab-data
    Last run:  2026-06-08T12:13:10.574483+08:00  ok

  b1648b3fbc3a [active]
    Name:      portfolio-lab-dashboard
    Last run:  2026-06-08T12:15:22.809918+08:00  ok
"""

    assert overlap.crontab_jobs_from_text(crontab_text) == {
        "portfolio-lab-data",
        "portfolio-lab-unified-dashboard",
    }
    assert overlap.hermes_jobs_from_text(hermes_text) == {
        "portfolio-lab-data",
        "portfolio-lab-dashboard",
    }
    assert overlap.find_overlap(crontab_text, hermes_text) == {"portfolio-lab-data"}


def test_tasker_jobs_from_cron_status_enabled_only(tmp_path: Path) -> None:
    overlap = _load_overlap_module()
    path = tmp_path / "cron_status.json"
    path.write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-data",
                        "enabled": True,
                        "manual_only": False,
                        "schedule": "5 * * * *",
                    },
                    {
                        "name": "portfolio-lab-eval",
                        "enabled": False,
                        "manual_only": False,
                        "schedule": "20 */2 * * *",
                    },
                    {
                        "name": "portfolio-lab-build",
                        "enabled": True,
                        "manual_only": True,
                        "schedule": None,
                    },
                    {
                        "name": "portfolio-lab-overlay-signals",
                        "enabled": True,
                        "manual_only": False,
                        "schedule": "40 * * * *",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    jobs = overlap.tasker_jobs_from_cron_status(path)
    assert jobs == {
        "portfolio-lab-data",
        "portfolio-lab-overlay-signals",
    }


def test_multi_backend_overlap_includes_tasker_crontab() -> None:
    overlap = _load_overlap_module()
    crontab = {
        "portfolio-lab-overlay-signals",
        "portfolio-lab-daily-pnl",
    }
    hermes: set[str] = set()
    tasker = {
        "portfolio-lab-overlay-signals",
        "portfolio-lab-data",
    }
    multi = overlap.find_multi_backend_overlaps(
        crontab_jobs=crontab,
        hermes_jobs=hermes,
        tasker_jobs=tasker,
    )
    assert multi["crontab∩tasker"] == {"portfolio-lab-overlay-signals"}
    assert multi["crontab∩hermes"] == set()
    assert multi["tasker∩hermes"] == set()
    assert overlap.any_overlap(multi) is True


def test_no_overlap_when_single_writer() -> None:
    overlap = _load_overlap_module()
    multi = overlap.find_multi_backend_overlaps(
        crontab_jobs={"portfolio-lab-attribution"},
        hermes_jobs=set(),
        tasker_jobs={"portfolio-lab-data"},
    )
    assert overlap.any_overlap(multi) is False
