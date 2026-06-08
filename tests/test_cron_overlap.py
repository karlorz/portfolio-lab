"""Tests for live Hermes/system-crontab overlap detection."""

from __future__ import annotations

import importlib.util
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
