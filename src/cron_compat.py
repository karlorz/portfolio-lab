"""
Cron backend compatibility — feature flag for dual-mode operation.

portfolio-lab supports three cron backends:
  - hermes   (Hermes Agent cron scheduler, 11 jobs active)
  - crontab  (system crontab, standalone without Hermes)
  - manual   (make <target> from terminal or Claude Code)

Set CRON_BACKEND in .env or export it before running Makefile targets.
All cron-executed code imports from here and branches on IS_HERMES for
logging, notifications, and state persistence paths that differ between
Hermes and non-Hermes environments.
"""

import os

BACKEND: str = os.getenv("CRON_BACKEND", "hermes")
IS_HERMES: bool = BACKEND == "hermes"
IS_CRONTAB: bool = BACKEND == "crontab"
IS_MANUAL: bool = BACKEND == "manual" or BACKEND == "claude-code"

# Cron targets that must stay in sync across all backends.
# When adding a new cron job, append its name here AND add:
#   - a Makefile target
#   - a crontab entry in crontab file
CRON_TARGETS = [
    "portfolio-lab-data",
    "portfolio-lab-dashboard",
    "portfolio-lab-health",
    "portfolio-lab-eval",
    "portfolio-lab-research",
    "portfolio-lab-wiki-sync",
    "portfolio-lab-build",
    "portfolio-lab-position-sync",
    "portfolio-lab-overlay-signals",
    "portfolio-lab-overlay-dashboard",
]

def active_backend() -> str:
    """Return the currently active cron backend. Discoverable at runtime."""
    return BACKEND

def cron_status_path() -> str:
    """Return the path to the cron status file (backend-agnostic)."""
    import sys
    from pathlib import Path
    # Resolve relative to project root regardless of cwd
    this_dir = Path(__file__).resolve().parent.parent
    return str(this_dir / "data" / "cron_status.json")
