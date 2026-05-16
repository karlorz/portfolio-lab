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
    "portfolio-lab-unified-dashboard",
]

# Expected max duration per job (seconds). Exceeding 2x this triggers alerts.
CRON_EXPECTED_DURATIONS = {
    "portfolio-lab-data": 300,      # 5 min — API calls + processing
    "portfolio-lab-dashboard": 120, # 2 min — static generation
    "portfolio-lab-health": 60,     # 1 min — lightweight check
    "portfolio-lab-eval": 600,      # 10 min — iterates all portfolios
    "portfolio-lab-research": 300,  # 5 min — research loops
    "portfolio-lab-wiki-sync": 120, # 2 min — git operations
    "portfolio-lab-build": 600,     # 10 min — tsc + bun build
    "portfolio-lab-position-sync": 300,  # 5 min — broker API
    "portfolio-lab-overlay-signals": 600,  # 10 min — 5 sequential modules
    "portfolio-lab-overlay-dashboard": 120,  # 2 min — JSON serialization
    "portfolio-lab-unified-dashboard": 120,  # 2 min — JSON serialization
}

# Guard configuration (applied by scripts/cron_guard.sh)
CRON_GUARD_CONFIG = {
    "max_load": 5,              # Defer if 1-min loadavg exceeds this
    "default_timeout": 600,     # Hard kill after N seconds
    "memory_mb": 3072,          # ulimit -v in MB (3GB)
    "lock_dir": "/tmp/portfolio-lab-locks",
}

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
