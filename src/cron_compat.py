"""
Cron backend compatibility — feature flag for dual-mode operation.

portfolio-lab supports three cron backends:
  - hermes   (Hermes Agent cron scheduler, 12 targets registered)
  - crontab  (system crontab, standalone without Hermes)
  - tasker   (project-local tasker service)
  - manual   (make <target> from terminal or Claude Code)

Set CRON_BACKEND in .env or export it before running Makefile targets.
All cron-executed code imports from here and branches on IS_HERMES for
logging, notifications, and state persistence paths that differ between
Hermes and non-Hermes environments.
"""

import os
from src.paths import DATA_DIR, LOCK_DIR

BACKEND: str = os.getenv("CRON_BACKEND", "hermes")
IS_HERMES: bool = BACKEND == "hermes"
IS_CRONTAB: bool = BACKEND == "crontab"
IS_TASKER: bool = BACKEND == "tasker"
IS_MANUAL: bool = BACKEND == "manual" or BACKEND == "claude-code"

# Cron targets that must stay in sync across all backends.
# When adding a new cron job, append its name here AND add:
#   - a Makefile target
#   - a crontab entry in crontab file
CRON_TARGETS = [
    "portfolio-lab-data",
    "portfolio-lab-dashboard",
    "portfolio-lab-eval",
    "portfolio-lab-research",
    "portfolio-lab-wiki-sync",
    "portfolio-lab-position-sync",
    "portfolio-lab-overlay-signals",
    "portfolio-lab-overlay-dashboard",
    "portfolio-lab-garch-risk",
    "portfolio-lab-mark-to-market",
    "portfolio-lab-daily-pnl",
    "portfolio-lab-attribution",
    "portfolio-lab-unified-dashboard",
    "portfolio-lab-health",
    "portfolio-lab-prune-logs",
    "portfolio-lab-prod-ideas",
    "portfolio-lab-fetch-trends",
    "portfolio-lab-daily-brief",
]

# S18b: optional suite segments — NOT production cron jobs (not in CRON_TARGETS /
# tasker / cron_status). Documented Makefile targets + commented crontab fallback
# only. Enable carefully: full suite ~28m under 6GB; unit segment is lighter.
# Mapping: job-id → Makefile target
OPTIONAL_SUITE_TARGETS = {
    "portfolio-lab-test-unit": "test-unit",       # generator+integration ignored
    "portfolio-lab-test-generator": "test-generator",
    "portfolio-lab-test-full": "test",            # full safe gate (3600s)
}

# Expected max duration per job (seconds). Exceeding 2x this triggers alerts.
CRON_EXPECTED_DURATIONS = {
    "portfolio-lab-data": 300,      # 5 min — bun fetch-data
    "portfolio-lab-dashboard": 180, # 3 min — static generation (Batch II DF3; wall ~116s)
    "portfolio-lab-eval": 600,      # 10 min — iterates all portfolios
    "portfolio-lab-research": 300,  # 5 min — research loops
    "portfolio-lab-wiki-sync": 120, # 2 min — git operations
    "portfolio-lab-position-sync": 60,   # 1 min — placeholder/no-op
    "portfolio-lab-overlay-signals": 600,  # 10 min — 6 sequential modules (+ alternative_data)
    "portfolio-lab-overlay-dashboard": 120,  # 2 min — JSON serialization
    "portfolio-lab-garch-risk": 120,  # 2 min — GARCH-CVaR computation
    "portfolio-lab-mark-to-market": 15,   # 15 sec — price update from prices.json
    "portfolio-lab-daily-pnl": 30,   # 30 sec — snapshot from portfolio state
    "portfolio-lab-attribution": 300,  # 5 min — attribution + adaptive weights
    "portfolio-lab-unified-dashboard": 120,  # 2 min — JSON serialization
    "portfolio-lab-health": 90,  # expected ~13–90s; wall 120 (Batch JO HT1)
    "portfolio-lab-prune-logs": 60,  # 1 min — bound tasker_logs growth + dead-log removal
    "portfolio-lab-prod-ideas": 60,  # 1 min — scan ops SSOT → channel delta (ML off)
    "portfolio-lab-fetch-trends": 300,  # 5 min — pytrends weekly refresh
    "portfolio-lab-daily-brief": 120,  # 2 min — template sections + optional LLM narrative
}

# Guard configuration (applied by scripts/cron_guard.sh)
CRON_GUARD_CONFIG = {
    "max_load": 5,              # Defer if 1-min loadavg exceeds this
    "default_timeout": 600,     # Hard kill after N seconds
    "memory_mb": 3072,          # ulimit -v in MB (3GB)
    "lock_dir": str(LOCK_DIR),
}

def active_backend() -> str:
    """Return the currently active cron backend. Discoverable at runtime."""
    return BACKEND

def cron_status_path() -> str:
    """Return the path to the cron status file (backend-agnostic)."""
    return str(DATA_DIR / "cron_status.json")
