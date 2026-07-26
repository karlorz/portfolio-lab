"""Governed constants for advisory signal-health hard-zero decisions."""

from __future__ import annotations

import os
from typing import Any

HARD_ZERO_ADR_ID = "ADR-006"
HARD_ZERO_ADR_PATH = (
    "projects/portfolio-lab/architecture/"
    "2026-07-25-advisory-signal-hard-zero-policy.md"
)
DEFAULT_UNHEALTHY_MIN_IC = 0.08
DEFAULT_MIN_LABELED_DAILY_COHORTS = 20


def unhealthy_min_ic() -> float:
    """Return the human-approved unhealthy-sleeve IC boundary."""
    try:
        return float(
            os.environ.get(
                "ENSEMBLE_UNHEALTHY_MIN_IC",
                str(DEFAULT_UNHEALTHY_MIN_IC),
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_UNHEALTHY_MIN_IC


def minimum_labeled_daily_cohorts() -> int:
    """Return the evidence floor required before advisory hard-zero."""
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "ENSEMBLE_HARD_ZERO_MIN_COHORTS",
                    str(DEFAULT_MIN_LABELED_DAILY_COHORTS),
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_MIN_LABELED_DAILY_COHORTS


def disclosure() -> dict[str, Any]:
    """Render the public governance contract without importing the voter."""
    return {
        "schema_version": "advisory-hard-zero-policy/v1",
        "decision": HARD_ZERO_ADR_ID,
        "decision_path": HARD_ZERO_ADR_PATH,
        "human_approved": True,
        "approved_at": "2026-07-25",
        "mode": "advisory_only",
        "live_authoritative": False,
        "unhealthy_min_ic": unhealthy_min_ic(),
        "min_labeled_daily_cohorts": minimum_labeled_daily_cohorts(),
        "shadow_collection": True,
        "target_allocations_unchanged": True,
        "reentry": (
            "A newly published daily-cohort report must no longer meet the "
            "hard-zero condition; no auto-invert or live promotion."
        ),
    }
