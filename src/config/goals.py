#!/usr/bin/env python3
"""
v7.10: Investment Goal Context Configuration

Reads data/goals.json and provides goal-aware risk budget scaling
for the ensemble voter and optimizer.

Usage:
    from src.config.goals import load_goals, get_risk_budget_multiplier

    goals = load_goals()
    risk_mult = get_risk_budget_multiplier(goals)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DATA_DIR = Path("~/projects/portfolio-lab/data").expanduser()

GOAL_TYPES = {
    "retirement", "major_purchase", "education", "emergency_fund",
    "income_generation", "wealth_preservation", "fire",
}

LIFE_STAGES = {"accumulation", "pre_retirement", "retirement", "decumulation"}

ACCOUNT_TYPES = {"taxable", "ira", "roth_ira", "401k", "trust", "joint"}

DEFAULT_GOALS: Dict[str, Any] = {
    "version": 1,
    "goals": [],
    "profile": {
        "life_stage": "accumulation",
        "account_type": "taxable",
        "risk_tolerance": "moderate",
    },
}


def load_goals(path: Path | None = None) -> Dict[str, Any]:
    """Load goals config from JSON file, returning defaults on failure."""
    filepath = path or (DATA_DIR / "goals.json")
    try:
        if not filepath.exists():
            logger.info("goals.json not found at %s, using defaults", filepath)
            return dict(DEFAULT_GOALS)
        with open(filepath) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "version" not in data:
            logger.warning("Invalid goals.json structure, using defaults")
            return dict(DEFAULT_GOALS)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read goals.json: %s, using defaults", e)
        return dict(DEFAULT_GOALS)


def validate_goals(goals: Dict[str, Any]) -> List[str]:
    """Validate goals config, returning list of error messages."""
    errors = []
    for i, goal in enumerate(goals.get("goals", [])):
        if goal.get("type") not in GOAL_TYPES:
            errors.append(f"Goal {i}: invalid type '{goal.get('type')}', must be one of {GOAL_TYPES}")
        if not goal.get("priority"):
            errors.append(f"Goal {i}: missing 'priority' field")
        horizon = goal.get("time_horizon_years")
        if horizon is not None and (not isinstance(horizon, (int, float)) or horizon <= 0):
            errors.append(f"Goal {i}: time_horizon_years must be positive, got {horizon}")
    profile = goals.get("profile", {})
    if profile.get("life_stage") not in LIFE_STAGES:
        errors.append(f"Profile: invalid life_stage '{profile.get('life_stage')}', must be one of {LIFE_STAGES}")
    return errors


def get_risk_budget_multiplier(goals: Dict[str, Any]) -> float:
    """Compute risk budget multiplier based on goals and life stage.

    Returns a value in [0.5, 1.0] that scales the ensemble voter's risk budget.
    """
    profile = goals.get("profile", {})
    life_stage = profile.get("life_stage", "accumulation")

    # Base multiplier by life stage
    stage_mult = {
        "accumulation": 1.0,
        "pre_retirement": 0.8,
        "retirement": 0.7,
        "decumulation": 0.7,
    }.get(life_stage, 1.0)

    # Check for short-horizon major purchases
    for goal in goals.get("goals", []):
        if goal.get("type") == "major_purchase":
            horizon = goal.get("time_horizon_years", 10)
            if horizon <= 3:
                stage_mult = min(stage_mult, 0.6)

    # Never drop below 0.5 -- preserve some risk-taking
    return max(stage_mult, 0.5)


def get_account_type_modifier(goals: Dict[str, Any]) -> Dict[str, Any]:
    """Return modifiers based on account type for tax-aware features.

    Returns dict with 'enable_tax_aware' boolean.
    """
    profile = goals.get("profile", {})
    account_type = profile.get("account_type", "taxable")

    is_taxable = account_type in ("taxable", "joint", "trust")
    return {
        "enable_tax_aware": is_taxable,
        "account_type": account_type,
    }
