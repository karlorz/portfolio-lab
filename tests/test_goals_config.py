"""
Tests for Investment Goal Context config reader.
Covers: schema loading, risk scaling, missing file fallback, invalid JSON.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.goals import (
    load_goals,
    validate_goals,
    get_risk_budget_multiplier,
    get_account_type_modifier,
    GOAL_TYPES,
    LIFE_STAGES,
)


@pytest.fixture
def sample_goals():
    return {
        "version": 1,
        "goals": [
            {
                "id": "house",
                "type": "major_purchase",
                "label": "House down payment",
                "target_amount": 200000,
                "time_horizon_years": 5,
                "priority": "high",
            }
        ],
        "profile": {
            "life_stage": "accumulation",
            "account_type": "taxable",
            "risk_tolerance": "moderate",
        },
    }


@pytest.fixture
def retirement_goals():
    return {
        "version": 1,
        "goals": [
            {
                "id": "retire",
                "type": "retirement",
                "label": "Retirement",
                "target_amount": 2000000,
                "time_horizon_years": 20,
                "priority": "high",
            }
        ],
        "profile": {
            "life_stage": "pre_retirement",
            "account_type": "ira",
            "risk_tolerance": "conservative",
        },
    }


class TestLoadGoals:
    def test_loads_valid_goals_file(self, sample_goals, tmp_path):
        goals_path = tmp_path / "goals.json"
        goals_path.write_text(json.dumps(sample_goals))
        result = load_goals(goals_path)
        assert result["version"] == 1
        assert len(result["goals"]) == 1

    def test_returns_defaults_when_file_missing(self, tmp_path):
        goals_path = tmp_path / "nonexistent.json"
        result = load_goals(goals_path)
        assert result["version"] == 1
        assert result["goals"] == []
        assert result["profile"]["life_stage"] == "accumulation"

    def test_returns_defaults_on_invalid_json(self, tmp_path):
        goals_path = tmp_path / "goals.json"
        goals_path.write_text("not valid json {{{")
        result = load_goals(goals_path)
        assert result["version"] == 1
        assert result["goals"] == []


class TestValidateGoals:
    def test_valid_goals_pass(self, sample_goals):
        errors = validate_goals(sample_goals)
        assert len(errors) == 0

    def test_rejects_invalid_goal_type(self, sample_goals):
        sample_goals["goals"][0]["type"] = "invalid_type"
        errors = validate_goals(sample_goals)
        assert len(errors) == 1
        assert "invalid_type" in errors[0]

    def test_rejects_unknown_life_stage(self, sample_goals):
        sample_goals["profile"]["life_stage"] = "adolescence"
        errors = validate_goals(sample_goals)
        assert len(errors) == 1

    def test_rejects_missing_priority(self, sample_goals):
        del sample_goals["goals"][0]["priority"]
        errors = validate_goals(sample_goals)
        assert len(errors) == 1

    def test_rejects_negative_horizon(self, sample_goals):
        sample_goals["goals"][0]["time_horizon_years"] = -1
        errors = validate_goals(sample_goals)
        assert len(errors) == 1


class TestRiskBudgetMultiplier:
    def test_accumulation_with_long_horizon(self, sample_goals):
        mult = get_risk_budget_multiplier(sample_goals)
        assert mult == 1.0

    def test_pre_retirement_reduces_risk(self, retirement_goals):
        mult = get_risk_budget_multiplier(retirement_goals)
        assert mult == 0.8

    def test_major_purchase_short_horizon_reduces_risk(self, sample_goals):
        sample_goals["goals"][0]["time_horizon_years"] = 2
        mult = get_risk_budget_multiplier(sample_goals)
        assert mult == 0.6

    def test_retirement_life_stage(self, retirement_goals):
        retirement_goals["profile"]["life_stage"] = "retirement"
        mult = get_risk_budget_multiplier(retirement_goals)
        assert mult == 0.7

    def test_empty_goals_returns_default(self):
        goals = {
            "version": 1,
            "goals": [],
            "profile": {"life_stage": "accumulation", "account_type": "taxable", "risk_tolerance": "moderate"},
        }
        mult = get_risk_budget_multiplier(goals)
        assert mult == 1.0


class TestAccountTypeModifier:
    def test_taxable_enables_tax_aware(self, sample_goals):
        modifier = get_account_type_modifier(sample_goals)
        assert modifier["enable_tax_aware"] is True

    def test_ira_disables_tax_aware(self, retirement_goals):
        modifier = get_account_type_modifier(retirement_goals)
        assert modifier["enable_tax_aware"] is False

    def test_roth_similar_to_ira(self, sample_goals):
        sample_goals["profile"]["account_type"] = "roth_ira"
        modifier = get_account_type_modifier(sample_goals)
        assert modifier["enable_tax_aware"] is False
