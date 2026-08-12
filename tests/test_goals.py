"""Tests for src/config/goals.py — Investment Goal Context Configuration."""

import json
import tempfile
from pathlib import Path
from src.config.goals import (
    load_goals,
    validate_goals,
    get_risk_budget_multiplier,
    get_account_type_modifier,
    DEFAULT_GOALS,
    GOAL_TYPES,
    LIFE_STAGES,
    ACCOUNT_TYPES,
)


class TestLoadGoals:
    """Tests for load_goals()."""

    def test_returns_defaults_when_file_missing(self):
        result = load_goals(Path("/nonexistent/goals.json"))
        assert result == DEFAULT_GOALS

    def test_loads_valid_json_file(self):
        data = {
            "version": 2,
            "goals": [{"type": "retirement", "priority": "high", "time_horizon_years": 20}],
            "profile": {"life_stage": "accumulation", "account_type": "ira"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_goals(Path(f.name))
        Path(f.name).unlink()
        assert result["version"] == 2
        assert len(result["goals"]) == 1

    def test_returns_defaults_on_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            result = load_goals(Path(f.name))
        Path(f.name).unlink()
        assert result == DEFAULT_GOALS

    def test_returns_defaults_on_missing_version_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"goals": []}, f)
            f.flush()
            result = load_goals(Path(f.name))
        Path(f.name).unlink()
        assert result == DEFAULT_GOALS

    def test_returns_defaults_on_non_dict_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            result = load_goals(Path(f.name))
        Path(f.name).unlink()
        assert result == DEFAULT_GOALS


class TestValidateGoals:
    """Tests for validate_goals()."""

    def test_valid_goals_no_errors(self):
        goals = {
            "version": 1,
            "goals": [
                {"type": "retirement", "priority": "high", "time_horizon_years": 25},
                {"type": "education", "priority": "medium", "time_horizon_years": 10},
            ],
            "profile": {"life_stage": "accumulation"},
        }
        errors = validate_goals(goals)
        assert errors == []

    def test_invalid_goal_type(self):
        goals = {
            "version": 1,
            "goals": [{"type": "invalid_type", "priority": "high"}],
        }
        errors = validate_goals(goals)
        assert any("invalid type" in e for e in errors)

    def test_missing_priority(self):
        goals = {
            "version": 1,
            "goals": [{"type": "retirement"}],
        }
        errors = validate_goals(goals)
        assert any("priority" in e for e in errors)

    def test_negative_time_horizon(self):
        goals = {
            "version": 1,
            "goals": [{"type": "retirement", "priority": "high", "time_horizon_years": -5}],
        }
        errors = validate_goals(goals)
        assert any("time_horizon_years" in e for e in errors)

    def test_zero_time_horizon(self):
        goals = {
            "version": 1,
            "goals": [{"type": "retirement", "priority": "high", "time_horizon_years": 0}],
        }
        errors = validate_goals(goals)
        assert any("time_horizon_years" in e for e in errors)

    def test_none_time_horizon_is_valid(self):
        goals = {
            "version": 1,
            "goals": [{"type": "retirement", "priority": "high", "time_horizon_years": None}],
        }
        errors = validate_goals(goals)
        assert all("time_horizon_years" not in e for e in errors)

    def test_missing_time_horizon_is_valid(self):
        goals = {
            "version": 1,
            "goals": [{"type": "retirement", "priority": "high"}],
        }
        errors = validate_goals(goals)
        assert all("time_horizon_years" not in e for e in errors)

    def test_invalid_life_stage(self):
        goals = {
            "version": 1,
            "goals": [],
            "profile": {"life_stage": "not_a_stage"},
        }
        errors = validate_goals(goals)
        assert any("life_stage" in e for e in errors)

    def test_empty_goals_list(self):
        goals = {
            "version": 1,
            "goals": [],
            "profile": {"life_stage": "accumulation"},
        }
        errors = validate_goals(goals)
        assert errors == []

    def test_missing_profile_section(self):
        goals = {"version": 1, "goals": []}
        errors = validate_goals(goals)
        # profile missing → life_stage defaults to None → flagged as invalid
        assert len(errors) == 1
        assert "life_stage" in errors[0]


class TestRiskBudgetMultiplier:
    """Tests for get_risk_budget_multiplier()."""

    def test_accumulation_returns_1_0(self):
        goals = {"profile": {"life_stage": "accumulation"}}
        assert get_risk_budget_multiplier(goals) == 1.0

    def test_pre_retirement_returns_0_8(self):
        goals = {"profile": {"life_stage": "pre_retirement"}}
        assert get_risk_budget_multiplier(goals) == 0.8

    def test_retirement_returns_0_7(self):
        goals = {"profile": {"life_stage": "retirement"}}
        assert get_risk_budget_multiplier(goals) == 0.7

    def test_decumulation_returns_0_7(self):
        goals = {"profile": {"life_stage": "decumulation"}}
        assert get_risk_budget_multiplier(goals) == 0.7

    def test_unknown_life_stage_defaults_to_1_0(self):
        goals = {"profile": {"life_stage": "unknown"}}
        assert get_risk_budget_multiplier(goals) == 1.0

    def test_missing_profile_defaults_to_1_0(self):
        goals = {}
        assert get_risk_budget_multiplier(goals) == 1.0

    def test_short_horizon_major_purchase_caps_at_0_6(self):
        goals = {
            "profile": {"life_stage": "accumulation"},
            "goals": [{"type": "major_purchase", "time_horizon_years": 2}],
        }
        assert get_risk_budget_multiplier(goals) == 0.6

    def test_long_horizon_major_purchase_no_cap(self):
        goals = {
            "profile": {"life_stage": "accumulation"},
            "goals": [{"type": "major_purchase", "time_horizon_years": 5}],
        }
        assert get_risk_budget_multiplier(goals) == 1.0

    def test_never_below_0_5(self):
        goals = {
            "profile": {"life_stage": "retirement"},
            "goals": [{"type": "major_purchase", "time_horizon_years": 1}],
        }
        # retirement=0.7, then min(0.7, 0.6)=0.6, then max(0.6, 0.5)=0.6
        assert get_risk_budget_multiplier(goals) >= 0.5

    def test_multiple_goals_with_non_purchase(self):
        goals = {
            "profile": {"life_stage": "accumulation"},
            "goals": [
                {"type": "retirement", "time_horizon_years": 20},
                {"type": "education", "time_horizon_years": 5},
            ],
        }
        result = get_risk_budget_multiplier(goals)
        assert result == 1.0  # Neither short-horizon purchase caps it


class TestAccountTypeModifier:
    """Tests for get_account_type_modifier()."""

    def test_taxable_enables_tax_aware(self):
        goals = {"profile": {"account_type": "taxable"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is True
        assert result["account_type"] == "taxable"

    def test_joint_enables_tax_aware(self):
        goals = {"profile": {"account_type": "joint"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is True

    def test_trust_enables_tax_aware(self):
        goals = {"profile": {"account_type": "trust"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is True

    def test_ira_disables_tax_aware(self):
        goals = {"profile": {"account_type": "ira"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is False

    def test_roth_ira_disables_tax_aware(self):
        goals = {"profile": {"account_type": "roth_ira"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is False

    def test_401k_disables_tax_aware(self):
        goals = {"profile": {"account_type": "401k"}}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is False

    def test_default_no_profile_is_taxable(self):
        goals = {}
        result = get_account_type_modifier(goals)
        assert result["enable_tax_aware"] is True
        assert result["account_type"] == "taxable"


class TestConstants:
    """Verify constant sets are as expected."""

    def test_goal_types(self):
        assert "retirement" in GOAL_TYPES
        assert "fire" in GOAL_TYPES
        assert len(GOAL_TYPES) == 7

    def test_life_stages(self):
        assert "accumulation" in LIFE_STAGES
        assert "decumulation" in LIFE_STAGES
        assert len(LIFE_STAGES) == 4

    def test_account_types(self):
        assert "taxable" in ACCOUNT_TYPES
        assert "401k" in ACCOUNT_TYPES
        assert len(ACCOUNT_TYPES) == 6
