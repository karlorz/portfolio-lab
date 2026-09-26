"""Default-on gates: unset/literal 1 stay on; other strings stay off."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.env_flags import env_literal_one
from src.monitor.decision_registry import record_backtest_experiment
from src.monitor.incident_manager import IncidentManager

NON_LITERAL = ("", "0", "true", "True", "yes", "on", "false", "no", "off", "2", "1 ")


@pytest.mark.parametrize("value", NON_LITERAL)
def test_record_backtest_skips_unless_literal_one(monkeypatch, value):
    monkeypatch.setenv("DECISION_REGISTRY_RECORD_BACKTEST", value)
    reg = MagicMock()
    assert (
        record_backtest_experiment(
            {"sharpe": 1.0},
            experiment_id="exp-default-on",
            registry=reg,
        )
        is None
    )
    reg.record_experiment.assert_not_called()


def test_record_backtest_runs_for_literal_one(monkeypatch):
    monkeypatch.setenv("DECISION_REGISTRY_RECORD_BACKTEST", "1")
    reg = MagicMock()
    reg.record_experiment.return_value = "exp-default-on"
    with patch(
        "src.monitor.decision_registry.evaluate_promotion_candidate",
        return_value={"failures": [], "eligible": True},
    ):
        assert (
            record_backtest_experiment(
                {"sharpe": 1.0},
                experiment_id="exp-default-on",
                registry=reg,
            )
            == "exp-default-on"
        )
    reg.record_experiment.assert_called_once()


def test_record_backtest_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("DECISION_REGISTRY_RECORD_BACKTEST", raising=False)
    reg = MagicMock()
    reg.record_experiment.return_value = "exp-unset"
    with patch(
        "src.monitor.decision_registry.evaluate_promotion_candidate",
        return_value={"failures": [], "eligible": True},
    ):
        assert (
            record_backtest_experiment(
                {"sharpe": 1.0},
                experiment_id="exp-unset",
                registry=reg,
            )
            == "exp-unset"
        )


@pytest.mark.parametrize("value", NON_LITERAL)
def test_alt_data_auto_project_off_unless_literal_one(monkeypatch, value):
    monkeypatch.setenv("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", value)
    assert env_literal_one("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", default="1") is False


def test_alt_data_auto_project_on_for_literal_one_and_default(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", "1")
    assert env_literal_one("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", default="1") is True
    monkeypatch.delenv("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", raising=False)
    assert env_literal_one("PORTFOLIO_LAB_ALT_DATA_AUTO_PROJECT", default="1") is True


@pytest.mark.parametrize("value", NON_LITERAL)
def test_incident_escalation_off_unless_literal_one(tmp_path, monkeypatch, value):
    monkeypatch.setenv("INCIDENT_KILL_SWITCH_ESCALATION_ENABLED", value)
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    assert manager.escalation_enabled is False


def test_incident_escalation_on_for_literal_one_and_default(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENT_KILL_SWITCH_ESCALATION_ENABLED", "1")
    on = IncidentManager(
        log_path=tmp_path / "incidents-on.jsonl",
        summary_path=tmp_path / "incidents-on.json",
    )
    assert on.escalation_enabled is True
    monkeypatch.delenv("INCIDENT_KILL_SWITCH_ESCALATION_ENABLED", raising=False)
    unset = IncidentManager(
        log_path=tmp_path / "incidents-unset.jsonl",
        summary_path=tmp_path / "incidents-unset.json",
    )
    assert unset.escalation_enabled is True
