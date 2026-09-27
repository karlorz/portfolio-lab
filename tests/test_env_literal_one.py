"""Unit tests for fail-closed env_literal_one helper."""

from __future__ import annotations

import pytest

from src.env_flags import env_choice, env_literal_one, env_paper_unless_false

TRUTHY_OFF = ("", "0", "true", "True", "yes", "on", "2", "1 ")


@pytest.mark.parametrize("value", TRUTHY_OFF)
def test_off_for_non_literal_one(monkeypatch, value):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_FLAG", value)
    assert env_literal_one("PORTFOLIO_LAB_TEST_FLAG") is False


def test_on_for_literal_one(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_FLAG", "1")
    assert env_literal_one("PORTFOLIO_LAB_TEST_FLAG") is True


def test_default_off_when_unset(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_LAB_TEST_FLAG", raising=False)
    assert env_literal_one("PORTFOLIO_LAB_TEST_FLAG") is False


def test_mapping_override_ignores_os(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_FLAG", "1")
    assert env_literal_one("PORTFOLIO_LAB_TEST_FLAG", env={"PORTFOLIO_LAB_TEST_FLAG": "true"}) is False
    assert env_literal_one("PORTFOLIO_LAB_TEST_FLAG", env={"PORTFOLIO_LAB_TEST_FLAG": "1"}) is True


def test_empty_default_still_requires_literal_one():
    assert env_literal_one("MISSING", default="", env={}) is False
    assert env_literal_one("MISSING", default="", env={"MISSING": "1"}) is True


PAPER_ON = ("", "true", "True", "1", "yes", "YES", "on", "paper")
PAPER_OFF = ("false", "False", "0", "no", "NO")


@pytest.mark.parametrize("value", PAPER_ON)
def test_paper_unless_false_stays_paper(monkeypatch, value):
    monkeypatch.setenv("ALPACA_PAPER", value)
    assert env_paper_unless_false() is True


@pytest.mark.parametrize("value", PAPER_OFF)
def test_paper_unless_false_explicit_live(monkeypatch, value):
    monkeypatch.setenv("ALPACA_PAPER", value)
    assert env_paper_unless_false() is False


def test_paper_unless_false_default_when_unset(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    assert env_paper_unless_false() is True


def test_paper_unless_false_mapping_override(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert env_paper_unless_false(env={"ALPACA_PAPER": "1"}) is True
    assert env_paper_unless_false(env={"ALPACA_PAPER": "0"}) is False



def test_env_choice_accepts_allowed(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_CHOICE", "probe")
    assert env_choice(
        "PORTFOLIO_LAB_TEST_CHOICE",
        allowed=("publication", "probe"),
        default="publication",
    ) == "probe"


def test_env_choice_unknown_fails_closed(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_CHOICE", "bogus")
    assert env_choice(
        "PORTFOLIO_LAB_TEST_CHOICE",
        allowed=("publication", "probe"),
        default="publication",
    ) == "publication"


def test_env_choice_casefold_returns_canonical(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_LAB_TEST_CHOICE", "LiVe")
    assert env_choice(
        "PORTFOLIO_LAB_TEST_CHOICE",
        allowed=("paper", "live"),
        default="paper",
        casefold=True,
    ) == "live"


def test_env_choice_default_when_unset(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_LAB_TEST_CHOICE", raising=False)
    assert env_choice(
        "PORTFOLIO_LAB_TEST_CHOICE",
        allowed=("paper", "live"),
        default="paper",
    ) == "paper"
