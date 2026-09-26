"""Unit tests for fail-closed env_literal_one helper."""

from __future__ import annotations

import pytest

from src.env_flags import env_literal_one

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
