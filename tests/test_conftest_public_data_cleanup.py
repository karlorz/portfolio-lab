"""Regression coverage for bare-pytest PUBLIC_DATA_DIR cleanup."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.synthetic_prices import (
    SYNTHETIC_PRICE_END,
    SYNTHETIC_PRICE_START,
)


def _project_conftest(pytestconfig):
    for _, plugin in pytestconfig.pluginmanager.list_name_plugin():
        plugin_file = getattr(plugin, "__file__", "")
        if plugin_file and Path(plugin_file).resolve() == Path(__file__).with_name(
            "conftest.py"
        ).resolve():
            return plugin
    raise AssertionError("project tests/conftest.py plugin was not loaded")


def test_owned_public_data_root_is_removed_at_session_finish(
    tmp_path: Path, monkeypatch, pytestconfig,
) -> None:
    conftest = _project_conftest(pytestconfig)

    owned_root = tmp_path / "plab-pytest-public-owned"
    public = owned_root / "data"
    public.mkdir(parents=True)
    (public / "prices.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_ROOT", owned_root)
    # Never let this mid-suite regression test touch the session-owned DB root.
    monkeypatch.setattr(conftest, "_ISOLATED_MARKET_DB_ROOT", None)
    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert not owned_root.exists()


def test_caller_provided_public_data_dir_is_not_removed(
    tmp_path: Path, monkeypatch, pytestconfig,
) -> None:
    conftest = _project_conftest(pytestconfig)

    caller_root = tmp_path / "caller-owned"
    caller_root.mkdir()
    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_DIR", caller_root)
    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_ROOT", None)
    monkeypatch.setattr(conftest, "_ISOLATED_MARKET_DB_ROOT", None)

    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert caller_root.is_dir()


def test_isolated_prices_are_seeded_synthetically(
    tmp_path: Path, pytestconfig,
) -> None:
    conftest = _project_conftest(pytestconfig)
    public = tmp_path / "isolated-public"
    public.mkdir()

    conftest._seed_isolated_public_fixtures(public)

    prices = json.loads((public / "prices.json").read_text(encoding="utf-8"))
    assert {"SPY", "GLD", "TLT", "IEF", "BTC", "^VIX", "^VIX3M"} <= prices.keys()
    assert prices["SPY"][0]["d"] == SYNTHETIC_PRICE_START.isoformat()
    assert prices["SPY"][-1]["d"] == SYNTHETIC_PRICE_END.isoformat()
    assert len(prices["SPY"]) >= 20 * 252
