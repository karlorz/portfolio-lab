"""Tests for scripts/mark_to_market.py price SSOT."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "mark_to_market.py"


def _load_mtm_module():
    spec = importlib.util.spec_from_file_location("mark_to_market_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_load_prices_follows_public_data_dir_env(tmp_path, monkeypatch):
    """PUBLIC_DATA_DIR must select prices SSOT (not hardcoded project public/)."""
    public = tmp_path / "www-data"
    public.mkdir()
    prices_path = public / "prices.json"
    prices_path.write_text(
        json.dumps(
            {
                "SPY": [{"d": "2026-07-15", "p": 111.11}],
                "GLD": [{"d": "2026-07-15", "p": 222.22}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(public))

    # Re-import path resolution as mark_to_market does at call time
    mtm = _load_mtm_module()
    prices = mtm.load_prices()
    assert prices["SPY"] == pytest.approx(111.11)
    assert prices["GLD"] == pytest.approx(222.22)


def test_load_prices_explicit_override_wins(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    path = other / "prices.json"
    path.write_text(json.dumps({"SPY": [{"d": "2026-07-15", "p": 333.33}]}), encoding="utf-8")
    # Even if PUBLIC_DATA_DIR points elsewhere, explicit path wins
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(tmp_path / "unused"))
    mtm = _load_mtm_module()
    prices = mtm.load_prices(prices_file=path)
    assert prices["SPY"] == pytest.approx(333.33)


def test_get_prices_file_matches_paths_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(tmp_path))
    mtm = _load_mtm_module()
    assert mtm.get_prices_file() == tmp_path / "prices.json"


def test_mark_to_market_uses_loaded_prices():
    mtm = _load_mtm_module()
    portfolio = {
        "mode": "paper",
        "cash": 0.0,
        "positions": {
            "SPY": {
                "shares": 10,
                "avg_price": 100.0,
                "current_price": 100.0,
                "value": 1000.0,
                "unrealized_pnl": 0.0,
                "weight": 1.0,
            }
        },
        "history": [],
    }
    updated = mtm.mark_to_market(portfolio, {"SPY": 200.0})
    assert updated["positions"]["SPY"]["current_price"] == 200.0
    assert updated["positions"]["SPY"]["value"] == 2000.0


def test_mark_to_market_declares_tzdata_dependency():
    """tzdata must be listed in pyproject.toml dependencies for systems lacking system tzdata (e.g. Alpine)."""
    import tomllib

    pyproject_path = REPO / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    tzdata_dep = [d for d in deps if d.startswith("tzdata")]
    assert len(tzdata_dep) == 1, "tzdata>=2024.1 must be in project dependencies"
    assert "tzdata>=2024.1" in tzdata_dep[0]


def test_mark_to_market_zoneinfo_resolution():
    """ZoneInfo('America/New_York') must construct successfully."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/New_York")
    assert str(tz) == "America/New_York"
