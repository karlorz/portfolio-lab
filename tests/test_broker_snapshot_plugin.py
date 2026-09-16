"""Opt-in broker-readonly-gateway plugin for daily brief only."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.monitor.broker_snapshot_plugin import (
    ENABLE_ENV,
    broker_snapshot_enabled,
    load_broker_snapshot_section,
)
from src.monitor.daily_brief import BriefSection, generate_brief_sections


@pytest.fixture
def sample_dashboard():
    return {
        "health": {"available": True, "status": "healthy", "alerts": []},
        "portfolio": {
            "available": True,
            "total_value": 250000,
            "positions": [
                {"symbol": "SPY", "weight": 44.0, "value": 110000},
            ],
        },
        "risk": {
            "available": True,
            "current_drawdown": -8.5,
            "var_95_daily": -1.2,
            "volatility_annual": 11.5,
        },
        "overlays": {"_meta": {"active_count": 2, "frozen_count": 0}},
        "regime": {"label": "normal"},
        "tca": {"scorecard": {"avg_slippage_bps": 2.0, "total_orders": 4}},
        "attribution": {
            "sources": [
                {"name": "TSFM Momentum", "total_return_bps": 1.2},
            ]
        },
    }


class _ReadOnlyPlugin:
    plugin_id = "broker_snapshot"
    read_only = True

    def brief_section(self, snapshot_path=None):
        return {
            "title": "Broker snapshot",
            "severity": "normal",
            "text": "Read-only IBKR/Futu snapshot loaded for daily brief.",
            "metadata": {"plugin_id": self.plugin_id, "read_only": True},
        }


class _MutatingPlugin:
    plugin_id = "broker_snapshot"
    read_only = False

    def brief_section(self, snapshot_path=None):
        raise AssertionError("mutating plugin must not be invoked")


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert broker_snapshot_enabled() is False
    assert load_broker_snapshot_section(plugin=_ReadOnlyPlugin()) is None


def test_enabled_without_plugin_skips(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    with patch(
        "src.monitor.broker_snapshot_plugin._discover_plugin",
        return_value=None,
    ):
        assert load_broker_snapshot_section() is None


def test_enabled_read_only_plugin_returns_section(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    section = load_broker_snapshot_section(plugin=_ReadOnlyPlugin())
    assert isinstance(section, BriefSection)
    assert section.name == "broker_snapshot"
    assert section.title == "Broker snapshot"
    assert section.severity == "normal"
    assert "Read-only" in section.data_text


def test_refuses_non_read_only_plugin(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    assert load_broker_snapshot_section(plugin=_MutatingPlugin()) is None


def test_discover_errors_do_not_crash_brief(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    with patch(
        "src.monitor.broker_snapshot_plugin._discover_plugin",
        side_effect=RuntimeError("entry point boom"),
    ):
        assert load_broker_snapshot_section() is None


def test_generate_brief_sections_omits_plugin_when_disabled(
    sample_dashboard, monkeypatch
):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    sections = generate_brief_sections(sample_dashboard)
    assert {s.name for s in sections}.isdisjoint({"broker_snapshot"})
    assert len(sections) == 7


def test_generate_brief_sections_includes_plugin_when_enabled(
    sample_dashboard, monkeypatch
):
    monkeypatch.setenv(ENABLE_ENV, "1")
    fake = BriefSection(
        name="broker_snapshot",
        title="Broker snapshot",
        severity="warning",
        data_text="No redacted snapshot file yet.",
    )
    with patch(
        "src.monitor.broker_snapshot_plugin.load_broker_snapshot_section",
        return_value=fake,
    ):
        sections = generate_brief_sections(sample_dashboard)
    names = [s.name for s in sections]
    assert "broker_snapshot" in names
    assert names.index("broker_snapshot") < names.index("action_items")


def test_sibling_plugin_satisfies_brief_contract(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    sibling_src = Path(__file__).resolve().parents[1].parent / "broker-readonly-gateway" / "src"
    if not sibling_src.is_dir():
        pytest.skip("sibling broker-readonly-gateway missing")
    sys.path.insert(0, str(sibling_src))
    try:
        from broker_readonly_gateway.plugin.lab import BrokerSnapshotBriefPlugin
    finally:
        sys.path.pop(0)
    section = load_broker_snapshot_section(plugin=BrokerSnapshotBriefPlugin())
    assert section is not None
    assert section.name == "broker_snapshot"
    assert section.severity == "warning"


def test_module_does_not_import_live_broker():
    import src.monitor.broker_snapshot_plugin as mod

    source = inspect.getsource(mod)
    assert "from src.broker" not in source
    assert "import src.broker" not in source
