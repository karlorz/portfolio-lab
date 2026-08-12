#!/usr/bin/env python3
"""
Tests for the C6 data-loader mixin extracted by Item 27 (2026-08-12):
``src/dashboard/sections_data_loaders.py`` ``_DataLoaderSectionsMixin``.

A1: getattr smoke — all 5 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_DataLoaderSectionsMixin`` (incl. ``_load_json_file``
    staticmethod semantics).
A2: behavior-equality — canned fixtures for ``_get_yield_curve_data``
    (yields + duration-allocation regime table + provenance merge) and
    ``_load_garch_cvar_data`` (defaults + flat-format % normalization).
    Lazy-import contract: patches on ``src.dashboard.generator.*`` are
    respected (never read the real yields.json / data dir).
"""
import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch


from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_data_loaders import _DataLoaderSectionsMixin

DATA_LOADER_NAMES = (
    "_load_broker_data",
    "_load_garch_cvar_data",
    "_load_entropy_data",
    "_get_yield_curve_data",
    "_load_json_file",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 5 C6 names resolve via DashboardGenerator MRO and the mixin."""
    for name in DATA_LOADER_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_DataLoaderSectionsMixin, name), name


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)  # Monday

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def _make_gen(tmp_path):
    """Generator shell with a real (empty) prices table connection."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = conn
    return gen


def test_a2_yield_curve_steep_regime_and_weights(tmp_path):
    """spread>100 → steep regime, tlt 0.70; provenance merged from PUBLIC_DIR."""
    yields_data = [
        {"date": "2026-06-01", "spread2s10s": 40, "dgs2": 4.0, "dgs10": 4.4},
        {"date": "2026-07-01", "spread2s10s": 120, "dgs2": 3.8, "dgs10": 5.0},
    ]
    (tmp_path / "yields.json").write_text(json.dumps(yields_data))
    gen = _make_gen(tmp_path)

    with patch("src.dashboard.generator.YIELDS_JSON", tmp_path / "yields.json"):
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.datetime", FakeDateTime):
                result = gen._get_yield_curve_data()

    yc = result["yield_curve"]
    assert yc["spread2s10s"] == 120
    assert yc["duration_regime"] == "steep"
    assert yc["asof"] == "2026-07-01"
    assert yc["asof_lag_weekdays"] == 3  # Thu+Fri+Mon after asof
    assert yc["status"] == "ok"
    assert yc["spread_history"] == [40, 120]
    da = result["duration_allocation"]
    assert da["weights"] == {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00}
    assert da["duration_regime"] == "steep"
    assert da["source"] == "yield_curve_regime_table"
    gen.conn.close()


def test_a2_yield_curve_missing_file_returns_empty_shape(tmp_path):
    """No yields.json → result dict with null sections, no crash."""
    gen = _make_gen(tmp_path)
    with patch("src.dashboard.generator.YIELDS_JSON", tmp_path / "nope.json"):
        result = gen._get_yield_curve_data()
    assert result == {"yield_curve": None, "duration_allocation": None}
    gen.conn.close()


def test_a2_garch_cvar_flat_format_normalization(tmp_path):
    """Flat-format .health_report.json: % values normalized to decimals."""
    health = {
        "garch_filtered": True,
        "cvar_95": -1.79,
        "var_95": -1.27,
        "cvar_ratio": 1.5,
        "filter_active": True,
        "conditional_volatility_current": 1.2,
        "garch_persistence": 0.9,
    }
    (tmp_path / ".health_report.json").write_text(json.dumps(health))
    gen = _make_gen(tmp_path)

    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        result = gen._load_garch_cvar_data()

    assert result["cvar_95"] == -0.0179  # -1.79 / 100
    assert result["var_95"] == -0.0127  # -1.27 / 100
    assert result["cvar_ratio"] == 1.5
    assert result["garch_active"] is True
    assert result["current_volatility"] == 0.012  # 1.2 / 100
    assert result["volatility_clustering"] == "elevated"  # persistence 0.9
    # Conformal cross-check skipped on empty table: defaults preserved
    assert result["conformal_cvar_95"] is None
    assert result["conformal_var_95"] is None
    gen.conn.close()


def test_a2_garch_cvar_defaults_without_health_file(tmp_path):
    """No .health_report.json → hardcoded defaults unchanged."""
    gen = _make_gen(tmp_path)
    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        result = gen._load_garch_cvar_data()
    assert result["cvar_95"] == -0.0179
    assert result["garch_active"] is True
    assert result["current_volatility"] == 0.012
    gen.conn.close()


def test_a2_load_json_file_static_semantics(tmp_path):
    """Static helper: missing file → None; non-dict payload → None; else dict."""
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"a": 1}))
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    for surface in (_DataLoaderSectionsMixin, DashboardGenerator):
        assert surface._load_json_file(tmp_path / "missing.json") is None
        assert surface._load_json_file(ok) == {"a": 1}
        assert surface._load_json_file(bad) is None
