#!/usr/bin/env python3
"""
Tests for the C9 graduation/explainability/risk-decomposition mixin extracted
by Item 25 (2026-08-12): ``src/dashboard/sections_graduation.py``
``_GraduationExplainabilitySectionsMixin``.

A1: getattr smoke — all 5 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_GraduationExplainabilitySectionsMixin`` (nested ``_stamp``
    moves inside ``_load_risk_decomposition_signal_section``).
A2: behavior-equality — canned fixtures for
    ``_paper_trading_summary_for_dashboard`` (incl. FakeDateTime deferral
    and perf-file override via patched DATA_DIR) and
    ``_latest_stale_explainability_metadata``.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_graduation import _GraduationExplainabilitySectionsMixin

GRADUATION_NAMES = (
    "_graduation_display_value",
    "_paper_trading_summary_for_dashboard",
    "_latest_stale_explainability_metadata",
    "_build_unavailable_explainability_payload",
    "_load_risk_decomposition_signal_section",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 5 C9 names resolve via DashboardGenerator MRO and the mixin."""
    for name in GRADUATION_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_GraduationExplainabilitySectionsMixin, name), name


def test_a2_graduation_display_value_canned_inputs():
    """Checklist value formatting identical via both surfaces."""
    for surface in (_GraduationExplainabilitySectionsMixin, DashboardGenerator):
        assert surface._graduation_display_value(True) == "yes"
        assert surface._graduation_display_value(False) == "no"
        assert surface._graduation_display_value(3) == "3"
        assert surface._graduation_display_value(0.0) == "0"
        assert surface._graduation_display_value(123.45) == "123.5"
        assert surface._graduation_display_value(1.234) == "1.23"
        assert surface._graduation_display_value(0.1234) == "0.1234"
        assert surface._graduation_display_value("n/a") == "n/a"


def test_a2_paper_trading_summary_history_path():
    """Summary built from portfolio history (start/current values + dates)."""
    state = {
        "portfolio": {
            "cash": 0.0,
            "positions": {},
            "history": [
                {"timestamp": "2026-06-01T00:00:00Z", "total_value": 100_000.0},
                {"timestamp": "2026-07-01T00:00:00Z", "total_value": 105_000.0},
            ],
        }
    }
    with patch("src.dashboard.generator.DATA_DIR", __import__("pathlib").Path("/nonexistent")):
        result = _GraduationExplainabilitySectionsMixin._paper_trading_summary_for_dashboard(
            state, days_elapsed=30, days_required=60
        )

    assert result == {
        "start_date": "2026-06-01",
        "initial_capital": 100000.0,
        "current_value": 105000.0,
        "days_elapsed": 30,
        "days_required": 60,
    }


def test_a2_paper_trading_summary_perf_file_override(tmp_path):
    """paper-trading-performance-*.json overrides history values."""
    perf = {
        "date": "2026-05-15",
        "performance": {"start_value": 99_000.0, "current_value": 101_000.0},
    }
    (tmp_path / "paper-trading-performance-2026-07-01.json").write_text(
        json.dumps(perf)
    )
    state = {"portfolio": {"cash": 0.0, "positions": {}, "history": []}}
    with patch("src.dashboard.generator.DATA_DIR", tmp_path):
        result = _GraduationExplainabilitySectionsMixin._paper_trading_summary_for_dashboard(
            state, days_elapsed=None, days_required="60"
        )

    assert result["start_date"] == "2026-05-15"
    assert result["initial_capital"] == 99000.0
    assert result["current_value"] == 101000.0
    assert result["days_elapsed"] == 0  # None → 0
    assert result["days_required"] == 60  # str → int


def test_a2_paper_trading_summary_deferred_today_fallback():
    """No history/perf → start_date falls back to deferred datetime (FakeDateTime)."""
    state = {"portfolio": {"cash": 1_000.0, "positions": {"SPY": {"value": 2_000.0}}}}

    class FakeDateTime(datetime):
        _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls._value.replace(tzinfo=None)
            return cls._value.astimezone(tz)

    with patch("src.dashboard.generator.datetime", FakeDateTime):
        with patch(
            "src.dashboard.generator.DATA_DIR",
            __import__("pathlib").Path("/nonexistent"),
        ):
            result = _GraduationExplainabilitySectionsMixin._paper_trading_summary_for_dashboard(
                state, days_elapsed=1, days_required=2
            )

    assert result["start_date"] == "2026-07-06"
    assert result["initial_capital"] == 100000.0  # no history → default
    assert result["current_value"] == 3000.0  # cash + positions


def test_a2_latest_stale_explainability_metadata(tmp_path):
    """Newest dated file wins; analysis_date surfaces; missing dir → {}."""
    assert (
        _GraduationExplainabilitySectionsMixin._latest_stale_explainability_metadata(
            tmp_path
        )
        == {}
    )

    (tmp_path / "explainability_2026-07-01.json").write_text(
        json.dumps({"analysis_date": "2026-07-01"})
    )
    (tmp_path / "explainability_2026-07-05.json").write_text(
        json.dumps({"analysis_date": "2026-07-05"})
    )
    (tmp_path / "explainability_000_broken.json").write_text("{not json")

    result = _GraduationExplainabilitySectionsMixin._latest_stale_explainability_metadata(
        tmp_path
    )
    assert result["stale_source_file"] == "explainability_2026-07-05.json"
    assert result["stale_analysis_date"] == "2026-07-05"


def test_a2_latest_stale_explainability_metadata_read_error():
    """Unparseable newest file surfaces stale_read_error instead of crashing."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        (Path(d) / "explainability_zzz_broken.json").write_text("{not json")
        result = _GraduationExplainabilitySectionsMixin._latest_stale_explainability_metadata(
            Path(d)
        )
        assert result["stale_source_file"] == "explainability_zzz_broken.json"
        assert "stale_read_error" in result


def test_a2_load_risk_decomposition_live_path_fallback(tmp_path):
    """Live decompose missing → sidecar fallback stamped with deferred ts."""
    sidecar = tmp_path / "risk_decomposition.json"
    sidecar.write_text(json.dumps({"status": "ok", "risk": 0.4}))
    gen = DashboardGenerator.__new__(DashboardGenerator)

    class FakeDateTime(datetime):
        _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls._value.replace(tzinfo=None)
            return cls._value.astimezone(tz)

    with patch("src.dashboard.generator.datetime", FakeDateTime):
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            # decompose_portfolio is lazily imported inside the method; force
            # the fallback by making the import fail.
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *a, **k):
                if name == "src.monitor.risk_decomposition":
                    raise ImportError("no scipy")
                return real_import(name, *a, **k)

            with patch("builtins.__import__", fake_import):
                result = gen._load_risk_decomposition_signal_section()

    expected_ts = FakeDateTime.now(timezone.utc).isoformat()
    # status is popped for successful payloads (must not look unavailable)
    assert result == {"risk": 0.4, "generated_at": expected_ts, "timestamp": expected_ts}
