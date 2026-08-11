#!/usr/bin/env python3
"""
Regression tests for the C1 overlay/regime-prep mixin extracted by Item 23
(2026-08-12): ``src/dashboard/sections_overlay.py`` ``_OverlaySectionsMixin``
(test file owed by Item 23 acceptance gap #1).

A1: getattr smoke — all 10 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_OverlaySectionsMixin``.
A2: behavior-equality — canned fixtures for the pure members
    (``_coerce_vix_level``, ``_is_populated_overlay_section``,
    ``_unavailable_zero_dte_payload`` with FakeDateTime deferral).
"""
from datetime import datetime, timezone
from unittest.mock import patch

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_overlay import _OverlaySectionsMixin

OVERLAY_NAMES = (
    "_unavailable_zero_dte_payload",
    "_unavailable_closing_auction_payload",
    "_is_populated_overlay_section",
    "_get_overlay_data",
    "_record_ic_data",
    "_generate_two_stage_regime",
    "_generate_bocd_regime",
    "_coerce_vix_level",
    "_enrich_regime_vix",
    "_load_signal_generation_context",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 10 C1 names resolve via DashboardGenerator MRO and the mixin."""
    for name in OVERLAY_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_OverlaySectionsMixin, name), name


def test_a2_coerce_vix_level_canned_inputs():
    """Positive finite levels pass; NaN/zero/negative/garbage → None (both)."""
    for surface in (_OverlaySectionsMixin, DashboardGenerator):
        assert surface._coerce_vix_level(None) is None
        assert surface._coerce_vix_level("18.5") == 18.5
        assert surface._coerce_vix_level(0) is None
        assert surface._coerce_vix_level(-3) is None
        assert surface._coerce_vix_level(float("nan")) is None
        assert surface._coerce_vix_level("garbage") is None


def test_a2_is_populated_overlay_section_canned_inputs():
    """Placeholders/unavailable payloads are not populated; real payloads are."""
    mixin = _OverlaySectionsMixin()
    gen = DashboardGenerator.__new__(DashboardGenerator)
    for surface in (mixin, gen):
        assert surface._is_populated_overlay_section(None) is False
        assert surface._is_populated_overlay_section({}) is False
        assert (
            surface._is_populated_overlay_section(
                {"status": "unavailable", "active": False}
            )
            is False
        )
        assert (
            surface._is_populated_overlay_section(
                {"positions": [{"symbol": "SPY"}], "active": True}
            )
            is True
        )
        assert (
            surface._is_populated_overlay_section(
                {"status": "ok", "collar_ratio": 0.5}
            )
            is True
        )


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def test_a2_unavailable_zero_dte_payload_deferred_timestamp():
    """Zero-DTE placeholder: FakeDateTime-deferred timestamp, honesty fields."""
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        result = DashboardGenerator._unavailable_zero_dte_payload()

    expected_ts = FakeDateTime.now().isoformat()
    assert result["generated_at"] == expected_ts
    assert result["timestamp"] == expected_ts
    assert result["active"] is False
    assert result["runtime_status"] == "unavailable_no_producer"
    assert result["live_authoritative"] is False
    assert result["reason"] == "zero_dte producer not wired into overlay merge"


def test_a2_unavailable_closing_auction_payload_deferred_timestamp():
    """Closing-auction placeholder: FakeDateTime-deferred timestamp."""
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        result = DashboardGenerator._unavailable_closing_auction_payload()

    expected_ts = FakeDateTime.now().isoformat()
    assert result["generated_at"] == expected_ts
    assert result["timestamp"] == expected_ts
    assert result["status"] == "unavailable"
    assert result["runtime_status"] == "unavailable_no_producer"
