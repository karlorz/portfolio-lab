#!/usr/bin/env python3
"""
Tests for the C4 stacking/signal-section mixin extracted by Item 24
(2026-08-12): ``src/dashboard/sections_stacking.py`` ``_StackingSectionsMixin``.

A1: getattr smoke — all 5 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_StackingSectionsMixin``.
A2: behavior-equality — canned fixtures for
    ``_build_stacking_feature_count_metadata`` and
    ``_apply_signal_postprocessors`` (stub builder), plus FakeDateTime
    deferred-timestamp checks for the two stacking dashboard builders.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_stacking import _StackingSectionsMixin

STACKING_NAMES = (
    "_build_stacking_feature_count_metadata",
    "_build_stacking_no_model_dashboard",
    "_build_stacking_model_dashboard",
    "_build_optional_signal_sections",
    "_apply_signal_postprocessors",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 5 C4 names resolve via DashboardGenerator MRO and the mixin."""
    for name in STACKING_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_StackingSectionsMixin, name), name


def test_a2_feature_count_metadata_no_model():
    """Dormant integrator → unavailable_no_model semantics (identical via both)."""
    integrator = SimpleNamespace(model=None, metadata=None)
    expected = {
        "feature_count": None,
        "feature_count_metadata_available": False,
        "feature_count_source": "unavailable_no_model",
        "source_roster": [],
        "source_roster_version": "unavailable_no_model",
        "fallback_semantics": "no_model_feature_count_unavailable",
    }
    assert _StackingSectionsMixin._build_stacking_feature_count_metadata(
        integrator
    ) == expected
    assert DashboardGenerator._build_stacking_feature_count_metadata(
        integrator
    ) == expected


def test_a2_feature_count_metadata_model_loaded():
    """Model + metadata → model_metadata semantics with roster passthrough."""
    metadata = SimpleNamespace(
        feature_count=42,
        source_roster=["base_a", "base_b"],
        source_roster_version="v1.2",
        fallback_semantics="available",
    )
    integrator = SimpleNamespace(model=object(), metadata=metadata)
    expected = {
        "feature_count": 42,
        "feature_count_metadata_available": True,
        "feature_count_source": "model_metadata",
        "source_roster": ["base_a", "base_b"],
        "source_roster_version": "v1.2",
        "fallback_semantics": "available",
    }
    assert _StackingSectionsMixin._build_stacking_feature_count_metadata(
        integrator
    ) == expected
    assert DashboardGenerator._build_stacking_feature_count_metadata(
        integrator
    ) == expected


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def test_a2_no_model_dashboard_deferred_timestamp_and_metadata():
    """Dormant dashboard: FakeDateTime-deferred timestamp + metadata merge."""
    integrator = SimpleNamespace(model=None, metadata=None)
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        result = DashboardGenerator._build_stacking_no_model_dashboard(integrator)

    expected_ts = FakeDateTime.now(timezone.utc).isoformat()
    assert result["timestamp"] == expected_ts
    assert result["generated_at"] == expected_ts
    assert result["active"] is False
    assert result["runtime_status"] == "unavailable_no_model"
    assert result["live_authoritative"] is False
    assert result["feature_count_metadata_available"] is False
    assert result["feature_count_source"] == "unavailable_no_model"


def test_a2_model_dashboard_deferred_timestamp_and_metadata():
    """Model-backed dashboard: FakeDateTime-deferred timestamp + metadata merge."""
    metadata = SimpleNamespace(
        feature_count=7,
        source_roster=[],
        source_roster_version="v3",
        fallback_semantics="available",
    )
    integrator = SimpleNamespace(model=object(), metadata=metadata)
    prediction = SimpleNamespace(
        direction="bullish",
        confidence=0.61,
        probability_bullish=0.61,
        probability_bearish=0.2,
        probability_neutral=0.19,
        fallback_used=False,
        model_version="stack_v3",
        latency_ms=12.5,
    )
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        result = DashboardGenerator._build_stacking_model_dashboard(
            integrator, prediction
        )

    expected_ts = FakeDateTime.now(timezone.utc).isoformat()
    assert result["timestamp"] == expected_ts
    assert result["generated_at"] == expected_ts
    assert result["active"] is True
    assert result["runtime_status"] == "model_loaded"
    assert result["live_authoritative"] is False
    assert result["prediction_direction"] == "bullish"
    assert result["feature_count"] == 7
    assert result["feature_count_metadata_available"] is True
    assert result["latency_ms"] == 12.5


def test_a2_apply_signal_postprocessors_stub_builder_passthrough():
    """Postprocessors delegate to the builder; MRO resolves on real instances."""
    builder = SimpleNamespace(
        apply_postprocessors=lambda output, context: {
            "delegated": True,
            "output_keys": sorted(output.keys()),
            "context_regime": context["current_regime"],
        }
    )
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._get_signal_section_builder = lambda: builder

    output = {"a": 1}
    context = {"current_regime": "normal"}
    result = gen._apply_signal_postprocessors(output, context)

    assert result == {
        "delegated": True,
        "output_keys": ["a"],
        "context_regime": "normal",
    }


def test_a2_optional_signal_sections_stub_builder_passthrough():
    """Optional sections delegate to the builder via MRO."""
    builder = SimpleNamespace(
        build_optional_sections=lambda output, context: {"optional": True}
    )
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._get_signal_section_builder = lambda: builder

    assert gen._build_optional_signal_sections({"b": 2}, {}) == {"optional": True}
