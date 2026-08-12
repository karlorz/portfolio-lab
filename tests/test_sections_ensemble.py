#!/usr/bin/env python3
"""
Regression tests for the C2 ensemble mixin extracted by Item 19 (2026-08-12):
``src/dashboard/sections_ensemble.py`` ``_EnsembleSectionsMixin`` (test file
owed by the TEST-GAP coverage gap — module has zero direct test references).

A1: getattr smoke — all 17 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_EnsembleSectionsMixin``.
A2: behavior-equality — canned fixtures for the pure members
    (``_build_ensemble_source_breakdown``, ``_format_ensemble_source_label``,
    ``_get_configured_ensemble_source_weights`` with a tmp weights file).
"""
from types import SimpleNamespace

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_ensemble import _EnsembleSectionsMixin

ENSEMBLE_NAMES = (
    "_build_ensemble_source_breakdown",
    "_build_ensemble_source_count_metadata",
    "_get_configured_ensemble_source_weights",
    "_format_ensemble_source_label",
    "_google_trends_inactive_disclosure",
    "_evaluate_ic_reentry",
    "_signal_health_metrics_map",
    "_alt_data_component_bias_diagnostic",
    "_health_recovery_hint",
    "_international_activation_disclosure",
    "_label_alignment_diagnostic",
    "_inactive_signal_shadow_checklist",
    "_zero_baseline_shadow_checklist",
    "_build_configured_source_status",
    "_ensemble_active_weights_rollup",
    "_build_ensemble_adaptive_learning_disclosure",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 17 C2 names resolve via DashboardGenerator MRO and the mixin."""
    for name in ENSEMBLE_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_EnsembleSectionsMixin, name), name


def test_a2_format_ensemble_source_label_canned_inputs():
    """Source identifiers title-case with underscores replaced (both)."""
    for surface in (_EnsembleSectionsMixin, DashboardGenerator):
        assert surface._format_ensemble_source_label("multi_speed_momentum") == (
            "Multi Speed Momentum"
        )
        assert surface._format_ensemble_source_label("google_trends") == (
            "Google Trends"
        )
        assert surface._format_ensemble_source_label("") == ""


def _vote(value, source, confidence=0.75, weight=0.5, is_active=True):
    return SimpleNamespace(
        value=value,
        source=SimpleNamespace(value=source),
        confidence=confidence,
        weight=weight,
        is_active=is_active,
        explanation="test explanation",
    )


def test_a2_build_ensemble_source_breakdown_canned_inputs():
    """Vote serialization: rounding, direction, strength, inactive disclosure."""
    mixin = _EnsembleSectionsMixin()
    gen = DashboardGenerator.__new__(DashboardGenerator)
    for surface in (mixin, gen):
        entries = surface._build_ensemble_source_breakdown(
            [
                _vote(0.25, "multi_speed_momentum"),
                _vote(-0.25, "cross_asset_rv"),
                _vote(0.0, "google_trends", is_active=False),
            ]
        )
        assert entries[0] == {
            "source": "multi_speed_momentum",
            "value": 0.25,
            "direction": "bullish",
            "strength": 0.25,
            "confidence": 0.75,
            "weight": 0.5,
            "is_active": True,
        }
        assert entries[1]["direction"] == "bearish"
        assert entries[2]["direction"] == "neutral"
        assert entries[2]["is_active"] is False
        assert entries[2]["inactive_explanation"] == "test explanation"


def test_a2_get_configured_ensemble_source_weights_tmp_file(monkeypatch, tmp_path):
    """Configured weights file wins over REGIME_WEIGHTS fallback (both)."""
    weights_file = tmp_path / "ensemble_weights.json"
    weights_file.write_text(
        '{"normal": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}}', encoding="utf-8"
    )
    monkeypatch.setenv("ENSEMBLE_WEIGHTS_FILE", str(weights_file))
    mixin = _EnsembleSectionsMixin()
    gen = DashboardGenerator.__new__(DashboardGenerator)
    for surface in (mixin, gen):
        weights = surface._get_configured_ensemble_source_weights("normal")
        assert weights == {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
