#!/usr/bin/env python3
"""
Direct unit tests for ``src/signals/signal_source.py`` ``SignalSource``
(consolidated canonical enum — previously 3 divergent duplicates) — test
file owed by the TEST-GAP coverage gap (module has zero direct test
references; indirect coverage via test_generator only).

Pins the 9-member canonical set and its lowercase-snake values. This is the
drift alarm for v806 / v3.23-class member additions: the same set is consumed
at ``generator.py:1476`` (``canonical_sources = {source.value for source in
SignalSource}``) and ``sections_ensemble.py:65`` (``[source.value for source
in SignalSource]``), so a new enum member must update this file too.
"""
from src.signals.signal_source import SignalSource, __all__

CANONICAL_SIGNAL_VALUES = {
    "multi_speed_momentum",
    "cross_asset_rv",
    "international_momentum",
    "alternative_data",
    "cross_asset_regime_arb",
    "unified_overlay",
    "multi_timeframe_fusion",  # v806
    "google_trends",  # replaces behavioral_sentiment
    "vix_term_structure",  # v3.23 intraday vol timing
}


def test_member_count_is_nine():
    """9 members — the canonical set shared by ensemble/stacking/health."""
    assert len(SignalSource) == 9


def test_member_values_unique_lowercase_snake():
    """Values are unique lowercase-snake strings (voter-key contract)."""
    values = [member.value for member in SignalSource]
    assert len(values) == len(set(values))
    for value in values:
        assert value == value.lower()
        assert " " not in value
        assert value


def test_value_round_trip():
    """Every value maps back to its own member (enum lookup contract)."""
    for member in SignalSource:
        assert SignalSource(member.value) is member
    assert SignalSource("multi_speed_momentum") is SignalSource.MULTI_SPEED_MOM
    assert SignalSource("vix_term_structure") is SignalSource.VIX_TERM_STRUCTURE


def test_canonical_value_set_pinned():
    """{s.value for s in SignalSource} == the 9 pinned values (drift alarm)."""
    assert {s.value for s in SignalSource} == CANONICAL_SIGNAL_VALUES


def test_dunder_all_exports_enum_only():
    """__all__ pins the public surface to the enum itself."""
    assert __all__ == ["SignalSource"]
