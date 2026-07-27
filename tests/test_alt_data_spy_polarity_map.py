"""SPY polarity map for alternative_data — Batch DC pattern (no auto-invert).

The alternative_data composite is a risk-appetite score (positive=risk_on) that
is counter-cyclical for SPY in the IC measurement window (ic_raw=-0.087,
ic_sign_flipped=+0.087, signal_positive_rate=98%).  The fix adds a SPY polarity
map in get_signal_snapshot() — a deliberate classifier output mapping, NOT a
runtime auto-invert (auto_invert_policy: disabled by design).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

from src.signals.alternative_data_signal import AlternativeDataSignalGenerator


def _make_signal(composite: float, regime: str = "bull") -> SimpleNamespace:
    """Build a minimal EnsembleSignal-like object for load_latest_signal mock."""
    return SimpleNamespace(
        timestamp="2026-07-27T00:00:00+00:00",
        confidence=0.75,
        probability=0.8,
        regime=regime,
        raw_data={"composite_score": composite, "components": {}},
    )


def test_alt_data_spy_polarity_inverts_positive_composite() -> None:
    """Positive composite (risk_on) → negative SPY ensemble value."""
    gen = AlternativeDataSignalGenerator()
    composite = 0.5
    raw_tanh = math.tanh(composite / 0.5)
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(composite, "bull")):
        snap = gen.get_signal_snapshot()
    assert snap.value < 0, f"expected inverted (negative) value, got {snap.value}"
    assert abs(snap.value - (-raw_tanh)) < 1e-6


def test_alt_data_spy_polarity_inverts_negative_composite() -> None:
    """Negative composite (risk_off) → positive SPY ensemble value."""
    gen = AlternativeDataSignalGenerator()
    composite = -0.5
    raw_tanh = math.tanh(composite / 0.5)
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(composite, "bear")):
        snap = gen.get_signal_snapshot()
    assert snap.value > 0, f"expected inverted (positive) value, got {snap.value}"
    assert abs(snap.value - (-raw_tanh)) < 1e-6


def test_alt_data_spy_polarity_zero_stays_zero() -> None:
    """Zero composite with neutral regime → zero SPY value (not -0.0)."""
    gen = AlternativeDataSignalGenerator()
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.0, "neutral")):
        snap = gen.get_signal_snapshot()
    assert snap.value == 0.0
    assert not math.copysign(1, snap.value) < 0  # not negative zero


def test_alt_data_polarity_policy_metadata() -> None:
    """Snapshot metadata includes polarity_policy for provenance cohort tracking."""
    gen = AlternativeDataSignalGenerator()
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.3, "bull")):
        snap = gen.get_signal_snapshot()
    assert snap.metadata.get("polarity_policy") == "no_auto_invert_spy_mapped"


def test_alt_data_spy_polarity_asset_signals_spy() -> None:
    """asset_signals['SPY'] matches the SPY-mapped ensemble value."""
    gen = AlternativeDataSignalGenerator()
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.4, "bull")):
        snap = gen.get_signal_snapshot()
    assert "SPY" in snap.asset_signals
    assert snap.asset_signals["SPY"] == snap.value


def test_alt_data_spy_polarity_regime_fallback_inverted() -> None:
    """Regime fallback (composite=0.0) is also SPY-mapped: bull 0.4→-0.4, bear -0.4→+0.4."""
    gen = AlternativeDataSignalGenerator()

    # bull regime fallback
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.0, "bull")):
        snap_bull = gen.get_signal_snapshot()
    assert snap_bull.value == -0.4, f"bull fallback should invert to -0.4, got {snap_bull.value}"

    # bear regime fallback
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.0, "bear")):
        snap_bear = gen.get_signal_snapshot()
    assert snap_bear.value == 0.4, f"bear fallback should invert to +0.4, got {snap_bear.value}"


def test_alt_data_classifier_raw_preserved() -> None:
    """metadata['composite_raw'] preserves the original composite sign (not inverted)."""
    gen = AlternativeDataSignalGenerator()
    composite = 0.5
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(composite, "bull")):
        snap = gen.get_signal_snapshot()
    assert snap.metadata.get("composite_raw") == composite
    # The SPY value is inverted, but composite_raw is the original
    assert snap.metadata["composite_raw"] > 0
    assert snap.value < 0


def test_alt_data_value_scale_reflects_spy_mapping() -> None:
    """value_scale metadata reflects the SPY polarity mapping."""
    gen = AlternativeDataSignalGenerator()
    with patch.object(gen, "load_latest_signal", return_value=_make_signal(0.3, "bull")):
        snap = gen.get_signal_snapshot()
    scale = snap.metadata.get("value_scale", "")
    assert "spy_mapped" in scale, f"value_scale should reflect SPY mapping, got {scale!r}"
