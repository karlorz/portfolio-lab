"""Batch DC: regime_arb EQUITY_ROTATION polarity + no auto-invert diagnostics."""

from __future__ import annotations

from src.dashboard.generator import DashboardGenerator
from src.signals.cross_asset_regime_arb import (
    AssetRegime,
    AssetRegimeReading,
    BondRegime,
    BondRegimeReading,
    CrossAssetRegimeArbDetector,
    DivergencePattern,
    GoldRegime,
    GoldRegimeReading,
)


def test_equity_rotation_bull_is_positive() -> None:
    det = CrossAssetRegimeArbDetector()
    equity = AssetRegimeReading("SPY", 0.10, 0.12, AssetRegime.BULL, 0.8)
    bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.3)
    gold = GoldRegimeReading("GLD", -0.05, GoldRegime.WEAK, 0.7)
    pattern, value, _ = det._classify_divergence(equity, bonds, gold)
    assert pattern == DivergencePattern.EQUITY_ROTATION
    assert value > 0


def test_equity_rotation_bear_is_negative() -> None:
    """Batch DC: prior always-positive catch-all inverted IC vs SPY labels."""
    det = CrossAssetRegimeArbDetector()
    equity = AssetRegimeReading("SPY", -0.10, 0.20, AssetRegime.BEAR, 0.8)
    bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.3)
    gold = GoldRegimeReading("GLD", -0.05, GoldRegime.WEAK, 0.7)
    pattern, value, _ = det._classify_divergence(equity, bonds, gold)
    assert pattern == DivergencePattern.EQUITY_ROTATION
    assert value < 0


def test_risk_rotation_spy_not_long_bias() -> None:
    det = CrossAssetRegimeArbDetector()
    # Force RISK_ROTATION via classify then map through get_ensemble_signal path
    equity = AssetRegimeReading("SPY", -0.12, 0.25, AssetRegime.BEAR, 0.9)
    bonds = BondRegimeReading("TLT", 0.0, BondRegime.STABLE, 0.3)
    gold = GoldRegimeReading("GLD", 0.10, GoldRegime.STRONG, 0.9)
    pattern, value, _ = det._classify_divergence(equity, bonds, gold)
    assert pattern == DivergencePattern.RISK_ROTATION
    assert value > 0  # alert magnitude stays positive on classifier
    # Build a minimal scan-like object via monkeypatch scan return is heavy;
    # unit-test the mapping logic by calling get_ensemble_signal with stub scan.
    from types import SimpleNamespace
    from unittest.mock import patch

    div = SimpleNamespace(
        pattern=pattern,
        signal_value=value,
        explanation="test",
        persistence_days=0,
        equity_regime=AssetRegime.BEAR,
        bond_regime=BondRegime.STABLE,
        gold_regime=GoldRegime.STRONG,
    )
    fake = SimpleNamespace(
        active=True,
        overall_conviction=0.8,
        timestamp="t",
        signal_value=value,
        divergence=div,
        equity=equity,
        bonds=bonds,
        gold=gold,
    )
    with patch.object(det, "scan", return_value=fake):
        out = det.get_ensemble_signal()
    assert out["signal_value"] < 0  # SPY-facing ensemble value defensive
    assert out["asset_signals"]["SPY"] < 0
    assert out["asset_signals"]["GLD"] > 0
    assert out.get("auto_invert_policy") is None or out.get("polarity_policy")
    assert out.get("polarity_policy") == "no_auto_invert_spy_mapped"


def test_label_alignment_exposes_auto_invert_disabled() -> None:
    # Live diagnostic may vary; shape contract on keys when data present
    diag = DashboardGenerator._label_alignment_diagnostic("cross_asset_regime_arb")
    if diag is None:
        return
    assert diag.get("auto_invert_policy") == "disabled"
    assert "signal_positive_rate" in diag
    assert "ic_raw" in diag
    assert "ic_sign_flipped" in diag
