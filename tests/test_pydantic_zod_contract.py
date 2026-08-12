"""Cross-language contract: Pydantic SIGNAL_MODELS keys vs Zod typed signal panels."""

from __future__ import annotations

from pathlib import Path


from src.monitor.signal_schemas import SIGNAL_MODELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_TS = PROJECT_ROOT / "src" / "schemas" / "signals.ts"

# Pydantic sections with a dedicated Zod schema (not OptionalPanelObject passthrough).
PYDANTIC_TO_ZOD_TYPED: dict[str, str] = {
    "ensemble_voting": "ensemble_voting",  # optional panel object — contract checks key presence in TS
    "garch_cvar": "GarchCvarSchema",
    "smart_rebalance": "SmartRebalanceSchema",
    "regime": "RegimeSchema",
    "yield_curve": "YieldCurveSchema",
    "ic_decay": "IcDecaySchema",
    "signal_wfe": "SignalWFESchema",
    "hedge_selector": "HedgeSelectorSchema",
    "fred_macro": "FredMacroSchema",
    "bocd_regime": "bocd_regime",
    "ramp": "ramp",
    "gold_tlt_correlation": "gold_tlt_correlation",
}

ACTIVE_SIGNALS_JSON_PANEL_SCHEMAS: dict[str, str] = {
    "crypto_allocation": "CryptoAllocationSchema",
    "calendar_seasonality": "CalendarSeasonalitySchema",
    "ensemble_voting": "EnsembleVotingSchema",
    "alternative_data": "AlternativeDataSchema",
    "factor_rotation": "FactorRotationSignalSchema",
    "stacking_ensemble": "StackingEnsembleSchema",
    "convexity_harvest": "ConvexityHarvestSchema",
    "llm_sentiment": "LLMSentimentSchema",
    "sector_rotation": "SectorRotationSchema",
    "factor_rotation_dashboard": "FactorRotationDashboardSchema",
    "collar": "CollarSchema",
    "kurtosis_regime": "KurtosisRegimeSchema",
    "volatility_parity": "VolatilityParitySchema",
}

# Integrator REGIME_WEIGHTS (momentum/macro/…) ≠ ensemble REGIME_WEIGHTS (SignalSource).
# Documented exception — not the same constant.
INTEGRATOR_ONLY_PYDANTIC = frozenset(
    {
        "signal_snapshot",
        "two_stage_regime",
        "portfolio_explainability",
    }
)


def _signals_ts_text() -> str:
    assert SIGNALS_TS.exists(), f"missing {SIGNALS_TS}"
    return SIGNALS_TS.read_text()


def test_pydantic_signal_models_non_empty() -> None:
    assert len(SIGNAL_MODELS) >= 10


def test_zod_exports_cover_core_pydantic_sections() -> None:
    """Every Pydantic SIGNAL_MODELS key must appear as a signals.json section in Zod."""
    ts = _signals_ts_text()
    missing: list[str] = []
    for key in sorted(SIGNAL_MODELS.keys()):
        if key in INTEGRATOR_ONLY_PYDANTIC:
            continue
        # Key referenced in SignalsDataObjectSchema or nested schema file
        if f"{key}:" not in ts and f'"{key}"' not in ts:
            missing.append(key)
    assert not missing, f"Pydantic keys missing from TS signals schemas: {missing}"


def test_typed_zod_schemas_exist_for_dashboard_panels() -> None:
    ts = _signals_ts_text()
    for pydantic_key, zod_hint in PYDANTIC_TO_ZOD_TYPED.items():
        assert pydantic_key in SIGNAL_MODELS, pydantic_key
        if zod_hint.endswith("Schema"):
            assert zod_hint in ts, f"expected export {zod_hint} for {pydantic_key}"


def test_active_signals_json_panels_do_not_use_optional_passthrough() -> None:
    """Active signals.json panels must use dedicated Zod schemas, not any-record passthrough."""
    ts = _signals_ts_text()
    for panel_key, schema_name in ACTIVE_SIGNALS_JSON_PANEL_SCHEMAS.items():
        assert f"export const {schema_name}" in ts, f"missing dedicated schema for {panel_key}"
        assert f"{panel_key}: z.optional(OptionalPanelObjectSchema)" not in ts
        assert f"{panel_key}: z.optional(z.record(z.string(), z.unknown()))" not in ts


def test_ensemble_voter_regime_weights_loads_without_ml() -> None:
    """Canonical ensemble REGIME_WEIGHTS (SignalSource keys) loads in safe mode."""
    from src.strategy.ensemble_voter import REGIME_WEIGHTS, Regime

    assert Regime.NORMAL in REGIME_WEIGHTS or len(REGIME_WEIGHTS) > 0
    for regime, weights in REGIME_WEIGHTS.items():
        assert isinstance(weights, dict)
        for w in weights.values():
            assert 0 <= w <= 1.5


def test_integrator_regime_weights_documented_separate_domain() -> None:
    """Integrator uses legacy momentum/macro weight names — not ensemble SignalSource."""
    from src.signals import integrator

    assert "momentum" in integrator.REGIME_WEIGHTS.get("bull", {})
    assert "neutral" in integrator.REGIME_WEIGHTS
