"""Parity contracts for the extracted signal-section collaborator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.dashboard import generator as generator_module
from src.dashboard.generator import DashboardGenerator
from src.dashboard.signal_section_builder import SignalSectionBuilder


class _UnavailableSignal:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError("unavailable in deterministic builder test")


class _FakeRebalanceGate:
    def get_status(self) -> dict[str, object]:
        return {"remaining_budget_pct": 100.0}


def _base_owner() -> MagicMock:
    owner = MagicMock(spec=DashboardGenerator)
    owner._get_yield_curve_data.return_value = {
        "yield_curve": {},
        "duration_allocation": {},
    }
    owner._load_broker_data.return_value = {
        "drift": {"max_drift_pct": 1.25},
    }
    owner._load_garch_cvar_data.return_value = {}
    owner._load_entropy_data.return_value = {}
    owner._get_overlay_data.return_value = {}
    owner._generate_ml_signals.return_value = {"available": False}
    owner._generate_marl_status.return_value = {"available": False}
    owner._build_allocation_surface_roles.return_value = {
        "routed_surface": "target_allocations",
    }
    owner._build_regime_authority.return_value = {
        "live_controller": "signals.json.target_allocations",
    }
    owner._build_regime_allocation_diagnostic.return_value = {}
    owner._generate_sector_momentum_signals.return_value = None
    owner._get_hedge_selector_signal.return_value = None
    owner._enrich_regime_vix.side_effect = lambda regime, **_kwargs: regime
    owner._unavailable_zero_dte_payload.return_value = {"status": "unavailable"}
    owner._unavailable_closing_auction_payload.return_value = {
        "status": "unavailable",
    }
    owner._is_populated_overlay_section.return_value = False
    owner._load_risk_decomposition_signal_section.return_value = None
    return owner


def test_build_base_sections_preserves_complete_keys_and_live_allocations(
    monkeypatch,
) -> None:
    owner = _base_owner()
    allocations = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    drift_alerts: list[float] = []

    monkeypatch.setattr(
        "src.strategy.factor_rotation.FactorMomentumEngine",
        _UnavailableSignal,
    )
    monkeypatch.setattr(
        "src.strategy.convexity_harvest.ConvexityHarvestStrategy",
        _UnavailableSignal,
    )
    monkeypatch.setattr(
        "src.strategy.regime_sentiment.RegimeSentimentPipeline",
        _UnavailableSignal,
    )
    monkeypatch.setattr(
        "src.signals.behavioral_sentiment.BehavioralSentimentSignal",
        _UnavailableSignal,
    )
    monkeypatch.setattr(
        "src.signals.stacking_integrator.StackingIntegrator",
        _UnavailableSignal,
    )
    monkeypatch.setattr(
        "src.rebalancing.integration.SmartRebalanceGate",
        _FakeRebalanceGate,
    )
    monkeypatch.setattr(
        "src.monitor.alerting.check_drift_and_alert",
        drift_alerts.append,
    )
    monkeypatch.setattr(
        "src.dashboard.generator.load_kill_switch_payload",
        lambda _data_dir: {},
    )
    monkeypatch.setattr(
        "src.dashboard.generator._apply_kill_to_smart_rebalance",
        lambda payload, _kill: payload,
    )

    result = SignalSectionBuilder(owner, generator_module).build_base_sections(
        {
            "vix_level": 15.0,
            "trend_regime": "neutral",
            "current_regime": "NORMAL",
            "regime_data": {"regime": "normal", "vix": 15.0},
            "latest": {"SPY": 500.0},
            "positions": [],
            "cash": 100_000.0,
            "total_value": 100_000.0,
            "target_alloc": allocations,
            "orders": [],
        }
    )

    assert set(result) == {
        "regime",
        "target_allocations",
        "allocation_surface_roles",
        "regime_authority",
        "regime_allocation_diagnostic",
        "current_positions",
        "cash",
        "total_value",
        "latest_prices",
        "recent_orders",
        "ml_signals",
        "marl_status",
        "factor_rotation",
        "yield_curve",
        "duration_allocation",
        "convexity_harvest",
        "volatility_parity",
        "llm_sentiment",
        "ensemble_voting",
        "sector_rotation",
        "alternative_data",
        "behavioral_sentiment",
        "collar",
        "crypto_allocation",
        "calendar_seasonality",
        "kurtosis_regime",
        "vix_term_structure",
        "zero_dte",
        "closing_auction",
        "stacking_ensemble",
        "factor_rotation_dashboard",
        "smart_rebalance",
        "broker",
        "garch_cvar",
        "entropy",
        "bond_momentum",
        "hedge_selector",
        "risk_decomposition",
    }
    assert result["target_allocations"] is allocations
    assert result["target_allocations"] == {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    }
    assert result["allocation_surface_roles"]["routed_surface"] == "target_allocations"
    assert result["marl_status"]["available"] is False
    assert drift_alerts == [1.25]
    owner._generate_ml_signals.assert_called_once_with()


def test_optional_sections_preserve_null_circuit_breaker_key() -> None:
    owner = MagicMock(spec=DashboardGenerator)

    with patch("src.monitor.rebalance_health.generate", return_value={"status": "ok"}):
        with patch("src.broker.circuit_breaker.get_circuit_state", return_value=None):
            result = SignalSectionBuilder(owner, generator_module).build_optional_sections(
                {},
                {},
            )

    assert result == {
        "rebalance_health": {"status": "ok"},
        "broker_circuit_breaker": None,
    }
