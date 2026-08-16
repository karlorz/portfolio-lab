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
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.dashboard import generator as generator_module
from src.dashboard.generator import DashboardGenerator
from src.dashboard.signal_section_builder import (
    SignalSectionBuilder,
    _warn_bs_section_if_slow,
)
from src.dashboard.sections_overlay import _OverlaySectionsMixin
from src.data.behavioral_sentiment_fetcher import (
    BehavioralSentimentSnapshot,
    OptionsSentiment,
    RetailFlow,
    SocialIntensity,
)
import logging

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

# =========================================================================
# Item 12 (2026-08-16): behavioral_sentiment provenance block (s1),
# reddit_data_source surfacing (s2), fetch-aware stall watchdog (s3)
# =========================================================================

class _UnavailableProducer:
    """Stand-in raising producer: deterministic builder harness."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError("unavailable in deterministic builder test")


class _FakeRebalanceGate:
    def get_status(self) -> dict:
        return {"remaining_budget_pct": 100.0}


class _FakeBehavioralSignal:
    regime_suppressed = False
    composite_score = 0.5
    signal_type = "neutral"
    confidence = 0.8
    equity_shift_pct = 0.0
    z_score = 0.1
    vix = 15.0


class _FakeBehavioralSignalGen:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_signal(self, _snapshot) -> _FakeBehavioralSignal:
        return _FakeBehavioralSignal()

    def get_status(self) -> dict:
        return {"signal_count_5d": 0}


class _FakeBehavioralFetcher:
    """Fetcher stand-in returning the canned snapshot set by each test."""

    snapshot = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def fetch_snapshot(self) -> BehavioralSentimentSnapshot:
        assert self.snapshot is not None, "test must set _FakeBehavioralFetcher.snapshot"
        return self.snapshot


def _make_snapshot(provenance=None, reddit_source="proxy") -> BehavioralSentimentSnapshot:
    now = datetime.now(timezone.utc).isoformat()
    return BehavioralSentimentSnapshot(
        timestamp=now,
        options=OptionsSentiment(
            timestamp=now, skew_index=120.0, vix=15.0, vix9d=13.5,
            vix9d_ratio=0.9, put_call_ratio=0.52, fear_greed_score=0.0,
        ),
        retail=RetailFlow(
            timestamp=now, retail_call_put_ratio=1.2, retail_buy_sell_imbalance=0.1,
            retail_top_100_correlation=-0.2, small_lot_premium_ratio=0.8,
        ),
        social=SocialIntensity(
            timestamp=now, mention_velocity_7d=1.0, sentiment_divergence=0.0,
            bot_activity_flag=False, influencer_concentration=0.15,
            reddit_data_source=reddit_source,
        ),
        composite_score=0.5,
        signal_type="neutral",
        confidence=0.8,
        data_fresh=True,
        options_provenance=(
            provenance
            if provenance is not None
            else {"^CPCE": "live", "^VIX": "live", "^VIX9D": "live", "^SKEW": "live"}
        ),
    )


def _build_base_sections_with_behavioral(monkeypatch) -> dict:
    """Run build_base_sections with every producer stubbed except behavioral."""
    owner = MagicMock(spec=DashboardGenerator)
    owner._get_yield_curve_data.return_value = {"yield_curve": {}, "duration_allocation": {}}
    owner._load_broker_data.return_value = {"drift": {"max_drift_pct": 1.25}}
    owner._load_garch_cvar_data.return_value = {}
    owner._load_entropy_data.return_value = {}
    owner._get_overlay_data.return_value = {}
    owner._generate_ml_signals.return_value = {"available": False}
    owner._generate_marl_status.return_value = {"available": False}
    owner._build_allocation_surface_roles.return_value = {"routed_surface": "target_allocations"}
    owner._build_regime_authority.return_value = {"live_controller": "signals.json.target_allocations"}
    owner._build_regime_allocation_diagnostic.return_value = {}
    owner._generate_sector_momentum_signals.return_value = None
    owner._get_hedge_selector_signal.return_value = None
    owner._enrich_regime_vix.side_effect = lambda regime, **_kwargs: regime
    owner._unavailable_zero_dte_payload.return_value = {"status": "unavailable"}
    owner._unavailable_closing_auction_payload.return_value = {"status": "unavailable"}
    owner._is_populated_overlay_section.return_value = False
    owner._load_risk_decomposition_signal_section.return_value = None

    monkeypatch.setattr("src.strategy.factor_rotation.FactorMomentumEngine", _UnavailableProducer)
    monkeypatch.setattr("src.strategy.convexity_harvest.ConvexityHarvestStrategy", _UnavailableProducer)
    monkeypatch.setattr("src.strategy.regime_sentiment.RegimeSentimentPipeline", _UnavailableProducer)
    monkeypatch.setattr("src.signals.stacking_integrator.StackingIntegrator", _UnavailableProducer)
    monkeypatch.setattr("src.rebalancing.integration.SmartRebalanceGate", _FakeRebalanceGate)
    monkeypatch.setattr("src.monitor.alerting.check_drift_and_alert", lambda _pct: None)
    monkeypatch.setattr("src.dashboard.generator.load_kill_switch_payload", lambda _data_dir: {})
    monkeypatch.setattr(
        "src.dashboard.generator._apply_kill_to_smart_rebalance",
        lambda payload, _kill: payload,
    )
    monkeypatch.setattr(
        "src.signals.behavioral_sentiment.BehavioralSentimentSignal",
        _FakeBehavioralSignalGen,
    )
    monkeypatch.setattr(
        "src.data.behavioral_sentiment_fetcher.BehavioralSentimentFetcher",
        _FakeBehavioralFetcher,
    )

    return SignalSectionBuilder(owner, generator_module).build_base_sections(
        {
            "vix_level": 15.0,
            "trend_regime": "neutral",
            "current_regime": "NORMAL",
            "regime_data": {"regime": "normal", "vix": 15.0},
            "latest": {"SPY": 500.0},
            "positions": [],
            "cash": 100_000.0,
            "total_value": 100_000.0,
            "target_alloc": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "orders": [],
        }
    )


def test_behavioral_section_emits_live_provenance_block(monkeypatch):
    """Item 12 s1: builder emits additive provenance block with live flags."""
    _FakeBehavioralFetcher.snapshot = _make_snapshot()
    result = _build_base_sections_with_behavioral(monkeypatch)
    assert result["behavioral_sentiment"]["provenance"] == {
        "put_call_ratio": "live",
        "vix": "live",
        "vix9d": "live",
        "skew_index": "live",
    }


def test_behavioral_section_provenance_fallback_and_unknown(monkeypatch):
    """Item 12 s1: fallback reasons pass through; old rows read 'unknown'."""
    _FakeBehavioralFetcher.snapshot = _make_snapshot(
        provenance={
            "^CPCE": "fallback:network",
            "^VIX": "live",
            "^VIX9D": "live",
            "^SKEW": "fallback:estimated_from_vix",
        }
    )
    result = _build_base_sections_with_behavioral(monkeypatch)
    bs = result["behavioral_sentiment"]
    assert bs["provenance"]["put_call_ratio"] == "fallback:network"
    assert bs["provenance"]["skew_index"] == "fallback:estimated_from_vix"

    _FakeBehavioralFetcher.snapshot = _make_snapshot(provenance={})
    result = _build_base_sections_with_behavioral(monkeypatch)
    assert result["behavioral_sentiment"]["provenance"]["put_call_ratio"] == "unknown"


def test_behavioral_section_surfaces_reddit_data_source(monkeypatch):
    """Item 12 s2: social block carries reddit_data_source (proxy|reddit_api)."""
    _FakeBehavioralFetcher.snapshot = _make_snapshot(reddit_source="proxy")
    result = _build_base_sections_with_behavioral(monkeypatch)
    assert result["behavioral_sentiment"]["social"]["reddit_data_source"] == "proxy"

    _FakeBehavioralFetcher.snapshot = _make_snapshot(reddit_source="reddit_api")
    result = _build_base_sections_with_behavioral(monkeypatch)
    assert result["behavioral_sentiment"]["social"]["reddit_data_source"] == "reddit_api"


class _FakeWatchdogDateTime(datetime):
    """Deterministic now() for the stall-watchdog seam (Item 12 s3)."""

    _value = datetime(2026, 7, 6, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value
        return cls._value.replace(tzinfo=timezone.utc).astimezone(tz)


def test_watchdog_absorbs_fresh_fetch_cost(caplog):
    """4.2s section (Item 10 fresh CBOE page fetch) must NOT warn (>= 6.0)."""
    caplog.set_level(logging.WARNING)
    _FakeWatchdogDateTime._value = datetime(2026, 7, 6, 12, 0, 0)
    started = _FakeWatchdogDateTime.now() - timedelta(seconds=4.2)
    with patch("src.dashboard.signal_section_builder.datetime", _FakeWatchdogDateTime):
        _warn_bs_section_if_slow(started)
    assert not any("stall watchdog" in r.message for r in caplog.records)


def test_watchdog_still_fires_on_real_stall(caplog):
    """7s+ section must still warn (real-stall detection preserved)."""
    caplog.set_level(logging.WARNING)
    _FakeWatchdogDateTime._value = datetime(2026, 7, 6, 12, 0, 0)
    started = _FakeWatchdogDateTime.now() - timedelta(seconds=7.0)
    with patch("src.dashboard.signal_section_builder.datetime", _FakeWatchdogDateTime):
        _warn_bs_section_if_slow(started)
    assert any("stall watchdog" in r.message for r in caplog.records)
