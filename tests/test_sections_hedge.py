#!/usr/bin/env python3
"""
Regression tests for the C8 hedge/tsmom/staleness mixin extracted by Item 20
(2026-08-12): ``src/dashboard/sections_hedge.py`` ``_HedgeSectionsMixin``
(test file owed by the TEST-GAP coverage gap — module has zero direct test
references).

A1: getattr smoke — all 10 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_HedgeSectionsMixin``.
A2: behavior-equality — canned fixtures for the pure members
    (``_extract_vix_term_structure_signal``, ``_resolve_hedge_vix_level``,
    ``_is_unavailable_signal_block``, ``_normalized_signal_timestamp``),
    the deferred-timestamp placeholder (``_build_unavailable_hedge_selector_signal``),
    the fake-conn regime gate (``_is_msm_gated``), the unavailable short-circuit
    of ``_get_hedge_selector_signal``, and the full staleness scan
    (``_check_signal_staleness`` with FakeDateTime deferral).
"""
from datetime import datetime, timezone
from unittest.mock import patch

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_hedge import _HedgeSectionsMixin

HEDGE_NAMES = (
    "_is_msm_gated",
    "_extract_vix_term_structure_signal",
    "_resolve_hedge_vix_level",
    "_build_unavailable_hedge_selector_signal",
    "_get_hedge_selector_signal",
    "_normalized_signal_timestamp",
    "_is_unavailable_signal_block",
    "_check_signal_staleness",
    "_apply_staleness_decay",
    "_run_spc_monitor",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 10 C8 names resolve via DashboardGenerator MRO and the mixin."""
    for name in HEDGE_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_HedgeSectionsMixin, name), name


def test_a2_extract_vix_term_structure_signal_canned_inputs():
    """signal_value float extraction; non-dict/None/garbage → None (both)."""
    for surface in (_HedgeSectionsMixin, DashboardGenerator):
        assert surface._extract_vix_term_structure_signal(None) is None
        assert surface._extract_vix_term_structure_signal("not-a-dict") is None
        assert surface._extract_vix_term_structure_signal({}) is None
        assert (
            surface._extract_vix_term_structure_signal({"signal_value": 0.35})
            == 0.35
        )
        assert (
            surface._extract_vix_term_structure_signal({"signal_value": "0.35"})
            == 0.35
        )
        assert (
            surface._extract_vix_term_structure_signal({"signal_value": None})
            is None
        )
        assert (
            surface._extract_vix_term_structure_signal({"signal_value": "garbage"})
            is None
        )


def test_a2_resolve_hedge_vix_level_canned_inputs():
    """Direct level wins; vix_spot fallback; unparseable → None (both)."""
    for surface in (_HedgeSectionsMixin, DashboardGenerator):
        assert surface._resolve_hedge_vix_level(18.5, None) == 18.5
        assert surface._resolve_hedge_vix_level(18.5, {"vix_spot": 17.2}) == 18.5
        assert surface._resolve_hedge_vix_level(None, {"vix_spot": 17.2}) == 17.2
        assert surface._resolve_hedge_vix_level(None, {"vix_spot": "17.2"}) == 17.2
        assert surface._resolve_hedge_vix_level(None, None) is None
        assert surface._resolve_hedge_vix_level(None, "not-a-dict") is None
        assert surface._resolve_hedge_vix_level(None, {"vix_spot": None}) is None
        assert (
            surface._resolve_hedge_vix_level(None, {"vix_spot": "garbage"}) is None
        )


def test_a2_is_unavailable_signal_block_canned_inputs():
    """Unavailable/disabled/missing/synthetic/fallback/error → True (both)."""
    for surface in (_HedgeSectionsMixin, DashboardGenerator):
        assert surface._is_unavailable_signal_block(None) is True
        assert surface._is_unavailable_signal_block([]) is False
        assert surface._is_unavailable_signal_block({}) is False
        assert surface._is_unavailable_signal_block({"status": "unavailable"}) is True
        assert surface._is_unavailable_signal_block({"status": "disabled"}) is True
        assert surface._is_unavailable_signal_block({"status": "missing"}) is True
        assert surface._is_unavailable_signal_block({"status": "ok"}) is False
        assert surface._is_unavailable_signal_block({"source_mode": "synthetic"}) is True
        assert surface._is_unavailable_signal_block({"source_mode": "last_good"}) is True
        assert surface._is_unavailable_signal_block({"source_mode": "fallback"}) is True
        assert surface._is_unavailable_signal_block({"source_mode": "live"}) is False
        assert surface._is_unavailable_signal_block({"cache_status": "degraded"}) is True
        assert surface._is_unavailable_signal_block({"cache_status": "ok"}) is False
        assert surface._is_unavailable_signal_block({"error": "boom"}) is True


def test_a2_normalized_signal_timestamp_canned_inputs():
    """Preferred field, fallback fields, and the daily allow_date path (both)."""
    for surface in (_HedgeSectionsMixin, DashboardGenerator):
        assert surface._normalized_signal_timestamp(None, "generated_at") is None
        assert (
            surface._normalized_signal_timestamp(
                {"generated_at": "2026-07-06T10:00:00Z"}, "generated_at"
            )
            == "2026-07-06T10:00:00Z"
        )
        # Fallback: preferred field absent → generated_at used.
        assert (
            surface._normalized_signal_timestamp(
                {"generated_at": "2026-07-06T10:00:00Z"}, "timestamp"
            )
            == "2026-07-06T10:00:00Z"
        )
        # Empty-string fields are skipped; no usable timestamp → None.
        assert (
            surface._normalized_signal_timestamp(
                {"generated_at": "", "timestamp": ""}, "timestamp"
            )
            is None
        )
        # Daily date only honored with allow_date=True → end-of-UTC-day.
        assert (
            surface._normalized_signal_timestamp({"date": "2026-07-06"}, "generated_at")
            is None
        )
        assert (
            surface._normalized_signal_timestamp(
                {"date": "2026-07-06"}, "generated_at", allow_date=True
            )
            == "2026-07-06T23:59:59+00:00"
        )
        # Unparseable date falls back to the raw value.
        assert (
            surface._normalized_signal_timestamp(
                {"date": "not-a-date"}, "generated_at", allow_date=True
            )
            == "not-a-date"
        )


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def test_a2_build_unavailable_hedge_selector_signal_deferred_timestamp():
    """VIX-unavailable placeholder: FakeDateTime-deferred timestamp, pins."""
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        for surface in (_HedgeSectionsMixin, DashboardGenerator):
            result = surface._build_unavailable_hedge_selector_signal(
                "high_vol", 0.35
            )

            expected_ts = FakeDateTime.now(timezone.utc).isoformat()
            assert result["generated_at"] == expected_ts
            assert result["available"] is False
            assert result["regime"] == "high_vol"
            assert result["primary_hedge"] == "none"
            assert result["gate_reason"] == "vix_unavailable"
            assert result["canonical_controller"] == "hedge_selector"
            assert result["vixy_role"] == "diagnostic_sizing_helper"
            assert result["term_structure_signal"] == 0.35
            assert result["term_structure_gate"] is False
            assert result["regime_confidence"] == 0.0


def test_a2_get_hedge_selector_signal_unavailable_path():
    """No VIX level and no term structure → unavailable payload, no selector."""
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        mixin = _HedgeSectionsMixin()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        for surface in (mixin, gen):
            result = surface._get_hedge_selector_signal(None, "normal")
            assert isinstance(result, dict)
            assert result["available"] is False
            assert result["gate_reason"] == "vix_unavailable"
            assert result["regime"] == "normal"
            assert result["generated_at"] == FakeDateTime.now(timezone.utc).isoformat()


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, *args, **kwargs):
        return self

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._cursor = _FakeCursor(row)

    def cursor(self):
        return self._cursor


class _RaisingCursor(_FakeCursor):
    def execute(self, *args, **kwargs):
        raise RuntimeError("regime query failed")


def test_a2_is_msm_gated_canned_regimes(monkeypatch):
    """HIGH_VOL/CRISIS gate MSM off; normal stays on (mixin surface)."""
    monkeypatch.setattr(_HedgeSectionsMixin, "_last_regime", "normal")
    mixin = _HedgeSectionsMixin()
    for regime, expected in (
        ("high_vol", True),
        ("crisis", True),
        ("normal", False),
        ("", False),
    ):
        mixin.conn = _FakeConn((regime,))
        assert mixin._is_msm_gated() is expected, regime


def test_a2_is_msm_gated_query_failure_fallback(monkeypatch):
    """Transient query failure falls back to last-known regime (both)."""
    monkeypatch.setattr(_HedgeSectionsMixin, "_last_regime", "normal")
    mixin = _HedgeSectionsMixin()
    gen = DashboardGenerator.__new__(DashboardGenerator)
    for surface in (mixin, gen):
        surface.conn = _FakeConn(None)
        surface.conn._cursor = _RaisingCursor(None)
        assert surface._is_msm_gated() is False

    monkeypatch.setattr(_HedgeSectionsMixin, "_last_regime", "high_vol")
    mixin2 = _HedgeSectionsMixin()
    mixin2.conn = _FakeConn(None)
    mixin2.conn._cursor = _RaisingCursor(None)
    assert mixin2._is_msm_gated() is True


def test_a2_check_signal_staleness_fresh_and_daily(monkeypatch):
    """Fresh + daily-date signals stay healthy; missing required ones go stale."""
    monkeypatch.setattr(_HedgeSectionsMixin, "SIGNAL_STALENESS_TTL_HOURS", 4)
    monkeypatch.setattr(_HedgeSectionsMixin, "STALENESS_DECAY_TAU_HOURS", 2.0)
    monkeypatch.setattr(
        "src.dashboard.generator.load_alternative_data_producer_timestamp",
        lambda *args, **kwargs: None,
    )

    signal_data = {
        "ensemble_voting": {"generated_at": "2026-07-06T10:00:00+00:00"},  # 2h old
        "convexity_harvest": {"date": "2026-07-06"},  # daily → end-of-day → fresh
    }
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        mixin = _HedgeSectionsMixin()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        for surface in (mixin, gen):
            result = surface._check_signal_staleness(signal_data)

            # 23 timestamped keys: 18 optional-advisory + 5 required.
            assert result["total_count"] == 23
            assert result["optional_count"] == 18
            assert result["required_count"] == 5
            assert result["ttl_hours"] == 4
            assert result["decay_tau_hours"] == 2.0
            # Missing required keys go stale; missing optional ones are unavailable.
            assert result["stale_signals"] == [
                "alternative_data",
                "garch_cvar",
                "smart_rebalance",
                "rebalance_health",
            ]
            assert len(result["unavailable_signals"]) == 17
            assert result["projection_lag_signals"] == []
            assert result["healthy_count"] == 2
            assert result["checked_at"] == "2026-07-06T12:00:00+00:00"
            # Fresh ensemble_voting: 2h age → exp(-2/2) decay.
            assert result["signal_age_hours"]["ensemble_voting"] == 2.0
            assert result["staleness_decay"]["ensemble_voting"] == 0.3679
            # Daily date normalized to end-of-UTC-day → zero age, full decay.
            assert (
                result["signal_timestamps"]["convexity_harvest"]
                == "2026-07-06T23:59:59+00:00"
            )
            assert result["signal_age_hours"]["convexity_harvest"] == 0.0
            assert result["staleness_decay"]["convexity_harvest"] == 1.0


def test_a2_check_signal_staleness_stale_required(monkeypatch):
    """A required signal past the 4h TTL lands in stale_signals (both)."""
    monkeypatch.setattr(_HedgeSectionsMixin, "SIGNAL_STALENESS_TTL_HOURS", 4)
    monkeypatch.setattr(_HedgeSectionsMixin, "STALENESS_DECAY_TAU_HOURS", 2.0)
    monkeypatch.setattr(
        "src.dashboard.generator.load_alternative_data_producer_timestamp",
        lambda *args, **kwargs: None,
    )

    signal_data = {"ensemble_voting": {"generated_at": "2026-07-01T00:00:00Z"}}
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        mixin = _HedgeSectionsMixin()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        for surface in (mixin, gen):
            result = surface._check_signal_staleness(signal_data)

            # 5d12h = 132h old → stale, decay fully collapsed to 0.0.
            assert result["signal_age_hours"]["ensemble_voting"] == 132.0
            assert result["staleness_decay"]["ensemble_voting"] == 0.0
            assert result["stale_signals"] == [
                "ensemble_voting",
                "alternative_data",
                "garch_cvar",
                "smart_rebalance",
                "rebalance_health",
            ]
            assert len(result["unavailable_signals"]) == 18
            assert result["healthy_count"] == 0
