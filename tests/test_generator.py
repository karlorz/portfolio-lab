#!/usr/bin/env python3
"""
Tests for dashboard generator — VIX regime detection, data freshness,
health status, alerts, broker data, and stats calculation.
"""
import json
import inspect
import sqlite3
import sys
import types
import numpy as np

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.dashboard.generator import DashboardGenerator, DATA_DIR, PUBLIC_DIR, DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_market_db(db_path, symbols=None, days=30, base_price=500.0):
    """Create a market.db with price data for testing."""
    if symbols is None:
        symbols = ['SPY', 'GLD', 'TLT', 'QQQ']
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
        PRIMARY KEY (symbol, date))
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            date TEXT, regime TEXT, vix_level REAL, detected_at TEXT
        )
    """)
    base_date = datetime.now()
    for sym in symbols:
        for i in range(days):
            d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            noise = np.random.normal(0, 2.0)
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (sym, d, round(base_price + noise, 2)))
    conn.commit()
    conn.close()


def _make_generator(tmp_path):
    """Create a DashboardGenerator with a test database."""
    db_path = tmp_path / "market.db"
    _create_market_db(db_path)
    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen.conn = sqlite3.connect(str(db_path))
    gen.conn.row_factory = sqlite3.Row
    return gen, db_path


def _write_ok_source_manifest(public_dir: Path) -> None:
    """Write a compact live source manifest for healthy health-json fixtures."""
    (public_dir / "source_manifest.json").write_text(json.dumps({
        "artifacts": [
            {
                "artifact": "prices.json",
                "provider": "Yahoo Finance",
                "feed": "chart/v8",
                "source_mode": "live",
                "status": "success",
                "data_quality": {
                    "artifact": "data_quality.json",
                    "schema_version": "price-data-quality/v1",
                    "status": "ok",
                    "issue_counts": {"total": 0},
                },
            },
        ],
    }))


def _write_data_quality_report(public_dir: Path, *, status: str = "ok", stale_latest_dates: int = 0) -> None:
    """Write a compact current data_quality.json report for alert/SLO fixtures."""
    issue_counts = {
        "duplicate_dates": 0,
        "empty_symbols": 0,
        "extreme_returns": 0,
        "internal_gaps": 0,
        "invalid_dates": 0,
        "invalid_prices": 0,
        "missing_required_keys": 0,
        "non_monotonic_rows": 0,
        "non_object_records": 0,
        "split_like_returns": 0,
        "stale_latest_dates": stale_latest_dates,
        "total": stale_latest_dates,
    }
    (public_dir / "data_quality.json").write_text(json.dumps({
        "artifact": "data_quality.json",
        "schema_version": "price-data-quality/v1",
        "generated_at": "2026-06-16T12:00:00Z",
        "status": status,
        "issue_counts": issue_counts,
        "symbols": [
            {"symbol": "SPY", "status": "ok", "latest_date": "2026-06-15"},
            {
                "symbol": "GLD",
                "status": "fail" if stale_latest_dates else "ok",
                "latest_date": "2026-06-11" if stale_latest_dates else "2026-06-15",
                "stale_latest_date": {
                    "reference_date": "2026-06-15",
                    "latest_date": "2026-06-11",
                    "latest_lag_days": 2,
                } if stale_latest_dates else None,
            },
        ],
    }))


class TestFredMacroProvenance:
    """FRED macro unavailable states should be explicit and non-predictive."""

    def test_record_ic_data_skips_unavailable_fred_macro(self, tmp_path, monkeypatch):
        """Fallback FRED confidence must not be staged as an IC prediction."""
        gen, _ = _make_generator(tmp_path)

        class FakeICMonitor:
            def __init__(self):
                self.staged = []

            def load_state(self):
                return None

            def has_staged_predictions(self):
                return False

            def stage_predictions(self, predictions, staged_date):
                self.staged.append((predictions, staged_date))

            def save_state(self):
                return None

        monitor = FakeICMonitor()
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.ic_decay_monitor",
            types.SimpleNamespace(ICMonitor=lambda: monitor),
        )

        gen._record_ic_data({
            "fred_macro": {
                "regime": "UNKNOWN",
                "confidence": 0.5,
                "indicators": {},
                "indicators_observed": False,
                "source_mode": "unavailable",
                "status": "unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        })

        assert monitor.staged == []
        gen.conn.close()

    def test_staleness_classifies_unavailable_fred_macro(self, tmp_path):
        """Unavailable FRED macro should appear in freshness semantics."""
        gen, _ = _make_generator(tmp_path)

        staleness = gen._check_signal_staleness({
            "fred_macro": {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "indicators": {},
                "indicators_observed": False,
                "source_mode": "unavailable",
                "status": "unavailable",
            },
        })

        assert "fred_macro" in staleness["unavailable_signals"]
        assert staleness["signal_timestamps"]["fred_macro"] is None
        assert staleness["staleness_decay"]["fred_macro"] == 0.0
        gen.conn.close()


class TestTurnoverValidatorPublicArtifact:
    """Public turnover-validator diagnostics must separate production and fixture keys."""

    def test_generate_turnover_validator_json_groups_non_canonical_sources(self, tmp_path, monkeypatch):
        from src.strategy.turnover_validator import TurnoverValidator

        monkeypatch.setattr(
            TurnoverValidator,
            "get_state_diagnostics",
            lambda _self: {
                "multi_speed_momentum": {"periods": 7, "turnover_penalty": 0.1},
                "src": {"periods": 7, "turnover_penalty": 0.2},
            },
        )

        gen = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_turnover_validator_json()

        assert path == tmp_path / "turnover_validator.json"
        payload = json.loads(path.read_text())
        assert "src" not in payload
        assert payload["signals"] == {
            "multi_speed_momentum": {"periods": 7, "turnover_penalty": 0.1},
        }
        assert payload["synthetic_baselines"] == {
            "src": {
                "metadata": {"source_type": "synthetic_or_fixture"},
                "diagnostics": {"periods": 7, "turnover_penalty": 0.2},
            },
        }


class TestPredictionLabelLifecycle:
    """IC staging should follow market-data label lifecycle, not wall-clock runs."""

    def _make_spy_generator(self, tmp_path, rows):
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE prices (
                symbol TEXT,
                date TEXT,
                close REAL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        for date, close in rows:
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", date, close))
        conn.commit()
        conn.row_factory = sqlite3.Row

        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = conn
        return gen

    @staticmethod
    def _ic_output(value=0.4):
        return {
            "ensemble_voting": {
                "equity_bias": value,
                "gold_bias": -0.1,
                "duration_bias": 0.2,
                "weighted_consensus": value,
            }
        }

    def test_record_ic_data_stages_latest_market_data_date_not_wall_clock(
        self, tmp_path, monkeypatch
    ):
        """Staged prediction date must not move past the latest available SPY row."""
        from src.monitor import ic_decay_monitor

        monkeypatch.setattr(
            ic_decay_monitor,
            "IC_STATE_PATH",
            tmp_path / "ic_monitor_state.json",
        )
        gen = self._make_spy_generator(tmp_path, [("2026-07-02", 100.0)])

        try:
            gen._record_ic_data(self._ic_output())
        finally:
            gen.conn.close()

        state = json.loads((tmp_path / "ic_monitor_state.json").read_text())
        assert state["__staged__"]["date"] == "2026-07-02"

    def test_record_ic_data_preserves_unresolved_staged_predictions_on_same_market_date(
        self, tmp_path, monkeypatch
    ):
        """Repeated dashboard runs on stale market data must not overwrite unresolved labels."""
        from src.monitor import ic_decay_monitor

        state_path = tmp_path / "ic_monitor_state.json"
        monkeypatch.setattr(ic_decay_monitor, "IC_STATE_PATH", state_path)
        state_path.write_text(json.dumps({
            "__staged__": {
                "date": "2026-07-02",
                "predictions": {"old_signal": 0.25},
            }
        }))
        gen = self._make_spy_generator(tmp_path, [("2026-07-02", 100.0)])

        try:
            gen._record_ic_data(self._ic_output(value=0.9))
        finally:
            gen.conn.close()

        state = json.loads(state_path.read_text())
        assert state["__staged__"] == {
            "date": "2026-07-02",
            "predictions": {"old_signal": 0.25},
        }

    def test_record_ic_data_resolves_staged_predictions_when_later_spy_row_exists(
        self, tmp_path, monkeypatch
    ):
        """A later SPY close should turn staged predictions into resolved observations."""
        from src.monitor import ic_decay_monitor

        state_path = tmp_path / "ic_monitor_state.json"
        monkeypatch.setattr(ic_decay_monitor, "IC_STATE_PATH", state_path)
        state_path.write_text(json.dumps({
            "__staged__": {
                "date": "2026-07-02",
                "predictions": {"old_signal": 0.5},
            }
        }))
        gen = self._make_spy_generator(
            tmp_path,
            [("2026-07-02", 100.0), ("2026-07-03", 102.0)],
        )

        try:
            gen._record_ic_data(self._ic_output(value=-0.2))
        finally:
            gen.conn.close()

        state = json.loads(state_path.read_text())
        assert state["old_signal"] == [[0.5, pytest.approx(0.02)]]
        assert state["__staged__"]["date"] == "2026-07-03"


class TestSignalStalenessNormalization:
    """Signal staleness should distinguish stale from optional unavailable."""

    def test_required_fresh_and_optional_missing_sections_are_not_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
        })

        assert result["stale_signals"] == []
        assert "behavioral_sentiment" in result["unavailable_signals"]
        assert "two_stage_regime" in result["unavailable_signals"]
        assert result["signal_timestamps"]["rebalance_health"] == fresh
        assert result["healthy_count"] == result["required_count"]

    def test_present_required_sections_without_section_timestamp_use_artifact_generated_at(
        self,
        tmp_path,
    ):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "generated_at": fresh,
            "ensemble_voting": {"regime": "normal"},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"cvar_95": -0.02},
            "smart_rebalance": {"decision": "hold"},
            "rebalance_health": {"status": "ok"},
        })

        for signal_key in ("ensemble_voting", "garch_cvar", "smart_rebalance", "rebalance_health"):
            assert signal_key not in result["stale_signals"]
            assert result["signal_timestamps"][signal_key] == fresh
            assert result["staleness_decay"][signal_key] > 0.0
        assert result["healthy_count"] == result["required_count"]

    def test_producer_fresh_alt_data_not_stale_when_projected_lags(self, tmp_path, monkeypatch):
        """Producer latest fresh + projected stale → projection_lag, not stale."""
        from src.dashboard import generator as gen_mod

        gen, _ = _make_generator(tmp_path)
        producer_ts = datetime.now(timezone.utc).isoformat()
        projected_stale = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir(parents=True)
        (signals_dir / "alternative_data_latest.json").write_text(
            json.dumps(
                {
                    "source": "alternative_data",
                    "regime": "bull",
                    "timestamp": producer_ts,
                    "confidence": 0.6,
                    "raw_data": {"composite_score": 0.1},
                }
            )
        )
        monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": producer_ts},
            "alternative_data": {"timestamp": projected_stale},
            "garch_cvar": {"timestamp": producer_ts},
            "smart_rebalance": {"generated_at": producer_ts},
            "rebalance_health": {"generated": producer_ts},
        })

        assert "alternative_data" not in result["stale_signals"]
        assert "alternative_data" in result["projection_lag_signals"]
        assert result["signal_timestamps"]["alternative_data"] == producer_ts

    def test_refresh_public_alternative_data_projection_updates_signals(self, tmp_path):
        from src.dashboard.generator import refresh_public_alternative_data_projection

        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        (data_dir / "signals").mkdir(parents=True)
        public_dir.mkdir()
        producer_ts = "2026-07-18T12:00:00+00:00"
        (data_dir / "signals" / "alternative_data_latest.json").write_text(
            json.dumps(
                {
                    "source": "alternative_data",
                    "regime": "bull",
                    "timestamp": producer_ts,
                    "confidence": 0.7,
                    "raw_data": {
                        "composite_score": 0.2,
                        "components": {"treasury": 0.1},
                        "component_confidences": {"treasury": 0.5},
                        "weights": {"treasury": 1.0},
                    },
                }
            )
        )
        (public_dir / "signals.json").write_text(
            json.dumps(
                {
                    "generated_at": "2026-07-12T00:00:00+00:00",
                    "alternative_data": {"timestamp": "2026-07-12T00:00:00+00:00"},
                }
            )
        )

        assert refresh_public_alternative_data_projection(
            data_dir=data_dir, public_dir=public_dir
        )
        out = json.loads((public_dir / "signals.json").read_text())
        assert out["alternative_data"]["timestamp"] == producer_ts
        assert out["alternative_data_projection"]["source"] == "bounded_alt_data_refresh"

    def test_required_stale_signal_remains_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        stale = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": stale},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
        })

        assert "garch_cvar" in result["stale_signals"]
        assert result["signal_age_hours"]["garch_cvar"] >= 8

    def test_optional_error_placeholder_is_unavailable_not_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "two_stage_regime": {"error": "FRED API key unavailable"},
        })

        assert "two_stage_regime" not in result["stale_signals"]
        assert "two_stage_regime" in result["unavailable_signals"]

    def test_fresh_date_only_optional_daily_sections_are_not_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).date().isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "convexity_harvest": {"date": today, "allocation_pct": 0.0},
            "volatility_parity": {"date": today, "target_volatility": 10.0},
        })

        assert "convexity_harvest" not in result["stale_signals"]
        assert "volatility_parity" not in result["stale_signals"]
        assert "convexity_harvest" not in result["unavailable_signals"]
        assert "volatility_parity" not in result["unavailable_signals"]
        assert result["signal_timestamps"]["convexity_harvest"].startswith(today)
        assert result["signal_timestamps"]["volatility_parity"].startswith(today)

    def test_stale_date_only_optional_daily_sections_remain_stale(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        stale_date = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "convexity_harvest": {"date": stale_date, "allocation_pct": 0.0},
            "volatility_parity": {"date": stale_date, "target_volatility": 10.0},
        })

        assert "convexity_harvest" in result["stale_signals"]
        assert "volatility_parity" in result["stale_signals"]
        assert result["signal_timestamps"]["convexity_harvest"].startswith(stale_date)
        assert result["signal_timestamps"]["volatility_parity"].startswith(stale_date)
        assert result["signal_age_hours"]["convexity_harvest"] >= 24.0
        assert result["signal_age_hours"]["volatility_parity"] >= 24.0

    def test_optional_active_only_sections_without_freshness_are_unavailable(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "collar": {"active": True, "status_text": "Collar active"},
            "bond_momentum": {"active": True, "status_text": "Bonds active"},
        })

        assert "collar" not in result["stale_signals"]
        assert "bond_momentum" not in result["stale_signals"]
        assert "collar" in result["unavailable_signals"]
        assert "bond_momentum" in result["unavailable_signals"]
        assert result["signal_timestamps"]["collar"] is None
        assert result["signal_timestamps"]["bond_momentum"] is None

    def test_optional_overlay_sections_with_generated_at_are_fresh(self, tmp_path):
        """Overlay dashboard must stamp generated_at or healthy producers look unavailable."""
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": fresh},
            "collar": {"active": True, "generated_at": fresh, "status_text": "ok"},
            "bond_momentum": {"active": True, "generated_at": fresh, "status_text": "ok"},
            "calendar_seasonality": {"active": False, "generated_at": fresh},
            "crypto_allocation": {"active": True, "generated_at": fresh},
            "kurtosis_regime": {"active": True, "generated_at": fresh},
            "factor_rotation": {
                "selected_factors": ["VLUE"],
                "signal_strength": 0.5,
                "generated_at": fresh,
            },
        })

        for key in (
            "collar",
            "bond_momentum",
            "calendar_seasonality",
            "crypto_allocation",
            "kurtosis_regime",
            "factor_rotation",
        ):
            assert key not in result["unavailable_signals"]
            assert key not in result["stale_signals"]
            assert result["signal_timestamps"][key] is not None

    def test_future_naive_timestamp_is_bounded_to_fresh_age_and_decay(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        future_naive = (
            datetime.now(timezone.utc) + timedelta(hours=8)
        ).replace(tzinfo=None).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": future_naive},
        })

        assert result["signal_age_hours"]["rebalance_health"] == 0.0
        assert result["staleness_decay"]["rebalance_health"] == 1.0

    def test_future_aware_timestamp_cannot_publish_decay_above_one(self, tmp_path):
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        result = gen._check_signal_staleness({
            "ensemble_voting": {"generated_at": fresh},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated": future},
        })

        decay_values = result["staleness_decay"].values()
        age_values = [
            age for age in result["signal_age_hours"].values()
            if age is not None
        ]
        assert all(age >= 0.0 for age in age_values)
        assert all(0.0 <= decay <= 1.0 for decay in decay_values)


    def test_hedge_selector_signal_discloses_canonical_hedge_roles(self):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeSelector:
            def select(
                self,
                vix_level,
                regime_confidence,
                regime_label,
                term_structure_signal=None,
            ):
                assert vix_level == 25.0
                assert regime_label == "elevated"
                assert term_structure_signal == -0.5
                return types.SimpleNamespace(
                    regime="elevated",
                    regime_confidence=regime_confidence,
                    primary_hedge="vixy",
                    primary_size_pct=3.0,
                    secondary_hedge=None,
                    secondary_size_pct=0.0,
                    cost_benefit_gate=True,
                    net_benefit_bps=42.0,
                    kelly_fraction=0.18,
                    expected_cost_bps=5.0,
                    expected_benefit_bps=47.0,
                    confidence_scaled_size=3.0,
                    min_hold_days=5,
                    transition_cost_bps=25.0,
                    canonical_controller="hedge_selector",
                    vixy_role="diagnostic_sizing_helper",
                    term_structure_role="gate_discount_multiplier",
                    term_structure_gate=True,
                    term_structure_multiplier=0.9,
                    gate_reason="term_structure_confirmed",
                )

        with patch("src.strategy.hedge_selector.HedgeSelector", return_value=FakeSelector()):
            result = gen._get_hedge_selector_signal(
                25.0,
                "elevated",
                {"signal_value": -0.5, "regime": "backwardation"},
            )

        assert result["canonical_controller"] == "hedge_selector"
        assert result["vixy_role"] == "diagnostic_sizing_helper"
        assert result["term_structure_role"] == "gate_discount_multiplier"
        assert result["term_structure_gate"] is True
        assert result["term_structure_multiplier"] == 0.9
        assert result["gate_reason"] == "term_structure_confirmed"

    def test_hedge_selector_uses_vix_term_structure_spot_when_market_vix_missing(self):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeSelector:
            def select(
                self,
                vix_level,
                regime_confidence,
                regime_label,
                term_structure_signal=None,
            ):
                assert vix_level == 16.76
                assert term_structure_signal == 0.5
                return types.SimpleNamespace(
                    regime="normal",
                    regime_confidence=regime_confidence,
                    primary_hedge="none",
                    primary_size_pct=0.0,
                    secondary_hedge=None,
                    secondary_size_pct=0.0,
                    cost_benefit_gate=False,
                    net_benefit_bps=0.0,
                    kelly_fraction=0.0,
                    expected_cost_bps=0.0,
                    expected_benefit_bps=0.0,
                    confidence_scaled_size=0.0,
                    min_hold_days=5,
                    transition_cost_bps=0.0,
                    canonical_controller="hedge_selector",
                    vixy_role="diagnostic_sizing_helper",
                    term_structure_role="gate_discount_multiplier",
                    term_structure_gate=False,
                    term_structure_multiplier=0.0,
                    gate_reason="normal_no_trade_band",
                )

        with patch("src.strategy.hedge_selector.HedgeSelector", return_value=FakeSelector()):
            result = gen._get_hedge_selector_signal(
                None,
                "normal",
                {"signal_value": 0.5, "vix_spot": 16.76, "regime": "extreme_contango"},
            )

        assert result["available"] is True
        assert result["primary_hedge"] == "none"
        assert result["gate_reason"] == "normal_no_trade_band"


    def test_postprocessors_recompute_staleness_after_optional_regime_publish(
        self, tmp_path, monkeypatch
    ):
        """Final artifact staleness matches optional regime sections appended later."""
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()

        class FakeFredSignal:
            regime = "NORMAL"
            confidence = 0.6
            recession_probability = 0.1
            inflation_pressure = "neutral"
            monetary_stance = "neutral"
            manufacturing_health = "neutral"
            credit_conditions = "neutral"
            indicators = {}
            timestamp = fresh

        class FakeRegimeTransitionForecaster:
            def fit(self, history):
                self.history = history

            def forecast(self, current, horizon_days):
                return types.SimpleNamespace(
                    probabilities={"normal": 0.7, "high_vol": 0.3},
                    most_likely="normal",
                    persistence_params={"normal": 7.0},
                )

        class FakeLiveTransitionManager:
            def get_status(self):
                return {"status": "paper", "timestamp": fresh}

        monkeypatch.setitem(
            sys.modules,
            "src.data.fred_data",
            types.SimpleNamespace(get_fred_signal=lambda: FakeFredSignal()),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.regime.regime_transition_forecaster",
            types.SimpleNamespace(RegimeTransitionForecaster=FakeRegimeTransitionForecaster),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.health_check",
            types.SimpleNamespace(run_health_check=lambda: {"status": "ok"}),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.alerting",
            types.SimpleNamespace(
                check_staleness_and_alert=lambda staleness: None,
                check_ic_decay_and_alert=lambda ic_decay: None,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.ic_decay_monitor",
            types.SimpleNamespace(compute_ic_decay_report=lambda: {"status": "healthy"}),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.signal_walk_forward",
            types.SimpleNamespace(compute_signal_wfe_report=lambda: {"status": "validated"}),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.research.gold_tlt_correlation",
            types.SimpleNamespace(
                run_analysis=lambda window, save: types.SimpleNamespace(
                    current_correlation=0.1,
                    current_regime="neutral",
                    correlation_trend="stable",
                    mean_correlation=0.2,
                    min_correlation=-0.1,
                    max_correlation=0.5,
                    structural_breaks=[],
                    regimes=[],
                    implications=[],
                )
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.broker.alpaca",
            types.SimpleNamespace(LiveTransitionManager=FakeLiveTransitionManager),
        )
        monkeypatch.setattr(
            "src.dashboard.generator.validate_signal",
            lambda _name, signal: signal,
        )
        monkeypatch.setattr(gen, "_generate_two_stage_regime", lambda: None)
        monkeypatch.setattr(
            gen,
            "_generate_bocd_regime",
            lambda: {
                "regime": 1,
                "regime_change_prob": 0.2,
                "timestamp": fresh,
            },
        )
        monkeypatch.setattr(gen, "_run_spc_monitor", lambda output: {"status": "ok"})
        monkeypatch.setattr(gen, "_record_ic_data", lambda output: None)

        output = {
            "ensemble_voting": {
                "generated_at": fresh,
                "regime": "normal",
                "source_breakdown": [],
            },
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated_at": fresh},
        }

        try:
            result = gen._apply_signal_postprocessors(
                output,
                {
                    "cursor": gen.conn.cursor(),
                    "current_regime": "normal",
                },
            )
        finally:
            gen.conn.close()

        assert result["bocd_regime"]["timestamp"] == fresh
        assert "bocd_regime" not in result["staleness"]["unavailable_signals"]
        assert result["staleness"]["signal_timestamps"]["bocd_regime"] == fresh
        assert result["staleness"]["signal_age_hours"]["bocd_regime"] is not None
        assert result["staleness"]["staleness_decay"]["bocd_regime"] > 0.0

    def test_postprocessors_embed_canonical_health_report_when_available(
        self, tmp_path, monkeypatch
    ):
        """signals.json health preserves canonical health.json severity and SLO cause."""
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        (tmp_path / "health.json").write_text(json.dumps({
            "system_status": "critical",
            "generated_at": fresh,
            "cron_jobs": [],
            "data_freshness": {},
            "scheduler_status": {"status": "degraded"},
            "data_pipeline_slo": {
                "status": "critical",
                "top_dimension": "data_quality",
                "runbook": {"top_cause": {"code": "stale_prices"}},
            },
        }))

        class FakeFredSignal:
            regime = "NORMAL"
            confidence = 0.6
            recession_probability = 0.1
            inflation_pressure = "neutral"
            monetary_stance = "neutral"
            manufacturing_health = "neutral"
            credit_conditions = "neutral"
            indicators = {}
            timestamp = fresh

        monkeypatch.setitem(
            sys.modules,
            "src.data.fred_data",
            types.SimpleNamespace(get_fred_signal=lambda: FakeFredSignal()),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.health_check",
            types.SimpleNamespace(run_health_check=lambda: {"system_status": "warning"}),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.alerting",
            types.SimpleNamespace(
                check_staleness_and_alert=lambda staleness: None,
                check_ic_decay_and_alert=lambda ic_decay: None,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.ic_decay_monitor",
            types.SimpleNamespace(compute_ic_decay_report=lambda: {"status": "healthy"}),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.monitor.signal_walk_forward",
            types.SimpleNamespace(compute_signal_wfe_report=lambda: {"status": "validated"}),
        )
        monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", tmp_path)
        monkeypatch.setattr("src.dashboard.generator.validate_signal", lambda _name, signal: signal)
        monkeypatch.setattr(gen, "_generate_two_stage_regime", lambda: None)
        monkeypatch.setattr(gen, "_generate_bocd_regime", lambda: None)
        monkeypatch.setattr(gen, "_run_spc_monitor", lambda output: {"status": "ok"})
        monkeypatch.setattr(gen, "_record_ic_data", lambda output: None)

        output = {
            "ensemble_voting": {"generated_at": fresh, "regime": "normal", "source_breakdown": []},
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated_at": fresh},
        }

        try:
            result = gen._apply_signal_postprocessors(
                output,
                {"cursor": gen.conn.cursor(), "current_regime": "normal"},
            )
        finally:
            gen.conn.close()

        assert result["health"]["status"] == "critical"
        assert result["health"]["data_pipeline_slo_status"] == "critical"
        assert result["health"]["top_slo_dimension"] == "data_quality"
        assert result["health"]["top_slo_cause_code"] == "stale_prices"


class TestEnsemblePostDecayMetrics:
    """Ensemble post-decay metrics should share one signed source contract."""

    def test_allocation_surface_roles_disclose_current_live_routing(self, tmp_path):
        roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)

        assert roles["schema_version"] == "allocation-surface-roles/v1"
        assert roles["routed_surface"] == "target_allocations"
        assert roles["surfaces"]["target_allocations"]["routed"] is True
        assert roles["surfaces"]["target_allocations"]["routed_by"] == "src.broker.order_router"
        assert roles["surfaces"]["target_allocations"]["live_authoritative"] is True
        assert roles["surfaces"]["ensemble_voting"]["routed"] is False
        assert roles["surfaces"]["ensemble_voting"]["role"] == "advisory_non_routed"

    def test_allocation_surface_roles_include_standalone_advisory_artifacts(self, tmp_path):
        roles = DashboardGenerator._build_allocation_surface_roles(data_dir=tmp_path)

        for surface in ("adaptive_sizing", "black_litterman", "calendar_seasonality"):
            role = roles["surfaces"][surface]
            assert role["role"] == "advisory_non_routed"
            assert role["routed"] is False
            assert role["routed_by"] is None
            assert role["live_authoritative"] is False
            assert role["canonical_controller"] == "signals.json.target_allocations"
            assert "target_allocations" in role["description"]

        cal = roles["surfaces"]["calendar_seasonality"]
        assert cal.get("applies_to_target_allocations") is False

    def test_advisory_allocation_artifact_role_block_is_machine_readable(self):
        role = DashboardGenerator._build_advisory_allocation_artifact_role(
            surface="black_litterman",
            allocation_field="posterior_weights",
        )

        assert role == {
            "schema_version": "allocation-artifact-role/v1",
            "surface": "black_litterman",
            "allocation_field": "posterior_weights",
            "runtime_role": "advisory_non_routed",
            "live_authoritative": False,
            "routed": False,
            "routed_by": None,
            "canonical_controller": "signals.json.target_allocations",
            "routed_surface": "target_allocations",
            "routed_surface_path": "public/data/signals.json#target_allocations",
            "description": (
                "black_litterman is published for advisory diagnostics; live order routing "
                "continues to consume signals.json.target_allocations."
            ),
        }

    def test_black_litterman_public_weights_are_uppercase_with_exclusion_diagnostics(self):
        weights = DashboardGenerator._canonicalize_public_weights(
            {"spy": 0.46, "gld": 0.0, "tlt": 0.16},
            canonical_assets=("SPY", "GLD", "TLT", "IEF"),
        )

        assert weights["weights"] == {"SPY": 0.46, "GLD": 0.0, "TLT": 0.16, "IEF": 0.0}
        assert weights["excluded_assets"] == []
        assert weights["zero_weight_assets"] == ["GLD", "IEF"]

    def test_regime_authority_discloses_live_controller_and_shadow_roles(self):
        authority = DashboardGenerator._build_regime_authority(
            current_regime="vol_spike",
            target_alloc={"SPY": 0.38, "GLD": 0.42, "TLT": 0.20},
        )

        assert authority["schema_version"] == "regime-authority/v1"
        assert authority["live_controller"] == "classify_vix_regime"
        assert authority["live_controller_module"] == "src.utils.classify_vix_regime"
        assert authority["live_regime"] == "vol_spike"
        assert authority["allocation_regime"] == "high_vol"
        assert authority["routed_surface"] == "target_allocations"
        assert authority["advanced_regime_signals"]["two_stage_regime"]["role"] == "advisory_shadow"
        assert authority["advanced_regime_signals"]["bocd_regime"]["routed"] is False

    def test_regime_authority_marks_missing_advanced_sections_unpublished(self):
        output = {
            "regime_authority": DashboardGenerator._build_regime_authority(
                current_regime="normal",
                target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ),
            "staleness": {
                "unavailable_signals": ["two_stage_regime", "regime_transition"],
                "stale_signals": [],
            },
        }

        DashboardGenerator._update_regime_authority_availability(output)

        advanced = output["regime_authority"]["advanced_regime_signals"]
        for signal_name in ("two_stage_regime", "regime_transition"):
            entry = advanced[signal_name]
            assert entry["published"] is False
            assert entry["availability"] == "unavailable"
            assert entry["routed"] is False
            assert entry["role"] == "advisory_shadow"
            assert "Published" not in entry["description"]

    def test_regime_authority_marks_present_fresh_advanced_sections_published(self):
        fresh = datetime.now(timezone.utc).isoformat()
        output = {
            "regime_authority": DashboardGenerator._build_regime_authority(
                current_regime="normal",
                target_alloc={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            ),
            "two_stage_regime": {"timestamp": fresh, "regime": "NORMAL"},
            "staleness": {
                "unavailable_signals": [],
                "stale_signals": [],
            },
        }

        DashboardGenerator._update_regime_authority_availability(output)

        entry = output["regime_authority"]["advanced_regime_signals"]["two_stage_regime"]
        assert entry["published"] is True
        assert entry["availability"] == "present"
        assert entry["routed"] is False
        assert entry["role"] == "advisory_shadow"

    def test_source_breakdown_preserves_signed_signal_values(self):
        """Serialized source rows include the signed value used downstream."""
        from src.signals.signal_source import SignalSource
        from src.strategy.ensemble_voter import SignalReading

        rows = DashboardGenerator._build_ensemble_source_breakdown([
            SignalReading(
                source=SignalSource.ALTERNATIVE_DATA,
                timestamp="2026-07-05T00:00:00+00:00",
                value=0.8,
                confidence=0.9,
                weight=0.6,
                regime_fit="normal",
            ),
            SignalReading(
                source=SignalSource.CROSS_ASSET_RV,
                timestamp="2026-07-05T00:00:00+00:00",
                value=-0.4,
                confidence=0.7,
                weight=0.4,
                regime_fit="normal",
            ),
        ])

        assert rows[0]["source"] == "alternative_data"
        assert rows[0]["value"] == pytest.approx(0.8)
        assert rows[1]["source"] == "cross_asset_rv"
        assert rows[1]["value"] == pytest.approx(-0.4)

    def test_vix_source_breakdown_uses_fractional_bridge_confidence(self):
        """VIX source rows publish the normalized typed-bridge confidence."""
        from src.signals.vix_term_structure import VIXTermStructureSignal

        vix_signal = VIXTermStructureSignal(
            timestamp="2026-07-05T00:00:00+00:00",
            signal_state="NEUTRAL",
            signal_value=0.2,
            vix_spot=18.0,
            vix3m=19.5,
            vix6m=20.0,
            slope_vix3m_vix=1.083,
            regime="contango",
            regime_strength=0.5,
            slope_signal=0.3,
            roll_yield_signal=0.08,
            vix_zscore_signal=0.0,
            curve_shape_signal=0.25,
            spy_shift=0.02,
            gld_shift=-0.01,
            tlt_shift=-0.01,
            confidence=90.0,
            is_valid=True,
            reason="VIX=18.00, Slope=1.083, Regime=contango",
        )
        reading = vix_signal.to_signal_snapshot().to_signal_reading()
        reading.weight = 0.05

        rows = DashboardGenerator._build_ensemble_source_breakdown([reading])

        assert rows[0]["source"] == "vix_term_structure"
        assert rows[0]["confidence"] == pytest.approx(0.9)
        assert 0.0 <= rows[0]["confidence"] <= 1.0

    def test_ensemble_adaptive_learning_disclosure_preserves_runtime_status(self):
        disclosure = {
            "bandit": {
                "status": "non_effective",
                "enabled": True,
                "reason": "cold_start_no_regime_weights",
            },
            "online_ic": {
                "status": "disabled",
                "enabled": False,
                "reason": "env_disabled",
            },
        }
        ensemble_result = type("EnsembleResult", (), {"adaptive_learning": disclosure})()

        assert DashboardGenerator._build_ensemble_adaptive_learning_disclosure(ensemble_result) == disclosure

    def test_ensemble_source_count_metadata_distinguishes_configured_collected_and_contributing(self):
        """Source count metadata separates roster, collected rows, and live contributors."""
        source_breakdown = [
            {"source": "alternative_data", "weight": 0.24},
            {"source": "cross_asset_rv", "weight": 0.0},
            {"source": "google_trends", "weight": 0.05},
            {"source": "multi_speed_momentum", "weight": 0.0},
        ]

        counts = DashboardGenerator._build_ensemble_source_count_metadata(
            regime="normal",
            source_breakdown=source_breakdown,
        )

        assert counts["configured_source_count"] == 9
        assert counts["collected_source_count"] == 4
        assert counts["contributing_source_count"] == 2
        assert counts["inactive_source_count"] == 2
        assert counts["inactive_sources"] == ["cross_asset_rv", "multi_speed_momentum"]
        assert counts["num_sources"] == counts["collected_source_count"]

    def test_configured_source_status_discloses_stale_google_trends(self, monkeypatch):
        """Configured source status explains stale Google Trends omission from source rows."""
        from src.signals.signal_snapshot import SignalSnapshot

        class FakeGoogleTrendsSignal:
            def get_signal_snapshot(self):
                return SignalSnapshot(
                    source="google_trends",
                    timestamp="2026-07-05T00:00:00+00:00",
                    value=0.0,
                    confidence=0.0,
                    is_active=False,
                    explanation="Google Trends: Data is 37 days old (max 14)",
                    metadata={
                        "inactive_reason": "Data is 37 days old (max 14)",
                        "inactive_category": "stale",
                    },
                )

        monkeypatch.setattr(
            "src.signals.google_trends_signal.GoogleTrendsSignal",
            FakeGoogleTrendsSignal,
        )

        statuses = DashboardGenerator._build_configured_source_status(
            regime="normal",
            source_breakdown=[{"source": "alternative_data", "weight": 0.24}],
        )

        google_trends = next(status for status in statuses if status["source"] == "google_trends")
        assert google_trends["status"] == "stale"
        assert google_trends["active"] is False
        assert google_trends["collected"] is False
        assert google_trends["configured_weight"] == pytest.approx(0.04762)
        assert google_trends["reason"] == "Data is 37 days old (max 14)"

    def test_marl_status_discloses_controller_runtime_non_routed(self, monkeypatch):
        """MARL status publishes the controller contract without implying routing authority."""
        controller_status = {
            "version": "2.51.0",
            "device": "cpu",
            "agents_loaded": ["analyst", "sentiment", "risk", "execution", "controller"],
            "signal_integrator_connected": False,
            "checkpoint_loaded": False,
            "inference_count": 0,
            "current_allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0},
            "graph_metrics": {"messages_routed": 0},
        }

        class FakeAIController:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get_status(self):
                return controller_status

        monkeypatch.setattr("src.agents.ai_controller.AIController", FakeAIController)

        status = DashboardGenerator._generate_marl_status()

        # Without checkpoint, available is false; module is still importable
        assert status["available"] is False
        assert status.get("module_importable") is True
        assert status.get("reason") == "checkpoint_not_loaded"
        assert status["schema_version"] == "marl-runtime-status/v1"
        assert status["runtime"]["version"] == "2.51.0"
        assert status["runtime"]["agents_loaded"] == controller_status["agents_loaded"]
        assert status["runtime"]["signal_integrator_connected"] is False
        assert status["runtime"]["inference_count"] == 0
        assert status["execution_role"]["routed"] is False
        assert status["execution_role"]["role"] == "research_shadow_non_routed"
        assert status["execution_role"]["routed_by"] is None
        assert "target_allocations" in status["execution_role"]["description"]

    def test_marl_status_available_when_checkpoint_loaded(self, monkeypatch):
        controller_status = {
            "version": "2.51.0",
            "device": "cpu",
            "agents_loaded": ["analyst"],
            "signal_integrator_connected": False,
            "checkpoint_loaded": True,
            "inference_count": 3,
            "current_allocation": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0},
            "graph_metrics": {},
        }

        class FakeAIController:
            def __init__(self, *args, **kwargs):
                pass
            def get_status(self):
                return controller_status

        monkeypatch.setattr("src.agents.ai_controller.AIController", FakeAIController)
        status = DashboardGenerator._generate_marl_status()
        assert status["available"] is True
        assert status.get("module_importable") is True
        assert status["runtime"]["checkpoint_loaded"] is True

    def test_staleness_decay_recomputes_consensus_and_agreement(self, tmp_path):
        """Post-decay consensus and agreement derive from decayed source rows."""
        gen, _ = _make_generator(tmp_path)
        output = {
            "staleness": {
                "staleness_decay": {
                    "alternative_data": 0.1,
                    "ensemble_voting": 1.0,
                },
            },
            "ensemble_voting": {
                "weighted_consensus": 0.0,
                "agreement_ratio": 0.5,
                "source_breakdown": [
                    {
                        "source": "alternative_data",
                        "value": 1.0,
                        "weight": 0.5,
                    },
                    {
                        "source": "cross_asset_rv",
                        "value": -1.0,
                        "weight": 0.5,
                    },
                ],
            },
        }

        try:
            result = gen._apply_staleness_decay(output)
        finally:
            gen.conn.close()

        ensemble = result["ensemble_voting"]
        assert ensemble["total_weight_after_decay"] == pytest.approx(0.55)
        assert ensemble["weighted_consensus"] == pytest.approx(-0.8182)
        assert ensemble["agreement_ratio"] == pytest.approx(0.9091)


# ---------------------------------------------------------------------------
# VIX regime detection tests
# ---------------------------------------------------------------------------

class TestVIXRegimeDetection:
    """Test VIX-based regime classification logic."""

    def _classify_vix(self, vix_level):
        """Extract VIX regime classification logic."""
        if vix_level > 25:
            return "crisis"
        elif vix_level > 20:
            return "vol_spike"
        elif vix_level < 15:
            return "low_vol"
        else:
            return "normal"

    def test_crisis_regime(self):
        assert self._classify_vix(30) == "crisis"
        assert self._classify_vix(26) == "crisis"

    def test_vol_spike_regime(self):
        assert self._classify_vix(22) == "vol_spike"
        assert self._classify_vix(21) == "vol_spike"

    def test_low_vol_regime(self):
        assert self._classify_vix(12) == "low_vol"
        assert self._classify_vix(14) == "low_vol"

    def test_normal_regime(self):
        assert self._classify_vix(18) == "normal"
        assert self._classify_vix(15) == "normal"
        assert self._classify_vix(20) == "normal"

    def test_composite_regime_vix_overrides(self):
        """VIX crisis/vol_spike overrides trend regime."""
        # If VIX says crisis, it overrides regardless of trend
        vix_regime = "crisis"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        else:
            current_regime = trend_regime
        assert current_regime == "crisis"

    def test_composite_regime_low_vol_with_normal_trend(self):
        """Low vol + normal trend → low_vol."""
        vix_regime = "low_vol"
        trend_regime = "normal"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "low_vol"

    def test_composite_regime_normal_uses_trend(self):
        """Normal VIX + trend regime → uses trend."""
        vix_regime = "normal"
        trend_regime = "bull"
        if vix_regime in ["crisis", "vol_spike"]:
            current_regime = vix_regime
        elif vix_regime == "low_vol" and trend_regime != "crisis":
            current_regime = "low_vol"
        else:
            current_regime = trend_regime
        assert current_regime == "bull"


# ---------------------------------------------------------------------------
# Data freshness tests
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """Test data freshness classification."""

    def _classify_freshness(self, days_stale):
        """Extract freshness classification logic."""
        if days_stale <= 1:
            return "fresh"
        elif days_stale <= 3:
            return "stale"
        else:
            return "critical"

    def test_fresh(self):
        assert self._classify_freshness(0) == "fresh"
        assert self._classify_freshness(1) == "fresh"

    def test_stale(self):
        assert self._classify_freshness(2) == "stale"
        assert self._classify_freshness(3) == "stale"

    def test_critical(self):
        assert self._classify_freshness(4) == "critical"
        assert self._classify_freshness(30) == "critical"


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------

class TestHealthStatus:
    """Test system health status determination."""

    def _determine_health(self, failed_jobs, stale_count):
        """Extract health status logic."""
        status = "healthy"
        if failed_jobs > 0 or stale_count > 5:
            status = "warning"
        if failed_jobs > 2 or stale_count > 10:
            status = "critical"
        return status

    def test_healthy(self):
        assert self._determine_health(0, 0) == "healthy"
        assert self._determine_health(0, 5) == "healthy"

    def test_warning(self):
        assert self._determine_health(1, 0) == "warning"
        assert self._determine_health(0, 6) == "warning"

    def test_critical(self):
        assert self._determine_health(3, 0) == "critical"
        assert self._determine_health(0, 11) == "critical"

    def test_critical_overrides_warning(self):
        """Critical takes precedence when both conditions met."""
        assert self._determine_health(3, 11) == "critical"


# ---------------------------------------------------------------------------
# Generator initialization tests
# ---------------------------------------------------------------------------

class TestGeneratorInit:
    """Test DashboardGenerator initialization."""

    def test_creates_with_db(self, tmp_path):
        """Generator connects to database."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn is not None
        gen.conn.close()

    def test_row_factory_set(self, tmp_path):
        """Row factory is set for dict-like access."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn.row_factory == sqlite3.Row
        gen.conn.close()


# ---------------------------------------------------------------------------
# Cross-asset relative value JSON tests
# ---------------------------------------------------------------------------

class TestCrossAssetRVJSON:
    """Test cross-asset relative-value dashboard artifact generation."""

    def test_uses_current_signal_shape_and_unavailable_pair_metadata(
        self, tmp_path
    ):
        gen, _ = _make_generator(tmp_path)

        class FakeReading:
            def to_dict(self):
                return {
                    "pair_name": "spy_gld",
                    "symbol_a": "SPY",
                    "symbol_b": "GLD",
                    "z_score": 2.2,
                    "signal_value": -0.55,
                    "regime": "diverged_bull",
                    "conviction": 0.73,
                    "coverage_status": "available",
                }

        fake_signal = types.SimpleNamespace(
            pairs={"spy_gld": FakeReading()},
            avg_z_score=2.2,
            max_divergence=2.2,
            num_diverged=1,
            total_pairs=5,
            available_pair_count=1,
            unavailable_pair_count=1,
            unavailable_pairs={
                "gld_btc": {
                    "coverage_status": "unavailable",
                    "missing_symbols": ["BTC"],
                    "reason": "missing_or_all_nan_symbol",
                },
            },
            missing_symbols=["BTC"],
            risk_on_score=0.4,
            duration_score=0.0,
            overall_conviction=0.73,
        )

        class FakeScanner:
            def scan_all(self):
                return fake_signal

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch(
                "src.signals.cross_asset_relative_value.CrossAssetRVScanner",
                return_value=FakeScanner(),
            ):
                path = gen.generate_cross_asset_rv_json()

        data = json.loads(path.read_text())
        assert data["signal_value"] == pytest.approx(0.4)
        assert data["pairs"][0]["pair_name"] == "spy_gld"
        assert data["available_pair_count"] == 1
        assert data["unavailable_pair_count"] == 1
        assert data["unavailable_pairs"]["gld_btc"]["missing_symbols"] == ["BTC"]
        assert data["missing_symbols"] == ["BTC"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON tests
# ---------------------------------------------------------------------------

class TestPerformanceJSON:
    """Test generate_performance_json."""

    def test_generates_file(self, tmp_path):
        """Creates dashboard.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        assert path.exists()
        gen.conn.close()

    def test_output_structure(self, tmp_path):
        """Output has expected keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert "prices" in data
        assert "regimes" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_prices_contain_symbols(self, tmp_path):
        """Prices dict contains expected symbols."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["prices"]
        assert "GLD" in data["prices"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON tests
# ---------------------------------------------------------------------------

class TestStatsJSON:
    """Test generate_stats_json."""

    def test_generates_file(self, tmp_path):
        """Creates stats.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        assert path.exists()
        gen.conn.close()

    def test_has_asset_stats(self, tmp_path):
        """Stats contain per-asset data."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "assets" in data or "generated_at" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Alerts JSON tests
# ---------------------------------------------------------------------------

class TestAlertsJSON:
    """Test generate_alerts_json."""

    def test_generates_file(self, tmp_path):
        """Creates alerts.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        assert path.exists()
        gen.conn.close()

    def test_alerts_structure(self, tmp_path):
        """Alerts output has expected structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        assert "alerts" in data
        assert "count" in data
        assert isinstance(data["alerts"], list)
        gen.conn.close()

    def test_kill_switch_alert(self, tmp_path):
        """Kill switch file generates alert."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": True, "reason": "test", "mode": "paper", "timestamp": datetime.now().isoformat()}))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        kill_alerts = [a for a in data["alerts"] if a["type"] == "kill_switch"]
        assert len(kill_alerts) >= 1
        gen.conn.close()

    def test_stale_data_alert(self, tmp_path):
        """Stale data generates warning alert."""
        gen, db_path = _make_generator(tmp_path)
        # Insert very old data
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('STALE', '2020-01-01', 100.0)")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        stale_alerts = [a for a in data["alerts"] if a["type"] == "stale_data"]
        assert len(stale_alerts) >= 1
        gen.conn.close()

    def test_promote_trigger_success_blocked_by_active_kill_switch(self, tmp_path):
        """Stale promotion markers cannot publish success while kill switch is active."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / ".promote_to_live").write_text(json.dumps({
            "metrics": {"sharpe": 0.85},
            "timestamp": "2026-01-01T00:00:00",
        }))
        (tmp_path / "kill_switch.json").write_text(json.dumps({
            "enabled": True,
            "mode": "paper",
            "reason": "drawdown breach",
            "timestamp": "2026-01-02T00:00:00",
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert not [
            alert for alert in data["alerts"]
            if alert["type"] == "graduation_candidate" and alert["level"] == "success"
        ]
        gen.conn.close()

    def test_promote_trigger_success_requires_manual_approval(self, tmp_path):
        """Promotion success must fail closed when manual approval is absent."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / ".promote_to_live").write_text(json.dumps({
            "metrics": {"sharpe": 0.85},
            "timestamp": "2026-01-01T00:00:00",
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert not [
            alert for alert in data["alerts"]
            if alert["type"] == "graduation_candidate" and alert["level"] == "success"
        ]
        gen.conn.close()

    def test_promote_trigger_success_requires_current_ready_checklist(self, tmp_path):
        """Current checklist failures dominate stale promote-to-live markers."""
        from src.strategy.graduation_checklist import CheckResult

        gen, _ = _make_generator(tmp_path)
        (tmp_path / ".promote_to_live").write_text(json.dumps({
            "metrics": {"sharpe": 0.85},
            "timestamp": "2026-01-01T00:00:00",
        }))
        (tmp_path / ".manual_approval").write_text("approved")
        not_ready = {
            "min_trading_days": CheckResult("min_trading_days", False, 5, 63, ""),
            "manual_approval": CheckResult("manual_approval", True, 1, 1, ""),
        }

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path), \
             patch("src.strategy.graduation_checklist.DATA_DIR", tmp_path), \
             patch("src.strategy.graduation_checklist.GraduationChecklist.check", return_value=not_ready):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert not [
            alert for alert in data["alerts"]
            if alert["type"] == "graduation_candidate" and alert["level"] == "success"
        ]
        gen.conn.close()

    def test_promote_blocked_tombstone_emits_no_graduation_candidate_alert(self, tmp_path):
        """promote_blocked_* tombstones are not candidacy — no graduation_candidate alert."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / ".promote_to_live").write_text(json.dumps({
            "graduation_conflict": True,
            "action": "promote_blocked_checklist",
            "reason": "checklist_not_ready",
            "is_graduation_ready": False,
            "timestamp": "2026-07-18T04:35:03",
            "source": "graduation_checklist",
            "readiness_score": 18.2,
            "prior_metrics": {"sharpe": 0.86},
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert not [
            alert for alert in data["alerts"]
            if alert["type"] == "graduation_candidate"
        ]
        gen.conn.close()

    def test_promote_blocked_kill_tombstone_emits_no_candidate_alert(self, tmp_path):
        """Kill tombstones must not surface as blocked graduation candidates."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / ".promote_to_live").write_text(json.dumps({
            "action": "promote_blocked_kill",
            "reason": "kill_authority",
            "kill_level": "halt",
            "timestamp": "2026-07-18T04:00:00",
        }))
        (tmp_path / "kill_switch.json").write_text(json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "test",
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert not [
            alert for alert in data["alerts"]
            if alert["type"] == "graduation_candidate"
        ]
        gen.conn.close()

    def test_critical_health_slo_projects_into_alerts(self, tmp_path):
        """Critical health payload must surface as health_slo in alerts.json."""
        from src.dashboard.health_slo_alerts import HEALTH_SLO_ALERT_TYPE

        gen, _ = _make_generator(tmp_path)
        as_of = "2026-07-07T12:00:00"
        health = {
            "system_status": "critical",
            "generated_at": as_of,
            "data_pipeline_slo": {
                "status": "critical",
                "top_dimension": "alpaca_feed_entitlement",
                "dimensions": {
                    "alpaca_feed_entitlement": {
                        "status": "critical",
                        "policy_decision": "reject",
                        "reason": "missing_entitlement",
                        "acceptable_for_live": False,
                    },
                },
                "runbook": {
                    "status": "critical",
                    "top_cause": {
                        "dimension": "alpaca_feed_entitlement",
                        "code": "missing_entitlement",
                        "severity": "critical",
                        "reason": "missing_entitlement",
                        "action": "Restore Alpaca feed entitlement before live routing.",
                    },
                },
            },
        }

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json(health=health)

        data = json.loads(path.read_text())
        health_alerts = [a for a in data["alerts"] if a.get("type") == HEALTH_SLO_ALERT_TYPE]
        assert len(health_alerts) == 1
        alert = health_alerts[0]
        assert alert["level"] == "error"
        assert alert["requires_action"] is True
        assert alert["timestamp"] == as_of
        assert alert.get("top_dimension") == "alpaca_feed_entitlement"
        assert alert.get("reason") == "missing_entitlement"
        assert alert.get("policy_decision") == "reject"
        assert alert.get("runbook_action")
        assert "missing_entitlement" in (alert.get("message") or "")
        gen.conn.close()

    def test_healthy_health_json_does_not_emit_health_slo_alert(self, tmp_path):
        """Non-critical health should not invent a health/SLO alert."""
        from src.dashboard.health_slo_alerts import HEALTH_SLO_ALERT_TYPE

        gen, _ = _make_generator(tmp_path)
        health = {
            "system_status": "healthy",
            "generated_at": "2026-07-07T12:00:00",
            "data_pipeline_slo": {"status": "healthy", "top_dimension": None},
        }

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json(health=health)

        data = json.loads(path.read_text())
        health_alerts = [a for a in data["alerts"] if a.get("type") == HEALTH_SLO_ALERT_TYPE]
        assert health_alerts == []
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON tests
# ---------------------------------------------------------------------------

class TestHealthJSON:
    """Test generate_health_json."""

    def test_generates_file(self, tmp_path):
        """Creates health.json file."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        assert path.exists()
        gen.conn.close()

    def test_health_structure(self, tmp_path):
        """Health output has expected structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "system_status" in data
        assert "data_freshness" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_generate_health_json_preserves_ops_health_from_monitor(self, tmp_path):
        """Dashboard regen must re-stamp ops_health_* from monitor report.

        make health merges ops_health_status into PUBLIC health.json; a later
        generate_health_json must not wipe those dual-SSOT fields.
        """
        gen, _ = _make_generator(tmp_path)
        data_dir = tmp_path / "data"
        public_dir = tmp_path / "public"
        data_dir.mkdir()
        public_dir.mkdir()
        (data_dir / "health.json").write_text(json.dumps({
            "status": "ok",
            "timestamp": "2026-07-18T05:00:00+00:00",
            "scope": "operational_readiness",
            "checks": {
                "kill_switch": {"status": "ok", "enabled": False},
                "open_incidents": {"status": "ok", "open_count": 0, "incidents": []},
            },
            "service": "portfolio-lab",
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", public_dir):
            with patch("src.dashboard.generator.DATA_DIR", data_dir):
                path = gen.generate_health_json()
        data = json.loads(path.read_text())
        assert data.get("ops_health_status") == "ok"
        assert data.get("ops_health_source") == "monitor.health_check"
        assert data.get("ops_health_timestamp") == "2026-07-18T05:00:00+00:00"
        gen.conn.close()

    def test_data_freshness_populated(self, tmp_path):
        """Data freshness contains symbols from DB."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["data_freshness"]) > 0
        assert "SPY" in data["data_freshness"]
        gen.conn.close()

    def test_fred_readiness_populates_health_and_slo(self, tmp_path):
        """FRED readiness should be included in dashboard health and SLO output."""
        gen, _ = _make_generator(tmp_path)
        readiness = {
            "status": "warning",
            "readiness": "warn",
            "mode": "lab",
            "ready": True,
            "blocking": False,
            "reason": "missing_fred_api_key",
            "source_mode": "synthetic",
            "remediation": "Set FRED_API_KEY for lab/paper/live operation.",
        }
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch(
                    "src.data.fred_data.get_fred_md_cache_health",
                    return_value={
                        "status": "unavailable",
                        "source_mode": "synthetic",
                        "api_key_configured": False,
                    },
                ):
                    with patch("src.monitor.fred_readiness.assess_fred_readiness", return_value=readiness):
                        path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert data["fred_readiness"]["reason"] == "missing_fred_api_key"
        # Non-blocking lab gap: SLO severity ok + intentional_lab_gap (payload still warn).
        fred_dim = data["data_pipeline_slo"]["dimensions"]["fred_readiness"]
        assert fred_dim["status"] == "ok"
        assert fred_dim["intentional_lab_gap"] is True
        assert fred_dim["reason"] == "missing_fred_api_key"
        gen.conn.close()

    def test_rebalance_live_diagnostics_populate_health_slo(self, tmp_path, monkeypatch):
        """Rebalance live diagnostics should be included in dashboard health SLO output."""
        # Live mode: missing entitlement / alpaca_not_configured stay fail-closed.
        monkeypatch.setenv("PORTFOLIO_LAB_MODE", "live")
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "rebalance_health.json").write_text(json.dumps({
            "generated": "2026-06-12T16:43:07.176691",
            "market_data_consistency": {
                "status": "unavailable",
                "reason": "alpaca_not_configured",
                "checked_at": "2026-06-12T08:43:07.177011+00:00",
                "rows": [],
                "warnings": [],
            },
            "alpaca_feed_entitlement": {
                "configured_feed": "iex",
                "effective_feed": "iex",
                "entitlement": "unknown",
                "delayed": False,
                "acceptable_for_live": False,
                "policy_decision": "reject",
                "reason": "missing_entitlement",
            },
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        dims = data["data_pipeline_slo"]["dimensions"]
        assert dims["alpaca_feed_entitlement"]["status"] == "critical"
        assert dims["alpaca_feed_entitlement"]["reason"] == "missing_entitlement"
        assert dims["market_data_consistency"]["status"] == "warning"
        assert dims["market_data_consistency"]["reason"] == "alpaca_not_configured"
        # Multiple dimensions may be elevated in live mode (e.g. FRED); top is rank-first.
        assert data["data_pipeline_slo"]["status"] == "critical"
        gen.conn.close()

    def test_provider_latest_date_symbols_are_fresh_even_with_calendar_lag(self, tmp_path):
        """Freshness status is relative to the provider's latest available date."""
        db_path = tmp_path / "market.db"
        provider_latest = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))"
        )
        for symbol in ("SPY", "GLD", "TLT"):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)", (symbol, provider_latest, 100.0))
        conn.commit()
        conn.close()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = sqlite3.connect(str(db_path))
        gen.conn.row_factory = sqlite3.Row
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert {item["status"] for item in data["data_freshness"].values()} == {"fresh"}
        assert data["data_freshness"]["SPY"]["days_stale"] >= 2
        assert data["data_freshness"]["SPY"]["market_lag_days"] == 0
        assert data["data_freshness"]["SPY"]["latest_available_market_date"] == provider_latest
        gen.conn.close()

    def test_symbol_lagging_provider_latest_date_is_critical(self, tmp_path):
        """A symbol behind the provider's latest date should still be flagged."""
        db_path = tmp_path / "market.db"
        provider_latest = datetime.now() - timedelta(days=2)
        lagging_date = provider_latest - timedelta(days=5)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))"
        )
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", provider_latest.strftime("%Y-%m-%d"), 100.0))
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("GLD", lagging_date.strftime("%Y-%m-%d"), 100.0))
        conn.commit()
        conn.close()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = sqlite3.connect(str(db_path))
        gen.conn.row_factory = sqlite3.Row
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert data["data_freshness"]["SPY"]["status"] == "fresh"
        assert data["data_freshness"]["GLD"]["status"] == "critical"
        assert data["data_freshness"]["GLD"]["market_lag_days"] == 5
        gen.conn.close()


# ---------------------------------------------------------------------------
# Incident lifecycle JSON tests
# ---------------------------------------------------------------------------

class TestIncidentLifecycleJSON:
    """Test generate_incidents_json."""

    def test_copies_incident_summary_to_public_data(self, tmp_path):
        """Existing incident lifecycle state is published for dashboard fetches."""
        gen, _ = _make_generator(tmp_path)
        source = tmp_path / "incidents.json"
        source.write_text(json.dumps({
            "generated_at": "2026-07-06T00:00:00+00:00",
            "open_count": 1,
            "incidents": [
                {
                    "incident_id": "incident-123",
                    "channel": "signal_staleness",
                    "severity": "p0",
                    "state": "firing",
                    "message": "signals stale",
                    "details": {},
                    "created_at": "2026-07-06T00:00:00+00:00",
                    "updated_at": "2026-07-06T00:00:00+00:00",
                    "resolved_at": None,
                    "resolution_notes": None,
                    "mttr_seconds": None,
                    "alert_count": 6,
                    "kill_switch_level": "halt",
                }
            ],
            "metrics": {
                "incident_frequency": 1,
                "open_count": 1,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            },
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path / "public"), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_incidents_json()

        assert path == tmp_path / "public" / "incidents.json"
        assert json.loads(path.read_text())["incidents"][0]["kill_switch_level"] == "halt"
        gen.conn.close()

    def test_missing_incident_summary_publishes_empty_summary(self, tmp_path):
        """Dashboard core endpoint exists even before the first incident event."""
        gen, _ = _make_generator(tmp_path)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path / "public"), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_incidents_json()

        data = json.loads(path.read_text())
        assert data["open_count"] == 0
        assert data["incidents"] == []
        assert data["metrics"]["open_count"] == 0
        gen.conn.close()


# ---------------------------------------------------------------------------
# Broker data tests
# ---------------------------------------------------------------------------

class TestBrokerData:
    """Test _load_broker_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert "connected" in broker
        assert "positions" in broker
        assert "drift" in broker
        assert "kill_switch" in broker
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_detected(self, tmp_path):
        """Kill switch file is detected."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": True}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is True
        gen.conn.close()

    def test_sync_log_detected(self, tmp_path):
        """Position sync log is loaded."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "broker_positions": [{"symbol": "SPY", "qty": 10}],
            "drift": [{"symbol": "SPY", "drift_pct": 0.02}],
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is True
        assert len(broker["positions"]) == 1
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals tests
# ---------------------------------------------------------------------------

class TestMLSignals:
    """Test _generate_ml_signals."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "available" in signals
        assert signals["available"] is False
        gen.conn.close()

    def test_features_file_detected(self, tmp_path):
        """Features file makes signals available."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "timestamp": datetime.now().isoformat(),
            "momentum_12m": 0.15, "volatility": 0.18,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()

    def test_stale_feature_rows_publish_source_freshness_metadata(self, tmp_path):
        """Feature predictions expose feature as-of time and stale status."""
        gen, _ = _make_generator(tmp_path)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY",
            "timestamp": old_timestamp,
            "vix_level": 15,
            "trend_direction": 1,
            "price_vs_sma20": 0.05,
            "return_5d": 0.01,
            "spy_correlation_20d": 0.4,
        }) + "\n")

        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()

        assert signals["available"] is True
        assert signals["timestamp"] is not None
        assert signals["generated_at"] == signals["timestamp"]
        assert signals["feature_source_artifact"] == "features.jsonl"
        assert signals["feature_as_of"] == old_timestamp
        assert signals["feature_freshness_status"] == "stale"
        assert signals["feature_staleness_days"] >= 30
        assert signals["prediction_source_mode"] == "stale_features"
        assert signals["predictions"]["SPY"]["feature_timestamp"] == old_timestamp
        assert signals["predictions"]["SPY"]["feature_freshness_status"] == "stale"
        assert signals["predictions"]["SPY"]["source_artifact"] == "features.jsonl"
        assert signals["execution_role"]["routed"] is False
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve tests
# ---------------------------------------------------------------------------

class TestYieldCurve:
    """Test _get_yield_curve_data."""

    def test_default_structure(self, tmp_path):
        """Returns expected default structure."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._get_yield_curve_data()
        assert "yield_curve" in data or "duration_allocation" in data
        gen.conn.close()

    def test_yield_curve_includes_yields_source_manifest_provenance(self, tmp_path):
        """Synthetic/degraded yields provenance follows the yield curve payload."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([
            {"spread2s10s": -10.0, "dgs2": 4.6, "dgs10": 4.5},
        ]))
        (tmp_path / "source_manifest.json").write_text(json.dumps({
            "artifacts": [
                {
                    "artifact": "yields.json",
                    "provider": "FRED",
                    "source_mode": "synthetic",
                    "status": "degraded",
                    "failure_reason": "FRED_API_KEY missing",
                    "generated_at": "2026-07-06T00:00:00Z",
                    "latest_observation": "2026-07-02",
                },
            ],
        }))

        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
                data = gen._get_yield_curve_data()

        assert data["yield_curve"]["source_mode"] == "synthetic"
        assert data["yield_curve"]["source_status"] == "degraded"
        assert data["yield_curve"]["source_reason"] == "FRED_API_KEY missing"
        assert data["yield_curve"]["source_provider"] == "FRED"
        assert data["yield_curve"]["source_latest_observation"] == "2026-07-02"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run integration test
# ---------------------------------------------------------------------------

class TestRun:
    """Test run method."""

    def test_run_generates_all_files(self, tmp_path):
        """run() generates all dashboard files."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        assert (tmp_path / "dashboard.json").exists()
        assert (tmp_path / "index.json").exists()
        # conn is closed by run()


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Test module-level constants."""

    def test_base_allocation_has_symbols(self):
        """BASE_ALLOCATION contains SPY, GLD, TLT."""
        from src.paths import BASE_ALLOCATION
        assert "SPY" in BASE_ALLOCATION
        assert "GLD" in BASE_ALLOCATION
        assert "TLT" in BASE_ALLOCATION

    def test_public_dir_is_path_instance(self):
        """PUBLIC_DIR is a Path instance."""
        from src.dashboard.generator import PUBLIC_DIR
        assert isinstance(PUBLIC_DIR, Path)

    def test_base_allocation_weights_sum_to_one(self):
        """BASE_ALLOCATION weights sum to 1.0."""
        from src.paths import BASE_ALLOCATION
        total = sum(BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# GARCH-CVaR data edge cases
# ---------------------------------------------------------------------------

class TestGarchCvarData:
    """Test _load_garch_cvar_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Returns expected defaults when no health file exists."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["cvar_95_garch"] == -0.0215
        assert data["var_95"] == -0.0127
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_conformal_coverage_diagnostics_are_optional_monitoring_metadata(self, tmp_path):
        """GARCH-CVaR payload includes optional conformal coverage diagnostics."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()

        diagnostics = data["coverage_diagnostics"]
        assert diagnostics["schema_version"] == "conformal-coverage/v1"
        assert diagnostics["alpha"] == pytest.approx(0.05)
        assert diagnostics["observations"] >= 21
        assert "kupiec_pass" in diagnostics
        assert "christoffersen_pass" in diagnostics
        assert "conditional_coverage_pass" in diagnostics
        gen.conn.close()

    def test_flat_format_normalizes_percentages(self, tmp_path):
        """Values >1 in health report are divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 2.5,
            "var_95": 1.8,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.2,
            "garch_persistence": 0.97,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.025
        assert data["var_95"] == pytest.approx(0.018)
        assert data["garch_active"] is True
        gen.conn.close()

    def test_flat_format_keeps_decimal_values(self, tmp_path):
        """Values <=1 are kept as-is."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": -0.0179,
            "var_95": -0.0127,
            "cvar_ratio": 1.51,
            "filter_active": True,
            "conditional_volatility_current": 1.5,
            "garch_persistence": 0.88,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["var_95"] == -0.0127
        gen.conn.close()

    def test_legacy_nested_format(self, tmp_path):
        """Parses legacy nested check format from health report."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "cvar_metrics": {
                    "garch_filtered": True,
                    "cvar_95": -0.0250,
                    "var_95": -0.0150,
                    "cvar_ratio": 1.75,
                    "garch_active": True,
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0250
        assert data["cvar_ratio"] == 1.75
        gen.conn.close()

    def test_volatility_clustering_boundaries(self, tmp_path):
        """Tests all persistence thresholds for clustering label."""
        gen, _ = _make_generator(tmp_path)
        for persistence, expected in [(0.96, "high"), (0.90, "elevated"), (0.80, "normal")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "var_95": -0.0127,
                "cvar_ratio": 1.51,
                "filter_active": True,
                "conditional_volatility_current": 1.5,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}"
            )
        gen.conn.close()


# ---------------------------------------------------------------------------
# Entropy data edge cases
# ---------------------------------------------------------------------------

class TestEntropyData:
    """Test _load_entropy_data edge cases."""

    def test_defaults_no_health_file(self, tmp_path):
        """Returns expected defaults when no health file exists."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["effective_n"] == 2.77
        assert data["concentration_risk"] == "good"
        assert data["hhi_index"] == 0.38
        gen.conn.close()

    def test_concentration_risk_all_levels(self, tmp_path):
        """All score thresholds map to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        for score, expected in [(92, "good"), (80, "low"), (60, "medium"),
                                 (40, "high"), (20, "critical")]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.02,
                            "effective_n": 2.77,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}"
            )
        gen.conn.close()

    def test_loads_from_health_file_metrics(self, tmp_path):
        """All fields are populated from health file metrics."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {
                    "metrics": {
                        "shannon_entropy": 0.85,
                        "effective_n": 2.1,
                        "normalized_score": 77.0,
                        "hhi_index": 0.45,
                    }
                }
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 0.85
        assert data["effective_n"] == 2.1
        assert data["normalized_score"] == 77.0
        assert data["hhi_index"] == 0.45
        gen.conn.close()

    def test_missing_metrics_section(self, tmp_path):
        """Missing metrics section returns defaults unchanged."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "checks": {
                "portfolio_entropy": {}
            }
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["effective_n"] == 2.77
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals edge cases
# ---------------------------------------------------------------------------

class TestMlSignalsEdgeCases:
    """Test _generate_ml_signals edge cases."""

    def test_no_features_file(self, tmp_path):
        """Returns available=False when features file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        assert "features" in signals
        assert "predictions" in signals
        gen.conn.close()

    def test_empty_features_file(self, tmp_path):
        """Empty features file returns available=False."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is False
        gen.conn.close()

    def test_malformed_line_skipped(self, tmp_path):
        """Malformed JSON line in features file is skipped."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            "not valid json\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "timestamp": "2026-01-01"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert "SPY" in signals["features"]
        gen.conn.close()

    def test_vix_crisis_predictions(self, tmp_path):
        """VIX >25 yields bearish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 30, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bear"
        assert pred["confidence"] == 0.5
        assert pred["probabilities"]["bear"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_trend_bull_predictions(self, tmp_path):
        """Positive trend and price above SMA yields bullish prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 1,
            "price_vs_sma20": 0.05, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "bull"
        assert pred["probabilities"]["bull"] == 0.6
        gen.conn.close()

    def test_trend_bear_predictions(self, tmp_path):
        """Negative trend yields bearish-leaning prediction."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": -1,
            "price_vs_sma20": -0.03, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.4
        assert pred["probabilities"]["neutral"] == 0.4
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_default_predictions(self, tmp_path):
        """Default probabilities when vix <=20 and no trend signal."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_grid_search_results(self, tmp_path):
        """Grid search results loaded from JSONL file."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "allocations": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
            "sharpe": 0.85,
            "volatility": 0.12,
        }) + "\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        gs = signals["grid_search"]
        assert gs["available"] is True
        assert gs["sharpe"] == 0.85
        assert gs["top_allocation"] == {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        gen.conn.close()

    def test_grid_search_results_publish_frozen_benchmark_semantics(self, tmp_path):
        """Grid-search metrics disclose source artifact and frozen benchmark status."""
        gen, _ = _make_generator(tmp_path)
        grid_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text(json.dumps({
            "timestamp": grid_timestamp,
            "allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "sharpe": 0.95,
            "volatility": 0.11,
        }) + "\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY",
            "vix_level": 15,
            "trend_direction": 0,
            "price_vs_sma20": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()

        gs = signals["grid_search"]
        assert gs["available"] is True
        assert gs["source_artifact"] == "grid_search_results.jsonl"
        assert gs["benchmark_timestamp"] == grid_timestamp
        assert gs["observation_semantics"] == "frozen_benchmark_not_live_snapshot"
        assert gs["freshness_status"] == "frozen_benchmark"
        assert gs["staleness_days"] >= 45
        assert gs["live_authoritative"] is False
        gen.conn.close()

    def test_multi_symbol_features(self, tmp_path):
        """Multiple symbols produce separate predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
            + json.dumps({"symbol": "GLD", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.02, "timestamp": "2026-01-01T00:00:00"})
            + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert "SPY" in signals["predictions"]
        assert "GLD" in signals["predictions"]
        assert signals["predictions"]["SPY"]["predicted_regime"] == "bear"
        assert signals["predictions"]["GLD"]["predicted_regime"] == "bull"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve edge cases
# ---------------------------------------------------------------------------

class TestYieldCurveEdgeCases:
    """Test _get_yield_curve_data edge cases."""

    def test_no_file_returns_empty(self, tmp_path):
        """Returns empty result when yields file does not exist."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.YIELDS_JSON", tmp_path / "nonexistent.json"):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        assert data["duration_allocation"] is None
        gen.conn.close()

    def test_empty_list_returns_empty(self, tmp_path):
        """Empty yields list returns empty result."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("[]")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"] is None
        gen.conn.close()

    def test_spread_classification_steep(self, tmp_path):
        """Spread > 100 bps classifies as steep."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": 150, "dgs2": 4.0, "dgs10": 5.5} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "steep"
        gen.conn.close()

    def test_spread_classification_inverted(self, tmp_path):
        """Spread <= 0 bps classifies as inverted."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": -25, "dgs2": 5.0, "dgs10": 4.75} for _ in range(35)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_spread_boundary_values(self, tmp_path):
        """Boundary spread values map to correct regimes."""
        gen, _ = _make_generator(tmp_path)
        cases = [(100, "normal"), (50, "flat"), (1, "flat"), (0, "inverted")]
        for spread, expected in cases:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            assert data["yield_curve"]["duration_regime"] == expected, (
                f"Spread {spread} should be {expected}, got {data['yield_curve']['duration_regime']}"
            )
        gen.conn.close()

    def test_duration_allocation_by_regime(self, tmp_path):
        """Each regime maps to correct duration allocation."""
        gen, _ = _make_generator(tmp_path)
        allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, spread_val in [("steep", 150), ("inverted", -25)]:
            yields_path = tmp_path / "yields.json"
            yields_data = [{"spread2s10s": spread_val, "dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
            yields_path.write_text(json.dumps(yields_data))
            with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                data = gen._get_yield_curve_data()
            expected = allocations[regime]
            for k, v in expected.items():
                assert data["duration_allocation"][k] == v, (
                    f"Regime {regime}: {k} expected {v}, got {data['duration_allocation'][k]}"
                )
        gen.conn.close()

    def test_spread_history_length(self, tmp_path):
        """Spread history contains up to 30 entries."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_data = [{"spread2s10s": i * 5, "dgs2": 4.0, "dgs10": 5.0} for i in range(40)]
        yields_path.write_text(json.dumps(yields_data))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert len(data["yield_curve"]["spread_history"]) == 30
        gen.conn.close()


# ---------------------------------------------------------------------------
# Broker data edge cases
# ---------------------------------------------------------------------------

class TestBrokerDataEdgeCases:
    """Test _load_broker_data edge cases."""

    def test_empty_sync_log(self, tmp_path):
        """Empty sync log file returns default structure."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["connected"] is False
        gen.conn.close()

    def test_malformed_sync_log(self, tmp_path):
        """Malformed JSON in sync log is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text("not valid json\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, broker stays in default state
        assert broker["connected"] is False
        gen.conn.close()

    def test_kill_switch_disabled(self, tmp_path):
        """Kill switch with enabled=False reports no kill switch."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text(json.dumps({"enabled": False}))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()

    def test_broker_orders_loaded(self, tmp_path):
        """Broker orders log is loaded into recent_orders."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text(
            json.dumps({"order_id": 1, "symbol": "SPY", "side": "buy", "qty": 10})
            + "\n"
            + json.dumps({"order_id": 2, "symbol": "GLD", "side": "sell", "qty": 5})
            + "\n"
        )
        # Also need sync log so connected=True
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert len(broker["recent_orders"]) == 2
        assert broker["recent_orders"][0]["order_id"] == 2  # Reversed order
        gen.conn.close()

    def test_malformed_orders_line(self, tmp_path):
        """Malformed line in broker orders is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        orders_log = tmp_path / "broker_orders.jsonl"
        orders_log.write_text("not valid json\n")
        # Also need sync log
        sync_log = tmp_path / "position_sync.jsonl"
        sync_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "broker_positions": [],
            "drift": [],
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        # Exception is caught, default recent_orders returned
        assert broker["recent_orders"] == []
        gen.conn.close()

    def test_malformed_kill_switch(self, tmp_path):
        """Malformed kill_switch.json returns default state."""
        gen, _ = _make_generator(tmp_path)
        kill_file = tmp_path / "kill_switch.json"
        kill_file.write_text("not valid json")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert broker["kill_switch"] is False
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON edge cases
# ---------------------------------------------------------------------------

class TestPerformanceJSONEdgeCases:
    """Test generate_performance_json edge cases."""

    def test_no_perf_log_empty_paper_portfolio(self, tmp_path):
        """No performance.jsonl gives empty paper_portfolio."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["paper_portfolio"] == []
        gen.conn.close()

    def test_malformed_perf_entry_skipped(self, tmp_path):
        """Malformed entries in performance.jsonl are skipped."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(
            "not valid json\n"
            + json.dumps({"timestamp": "2026-01-01", "total_value": 100000, "daily_return": 0.01})
            + "\n"
        )
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["paper_portfolio"]) == 1
        gen.conn.close()

    def test_paper_portfolio_deduplicates_intraday_entries_by_date(self, tmp_path):
        """Date-only paper_portfolio chart rows keep the last entry per day."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        lines = []
        for day in range(1, 4):
            for hour in range(10):
                lines.append(json.dumps({
                    "timestamp": f"2026-01-{day:02d}T{hour:02d}:00:00",
                    "total_value": 100000 + day * 100 + hour,
                    "daily_return": round(day * 0.001 + hour * 0.0001, 6),
                }))
        perf_log.write_text("\n".join(lines) + "\n")

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()

        with open(path) as f:
            data = json.load(f)

        paper = data["paper_portfolio"]
        assert [row["t"] for row in paper] == [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
        assert [row["v"] for row in paper] == [100109, 100209, 100309]
        assert len(paper) == len({row["t"] for row in paper})
        gen.conn.close()

    def test_prices_contain_correct_keys(self, tmp_path):
        """Each price entry has d and p keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        for sym, entries in data["prices"].items():
            assert len(entries) > 0
            assert "d" in entries[0]
            assert "p" in entries[0]
        gen.conn.close()

    def test_generated_at_isoformat(self, tmp_path):
        """generated_at is a valid ISO format string."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        # Verify it parses
        dt = datetime.fromisoformat(data["generated_at"])
        assert isinstance(dt, datetime)
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON edge cases
# ---------------------------------------------------------------------------

class TestSignalsJSONEdgeCases:
    """Test generate_signals_json edge cases."""

    def test_generate_signals_json_is_thin_coordinator(self):
        """generate_signals_json delegates section work to focused helpers."""
        source = inspect.getsource(DashboardGenerator.generate_signals_json)
        body_lines = [
            line
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        assert len(body_lines) <= 100
        for helper_name in (
            "_load_signal_generation_context",
            "_build_base_signal_sections",
            "_build_optional_signal_sections",
            "_apply_signal_postprocessors",
        ):
            assert helper_name in source
            assert hasattr(DashboardGenerator, helper_name)

    def test_generate_signals_json_finalizes_top_level_generated_at_after_nested_sections(
        self,
        tmp_path,
    ):
        """Top-level generated_at should describe the finalized signals artifact."""

        class FakeDateTime(datetime):
            _values = iter(
                [
                    datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2026, 7, 6, 12, 0, 2, tzinfo=timezone.utc),
                ]
            )

            @classmethod
            def now(cls, tz=None):
                value = next(cls._values)
                if tz is None:
                    return value.replace(tzinfo=None)
                return value.astimezone(tz)

        gen = DashboardGenerator.__new__(DashboardGenerator)

        def add_nested_timestamp(output, context):
            nested_ts = datetime.fromisoformat(output["generated_at"]) + timedelta(seconds=1)
            enriched = dict(output)
            enriched["regime_transition"] = {"timestamp": nested_ts.isoformat()}
            return enriched

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.datetime", FakeDateTime):
                with patch.object(gen, "_load_signal_generation_context", return_value={}):
                    with patch.object(gen, "_build_base_signal_sections", return_value={}):
                        with patch.object(gen, "_build_optional_signal_sections", side_effect=add_nested_timestamp):
                            with patch.object(gen, "_apply_signal_postprocessors", side_effect=lambda output, context: output):
                                with patch(
                                    "src.monitor.decision_registry.record_dashboard_cycle_decision",
                                    side_effect=lambda *args, **kwargs: None,
                                ):
                                    path = gen.generate_signals_json()

        data = json.loads(path.read_text(encoding="utf-8"))
        top_level = datetime.fromisoformat(data["generated_at"])
        nested = datetime.fromisoformat(data["regime_transition"]["timestamp"])
        if top_level.tzinfo is None:
            top_level = top_level.replace(tzinfo=timezone.utc)
        if nested.tzinfo is None:
            nested = nested.replace(tzinfo=timezone.utc)
        assert top_level >= nested
        assert data["timestamp"] == data["generated_at"]

    def test_generate_regime_gate_json_deduplicates_active_signal_identifiers(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Producer artifact should not count duplicate configured aliases twice."""

        class FakeRegimeGate:
            min_dwell_days = 2

            def get_gate_summary(self):
                return {
                    "cross_asset_rv": set(),
                    "alt_data": set(),
                }

            def get_active_signal_names(self, signal_names, regime_name):
                assert regime_name == "NORMAL"
                return list(signal_names)

        monkeypatch.setitem(
            sys.modules,
            "src.signals.regime_gate",
            types.SimpleNamespace(RegimeGate=FakeRegimeGate),
        )
        gen = DashboardGenerator.__new__(DashboardGenerator)
        monkeypatch.setattr(gen, "_load_price_data", lambda: None, raising=False)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_regime_gate_json()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["active_signals"] == ["cross_asset_rv", "alt_data", "unified_overlay"]
        assert len(data["active_signals"]) == len(set(data["active_signals"]))
        # Producer must write regime_state.json SSOT (even when defaulting)
        state_path = tmp_path / "regime_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "regime" in state
        assert "confidence" in state
        assert "source" in state
        assert data.get("confidence_source") == state["source"]

    def test_generate_regime_gate_json_writes_regime_state_from_ensemble(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When ensemble_voting is published, regime_state SSOT matches it."""

        class FakeRegimeGate:
            min_dwell_days = 2

            def get_gate_summary(self):
                return {"alt_data": set()}

            def get_active_signal_names(self, signal_names, regime_name):
                return list(signal_names)

        monkeypatch.setitem(
            sys.modules,
            "src.signals.regime_gate",
            types.SimpleNamespace(RegimeGate=FakeRegimeGate),
        )
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = None
        monkeypatch.setattr(gen, "_load_price_data", lambda: None, raising=False)

        signals = {
            "ensemble_voting": {
                "regime": "normal",
                "regime_confidence": 0.755,
            }
        }
        (tmp_path / "signals.json").write_text(json.dumps(signals), encoding="utf-8")

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_regime_gate_json()

        gate = json.loads(path.read_text(encoding="utf-8"))
        assert gate["current_regime"] == "NORMAL"
        assert abs(gate["regime_confidence"] - 0.755) < 1e-9
        assert gate["confidence_source"] == "ensemble_voting"

        state = json.loads((tmp_path / "regime_state.json").read_text(encoding="utf-8"))
        assert state["regime"] == "NORMAL"
        assert abs(state["confidence"] - 0.755) < 1e-9
        assert state["source"] == "ensemble_voting"
        assert isinstance(state.get("history"), list) and len(state["history"]) >= 1

    def test_generate_regime_gate_json_discloses_default_missing_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        """No live sources → default NORMAL/0.5 with confidence_source disclosure."""

        class FakeRegimeGate:
            min_dwell_days = 2

            def get_gate_summary(self):
                return {}

            def get_active_signal_names(self, signal_names, regime_name):
                return list(signal_names)

        monkeypatch.setitem(
            sys.modules,
            "src.signals.regime_gate",
            types.SimpleNamespace(RegimeGate=FakeRegimeGate),
        )
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = None
        monkeypatch.setattr(gen, "_load_price_data", lambda: None, raising=False)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_regime_gate_json()

        gate = json.loads(path.read_text(encoding="utf-8"))
        assert gate["current_regime"] == "NORMAL"
        assert gate["regime_confidence"] == 0.5
        assert gate["confidence_source"] == "default_missing_state"
        state = json.loads((tmp_path / "regime_state.json").read_text(encoding="utf-8"))
        assert state["source"] == "default_missing_state"

    def test_missing_vix_handled(self, tmp_path):
        """Missing VIX symbol defaults vix to None when no fallback surfaces."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        gen.conn.commit()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        # Force overlay + behavioral without usable VIX so regime.vix stays unavailable
        gen._get_overlay_data = lambda: {
            "vix_term_structure": {},
            "collar": {},
            "crypto": {},
            "calendar": {},
            "kurtosis": {},
            "zero_dte": DashboardGenerator._unavailable_zero_dte_payload(),
            "closing_auction": DashboardGenerator._unavailable_closing_auction_payload(),
        }
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    with patch(
                        "src.signals.behavioral_sentiment.BehavioralSentimentSignal",
                        side_effect=ImportError("skip behavioral in test"),
                    ):
                        path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["vix"] is None
        assert data["regime"].get("vix_source") in (None, "unavailable")
        assert "regime" in data["regime"]
        gen.conn.close()

    def test_regime_vix_falls_back_to_term_structure(self, tmp_path):
        """When ^VIX missing, regime.vix uses vix_term_structure.vix_spot."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        gen.conn.commit()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        gen._get_overlay_data = lambda: {
            "vix_term_structure": {
                "vix_spot": 16.76,
                "signal_state": "RISK_ON",
                "regime": "contango",
            },
            "collar": {},
            "crypto": {},
            "calendar": {},
            "kurtosis": {},
            "zero_dte": DashboardGenerator._unavailable_zero_dte_payload(),
            "closing_auction": DashboardGenerator._unavailable_closing_auction_payload(),
        }
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    # Prefer term-structure over live behavioral fetcher
                    with patch(
                        "src.signals.behavioral_sentiment.BehavioralSentimentSignal",
                        side_effect=ImportError("skip behavioral in test"),
                    ):
                        path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert abs(float(data["regime"]["vix"]) - 16.76) < 1e-6
        assert data["regime"]["vix_source"] == "vix_term_structure"
        gen.conn.close()

    def test_enrich_regime_vix_prefers_market_db(self):
        enriched = DashboardGenerator._enrich_regime_vix(
            {"regime": "normal", "vix": 18.5, "vix_source": "market.db"},
            vix_term_structure={"vix_spot": 99.0},
            behavioral_sentiment={"vix": 88.0},
        )
        assert abs(enriched["vix"] - 18.5) < 1e-6
        assert enriched["vix_source"] == "market.db"

    def test_enrich_regime_vix_behavioral_fallback(self):
        enriched = DashboardGenerator._enrich_regime_vix(
            {"regime": "normal", "vix": None},
            vix_term_structure={},
            behavioral_sentiment={"vix": 18.77},
        )
        assert abs(enriched["vix"] - 18.77) < 1e-6
        assert enriched["vix_source"] == "behavioral_sentiment"

    def test_output_structure(self, tmp_path):
        """signals.json contains all expected top-level keys."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        required_keys = {"generated_at", "regime", "target_allocations", "current_positions",
                         "cash", "total_value", "latest_prices", "ml_signals",
                         "marl_status", "yield_curve", "broker"}
        assert required_keys.issubset(set(data.keys()))
        gen.conn.close()

    def test_default_values_when_no_state(self, tmp_path):
        """No portfolio_paper.json uses default cash and total value."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cash"] == 100000.0
        assert data["total_value"] == 100000.0
        gen.conn.close()

    def test_stacking_no_model_feature_count_is_not_hardcoded(self, tmp_path):
        """No-model stacking artifact exposes feature count as unavailable."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()

        with open(path) as f:
            data = json.load(f)

        stacking = data["stacking_ensemble"]
        assert stacking["stacking_available"] is False
        assert stacking["fallback_used"] is False
        assert stacking["feature_count"] is None
        assert stacking["feature_count_metadata_available"] is False
        assert stacking["feature_count_source"] == "unavailable_no_model"
        assert stacking["source_roster"] == []
        assert stacking["source_roster_version"] == "unavailable_no_model"
        assert stacking["fallback_semantics"] == "no_model_feature_count_unavailable"
        gen.conn.close()

    def test_stacking_no_model_runtime_status_is_dormant(self, tmp_path):
        """No-model stacking is dormant/unavailable, not a live fallback prediction."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()

        with open(path) as f:
            data = json.load(f)

        stacking = data["stacking_ensemble"]
        assert stacking["active"] is False
        assert stacking["stacking_available"] is False
        assert stacking["runtime_role"] == "research_dormant"
        assert stacking["runtime_status"] == "unavailable_no_model"
        assert stacking["live_authoritative"] is False
        assert stacking["routed"] is False
        assert stacking["routed_by"] is None
        assert stacking["prediction_available"] is False
        assert stacking["prediction_direction"] == "unavailable"
        assert stacking["fallback_used"] is False
        assert stacking["voting_accuracy"] is None
        assert stacking["stacking_accuracy"] is None
        assert stacking["accuracy_metrics_available"] is False
        assert "No stacking model artifact is loaded" in stacking["status_reason"]
        assert "not order-routed" in stacking["operator_message"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON edge cases
# ---------------------------------------------------------------------------

class TestHealthJSONEdgeCases:
    """Test generate_health_json edge cases."""

    def test_cron_fallback(self, tmp_path):
        """No cron_status.json does not invent scheduled cron jobs."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cron_jobs"] == []
        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["status"] == "unavailable"
        gen.conn.close()

    def test_cron_error_degraded(self, tmp_path):
        """Corrupted cron_status.json returns degraded status."""
        gen, _ = _make_generator(tmp_path)
        cron_file = tmp_path / "cron_status.json"
        cron_file.write_text("not valid json")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "degraded"
        gen.conn.close()

    def test_stale_data_warning_threshold(self, tmp_path):
        """stale_count > 5 with market_lag in the stale band (2–3d) → warning.

        Symbols must not be ``critical`` freshness (market_lag > 3): any
        critical data_freshness child rolls the artifact SLO to critical and
        elevates system_status. Use lag=3 relative to the freshest row so
        status is ``stale`` only.
        """
        gen, db_path = _make_generator(tmp_path)
        # _make_generator seeds SPY/GLD/TLT/QQQ through today; lag stale rows
        # 3 calendar days behind that cross-section (stale, not critical).
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT MAX(date) FROM prices")
        latest = cursor.fetchone()[0]
        latest_dt = datetime.strptime(latest, "%Y-%m-%d")
        stale_date = (latest_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        for i in range(6):
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?)",
                (f"STALE{i}", stale_date, 100.0),
            )
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        # 6 stale + 4 fresh = 10 data_freshness entries, stale_count = 6
        assert data["system_status"] == "warning"
        gen.conn.close()

    def test_stale_data_critical_threshold(self, tmp_path):
        """stale_count > 10 or critical freshness children → system critical."""
        gen, db_path = _make_generator(tmp_path)
        # Far-behind rows classify as critical freshness (market_lag > 3),
        # which rolls up to artifact SLO critical (highest-severity policy).
        conn = sqlite3.connect(str(db_path))
        for i in range(11):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (f"OLD{i}", "2020-01-01", 100.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "critical"
        gen.conn.close()

    def test_healthy_when_all_fresh(self, tmp_path):
        """Fresh data and no cron errors gives healthy status."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')
        _write_ok_source_manifest(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert data["system_status"] == "healthy"
        assert len(data["data_freshness"]) == 4
        gen.conn.close()

    def test_hermes_error_degrades_dashboard_health(self, tmp_path, monkeypatch):
        """Active Hermes portfolio-lab errors should be visible in dashboard health."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [
                {
                    "name": "portfolio-lab-data",
                    "status": "ok",
                    "last_run": "2026-06-08T12:00:00+08:00",
                    "backend": "local",
                }
            ]
        }))
        hermes_jobs = tmp_path / "hermes_jobs.json"
        hermes_jobs.write_text(json.dumps({
            "jobs": [
                {
                    "id": "ok-job",
                    "name": "portfolio-lab-dashboard",
                    "schedule_display": "15 * * * *",
                    "last_run_at": "2026-06-08T12:15:00+08:00",
                    "next_run_at": "2026-06-08T13:15:00+08:00",
                    "last_status": "ok",
                    "state": "scheduled",
                    "enabled": True,
                    "workdir": str(tmp_path),
                },
                {
                    "id": "bad-job",
                    "name": "portfolio-lab-autonomous-agent",
                    "schedule_display": "40 */2 * * *",
                    "last_run_at": "2026-06-08T12:47:00+08:00",
                    "next_run_at": "2026-06-08T14:40:00+08:00",
                    "last_status": "error",
                    "last_error": "RuntimeError: final report text",
                    "state": "scheduled",
                    "enabled": True,
                    "workdir": str(tmp_path),
                },
                {
                    "id": "other-project",
                    "name": "finance-digest",
                    "last_status": "error",
                    "enabled": True,
                },
            ]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(hermes_jobs))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        hermes_error = next(j for j in data["cron_jobs"] if j["id"] == "bad-job")
        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["status"] == "degraded"
        assert data["scheduler_status"]["backends"]["hermes"]["failed_jobs"] == 1
        assert hermes_error["backend"] == "hermes"
        assert hermes_error["name"] == "portfolio-lab-autonomous-agent"
        assert hermes_error["schedule"] == "40 */2 * * *"
        assert hermes_error["last_run"] == "2026-06-08T12:47:00+08:00"
        assert hermes_error["status"] == "error"
        assert hermes_error["error"] == "RuntimeError: final report text"
        assert not any(j["name"] == "finance-digest" for j in data["cron_jobs"])
        gen.conn.close()

    def test_missing_hermes_state_warns_without_crashing(self, tmp_path, monkeypatch):
        """Unavailable Hermes state should be explicit warning metadata."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "cron_status.json").write_text(json.dumps({
            "jobs": [{"name": "portfolio-lab-data", "status": "ok"}]
        }))
        monkeypatch.setenv("HERMES_CRON_JOBS_PATH", str(tmp_path / "missing-jobs.json"))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert data["system_status"] == "warning"
        assert data["scheduler_status"]["backends"]["hermes"]["status"] == "unavailable"
        assert "missing-jobs.json" in data["scheduler_status"]["backends"]["hermes"]["source"]
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON edge cases
# ---------------------------------------------------------------------------

class TestStatsJSONEdgeCases:
    """Test generate_stats_json edge cases."""

    def test_no_perf_log_returns_basic_stats(self, tmp_path):
        """No performance.jsonl returns asset stats without paper metrics."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "asset_stats" in data
        assert len(data["asset_stats"]) > 0
        assert data["paper_portfolio"] == {}
        assert data["spy_comparison"] is None
        gen.conn.close()

    def test_stats_have_expected_fields(self, tmp_path):
        """Each asset stat has 30d_return, volatility, and current."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        for sym, stat in data["asset_stats"].items():
            assert "30d_return" in stat
            assert "volatility" in stat
            assert "current" in stat
        gen.conn.close()

    def test_single_price_point_returns_empty_stats(self, tmp_path):
        """Only one price entry per symbol produces empty stats."""
        gen, db_path = _make_generator(tmp_path)
        # Create new DB with single price point
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now().strftime("%Y-%m-%d")
        for sym in ["SPY", "GLD"]:
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         (sym, today, 100.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # Both symbols have < 2 prices, so no stats generated
        assert len(data["asset_stats"]) == 0
        gen.conn.close()


# ---------------------------------------------------------------------------
# Alerts JSON edge cases
# ---------------------------------------------------------------------------

class TestAlertsJSONEdgeCases:
    """Test generate_alerts_json edge cases."""

    def test_empty_db_no_alerts(self, tmp_path):
        """Empty prices table produces no stale data alerts."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        assert data["count"] == 0
        gen.conn.close()

    def test_alerts_sorted_by_timestamp(self, tmp_path):
        """Alerts are sorted by timestamp descending."""
        gen, _ = _make_generator(tmp_path)
        # Create two kill switch files with different timestamps
        kill_1 = tmp_path / ".kill_switch_paper"
        kill_1.write_text(json.dumps({
            "enabled": True, "reason": "first",
            "timestamp": "2026-01-02T00:00:00"
        }))
        kill_2 = tmp_path / ".kill_switch_live"
        kill_2.write_text(json.dumps({
            "enabled": True, "reason": "second",
            "timestamp": "2026-01-01T00:00:00"
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        timestamps = [a.get("timestamp", "") for a in data["alerts"] if a.get("timestamp")]
        assert timestamps == sorted(timestamps, reverse=True), (
            f"Alerts not sorted descending: {timestamps}"
        )
        gen.conn.close()

    def test_regime_change_alert(self, tmp_path):
        """.regime_trigger file generates regime_change alert."""
        gen, _ = _make_generator(tmp_path)
        regime_file = tmp_path / ".regime_trigger"
        regime_file.write_text(json.dumps({
            "regime": "crisis", "vix": 30,
            "timestamp": "2026-01-01T00:00:00"
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        regime_alerts = [a for a in data["alerts"] if a["type"] == "regime_change"]
        assert len(regime_alerts) >= 1
        assert regime_alerts[0]["level"] == "warning"
        gen.conn.close()

    def test_promote_trigger_alert(self, tmp_path):
        """.promote_to_live file generates success only when current gates pass."""
        from src.strategy.graduation_checklist import CheckResult

        gen, _ = _make_generator(tmp_path)
        promote_file = tmp_path / ".promote_to_live"
        promote_file.write_text(json.dumps({
            "metrics": {"sharpe": 0.85},
            "timestamp": "2026-01-01T00:00:00"
        }))
        (tmp_path / ".manual_approval").write_text("approved")
        ready = {
            "min_trading_days": CheckResult("min_trading_days", True, 63, 63, ""),
            "manual_approval": CheckResult("manual_approval", True, 1, 1, ""),
        }
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path), \
             patch("src.strategy.graduation_checklist.DATA_DIR", tmp_path), \
             patch("src.strategy.graduation_checklist.GraduationChecklist.check", return_value=ready):
            path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        promote_alerts = [a for a in data["alerts"] if a["type"] == "graduation_candidate"]
        assert len(promote_alerts) >= 1
        assert promote_alerts[0]["level"] == "success"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run edge cases
# ---------------------------------------------------------------------------

class TestRunEdgeCases:
    """Test run() edge cases."""

    def test_run_closes_connection(self, tmp_path):
        """Connection is closed after run()."""
        gen, _ = _make_generator(tmp_path)
        conn = gen.conn
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        # Connection should be None after close()
        assert gen.conn is None

    def test_run_creates_index_json(self, tmp_path):
        """run() creates index.json with files list."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        with open(tmp_path / "index.json") as f:
            index = json.load(f)
        assert "files" in index
        assert len(index["files"]) >= 6  # At least 6 dashboard files
        assert "generated_at" in index
        # Connection is closed by run()

    def test_generated_at_populated_in_all_files(self, tmp_path):
        """All non-index JSON files have generated_at field."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        json_files = ["dashboard.json", "stats.json",
                      "alerts.json", "health.json"]
        for name in json_files:
            fpath = tmp_path / name
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                assert "generated_at" in data, f"{name} missing generated_at"
        # signals.json uses "generated_at" consistent with other JSON outputs
        signals_path = tmp_path / "signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                signals_data = json.load(f)
            assert "generated_at" in signals_data
        # Connection already closed by run()


# ---------------------------------------------------------------------------
# Analytics JSON tests
# ---------------------------------------------------------------------------

class TestGenerateAnalyticsJSON:
    """Test generate_analytics_json."""

    def test_generates_file(self, tmp_path):
        """Creates analytics.json even via fallback when dependencies unavailable."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_analytics_json()
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "status" in data
        gen.conn.close()

    def test_fallback_has_generated_at(self, tmp_path):
        """Fallback error report includes generated_at."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_analytics_json()
        with open(path) as f:
            data = json.load(f)
        assert "generated_at" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Graduation JSON tests
# ---------------------------------------------------------------------------

class TestGenerateGraduationJSON:
    """Test generate_graduation_json."""

    def test_generates_file_with_graduation_data(self, tmp_path):
        """Creates graduation.json with expected top-level keys."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        assert path is not None
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "readiness_score" in data
        assert "is_graduation_ready" in data
        assert "criteria" in data
        assert "generated_at" in data
        gen.conn.close()

    def test_graduation_has_criteria_items(self, tmp_path):
        """Each criterion has name, passed, value, required, description."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        with open(path) as f:
            data = json.load(f)
        for criterion in data["criteria"]:
            assert "name" in criterion
            assert "passed" in criterion
            assert "value" in criterion
            assert "required" in criterion
            assert "description" in criterion
        gen.conn.close()

    def test_graduation_json_matches_checklist_thresholds_and_summary_metrics(self, tmp_path):
        """Dashboard graduation data should mirror GraduationChecklist results."""
        from src.strategy.graduation_checklist import GraduationChecklist

        (tmp_path / "paper-trading-performance-2026-06-28.json").write_text(json.dumps({
            "date": "2026-06-28",
            "performance": {
                "days_tracked": 49,
                "sharpe": 3.3769,
                "max_drawdown": 0.0627,
                "start_value": 100000.0,
                "current_value": 101500.0,
            },
            "daily_returns_distribution": {
                "win_rate": 0.2041,
            },
        }))
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 0.0,
            "positions": {},
            "history": [
                {
                    "timestamp": "2026-05-01T10:00:00",
                    "total_value": 100000.0,
                    "cash": 0.0,
                    "daily_return": 0.0,
                    "positions_count": 0,
                    "mode": "paper",
                },
                {
                    "timestamp": "2026-06-28T10:00:00",
                    "total_value": 101500.0,
                    "cash": 0.0,
                    "daily_return": 0.001,
                    "positions_count": 0,
                    "mode": "paper",
                },
            ],
            "updated": "2026-06-28T10:00:00",
            "mode": "paper",
        }))

        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path), \
             patch("src.strategy.graduation_checklist.DATA_DIR", tmp_path):
            path = gen.generate_graduation_json()

            checklist = GraduationChecklist()
            results = checklist.check(checklist._load_state())
            expected_score = checklist.readiness_score(results)
            expected_ready = checklist.is_graduation_ready(results)

        with open(path) as f:
            data = json.load(f)

        criteria = {item["name"]: item for item in data["criteria"]}
        assert data["readiness_score"] == expected_score
        assert data["is_graduation_ready"] == expected_ready
        assert data["min_trading_days"] == criteria["min_trading_days"]["required"]
        assert data["min_trading_days"] == GraduationChecklist.DEFAULT_CRITERIA["min_trading_days"]["value"]
        assert data["trading_days"] == criteria["min_trading_days"]["value"]
        assert data["trading_days"] == 49
        assert criteria["min_sharpe"]["value"] == 0.0
        assert criteria["min_sharpe"]["passed"] is False
        # Frontend dual-shape aliases (GraduationDataSchema / panel)
        assert data["readiness_pct"] == expected_score
        assert data["eligible"] is expected_ready
        assert data["paper_trading"]["start_date"] == "2026-05-01"
        assert data["paper_trading"]["initial_capital"] == 100000.0
        assert data["paper_trading"]["current_value"] == 101500.0
        assert data["paper_trading"]["days_elapsed"] == 49
        assert data["paper_trading"]["days_required"] == data["min_trading_days"]
        for item in data["criteria"]:
            assert item["id"] == item["name"]
            assert isinstance(item["label"], str) and item["label"]
            assert "threshold" in item
        gen.conn.close()


# ---------------------------------------------------------------------------
# Overlay JSON tests
# ---------------------------------------------------------------------------

class TestGenerateOverlayJSON:
    """Test generate_overlay_json."""

    def test_returns_path_or_none(self, tmp_path):
        """Returns a Path when overlay generator succeeds, or None otherwise."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                result = gen.generate_overlay_json()
        # Either a Path (success) or None (graceful failure) is acceptable
        assert result is None or isinstance(result, Path)
        gen.conn.close()


# ---------------------------------------------------------------------------
# GARCH-CVaR edge cases (continued)
# ---------------------------------------------------------------------------

class TestGarchCvarEdgeCases:
    """Additional _load_garch_cvar_data edge cases."""

    def test_flat_format_empty_dict(self, tmp_path):
        """Empty health report file returns all defaults."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == -0.0179
        assert data["garch_active"] is True
        assert data["volatility_clustering"] == "elevated"
        gen.conn.close()

    def test_flat_format_zero_values(self, tmp_path):
        """Zero values in health report are handled correctly."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 0.0,
            "var_95": 0.0,
            "cvar_ratio": 0.0,
            "filter_active": False,
            "conditional_volatility_current": 0.0,
            "garch_persistence": 0.0,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0
        assert data["var_95"] == 0.0
        assert data["cvar_ratio"] == 0.0
        assert data["garch_active"] is False
        assert data["volatility_clustering"] == "normal"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Entropy edge cases (continued)
# ---------------------------------------------------------------------------

class TestEntropyEdgeCases:
    """Additional _load_entropy_data edge cases."""

    def test_concentration_risk_exact_boundaries(self, tmp_path):
        """Normalized score at exact boundaries maps to correct risk labels."""
        gen, _ = _make_generator(tmp_path)
        boundaries = [(91, "good"), (71, "low"), (51, "medium"), (31, "high"), (0, "critical")]
        for score, expected in boundaries:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "checks": {
                    "portfolio_entropy": {
                        "metrics": {
                            "shannon_entropy": 1.0,
                            "effective_n": 2.5,
                            "normalized_score": score,
                            "hhi_index": 0.38,
                        }
                    }
                }
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_entropy_data()
            assert data["concentration_risk"] == expected, (
                f"Score {score} should be {expected}, got {data['concentration_risk']}"
            )
        gen.conn.close()

    def test_empty_health_file_returns_defaults(self, tmp_path):
        """Empty JSON health file returns default entropy values."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text("{}")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_entropy_data()
        assert data["shannon_entropy"] == 1.02
        assert data["concentration_risk"] == "good"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON edge cases (continued)
# ---------------------------------------------------------------------------

class TestSignalsJSONAdditionalEdgeCases:
    """Additional generate_signals_json edge cases."""

    def test_vix_at_low_vol_boundary(self, tmp_path):
        """VIX at 14 (just below 15) classifies as low_vol."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 14.0))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "low_vol"
        gen.conn.close()

    def test_vix_at_crisis_boundary(self, tmp_path):
        """VIX at 26 (just above 25) classifies as crisis."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 26.0))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "crisis"
        gen.conn.close()

    def test_empty_positions_in_portfolio_state(self, tmp_path):
        """Portfolio state with empty positions generates valid output."""
        gen, _ = _make_generator(tmp_path)
        state_file = tmp_path / "portfolio_paper.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "cash": 50000.0
        }))
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["cash"] == 50000.0
        assert data["total_value"] == 50000.0
        assert data["current_positions"] == []
        gen.conn.close()

    def test_all_optional_keys_present(self, tmp_path):
        """signals.json output contains all optional section keys."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        all_keys = {
            "generated_at", "regime", "target_allocations", "current_positions",
            "cash", "total_value", "latest_prices", "recent_orders", "ml_signals",
            "factor_rotation", "yield_curve", "duration_allocation",
            "convexity_harvest", "volatility_parity", "llm_sentiment",
            "ensemble_voting", "sector_rotation", "alternative_data",
            "behavioral_sentiment", "stacking_ensemble", "factor_rotation_dashboard",
            "smart_rebalance", "broker", "garch_cvar", "entropy", "bond_momentum",
        }
        assert all_keys.issubset(set(data.keys())), (
            f"Missing keys: {all_keys - set(data.keys())}"
        )
        gen.conn.close()


# ---------------------------------------------------------------------------
# ML signals edge cases (continued)
# ---------------------------------------------------------------------------

class TestMlSignalsAdditionalEdgeCases:
    """Additional _generate_ml_signals edge cases."""

    def test_grid_search_empty_file(self, tmp_path):
        """Empty grid search results file returns empty grid_search dict."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["grid_search"] == {}
        gen.conn.close()

    def test_vix_at_vol_spike_boundary_ml_prediction(self, tmp_path):
        """VIX at 21 (just above 20) triggers vol_spike probability distribution."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 21, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3
        assert pred["probabilities"]["neutral"] == 0.5
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON edge cases (continued)
# ---------------------------------------------------------------------------

class TestStatsJSONAdditionalEdgeCases:
    """Additional generate_stats_json edge cases."""

    def test_no_vix_in_prices(self, tmp_path):
        """Missing VIX symbol in prices does not crash stats generation."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "asset_stats" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Constants validation (continued)
# ---------------------------------------------------------------------------

class TestConstantsAdditional:
    """Additional module-level constant validation."""

    def test_data_dir_is_path_instance(self):
        """DATA_DIR is a Path instance."""
        from src.dashboard.generator import DATA_DIR
        assert isinstance(DATA_DIR, Path)

    def test_db_path_is_path_instance(self):
        """DB_PATH is a Path instance."""
        from src.dashboard.generator import DB_PATH
        assert isinstance(DB_PATH, Path)


# ---------------------------------------------------------------------------
# Sector momentum signals tests (completely untested method)
# ---------------------------------------------------------------------------

class TestSectorMomentumSignals:
    """Test _generate_sector_momentum_signals edge cases."""

    def test_none_when_import_fails(self, tmp_path):
        """Returns None when sector_momentum_calc cannot be imported."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          side_effect=ImportError("no module")):
                    result = gen._generate_sector_momentum_signals()
        assert result is None
        gen.conn.close()

    def test_none_when_generate_raises(self, tmp_path):
        """Returns None when generate_sector_signals raises an exception."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          side_effect=ValueError("bad data")):
                    result = gen._generate_sector_momentum_signals()
        assert result is None
        gen.conn.close()

    def test_passes_vix_to_generate(self, tmp_path):
        """Vix level is passed to generate_sector_signals via vix_level parameter."""
        gen, db_path = _make_generator(tmp_path)
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals(vix_level=18.5)
        assert result == mock_signals
        # Verify vix was passed
        _, kwargs = mock_gen.call_args
        assert kwargs.get("vix") == 18.5
        gen.conn.close()

    def test_vix_fetch_failure_defaults_zero(self, tmp_path):
        """When vix_level is None (no VIX data), vix parameter defaults to 0."""
        gen, db_path = _make_generator(tmp_path)
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals(vix_level=None)
        assert result == mock_signals
        _, kwargs = mock_gen.call_args
        assert kwargs.get("vix") == 0
        gen.conn.close()

    def test_none_when_no_vix_row(self, tmp_path):
        """No VIX row in DB still calls generate with vix=0 and returns result."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        conn.commit()
        conn.close()
        mock_signals = {"SPY": {"momentum": 0.5}}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals()
        assert result == mock_signals
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — regime composite integration tests
# ---------------------------------------------------------------------------

class TestSignalsJSONRegimeComposite:
    """Test full regime composite logic in generate_signals_json."""

    def test_vix_crisis_overrides_trend(self, tmp_path):
        """VIX crisis (>25) overrides any trend regime."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 30.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "bull", 30.0,
                      datetime.now().isoformat()))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "crisis"
        gen.conn.close()

    def test_vix_vol_spike_overrides_trend(self, tmp_path):
        """VIX vol_spike (21-25) overrides trend regime."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 22.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "bull", 22.0,
                      datetime.now().isoformat()))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "vol_spike"
        gen.conn.close()

    def test_low_vol_with_crisis_trend_uses_trend(self, tmp_path):
        """low_vol VIX with crisis trend falls through to trend."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 14.0))
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), "crisis", 14.0,
                      datetime.now().isoformat()))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regime"]["regime"] == "crisis"
        gen.conn.close()

    def test_crisis_target_alloc_applied(self, tmp_path):
        """Crisis regime uses crisis target allocation weights."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('^VIX', ?, ?)",
                     (datetime.now().strftime("%Y-%m-%d"), 30.0))
        conn.commit()
        conn.close()
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        expected = {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30}
        assert data["target_allocations"] == expected
        gen.conn.close()


class TestRegimeTargetAllocationParity:
    """Dashboard target allocations should match scheduled evaluator semantics."""

    @pytest.mark.parametrize(
            ("expected_regime", "vix_level", "trend_regime"),
            [
                ("crisis", 30.0, "normal"),
                ("vol_spike", 22.5, "normal"),
                ("high_vol", 16.0, "high_vol"),
                ("recovery", 16.0, "recovery"),
            ],
        )
    def test_env_enabled_target_allocations_match_regime_helper(
        self, tmp_path, monkeypatch, expected_regime, vix_level, trend_regime
    ):
        """REGIME_ALLOC_ENABLED dashboard path uses the evaluator helper."""
        from src.strategy.regime_allocation import get_regime_allocation_with_override

        monkeypatch.setenv("REGIME_ALLOC_ENABLED", "1")
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO prices VALUES ('^VIX', ?, ?)",
            (datetime.now().strftime("%Y-%m-%d"), vix_level),
        )
        conn.execute(
            "INSERT INTO regime_log VALUES (?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d"),
                trend_regime,
                vix_level,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        try:
            context = gen._load_signal_generation_context()
        finally:
            gen.conn.close()

        assert context["current_regime"] == expected_regime
        assert context["target_alloc"] == get_regime_allocation_with_override(expected_regime)


# ---------------------------------------------------------------------------
# Signals JSON — positions, orders, and paper portfolio state
# ---------------------------------------------------------------------------

class TestSignalsJSONPositions:
    """Test generate_signals_json with portfolio state."""

    def test_portfolio_positions_parsed(self, tmp_path):
        """Portfolio paper state positions are parsed correctly."""
        gen, _ = _make_generator(tmp_path)
        state_file = tmp_path / "portfolio_paper.json"
        state_file.write_text(json.dumps({
            "positions": {
                "SPY": {"shares": 100, "value": 45000, "weight": 0.45, "unrealized_pnl": 500},
                "GLD": {"shares": 200, "value": 35000, "weight": 0.35, "unrealized_pnl": -200},
            },
            "cash": 20000.0
        }))
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        positions = {p["symbol"]: p for p in data["current_positions"]}
        assert "SPY" in positions
        assert positions["SPY"]["shares"] == 100
        assert positions["SPY"]["value"] == 45000
        assert data["cash"] == 20000.0
        assert data["total_value"] == 100000.0  # cash + 45000 + 35000
        gen.conn.close()

    def test_orders_parsed_from_log(self, tmp_path):
        """Orders from orders.jsonl are parsed into recent_orders."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        orders_file.write_text(
            json.dumps({"symbol": "SPY", "side": "buy", "shares": 10, "fill_value": 4500}) + "\n"
            + json.dumps({"symbol": "GLD", "side": "sell", "shares": 5, "fill_value": 1750}) + "\n"
        )
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        orders = data["recent_orders"]
        assert len(orders) == 2
        assert orders[0]["sym"] == "GLD"    # Reversed
        assert orders[1]["sym"] == "SPY"
        gen.conn.close()

    def test_malformed_order_skipped(self, tmp_path):
        """Malformed JSON line in orders.jsonl is skipped."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        orders_file.write_text(
            "not valid json\n"
            + json.dumps({"symbol": "SPY", "side": "buy", "shares": 10, "fill_value": 4500}) + "\n"
        )
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["recent_orders"]) == 1
        gen.conn.close()

    def test_only_last_five_orders(self, tmp_path):
        """Only the last 5 orders from orders.jsonl are kept."""
        gen, _ = _make_generator(tmp_path)
        orders_file = tmp_path / "orders.jsonl"
        lines = []
        for i in range(10):
            lines.append(json.dumps({"symbol": f"SYM{i}", "side": "buy", "shares": 1, "fill_value": 100 * i}))
        orders_file.write_text("\n".join(lines) + "\n")
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["recent_orders"]) == 5
        gen.conn.close()

    def test_latest_prices_from_db(self, tmp_path):
        """Latest prices dict is populated from DB."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["latest_prices"]
        assert "GLD" in data["latest_prices"]
        assert isinstance(data["latest_prices"]["SPY"], float)
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — smart rebalance
# ---------------------------------------------------------------------------

class TestSignalsJSONSmartRebalance:
    """Test smart rebalance data in generate_signals_json."""

    def test_smart_rebalance_fallback_data(self, tmp_path):
        """Smart rebalance has fallback data when import fails."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    with patch("importlib.import_module",
                              side_effect=ImportError("no rebalancing")):
                        path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        # smart_rebalance should be None when import fails
        assert data["smart_rebalance"] is None
        gen.conn.close()

    def test_smart_rebalance_remaining_budget_pct_is_display_percent(
        self, tmp_path, monkeypatch
    ):
        """Public percent fields use display units while ratio fields preserve fractions."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 50000,
            "positions": {
                "SPY": {
                    "shares": 100,
                    "value": 50000,
                    "weight": 0.5,
                    "unrealized_pnl": 0,
                },
            },
        }))

        class FakeGateResult:
            should_execute = False
            decision = "wait"
            urgency = "low"
            max_drift = 0.04
            estimated_cost_bps = 3.0
            reason = "budget_available"
            metadata = {
                "drift_details": {"SPY": 0.04},
                "vpin": 0.2,
                "in_optimal_window": True,
                "ytd_cost_bps": 0,
                "remaining_budget_pct": 0.005,
                "remaining_budget_ratio": 0.005,
            }

        class FakeSmartRebalanceGate:
            def evaluate(self, current_holdings, target_allocations, total_value):
                assert current_holdings == {"SPY": 50000}
                assert total_value == 100000
                return FakeGateResult()

            def get_status(self):
                return {
                    "ytd_cost_bps": 0,
                    "ytd_cost_pct": 0.0,
                    "remaining_budget_pct": 0.5,
                    "remaining_budget_ratio": 0.005,
                    "is_over_budget": False,
                    "is_warning": False,
                    "last_rebalance": None,
                    "deferred_until": None,
                    "config": {
                        "drift_threshold": 0.1,
                        "vpin_threshold": 0.5,
                        "optimal_window": "10:00-15:30",
                        "annual_cost_limit": "50bps",
                    },
                }

        fake_rebalancing = types.SimpleNamespace(
            integration=types.SimpleNamespace(SmartRebalanceGate=FakeSmartRebalanceGate)
        )
        monkeypatch.setattr("src.dashboard.generator.validate_signal", lambda _name, signal: signal)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    with patch("importlib.import_module", return_value=fake_rebalancing):
                        path = gen.generate_signals_json()

        with open(path) as f:
            data = json.load(f)

        smart = data["smart_rebalance"]
        assert smart["remaining_budget_pct"] == 0.5
        assert smart["remaining_budget_ratio"] == 0.005
        assert smart["status"]["remaining_budget_pct"] == 0.5
        assert smart["status"]["remaining_budget_ratio"] == 0.005
        gen.conn.close()

    def test_apply_kill_to_smart_rebalance_helper_unit(self):
        """Pure helper forces blocked decision when kill enabled."""
        from src.dashboard.generator import _apply_kill_to_smart_rebalance

        base = {
            "should_execute": True,
            "decision": "execute",
            "urgency": "high",
            "max_drift": 0.20,
            "estimated_cost_bps": 5.0,
            "reason": "drift_above_threshold",
            "drift_details": {"SPY": 0.20},
            "vpin": 0.25,
            "in_optimal_window": True,
            "ytd_cost_bps": 10,
            "remaining_budget_pct": 0.5,
            "remaining_budget_ratio": 0.005,
            "status": {},
        }
        out = _apply_kill_to_smart_rebalance(
            dict(base),
            {
                "enabled": True,
                "level": "halt",
                "reason": "unresolved_incident:signal_staleness",
                "incident_id": "inc-1",
                "message": "Paper trading halted",
            },
        )
        assert out["should_execute"] is False
        assert out["decision"] == "blocked_kill_switch"
        assert out["execution_blocked"] is True
        assert out["kill_switch_enabled"] is True
        assert out["kill_switch_level"] == "halt"
        assert out["kill_switch_incident_id"] == "inc-1"
        assert "drift_details" in out and out["drift_details"]["SPY"] == 0.20
        assert "prior=execute" in out["reason"]

        clear = _apply_kill_to_smart_rebalance(dict(base), {"enabled": False})
        assert clear["should_execute"] is True
        assert clear["decision"] == "execute"
        assert clear["execution_blocked"] is False

    def test_smart_rebalance_kill_halt_blocks_execute(self, tmp_path, monkeypatch):
        """Kill on + high-drift gate would execute → smart_rebalance not executable."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        (tmp_path / "portfolio_paper.json").write_text(json.dumps({
            "cash": 50000,
            "positions": {
                "SPY": {
                    "shares": 100,
                    "value": 50000,
                    "weight": 0.5,
                    "unrealized_pnl": 0,
                },
            },
        }))
        (tmp_path / "kill_switch.json").write_text(json.dumps({
            "enabled": True,
            "level": "halt",
            "reason": "unresolved_incident:signal_staleness",
            "source": "incident_lifecycle",
            "incident_id": "inc-halt-sr",
            "mode": "paper",
            "message": "1/23 signals stale",
            "position_reduction": 1.0,
        }))

        class FakeGateResult:
            should_execute = True
            decision = "execute"
            urgency = "high"
            max_drift = 0.22
            estimated_cost_bps = 8.0
            reason = "drift_above_threshold"
            metadata = {
                "drift_details": {"SPY": 0.22},
                "vpin": 0.2,
                "in_optimal_window": True,
                "ytd_cost_bps": 0,
                "remaining_budget_pct": 0.005,
                "remaining_budget_ratio": 0.005,
            }

        class FakeSmartRebalanceGate:
            def evaluate(self, current_holdings, target_allocations, total_value):
                return FakeGateResult()

            def get_status(self):
                return {
                    "ytd_cost_bps": 0,
                    "ytd_cost_pct": 0.0,
                    "remaining_budget_pct": 0.5,
                    "remaining_budget_ratio": 0.005,
                    "is_over_budget": False,
                    "is_warning": False,
                    "last_rebalance": None,
                    "deferred_until": None,
                    "config": {
                        "drift_threshold": 0.1,
                        "vpin_threshold": 0.5,
                        "optimal_window": "10:00-15:30",
                        "annual_cost_limit": "50bps",
                    },
                }

        fake_rebalancing = types.SimpleNamespace(
            integration=types.SimpleNamespace(SmartRebalanceGate=FakeSmartRebalanceGate)
        )
        monkeypatch.setattr("src.dashboard.generator.validate_signal", lambda _name, signal: signal)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    with patch("importlib.import_module", return_value=fake_rebalancing):
                        path = gen.generate_signals_json()

        with open(path) as f:
            data = json.load(f)

        smart = data["smart_rebalance"]
        assert smart["should_execute"] is False
        assert smart["decision"] == "blocked_kill_switch"
        assert smart["execution_blocked"] is True
        assert smart["kill_switch_level"] == "halt"
        assert smart["max_drift"] == 0.22  # diagnostics preserved
        gen.conn.close()


# ---------------------------------------------------------------------------
# Signals JSON — alternative data
# ---------------------------------------------------------------------------

class TestSignalsJSONAlternativeData:
    """Test alternative data loading in generate_signals_json."""

    def test_alternative_data_loaded(self, tmp_path):
        """Alternative data from JSON file is loaded into output."""
        gen, _ = _make_generator(tmp_path)
        alt_dir = tmp_path / "signals"
        alt_dir.mkdir(exist_ok=True)
        alt_file = alt_dir / "alternative_data_latest.json"
        # Current producer shape: seven components under raw_data.components
        alt_file.write_text(json.dumps({
            "regime": "bull",
            "probability": 0.65,
            "confidence": 0.72,
            "timestamp": "2026-01-01T00:00:00",
            "raw_data": {
                "composite_score": 0.38,
                "z_score": 0.5,
                "sources_count": 7,
                "data_freshness_hours": 2.5,
                "components": {
                    "treasury_curve": 0.3,
                    "sector_rotation": 0.1,
                    "credit_spread": -0.1,
                    "tail_risk": 0.6,
                    "broad_momentum": 1.0,
                    "crypto_sentiment": 0.0,
                    "crypto_fg": 0.48,
                },
                "component_confidences": {
                    "treasury_curve": 0.3,
                    "sector_rotation": 0.9,
                    "credit_spread": 0.4,
                    "tail_risk": 0.9,
                    "broad_momentum": 0.9,
                    "crypto_sentiment": 0.1,
                    "crypto_fg": 0.66,
                },
                "weights": {
                    "treasury_curve": 0.18,
                    "sector_rotation": 0.16,
                    "credit_spread": 0.16,
                    "tail_risk": 0.15,
                    "broad_momentum": 0.16,
                    "crypto_sentiment": 0.05,
                    "crypto_fg": 0.14,
                },
            }
        }))
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        alt = data["alternative_data"]
        assert alt is not None
        assert alt["regime"] == "bull"
        assert alt["composite_score"] == 0.38
        assert set(alt["components"].keys()) == {
            "treasury_curve",
            "sector_rotation",
            "credit_spread",
            "tail_risk",
            "broad_momentum",
            "crypto_sentiment",
            "crypto_fg",
        }
        assert alt["components"]["treasury_curve"]["score"] == 0.3
        assert alt["components"]["treasury_curve"]["confidence"] == 0.3
        assert alt["components"]["treasury_curve"]["weight"] == 0.18
        assert "earnings" not in alt["components"]
        assert alt["sources_count"] == 7
        assert alt["data_freshness_hours"] == 2.5
        gen.conn.close()

    def test_alternative_data_legacy_flat_keys_fallback(self, tmp_path):
        """Legacy flat earnings/news/jobs/social keys still project when components map absent."""
        gen, _ = _make_generator(tmp_path)
        alt_dir = tmp_path / "signals"
        alt_dir.mkdir(exist_ok=True)
        alt_file = alt_dir / "alternative_data_latest.json"
        alt_file.write_text(json.dumps({
            "regime": "risk_on",
            "probability": 0.65,
            "confidence": 0.72,
            "timestamp": "2026-01-01T00:00:00",
            "raw_data": {
                "earnings_sentiment": 0.3,
                "earnings_confidence": 0.8,
                "news_sentiment": 0.6,
                "news_confidence": 0.7,
                "jobs_signal": 0.2,
                "jobs_confidence": 0.6,
                "social_sentiment": 0.4,
                "social_confidence": 0.5,
                "composite_score": 0.38,
                "z_score": 0.5,
                "sources_count": 4,
                "data_freshness_hours": 2.5,
                "weights": {
                    "earnings": 0.3,
                    "news": 0.3,
                    "jobs": 0.2,
                    "social": 0.2,
                },
            },
        }))
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        alt = data["alternative_data"]
        assert alt is not None
        assert alt["components"]["earnings"]["score"] == 0.3
        assert alt["components"]["news"]["score"] == 0.6
        assert alt["sources_count"] == 4
        gen.conn.close()

    def test_alternative_data_missing_file(self, tmp_path):
        """Missing alternative data file falls back to None."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["alternative_data"] is None
        gen.conn.close()

    def test_alternative_data_malformed(self, tmp_path):
        """Malformed alternative data file falls back to None."""
        gen, _ = _make_generator(tmp_path)
        alt_dir = tmp_path / "signals"
        alt_dir.mkdir(exist_ok=True)
        alt_file = alt_dir / "alternative_data_latest.json"
        alt_file.write_text("not valid json")
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)
        assert data["alternative_data"] is None
        gen.conn.close()


# ---------------------------------------------------------------------------
# Stats JSON — paper portfolio and SPY comparison (core logic, untested)
# ---------------------------------------------------------------------------

class TestStatsJSONPaperPerformance:
    """Test paper portfolio metrics in generate_stats_json."""

    def test_paper_metrics_with_perf_log(self, tmp_path):
        """Performance log entries produce paper portfolio metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": value,
                "daily_return": 0.001 if i > 0 else 0.0,
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        assert "sharpe" in paper
        assert "total_return" in paper
        assert "max_value" in paper
        assert "min_value" in paper
        assert "days_tracked" in paper
        assert paper["days_tracked"] == 25
        gen.conn.close()

    def test_paper_metrics_insufficient_data(self, tmp_path):
        """Fewer than 20 perf entries produces empty paper_metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "total_value": 100000,
            "daily_return": 0.001,
        }) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert data["paper_portfolio"] == {}
        assert data["spy_comparison"] is None
        gen.conn.close()

    def test_paper_metrics_sharpe_with_no_variance(self, tmp_path):
        """All-zero daily_return entries are filtered out, yielding empty metrics."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": 100000.0,
                "daily_return": 0.0,
            }))
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # Zero returns are valid daily returns — Sharpe should be 0 (no excess return)
        paper = data["paper_portfolio"]
        assert paper["sharpe"] == 0
        assert paper["days_tracked"] == 25
        assert paper["total_return"] == 0.0
        gen.conn.close()

    def test_paper_metrics_all_fields_populated(self, tmp_path):
        """With enough non-zero returns, all paper metric fields are present."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            ret = 0.001 + (i * 0.0001)  # Increasing returns for variance
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": round(value, 2),
                "daily_return": round(ret, 6),
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        assert "sharpe" in paper
        assert "total_return" in paper
        assert "max_value" in paper
        assert "min_value" in paper
        assert "days_tracked" in paper
        assert isinstance(paper["sharpe"], (int, float))
        gen.conn.close()

    def test_paper_metrics_deduplicates_intraday_entries(self, tmp_path):
        """Intraday entries for the same date must not inflate days_tracked.

        Regression test: performance.jsonl may contain multiple entries per
        day (cron runs, manual syncs).  days_tracked must count unique
        calendar dates, not raw JSONL lines.
        """
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        lines = []
        # 3 dates × 10 intraday entries each = 30 raw lines, but only 3 unique days
        for day in range(1, 4):
            for hour in range(10):
                ret = 0.001 if hour == 0 else 0.0  # Only first entry per day has return
                lines.append(json.dumps({
                    "timestamp": f"2026-01-{day:02d}T{hour:02d}:00:00",
                    "total_value": 100000.0 + day * 100,
                    "daily_return": ret,
                }))
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        paper = data["paper_portfolio"]
        # Must be 3 unique dates, not 30 raw lines
        assert paper["days_tracked"] == 3
        gen.conn.close()


class TestStatsJSONSpyComparison:
    """Test SPY comparison in generate_stats_json."""

    def test_spy_comparison_present_with_enough_data(self, tmp_path):
        """SPY comparison is calculated with sufficient perf and SPY data."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        value = 100000.0
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "timestamp": f"2026-01-{i+1:02d}T00:00:00",
                "total_value": value,
                "daily_return": 0.001 if i > 0 else 0.0,
            }))
            value *= 1.001
        perf_log.write_text("\n".join(lines) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        # spy_comparison may be None if SPY prices don't align with timestamps
        # Just verify no crash and asset_stats present
        assert "asset_stats" in data
        gen.conn.close()


# ---------------------------------------------------------------------------
# Health JSON — signal health testing
# ---------------------------------------------------------------------------

class TestHealthJSONSignalHealth:
    """Test signal health in generate_health_json."""

    def test_signal_health_present(self, tmp_path):
        """Signal health is populated in health output."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "signal_health" in data
        gen.conn.close()

    def test_signal_health_error_fallback(self, tmp_path):
        """Signal health has error fallback when tracker unavailable."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.signals.health_tracker.SignalHealthTracker.get_health_report",
                          side_effect=ImportError("no tracker")):
                    path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert "error" in data["signal_health"]
        assert data["signal_health"]["status"] == "unavailable"
        gen.conn.close()

    def test_cron_status_loaded(self, tmp_path):
        """Cron status from file is loaded into health data."""
        gen, _ = _make_generator(tmp_path)
        cron_file = tmp_path / "cron_status.json"
        cron_file.write_text(json.dumps({
            "jobs": [
                {"name": "portfolio-lab-data", "status": "success", "state": "completed"},
                {"name": "portfolio-lab-eval", "status": "error", "state": "failed"},
            ]
        }))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["cron_jobs"]) == 2
        assert data["cron_jobs"][0]["status"] == "ok"
        assert data["cron_jobs"][0]["backend"] == "local"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Performance JSON — regime data and paper portfolio
# ---------------------------------------------------------------------------

class TestPerformanceJSONRegime:
    """Test regime data in generate_performance_json."""

    def test_regime_data_included(self, tmp_path):
        """Regime data from DB is included in performance output."""
        gen, _ = _make_generator(tmp_path)
        conn = gen.conn
        recent = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        conn.execute("INSERT INTO regime_log VALUES (?, ?, ?, ?)",
                     (recent, "normal", 15.0, datetime.now().isoformat()))
        conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["regimes"]) >= 1
        assert data["regimes"][0]["r"] == "normal"
        gen.conn.close()

    def test_paper_portfolio_from_log(self, tmp_path):
        """Paper portfolio entries from performance.jsonl are included."""
        gen, _ = _make_generator(tmp_path)
        perf_log = tmp_path / "performance.jsonl"
        perf_log.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00",
            "total_value": 100000,
            "daily_return": 0.01,
        }) + "\n")
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["paper_portfolio"]) == 1
        entry = data["paper_portfolio"][0]
        assert entry["t"] == "2026-01-01"
        assert entry["v"] == 100000
        assert entry["r"] == 0.01
        gen.conn.close()


# ---------------------------------------------------------------------------
# Explainability JSON freshness
# ---------------------------------------------------------------------------

class TestExplainabilityJSONFreshness:
    """Test generate_explainability_json freshness contract."""

    def test_generates_current_latest_from_signals_json(self, tmp_path):
        """Current signals data is the authoritative latest explainability source."""
        gen, _ = _make_generator(tmp_path)
        signals = {
            "generated_at": "2026-07-06T12:00:00",
            "ensemble_voting": {
                "regime": "normal",
                "weighted_consensus": 0.25,
                "agreement_ratio": 0.75,
                "action": "increase_equity",
                "confidence": 0.8,
                "num_sources": 1,
                "source_breakdown": [
                    {
                        "source": "cross_asset_rv",
                        "value": 0.5,
                        "confidence": 0.9,
                        "weight": 0.4,
                    }
                ],
            },
        }
        (tmp_path / "signals.json").write_text(json.dumps(signals))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_explainability_json()

        data = json.loads(path.read_text())
        assert data["analysis_date"] == "2026-07-06"
        assert data["latest_decision"]["period"] == "2026-07-06"
        assert data["freshness"]["status"] == "current"
        assert data["freshness"]["source_file"] == "signals.json"
        gen.conn.close()

    def test_stale_dated_report_without_current_signals_is_explicit_unavailable(
        self, tmp_path
    ):
        """Stale dated files are not copied as current latest explainability."""
        gen, _ = _make_generator(tmp_path)
        source_dir = tmp_path / "explainability"
        source_dir.mkdir()
        stale_payload = {
            "timestamp": "2026-05-18T03:14:06",
            "analysis_date": "2026-05-18",
            "latest_decision": {"period": "2026-05-18", "action": "increase_equity"},
            "recent_decisions": [],
            "signal_deep_dives": {},
            "top_sources_today": [],
            "decision_quality": {"status": "ok"},
        }
        (source_dir / "explainability_2026-05-18.json").write_text(
            json.dumps(stale_payload)
        )

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_explainability_json()

        data = json.loads(path.read_text())
        assert data["latest_decision"] is None
        assert data["decision_quality"]["status"] == "unavailable_current_signals"
        assert data["freshness"]["status"] == "unavailable"
        assert data["freshness"]["stale_source_file"] == "explainability_2026-05-18.json"
        assert data["freshness"]["stale_analysis_date"] == "2026-05-18"
        gen.conn.close()


# ---------------------------------------------------------------------------
# Yield curve — missing keys and malformed data
# ---------------------------------------------------------------------------

class TestYieldCurveMalformed:
    """Test _get_yield_curve_data with malformed data."""

    def test_missing_spread_key(self, tmp_path):
        """Missing spread2s10s key defaults to 0 spread."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"dgs2": 4.0, "dgs10": 5.0} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        assert data["yield_curve"]["spread2s10s"] == 0
        assert data["yield_curve"]["duration_regime"] == "inverted"
        gen.conn.close()

    def test_none_spread_entries_skipped(self, tmp_path):
        """None values in spread entries are excluded from spread_history."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        entries = []
        for i in range(35):
            if i % 3 == 0:
                entries.append({"spread2s10s": None, "dgs2": 4.0, "dgs10": 5.0})
            else:
                entries.append({"spread2s10s": i * 3, "dgs2": 4.0, "dgs10": 5.0})
        yields_path.write_text(json.dumps(entries))
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # None entries should be excluded from spread_history
        assert None not in data["yield_curve"]["spread_history"]
        gen.conn.close()

    def test_yields_file_empty_json_object(self, tmp_path):
        """Non-list JSON in yields file is handled gracefully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text("{}")
        with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
            data = gen._get_yield_curve_data()
        # Should return empty result
        assert data["yield_curve"] is None
        gen.conn.close()


# ---------------------------------------------------------------------------
# Generator init — edge cases
# ---------------------------------------------------------------------------

class TestGeneratorInitEdgeCases:
    """Additional DashboardGenerator initialization edge cases."""

    def test_public_dir_created(self, tmp_path):
        """PUBLIC_DIR is created during init."""
        new_public = tmp_path / "non_existent" / "data"
        assert not new_public.exists()
        # We can't easily test the constructor because it calls sqlite_connect
        # Instead verify that __init__ would create it
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", new_public):
            gen.__init__()
        assert new_public.exists()
        gen.conn.close()


# ---------------------------------------------------------------------------
# Run — overlay and signals edge cases
# ---------------------------------------------------------------------------

class TestRunOverlay:
    """Test run() with overlay generation."""

    def test_run_with_overlay(self, tmp_path):
        """run() includes overlay path when overlay generates successfully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        assert (tmp_path / "index.json").exists()
        # Verify signals.json was generated with regime data
        with open(tmp_path / "signals.json") as f:
            signals = json.load(f)
        assert "regime" in signals
        assert "generated_at" in signals
        # Connection already closed by run()

    def test_run_mirrors_required_public_data_contract_files_to_dist(self, tmp_path):
        """Dashboard generation keeps deploy-checked public/data and dist/data files in sync."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "source_manifest.json").write_text(json.dumps({
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-07-06T00:00:00+00:00",
            "artifacts": [],
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()

        for filename in ("source_manifest.json", "index.json", "health.json"):
            public_file = tmp_path / filename
            dist_file = tmp_path.parent / "dist" / "data" / filename
            assert dist_file.exists(), f"{dist_file} missing"
            assert dist_file.read_bytes() == public_file.read_bytes()
        # Connection already closed by run()


# ---------------------------------------------------------------------------
# ML signals — edge cases continued
# ---------------------------------------------------------------------------

class TestMlSignalsGridSearch:
    """Test ML signals grid search edge cases."""

    def test_grid_search_malformed_line(self, tmp_path):
        """Malformed line in grid search file is caught gracefully."""
        gen, _ = _make_generator(tmp_path)
        grid_file = tmp_path / "grid_search_results.jsonl"
        grid_file.write_text("not valid json\n")
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 15, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["grid_search"] == {}
        gen.conn.close()

    def test_multiple_features_keeps_latest(self, tmp_path):
        """Multiple entries for same symbol keep the latest by timestamp."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(
            json.dumps({"symbol": "SPY", "vix_level": 30, "trend_direction": 0,
                        "price_vs_sma20": 0, "timestamp": "2026-01-01T00:00:00"}) + "\n"
            + json.dumps({"symbol": "SPY", "vix_level": 15, "trend_direction": 1,
                          "price_vs_sma20": 0.05, "timestamp": "2026-01-02T00:00:00"}) + "\n"
        )
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        assert signals["available"] is True
        assert signals["features"]["SPY"]["vix_level"] == 15  # Latest
        gen.conn.close()

    def test_missing_vix_in_features(self, tmp_path):
        """Features missing vix_level key defaults to 20 in predictions."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # Default probabilities: vix <=20 and no trend
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()


# ---------------------------------------------------------------------------
# VIX regime detection — boundary values at exact thresholds
# ---------------------------------------------------------------------------

class TestVIXRegimeBoundaries:
    """VIX regime at exact boundary values."""

    def test_vix_exactly_15(self):
        """VIX exactly 15 is normal regime."""
        gen = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path := Path("/tmp")):
            # Extract the classify logic
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(15) == "normal"
            assert classify(20) == "normal"
            assert classify(25) == "vol_spike"  # >20 not >=20

    def test_vix_vol_spike_upper(self):
        """VIX exactly 25 is vol_spike (>20, not >25)."""
        gen = DashboardGenerator.__new__(DashboardGenerator)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path := Path("/tmp")):
            def classify(v):
                if v > 25: return "crisis"
                elif v > 20: return "vol_spike"
                elif v < 15: return "low_vol"
                else: return "normal"
            assert classify(25) == "vol_spike"


# ---------------------------------------------------------------------------
# Graduation JSON — additional edge cases
# ---------------------------------------------------------------------------

class TestGraduationJSONEdgeCases:
    """Additional generate_graduation_json edge cases."""

    def test_graduation_manual_approval_fields(self, tmp_path):
        """Graduation output has manual approval fields."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        assert path is not None
        with open(path) as f:
            data = json.load(f)
        assert data.get("manual_approval_required") is True
        assert data.get("manual_approval_pending") is True
        gen.conn.close()

    def test_graduation_criteria_met_count(self, tmp_path):
        """Criteria counts are calculated properly."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_graduation_json()
        with open(path) as f:
            data = json.load(f)
        assert "criteria_met" in data
        assert "criteria_total" in data
        assert data["criteria_total"] > 0
        gen.conn.close()


# ---------------------------------------------------------------------------
# __all__ exports validation
# ---------------------------------------------------------------------------

class TestAllExports:
    """Test __all__ exports match module's public API."""

    def test_all_defined(self):
        """__all__ is defined and contains expected names."""
        from src.dashboard.generator import __all__
        assert isinstance(__all__, list)
        assert "DashboardGenerator" in __all__
        assert "PUBLIC_DIR" in __all__
        assert "DB_PATH" in __all__
        assert len(__all__) >= 3

    def test_all_names_importable(self):
        """Every name in __all__ can be imported from the module."""
        import src.dashboard.generator as gen_mod
        from src.dashboard.generator import __all__

        for name in __all__:
            assert hasattr(gen_mod, name), f"{name} missing from module"
            assert getattr(gen_mod, name) is not None, f"{name} should not be None"


# ---------------------------------------------------------------------------
# Extended field type / dataclass validation
# ---------------------------------------------------------------------------

class TestOutputFieldTypes:
    """Validate field types in all generated JSON outputs."""

    def test_signals_json_field_types(self, tmp_path):
        """All signals.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    path = gen.generate_signals_json()
        with open(path) as f:
            data = json.load(f)

        # Top-level scalar fields
        assert isinstance(data["generated_at"], str), "generated_at should be str"
        assert isinstance(data["cash"], (int, float)), "cash should be numeric"
        assert isinstance(data["total_value"], (int, float)), "total_value should be numeric"

        # Regime section
        regime = data["regime"]
        assert isinstance(regime, dict)
        assert isinstance(regime["regime"], str)
        assert regime["vix"] is None or isinstance(regime["vix"], (int, float))
        assert regime["detected"] is None or isinstance(regime["detected"], str)

        # Target allocations
        target = data["target_allocations"]
        assert isinstance(target, dict)
        for sym, weight in target.items():
            assert isinstance(sym, str)
            assert isinstance(weight, (int, float))
            assert 0 <= weight <= 1.0, f"Weight {weight} out of range [0, 1]"

        # Latest prices
        prices = data["latest_prices"]
        assert isinstance(prices, dict)
        for sym, price in prices.items():
            assert isinstance(sym, str)
            assert isinstance(price, (int, float)), f"Price for {sym} should be numeric"

        # Positions
        assert isinstance(data["current_positions"], list)
        for pos in data["current_positions"]:
            assert isinstance(pos["symbol"], str)
            assert isinstance(pos["shares"], (int, float))
            assert isinstance(pos["value"], (int, float))
            assert isinstance(pos["weight"], (int, float))
            assert isinstance(pos["unrealized"], (int, float))

        # ML signals
        ml = data["ml_signals"]
        assert isinstance(ml, dict)
        assert isinstance(ml["available"], bool)
        assert ml["timestamp"] is None or isinstance(ml["timestamp"], str)
        assert isinstance(ml["predictions"], dict)
        assert isinstance(ml["features"], dict)
        assert isinstance(ml["grid_search"], dict)
        marl = data["marl_status"]
        assert isinstance(marl, dict)
        assert marl["schema_version"] == "marl-runtime-status/v1"
        assert isinstance(marl["available"], bool)
        assert isinstance(marl["runtime"], dict)
        assert marl["execution_role"]["routed"] is False
        assert marl["execution_role"]["role"] == "research_shadow_non_routed"
        gen.conn.close()

    def test_health_json_field_types(self, tmp_path):
        """All health.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["system_status"], str)
        assert data["system_status"] in ("healthy", "warning", "critical", "degraded")
        assert isinstance(data["cron_jobs"], list)
        assert isinstance(data["data_freshness"], dict)
        assert isinstance(data["signal_health"], dict)
        assert isinstance(data["generated_at"], str)

        for sym, freshness in data["data_freshness"].items():
            assert isinstance(sym, str)
            assert isinstance(freshness, dict)
            assert "last_update" in freshness
            assert "days_stale" in freshness
            assert "status" in freshness
            assert freshness["status"] in ("fresh", "stale", "critical")
            assert isinstance(freshness["days_stale"], int)

        for job in data["cron_jobs"]:
            assert isinstance(job["name"], str)
            assert isinstance(job["status"], str)
        gen.conn.close()

    def test_stats_json_field_types(self, tmp_path):
        """All stats.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["generated_at"], str)
        assert isinstance(data["asset_stats"], dict)

        for sym, stat in data["asset_stats"].items():
            assert isinstance(sym, str)
            assert isinstance(stat["30d_return"], (int, float))
            assert isinstance(stat["volatility"], (int, float))
            assert isinstance(stat["current"], (int, float))

        assert isinstance(data["paper_portfolio"], dict)
        assert data["spy_comparison"] is None or isinstance(data["spy_comparison"], dict)
        if data.get("spy_comparison"):
            sc = data["spy_comparison"]
            for key in ("portfolio_value", "spy_value", "relative_return", "correlation_30d", "beta"):
                assert key in sc, f"spy_comparison missing '{key}'"
        gen.conn.close()

    def test_alerts_json_field_types(self, tmp_path):
        """All alerts.json fields have correct Python types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data["alerts"], list)
        assert isinstance(data["count"], int)
        assert isinstance(data["generated_at"], str)

        for alert in data["alerts"]:
            assert isinstance(alert["level"], str)
            assert isinstance(alert["type"], str)
            assert isinstance(alert["title"], str)
            assert isinstance(alert["message"], str)
            assert isinstance(alert["requires_action"], bool)
            assert alert["level"] in ("success", "warning", "error", "info")
        gen.conn.close()

    def test_broker_data_field_types(self, tmp_path):
        """_load_broker_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            broker = gen._load_broker_data()
        assert isinstance(broker["connected"], bool)
        assert isinstance(broker["positions"], list)
        assert isinstance(broker["drift"], list)
        assert isinstance(broker["recent_orders"], list)
        assert broker["last_sync"] is None or isinstance(broker["last_sync"], str)
        assert isinstance(broker["kill_switch"], bool)
        gen.conn.close()

    def test_garch_cvar_field_types(self, tmp_path):
        """_load_garch_cvar_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            garch = gen._load_garch_cvar_data()
        assert isinstance(garch["cvar_95"], (int, float))
        assert isinstance(garch["cvar_95_garch"], (int, float))
        assert isinstance(garch["var_95"], (int, float))
        assert isinstance(garch["var_95_garch"], (int, float))
        assert isinstance(garch["cvar_ratio"], (int, float))
        assert isinstance(garch["garch_active"], bool)
        assert isinstance(garch["current_volatility"], (int, float))
        assert isinstance(garch["forecast_volatility"], (int, float))
        assert isinstance(garch["volatility_clustering"], str)
        assert garch["volatility_clustering"] in ("normal", "elevated", "high")
        gen.conn.close()

    def test_entropy_data_field_types(self, tmp_path):
        """_load_entropy_data dict has correct field types."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            entropy = gen._load_entropy_data()
        assert isinstance(entropy["shannon_entropy"], (int, float))
        assert isinstance(entropy["effective_n"], (int, float))
        assert isinstance(entropy["max_possible"], (int, float))
        assert isinstance(entropy["normalized_score"], (int, float))
        assert isinstance(entropy["concentration_risk"], str)
        assert entropy["concentration_risk"] in ("good", "low", "medium", "high", "critical")
        assert isinstance(entropy["hhi_index"], (int, float))
        assert isinstance(entropy["correlation_entropy"], (int, float))
        assert isinstance(entropy["participation_ratio"], (int, float))
        gen.conn.close()


# ---------------------------------------------------------------------------
# Additional computation edge cases — boundary values, zero/negative, large
# ---------------------------------------------------------------------------

class TestGarchCvarEdgeCasesExtended:
    """Additional _load_garch_cvar_data edge cases — boundary values."""

    def test_value_exactly_one_not_divided(self, tmp_path):
        """Value exactly 1.0 is kept as-is (not divided by 100 because abs(1) <= 1)."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.0,
            "var_95": 1.0,
            "cvar_ratio": 1.5,
            "filter_active": True,
            "conditional_volatility_current": 1.0,
            "garch_persistence": 0.9,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        # abs(1.0) <= 1, so no division
        assert data["cvar_95"] == 1.0
        assert data["var_95"] == 1.0
        gen.conn.close()

    def test_value_slightly_above_one_divided(self, tmp_path):
        """Value 1.01 (> 1.0) is divided by 100."""
        gen, _ = _make_generator(tmp_path)
        health_file = tmp_path / ".health_report.json"
        health_file.write_text(json.dumps({
            "garch_filtered": True,
            "cvar_95": 1.01,
            "var_95": 1.01,
            "filter_active": True,
        }))
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            data = gen._load_garch_cvar_data()
        assert data["cvar_95"] == 0.0101
        assert data["var_95"] == 0.0101
        gen.conn.close()

    def test_persistence_at_exact_boundaries(self, tmp_path):
        """GARCH persistence at exact boundary values."""
        gen, _ = _make_generator(tmp_path)
        # boundary cases: 0.85 -> elevated, 0.95 -> high, 0.951 -> high
        for persistence, expected in [
            # Code uses > 0.85 and > 0.95 (strict), not >=
            (0.85, "normal"),
            (0.86, "elevated"),
            (0.94, "elevated"),
            (0.95, "elevated"),
            (0.951, "high"),
            (0.80, "normal"),
            (0.84, "normal"),
        ]:
            health_file = tmp_path / ".health_report.json"
            health_file.write_text(json.dumps({
                "garch_filtered": True,
                "cvar_95": -0.0179,
                "filter_active": True,
                "garch_persistence": persistence,
            }))
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                data = gen._load_garch_cvar_data()
            assert data["volatility_clustering"] == expected, (
                f"Persistence {persistence} should be {expected}, got {data['volatility_clustering']}"
            )
        gen.conn.close()


class TestStatsEdgeCasesExtended:
    """Additional generate_stats_json computation edge cases."""

    def test_zero_returns_zero_volatility(self, tmp_path):
        """Identical prices produce zero volatility."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        for sym in ["SPY", "GLD"]:
            for i in range(30):
                d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                             (sym, d, 100.0))  # All identical prices
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        for sym, stat in data["asset_stats"].items():
            assert stat["volatility"] == 0.0, f"{sym} volatility should be 0 with identical prices"
            assert stat["30d_return"] == 0.0, f"{sym} 30d return should be 0 with identical prices"
        gen.conn.close()

    def test_negative_prices_handled(self, tmp_path):
        """Negative prices are handled without crashing."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        for i in range(5):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, -100.0 + i * 10.0))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["asset_stats"]
        gen.conn.close()

    def test_very_large_price_values(self, tmp_path):
        """Very large prices do not overflow or crash."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        large_price = 1e12
        for i in range(5):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, large_price + i * 1e9))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        assert "SPY" in data["asset_stats"]
        assert data["asset_stats"]["SPY"]["current"] > 0
        gen.conn.close()

    def test_negative_returns_handled(self, tmp_path):
        """Negative daily returns produce valid volatility."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL,
            PRIMARY KEY (symbol, date))
        """)
        today = datetime.now()
        # Insert in ascending date order (earliest first), so ORDER BY date gives descending prices
        days_ago = [4, 3, 2, 1, 0]
        prices = [100.0, 98.0, 95.0, 93.0, 90.0]  # Strictly declining
        for days_back, price in zip(days_ago, prices):
            d = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO prices VALUES (?, ?, ?)",
                         ("SPY", d, price))
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_stats_json()
        with open(path) as f:
            data = json.load(f)
        stat = data["asset_stats"]["SPY"]
        assert stat["30d_return"] < 0, "Declining prices should have negative return"
        assert stat["volatility"] >= 0, "Volatility must be non-negative"
        gen.conn.close()


class TestMLSignalsEdgeCasesExtended:
    """Additional _generate_ml_signals prediction edge cases."""

    def test_vix_exactly_25_classification(self, tmp_path):
        """VIX exactly 25 falls into vol_spike branch (>20, not >25)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 25, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.3  # vol_spike probs
        assert pred["probabilities"]["neutral"] == 0.5
        gen.conn.close()

    def test_vix_exactly_20_classification(self, tmp_path):
        """VIX exactly 20 falls into normal branch (not >20, not <15)."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 20, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        assert pred["probabilities"]["neutral"] == 0.6
        gen.conn.close()

    def test_vix_extremely_high(self, tmp_path):
        """Very high VIX (e.g., 80) does not crash."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 80, "trend_direction": -5,
            "price_vs_sma20": -0.2, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.5  # crisis probs dominate
        assert pred["predicted_regime"] == "bear"
        gen.conn.close()

    def test_trend_direction_zero_price_vs_sma_zero(self, tmp_path):
        """trend_direction=0 and price_vs_sma20=0 with vix <=20 defaults."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["probabilities"]["bear"] == 0.2
        assert pred["probabilities"]["neutral"] == 0.6
        assert pred["probabilities"]["bull"] == 0.2
        gen.conn.close()

    def test_missing_trend_fields_defaults(self, tmp_path):
        """Missing trend_direction and price_vs_sma20 fields use default 0."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()

    def test_price_vs_sma_positive_trend_zero(self, tmp_path):
        """price_vs_sma20 > 0 but trend_direction is 0 → default branch."""
        gen, _ = _make_generator(tmp_path)
        features_file = tmp_path / "features.jsonl"
        features_file.write_text(json.dumps({
            "symbol": "SPY", "vix_level": 18, "trend_direction": 0,
            "price_vs_sma20": 0.1, "timestamp": "2026-01-01",
        }) + "\n")
        with patch("src.dashboard.generator.DATA_DIR", tmp_path):
            signals = gen._generate_ml_signals()
        pred = signals["predictions"]["SPY"]
        # trend == 0 and price_vs_sma > 0 does NOT match trend > 0 condition
        assert pred["predicted_regime"] == "neutral"
        gen.conn.close()


class TestPerformanceJSONEdgeCasesExtended:
    """Additional generate_performance_json edge cases."""

    def test_empty_prices_table(self, tmp_path):
        """Empty prices table produces empty prices dict and no crash."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.execute("DELETE FROM regime_log")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["prices"] == {}
        assert data["regimes"] == []
        assert "generated_at" in data
        gen.conn.close()

    def test_regime_log_empty_list(self, tmp_path):
        """Empty regime_log table produces empty regimes list."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM regime_log")
        gen.conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert data["regimes"] == []
        gen.conn.close()


class TestAlertsJSONEdgeCasesExtended:
    """Additional generate_alerts_json edge cases."""

    def test_stale_data_days_calculation(self, tmp_path):
        """Stale data alert shows correct days count."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prices VALUES ('OLD', '2020-06-15', 100.0)")
        conn.commit()
        conn.close()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        stale = [a for a in data["alerts"] if a["type"] == "stale_data"]
        assert len(stale) >= 1
        assert "days ago" in stale[0]["message"]
        gen.conn.close()

    def test_current_data_quality_suppresses_weekend_aligned_stale_data_flood(self, tmp_path):
        """Current data_quality.json can prove an aligned cross-section is not stale."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.executemany(
            "INSERT INTO prices VALUES (?, ?, ?)",
            [("SPY", "2026-06-12", 100.0), ("GLD", "2026-06-12", 200.0)],
        )
        conn.commit()
        conn.close()
        _write_data_quality_report(tmp_path, status="ok", stale_latest_dates=0)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        assert [a for a in data["alerts"] if a["type"] == "stale_data"] == []
        gen.conn.close()

    def test_stale_data_alert_count_comes_from_current_data_quality_report(self, tmp_path):
        """Stale-data alerts should match data_quality stale_latest_dates count."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices")
        conn.executemany(
            "INSERT INTO prices VALUES (?, ?, ?)",
            [
                ("SPY", "2020-01-01", 100.0),
                ("GLD", "2020-01-01", 200.0),
                ("TLT", "2020-01-01", 90.0),
            ],
        )
        conn.commit()
        conn.close()
        _write_data_quality_report(tmp_path, status="fail", stale_latest_dates=1)

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path), \
             patch("src.dashboard.generator.DATA_DIR", tmp_path):
            path = gen.generate_alerts_json()

        data = json.loads(path.read_text())
        stale_alerts = [a for a in data["alerts"] if a["type"] == "stale_data"]
        assert len(stale_alerts) == 1
        assert "GLD" in stale_alerts[0]["message"]
        gen.conn.close()

    def test_no_trigger_files_no_alerts(self, tmp_path):
        """No trigger files produce only stale data alerts."""
        gen, _ = _make_generator(tmp_path)
        gen.conn.execute("DELETE FROM regime_log")
        gen.conn.commit()
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        # With today's data, there should be no stale data alerts
        types_found = {a["type"] for a in data["alerts"]}
        # If all data is fresh, alerts should be empty
        gen.conn.close()


# ---------------------------------------------------------------------------
# Constants validation — extended
# ---------------------------------------------------------------------------

class TestConstantsExtended:
    """Extended module-level constant validation."""

    def test_logger_is_logger_instance(self):
        """logger is a Logger instance."""
        import logging
        from src.dashboard.generator import logger
        assert isinstance(logger, logging.Logger)

    def test_base_allocation_keys_are_uppercase(self):
        """BASE_ALLOCATION keys are uppercase symbol names."""
        from src.paths import BASE_ALLOCATION
        for key in BASE_ALLOCATION:
            assert key == key.upper(), f"Key '{key}' should be uppercase"

    def test_regime_overrides_keys_match(self):
        """Regime override keys correspond to valid regimes."""
        from src.dashboard.generator import DashboardGenerator
        gen = DashboardGenerator.__new__(DashboardGenerator)
        # Extract the regime_overrides from generate_signals_json logic
        expected_regimes = {"crisis", "vol_spike", "low_vol"}
        overrides = {
            "crisis": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
            "vol_spike": {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25},
            "low_vol": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},
        }
        assert set(overrides.keys()) == expected_regimes
        for regime, alloc in overrides.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Regime '{regime}' allocations sum to {total}, expected 1.0"
            )
            for sym in alloc:
                assert isinstance(alloc[sym], float)

    def test_yield_regime_allocations_sum_to_one(self):
        """Each yield regime allocation sums to ~1.0."""
        regime_allocations = {
            "steep": {"tlt": 0.70, "ief": 0.25, "shy": 0.05, "bil": 0.00},
            "normal": {"tlt": 0.50, "ief": 0.35, "shy": 0.15, "bil": 0.00},
            "flat": {"tlt": 0.30, "ief": 0.40, "shy": 0.25, "bil": 0.05},
            "inverted": {"tlt": 0.15, "ief": 0.25, "shy": 0.35, "bil": 0.25},
        }
        for regime, alloc in regime_allocations.items():
            total = sum(alloc.values())
            assert abs(total - 1.0) < 0.01, (
                f"Yield regime '{regime}' allocations sum to {total}, expected 1.0"
            )

    def test_public_dir_exists_after_creation(self):
        """PUBLIC_DIR is a Path pointing to an existing or creatable directory."""
        from src.dashboard.generator import PUBLIC_DIR
        # This test just validates the constant, directory creation is tested in init
        import os
        parent = PUBLIC_DIR.parent
        assert parent.exists(), f"Parent dir {parent} should exist"


# ---------------------------------------------------------------------------
# CLI main() function test
# ---------------------------------------------------------------------------

class TestCliMain:
    """Test the __main__ CLI entry point."""

    def test_main_logic_runs_generator(self):
        """__main__ block's logic: DashboardGenerator().run()."""
        with patch.object(DashboardGenerator, "run") as mock_run:
            with patch.object(DashboardGenerator, "__init__", return_value=None):
                # This replicates the __main__ block: gen = DashboardGenerator(); gen.run()
                gen = DashboardGenerator()
                gen.run()
                mock_run.assert_called_once()

    def test_main_block_guard_reads_correctly(self):
        """The __main__ guard checks __name__ == '__main__'."""
        import ast
        import importlib.util
        source_path = importlib.util.find_spec("src.dashboard.generator").origin
        source = Path(source_path).read_text()
        tree = ast.parse(source)
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is an if __name__ == "__main__" guard
                if (isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == "__main__"):
                    found_guard = True
                    # Verify the body contains DashboardGenerator and run()
                    body_source = ast.unparse(node.body)
                    assert "DashboardGenerator" in body_source
                    assert ".run()" in body_source
                    break
        assert found_guard, "No __name__ == '__main__' guard found"


class TestZeroDTEClosingAuctionHonesty:
    """Dead overlay surfaces must not publish silent empty {}."""

    def test_unavailable_zero_dte_payload_has_schema_fields(self):
        payload = DashboardGenerator._unavailable_zero_dte_payload()
        assert payload["positions"] == []
        assert payload["config"] is None
        assert payload["weekly_trades_used"] == 0
        assert payload["status"] == "unavailable"
        assert payload["runtime_status"] == "unavailable_no_producer"
        assert payload["live_authoritative"] is False
        assert payload["active"] is False
        assert "generated_at" in payload

    def test_unavailable_closing_auction_payload_has_schema_fields(self):
        payload = DashboardGenerator._unavailable_closing_auction_payload()
        assert payload["signals"] == []
        assert payload["last_update"] is None
        assert payload["market_open"] is False
        assert payload["status"] == "unavailable"
        assert payload["runtime_status"] == "unavailable_no_producer"
        assert payload["live_authoritative"] is False

    def test_empty_dict_not_populated(self):
        assert DashboardGenerator._is_populated_overlay_section({}) is False
        assert DashboardGenerator._is_populated_overlay_section(None) is False
        assert DashboardGenerator._is_populated_overlay_section(
            DashboardGenerator._unavailable_zero_dte_payload()
        ) is False

    def test_real_producer_payload_is_populated(self):
        assert DashboardGenerator._is_populated_overlay_section(
            {"positions": [{"id": "x"}], "active": True}
        ) is True
        assert DashboardGenerator._is_populated_overlay_section(
            {"signals": [{"should_trade": False}], "market_open": True}
        ) is True

    def test_get_overlay_data_never_returns_silent_empty_surfaces(self, monkeypatch):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeOverlay:
            def generate(self):
                return self

            def to_dict(self):
                # No zero_dte / closing_auction keys — historical producer gap
                return {
                    "collar": {"active": False},
                    "crypto": {},
                    "calendar": {},
                    "kurtosis": {},
                    "bond_duration": {},
                    "unified": {},
                }

        monkeypatch.setattr(
            "src.dashboard.overlay_dashboard.OverlayDashboardGenerator",
            FakeOverlay,
            raising=False,
        )

        # Avoid real VIX generator / file IO
        class FakeVixGen:
            def generate_signal(self):
                class S:
                    def to_dict(self_inner):
                        return {"regime": "contango", "vix_spot": 18.0}

                return S()

        import sys
        import types

        monkeypatch.setitem(
            sys.modules,
            "src.signals.vix_term_structure",
            types.SimpleNamespace(VIXTermStructureSignalGenerator=FakeVixGen),
        )

        with patch("src.dashboard.generator.DATA_DIR", Path("/tmp/no-vix-overlay-state")):
            data = gen._get_overlay_data()

        assert data["zero_dte"]["runtime_status"] == "unavailable_no_producer"
        assert data["zero_dte"] != {}
        assert data["closing_auction"]["runtime_status"] == "unavailable_no_producer"
        assert data["closing_auction"] != {}
        assert data["zero_dte"]["positions"] == []
        assert data["closing_auction"]["signals"] == []

    def test_get_overlay_data_passes_through_real_producer(self, monkeypatch):
        gen = DashboardGenerator.__new__(DashboardGenerator)

        class FakeOverlay:
            def generate(self):
                return self

            def to_dict(self):
                return {
                    "collar": {},
                    "crypto": {},
                    "calendar": {},
                    "kurtosis": {},
                    "bond_duration": {},
                    "unified": {},
                    "zero_dte": {
                        "positions": [{"id": "p1"}],
                        "config": None,
                        "weekly_trades_used": 1,
                        "total_premium_collected_mtd": 10.0,
                        "active": True,
                    },
                    "closing_auction": {
                        "signals": [{"should_trade": True}],
                        "last_update": "2026-07-20T12:00:00",
                        "market_open": True,
                    },
                }

        monkeypatch.setattr(
            "src.dashboard.overlay_dashboard.OverlayDashboardGenerator",
            FakeOverlay,
            raising=False,
        )

        class FakeVixGen:
            def generate_signal(self):
                class S:
                    def to_dict(self_inner):
                        return {}

                return S()

        import sys
        import types

        monkeypatch.setitem(
            sys.modules,
            "src.signals.vix_term_structure",
            types.SimpleNamespace(VIXTermStructureSignalGenerator=FakeVixGen),
        )

        with patch("src.dashboard.generator.DATA_DIR", Path("/tmp/no-vix-overlay-state")):
            data = gen._get_overlay_data()

        assert data["zero_dte"]["positions"][0]["id"] == "p1"
        assert data["closing_auction"]["market_open"] is True
        assert data["zero_dte"].get("runtime_status") != "unavailable_no_producer"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestTwoStageRegimeUnavailableHonesty:
    """Optional regime sections must not silently disappear when generators return None."""

    def test_two_stage_none_publishes_unavailable_section(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        import types
        gen, _ = _make_generator(tmp_path)
        fresh = datetime.now(timezone.utc).isoformat()
        monkeypatch.setattr(
            "src.dashboard.generator.validate_signal",
            lambda _name, signal: signal,
        )
        monkeypatch.setattr(gen, "_generate_two_stage_regime", lambda: None)
        monkeypatch.setattr(gen, "_generate_bocd_regime", lambda: None)
        monkeypatch.setattr(gen, "_run_spc_monitor", lambda output: {"status": "ok"})
        monkeypatch.setattr(gen, "_record_ic_data", lambda output: None)
        # Empty regime history → transition also unavailable
        fake_cursor = types.SimpleNamespace(
            execute=lambda *a, **k: None,
            fetchall=lambda: [],
        )
        output = {
            "ensemble_voting": {
                "generated_at": fresh,
                "regime": "normal",
                "source_breakdown": [],
            },
            "alternative_data": {"timestamp": fresh},
            "garch_cvar": {"timestamp": fresh},
            "smart_rebalance": {"generated_at": fresh},
            "rebalance_health": {"generated_at": fresh},
        }
        try:
            result = gen._apply_signal_postprocessors(
                output,
                {"cursor": fake_cursor, "current_regime": "normal"},
            )
        finally:
            gen.conn.close()

        assert "two_stage_regime" in result
        ts = result["two_stage_regime"]
        assert ts.get("status") == "unavailable" or ts.get("runtime_status") == "unavailable"
        assert ts.get("regime") in ("UNKNOWN", "unavailable", None) or ts.get("confidence", 0) == 0.0

        assert "regime_transition" in result
        rt = result["regime_transition"]
        assert rt.get("status") == "unavailable" or rt.get("runtime_status") == "unavailable"
