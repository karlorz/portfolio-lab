#!/usr/bin/env python3
"""
Generator staleness tests — signal staleness normalization class
(TEST-GENERATOR-SPLIT s2, 2026-08-12).

Moved verbatim from tests/test_generator.py (TestSignalStalenessNormalization)
— no tests renamed or weakened. Shared helpers live in tests/helpers.py (plain
module; the autouse fixture below is duplicated verbatim per split file —
never move it to conftest.py, it would pollute the full ~15k-test suite).
"""
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator
from tests.helpers import _make_generator


@pytest.fixture(autouse=True)
def _isolate_live_ensemble_and_ic_health(request, monkeypatch):
    """Keep generator tests off live SignalHealthTracker.compute_ic / compute_vote.

    gen.run() and generate_health_json() otherwise call get_health_report() which
    runs hundreds of Spearman IC queries (~15–35s each on lab hosts). That was
    stalling make-test around the TestRun / health-json region (~44%).

    Opt out with @pytest.mark.allow_live_signal_health when a test intentionally
    exercises the real tracker (or already patches get_health_report itself).
    """
    if request.node.get_closest_marker("allow_live_signal_health"):
        yield
        return

    from src.strategy.ensemble_voter import EnsembleVote, Regime

    def _fake_vote(self, *args, **kwargs):
        return EnsembleVote(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=1,
            weighted_consensus=0.1,
            agreement_ratio=0.5,
            equity_bias=0.1,
            duration_bias=0.0,
            gold_bias=0.0,
            action="neutral",
            confidence=0.5,
            reasoning="test-isolation",
            source_votes=[],
        )

    def _fake_bl_views(self, *args, **kwargs):
        from src.strategy.black_litterman_mapper import map_biases_to_views

        views = map_biases_to_views(
            0.1, 0.0, 0.0, health_scores=None, tau=0.15, prior="equal"
        )
        return {
            "views": views,
            "tau": 0.15,
            "prior": "equal",
            "health_scores_used": {},
            "equity_bias": 0.1,
            "duration_bias": 0.0,
            "gold_bias": 0.0,
        }

    def _fake_signal_health_section(**kwargs):
        return {
            "status": "ok",
            "sources": {},
            "summary": {"healthy": 0, "warning": 0, "critical": 0, "total": 0},
            "label_resolve": {"resolved": 0, "pending": 0, "skipped": True},
        }

    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.compute_vote",
        _fake_vote,
        raising=False,
    )
    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.get_bl_views",
        _fake_bl_views,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.signal_health_section.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.generator.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    yield
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
        signals_body = {
            "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            "generated_at": "2026-07-12T00:00:00+00:00",
            "alternative_data": {"timestamp": "2026-07-12T00:00:00+00:00"},
        }
        (public_dir / "signals.json").write_text(json.dumps(signals_body))
        (data_dir / "signals.json").write_text(json.dumps(signals_body))

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

