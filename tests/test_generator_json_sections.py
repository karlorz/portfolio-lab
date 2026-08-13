#!/usr/bin/env python3
"""
Generator JSON-section tests — all generate_*_json section classes
(TEST-GENERATOR-SPLIT s6, 2026-08-12).

Moved verbatim from tests/test_generator.py (30 JSON-section classes per the
15:35Z corrected move-set table) — no tests renamed or weakened. Shared
helpers live in tests/helpers.py (plain module; the autouse fixture below is
duplicated verbatim per split file — never move it to conftest.py, it would
pollute the full ~15k-test suite).
"""
import inspect
import json
import sqlite3
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.generator import DashboardGenerator
from tests.helpers import (
    _make_generator,
    _write_data_quality_report,
    _write_ok_source_manifest,
)


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
                    "missing_symbols": ["BTC-USD"],
                    "reason": "missing_or_all_nan_symbol",
                },
            },
            missing_symbols=["BTC-USD"],
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
        assert data["unavailable_pairs"]["gld_btc"]["missing_symbols"] == ["BTC-USD"]
        assert data["missing_symbols"] == ["BTC-USD"]
        gen.conn.close()

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

    def test_alerting_block_disclosed(self, tmp_path, monkeypatch):
        """alerts.json carries webhook disclosure; URL itself never present."""
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("ALERT_WEBHOOK_URL_FILE", raising=False)
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_alerts_json()
        with open(path) as f:
            data = json.load(f)
        assert data["alerting"] == {
            "webhook_configured": False,
            "webhook_source": "none",
        }
        payload_text = json.dumps(data)
        assert "https://" not in payload_text  # secret safety
        gen.conn.close()
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
        """Freshness status is session-relative: a Friday bar on a Sunday as-of
        is zero missed sessions even though calendar age is two days."""
        db_path = tmp_path / "market.db"
        provider_latest = "2026-08-07"  # fixed Friday
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

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                # Sunday 2026-08-09 16:00 ET: Friday bars are fully fresh.
                # Mimic real datetime.now(): naive without tz, aware with tz.
                base = cls(2026, 8, 9, 20, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch(
                    "src.dashboard.data_freshness_section.datetime", _FrozenDatetime
                ):
                    path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert {item["status"] for item in data["data_freshness"].values()} == {"fresh"}
        assert data["data_freshness"]["SPY"]["days_stale"] == 2
        assert data["data_freshness"]["SPY"]["market_lag_days"] == 0
        assert data["data_freshness"]["SPY"]["missed_market_sessions"] == 0
        assert data["data_freshness"]["SPY"]["latest_available_market_date"] == provider_latest
        gen.conn.close()

    def test_symbol_lagging_provider_latest_date_is_critical(self, tmp_path):
        """A symbol genuinely behind the provider's latest completed session
        is flagged critical in missed-session units."""
        db_path = tmp_path / "market.db"
        provider_latest = "2026-08-07"  # fixed Friday
        lagging_date = "2026-08-02"  # Sunday: bar's update session is Fri 2026-07-31
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, PRIMARY KEY (symbol, date))"
        )
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", provider_latest, 100.0))
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("GLD", lagging_date, 100.0))
        conn.commit()
        conn.close()
        gen = DashboardGenerator.__new__(DashboardGenerator)
        gen.conn = sqlite3.connect(str(db_path))
        gen.conn.row_factory = sqlite3.Row
        (tmp_path / "cron_status.json").write_text('{"jobs": []}')

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                # Sunday 2026-08-09 16:00 ET.
                base = cls(2026, 8, 9, 20, 0)
                return base.replace(tzinfo=tz) if tz is not None else base

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch(
                    "src.dashboard.data_freshness_section.datetime", _FrozenDatetime
                ):
                    path = gen.generate_health_json()

        with open(path) as f:
            data = json.load(f)
        assert data["data_freshness"]["SPY"]["status"] == "fresh"
        assert data["data_freshness"]["GLD"]["status"] == "critical"
        assert data["data_freshness"]["GLD"]["market_lag_days"] == 5
        assert data["data_freshness"]["GLD"]["missed_market_sessions"] == 5
        gen.conn.close()

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

class TestSignalsJSONEdgeCases:
    """Test generate_signals_json edge cases."""

    def test_signal_section_helpers_delegate_to_builder(self, tmp_path):
        """The generator keeps orchestration while its collaborator owns sections."""
        gen, _ = _make_generator(tmp_path)
        builder = MagicMock()
        context = {"current_regime": "NORMAL"}
        base = {"target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}
        optional = {**base, "rebalance_health": {}}
        final = {**optional, "health": {"status": "ok"}}
        builder.build_base_sections.return_value = base
        builder.build_optional_sections.return_value = optional
        builder.apply_postprocessors.return_value = final

        with patch.object(gen, "_get_signal_section_builder", return_value=builder):
            assert gen._build_base_signal_sections(context) is base
            assert gen._build_optional_signal_sections(base, context) is optional
            assert gen._apply_signal_postprocessors(optional, context) is final

        builder.build_base_sections.assert_called_once_with(context)
        builder.build_optional_sections.assert_called_once_with(base, context)
        builder.apply_postprocessors.assert_called_once_with(optional, context)
        gen.conn.close()

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
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}

        def add_nested_timestamp(output, context):
            nested_ts = datetime.fromisoformat(output["generated_at"]) + timedelta(seconds=1)
            enriched = dict(output)
            enriched["regime_transition"] = {"timestamp": nested_ts.isoformat()}
            return enriched

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", data_dir):
                with patch("src.dashboard.generator.datetime", FakeDateTime):
                    with patch.object(gen, "_load_signal_generation_context", return_value={}):
                        with patch.object(
                            gen,
                            "_build_base_signal_sections",
                            return_value={"target_allocations": champion},
                        ):
                            with patch.object(
                                gen,
                                "_build_optional_signal_sections",
                                side_effect=add_nested_timestamp,
                            ):
                                with patch.object(
                                    gen,
                                    "_apply_signal_postprocessors",
                                    side_effect=lambda output, context: output,
                                ):
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
        assert data["target_allocations"] == champion

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

class TestHealthJSONEdgeCases:
    """Test generate_health_json edge cases."""

    def test_health_projects_bounded_ic_decay_summary(self, tmp_path, monkeypatch):
        """Public health carries bounded quality evidence without raw history."""
        gen, _ = _make_generator(tmp_path)
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "ic_decay_critical_minimum.json").read_text(
                encoding="utf-8"
            )
        )
        raw_report = {
            "status": fixture["status"],
            "signals": fixture["signals"],
            "pending_predictions": fixture["staged_pending_predictions"]["count"],
            "pending_scope": fixture["staged_pending_predictions"]["scope"],
            "staged_prediction_names": fixture["staged_pending_predictions"]["signal_names"],
            "staged_date": fixture["staged_pending_predictions"]["date"],
            "pending_rows": fixture["historical_unlabeled_backlog"]["rows"],
            "pending_rows_scope": fixture["historical_unlabeled_backlog"]["scope"],
            "pending_dates": fixture["historical_unlabeled_backlog"]["dates"],
        }
        monkeypatch.setattr(
            "src.monitor.ic_decay_monitor.compute_ic_decay_report",
            lambda: raw_report,
        )

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_health_json()

        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["ic_decay_summary"]
        assert summary["status"] == "critical"
        assert summary["critical_signals"] == ["ensemble_consensus", "ensemble_duration"]
        assert summary["staged_pending_predictions"] == 7
        assert summary["historical_unlabeled_rows"] == 1663
        assert "historical_database" not in json.dumps(summary)
        gen.conn.close()

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

    def test_stale_data_warning_threshold(self, tmp_path, monkeypatch):
        """stale_count > 5 with market_lag in the stale band (1d) → warning.

        Symbols must not be ``critical`` freshness (market_lag > 3): any
        critical data_freshness child rolls the artifact SLO to critical and
        elevates system_status. Lag 1 calendar day behind the freshest row
        and pin the freshness clock to 22:00Z on the freshest date so the
        session-aware classifier deterministically reports exactly 1 missed
        session (stale, not critical) at any wall-clock time (GAP-1).
        """
        gen, db_path = _make_generator(tmp_path)
        # _make_generator seeds SPY/GLD/TLT/QQQ through today; lag stale rows
        # 1 calendar day behind that cross-section (stale, not critical).
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT MAX(date) FROM prices")
        latest = cursor.fetchone()[0]
        latest_dt = datetime.strptime(latest, "%Y-%m-%d")
        stale_date = (latest_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        for i in range(6):
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?)",
                (f"STALE{i}", stale_date, 100.0),
            )
        conn.commit()
        conn.close()

        class FakeFreshnessClock(datetime):
            """Pin freshness evaluation to 22:00Z on the freshest bar's date."""

            _value = datetime(
                latest_dt.year,
                latest_dt.month,
                latest_dt.day,
                22,
                0,
                tzinfo=timezone.utc,
            )

            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return cls._value.replace(tzinfo=None)
                return cls._value.astimezone(tz)

        monkeypatch.setattr(
            "src.dashboard.data_freshness_section.datetime", FakeFreshnessClock
        )
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
        # Honesty: keep raw implausible Sharpe (>3.0) — never coerce to 0.0.
        # Gate still fails (Batch AE / graduation_checklist._sharpe_plausibility).
        assert criteria["min_sharpe"]["value"] == 3.38
        assert criteria["min_sharpe"]["passed"] is False
        assert "implausible" in (criteria["min_sharpe"]["description"] or "").lower()
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

    def test_vix_fetch_failure_passes_none(self, tmp_path):
        """When vix_level is None (no VIX data), pass None — never fake 0.0."""
        gen, db_path = _make_generator(tmp_path)
        mock_signals = {"SPY": {"momentum": 0.5}, "vix": None, "vix_source": "unavailable"}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals(vix_level=None)
        assert result == mock_signals
        args, kwargs = mock_gen.call_args
        # Positional or keyword — value must be None, not 0
        passed_vix = kwargs.get("vix") if "vix" in kwargs else (args[1] if len(args) > 1 else "MISSING")
        assert passed_vix is None
        gen.conn.close()

    def test_resolve_hedge_vix_falls_back_to_term_structure(self):
        """market.db missing ^VIX → use vix_term_structure.vix_spot for sector/hedge."""
        assert DashboardGenerator._resolve_hedge_vix_level(None, {"vix_spot": 16.76}) == 16.76
        assert DashboardGenerator._resolve_hedge_vix_level(18.5, {"vix_spot": 16.76}) == 18.5
        assert DashboardGenerator._resolve_hedge_vix_level(None, {}) is None
        assert DashboardGenerator._resolve_hedge_vix_level(None, None) is None

    def test_none_when_no_vix_row(self, tmp_path):
        """No VIX row still calls generate with vix=None (honest unknown)."""
        gen, db_path = _make_generator(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM prices WHERE symbol = '^VIX'")
        conn.commit()
        conn.close()
        mock_signals = {"SPY": {"momentum": 0.5}, "vix": None}
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.strategy.sector_momentum_calc.generate_sector_signals",
                          return_value=mock_signals) as mock_gen:
                    result = gen._generate_sector_momentum_signals()
        assert result == mock_signals
        args, kwargs = mock_gen.call_args
        passed_vix = kwargs.get("vix") if "vix" in kwargs else (args[1] if len(args) > 1 else "MISSING")
        assert passed_vix is None
        gen.conn.close()

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

    def test_crisis_regime_keeps_champion_target_alloc_and_advisory_candidate(
        self, tmp_path
    ):
        """Current hard rule: crisis regime does not rewrite live target_allocations."""
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
        assert data["target_allocations"] == {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        diagnostic = data["regime_allocation_diagnostic"]
        assert diagnostic["routed"] is False
        assert diagnostic["live_authoritative"] is False
        assert diagnostic["candidate_target_allocations"] == {
            "SPY": 0.20,
            "GLD": 0.50,
            "TLT": 0.30,
        }
        gen.conn.close()

class TestRegimeTargetAllocationParity:
    """Dashboard target allocations preserve current champion authority."""

    @pytest.mark.parametrize(
            ("expected_regime", "vix_level", "trend_regime"),
            [
                ("crisis", 30.0, "normal"),
                ("vol_spike", 22.5, "normal"),
                ("high_vol", 16.0, "high_vol"),
                ("recovery", 16.0, "recovery"),
            ],
        )
    def test_target_allocations_stay_champion_when_regime_changes(
        self, tmp_path, monkeypatch, expected_regime, vix_level, trend_regime
    ):
        """VIX/trend regimes remain advisory under the current hard rule."""
        from src.paths import BASE_ALLOCATION

        monkeypatch.delenv("REGIME_ALLOC_ENABLED", raising=False)
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
        assert context["target_alloc"] == BASE_ALLOCATION

    def test_default_dashboard_target_allocations_stay_champion_in_vol_spike(
        self, tmp_path, monkeypatch
    ):
        """Current hard rule: VIX regimes do not rewrite live target_allocations."""
        from src.paths import BASE_ALLOCATION
        from src.dashboard.generator import DashboardGenerator

        monkeypatch.delenv("REGIME_ALLOC_ENABLED", raising=False)

        gen = DashboardGenerator.__new__(DashboardGenerator)
        target = gen._resolve_live_target_allocations_for_regime("vol_spike")

        assert target == BASE_ALLOCATION

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

    @pytest.mark.allow_live_signal_health
    def test_signal_health_error_fallback(self, tmp_path):
        """Signal health has error fallback when tracker unavailable."""
        # Needs real build_signal_health_section path (opt out of isolation fixture).
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

class TestPerformanceJSONRegime:
    """Test regime data in generate_performance_json."""

    def test_regime_data_included(self, tmp_path):
        """Regime data from canonical JSONL is included in performance output."""
        gen, _ = _make_generator(tmp_path)
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        (tmp_path / "regime_log.json").write_text(
            json.dumps(
                {
                    "regime": "NORMAL",
                    "vix_level": 15.0,
                    "detected_at": f"{recent}T12:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                path = gen.generate_performance_json()
        with open(path) as f:
            data = json.load(f)
        assert len(data["regimes"]) >= 1
        assert data["regimes"][0]["r"] == "normal"
        assert data["regimes"][0] == {"d": recent, "r": "normal", "v": 15.0}
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
        # If all data is fresh, alerts should be empty
        assert data["alerts"] == []
        gen.conn.close()

