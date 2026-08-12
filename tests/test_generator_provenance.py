#!/usr/bin/env python3
"""
Generator provenance tests — FRED macro, turnover-validator artifact, and
IC label-lifecycle classes (TEST-GENERATOR-SPLIT s1, 2026-08-12).

Moved verbatim from tests/test_generator.py (TestFredMacroProvenance,
TestTurnoverValidatorPublicArtifact, TestPredictionLabelLifecycle) — no
tests renamed or weakened. Shared helpers live in tests/helpers.py (plain
module; the autouse fixture below is duplicated verbatim per split file —
never move it to conftest.py, it would pollute the full ~15k-test suite).
"""
import json
import sqlite3
import sys
import types
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator, PUBLIC_DIR
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
        # Per-signal staged shape (Task 2B): entries keyed by identity carry the
        # latest market-data prediction date, not the wall-clock run date.
        staged = state["__staged_v2__"]
        equity = next(e for e in staged if e["signal"] == "ensemble_equity")
        assert equity["prediction_date"] == "2026-07-02"
        assert equity["metadata"]["prediction_date"] == "2026-07-02"
        assert equity["metadata"]["prediction_field"] == "ensemble_voting.equity_bias"
        assert equity["metadata"]["prediction_transform"] == "identity"

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
        # Legacy single-slot staging migrates to per-signal identities; the
        # unresolved old cohort survives alongside the new cohort.
        staged = {e["signal"]: e for e in state["__staged_v2__"]}
        assert staged["old_signal"]["prediction"] == 0.25
        assert staged["old_signal"]["prediction_date"] == "2026-07-02"
        assert staged["ensemble_equity"]["prediction"] == 0.9

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
        resolved_meta = state["__observation_metadata__"]["old_signal"][0]
        assert resolved_meta["prediction_date"] == "2026-07-02"
        assert resolved_meta["realized_start_date"] == "2026-07-03"
        assert resolved_meta["resolved_date"] == "2026-07-03"
        assert resolved_meta["target_asset"] == "SPY"
        assert resolved_meta["realized_horizon_sessions"] == 1
        # New cohort staged at the latest market-data date.
        staged = {e["signal"]: e for e in state["__staged_v2__"]}
        assert staged["ensemble_equity"]["prediction_date"] == "2026-07-03"
        assert staged["ensemble_equity"]["prediction"] == pytest.approx(-0.2)

    def test_record_ic_data_counts_realized_market_sessions(self, tmp_path, monkeypatch):
        """A multi-session label records the actual SPY session span."""
        from src.monitor import ic_decay_monitor

        state_path = tmp_path / "ic_monitor_state.json"
        monkeypatch.setattr(ic_decay_monitor, "IC_STATE_PATH", state_path)
        state_path.write_text(json.dumps({
            "__staged__": {
                "date": "2026-07-02",
                "predictions": {"ensemble_equity": 0.5},
                "prediction_metadata": {
                    "ensemble_equity": {
                        "prediction_date": "2026-07-02",
                        "prediction_field": "ensemble_voting.equity_bias",
                        "prediction_transform": "identity",
                        "intended_horizon_sessions": 1,
                        "metric_axis": "time_series_rank_correlation",
                        "metric_kind": "correlation",
                        "contract_version": "ic-observation-metadata/v2",
                    },
                },
            },
        }))
        gen = self._make_spy_generator(
            tmp_path,
            [
                ("2026-07-02", 100.0),
                ("2026-07-03", 101.0),
                ("2026-07-06", 103.0),
            ],
        )

        try:
            gen._record_ic_data(self._ic_output(value=-0.2))
        finally:
            gen.conn.close()

        state = json.loads(state_path.read_text())
        metadata = state["__observation_metadata__"]["ensemble_equity"][0]
        assert metadata["realized_start_date"] == "2026-07-03"
        assert metadata["resolved_date"] == "2026-07-06"
        assert metadata["realized_horizon_sessions"] == 2
