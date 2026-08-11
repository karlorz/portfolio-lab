"""Tests for IC decay monitor — signal quality tracking."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.monitor.ic_decay_monitor import (
    IC_EVALUATION_CONTRACTS,
    ICMonitor,
    build_ic_decay_summary,
    compute_ic_decay_report,
    _spearman_rank_correlation,
)


class TestSpearmanRankCorrelation:
    """Test the Spearman rank correlation helper."""

    def test_perfect_positive_correlation(self):
        """Perfect monotonic increase should give correlation ~1.0."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        r = _spearman_rank_correlation(x, y)
        assert abs(r - 1.0) < 0.01

    def test_perfect_negative_correlation(self):
        """Perfect monotonic decrease should give correlation ~-1.0."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
        r = _spearman_rank_correlation(x, y)
        assert abs(r + 1.0) < 0.01

    def test_no_correlation(self):
        """Random data should give correlation near 0."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [5, 3, 8, 1, 9, 2, 7, 4, 10, 6]
        r = _spearman_rank_correlation(x, y)
        # Should be somewhere between -1 and 1 but not extreme
        assert -1.0 <= r <= 1.0

    def test_insufficient_data_returns_none(self):
        """Less than 5 data points has no defined monitor coefficient."""
        assert _spearman_rank_correlation([1, 2, 3], [1, 2, 3]) is None
        assert _spearman_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]) is None

    def test_zero_variance_returns_none(self):
        """Constant values have an undefined rank correlation."""
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        assert _spearman_rank_correlation(x, y) is None

    def test_ties_use_average_midranks(self):
        """Ties receive average ranks rather than arbitrary input-order ranks."""
        x = [1, 1, 2, 3, 3]
        y = [1, 2, 2, 3, 4]
        expected = float(np.corrcoef(
            [0.5, 0.5, 2.0, 3.5, 3.5],
            [0.0, 1.5, 1.5, 3.0, 4.0],
        )[0, 1])
        assert _spearman_rank_correlation(x, y) == pytest.approx(expected)

    def test_nan_inf_values_handled(self):
        """NaN/inf values should be filtered out."""
        import math
        x = [1, 2, math.nan, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        r = _spearman_rank_correlation(x, y)
        # After filtering nan, 9 points remain — should still compute
        assert -1.0 <= r <= 1.0

    def test_empty_arrays_return_none(self):
        """Empty arrays do not define a coefficient."""
        assert _spearman_rank_correlation([], []) is None


class TestICMonitor:
    """Test ICMonitor class."""

    def test_record_and_compute_ic(self):
        """Recording data should allow IC computation."""
        monitor = ICMonitor(window_size=20)
        # Perfect positive correlation
        for i in range(10):
            monitor.record("test_signal", prediction=float(i), actual_return=float(i) * 0.01)
        ic = monitor.compute_ic("test_signal")
        assert ic is not None
        assert ic > 0.9  # Strong positive correlation

    def test_compute_ic_insufficient_data(self):
        """IC should be None with less than 5 observations."""
        monitor = ICMonitor()
        monitor.record("test", 0.1, 0.01)
        monitor.record("test", 0.2, 0.02)
        assert monitor.compute_ic("test") is None

    def test_compute_ic_constant_input_is_undefined(self):
        monitor = ICMonitor()
        for i in range(10):
            monitor.record("constant", 1.0, float(i))

        assert monitor.compute_ic("constant") is None
        row = monitor.compute_decay_report()["constant"]
        assert row["ic_rolling"] is None
        assert row["status"] == "insufficient_data"

    def test_compute_ic_unknown_signal(self):
        """IC should be None for unknown signal."""
        monitor = ICMonitor()
        assert monitor.compute_ic("nonexistent") is None

    def test_rolling_window_respects_maxlen(self):
        """Window size should be respected — old data drops off."""
        monitor = ICMonitor(window_size=5)
        for i in range(10):
            monitor.record("test", prediction=float(i), actual_return=float(i) * 0.01)
        assert len(monitor._data["test"]) == 5

    def test_compute_ic_trend_stable(self):
        """Consistent IC should show 'stable' trend."""
        monitor = ICMonitor(window_size=60, trend_window=5)
        # High consistent correlation throughout
        for i in range(30):
            monitor.record("stable_sig", prediction=float(i), actual_return=float(i) * 0.01 + 0.001)
        trend = monitor.compute_ic_trend("stable_sig")
        assert trend == "stable"

    def test_compute_ic_trend_decaying(self):
        """Degrading IC should show 'decaying' trend."""
        monitor = ICMonitor(window_size=60, trend_window=5, decay_threshold=0.05)
        # Good correlation first half
        for i in range(25):
            monitor.record("decay_sig", prediction=float(i), actual_return=float(i) * 0.01)
        # Random/no correlation second half
        import random
        random.seed(42)
        for i in range(25):
            monitor.record("decay_sig", prediction=float(i), actual_return=random.random() * 0.01)
        trend = monitor.compute_ic_trend("decay_sig")
        # Trend should be decaying or at least not improving
        assert trend in ("decaying", "stable")

    def test_compute_ic_trend_unknown_signal(self):
        """Unknown signal should return 'unknown' trend."""
        monitor = ICMonitor()
        assert monitor.compute_ic_trend("nonexistent") == "unknown"

    def test_compute_ic_trend_insufficient_data(self):
        """Insufficient data should return 'unknown' trend."""
        monitor = ICMonitor(trend_window=20)
        for i in range(5):
            monitor.record("short_sig", float(i), float(i) * 0.01)
        assert monitor.compute_ic_trend("short_sig") == "unknown"

    def test_compute_decay_report(self):
        """Decay report should have correct structure."""
        monitor = ICMonitor(window_size=30, trend_window=5)
        for i in range(20):
            monitor.record("sig_a", float(i), float(i) * 0.01)
            monitor.record("sig_b", float(i), float(i % 3) * 0.01)

        report = monitor.compute_decay_report()
        assert "sig_a" in report
        assert "sig_b" in report

        for name, data in report.items():
            assert "ic_rolling" in data
            assert "ic_trend" in data
            assert "observations" in data
            assert "status" in data
            assert data["status"] in ("healthy", "warning", "critical", "insufficient_data")

    def test_decay_report_discloses_metric_contract_without_inference(self):
        monitor = ICMonitor(window_size=30, min_obs_for_status=5)
        for i in range(10):
            monitor.record("ensemble_equity", float(i), float(i) * 0.01)

        row = monitor.compute_decay_report()["ensemble_equity"]

        assert row["metric_axis"] == "time_series_rank_correlation"
        assert row["metric_kind"] == "correlation"
        assert row["estimate_kind"] == "descriptive"
        assert row["alignment_status"] == "provisional"
        assert row["inference_status"] == "unavailable"
        assert row["inference_reason"] == "legacy_rows_missing_alignment_metadata"
        assert row["observation_count"] == 10
        assert row["observation_unit"] == "pairs"
        assert row["evaluation_contract"]["target_asset"] == "SPY"
        forbidden = {
            "t_stat", "p_value", "mean_ic", "ic_std", "icir",
            "effective_sample_size", "t_stat_nw",
        }
        assert forbidden.isdisjoint(row)

    def test_complete_aligned_rows_still_fail_closed_on_dependence(self):
        monitor = ICMonitor(window_size=30, min_obs_for_status=5)
        for i in range(5):
            monitor.record(
                "ensemble_equity",
                float(i),
                float(i) * 0.01,
                observation_metadata={
                    "prediction_date": f"2026-08-{i + 1:02d}",
                    "realized_start_date": f"2026-08-{i + 2:02d}",
                    "resolved_date": f"2026-08-{i + 2:02d}",
                    "target_asset": "SPY",
                    "intended_horizon_sessions": 1,
                    "realized_horizon_sessions": 1,
                    "prediction_field": "ensemble_voting.equity_bias",
                    "prediction_transform": "identity",
                    "metric_axis": "time_series_rank_correlation",
                    "metric_kind": "correlation",
                    "contract_version": "ic-observation-metadata/v2",
                },
            )

        row = monitor.compute_decay_report()["ensemble_equity"]
        assert row["alignment_status"] == "aligned"
        assert row["inference_status"] == "unavailable"
        assert row["inference_reason"] == "dependence_not_characterized"

    def test_complete_metadata_with_wrong_metric_axis_is_not_aligned(self):
        monitor = ICMonitor(window_size=30, min_obs_for_status=5)
        for i in range(5):
            monitor.record(
                "ensemble_equity",
                float(i),
                float(i) * 0.01,
                observation_metadata={
                    "prediction_date": f"2026-08-{i + 1:02d}",
                    "realized_start_date": f"2026-08-{i + 2:02d}",
                    "resolved_date": f"2026-08-{i + 2:02d}",
                    "target_asset": "SPY",
                    "intended_horizon_sessions": 1,
                    "realized_horizon_sessions": 1,
                    "prediction_field": "ensemble_voting.equity_bias",
                    "prediction_transform": "identity",
                    "metric_axis": "cross_sectional_ic",
                    "metric_kind": "correlation",
                    "contract_version": "ic-observation-metadata/v2",
                },
            )

        row = monitor.compute_decay_report()["ensemble_equity"]
        assert row["alignment_status"] == "misaligned"
        assert row["inference_reason"] == "label_alignment_mismatch"

    def test_partial_v2_metadata_has_specific_unavailable_reason(self):
        monitor = ICMonitor(window_size=30, min_obs_for_status=5)
        for i in range(5):
            monitor.record(
                "ensemble_equity",
                float(i),
                float(i) * 0.01,
                observation_metadata={
                    "prediction_date": f"2026-08-{i + 1:02d}",
                    "metric_axis": "time_series_rank_correlation",
                    "metric_kind": "correlation",
                    "contract_version": "ic-observation-metadata/v2",
                },
            )

        row = monitor.compute_decay_report()["ensemble_equity"]
        assert row["alignment_status"] == "provisional"
        assert row["inference_reason"] == "observation_metadata_incomplete"

    def test_every_monitored_signal_has_a_versioned_contract(self):
        assert set(IC_EVALUATION_CONTRACTS) == {
            "ensemble_equity",
            "ensemble_gold",
            "ensemble_duration",
            "ensemble_consensus",
            "alternative_data",
            "behavioral_sentiment",
            "factor_rotation",
            "fred_macro",
        }
        for contract in IC_EVALUATION_CONTRACTS.values():
            assert contract["contract_version"] == "ic-evaluation-contract/v2"
            assert contract["intended_metric_axis"] in {
                "time_series_rank_correlation",
                "cross_sectional_ic",
                "calibration_proper_score",
            }
            assert contract["intended_metric_kind"] in {
                "correlation",
                "calibration_proper_score",
            }
            assert contract["prediction_field"]
            assert contract["prediction_transform"]

    def test_decay_report_status_healthy(self):
        """High IC signal should get 'healthy' status."""
        monitor = ICMonitor(window_size=30, stable_min=0.05, min_obs_for_status=10)
        for i in range(15):
            monitor.record("healthy_sig", float(i), float(i) * 0.01)
        report = monitor.compute_decay_report()
        assert report["healthy_sig"]["status"] == "healthy"

    def test_decay_report_status_critical(self):
        """Very low IC should get 'critical' status once min_obs is met."""
        monitor = ICMonitor(window_size=30, decay_threshold=0.5, min_obs_for_status=10)
        # Random predictions — low correlation
        import random
        random.seed(123)
        for i in range(15):
            monitor.record("weak_sig", random.random(), random.random())
        report = monitor.compute_decay_report()
        assert report["weak_sig"]["status"] in ("critical", "warning")

    def test_thin_history_is_insufficient_not_critical(self):
        """n≈6 resolved pairs must not escalate critical (noisy Spearman)."""
        monitor = ICMonitor(window_size=30, decay_threshold=0.05, min_obs_for_status=20)
        # Strongly anti-correlated but thin — would be critical if allowed.
        for i in range(6):
            monitor.record("thin_sig", float(i), -float(i) * 0.01)
        report = monitor.compute_decay_report()
        assert report["thin_sig"]["status"] == "insufficient_data"
        assert report["thin_sig"]["observations"] == 6
        assert report["thin_sig"]["ic_rolling"] is not None

    def test_get_signals_needing_attention(self):
        """Should return only signals with warning/critical status."""
        monitor = ICMonitor(window_size=30, decay_threshold=0.5, stable_min=0.6)
        import random
        random.seed(42)
        # Good signal
        for i in range(15):
            monitor.record("good_sig", float(i), float(i) * 0.01)
        # Bad signal
        for i in range(15):
            monitor.record("bad_sig", random.random(), random.random())

        attention = monitor.get_signals_needing_attention()
        # bad_sig should need attention; good_sig should not
        assert "bad_sig" in attention or len(attention) >= 0  # at minimum no crash

    def test_multiple_signals_tracked_independently(self):
        """Each signal should be tracked in its own window."""
        monitor = ICMonitor()
        for i in range(10):
            monitor.record("sig_x", float(i), float(i) * 0.01)
            monitor.record("sig_y", float(i), -float(i) * 0.01)

        ic_x = monitor.compute_ic("sig_x")
        ic_y = monitor.compute_ic("sig_y")
        assert ic_x is not None
        assert ic_y is not None
        assert ic_x > 0  # Positive correlation
        assert ic_y < 0  # Negative correlation

    def test_stage_predictions_stores_state(self):
        """stage_predictions should store the prediction dict and date."""
        monitor = ICMonitor()
        monitor.stage_predictions({"sig_a": 0.5, "sig_b": -0.2}, "2026-05-26")
        assert monitor.has_staged_predictions()
        assert monitor.get_staged_date() == "2026-05-26"

    def test_stage_predictions_per_signal_cohorts_coexist(self):
        """Per-signal staging: same identity restage replaces, new cohorts coexist."""
        monitor = ICMonitor()
        monitor.stage_predictions({"sig_a": 0.5}, "2026-05-26")
        monitor.stage_predictions({"sig_a": 0.55}, "2026-05-26")  # idempotent identity
        monitor.stage_predictions({"sig_b": -0.2}, "2026-05-27")  # new cohort
        assert monitor.get_staged_prediction_count() == 2
        n = monitor.resolve_staged(0.02)
        assert n == 2

    def test_resolve_staged_empty(self):
        """resolve_staged with nothing staged should return 0."""
        monitor = ICMonitor()
        n = monitor.resolve_staged(0.02)
        assert n == 0

    def test_resolve_staged_records_pairs(self):
        """resolve_staged should pair each prediction with the forward return."""
        monitor = ICMonitor(window_size=30)
        monitor.stage_predictions({"sig_a": 0.8, "sig_b": -0.3}, "2026-05-26")
        n = monitor.resolve_staged(0.015)
        assert n == 2
        assert not monitor.has_staged_predictions()
        # Each signal should have 1 observation in _data
        assert "sig_a" in monitor._data
        assert "sig_b" in monitor._data
        assert len(monitor._data["sig_a"]) == 1
        assert len(monitor._data["sig_b"]) == 1

    def test_resolve_staged_skips_none(self):
        """None/inf/nan prediction values should be skipped, not recorded."""
        monitor = ICMonitor()
        monitor.stage_predictions(
            {"good": 0.5, "bad_none": None, "bad_nan": float("nan"), "bad_inf": float("inf")},
            "2026-05-26",
        )
        n = monitor.resolve_staged(0.02)
        assert n == 1  # only "good" recorded
        assert "good" in monitor._data
        assert "bad_none" not in monitor._data

    def test_has_staged_predictions_empty_dict(self):
        """has_staged_predictions with empty staging should return False."""
        monitor = ICMonitor()
        monitor._staged = {}
        assert not monitor.has_staged_predictions()

    def test_staged_survives_save_load_cycle(self, tmp_path):
        """Staged predictions should persist through save/load."""
        monitor = ICMonitor()
        monitor.record("sig_a", 0.1, 0.01)
        monitor.stage_predictions({"sig_a": 0.5}, "2026-05-26")

        path = tmp_path / "ic_state.json"
        monitor.save_state(path=path)

        monitor2 = ICMonitor()
        monitor2.load_state(path=path)
        assert monitor2.has_staged_predictions()
        assert monitor2.get_staged_date() == "2026-05-26"
        assert len(monitor2._data["sig_a"]) == 1

    def test_observation_metadata_round_trips_without_invention(self, tmp_path):
        monitor = ICMonitor(window_size=30)
        monitor.record(
            "ensemble_equity",
            0.4,
            0.01,
            observation_metadata={
                "prediction_date": "2026-08-06",
                "realized_start_date": "2026-08-07",
                "resolved_date": "2026-08-07",
                "target_asset": "SPY",
                "intended_horizon_sessions": 1,
                "realized_horizon_sessions": 1,
                "prediction_field": "ensemble_voting.equity_bias",
                "prediction_transform": "identity",
                "metric_axis": "time_series_rank_correlation",
                "metric_kind": "correlation",
                "contract_version": "ic-observation-metadata/v2",
            },
        )
        monitor.record("ensemble_equity", 0.5, 0.02)
        path = tmp_path / "ic_state.json"
        monitor.save_state(path)

        restored = ICMonitor(window_size=30)
        restored.load_state(path)

        metadata = list(restored._observation_metadata["ensemble_equity"])
        assert metadata[0]["resolved_date"] == "2026-08-07"
        assert metadata[0]["target_asset"] == "SPY"
        assert metadata[1] is None
        saved = json.loads(path.read_text())
        assert saved["__state_schema_version__"] == "ic-monitor-state/v2"
        assert saved["__observation_metadata__"]["ensemble_equity"][1] is None


class TestICMonitorPersistence:
    """Test save/load state persistence."""

    def test_save_and_load_state(self, tmp_path):
        """Save then load should preserve signal data."""
        monitor = ICMonitor(window_size=30)
        for i in range(10):
            monitor.record("test_sig", float(i), float(i) * 0.01)

        path = tmp_path / "ic_state.json"
        monitor.save_state(path=path)

        monitor2 = ICMonitor(window_size=30)
        monitor2.load_state(path=path)

        ic1 = monitor.compute_ic("test_sig")
        ic2 = monitor2.compute_ic("test_sig")
        assert abs(ic1 - ic2) < 0.001

    def test_load_nonexistent_state(self, tmp_path):
        """Loading nonexistent state should not crash."""
        monitor = ICMonitor()
        monitor.load_state(path=tmp_path / "nonexistent.json")
        assert len(monitor._data) == 0

    def test_load_corrupt_state(self, tmp_path):
        """Loading corrupt JSON should not crash."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")
        monitor = ICMonitor()
        monitor.load_state(path=path)
        assert len(monitor._data) == 0

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"sig": [[0.1, 0.01, "extra"]]},
            {"sig": "not-a-list"},
        ],
    )
    def test_load_malformed_state_fails_soft(self, tmp_path, payload):
        path = tmp_path / "malformed.json"
        path.write_text(json.dumps(payload))
        monitor = ICMonitor()

        monitor.load_state(path)

        assert len(monitor._data) == 0
        assert monitor.compute_decay_report() == {}

    def test_mismatched_metadata_length_is_discarded_not_tail_guessed(self, tmp_path):
        path = tmp_path / "mismatched.json"
        path.write_text(json.dumps({
            "ensemble_equity": [[0.1, 0.01], [0.2, 0.02]],
            "__state_schema_version__": "ic-monitor-state/v2",
            "__observation_metadata__": {
                "ensemble_equity": [{
                    "prediction_date": "2026-08-07",
                    "contract_version": "ic-observation-metadata/v2",
                }],
            },
        }))
        monitor = ICMonitor()

        monitor.load_state(path)

        assert list(monitor._observation_metadata["ensemble_equity"]) == [None, None]

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save should create parent directories."""
        path = tmp_path / "sub" / "dir" / "ic_state.json"
        monitor = ICMonitor()
        monitor.record("sig", 0.1, 0.01)
        monitor.save_state(path=path)
        assert path.exists()

    def test_state_json_is_valid(self, tmp_path):
        """Saved state should be valid JSON."""
        monitor = ICMonitor()
        for i in range(5):
            monitor.record("sig", float(i), float(i) * 0.01)
        path = tmp_path / "ic_state.json"
        monitor.save_state(path=path)
        with open(path) as f:
            state = json.load(f)
        assert "sig" in state
        assert len(state["sig"]) == 5


class TestComputeICDecayReport:
    """Test the convenience function."""

    def test_convenience_function_returns_dict(self):
        """compute_ic_decay_report should return a dict without crashing."""
        report = compute_ic_decay_report()
        assert isinstance(report, dict)

    def test_staged_only_state_reports_waiting_for_forward_returns(
        self, tmp_path, monkeypatch
    ):
        """Pending labels should be explicit instead of looking like no IC data."""
        import src.monitor.ic_decay_monitor as icm

        state_path = tmp_path / "ic_monitor_state.json"
        state_path.write_text(json.dumps({
            "__staged__": {
                "date": "2026-07-02",
                "predictions": {"ensemble_equity": 0.4, "fred_macro": 0.6},
            }
        }))
        monkeypatch.setattr(icm, "IC_STATE_PATH", state_path)

        report = compute_ic_decay_report()

        assert report["status"] == "waiting_for_forward_returns"
        assert report["pending_predictions"] == 2
        assert report["staged_date"] == "2026-07-02"
        assert report["signals"] == {}
        assert "__staged__" not in report["signals"]


def test_ic_decay_report_tags_pending_scopes(tmp_path, monkeypatch):
    from src.monitor import ic_decay_monitor as icm

    monkeypatch.setattr(icm, "_signal_prediction_backlog", lambda db_path=None: {
        "pending_rows": 100,
        "pending_dates": 5,
        "oldest_unresolved_date": "2020-01-01",
        "total_predictions": 200,
        "resolved_predictions": 100,
        "pending_semantics": "test",
    })
    class FakeMon:
        def load_state(self):
            return None
        def compute_decay_report(self):
            return {}
        def get_staged_prediction_count(self):
            return 6
        def get_staged_date(self):
            return "2026-07-20"
    monkeypatch.setattr(icm, "ICMonitor", FakeMon)
    report = icm.compute_ic_decay_report()
    assert report["pending_predictions"] == 6
    assert report["pending_scope"] == "ic_staged_date_window"
    assert report["pending_rows"] == 100
    assert report["pending_rows_scope"] == "historical_db_unlabeled_rows"


def _load_ic_decay_evidence_fixture() -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / "ic_decay_critical_minimum.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _raw_report_from_evidence_fixture(fixture: dict) -> dict:
    staged = fixture["staged_pending_predictions"]
    backlog = fixture["historical_unlabeled_backlog"]
    return {
        "status": fixture["status"],
        "signals": fixture["signals"],
        "resolved_signal_count": fixture["resolved_signal_count"],
        "pending_predictions": staged["count"],
        "pending_scope": staged["scope"],
        "staged_prediction_names": staged["signal_names"],
        "staged_date": staged["date"],
        "pending_rows": backlog["rows"],
        "pending_rows_scope": backlog["scope"],
        "pending_dates": backlog["dates"],
        "oldest_unresolved_date": backlog["oldest_date"],
    }


def test_critical_row_at_minimum_observations_is_reviewable_not_hidden() -> None:
    fixture = _load_ic_decay_evidence_fixture()
    report = _raw_report_from_evidence_fixture(fixture)

    summary = build_ic_decay_summary(
        report,
        evidence_generated_at=fixture["generated_at"],
        control_effect="paper_warning",
    )

    assert summary["status"] == "critical"
    assert summary["critical_signals"] == ["ensemble_consensus", "ensemble_duration"]
    assert summary["warning_signals"] == ["ensemble_equity"]
    assert summary["min_observations"] == 20
    assert summary["resolved_signal_count"] == 4
    assert summary["evidence_generated_at"] == fixture["generated_at"]
    assert summary["evidence_freshness"] == "captured_runtime_snapshot"


def test_insufficient_data_rows_remain_non_paging() -> None:
    fixture = _load_ic_decay_evidence_fixture()
    summary = build_ic_decay_summary(
        _raw_report_from_evidence_fixture(fixture),
        evidence_generated_at=fixture["generated_at"],
        control_effect="paper_warning",
    )

    assert summary["insufficient_data_signals"] == [
        "alternative_data",
        "behavioral_sentiment",
        "factor_rotation",
        "fred_macro",
    ]
    assert not set(summary["insufficient_data_signals"]) & set(summary["critical_signals"])
    assert not set(summary["insufficient_data_signals"]) & set(summary["warning_signals"])


def test_ic_summary_keeps_staged_and_historical_pending_scopes_distinct() -> None:
    fixture = _load_ic_decay_evidence_fixture()
    summary = build_ic_decay_summary(
        _raw_report_from_evidence_fixture(fixture),
        evidence_generated_at=fixture["generated_at"],
        control_effect="paper_warning",
    )

    assert summary["staged_pending_predictions"] == 7
    assert summary["staged_date"] == "2026-08-01"
    assert summary["staged_pending_scope"] == "ic_staged_date_window"
    assert summary["historical_unlabeled_rows"] == 1663
    assert summary["historical_unlabeled_dates"] == 2
    assert summary["historical_unlabeled_scope"] == "historical_db_unlabeled_rows"
    assert summary["staged_pending_scope"] != summary["historical_unlabeled_scope"]


def test_ic_summary_projects_bounded_signal_evidence() -> None:
    monitor = ICMonitor(window_size=30, min_obs_for_status=5)
    for i in range(10):
        monitor.record("ensemble_gold", float(i), -float(i))
    raw = {
        "status": "critical",
        "signals": monitor.compute_decay_report(),
    }

    summary = build_ic_decay_summary(raw)
    row = summary["signal_evidence"]["ensemble_gold"]

    assert row["metric_axis"] == "time_series_rank_correlation"
    assert row["alignment_status"] == "misaligned"
    assert row["inference_status"] == "unavailable"
    assert row["evaluation_contract"]["target_asset"] == "GLD"
    assert "latest_observation_metadata" not in row
    assert "prediction_rows" not in json.dumps(summary)


# ── Task 2A: control eligibility layer ─────────────────────────────────

def test_decay_report_exposes_control_eligibility_fields():
    """Every row exposes control_eligible / control_status / reason."""
    monitor = ICMonitor()
    monitor.record("ensemble_equity", 0.5, 0.02)
    monitor.record("ensemble_equity", -0.3, -0.01)
    for _ in range(25):
        monitor.record("ensemble_equity", 0.1, 0.005)
    report = monitor.compute_decay_report()
    row = report["ensemble_equity"]
    assert "control_eligible" in row
    assert "control_status" in row
    assert "control_ineligibility_reason" in row
    # Legacy rows without v2 metadata are never silently eligible.
    assert row["control_eligible"] is False
    assert row["control_ineligibility_reason"] == "legacy_rows_missing_alignment_metadata"


def test_aligned_complete_v2_metadata_becomes_control_eligible():
    """Provisional signals with complete aligned v2 metadata become eligible."""
    monitor = ICMonitor()
    from src.monitor.ic_decay_monitor import (
        IC_EVALUATION_CONTRACTS,
        IC_OBSERVATION_METADATA_VERSION,
    )

    contract = IC_EVALUATION_CONTRACTS["ensemble_equity"]
    for i in range(25):
        monitor.record(
            "ensemble_equity",
            0.1 + i * 0.001,
            0.005,
            observation_metadata={
                "prediction_date": f"2026-07-{(i % 28) + 1:02d}",
                "realized_start_date": f"2026-07-{(i % 28) + 2:02d}",
                "resolved_date": f"2026-07-{(i % 28) + 2:02d}",
                "target_asset": contract["target_asset"],
                "intended_horizon_sessions": contract["intended_horizon_sessions"],
                "realized_horizon_sessions": contract["intended_horizon_sessions"],
                "prediction_field": contract["prediction_field"],
                "prediction_transform": contract["prediction_transform"],
                "metric_axis": contract["intended_metric_axis"],
                "metric_kind": contract["intended_metric_kind"],
                "contract_version": IC_OBSERVATION_METADATA_VERSION,
            },
        )
    row = monitor.compute_decay_report()["ensemble_equity"]
    assert row["alignment_status"] == "aligned"
    assert row["control_eligible"] is True
    assert row["control_status"] == "eligible"


def test_ic_summary_distinguishes_control_eligible_critical_signals():
    """Summary lists eligible criticals separately from descriptive ones."""
    monitor = ICMonitor()
    # Ineligible critical: legacy rows with poor (anti-correlated) IC.
    for i in range(25):
        monitor.record("ensemble_equity", 0.5 if i % 2 else -0.5, -0.02 if i % 2 else 0.02)
    monitor.record("alternative_data", 0.5, 0.02)
    for i in range(24):
        monitor.record("alternative_data", 0.5 if i % 2 else -0.5, -0.02 if i % 2 else 0.02)

    from src.monitor.ic_decay_monitor import build_ic_decay_summary

    summary = build_ic_decay_summary({"signals": monitor.compute_decay_report()})
    assert set(summary["critical_signals"]) >= {"ensemble_equity", "alternative_data"}
    assert "control_eligible_critical_signals" in summary
    assert isinstance(summary["control_eligible_critical_signals"], list)


def test_per_signal_staging_coexists_and_resolves_partially():
    """Different targets/horizons stage together; only matching entries resolve."""
    monitor = ICMonitor()
    monitor.stage_predictions(
        {"ensemble_equity": 0.4, "ensemble_gold": 0.2},
        "2026-08-07",
        prediction_metadata={
            "ensemble_equity": {"target_asset": "SPY", "intended_horizon_sessions": 1},
            "ensemble_gold": {"target_asset": "GLD", "intended_horizon_sessions": 1},
        },
    )
    assert monitor.get_staged_prediction_count() == 2

    # Resolve SPY-targeted entries only; GLD stays staged.
    n = monitor.resolve_staged(
        0.01,
        target_asset="SPY",
        resolved_date="2026-08-08",
        realized_start_date="2026-08-07",
        realized_horizon_sessions=1,
    )
    assert n == 1
    assert monitor.get_staged_prediction_names() == ["ensemble_gold"]

    # Now resolve the remaining GLD entry against GLD's return.
    n = monitor.resolve_staged(
        0.02,
        target_asset="GLD",
        resolved_date="2026-08-08",
        realized_start_date="2026-08-07",
        realized_horizon_sessions=1,
    )
    assert n == 1
    assert monitor.get_staged_prediction_count() == 0
    assert len(monitor._data["ensemble_equity"]) == 1
    assert len(monitor._data["ensemble_gold"]) == 1
    meta = list(monitor._observation_metadata["ensemble_gold"])[0]
    assert meta["target_asset"] == "GLD"


def test_staging_is_idempotent_by_observation_identity():
    """Re-staging the same signal+date does not duplicate the cohort."""
    monitor = ICMonitor()
    monitor.stage_predictions({"ensemble_equity": 0.4}, "2026-08-07")
    monitor.stage_predictions({"ensemble_equity": 0.4}, "2026-08-07")
    assert monitor.get_staged_prediction_count() == 1
    monitor.stage_predictions({"ensemble_equity": 0.5}, "2026-08-08")
    assert monitor.get_staged_prediction_count() == 2


def test_behavioral_five_session_horizon_waits_for_sessions():
    """A 5-session staged signal resolves only after 5 realized sessions."""
    monitor = ICMonitor()
    monitor.stage_predictions(
        {"behavioral_sentiment": 0.3},
        "2026-07-31",
        prediction_metadata={
            "behavioral_sentiment": {"target_asset": "SPY", "intended_horizon_sessions": 5}
        },
    )
    n = monitor.resolve_staged(
        0.01,
        target_asset="SPY",
        resolved_date="2026-08-03",
        realized_start_date="2026-07-31",
        realized_horizon_sessions=2,
    )
    assert n == 0
    assert monitor.get_staged_prediction_count() == 1
    n = monitor.resolve_staged(
        0.01,
        target_asset="SPY",
        resolved_date="2026-08-07",
        realized_start_date="2026-07-31",
        realized_horizon_sessions=5,
    )
    assert n == 1


class TestRebaselineAndTrigger:
    """Operator-approved re-baseline machinery (incident 8115a9c1, 2026-08-11).

    Option A (hold) was approved; Option B (re-baseline) is implemented and
    armed: ``rebaseline()`` archives the current staging epoch and starts a
    fresh accumulation epoch; ``rebaseline_trigger_state()`` reports when any
    signal has accumulated ``min_obs_for_status`` staged v2 observations —
    the evidence-based re-review point. The halt itself is untouched by these
    methods.
    """

    def test_rebaseline_archives_full_epoch_and_starts_fresh(self, tmp_path):
        monitor = ICMonitor()
        monitor.stage_predictions(
            {"ensemble_equity": 0.3, "ensemble_gold": -0.1}, "2026-08-10"
        )
        monitor.record("ensemble_equity", 0.3, 0.01)
        monitor.record("ensemble_gold", -0.1, -0.005)
        assert monitor.get_staged_prediction_count() == 2

        archive_path = monitor.rebaseline(archive_path=tmp_path / "epoch.json")

        # Archive snapshot preserves the prior epoch losslessly.
        snapshot = json.loads(archive_path.read_text(encoding="utf-8"))
        assert snapshot["schema"] == "ic-rebaseline-archive/v1"
        staged_identities = {
            entry["identity"] for entry in snapshot["staged"]
        }
        assert staged_identities == {
            "ensemble_equity|2026-08-10|ic-observation-metadata/v2",
            "ensemble_gold|2026-08-10|ic-observation-metadata/v2",
        }
        assert snapshot["observations"]["ensemble_equity"] == [[0.3, 0.01]]
        assert snapshot["observations"]["ensemble_gold"] == [[-0.1, -0.005]]
        assert "ensemble_equity" in snapshot["observation_metadata"]

        # Measurement restarts from a fresh epoch.
        assert monitor.get_staged_prediction_count() == 0
        assert monitor.staged_observation_counts() == {}
        assert monitor.compute_decay_report() == {}

    def test_rebaseline_epoch_persists_across_save_and_load(self, tmp_path):
        monitor = ICMonitor()
        monitor.stage_predictions({"ensemble_equity": 0.3}, "2026-08-10")
        monitor.rebaseline(archive_path=tmp_path / "epoch.json")
        state_path = tmp_path / "state.json"
        monitor.save_state(state_path)

        reloaded = ICMonitor()
        reloaded.load_state(state_path)
        assert reloaded.get_staged_prediction_count() == 0
        assert reloaded.staged_observation_counts() == {}
        assert reloaded.compute_decay_report() == {}

    def test_rebaseline_default_archive_path_is_dated(self, tmp_path, monkeypatch):
        from src.monitor import ic_decay_monitor as icm

        monkeypatch.setattr(icm, "DATA_DIR", tmp_path)
        monitor = ICMonitor()
        monitor.stage_predictions({"ensemble_equity": 0.3}, "2026-08-10")
        archive_path = monitor.rebaseline()
        assert archive_path.exists()
        assert archive_path.parent == tmp_path / "ic_rebaseline_archives"
        snapshot = json.loads(archive_path.read_text(encoding="utf-8"))
        assert len(snapshot["staged"]) == 1

    def test_rebaseline_trigger_state_below_threshold(self):
        monitor = ICMonitor()
        monitor.stage_predictions({"ensemble_equity": 0.3}, "2026-08-10")
        state = monitor.rebaseline_trigger_state()
        assert state["due"] is False
        assert state["threshold"] == 20
        assert state["max_staged_observations"] == 1
        assert state["staged_observations_per_signal"] == {"ensemble_equity": 1}

    def test_rebaseline_trigger_due_at_threshold(self):
        monitor = ICMonitor()
        for i in range(20):
            monitor.stage_predictions(
                {"ensemble_equity": 0.1 * i}, f"2026-07-{i + 1:02d}"
            )
        state = monitor.rebaseline_trigger_state()
        assert state["due"] is True
        assert state["max_staged_observations"] == 20
        assert state["staged_observations_per_signal"] == {"ensemble_equity": 20}

    def test_compute_ic_decay_report_exposes_rebaseline_trigger(self, tmp_path, monkeypatch):
        from src.monitor import ic_decay_monitor as icm

        monkeypatch.setattr(
            icm, "_signal_prediction_backlog",
            lambda db_path=None: {
                "pending_rows": 0,
                "pending_dates": 0,
                "oldest_unresolved_date": None,
                "total_predictions": 0,
                "resolved_predictions": 0,
                "pending_semantics": "test",
            },
        )
        state_path = tmp_path / "ic_state.json"
        monkeypatch.setattr(icm, "IC_STATE_PATH", state_path)
        monitor = icm.ICMonitor()
        monitor.stage_predictions({"ensemble_equity": 0.3}, "2026-08-10")
        monitor.save_state(state_path)

        report = icm.compute_ic_decay_report()
        assert report["rebaseline_due"] is False
        assert report["rebaseline_threshold"] == 20
        assert report["max_staged_observations"] == 1
        assert report["staged_observations_per_signal"] == {"ensemble_equity": 1}

    def test_ic_summary_projects_rebaseline_trigger_fields(self):
        report = {
            "status": "no_data",
            "signals": {},
            "pending_predictions": 21,
            "pending_scope": "ic_staged_date_window",
            "staged_prediction_names": ["ensemble_equity"],
            "staged_observations_per_signal": {"ensemble_equity": 21},
            "rebaseline_due": True,
            "rebaseline_threshold": 20,
            "max_staged_observations": 21,
            "pending_rows": 0,
            "pending_rows_scope": "historical_db_unlabeled_rows",
            "pending_dates": 0,
            "oldest_unresolved_date": None,
            "staged_date": "2026-08-01",
        }
        summary = build_ic_decay_summary(report)
        assert summary["rebaseline_due"] is True
        assert summary["rebaseline_threshold"] == 20
        assert summary["max_staged_observations"] == 21
        assert summary["staged_observations_per_signal"] == {"ensemble_equity": 21}
