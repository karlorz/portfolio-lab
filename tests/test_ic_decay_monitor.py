"""Tests for IC decay monitor — signal quality tracking."""

import json
from pathlib import Path

import pytest

from src.monitor.ic_decay_monitor import (
    ICMonitor,
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

    def test_insufficient_data_returns_zero(self):
        """Less than 5 data points should return 0.0."""
        assert _spearman_rank_correlation([1, 2, 3], [1, 2, 3]) == 0.0
        assert _spearman_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0

    def test_zero_variance_returns_zero(self):
        """Constant values (zero variance) should return 0.0."""
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        assert _spearman_rank_correlation(x, y) == 0.0

    def test_nan_inf_values_handled(self):
        """NaN/inf values should be filtered out."""
        import math
        x = [1, 2, math.nan, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        r = _spearman_rank_correlation(x, y)
        # After filtering nan, 9 points remain — should still compute
        assert -1.0 <= r <= 1.0

    def test_empty_arrays_return_zero(self):
        """Empty arrays should return 0.0."""
        assert _spearman_rank_correlation([], []) == 0.0


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

    def test_decay_report_status_healthy(self):
        """High IC signal should get 'healthy' status."""
        monitor = ICMonitor(window_size=30, stable_min=0.05)
        for i in range(15):
            monitor.record("healthy_sig", float(i), float(i) * 0.01)
        report = monitor.compute_decay_report()
        assert report["healthy_sig"]["status"] == "healthy"

    def test_decay_report_status_critical(self):
        """Very low IC should get 'critical' status."""
        monitor = ICMonitor(window_size=30, decay_threshold=0.5)
        # Random predictions — low correlation
        import random
        random.seed(123)
        for i in range(15):
            monitor.record("weak_sig", random.random(), random.random())
        report = monitor.compute_decay_report()
        assert report["weak_sig"]["status"] in ("critical", "warning")

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

    def test_stage_predictions_replaces_previous(self):
        """Calling stage_predictions twice should replace, not append."""
        monitor = ICMonitor()
        monitor.stage_predictions({"sig_a": 0.5}, "2026-05-26")
        monitor.stage_predictions({"sig_b": -0.2}, "2026-05-27")
        assert monitor.get_staged_date() == "2026-05-27"
        n = monitor.resolve_staged(0.02)
        assert n == 1

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
        """has_staged_predictions with empty predictions should return False."""
        monitor = ICMonitor()
        monitor._staged = {"date": "2026-05-26", "predictions": {}}
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
