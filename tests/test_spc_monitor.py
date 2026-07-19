"""Tests for src.monitor.spc_monitor — SPC signal quality monitoring."""

import pytest
from src.monitor.spc_monitor import SPCMonitor


class TestSPCMonitorRecord:
    def test_record_single_value(self):
        monitor = SPCMonitor()
        monitor.record("test_signal", 0.5)
        assert "test_signal" in monitor._windows
        assert len(monitor._windows["test_signal"]) == 1

    def test_record_multiple_values(self):
        monitor = SPCMonitor()
        for v in [0.3, 0.35, 0.4, 0.32, 0.38]:
            monitor.record("test_signal", v)
        assert len(monitor._windows["test_signal"]) == 5

    def test_record_respects_window_size(self):
        monitor = SPCMonitor(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            monitor.record("test_signal", v)
        assert len(monitor._windows["test_signal"]) == 3
        assert list(monitor._windows["test_signal"]) == [3.0, 4.0, 5.0]

    def test_record_multiple_signals(self):
        monitor = SPCMonitor()
        monitor.record("signal_a", 0.5)
        monitor.record("signal_b", 0.3)
        assert len(monitor._windows) == 2


class TestSPCMonitorFlags:
    def test_no_flags_with_stable_signal(self):
        monitor = SPCMonitor(window_size=10, sigma_threshold=3.0, consecutive_breach_limit=3)
        for _ in range(10):
            monitor.record("stable", 0.5)
        flags = monitor.check_flags()
        assert flags == []

    def test_flags_breaching_signal(self):
        monitor = SPCMonitor(window_size=10, sigma_threshold=3.0, consecutive_breach_limit=3)
        # Build a baseline with some variance (std != 0)
        for v in [0.3, 0.35, 0.4, 0.32, 0.38, 0.36, 0.34, 0.39, 0.33, 0.37]:
            monitor.record("drifty", v)
        # Now push 3 consecutive values far from mean
        for _ in range(3):
            monitor.record("drifty", 100.0)
        flags = monitor.check_flags()
        assert len(flags) == 1
        assert flags[0]["signal"] == "drifty"
        assert flags[0]["consecutive_breaches"] >= 3

    def test_no_flag_below_consecutive_limit(self):
        monitor = SPCMonitor(window_size=10, sigma_threshold=3.0, consecutive_breach_limit=3)
        for v in [0.3, 0.35, 0.4, 0.32, 0.38, 0.36, 0.34, 0.39, 0.33, 0.37]:
            monitor.record("drifty", v)
        # Only 2 breaches (below limit of 3)
        monitor.record("drifty", 100.0)
        monitor.record("drifty", 100.0)
        flags = monitor.check_flags()
        assert flags == []

    def test_breach_count_resets_on_normal_value(self):
        monitor = SPCMonitor(window_size=10, sigma_threshold=3.0, consecutive_breach_limit=3)
        for v in [0.3, 0.35, 0.4, 0.32, 0.38, 0.36, 0.34, 0.39, 0.33, 0.37]:
            monitor.record("test", v)
        monitor.record("test", 100.0)  # breach 1
        monitor.record("test", 100.0)  # breach 2
        monitor.record("test", 0.35)   # reset
        monitor.record("test", 100.0)  # breach 1 again
        flags = monitor.check_flags()
        assert flags == []  # Only 1 consecutive, not 3


class TestSPCMonitorStatus:
    def test_get_signal_status(self):
        monitor = SPCMonitor()
        for v in [0.3, 0.35, 0.4, 0.32, 0.38]:
            monitor.record("test", v)
        status = monitor.get_signal_status("test")
        assert status is not None
        assert status["signal"] == "test"
        assert status["sample_count"] == 5
        assert status["mean"] is not None
        assert status["std"] is not None
        assert not status["is_flagged"]

    def test_get_signal_status_unknown(self):
        monitor = SPCMonitor()
        assert monitor.get_signal_status("unknown") is None

    def test_get_all_status(self):
        monitor = SPCMonitor()
        monitor.record("a", 0.5)
        monitor.record("b", 0.3)
        all_status = monitor.get_all_status()
        assert "a" in all_status
        assert "b" in all_status

    def test_status_includes_control_limits(self):
        monitor = SPCMonitor()
        for v in [0.3, 0.35, 0.4, 0.32, 0.38]:
            monitor.record("test", v)
        status = monitor.get_signal_status("test")
        assert "ucl" in status  # upper control limit
        assert "lcl" in status  # lower control limit
        assert status["ucl"] > status["lcl"]


class TestSPCMonitorReset:
    def test_reset_specific_signal(self):
        monitor = SPCMonitor()
        monitor.record("a", 0.5)
        monitor.record("b", 0.3)
        monitor.reset("a")
        assert monitor.get_signal_status("a") is None
        assert monitor.get_signal_status("b") is not None

    def test_reset_all_signals(self):
        monitor = SPCMonitor()
        monitor.record("a", 0.5)
        monitor.record("b", 0.3)
        monitor.reset()
        assert monitor.get_signal_status("a") is None
        assert monitor.get_signal_status("b") is None


class TestSPCMonitorEdgeCases:
    def test_single_value_no_stats(self):
        monitor = SPCMonitor()
        monitor.record("test", 0.5)
        status = monitor.get_signal_status("test")
        assert status["mean"] is None  # need 2+ values for stats

    def test_zero_std(self):
        monitor = SPCMonitor()
        for _ in range(10):
            monitor.record("constant", 1.0)
        flags = monitor.check_flags()
        assert flags == []  # no breach when std=0

    def test_negative_values(self):
        monitor = SPCMonitor()
        for v in [-0.3, -0.35, -0.4, -0.32, -0.38]:
            monitor.record("neg", v)
        status = monitor.get_signal_status("neg")
        assert status["mean"] is not None
        assert status["mean"] < 0

    def test_large_values(self):
        monitor = SPCMonitor()
        for v in [1e6, 1e6 + 1, 1e6 - 1, 1e6, 1e6 + 2]:
            monitor.record("large", v)
        status = monitor.get_signal_status("large")
        assert status["mean"] is not None

    def test_configurable_parameters(self):
        monitor = SPCMonitor(
            window_size=5,
            sigma_threshold=2.0,
            consecutive_breach_limit=2,
        )
        assert monitor.window_size == 5
        assert monitor.sigma_threshold == 2.0
        assert monitor.consecutive_breach_limit == 2

    def test_flags_with_lower_breach(self):
        """Low values (below LCL) should also trigger flags."""
        monitor = SPCMonitor(window_size=10, sigma_threshold=3.0, consecutive_breach_limit=3)
        for v in [0.3, 0.35, 0.4, 0.32, 0.38, 0.36, 0.34, 0.39, 0.33, 0.37]:
            monitor.record("test", v)
        # Push 3 consecutive values far below mean
        for _ in range(3):
            monitor.record("test", -100.0)
        flags = monitor.check_flags()
        assert len(flags) == 1


def test_run_spc_monitor_status_not_ok_when_flags_present(tmp_path, monkeypatch):
    """Dashboard SPC block must surface non-ok status when flagged_signals non-empty."""
    import sqlite3
    from src.dashboard.generator import DashboardGenerator
    from src.monitor.spc_monitor import SPCMonitor

    # Build a monitor that already has a flagged signal
    mon = SPCMonitor(consecutive_breach_limit=3)
    # seed window + force breach count
    for i in range(10):
        mon.record("_ensemble_consensus", 0.5)
    mon._breach_counts["_ensemble_consensus"] = 55
    mon._limits["_ensemble_consensus"] = {
        "mean": 0.5, "std": 0.01, "ucl": 0.53, "lcl": 0.47
    }

    DashboardGenerator._spc_monitor = mon
    # Avoid save_state writing to real paths
    mon.save_state = lambda: None  # type: ignore
    mon.load_state = lambda: None  # type: ignore

    gen = DashboardGenerator.__new__(DashboardGenerator)
    out = gen._run_spc_monitor({"ensemble_voting": {"weighted_consensus": 0.9, "source_breakdown": []}})
    assert out.get("flagged_signals"), "expected flags from seeded monitor"
    assert out.get("status") != "ok", (
        f"spc.status must not stay ok with flags; got {out.get('status')}"
    )
    assert out.get("status") in {"alert", "warning", "breach", "critical", "degraded"}

    # Empty flags → ok
    mon2 = SPCMonitor(consecutive_breach_limit=3)
    mon2.save_state = lambda: None  # type: ignore
    DashboardGenerator._spc_monitor = mon2
    out_ok = gen._run_spc_monitor({"ensemble_voting": {"source_breakdown": []}})
    assert out_ok.get("status") == "ok"
    assert out_ok.get("flagged_signals") == []
    DashboardGenerator._spc_monitor = None
