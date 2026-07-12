"""Tests for src.monitor.alerting — webhook alerting + staleness checks."""

import os
import json
import time
from unittest.mock import patch, MagicMock

import pytest

from src.monitor.alerting import (
    AlertLevel,
    AlertChannel,
    send_alert,
    check_staleness_and_alert,
    check_drift_and_alert,
    check_ic_decay_and_alert,
    _last_alert_time,
    ALERT_MIN_INTERVAL_SECONDS,
)


@pytest.fixture(autouse=True)
def _isolate_incident_manager(tmp_path):
    """Keep send_alert incident side effects out of repo data/ during tests."""
    from src.monitor.incident_manager import IncidentManager

    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    with patch("src.monitor.alerting.get_incident_manager", return_value=manager):
        yield manager


class TestAlertLevel:
    def test_levels_exist(self):
        assert AlertLevel.PASS == "pass"
        assert AlertLevel.WARN == "warn"
        assert AlertLevel.HALT == "halt"


class TestAlertChannel:
    def test_channels_exist(self):
        assert AlertChannel.SIGNAL_STALENESS == "signal_staleness"
        assert AlertChannel.EVALUATOR_ERROR == "evaluator_error"
        assert AlertChannel.PORTFOLIO_DRIFT == "portfolio_drift"
        assert AlertChannel.CRON_FAILURE == "cron_failure"
        assert AlertChannel.IC_DECAY == "ic_decay"


class TestSendAlert:
    def test_disabled_when_no_webhook(self):
        """When ALERT_WEBHOOK_URL is empty, alerting is silently disabled."""
        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""):
            result = send_alert(
                AlertChannel.SIGNAL_STALENESS,
                AlertLevel.WARN,
                "test message",
            )
        assert result is True  # Silently succeeds (no-op)

    def test_disabled_webhook_still_records_incident(self, tmp_path):
        """Dashboard-only alerting should still persist incident lifecycle state."""
        from src.monitor.incident_manager import IncidentManager

        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )

        with patch("src.monitor.alerting.ALERT_WEBHOOK_URL", ""), \
             patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            result = send_alert(
                AlertChannel.PORTFOLIO_DRIFT,
                AlertLevel.WARN,
                "Portfolio drift exceeding 5%",
                details={"drift_pct": 7.5},
            )

        assert result is True
        lines = (tmp_path / "incidents.jsonl").read_text().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "opened"
        assert event["channel"] == "portfolio_drift"
        assert event["severity"] == "p2"

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_dedup_suppressed_alert_still_updates_incident(self, mock_urlopen, tmp_path):
        from src.monitor.incident_manager import IncidentManager

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        _last_alert_time.clear()
        manager = IncidentManager(
            log_path=tmp_path / "incidents.jsonl",
            summary_path=tmp_path / "incidents.json",
        )

        with patch("src.monitor.alerting.get_incident_manager", return_value=manager):
            assert send_alert(AlertChannel.IC_DECAY, AlertLevel.WARN, "IC warning") is True
            assert send_alert(AlertChannel.IC_DECAY, AlertLevel.WARN, "IC warning updated") is True

        assert mock_urlopen.call_count == 1
        events = [json.loads(line) for line in (tmp_path / "incidents.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events] == ["opened", "updated"]
        assert events[1]["message"] == "IC warning updated"

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_sends_webhook_post(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Clear dedup tracker
        _last_alert_time.clear()

        result = send_alert(
            AlertChannel.SIGNAL_STALENESS,
            AlertLevel.WARN,
            "2 signals stale",
            details={"stale": ["alt_data", "garch_cvar"]},
        )
        assert result is True
        mock_urlopen.assert_called_once()

        # Verify payload structure
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.method == "POST"
        payload = json.loads(request.data)
        assert payload["channel"] == "signal_staleness"
        assert payload["level"] == "warn"
        assert payload["message"] == "2 signals stale"
        assert payload["source"] == "portfolio-lab"
        assert "timestamp" in payload
        assert payload["details"]["stale"] == ["alt_data", "garch_cvar"]

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_dedup_suppresses_repeated_alerts(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        _last_alert_time.clear()

        # First alert goes through
        result1 = send_alert(AlertChannel.CRON_FAILURE, AlertLevel.HALT, "cron failed")
        assert result1 is True
        assert mock_urlopen.call_count == 1

        # Second identical alert is suppressed (within dedup interval)
        result2 = send_alert(AlertChannel.CRON_FAILURE, AlertLevel.HALT, "cron failed again")
        assert result2 is True  # Returns True (suppressed, not failed)
        assert mock_urlopen.call_count == 1  # Still only 1 call

    @patch("src.monitor.alerting.urllib.request.urlopen")
    @patch("src.monitor.alerting.ALERT_WEBHOOK_URL", "https://hooks.example.com/test")
    def test_webhook_failure_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")
        _last_alert_time.clear()

        result = send_alert(
            AlertChannel.EVALUATOR_ERROR,
            AlertLevel.WARN,
            "evaluator crashed",
        )
        assert result is False


class TestCheckStalenessAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_all_fresh_sends_pass(self, mock_send):
        staleness = {
            "stale_signals": [],
            "healthy_count": 5,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_some_stale_sends_warn(self, mock_send):
        staleness = {
            "stale_signals": ["ensemble_voting", "garch_cvar"],
            "healthy_count": 3,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.WARN

    @patch("src.monitor.alerting.send_alert")
    def test_all_stale_sends_halt(self, mock_send):
        staleness = {
            "stale_signals": ["ensemble_voting", "alternative_data", "garch_cvar", "behavioral_sentiment", "smart_rebalance"],
            "healthy_count": 0,
            "total_count": 5,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.HALT

    @patch("src.monitor.alerting.send_alert")
    def test_empty_data_no_alert(self, mock_send):
        staleness = {"stale_signals": [], "healthy_count": 0, "total_count": 0}
        check_staleness_and_alert(staleness)
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_unavailable_without_stale_sends_warn_not_pass(self, mock_send):
        """Empty stale + non-empty unavailable must not claim all-fresh PASS."""
        unavailable = [
            "behavioral_sentiment",
            "calendar_seasonality",
            "crypto_allocation",
            "factor_rotation",
            "stacking_ensemble",
            "kurtosis_regime",
            "collar",
            "bond_momentum",
            "risk_decomposition",
            "two_stage_regime",
            "regime_transition",
            "fred_macro",
        ]
        staleness = {
            "stale_signals": [],
            "unavailable_signals": unavailable,
            "healthy_count": 11,
            "total_count": 23,
            "required_count": 5,
            "optional_count": 18,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == AlertChannel.SIGNAL_STALENESS
        assert call_args[0][1] == AlertLevel.WARN
        message = call_args[0][2]
        assert "fresh" not in message.lower() or "partial" in message.lower() or "unavailable" in message.lower()
        assert "All 23 signals fresh" not in message
        assert "unavailable" in message.lower()
        details = call_args[1].get("details") or (call_args[0][3] if len(call_args[0]) > 3 else None)
        # kwargs preferred
        if call_args.kwargs:
            details = call_args.kwargs.get("details", details)
        assert details is not None
        assert details.get("unavailable_count") == 12
        assert details.get("unavailable_signals") == unavailable

    @patch("src.monitor.alerting.send_alert")
    def test_unavailable_count_int_without_list(self, mock_send):
        """Support unavailable_signals as a count when list is omitted."""
        staleness = {
            "stale_signals": [],
            "unavailable_signals": 12,
            "healthy_count": 11,
            "total_count": 23,
        }
        check_staleness_and_alert(staleness)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN
        assert "All 23 signals fresh" not in mock_send.call_args[0][2]


class TestCheckDriftAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_low_drift_pass(self, mock_send):
        check_drift_and_alert(3.2)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_moderate_drift_warn(self, mock_send):
        check_drift_and_alert(7.5)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN

    @patch("src.monitor.alerting.send_alert")
    def test_high_drift_halt(self, mock_send):
        check_drift_and_alert(12.0)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.HALT

    @patch("src.monitor.alerting.send_alert")
    def test_negative_drift_warn(self, mock_send):
        check_drift_and_alert(-6.0)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.WARN


class TestCheckIcDecayAndAlert:
    @patch("src.monitor.alerting.send_alert")
    def test_all_healthy_sends_pass(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "healthy", "ic_rolling": 0.15, "ic_trend": "stable"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.12, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.PASS

    @patch("src.monitor.alerting.send_alert")
    def test_warning_signal_sends_warn(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "warning", "ic_rolling": 0.07, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.12, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.WARN
        assert "alternative_data" in mock_send.call_args[0][2]

    @patch("src.monitor.alerting.send_alert")
    def test_critical_signal_sends_halt(self, mock_send):
        ic_decay = {
            "alternative_data": {"status": "critical", "ic_rolling": 0.02, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "warning", "ic_rolling": 0.07, "ic_trend": "decaying"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == AlertChannel.IC_DECAY
        assert mock_send.call_args[0][1] == AlertLevel.HALT
        assert "alternative_data" in mock_send.call_args[0][2]

    @patch("src.monitor.alerting.send_alert")
    def test_empty_data_no_alert(self, mock_send):
        check_ic_decay_and_alert({})
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_error_key_no_alert(self, mock_send):
        check_ic_decay_and_alert({"error": "monitor unavailable"})
        mock_send.assert_not_called()

    @patch("src.monitor.alerting.send_alert")
    def test_mixed_warning_and_critical_sends_halt(self, mock_send):
        """When both warning and critical signals exist, alert is HALT."""
        ic_decay = {
            "multi_speed_mom": {"status": "critical", "ic_rolling": 0.01, "ic_trend": "decaying"},
            "alternative_data": {"status": "warning", "ic_rolling": 0.08, "ic_trend": "decaying"},
            "cross_asset_rv": {"status": "healthy", "ic_rolling": 0.15, "ic_trend": "stable"},
        }
        check_ic_decay_and_alert(ic_decay)
        mock_send.assert_called_once()
        assert mock_send.call_args[0][1] == AlertLevel.HALT
        # Details should include both critical and warning signals
        details = mock_send.call_args[1].get("details") or mock_send.call_args[0][3] if len(mock_send.call_args[0]) > 3 else None
        # The function passes details as keyword arg
