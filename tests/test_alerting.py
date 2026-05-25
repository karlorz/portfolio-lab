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
    _last_alert_time,
    ALERT_MIN_INTERVAL_SECONDS,
)


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
