#!/usr/bin/env python3
"""
Tests for notifier.py — Alert dataclass, fingerprinting, rate limiting,
deduplication, state persistence, log management, and notification routing.
"""
import sys
import os
import json
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.utils.notifier import (
    Alert,
    NotificationManager,
    MIN_INTERVAL_SECONDS,
    MAX_ALERTS_PER_HOUR,
    ALERT_LOG,
    get_notifier,
    send_alert,
    notify_critical,
    notify_warning,
    notify_success,
    notify_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(**overrides):
    defaults = dict(
        level="warning",
        type="system",
        title="Test Alert",
        message="This is a test alert message",
        timestamp=datetime.now().isoformat(),
        requires_action=False,
        metadata=None,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def _make_notifier(tmp_path):
    """Create a NotificationManager with temp paths."""
    import src.utils.notifier as notifier_mod
    old_alert_log = notifier_mod.ALERT_LOG
    old_state = notifier_mod.NOTIFIER_STATE
    notifier_mod.ALERT_LOG = tmp_path / "alerts.jsonl"
    notifier_mod.NOTIFIER_STATE = tmp_path / ".notifier_state.json"
    mgr = NotificationManager()
    yield mgr
    notifier_mod.ALERT_LOG = old_alert_log
    notifier_mod.NOTIFIER_STATE = old_state


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_rate_limits(self):
        assert MIN_INTERVAL_SECONDS == 300
        assert MAX_ALERTS_PER_HOUR == 10


# ---------------------------------------------------------------------------
# Alert Dataclass Tests
# ---------------------------------------------------------------------------

class TestAlert:

    def test_fields(self):
        a = _make_alert(level="error", type="kill_switch", title="Kill!")
        assert a.level == "error"
        assert a.type == "kill_switch"
        assert a.title == "Kill!"

    def test_to_dict(self):
        a = _make_alert()
        d = a.to_dict()
        assert "level" in d
        assert "type" in d
        assert "title" in d
        assert "metadata" in d

    def test_to_dict_metadata_default(self):
        a = _make_alert(metadata=None)
        d = a.to_dict()
        assert d["metadata"] == {}

    def test_to_dict_metadata_preserved(self):
        a = _make_alert(metadata={"key": "value"})
        d = a.to_dict()
        assert d["metadata"]["key"] == "value"

    def test_fingerprint_deterministic(self):
        a = _make_alert()
        assert a.fingerprint() == a.fingerprint()

    def test_fingerprint_differs_by_content(self):
        a1 = _make_alert(title="Alert A")
        a2 = _make_alert(title="Alert B")
        assert a1.fingerprint() != a2.fingerprint()

    def test_fingerprint_length(self):
        a = _make_alert()
        assert len(a.fingerprint()) == 12

    def test_requires_action_default(self):
        a = _make_alert()
        assert a.requires_action is False


# ---------------------------------------------------------------------------
# NotificationManager — rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_allows_first_alert(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()
        assert mgr._check_rate_limit(a) is True

    def test_blocks_duplicate_within_window(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()
        mgr._record_sent(a)
        assert mgr._check_rate_limit(a) is False

    def test_allows_duplicate_after_window(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()
        mgr._record_sent(a)
        # Simulate time passing
        mgr.recent_alerts[a.fingerprint()] = datetime.now() - timedelta(seconds=MIN_INTERVAL_SECONDS + 1)
        assert mgr._check_rate_limit(a) is True

    def test_hourly_limit(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        mgr.hourly_count = MAX_ALERTS_PER_HOUR
        a = _make_alert()
        assert mgr._check_rate_limit(a) is False

    def test_hourly_reset(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        mgr.hourly_count = MAX_ALERTS_PER_HOUR
        mgr.hourly_reset = datetime.now() - timedelta(hours=2)
        a = _make_alert()
        assert mgr._check_rate_limit(a) is True

    def test_different_alerts_not_deduplicated(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a1 = _make_alert(title="Alert A")
        a2 = _make_alert(title="Alert B")
        mgr._record_sent(a1)
        assert mgr._check_rate_limit(a2) is True


# ---------------------------------------------------------------------------
# NotificationManager — record_sent
# ---------------------------------------------------------------------------

class TestRecordSent:

    def test_records_fingerprint(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()
        mgr._record_sent(a)
        assert a.fingerprint() in mgr.recent_alerts

    def test_increments_hourly_count(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        assert mgr.hourly_count == 0
        mgr._record_sent(_make_alert())
        assert mgr.hourly_count == 1


# ---------------------------------------------------------------------------
# NotificationManager — log management
# ---------------------------------------------------------------------------

class TestLogManagement:

    def test_log_creates_file(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        import src.utils.notifier as notifier_mod
        a = _make_alert()
        mgr._log_alert(a, sent=False)
        assert notifier_mod.ALERT_LOG.exists()

    def test_log_entry_format(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        import src.utils.notifier as notifier_mod
        a = _make_alert()
        mgr._log_alert(a, sent=True)
        with open(notifier_mod.ALERT_LOG) as f:
            entry = json.loads(f.readline())
        assert entry["level"] == "warning"
        assert entry["sent"] is True

    def test_get_recent_alerts_empty(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        alerts = mgr.get_recent_alerts(hours=24)
        assert alerts == []

    def test_get_recent_alerts_with_data(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        import src.utils.notifier as notifier_mod
        a = _make_alert()
        mgr._log_alert(a, sent=False)
        alerts = mgr.get_recent_alerts(hours=1)
        assert len(alerts) == 1

    def test_get_recent_alerts_filters_level(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        import src.utils.notifier as notifier_mod
        mgr._log_alert(_make_alert(level="error"), sent=False)
        mgr._log_alert(_make_alert(level="warning"), sent=False)
        errors = mgr.get_recent_alerts(hours=1, level="error")
        assert len(errors) == 1
        assert errors[0]["level"] == "error"


# ---------------------------------------------------------------------------
# NotificationManager — notify (async)
# ---------------------------------------------------------------------------

class TestNotify:

    def test_notify_logs_alert(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()

        async def run():
            with patch.object(mgr, 'send_telegram', new_callable=AsyncMock, return_value=False):
                return await mgr.notify(a, channels=['log'])

        result = asyncio.run(run())
        assert result['log'] is True

    def test_notify_telegram_channel(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)
        a = _make_alert()

        async def run():
            with patch.object(mgr, 'send_telegram', new_callable=AsyncMock, return_value=True) as mock:
                return await mgr.notify(a, channels=['telegram', 'log'])

        result = asyncio.run(run())
        assert result['telegram'] is True
        assert result['log'] is True


# ---------------------------------------------------------------------------
# send_alert convenience function
# ---------------------------------------------------------------------------

class TestSendAlert:

    def test_send_alert_creates_alert(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)

        async def run():
            with patch('src.utils.notifier.get_notifier', return_value=mgr):
                with patch.object(mgr, 'notify', new_callable=AsyncMock, return_value={'log': True}) as mock:
                    result = await send_alert('info', 'system', 'Test', 'Message')
                    assert mock.called
                    alert = mock.call_args[0][0]
                    assert alert.level == 'info'
                    assert alert.title == 'Test'
                    return result

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

class TestSeverityHelpers:

    def test_notify_critical(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)

        async def run():
            with patch('src.utils.notifier.get_notifier', return_value=mgr):
                with patch.object(mgr, 'notify', new_callable=AsyncMock, return_value={'log': True}) as mock:
                    await notify_critical("Title", "Message", type="kill_switch")
                    alert = mock.call_args[0][0]
                    assert alert.level == 'error'
                    assert alert.requires_action is True

        asyncio.run(run())

    def test_notify_warning(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)

        async def run():
            with patch('src.utils.notifier.get_notifier', return_value=mgr):
                with patch.object(mgr, 'notify', new_callable=AsyncMock, return_value={'log': True}) as mock:
                    await notify_warning("Title", "Message")
                    alert = mock.call_args[0][0]
                    assert alert.level == 'warning'
                    assert alert.requires_action is False

        asyncio.run(run())

    def test_notify_success(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)

        async def run():
            with patch('src.utils.notifier.get_notifier', return_value=mgr):
                with patch.object(mgr, 'notify', new_callable=AsyncMock, return_value={'log': True}) as mock:
                    await notify_success("Title", "Message")
                    alert = mock.call_args[0][0]
                    assert alert.level == 'success'

        asyncio.run(run())

    def test_notify_info(self, tmp_path):
        gen = _make_notifier(tmp_path)
        mgr = next(gen)

        async def run():
            with patch('src.utils.notifier.get_notifier', return_value=mgr):
                with patch.object(mgr, 'notify', new_callable=AsyncMock, return_value={'log': True}) as mock:
                    await notify_info("Title", "Message")
                    alert = mock.call_args[0][0]
                    assert alert.level == 'info'

        asyncio.run(run())
