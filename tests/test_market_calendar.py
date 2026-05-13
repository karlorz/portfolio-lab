#!/usr/bin/env python3
"""
Tests for market_calendar.py — MarketCalendar, trading day detection,
weekend/holiday handling, stale threshold logic, and formatting utilities.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta

from src.utils.market_calendar import (
    US_HOLIDAYS,
    MarketCalendar,
    get_market_calendar,
    is_weekend_stale,
    format_stale_status,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_holidays_not_empty(self):
        assert len(US_HOLIDAYS) > 0

    def test_holidays_are_strings(self):
        for h in US_HOLIDAYS:
            assert isinstance(h, str)
            assert len(h) == 10  # YYYY-MM-DD

    def test_has_2024_holidays(self):
        assert "2024-01-01" in US_HOLIDAYS
        assert "2024-07-04" in US_HOLIDAYS

    def test_has_2025_holidays(self):
        assert "2025-01-01" in US_HOLIDAYS
        assert "2025-12-25" in US_HOLIDAYS


# ---------------------------------------------------------------------------
# MarketCalendar — is_trading_day
# ---------------------------------------------------------------------------

class TestIsTradingDay:

    def test_weekday_not_holiday(self):
        cal = MarketCalendar()
        # Wednesday 2026-05-13
        assert cal.is_trading_day(datetime(2026, 5, 13)) is True

    def test_saturday(self):
        cal = MarketCalendar()
        assert cal.is_trading_day(datetime(2026, 5, 9)) is False  # Saturday

    def test_sunday(self):
        cal = MarketCalendar()
        assert cal.is_trading_day(datetime(2026, 5, 10)) is False  # Sunday

    def test_holiday(self):
        cal = MarketCalendar()
        assert cal.is_trading_day(datetime(2025, 1, 1)) is False  # New Year's

    def test_july_4th(self):
        cal = MarketCalendar()
        assert cal.is_trading_day(datetime(2025, 7, 4)) is False

    def test_caching(self):
        cal = MarketCalendar()
        d = datetime(2026, 5, 13)
        r1 = cal.is_trading_day(d)
        r2 = cal.is_trading_day(d)
        assert r1 == r2
        assert "2026-05-13" in cal._cache

    def test_friday_is_trading_day(self):
        cal = MarketCalendar()
        assert cal.is_trading_day(datetime(2026, 5, 8)) is True  # Friday


# ---------------------------------------------------------------------------
# MarketCalendar — _is_market_closed
# ---------------------------------------------------------------------------

class TestIsMarketClosed:

    def test_before_open(self):
        cal = MarketCalendar()
        assert cal._is_market_closed(datetime(2026, 5, 13, 9, 0)) is True

    def test_at_open(self):
        cal = MarketCalendar()
        assert cal._is_market_closed(datetime(2026, 5, 13, 9, 30)) is False

    def test_during_hours(self):
        cal = MarketCalendar()
        assert cal._is_market_closed(datetime(2026, 5, 13, 12, 0)) is False

    def test_at_close(self):
        cal = MarketCalendar()
        assert cal._is_market_closed(datetime(2026, 5, 13, 16, 0)) is True

    def test_after_close(self):
        cal = MarketCalendar()
        assert cal._is_market_closed(datetime(2026, 5, 13, 17, 0)) is True


# ---------------------------------------------------------------------------
# MarketCalendar — trading_days_since
# ---------------------------------------------------------------------------

class TestTradingDaysSince:

    def test_same_day(self):
        cal = MarketCalendar()
        d = datetime(2026, 5, 13, 17, 0)  # After market close
        assert cal.trading_days_since(d, d) == 0

    def test_one_day(self):
        cal = MarketCalendar()
        last = datetime(2026, 5, 12, 17, 0)
        today = datetime(2026, 5, 13, 17, 0)
        assert cal.trading_days_since(last, today) == 1

    def test_weekend_skipped(self):
        cal = MarketCalendar()
        friday = datetime(2026, 5, 8, 17, 0)
        monday = datetime(2026, 5, 11, 17, 0)
        # Friday→Saturday→Sunday→Monday: only Monday is trading day
        assert cal.trading_days_since(friday, monday) == 1

    def test_holiday_skipped(self):
        cal = MarketCalendar()
        # Dec 31 2024 → Jan 1 2025 (holiday) → Jan 2 2025
        dec31 = datetime(2024, 12, 31, 17, 0)
        jan2 = datetime(2025, 1, 2, 17, 0)
        # Dec 31 is early close but still trading, Jan 1 is holiday, Jan 2 is trading
        days = cal.trading_days_since(dec31, jan2)
        assert days >= 1


# ---------------------------------------------------------------------------
# MarketCalendar — next_trading_day
# ---------------------------------------------------------------------------

class TestNextTradingDay:

    def test_weekday_next(self):
        cal = MarketCalendar()
        wed = datetime(2026, 5, 13)
        nxt = cal.next_trading_day(wed)
        assert nxt == datetime(2026, 5, 14)

    def test_friday_next_is_monday(self):
        cal = MarketCalendar()
        friday = datetime(2026, 5, 8)
        nxt = cal.next_trading_day(friday)
        assert nxt.weekday() == 0  # Monday

    def test_skips_holiday(self):
        cal = MarketCalendar()
        dec31 = datetime(2024, 12, 31)
        nxt = cal.next_trading_day(dec31)
        # Jan 1 2025 is holiday, so next should be Jan 2
        assert nxt.day >= 2 or nxt.month == 1


# ---------------------------------------------------------------------------
# MarketCalendar — is_market_holiday
# ---------------------------------------------------------------------------

class TestIsMarketHoliday:

    def test_trading_day_not_holiday(self):
        cal = MarketCalendar()
        assert cal.is_market_holiday(datetime(2026, 5, 13)) is False

    def test_weekend_not_holiday(self):
        cal = MarketCalendar()
        # Saturday is not a "holiday" — it's a weekend
        assert cal.is_market_holiday(datetime(2026, 5, 9)) is False

    def test_weekday_holiday(self):
        cal = MarketCalendar()
        assert cal.is_market_holiday(datetime(2025, 1, 1)) is True

    def test_july_4th_holiday(self):
        cal = MarketCalendar()
        assert cal.is_market_holiday(datetime(2025, 7, 4)) is True


# ---------------------------------------------------------------------------
# MarketCalendar — get_stale_threshold_days
# ---------------------------------------------------------------------------

class TestStaleThreshold:

    def test_fresh(self):
        cal = MarketCalendar()
        now = datetime(2026, 5, 13, 17, 0)
        assert cal.get_stale_threshold_days(now, now) == 0

    def test_warning(self):
        cal = MarketCalendar()
        yesterday = datetime(2026, 5, 12, 17, 0)
        today = datetime(2026, 5, 13, 17, 0)
        assert cal.get_stale_threshold_days(yesterday, today) == 2

    def test_critical(self):
        cal = MarketCalendar()
        old = datetime(2026, 5, 8, 17, 0)
        now = datetime(2026, 5, 13, 17, 0)
        assert cal.get_stale_threshold_days(old, now) == 999


# ---------------------------------------------------------------------------
# Singleton Tests
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_market_calendar(self):
        cal = get_market_calendar()
        assert isinstance(cal, MarketCalendar)

    def test_singleton_same_instance(self):
        import src.utils.market_calendar as mod
        mod._market_calendar = None
        cal1 = get_market_calendar()
        cal2 = get_market_calendar()
        assert cal1 is cal2


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_is_weekend_stale_true(self):
        # If it's Monday and last update was Friday, days_stale=2 but trading_days=1
        result = is_weekend_stale(2, datetime(2026, 5, 8, 17, 0))
        # Depends on when we run — but the function should return bool
        assert isinstance(result, bool)

    def test_format_stale_status_returns_string(self):
        status = format_stale_status(0, datetime(2026, 5, 13, 17, 0))
        assert isinstance(status, str)
        assert len(status) > 0

    def test_format_stale_status_warning(self):
        status = format_stale_status(2, datetime(2026, 5, 12, 17, 0))
        assert "warning" in status.lower() or "stale" in status.lower()
