"""Market calendar utilities for trading day calculations.

Falls back to manual calendar if pandas_market_calendars not available.
"""
from datetime import datetime, timedelta
from typing import Optional

# Try to import pandas_market_calendars, but provide fallback
try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except ImportError:
    HAS_MCAL = False

# US Market holidays 2024-2026 (simplified)
US_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2024-12-31",  # Early close
    # 2025
    "2025-01-01", "2025-01-09", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-01",
}


class MarketCalendar:
    """NYSE market calendar for trading day calculations."""
    
    def __init__(self):
        self._cache = {}
        self._has_mcal = HAS_MCAL
        if self._has_mcal:
            try:
                self.nyse = mcal.get_calendar('NYSE')
            except Exception:
                self.nyse = None
                self._has_mcal = False
        else:
            self.nyse = None
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if date is a trading day."""
        date_key = date.strftime('%Y-%m-%d')
        if date_key in self._cache:
            return self._cache[date_key]
        
        # Weekend check
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            self._cache[date_key] = False
            return False
        
        # Holiday check
        if date_key in US_HOLIDAYS:
            self._cache[date_key] = False
            return False
        
        result = True
        self._cache[date_key] = result
        return result
    
    def trading_days_since(self, last_date: datetime, today: datetime = None) -> int:
        """Count trading days between last_date and today."""
        today = today or datetime.now()
        delta = (today.date() - last_date.date()).days
        
        count = 0
        for i in range(1, delta + 1):
            check_date = last_date + timedelta(days=i)
            if self.is_trading_day(check_date):
                count += 1
        
        # If today is trading day but market hasn't closed, don't count today
        if self.is_trading_day(today) and not self._is_market_closed(today):
            count = max(0, count - 1)
        
        return count
    
    def _is_market_closed(self, dt: datetime) -> bool:
        """Check if market is closed at given datetime."""
        # Simplified check: market hours 9:30-16:00 ET
        hour = dt.hour
        minute = dt.minute
        current_time = hour * 100 + minute
        return current_time < 930 or current_time >= 1600
    
    def next_trading_day(self, from_date: datetime = None) -> datetime:
        """Get next trading day."""
        from_date = from_date or datetime.now()
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def is_market_holiday(self, date: datetime) -> bool:
        """Check if date is a market holiday."""
        if self.is_trading_day(date):
            return False
        # Check if it's a weekday (potential holiday)
        return date.weekday() < 5
    
    def get_stale_threshold_days(self, last_update: datetime, now: datetime = None) -> int:
        """Calculate calendar days that equal stale threshold."""
        now = now or datetime.now()
        trading_days = self.trading_days_since(last_update, now)
        
        # Stale thresholds in trading days:
        # - warning: 1 trading day
        # - critical: 2 trading days
        if trading_days >= 2:
            return 999  # Critical
        elif trading_days >= 1:
            return 2   # Warning
        return 0  # Fresh


# Global instance
_market_calendar = None


def get_market_calendar() -> MarketCalendar:
    """Get singleton market calendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar()
    return _market_calendar


def is_weekend_stale(days_stale: int, last_update: datetime) -> bool:
    """Check if stale days are due to weekend/holiday."""
    cal = get_market_calendar()
    trading_days = cal.trading_days_since(last_update)
    # If trading days is 0, it's just weekend/holiday
    return trading_days == 0 and days_stale > 0


def format_stale_status(days_stale: int, last_update: datetime) -> str:
    """Format stale status considering market calendar."""
    cal = get_market_calendar()
    trading_days = cal.trading_days_since(last_update)
    
    if trading_days == 0:
        if days_stale <= 3:
            return "fresh (market closed)"
        else:
            return f"stale: {days_stale}d ({trading_days} trading days)"
    elif trading_days == 1:
        return f"warning: {days_stale}d ({trading_days} trading day)"
    else:
        return f"stale: {days_stale}d ({trading_days} trading days)"


if __name__ == "__main__":
    # Test
    cal = MarketCalendar()
    today = datetime.now()
    print(f"Today is trading day: {cal.is_trading_day(today)}")
    print(f"Next trading day: {cal.next_trading_day(today)}")
    
    # Test weekend handling
    friday = datetime(2026, 5, 8)
    monday = datetime(2026, 5, 11)
    print(f"Friday {friday.date()} is trading day: {cal.is_trading_day(friday)}")
    print(f"Saturday (weekend) is trading day: {cal.is_trading_day(friday + timedelta(days=1))}")
    print(f"Trading days since Friday to Monday: {cal.trading_days_since(friday, monday)}")
