"""
Market calendar utilities for portfolio-lab.
Handles NYSE trading days, market hours, and stale data detection.
"""

from datetime import datetime, timedelta
from typing import Optional, List

try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CAL = True
except ImportError:
    HAS_MARKET_CAL = False

class MarketCalendar:
    """NYSE market calendar wrapper."""
    
    def __init__(self):
        if HAS_MARKET_CAL:
            self.nyse = mcal.get_calendar('NYSE')
        else:
            self.nyse = None
    
    def is_trading_day(self, date: Optional[datetime] = None) -> bool:
        """Check if date is a NYSE trading day."""
        if date is None:
            date = datetime.now()
        
        if not HAS_MARKET_CAL:
            # Fallback: weekdays only, no holidays
            return date.weekday() < 5  # Monday=0, Friday=4
        
        try:
            schedule = self.nyse.schedule(start_date=date.strftime('%Y-%m-%d'), 
                                          end_date=date.strftime('%Y-%m-%d'))
            return len(schedule) > 0
        except:
            return date.weekday() < 5
    
    def is_market_open(self, dt: Optional[datetime] = None, tz='America/New_York') -> bool:
        """Check if market is currently open (9:30 AM - 4:00 PM ET)."""
        if dt is None:
            dt = datetime.now()
        
        if not self.is_trading_day(dt):
            return False
        
        # Simple hour check (9:30-16:00 ET)
        # For proper timezone handling, use pytz or zoneinfo
        hour = dt.hour
        minute = dt.minute
        
        # 9:30 = 9.5 hours, 16:00 = 16 hours
        market_time = hour + minute / 60
        return 9.5 <= market_time < 16.0
    
    def trading_days_since(self, last_date: datetime, today: Optional[datetime] = None) -> int:
        """Count actual trading days between last_date and today."""
        if today is None:
            today = datetime.now()
        
        if not HAS_MARKET_CAL:
            # Fallback: count weekdays
            count = 0
            current = last_date + timedelta(days=1)
            while current <= today:
                if current.weekday() < 5:
                    count += 1
                current += timedelta(days=1)
            return count
        
        try:
            schedule = self.nyse.schedule(start_date=last_date.strftime('%Y-%m-%d'),
                                          end_date=today.strftime('%Y-%m-%d'))
            # Don't count today if market hasn't closed yet
            if self.is_trading_day(today) and not self.is_market_closed_for_day(today):
                return max(0, len(schedule) - 1)
            return len(schedule)
        except:
            # Fallback
            return (today - last_date).days
    
    def is_market_closed_for_day(self, dt: Optional[datetime] = None) -> bool:
        """Check if market has closed for the day (after 4 PM)."""
        if dt is None:
            dt = datetime.now()
        return dt.hour >= 16
    
    def next_trading_day(self, from_date: Optional[datetime] = None) -> Optional[datetime]:
        """Find the next trading day after from_date."""
        if from_date is None:
            from_date = datetime.now()
        
        for i in range(1, 10):
            next_day = from_date + timedelta(days=i)
            if self.is_trading_day(next_day):
                return next_day
        return None
    
    def get_market_status(self, dt: Optional[datetime] = None) -> dict:
        """Get full market status summary."""
        if dt is None:
            dt = datetime.now()
        
        is_trading = self.is_trading_day(dt)
        is_open = self.is_market_open(dt) if is_trading else False
        
        status = {
            "is_trading_day": is_trading,
            "is_market_open": is_open,
            "market_time": None,
            "next_open": None,
            "last_close": None
        }
        
        if is_trading:
            if is_open:
                status["market_time"] = "open"
            elif dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
                status["market_time"] = "pre-market"
                status["next_open"] = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                status["market_time"] = "closed"
                next_day = self.next_trading_day(dt)
                if next_day:
                    status["next_open"] = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            status["market_time"] = "holiday/weekend"
            next_day = self.next_trading_day(dt)
            if next_day:
                status["next_open"] = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return status
    
    def should_expect_fresh_data(self, last_update: datetime, current: Optional[datetime] = None) -> bool:
        """Determine if data should be considered stale based on trading days."""
        if current is None:
            current = datetime.now()
        
        # If market closed, don't expect new data
        if not self.is_trading_day(current):
            return False
        
        # If market open but hasn't closed yet today, allow one trading day gap
        if self.is_trading_day(current) and not self.is_market_closed_for_day(current):
            # If last update was yesterday, that's ok
            trading_days = self.trading_days_since(last_update, current)
            return trading_days > 1
        
        # After market close, expect today's data
        trading_days = self.trading_days_since(last_update, current)
        return trading_days > 0


# Convenience functions for common use cases
def get_stale_status(last_update_str: str, current: Optional[datetime] = None) -> str:
    """
    Get stale status considering market calendar.
    Returns: 'fresh', 'stale', or 'critical'
    """
    if current is None:
        current = datetime.now()
    
    try:
        last_update = datetime.strptime(last_update_str, "%Y-%m-%d")
    except:
        return "unknown"
    
    calendar = MarketCalendar()
    
    # If not a trading day, data is fresh (market closed)
    if not calendar.is_trading_day(current):
        return "fresh"
    
    trading_days = calendar.trading_days_since(last_update, current)
    
    if trading_days == 0:
        return "fresh"
    elif trading_days == 1:
        return "fresh"  # Normal overnight gap
    elif trading_days <= 3:
        return "stale"
    else:
        return "critical"


if __name__ == "__main__":
    # Test the market calendar
    cal = MarketCalendar()
    now = datetime.now()
    
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Is trading day: {cal.is_trading_day(now)}")
    print(f"Is market open: {cal.is_market_open(now)}")
    print(f"Market status: {cal.get_market_status(now)}")
    
    # Test stale detection
    test_dates = [
        (now - timedelta(days=1), "Yesterday"),
        (now - timedelta(days=3), "3 days ago"),
        (now - timedelta(days=7), "7 days ago"),
    ]
    
    for date, label in test_dates:
        status = get_stale_status(date.strftime("%Y-%m-%d"))
        print(f"{label}: {status}")
