"""
Calendar Seasonality Signal Generator - v3.50 Implementation
Calendar-based seasonality filters for rebalancing execution timing.

Detects well-documented market anomalies to adjust rebalancing urgency:
- Turn-of-Month (TOM): Last trading day + first 3 days (+0.5% excess returns)
- Pre-Holiday: Day before major US market holidays (5-10x normal returns)
- Quarter-End: Last week of quarter (window dressing)
- Monday Effect: Historically weakest day
- Pre-FOMC: Day before Fed announcements (drift before decisions)
- Tax-Loss: December harvesting pressure

Expected impact: +0.01-0.02 Sharpe through 5-15 bps better execution annually.

Usage:
    python -m src.signals.calendar_seasonality check
    python -m src.signals.calendar_seasonality calendar 2026-01
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalendarWindow(Enum):
    TOM = "tom_window"              # Turn-of-Month
    PRE_HOLIDAY = "pre_holiday"     # Day before holiday
    POST_HOLIDAY = "post_holiday"   # Day after holiday
    QUARTER_END = "quarter_end"      # Last week of quarter
    MONDAY = "monday"                # Monday effect
    PRE_FOMC = "pre_fomc"           # Day before FOMC
    DECEMBER = "december"            # Tax-loss harvesting
    OPTIONS_EXPIRY = "options_expiry"  # Monthly OPEX


class SeasonalityEffect(Enum):
    POSITIVE = "positive"     # Favorable for execution
    NEUTRAL = "neutral"       # No effect
    NEGATIVE = "negative"     # Unfavorable — delay if possible
    AVOID = "avoid"           # Strongly unfavorable — defer


@dataclass
class CalendarSeasonalitySignal:
    """Complete calendar seasonality assessment for a given date."""
    assessment_date: str
    day_of_week: str
    is_trading_day: bool

    # Active windows
    active_windows: List[str]

    # Urgency modifier (0.0-1.0, multiplicative)
    urgency_modifier: float

    # Individual modifiers
    tom_modifier: float
    holiday_modifier: float
    quarter_end_modifier: float
    monday_modifier: float
    fomc_modifier: float
    december_modifier: float
    opex_modifier: float

    # Execution recommendation
    recommendation: str  # proceed, delay, wait, avoid
    effect: str          # positive, neutral, negative, avoid

    # Next upcoming window
    next_window: str
    next_window_date: str
    days_to_next_window: int

    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class NYSECalendar:
    """
    NYSE trading calendar with holiday detection.

    Covers major US market holidays. Uses NYSE observed dates
    (not necessarily the same as federal holidays).
    """

    # Fixed-date holidays (month, day)
    FIXED_HOLIDAYS = [
        (1, 1),     # New Year's Day
        (7, 4),     # Independence Day
        (12, 25),   # Christmas
    ]

    # Floating holidays (month, weekday rule) — week number, weekday (Mon=0)
    # Format: (month, week_number, weekday, name)
    # week_number: -1 = last occurrence in month
    FLOATING_HOLIDAYS = [
        (1, 3, 0, "MLK Day"),           # 3rd Monday of January
        (2, 3, 0, "Presidents Day"),    # 3rd Monday of February
        (5, -1, 0, "Memorial Day"),     # Last Monday of May
        (6, 19, None, "Juneteenth"),     # June 19 (fixed date)
        (9, 1, 0, "Labor Day"),         # 1st Monday of September
        (11, 4, 3, "Thanksgiving"),     # 4th Thursday of November
    ]

    # Good Friday: Friday before Easter (approximate — March 20 to April 23)
    # We use a simplified calculation

    def __init__(self, year: Optional[int] = None):
        self.year = year or date.today().year
        self._holidays: Optional[Set[date]] = None

    def _compute_easter(self, year: int) -> date:
        """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def _compute_holidays(self) -> Set[date]:
        """Compute all NYSE holidays for the year."""
        holidays: Set[date] = set()
        y = self.year

        # Fixed-date holidays
        for month, day in self.FIXED_HOLIDAYS:
            d = self._adjust_to_trading_day(date(y, month, day), "observed")
            holidays.add(d)

        # Juneteenth
        d = self._adjust_to_trading_day(date(y, 6, 19), "observed")
        holidays.add(d)

        # MLK Day: 3rd Monday of January
        holidays.add(self._nth_weekday(y, 1, 0, 3))

        # Presidents Day: 3rd Monday of February
        holidays.add(self._nth_weekday(y, 2, 0, 3))

        # Memorial Day: Last Monday of May
        holidays.add(self._nth_weekday(y, 5, 0, -1))

        # Labor Day: 1st Monday of September
        holidays.add(self._nth_weekday(y, 9, 0, 1))

        # Thanksgiving: 4th Thursday of November
        holidays.add(self._nth_weekday(y, 11, 3, 4))

        # Christmas — already in FIXED_HOLIDAYS

        # Good Friday (Friday before Easter)
        easter = self._compute_easter(y)
        good_friday = easter - timedelta(days=2)
        holidays.add(good_friday)

        return holidays

    @property
    def holidays(self) -> Set[date]:
        if self._holidays is None:
            self._holidays = self._compute_holidays()
        return self._holidays

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth occurrence of a weekday in a month.
        weekday: 0=Monday...6=Sunday. n: 1=first, -1=last."""
        if n > 0:
            first = date(year, month, 1)
            delta = (weekday - first.weekday()) % 7
            return first + timedelta(days=delta + (n - 1) * 7)
        else:
            # Last occurrence
            if month == 12:
                last = date(year, 12, 31)
            else:
                last = date(year, month + 1, 1) - timedelta(days=1)
            delta = (last.weekday() - weekday) % 7
            return last - timedelta(days=delta)

    @staticmethod
    def _adjust_to_trading_day(d: date, rule: str = "observed") -> date:
        """Adjust holiday to NYSE observed date.
        Saturday → Friday before, Sunday → Monday after."""
        if d.weekday() == 5:  # Saturday
            return d - timedelta(days=1)
        elif d.weekday() == 6:  # Sunday
            return d + timedelta(days=1)
        return d

    def is_holiday(self, d: date) -> bool:
        return d in self.holidays

    def is_trading_day(self, d: date) -> bool:
        """Check if a date is a trading day (Mon-Fri, not a holiday)."""
        if d.weekday() >= 5:  # Saturday/Sunday
            return False
        if d in self.holidays:
            return False
        return True

    def next_trading_day(self, d: date) -> date:
        """Find the next trading day after d."""
        d = d + timedelta(days=1)
        while not self.is_trading_day(d):
            d = d + timedelta(days=1)
        return d

    def previous_trading_day(self, d: date) -> date:
        """Find the previous trading day before d."""
        d = d - timedelta(days=1)
        while not self.is_trading_day(d):
            d = d - timedelta(days=1)
        return d

    def trading_days_between(self, start: date, end: date) -> List[date]:
        """Get all trading days in a date range."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days


class CalendarSeasonalityDetector:
    """
    Detects calendar-based seasonality windows for rebalancing timing.

    Urgency modifiers (multiplicative, 0.0-1.0):
    - TOM window: 0.70 (favorable, don't delay during TOM)
    - Pre-holiday: 0.50 (strongly favorable)
    - Quarter-end: 0.60 (moderately favorable)
    - Monday: 0.80 (slightly unfavorable — delay if low urgency)
    - Pre-FOMC: 0.65 (caution — wait for announcement)
    - December: 0.75 (tax-loss pressure)
    - OPEX: 0.85 (slight caution)

    Lower modifier = lower urgency = more likely to defer.
    """

    # Urgency modifiers for each window
    URGENCY_MODIFIERS = {
        CalendarWindow.TOM: 0.70,
        CalendarWindow.PRE_HOLIDAY: 0.50,
        CalendarWindow.POST_HOLIDAY: 0.90,
        CalendarWindow.QUARTER_END: 0.60,
        CalendarWindow.MONDAY: 0.80,
        CalendarWindow.PRE_FOMC: 0.65,
        CalendarWindow.DECEMBER: 0.75,
        CalendarWindow.OPTIONS_EXPIRY: 0.85,
    }

    # FOMC meeting dates (2026 — update annually)
    # Standard 8-meeting schedule: roughly every 6-7 weeks
    FOMC_2026_DATES = [
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
        date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 11, 4), date(2026, 12, 16),
    ]

    def __init__(self, year: Optional[int] = None):
        self.calendar = NYSECalendar(year=year)

    def get_urgency_modifier(self, d: Optional[date] = None) -> float:
        """
        Return the composite urgency modifier for a given date.
        Uses the minimum (most conservative) modifier across all active windows.
        """
        if d is None:
            d = date.today()

        modifiers = []
        windows = self._detect_windows(d)

        for window in windows:
            modifiers.append(self.URGENCY_MODIFIERS[window])

        if not modifiers:
            return 1.0

        # Use minimum modifier (most conservative / most deferred)
        return min(modifiers)

    def get_detailed_modifiers(self, d: Optional[date] = None) -> Dict[CalendarWindow, float]:
        """Get individual modifiers for each active window on a date."""
        if d is None:
            d = date.today()

        windows = self._detect_windows(d)
        result = {}
        for window in windows:
            result[window] = self.URGENCY_MODIFIERS[window]
        return result

    def _detect_windows(self, d: date) -> List[CalendarWindow]:
        """Detect all active calendar windows for a given date."""
        windows = []

        if not self.calendar.is_trading_day(d):
            return windows

        # Turn-of-Month: last trading day + first 3 trading days
        if self._is_tom_window(d):
            windows.append(CalendarWindow.TOM)

        # Pre-holiday
        if self._is_pre_holiday(d):
            windows.append(CalendarWindow.PRE_HOLIDAY)

        # Post-holiday
        if self._is_post_holiday(d):
            windows.append(CalendarWindow.POST_HOLIDAY)

        # Quarter-end: last 5 trading days of quarter
        if self._is_quarter_end(d):
            windows.append(CalendarWindow.QUARTER_END)

        # Monday effect
        if d.weekday() == 0:
            windows.append(CalendarWindow.MONDAY)

        # Pre-FOMC
        if self._is_pre_fomc(d):
            windows.append(CalendarWindow.PRE_FOMC)

        # December tax-loss
        if d.month == 12:
            windows.append(CalendarWindow.DECEMBER)

        # Options expiry: 3rd Friday of month
        if self._is_opex(d):
            windows.append(CalendarWindow.OPTIONS_EXPIRY)

        return windows

    def _is_tom_window(self, d: date) -> bool:
        """Check if date is in Turn-of-Month window.

        TOM = last trading day of month + first 3 trading days of next month.
        """
        # First 3 trading days of month
        first_trading_days = []
        probe = date(d.year, d.month, 1)
        while len(first_trading_days) < 3:
            if self.calendar.is_trading_day(probe):
                first_trading_days.append(probe)
            probe += timedelta(days=1)

        if d in first_trading_days:
            return True

        # Last trading day of month
        if d.month == 12:
            last_day = date(d.year, 12, 31)
        else:
            last_day = date(d.year, d.month + 1, 1) - timedelta(days=1)

        # Walk backward from last day to find last trading day
        last_trading = last_day
        while not self.calendar.is_trading_day(last_trading):
            last_trading -= timedelta(days=1)

        return d == last_trading

    def _is_pre_holiday(self, d: date) -> bool:
        """Check if date is the trading day before a market holiday."""
        next_day = self.calendar.next_trading_day(d)

        # Check if any day between d+1 and next_day is a holiday
        # (handles Friday before Monday holiday)
        probe = d + timedelta(days=1)
        while probe <= next_day:
            if self.calendar.is_holiday(probe):
                return True
            probe += timedelta(days=1)

        # Also check: is tomorrow a holiday?
        tomorrow = d + timedelta(days=1)
        return self.calendar.is_holiday(tomorrow)

    def _is_post_holiday(self, d: date) -> bool:
        """Check if date is the trading day after a market holiday."""
        prev_day = self.calendar.previous_trading_day(d)

        # Check if any day between prev_day and d is a holiday
        probe = prev_day + timedelta(days=1)
        while probe < d:
            if self.calendar.is_holiday(probe):
                return True
            probe += timedelta(days=1)

        yesterday = d - timedelta(days=1)
        return self.calendar.is_holiday(yesterday)

    def _is_quarter_end(self, d: date) -> bool:
        """Check if date is in the last 5 trading days of a quarter."""
        quarter_end_months = {3, 6, 9, 12}

        if d.month not in quarter_end_months:
            return False

        # Find the last day of the quarter month
        if d.month == 12:
            last_calendar_day = date(d.year, 12, 31)
        else:
            last_calendar_day = date(d.year, d.month + 1, 1) - timedelta(days=1)

        # Get last 5 trading days
        trading_days_in_range = []
        probe = last_calendar_day
        while len(trading_days_in_range) < 5:
            if self.calendar.is_trading_day(probe):
                trading_days_in_range.append(probe)
            probe -= timedelta(days=1)
            if probe.month != d.month and d.month != 12:
                # Went into previous month
                break

        return d in trading_days_in_range

    def _is_pre_fomc(self, d: date) -> bool:
        """Check if date is a trading day before a FOMC announcement."""
        for fomc_date in self.FOMC_2026_DATES:
            # Day before FOMC (if trading day)
            pre_date = fomc_date - timedelta(days=1)

            # Adjust for weekend
            while not self.calendar.is_trading_day(pre_date):
                pre_date -= timedelta(days=1)

            if d == pre_date:
                return True

            # Also check: d is the trading day before FOMC date
            # (handles FOMC on Wednesday, pre-FOMC on Tuesday)
            if self.calendar.is_trading_day(d):
                next_trade = self.calendar.next_trading_day(d)
                if next_trade == fomc_date:
                    return True

        return False

    def _is_opex(self, d: date) -> bool:
        """Check if date is monthly options expiration (3rd Friday)."""
        if d.weekday() != 4:  # Not Friday
            return False

        # 3rd Friday of month
        first = date(d.year, d.month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)

        return d == third_friday

    def classify_effect(self, urgency_modifier: float) -> SeasonalityEffect:
        """Classify seasonality effect based on urgency modifier."""
        if urgency_modifier >= 0.95:
            return SeasonalityEffect.NEUTRAL
        elif urgency_modifier >= 0.75:
            return SeasonalityEffect.POSITIVE
        elif urgency_modifier >= 0.50:
            return SeasonalityEffect.NEGATIVE
        else:
            return SeasonalityEffect.AVOID

    def get_recommendation(self, urgency_modifier: float) -> str:
        """Get execution recommendation based on modifier."""
        if urgency_modifier >= 0.95:
            return "proceed"
        elif urgency_modifier >= 0.75:
            return "proceed"  # Favorable, execute without delay
        elif urgency_modifier >= 0.60:
            return "delay"    # Wait for better window if low urgency
        elif urgency_modifier >= 0.50:
            return "wait"     # Strongly consider waiting
        else:
            return "avoid"    # Defer unless urgent

    def find_next_window(self, d: Optional[date] = None) -> Tuple[str, date, int]:
        """Find the next upcoming calendar window and days until it."""
        if d is None:
            d = date.today()

        # Look ahead up to 30 days
        for days_ahead in range(1, 31):
            probe = d + timedelta(days=days_ahead)
            if not self.calendar.is_trading_day(probe):
                continue

            windows = self._detect_windows(probe)
            if windows:
                # Return the most impactful window
                significant = [w for w in windows if w in {
                    CalendarWindow.TOM, CalendarWindow.PRE_HOLIDAY,
                    CalendarWindow.QUARTER_END, CalendarWindow.PRE_FOMC,
                }]
                if significant:
                    return (significant[0].value, probe, days_ahead)

        return ("none", d + timedelta(days=30), 30)


class CalendarSeasonalitySignalGenerator:
    """
    Main signal generator for calendar seasonality overlay.

    Generates signals used by the rebalance scheduler to adjust
    execution timing based on calendar effects.
    """

    OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "signals" / "calendar_seasonality.json"

    def __init__(self):
        self.detector = CalendarSeasonalityDetector()
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def generate_signal(self, d: Optional[date] = None) -> CalendarSeasonalitySignal:
        """Generate complete calendar seasonality signal for a date."""
        if d is None:
            d = date.today()

        is_trading = self.detector.calendar.is_trading_day(d)
        windows = self.detector._detect_windows(d) if is_trading else []
        modifier = self.detector.get_urgency_modifier(d)

        detailed = self.detector.get_detailed_modifiers(d)

        next_name, next_date, days_away = self.detector.find_next_window(d)

        effect = self.detector.classify_effect(modifier)
        recommendation = self.detector.get_recommendation(modifier)

        # Confidence decreases with more overlapping windows (more uncertainty)
        num_windows = len(windows)
        confidence = max(60.0, 95.0 - num_windows * 8.0) if is_trading else 0.0

        return CalendarSeasonalitySignal(
            assessment_date=d.isoformat(),
            day_of_week=d.strftime("%A"),
            is_trading_day=is_trading,
            active_windows=[w.value for w in windows],
            urgency_modifier=round(modifier, 2),
            tom_modifier=detailed.get(CalendarWindow.TOM, 1.0),
            holiday_modifier=min(
                detailed.get(CalendarWindow.PRE_HOLIDAY, 1.0),
                detailed.get(CalendarWindow.POST_HOLIDAY, 1.0),
            ),
            quarter_end_modifier=detailed.get(CalendarWindow.QUARTER_END, 1.0),
            monday_modifier=detailed.get(CalendarWindow.MONDAY, 1.0),
            fomc_modifier=detailed.get(CalendarWindow.PRE_FOMC, 1.0),
            december_modifier=detailed.get(CalendarWindow.DECEMBER, 1.0),
            opex_modifier=detailed.get(CalendarWindow.OPTIONS_EXPIRY, 1.0),
            recommendation=recommendation,
            effect=effect.value,
            next_window=next_name,
            next_window_date=next_date.isoformat(),
            days_to_next_window=days_away,
            confidence=round(confidence, 1),
        )

    def save_signal(self, signal: CalendarSeasonalitySignal):
        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(signal.to_dict(), f, indent=2)


def get_calendar_modifier(d: Optional[date] = None) -> float:
    """Convenience function for integration with rebalance scheduler."""
    detector = CalendarSeasonalityDetector()
    return detector.get_urgency_modifier(d)


def check_calendar(d: Optional[date] = None) -> CalendarSeasonalitySignal:
    """Quick calendar check convenience function."""
    generator = CalendarSeasonalitySignalGenerator()
    return generator.generate_signal(d)


def main():
    """CLI entry point."""
    import sys

    generator = CalendarSeasonalitySignalGenerator()
    signal = generator.generate_signal()

    print("=" * 60)
    print("CALENDAR SEASONALITY SIGNAL v3.50")
    print("=" * 60)
    print(f"Date: {signal.assessment_date} ({signal.day_of_week})")
    print(f"Trading Day: {signal.is_trading_day}")
    print()
    print("Active Windows:", signal.active_windows or ["none"])
    print(f"Urgency Modifier: {signal.urgency_modifier:.2f}")
    print(f"Recommendation: {signal.recommendation}")
    print(f"Effect: {signal.effect}")
    print()
    print("Individual Modifiers:")
    print(f"  TOM:          {signal.tom_modifier:.2f}")
    print(f"  Holiday:      {signal.holiday_modifier:.2f}")
    print(f"  Quarter-End:  {signal.quarter_end_modifier:.2f}")
    print(f"  Monday:       {signal.monday_modifier:.2f}")
    print(f"  Pre-FOMC:     {signal.fomc_modifier:.2f}")
    print(f"  December:     {signal.december_modifier:.2f}")
    print(f"  OPEX:         {signal.opex_modifier:.2f}")
    print()
    print(f"Next Window: {signal.next_window} ({signal.next_window_date}, "
          f"{signal.days_to_next_window}d away)")
    print(f"Confidence: {signal.confidence:.0f}%")
    print("=" * 60)

    if "--save" in sys.argv:
        generator.save_signal(signal)

    # Calendar view mode
    if len(sys.argv) > 1 and sys.argv[1] == "calendar":
        if len(sys.argv) > 2:
            try:
                year, month = sys.argv[2].split("-")
                year, month = int(year), int(month)
            except ValueError:
                year, month = date.today().year, date.today().month
        else:
            year, month = date.today().year, date.today().month

        print(f"\nCalendar View: {year}-{month:02d}")
        print("-" * 60)

        cal = NYSECalendar(year=year)
        detector = CalendarSeasonalityDetector(year=year)

        # Print all trading days in the month
        first = date(year, month, 1)
        if month == 12:
            last = date(year, 12, 31)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)

        for d in cal.trading_days_between(first, last):
            mod = detector.get_urgency_modifier(d)
            windows = [w.value for w in detector._detect_windows(d)]
            bar = "█" * int((1 - mod) * 10)
            print(f"  {d.isoformat()} ({d.strftime('%a')}) "
                  f"mod={mod:.2f} {bar} {windows or 'normal'}")


if __name__ == "__main__":
    main()
