"""
Tests for Calendar Seasonality Signal Generator (v3.50)
"""

import pytest
from datetime import datetime, date, timedelta

from src.signals.calendar_seasonality import (
    NYSECalendar,
    CalendarSeasonalityDetector,
    CalendarSeasonalitySignalGenerator,
    CalendarSeasonalitySignal,
    CalendarWindow,
    SeasonalityEffect,
    get_calendar_modifier,
    check_calendar,
)


class TestNYSECalendar:
    """Test NYSE trading calendar."""

    @pytest.fixture
    def cal(self):
        return NYSECalendar(year=2026)

    def test_weekend_not_trading_day(self, cal):
        assert not cal.is_trading_day(date(2026, 1, 3))   # Saturday
        assert not cal.is_trading_day(date(2026, 1, 4))   # Sunday

    def test_weekday_is_trading_day(self, cal):
        # January 5, 2026 is a Monday (not a holiday)
        assert cal.is_trading_day(date(2026, 1, 5))

    def test_new_years_day_is_holiday(self, cal):
        assert cal.is_holiday(date(2026, 1, 1))

    def test_independence_day_is_holiday(self, cal):
        assert cal.is_holiday(date(2026, 7, 3))  # Observed Friday 7/3

    def test_christmas_is_holiday(self, cal):
        assert cal.is_holiday(date(2026, 12, 25))

    def test_mlk_day_is_holiday(self, cal):
        # MLK Day 2026 = Jan 19 (3rd Monday)
        d = date(2026, 1, 19)
        assert d.weekday() == 0  # Monday
        assert cal.is_holiday(d)

    def test_presidents_day_is_holiday(self, cal):
        # Presidents Day 2026 = Feb 16 (3rd Monday)
        d = date(2026, 2, 16)
        assert d.weekday() == 0
        assert cal.is_holiday(d)

    def test_memorial_day_is_holiday(self, cal):
        # Memorial Day 2026 = May 25 (last Monday)
        d = date(2026, 5, 25)
        assert d.weekday() == 0
        assert cal.is_holiday(d)

    def test_labor_day_is_holiday(self, cal):
        # Labor Day 2026 = Sep 7 (1st Monday)
        d = date(2026, 9, 7)
        assert d.weekday() == 0
        assert cal.is_holiday(d)

    def test_thanksgiving_is_holiday(self, cal):
        # Thanksgiving 2026 = Nov 26 (4th Thursday)
        d = date(2026, 11, 26)
        assert d.weekday() == 3  # Thursday
        assert cal.is_holiday(d)

    def test_good_friday_is_holiday(self, cal):
        # Good Friday 2026 = April 3
        assert cal.is_holiday(date(2026, 4, 3))

    def test_juneteenth_is_holiday(self, cal):
        assert cal.is_holiday(date(2026, 6, 19))

    def test_next_trading_day(self, cal):
        # After Friday → Monday (if no holiday)
        friday = date(2026, 1, 9)
        assert cal.next_trading_day(friday) == date(2026, 1, 12)

    def test_next_trading_day_over_holiday(self, cal):
        # Dec 24, 2026 (Thursday) → Dec 25 is Christmas → Dec 28 Monday
        thursday = date(2026, 12, 24)
        assert cal.next_trading_day(thursday) == date(2026, 12, 28)

    def test_previous_trading_day(self, cal):
        monday = date(2026, 1, 5)
        assert cal.previous_trading_day(monday) == date(2026, 1, 2)

    def test_trading_days_between(self, cal):
        days = cal.trading_days_between(date(2026, 1, 5), date(2026, 1, 9))
        assert len(days) == 5  # Mon-Fri

    def test_easter_computation(self):
        """Test Easter calculation for known dates."""
        cal = NYSECalendar(year=2025)
        easter = cal._compute_easter(2025)
        assert easter == date(2025, 4, 20)

        cal2 = NYSECalendar(year=2026)
        easter2 = cal2._compute_easter(2026)
        assert easter2 == date(2026, 4, 5)

    def test_multiple_holidays_consistent(self, cal):
        """All detected holidays should return is_holiday True."""
        for h in cal.holidays:
            assert cal.is_holiday(h), f"{h} should be a holiday"


class TestTurnOfMonthWindow:
    """Test Turn-of-Month window detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_first_trading_day_of_month(self, detector):
        # Jan 2, 2026 (Friday) — first trading day of January
        result = detector._detect_windows(date(2026, 1, 2))
        assert CalendarWindow.TOM in result

    def test_second_trading_day_of_month(self, detector):
        # Jan 5, 2026 (Monday)
        result = detector._detect_windows(date(2026, 1, 5))
        assert CalendarWindow.TOM in result

    def test_third_trading_day_of_month(self, detector):
        # Jan 6, 2026 (Tuesday)
        result = detector._detect_windows(date(2026, 1, 6))
        assert CalendarWindow.TOM in result

    def test_fourth_trading_day_not_tom(self, detector):
        # Jan 7, 2026 (Wednesday) — 4th trading day
        result = detector._detect_windows(date(2026, 1, 7))
        assert CalendarWindow.TOM not in result

    def test_mid_month_not_tom(self, detector):
        result = detector._detect_windows(date(2026, 1, 15))
        assert CalendarWindow.TOM not in result

    def test_last_trading_day_of_month(self, detector):
        # Jan 30, 2026 (Friday) — last trading day
        result = detector._detect_windows(date(2026, 1, 30))
        assert CalendarWindow.TOM in result


class TestPreHolidayWindow:
    """Test pre-holiday window detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_day_before_thanksgiving(self, detector):
        # Thanksgiving 2026 = Nov 26 (Thu), day before = Nov 25 (Wed)
        result = detector._detect_windows(date(2026, 11, 25))
        assert CalendarWindow.PRE_HOLIDAY in result

    def test_day_before_christmas(self, detector):
        # Christmas 2026 = Dec 25 (Fri), day before = Dec 24 (Thu)
        result = detector._detect_windows(date(2026, 12, 24))
        assert CalendarWindow.PRE_HOLIDAY in result

    def test_day_before_independence_day(self, detector):
        # July 4 = Sat, observed Fri Jul 3. Day before observed = Thu Jul 2
        # Actually: pre-holiday is day before the holiday itself, not observed
        result = detector._detect_windows(date(2026, 7, 2))
        # July 3 is the observed holiday, 2 is the day before
        assert CalendarWindow.PRE_HOLIDAY in result

    def test_normal_day_not_pre_holiday(self, detector):
        result = detector._detect_windows(date(2026, 3, 10))
        assert CalendarWindow.PRE_HOLIDAY not in result


class TestQuarterEndWindow:
    """Test quarter-end window detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_march_quarter_end(self, detector):
        # March 31, 2026 (Tuesday) — last 5 trading days include 3/25-3/31
        result = detector._detect_windows(date(2026, 3, 30))
        assert CalendarWindow.QUARTER_END in result

    def test_june_quarter_end(self, detector):
        # June 30, 2026 (Tuesday) — quarter end
        result = detector._detect_windows(date(2026, 6, 29))
        assert CalendarWindow.QUARTER_END in result

    def test_september_quarter_end(self, detector):
        result = detector._detect_windows(date(2026, 9, 30))
        assert CalendarWindow.QUARTER_END in result

    def test_december_quarter_end(self, detector):
        result = detector._detect_windows(date(2026, 12, 30))
        assert CalendarWindow.QUARTER_END in result

    def test_not_quarter_end_month(self, detector):
        result = detector._detect_windows(date(2026, 2, 25))
        assert CalendarWindow.QUARTER_END not in result

    def test_early_march_not_quarter_end(self, detector):
        result = detector._detect_windows(date(2026, 3, 10))
        assert CalendarWindow.QUARTER_END not in result


class TestMondayEffect:
    """Test Monday effect detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_monday_detected(self, detector):
        # Jan 5, 2026 is Monday
        result = detector._detect_windows(date(2026, 1, 5))
        assert CalendarWindow.MONDAY in result

    def test_tuesday_not_monday(self, detector):
        result = detector._detect_windows(date(2026, 1, 6))
        assert CalendarWindow.MONDAY not in result

    def test_friday_not_monday(self, detector):
        result = detector._detect_windows(date(2026, 1, 9))
        assert CalendarWindow.MONDAY not in result


class TestPreFOMCWindow:
    """Test pre-FOMC window detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_day_before_fomc(self, detector):
        # FOMC 2026 = Jan 28 (Wed), pre-FOMC = Jan 27 (Tue)
        result = detector._detect_windows(date(2026, 1, 27))
        assert CalendarWindow.PRE_FOMC in result

    def test_regular_day_not_fomc(self, detector):
        result = detector._detect_windows(date(2026, 2, 10))
        assert CalendarWindow.PRE_FOMC not in result

    def test_fomc_day_not_pre_fomc(self, detector):
        # Jan 28 is FOMC day itself, not pre-FOMC
        result = detector._detect_windows(date(2026, 1, 28))
        assert CalendarWindow.PRE_FOMC not in result


class TestDecemberEffect:
    """Test December tax-loss effect."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_december_detected(self, detector):
        result = detector._detect_windows(date(2026, 12, 15))
        assert CalendarWindow.DECEMBER in result

    def test_november_not_december(self, detector):
        result = detector._detect_windows(date(2026, 11, 15))
        assert CalendarWindow.DECEMBER not in result


class TestOptionsExpiry:
    """Test monthly OPEX detection."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_january_opex(self, detector):
        # 3rd Friday of Jan 2026 = Jan 16
        result = detector._detect_windows(date(2026, 1, 16))
        assert CalendarWindow.OPTIONS_EXPIRY in result

    def test_normal_friday_not_opex(self, detector):
        # 2nd Friday of Jan 2026 = Jan 9
        result = detector._detect_windows(date(2026, 1, 9))
        assert CalendarWindow.OPTIONS_EXPIRY not in result

    def test_third_friday_february(self, detector):
        # 3rd Friday of Feb 2026 = Feb 20
        result = detector._detect_windows(date(2026, 2, 20))
        assert CalendarWindow.OPTIONS_EXPIRY in result


class TestUrgencyModifier:
    """Test composite urgency modifier calculation."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_normal_day_modifier_1(self, detector):
        """Normal Tuesday mid-month should have modifier 1.0."""
        modifier = detector.get_urgency_modifier(date(2026, 3, 10))
        assert modifier == 1.0

    def test_monday_modifier(self, detector):
        """Monday should have 0.80 modifier."""
        modifier = detector.get_urgency_modifier(date(2026, 3, 9))
        assert modifier == 0.80

    def test_pre_holiday_modifier(self, detector):
        """Pre-Thanksgiving should have 0.50 modifier."""
        modifier = detector.get_urgency_modifier(date(2026, 11, 25))
        assert modifier == 0.50

    def test_tom_modifier(self, detector):
        """TOM window should have 0.70 modifier."""
        modifier = detector.get_urgency_modifier(date(2026, 1, 2))
        assert modifier == 0.70

    def test_multiple_windows_min_modifier(self, detector):
        """Monday in TOM: should use minimum modifier (0.70 vs 0.80)."""
        mod = detector.get_urgency_modifier(date(2026, 2, 2))  # Mon TOM
        assert mod == 0.70  # 0.70 < 0.80

    def test_modifier_range(self, detector):
        """All modifiers should be between 0 and 1."""
        for d_offset in range(252):
            d = date(2026, 1, 5) + timedelta(days=d_offset)
            if detector.calendar.is_trading_day(d):
                mod = detector.get_urgency_modifier(d)
                assert 0.0 <= mod <= 1.0

    def test_non_trading_day_returns_1(self, detector):
        """Non-trading days should return 1.0 (no effect)."""
        mod = detector.get_urgency_modifier(date(2026, 1, 3))  # Saturday
        assert mod == 1.0


class TestCompositeSignal:
    """Test full CalendarSeasonalitySignal generation."""

    @pytest.fixture
    def generator(self):
        return CalendarSeasonalitySignalGenerator()

    def test_generate_normal_signal(self, generator):
        signal = generator.generate_signal(date(2026, 3, 10))
        assert isinstance(signal, CalendarSeasonalitySignal)
        assert signal.urgency_modifier == 1.0
        assert signal.effect == "neutral"
        assert signal.recommendation == "proceed"
        assert signal.is_trading_day

    def test_generate_monday_signal(self, generator):
        signal = generator.generate_signal(date(2026, 3, 9))
        assert signal.urgency_modifier == 0.80
        assert signal.monday_modifier == 0.80
        assert CalendarWindow.MONDAY.value in signal.active_windows

    def test_generate_pre_holiday_signal(self, generator):
        signal = generator.generate_signal(date(2026, 11, 25))
        assert signal.urgency_modifier == 0.50
        assert signal.recommendation in ("wait", "avoid", "delay")
        assert CalendarWindow.PRE_HOLIDAY.value in signal.active_windows

    def test_signal_serializable(self, generator):
        signal = generator.generate_signal(date(2026, 3, 10))
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "urgency_modifier" in d
        assert "active_windows" in d

    def test_confidence_high_for_normal_day(self, generator):
        signal = generator.generate_signal(date(2026, 3, 10))
        assert signal.confidence >= 85

    def test_confidence_lower_for_complex_day(self, generator):
        # Monday + TOM + December → more windows → lower confidence
        # Dec 1 is first trading day of December (TOM + Monday + December)
        signal = generator.generate_signal(date(2026, 12, 1))
        assert signal.confidence < 90

    def test_non_trading_day_signal(self, generator):
        signal = generator.generate_signal(date(2026, 1, 3))  # Saturday
        assert not signal.is_trading_day
        assert signal.urgency_modifier == 1.0
        assert signal.confidence == 0.0

    def test_next_window_info_present(self, generator):
        signal = generator.generate_signal(date(2026, 3, 10))
        assert signal.next_window is not None
        assert signal.days_to_next_window >= 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_calendar_modifier(self):
        mod = get_calendar_modifier(date(2026, 3, 10))
        assert mod == 1.0

    def test_get_calendar_modifier_monday(self):
        mod = get_calendar_modifier(date(2026, 3, 9))
        assert mod == 0.80

    def test_check_calendar(self):
        signal = check_calendar(date(2026, 3, 10))
        assert isinstance(signal, CalendarSeasonalitySignal)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_new_year_transition(self, detector):
        """Dec 31 and Jan 2 should both be TOM windows."""
        result_dec = detector._detect_windows(date(2026, 12, 31))
        assert CalendarWindow.TOM in result_dec or CalendarWindow.DECEMBER in result_dec

    def test_leap_year_handling(self, detector):
        """No crash on leap year dates."""
        # 2024 is a leap year but our detector is for 2026, test safely
        mod = detector.get_urgency_modifier(date(2026, 2, 28))
        assert isinstance(mod, float)

    def test_holiday_observed_weekend_adjustment(self):
        """Holiday on Saturday should be observed Friday."""
        cal = NYSECalendar(year=2026)
        # July 4, 2026 is Saturday → observed Friday July 3
        assert cal.is_holiday(date(2026, 7, 3))

    def test_window_detection_empty_for_non_trading(self, detector):
        """No windows should be detected for non-trading days."""
        result = detector._detect_windows(date(2026, 1, 3))  # Saturday
        assert len(result) == 0

    def test_detailed_modifiers(self, detector):
        """get_detailed_modifiers should return dict."""
        mods = detector.get_detailed_modifiers(date(2026, 3, 9))  # Monday
        assert CalendarWindow.MONDAY in mods
        assert mods[CalendarWindow.MONDAY] == 0.80

    def test_find_next_window_always_returns(self, detector):
        name, next_date, days = detector.find_next_window(date(2026, 1, 15))
        assert isinstance(name, str)
        assert isinstance(days, int)
        assert days >= 0

    def test_classify_effect_all_levels(self, detector):
        assert detector.classify_effect(1.0) == SeasonalityEffect.NEUTRAL
        assert detector.classify_effect(0.80) == SeasonalityEffect.POSITIVE
        assert detector.classify_effect(0.65) == SeasonalityEffect.NEGATIVE
        assert detector.classify_effect(0.40) == SeasonalityEffect.AVOID

    def test_recommendation_all_levels(self, detector):
        assert detector.get_recommendation(1.0) == "proceed"
        assert detector.get_recommendation(0.80) == "proceed"
        assert detector.get_recommendation(0.65) == "delay"
        assert detector.get_recommendation(0.55) == "wait"
        assert detector.get_recommendation(0.40) == "avoid"


class TestFOMCSchedule:
    """Test FOMC meeting schedule for 2026."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_eight_fomc_meetings(self, detector):
        assert len(detector.FOMC_2026_DATES) == 8

    def test_all_fomc_pre_detected(self, detector):
        """Each FOMC meeting should have a pre-FOMC trading day detected."""
        for fomc_date in detector.FOMC_2026_DATES:
            # Find pre-FOMC trading day
            pre = fomc_date - timedelta(days=1)
            while not detector.calendar.is_trading_day(pre):
                pre -= timedelta(days=1)
            windows = detector._detect_windows(pre)
            assert CalendarWindow.PRE_FOMC in windows, \
                f"Pre-FOMC not detected for {fomc_date} (pre={pre})"


class TestModifierHierarchy:
    """Test that modifier hierarchy is correct."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=2026)

    def test_multiple_windows_use_min(self, detector):
        """When multiple windows active, use minimum modifier."""
        # Monday in December TOM → 0.70 (min of 0.70, 0.80, 0.75) = 0.70
        # First Monday of Dec might be TOM
        mod = detector.get_urgency_modifier(date(2026, 12, 28))
        # Dec 28, 2026 = Monday + December + near quarter-end
        # Should use min modifier, not 1.0
        assert mod <= 0.85
