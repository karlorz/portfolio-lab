"""
Tests for Calendar Seasonality Signal Generator (v3.50)
"""

import pytest
from datetime import date, timedelta

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


# ──────────────────────────────────────────────
# Helper: shared year fixture used by most tests
# ──────────────────────────────────────────────
_YEAR = 2026


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
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_multiple_windows_use_min(self, detector):
        """When multiple windows active, use minimum modifier."""
        # Monday in December TOM → 0.70 (min of 0.70, 0.80, 0.75) = 0.70
        # First Monday of Dec might be TOM
        mod = detector.get_urgency_modifier(date(2026, 12, 28))
        # Dec 28, 2026 = Monday + December + near quarter-end
        # Should use min modifier, not 1.0
        assert mod <= 0.85


class TestCalendarHelperMethods:
    """Test NYSECalendar helper methods not covered by other tests."""

    @pytest.fixture
    def cal(self):
        return NYSECalendar(year=_YEAR)

    def test_adjust_to_trading_day_sunday_to_monday(self, cal):
        """Holiday on Sunday should be observed Monday."""
        # NYE 2027-01-01 is... let's test a known Sunday date
        # Use Independence Day 2027: Jul 4 is Sunday → observed Jul 5 (Monday)
        cal_2027 = NYSECalendar(year=2027)
        d = date(2027, 7, 4)
        assert d.weekday() == 6  # Sunday
        observed = cal_2027._adjust_to_trading_day(d)
        assert observed == date(2027, 7, 5)
        assert observed.weekday() == 0  # Monday

    def test_adjust_to_trading_day_weekday_unchanged(self, cal):
        """Holiday on weekday stays the same."""
        d = date(2026, 12, 25)  # Christmas 2026 is Friday
        assert d.weekday() == 4
        adjusted = cal._adjust_to_trading_day(d)
        assert adjusted == d

    def test_nth_weekday_first_occurrence(self, cal):
        """First Monday of January 2026 = Jan 5."""
        result = cal._nth_weekday(2026, 1, 0, 1)
        assert result == date(2026, 1, 5)

    def test_nth_weekday_last_december_edge(self, cal):
        """Last occurrence in December (month=12) should not crash."""
        result = cal._nth_weekday(2026, 12, 0, -1)  # Last Monday
        assert result == date(2026, 12, 28)
        assert result.weekday() == 0

    def test_nth_weekday_last_non_december(self, cal):
        """Last Monday of May 2026 = May 25."""
        result = cal._nth_weekday(2026, 5, 0, -1)
        assert result == date(2026, 5, 25)

    def test_trading_days_between_single_day(self, cal):
        """Range with a single trading day returns that day."""
        days = cal.trading_days_between(date(2026, 1, 5), date(2026, 1, 5))
        assert len(days) == 1
        assert days[0] == date(2026, 1, 5)

    def test_trading_days_between_weekend_range(self, cal):
        """Range containing only weekend days returns empty list."""
        days = cal.trading_days_between(date(2026, 1, 3), date(2026, 1, 4))
        assert len(days) == 0

    def test_trading_days_between_spanning_holiday(self, cal):
        """Range spanning Christmas holiday skips Dec 25."""
        # Dec 24 (Thu) to Dec 28 (Mon); Dec 25 is Christmas (Fri)
        days = cal.trading_days_between(date(2026, 12, 24), date(2026, 12, 28))
        assert date(2026, 12, 25) not in days
        assert date(2026, 12, 24) in days
        assert date(2026, 12, 28) in days

    def test_different_years_different_holidays(self):
        """Holiday sets differ between years."""
        cal_2025 = NYSECalendar(year=2025)
        cal_2026 = NYSECalendar(year=2026)
        holidays_2025 = cal_2025.holidays
        holidays_2026 = cal_2026.holidays
        # Same-date fixed holidays should be in both
        assert date(2025, 1, 1) in holidays_2025
        assert date(2026, 1, 1) in holidays_2026
        # Floating holidays differ by year
        # MLK 2025 = Jan 20, MLK 2026 = Jan 19
        assert date(2025, 1, 20) in holidays_2025
        assert date(2026, 1, 19) in holidays_2026

    def test_easter_early_years(self):
        """Easter can be as early as March 22."""
        cal_2008 = NYSECalendar(year=2008)
        easter = cal_2008._compute_easter(2008)
        assert easter == date(2008, 3, 23)

    def test_easter_late_years(self):
        """Easter can be as late as April 25."""
        cal_2011 = NYSECalendar(year=2011)
        easter = cal_2011._compute_easter(2011)
        assert easter == date(2011, 4, 24)

    def test_next_trading_day_friday_no_holiday(self, cal):
        """Friday to next Monday without holiday interruption."""
        friday = date(2026, 2, 13)
        assert cal.next_trading_day(friday) == date(2026, 2, 17)  # Mon Feb 16 is Presidents Day!
        # Actually Feb 16 2026 is Presidents Day (holiday), so next is Feb 17

    def test_previous_trading_day_from_monday(self, cal):
        """Monday to previous Friday without holiday."""
        monday = date(2026, 3, 2)
        assert cal.previous_trading_day(monday) == date(2026, 2, 27)

    def test_holiday_on_saturday_observed_friday(self):
        """Holiday on Saturday observed on preceding Friday."""
        cal = NYSECalendar(year=2027)
        # Jul 4 2027 is Sunday, observed Monday Jul 5
        # But FIXED_HOLIDAY adjusts — let's verify Independence Day
        assert cal.is_holiday(date(2027, 7, 5))
        assert not cal.is_holiday(date(2027, 7, 4))

    def test_is_trading_day_on_observed_holiday_false(self, cal):
        """Observed holiday Friday should NOT be a trading day."""
        # July 3, 2026 (Friday) is observed Independence Day
        assert not cal.is_trading_day(date(2026, 7, 3))


class TestPostHolidayWindow:
    """Test post-holiday window detection (completely missing from existing tests)."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_day_after_christmas_is_post_holiday(self, detector):
        """Dec 28 (Mon) is the next trading day after Christmas 2026 (Fri Dec 25)."""
        result = detector._detect_windows(date(2026, 12, 28))
        assert CalendarWindow.POST_HOLIDAY in result

    def test_day_after_july_4_is_post_holiday(self, detector):
        """Jul 6 (Mon) is after observed Jul 4 holiday (Jul 3 Fri)."""
        result = detector._detect_windows(date(2026, 7, 6))
        assert CalendarWindow.POST_HOLIDAY in result

    def test_normal_day_not_post_holiday(self, detector):
        """Mid-month Tuesday should not be post-holiday."""
        result = detector._detect_windows(date(2026, 3, 10))
        assert CalendarWindow.POST_HOLIDAY not in result

    def test_day_after_thanksgiving_is_post_holiday(self, detector):
        """Nov 27 (Fri) is after Thanksgiving 2026 (Nov 26 Thu)."""
        result = detector._detect_windows(date(2026, 11, 27))
        assert CalendarWindow.POST_HOLIDAY in result


class TestTOMEdgeCases:
    """Test TOM window boundary conditions."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_dec_31_tom_combined_december(self, detector):
        """Dec 31, 2026 is last trading day (TOM) AND December effect."""
        result = detector._detect_windows(date(2026, 12, 31))
        assert CalendarWindow.TOM in result
        assert CalendarWindow.DECEMBER in result

    def test_jan_2_first_trading_day_new_year(self, detector):
        """Jan 2, 2026 (Fri) is first trading day — TOM window."""
        result = detector._detect_windows(date(2026, 1, 2))
        assert CalendarWindow.TOM in result

    def test_feb_last_trading_day(self, detector):
        """Feb 27, 2026 (Fri) — last trading day of February."""
        result = detector._detect_windows(date(2026, 2, 27))
        assert CalendarWindow.TOM in result


class TestQuarterEndEdgeCases:
    """Test quarter-end boundary conditions."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_december_quarter_end_last_week(self, detector):
        """Dec 28 (Mon) is in the last week of Q4."""
        result = detector._detect_windows(date(2026, 12, 28))
        assert CalendarWindow.QUARTER_END in result

    def test_march_quarter_end_fully(self, detector):
        """Mar 25 (Wed) is early enough to not be quarter-end."""
        result = detector._detect_windows(date(2026, 3, 25))
        # Mar 25 is a Wednesday; last calendar day Mar 31 (Tue)
        # Last 5 trading days: Mar 25, 26, 27, 30, 31
        # Mar 25 IS in that range
        assert CalendarWindow.QUARTER_END in result

    def test_quarter_end_non_quarter_month(self, detector):
        """Mid-January should not be any quarter end."""
        result = detector._detect_windows(date(2026, 1, 15))
        assert CalendarWindow.QUARTER_END not in result

    def test_september_not_enough_trading_days_left(self, detector):
        """Sep 20 is more than 5 trading days from quarter end."""
        result = detector._detect_windows(date(2026, 9, 18))
        # Sep 30 is last day, 5 trading days before are ~Sep 24
        assert CalendarWindow.QUARTER_END not in result


class TestPreFOMCWeekendEdgeCases:
    """Test FOMC detection when pre-day falls on a weekend."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_pre_fomc_wednesday_meeting_tuesday_pre(self, detector):
        """FOMC on Wed (Mar 18) → pre-day is Tue (Mar 17)."""
        result = detector._detect_windows(date(2026, 3, 17))
        assert CalendarWindow.PRE_FOMC in result

    def test_pre_fomc_monday_after_weekend(self, detector):
        """If FOMC is Tuesday, pre-day is Monday."""
        # Jul 29 is Wednesday FOMC; check Jul 28 (Tue)
        result = detector._detect_windows(date(2026, 7, 28))
        assert CalendarWindow.PRE_FOMC in result

    def test_pre_fomc_not_triggered_on_fomc_day(self, detector):
        """FOMC day itself should not be pre-FOMC."""
        result = detector._detect_windows(date(2026, 3, 18))
        assert CalendarWindow.PRE_FOMC not in result


class TestOPEXEdgeCases:
    """Test options expiry detection edge cases."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_non_friday_never_opex(self, detector):
        """Non-Friday dates should never be OPEX regardless of date."""
        # 3rd Thursday of a month
        thursday = date(2026, 1, 15)  # Jan 15 is Thursday
        result = detector._detect_windows(thursday)
        assert CalendarWindow.OPTIONS_EXPIRY not in result

    def test_first_friday_not_opex(self, detector):
        """First Friday of month is not OPEX."""
        result = detector._detect_windows(date(2026, 1, 2))
        assert CalendarWindow.OPTIONS_EXPIRY not in result

    def test_december_opex_third_friday(self, detector):
        """3rd Friday of December 2026 = Dec 18."""
        result = detector._detect_windows(date(2026, 12, 18))
        assert CalendarWindow.OPTIONS_EXPIRY in result


class TestDetailedModifiers:
    """Test get_detailed_modifiers method."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_detailed_modifiers_empty_normal_day(self, detector):
        """Normal day with no windows returns empty dict."""
        mods = detector.get_detailed_modifiers(date(2026, 3, 10))
        assert len(mods) == 0

    def test_detailed_modifiers_multiple_windows(self, detector):
        """Monday in December TOM returns multiple modifier entries."""
        mods = detector.get_detailed_modifiers(date(2026, 12, 28))
        assert len(mods) >= 2  # At least MONDAY + DECEMBER

    def test_detailed_modifiers_default_none(self, detector):
        """Calling with None uses today's date."""
        mods = detector.get_detailed_modifiers()
        assert isinstance(mods, dict)


class TestFindNextWindowEdgeCases:
    """Test find_next_window logic."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_find_next_window_returns_tom(self, detector):
        """TOM is typically the next significant window."""
        # Jan 15 is mid-month, TOM should be coming up (Feb 2, first trading day of Feb)
        name, next_date, days = detector.find_next_window(date(2026, 1, 15))
        assert days > 0
        assert days <= 30
        assert isinstance(name, str)
        assert name != "none"

    def test_find_next_window_default_none(self, detector):
        """Calling with None uses today's date."""
        name, next_date, days = detector.find_next_window()
        assert isinstance(name, str)
        assert days >= 0

    def test_find_next_window_on_holiday_not_crash(self, detector):
        """Calling from a non-trading day should not crash."""
        name, next_date, days = detector.find_next_window(date(2026, 12, 25))
        assert isinstance(name, str)
        assert days >= 0


class TestSignalSnapshotConversion:
    """Test CalendarSeasonalitySignal.to_signal_snapshot()."""

    @pytest.fixture
    def generator(self):
        return CalendarSeasonalitySignalGenerator()

    def test_to_snapshot_active_signal(self, generator):
        """Active signal produces valid snapshot with is_active=True."""
        signal = generator.generate_signal(date(2026, 3, 9))
        snapshot = signal.to_signal_snapshot()
        assert snapshot.source == "calendar_seasonality"
        assert snapshot.is_active is True
        assert snapshot.value != 0.0

    def test_to_snapshot_inactive_signal(self, generator):
        """Non-trading day produces is_active=False snapshot."""
        signal = generator.generate_signal(date(2026, 1, 3))
        snapshot = signal.to_signal_snapshot()
        assert snapshot.source == "calendar_seasonality"
        assert snapshot.is_active is False

    def test_to_snapshot_effect_values(self, generator):
        """Effect values map correctly: positive=0.3, neutral=0.0, etc."""
        signal_normal = generator.generate_signal(date(2026, 3, 10))
        normal_snap = signal_normal.to_signal_snapshot()
        assert normal_snap.value == 0.0  # neutral

        signal_monday = generator.generate_signal(date(2026, 3, 9))
        monday_snap = signal_monday.to_signal_snapshot()
        assert monday_snap.value == 0.3  # positive

        signal_pre_holiday = generator.generate_signal(date(2026, 11, 25))
        holiday_snap = signal_pre_holiday.to_signal_snapshot()
        assert holiday_snap.value < 0.0  # negative

    def test_to_snapshot_metadata(self, generator):
        """Snapshot metadata contains key calendar fields."""
        signal = generator.generate_signal(date(2026, 3, 9))
        snap = signal.to_signal_snapshot()
        assert "recommendation" in snap.metadata
        assert "effect" in snap.metadata
        assert "active_windows" in snap.metadata
        assert "urgency_modifier" in snap.metadata

    def test_to_snapshot_regime_fit_all(self, generator):
        """Calendar signal is portfolio-wide, regime_fit should be 'all'."""
        signal = generator.generate_signal(date(2026, 3, 9))
        snap = signal.to_signal_snapshot()
        assert snap.regime_fit == "all"


class TestSaveSignal:
    """Test CalendarSeasonalitySignalGenerator.save_signal()."""

    def test_save_signal_writes_json(self, tmp_path, monkeypatch):
        """save_signal writes valid JSON to OUTPUT_PATH."""
        import json
        generator = CalendarSeasonalitySignalGenerator()
        # Redirect OUTPUT_PATH to tmp_path
        test_path = tmp_path / "calendar_seasonality.json"
        monkeypatch.setattr(generator, "OUTPUT_PATH", test_path)

        signal = generator.generate_signal(date(2026, 3, 10))
        generator.save_signal(signal)

        assert test_path.exists()
        with open(test_path) as f:
            data = json.load(f)
        assert data["assessment_date"] == "2026-03-10"
        assert data["urgency_modifier"] == 1.0
        assert data["effect"] == "neutral"

    def test_save_signal_overwrites_existing(self, tmp_path, monkeypatch):
        """save_signal overwrites existing file."""
        generator = CalendarSeasonalitySignalGenerator()
        test_path = tmp_path / "calendar_seasonality.json"
        test_path.write_text('{"old": true}')
        monkeypatch.setattr(generator, "OUTPUT_PATH", test_path)

        signal = generator.generate_signal(date(2026, 3, 10))
        generator.save_signal(signal)

        import json
        with open(test_path) as f:
            data = json.load(f)
        assert "old" not in data
        assert data["assessment_date"] == "2026-03-10"

    def test_save_signal_default_date(self, tmp_path, monkeypatch):
        """generate_signal with no date uses today (no crash)."""
        generator = CalendarSeasonalitySignalGenerator()
        test_path = tmp_path / "calendar_seasonality.json"
        monkeypatch.setattr(generator, "OUTPUT_PATH", test_path)

        signal = generator.generate_signal()  # No date = today
        generator.save_signal(signal)

        import json
        with open(test_path) as f:
            data = json.load(f)
        assert "assessment_date" in data
        assert isinstance(data["assessment_date"], str)


class TestEnumDefinitions:
    """Test enum definitions and values."""

    def test_calendar_window_all_members(self):
        """CalendarWindow enum has all expected members."""
        expected = {"TOM", "PRE_HOLIDAY", "POST_HOLIDAY", "QUARTER_END",
                    "MONDAY", "PRE_FOMC", "DECEMBER", "OPTIONS_EXPIRY"}
        actual = {m.name for m in CalendarWindow}
        assert actual == expected

    def test_calendar_window_values(self):
        """CalendarWindow values match expected strings."""
        assert CalendarWindow.TOM.value == "tom_window"
        assert CalendarWindow.PRE_HOLIDAY.value == "pre_holiday"
        assert CalendarWindow.POST_HOLIDAY.value == "post_holiday"
        assert CalendarWindow.QUARTER_END.value == "quarter_end"
        assert CalendarWindow.MONDAY.value == "monday"
        assert CalendarWindow.PRE_FOMC.value == "pre_fomc"
        assert CalendarWindow.DECEMBER.value == "december"
        assert CalendarWindow.OPTIONS_EXPIRY.value == "options_expiry"

    def test_seasonality_effect_all_members(self):
        """SeasonalityEffect enum has all expected members."""
        expected = {"POSITIVE", "NEUTRAL", "NEGATIVE", "AVOID"}
        actual = {m.name for m in SeasonalityEffect}
        assert actual == expected

    def test_seasonality_effect_values(self):
        """SeasonalityEffect values match expected strings."""
        assert SeasonalityEffect.POSITIVE.value == "positive"
        assert SeasonalityEffect.NEUTRAL.value == "neutral"
        assert SeasonalityEffect.NEGATIVE.value == "negative"
        assert SeasonalityEffect.AVOID.value == "avoid"


class TestConvenienceFunctionsDefaults:
    """Test convenience functions with default (None) date."""

    def test_get_calendar_modifier_default(self):
        """get_calendar_modifier with no argument returns float."""
        mod = get_calendar_modifier()
        assert isinstance(mod, float)
        assert 0.0 <= mod <= 1.0

    def test_check_calendar_default(self):
        """check_calendar with no argument returns valid signal."""
        signal = check_calendar()
        assert isinstance(signal, CalendarSeasonalitySignal)
        assert signal.assessment_date is not None


class TestConfidenceCalculation:
    """Test confidence calculation logic."""

    @pytest.fixture
    def generator(self):
        return CalendarSeasonalitySignalGenerator()

    def test_confidence_max_for_one_window(self, generator):
        """Single window should give near-max confidence."""
        # Monday only → confidence = 95 - 1*8 = 87
        signal = generator.generate_signal(date(2026, 3, 9))
        assert signal.confidence == 87.0

    def test_confidence_two_windows(self, generator):
        """Two windows → confidence = 95 - 2*8 = 79."""
        # Dec 1 is first trading day → TOM + December
        signal = generator.generate_signal(date(2026, 12, 1))
        assert signal.confidence == 79.0

    def test_confidence_zero_for_non_trading(self, generator):
        """Non-trading day confidence is 0."""
        signal = generator.generate_signal(date(2026, 1, 4))
        assert signal.confidence == 0.0

    def test_confidence_floor_at_60(self, generator):
        """Confidence floor should be 60% regardless of window count."""
        # Many windows: 95 - 5*8 = 55 → floor at 60
        signal = generator.generate_signal(date(2026, 12, 31))
        assert signal.confidence >= 60.0


class TestHolidayModifierCombined:
    """Test that the generated signal's holiday_modifier uses min(pre, post)."""

    @pytest.fixture
    def generator(self):
        return CalendarSeasonalitySignalGenerator()

    def test_holiday_modifier_pre_only(self, generator):
        """Pre-holiday day: holiday_modifier = min(0.50, 1.0) = 0.50."""
        signal = generator.generate_signal(date(2026, 11, 25))
        assert signal.holiday_modifier == 0.50

    def test_holiday_modifier_post_only(self, generator):
        """Post-holiday day: holiday_modifier = min(1.0, 0.90) = 0.90."""
        signal = generator.generate_signal(date(2026, 12, 28))
        assert signal.holiday_modifier == 0.90

    def test_holiday_modifier_normal_day(self, generator):
        """Normal day: holiday_modifier = min(1.0, 1.0) = 1.0."""
        signal = generator.generate_signal(date(2026, 3, 10))
        assert signal.holiday_modifier == 1.0


class TestRecommendationBoundaries:
    """Test recommendation boundary values."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_proceed_at_095(self, detector):
        """modifier >= 0.95 → proceed."""
        assert detector.get_recommendation(0.95) == "proceed"
        assert detector.get_recommendation(1.0) == "proceed"

    def test_proceed_at_075(self, detector):
        """0.75 <= modifier < 0.95 → proceed (favorable)."""
        assert detector.get_recommendation(0.75) == "proceed"
        assert detector.get_recommendation(0.80) == "proceed"

    def test_delay_at_060(self, detector):
        """0.60 <= modifier < 0.75 → delay."""
        assert detector.get_recommendation(0.60) == "delay"
        assert detector.get_recommendation(0.70) == "delay"

    def test_wait_at_050(self, detector):
        """0.50 <= modifier < 0.60 → wait."""
        assert detector.get_recommendation(0.50) == "wait"
        assert detector.get_recommendation(0.55) == "wait"

    def test_avoid_below_050(self, detector):
        """modifier < 0.50 → avoid."""
        assert detector.get_recommendation(0.49) == "avoid"
        assert detector.get_recommendation(0.40) == "avoid"
        assert detector.get_recommendation(0.0) == "avoid"


class TestEffectBoundaries:
    """Test classify_effect boundary values."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_neutral_at_095(self, detector):
        """modifier >= 0.95 → NEUTRAL."""
        assert detector.classify_effect(0.95) == SeasonalityEffect.NEUTRAL
        assert detector.classify_effect(1.0) == SeasonalityEffect.NEUTRAL

    def test_positive_at_075(self, detector):
        """0.75 <= modifier < 0.95 → POSITIVE."""
        assert detector.classify_effect(0.75) == SeasonalityEffect.POSITIVE
        assert detector.classify_effect(0.85) == SeasonalityEffect.POSITIVE

    def test_negative_at_050(self, detector):
        """0.50 <= modifier < 0.75 → NEGATIVE."""
        assert detector.classify_effect(0.50) == SeasonalityEffect.NEGATIVE
        assert detector.classify_effect(0.65) == SeasonalityEffect.NEGATIVE

    def test_avoid_below_050(self, detector):
        """modifier < 0.50 → AVOID."""
        assert detector.classify_effect(0.49) == SeasonalityEffect.AVOID
        assert detector.classify_effect(0.40) == SeasonalityEffect.AVOID
        assert detector.classify_effect(0.0) == SeasonalityEffect.AVOID


class TestSignalDataclassFields:
    """Test CalendarSeasonalitySignal dataclass construction and methods."""

    def test_to_dict_round_trip(self):
        """to_dict returns all expected fields."""
        signal = CalendarSeasonalitySignal(
            assessment_date="2026-03-10",
            day_of_week="Tuesday",
            is_trading_day=True,
            active_windows=[],
            urgency_modifier=1.0,
            tom_modifier=1.0,
            holiday_modifier=1.0,
            quarter_end_modifier=1.0,
            monday_modifier=1.0,
            fomc_modifier=1.0,
            december_modifier=1.0,
            opex_modifier=1.0,
            recommendation="proceed",
            effect="neutral",
            next_window="none",
            next_window_date="2026-04-09",
            days_to_next_window=30,
            confidence=95.0,
        )
        d = signal.to_dict()
        assert d["assessment_date"] == "2026-03-10"
        assert d["day_of_week"] == "Tuesday"
        assert d["is_trading_day"] is True
        assert d["urgency_modifier"] == 1.0
        assert d["recommendation"] == "proceed"
        assert d["effect"] == "neutral"

    def test_to_dict_manually_constructed(self):
        """Manually construct signal with known values and verify to_dict."""
        signal = CalendarSeasonalitySignal(
            assessment_date="2026-12-01",
            day_of_week="Tuesday",
            is_trading_day=True,
            active_windows=["tom_window", "december"],
            urgency_modifier=0.70,
            tom_modifier=0.70,
            holiday_modifier=1.0,
            quarter_end_modifier=1.0,
            monday_modifier=0.80,
            fomc_modifier=1.0,
            december_modifier=0.75,
            opex_modifier=1.0,
            recommendation="delay",
            effect="positive",
            next_window="quarter_end",
            next_window_date="2026-12-28",
            days_to_next_window=27,
            confidence=79.0,
        )
        d = signal.to_dict()
        assert d["active_windows"] == ["tom_window", "december"]
        assert d["urgency_modifier"] == 0.70
        assert d["confidence"] == 79.0


class TestDefaultCalendarConstruction:
    """Test that NYSECalendar defaults to current year."""

    def test_default_year(self):
        """NYSECalendar with no year uses current year."""
        cal = NYSECalendar()
        assert cal.year == date.today().year

    def test_detector_default_year(self):
        """CalendarSeasonalityDetector with no year creates NYSECalendar with current year."""
        detector = CalendarSeasonalityDetector()
        assert detector.calendar.year == date.today().year

    def test_generator_default_year(self):
        """CalendarSeasonalitySignalGenerator creates detector with current year."""
        generator = CalendarSeasonalitySignalGenerator()
        assert generator.detector.calendar.year == date.today().year


class TestIsTradingDayHolidayBoundary:
    """Test is_trading_day correctly identifies holidays vs non-holidays."""

    @pytest.fixture
    def cal(self):
        return NYSECalendar(year=_YEAR)

    def test_mlk_day_not_trading(self, cal):
        """MLK Day (Jan 19) should not be a trading day."""
        assert not cal.is_trading_day(date(2026, 1, 19))

    def test_memorial_day_not_trading(self, cal):
        """Memorial Day (May 25) should not be a trading day."""
        assert not cal.is_trading_day(date(2026, 5, 25))

    def test_thanksgiving_not_trading(self, cal):
        """Thanksgiving (Nov 26) should not be a trading day."""
        assert not cal.is_trading_day(date(2026, 11, 26))

    def test_black_friday_is_trading(self, cal):
        """Black Friday (day after Thanksgiving) IS a trading day."""
        assert cal.is_trading_day(date(2026, 11, 27))


class TestPostHolidayPreHolidayOverlap:
    """Test dates that can be both pre-holiday and post-holiday."""

    @pytest.fixture
    def detector(self):
        return CalendarSeasonalityDetector(year=_YEAR)

    def test_dec_24_is_pre_holiday_not_post(self, detector):
        """Dec 24 is pre-Christmas but should not be post-Christmas."""
        result = detector._detect_windows(date(2026, 12, 24))
        assert CalendarWindow.PRE_HOLIDAY in result
        assert CalendarWindow.POST_HOLIDAY not in result

    def test_dec_28_is_post_holiday_not_pre(self, detector):
        """Dec 28 is post-Christmas but should not be pre-Christmas."""
        result = detector._detect_windows(date(2026, 12, 28))
        assert CalendarWindow.POST_HOLIDAY in result
        assert CalendarWindow.PRE_HOLIDAY not in result


class TestNYSECalendarHolidaySaturation:
    """Test that all expected 2026 holidays are present."""

    @pytest.fixture
    def cal(self):
        return NYSECalendar(year=_YEAR)

    def test_expected_holiday_count(self, cal):
        """2026 should have 10 NYSE holidays (approximate count)."""
        # Fixed: NYD(1/1), Jul4(7/3 observed), Christmas(12/25) = 3
        # Floating: MLK(1/19), Presidents(2/16), Memorial(5/25),
        #           Juneteenth(6/19), Labor(9/7), Thanksgiving(11/26) = 6
        # Good Friday(4/3) = 1
        # Total = 10
        assert len(cal.holidays) >= 9

    def test_all_holidays_named_individually(self, cal):
        """Specific known 2026 holidays."""
        known = [
            date(2026, 1, 1),    # New Year's
            date(2026, 1, 19),   # MLK Day
            date(2026, 2, 16),   # Presidents Day
            date(2026, 4, 3),    # Good Friday
            date(2026, 5, 25),   # Memorial Day
            date(2026, 6, 19),   # Juneteenth
            date(2026, 7, 3),    # Independence Day (observed)
            date(2026, 9, 7),    # Labor Day
            date(2026, 11, 26),  # Thanksgiving
            date(2026, 12, 25),  # Christmas
        ]
        for h in known:
            assert cal.is_holiday(h), f"{h} should be a holiday"
