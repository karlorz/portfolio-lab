#!/usr/bin/env python3
"""
Tests for alternative_data_backfill.py — DailyAlternativeSignal dataclass,
crisis detection, signal generation, regime classification, composite scoring,
and metadata generation.
"""
import os
import json

from datetime import datetime
from unittest.mock import patch

from src.backtest.alternative_data_backfill import (
    DailyAlternativeSignal,
    AlternativeDataBackfill,
)


# ---------------------------------------------------------------------------
# DailyAlternativeSignal Tests
# ---------------------------------------------------------------------------

class TestDailyAlternativeSignal:

    def test_fields(self):
        signal = DailyAlternativeSignal(
            date="2020-03-15",
            earnings_sentiment=-0.5,
            news_sentiment=-0.8,
            jobs_growth=-0.9,
            social_sentiment=-0.4,
            composite_score=-0.65,
            regime="risk_off",
            confidence=0.85,
            z_score=-2.17,
            has_earnings=True,
            has_news=True,
            has_jobs=True,
            has_social=True,
        )
        assert signal.date == "2020-03-15"
        assert signal.regime == "risk_off"
        assert signal.confidence == 0.85

    def test_to_dict_all_fields(self):
        signal = DailyAlternativeSignal(
            date="2020-03-15",
            earnings_sentiment=-0.5,
            news_sentiment=-0.8,
            jobs_growth=-0.9,
            social_sentiment=-0.4,
            composite_score=-0.65,
            regime="risk_off",
            confidence=0.85,
            z_score=-2.17,
            has_earnings=True,
            has_news=True,
            has_jobs=True,
            has_social=True,
        )
        d = signal.__dict__
        expected_keys = {
            "date", "earnings_sentiment", "news_sentiment", "jobs_growth",
            "social_sentiment", "composite_score", "regime", "confidence",
            "z_score", "has_earnings", "has_news", "has_jobs", "has_social",
        }
        assert set(d.keys()) == expected_keys

    def test_field_types(self):
        signal = DailyAlternativeSignal(
            date="2020-03-15",
            earnings_sentiment=-0.5,
            news_sentiment=-0.8,
            jobs_growth=-0.9,
            social_sentiment=-0.4,
            composite_score=-0.65,
            regime="risk_off",
            confidence=0.85,
            z_score=-2.17,
            has_earnings=True,
            has_news=True,
            has_jobs=True,
            has_social=True,
        )
        assert isinstance(signal.date, str)
        assert isinstance(signal.earnings_sentiment, float)
        assert isinstance(signal.news_sentiment, float)
        assert isinstance(signal.jobs_growth, float)
        assert isinstance(signal.social_sentiment, float)
        assert isinstance(signal.composite_score, float)
        assert isinstance(signal.regime, str)
        assert isinstance(signal.confidence, float)
        assert isinstance(signal.z_score, float)
        assert isinstance(signal.has_earnings, bool)
        assert isinstance(signal.has_news, bool)
        assert isinstance(signal.has_jobs, bool)
        assert isinstance(signal.has_social, bool)


# ---------------------------------------------------------------------------
# AlternativeDataBackfill — constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_weights(self):
        backfill = AlternativeDataBackfill.__new__(AlternativeDataBackfill)
        assert backfill.WEIGHTS['earnings'] == 0.40
        assert backfill.WEIGHTS['news'] == 0.30
        assert backfill.WEIGHTS['jobs'] == 0.20
        assert backfill.WEIGHTS['social'] == 0.10

    def test_weights_sum_to_one(self):
        backfill = AlternativeDataBackfill.__new__(AlternativeDataBackfill)
        total = sum(backfill.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_crisis_dates(self):
        assert AlternativeDataBackfill.COVID_START == datetime(2020, 2, 20)
        assert AlternativeDataBackfill.COVID_BOTTOM == datetime(2020, 3, 23)
        assert AlternativeDataBackfill.BEAR_BOTTOM == datetime(2022, 10, 12)

    def test_all_crisis_constants(self):
        assert AlternativeDataBackfill.COVID_START == datetime(2020, 2, 20)
        assert AlternativeDataBackfill.COVID_BOTTOM == datetime(2020, 3, 23)
        assert AlternativeDataBackfill.COVID_RECOVERY == datetime(2020, 8, 1)
        assert AlternativeDataBackfill.INFLATION_PEAK == datetime(2022, 6, 1)
        assert AlternativeDataBackfill.BEAR_MARKET_2022 == datetime(2022, 1, 1)
        assert AlternativeDataBackfill.BEAR_BOTTOM == datetime(2022, 10, 12)
        assert AlternativeDataBackfill.RATE_HIKES_START == datetime(2022, 3, 1)

    def test_crisis_dates_chronological(self):
        assert AlternativeDataBackfill.COVID_START < AlternativeDataBackfill.COVID_BOTTOM
        assert AlternativeDataBackfill.COVID_BOTTOM < AlternativeDataBackfill.COVID_RECOVERY
        assert AlternativeDataBackfill.BEAR_MARKET_2022 < AlternativeDataBackfill.BEAR_BOTTOM
        assert AlternativeDataBackfill.RATE_HIKES_START < AlternativeDataBackfill.BEAR_BOTTOM
        assert AlternativeDataBackfill.COVID_RECOVERY < AlternativeDataBackfill.BEAR_MARKET_2022


# ---------------------------------------------------------------------------
# _is_crisis_period Tests
# ---------------------------------------------------------------------------

class TestIsCrisisPeriod:

    def test_normal_period(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2021, 6, 15))
        assert crisis is False
        assert crisis_type == 'normal'

    def test_covid_crash(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2020, 3, 10))
        assert crisis is True
        assert crisis_type == 'covid_crash'

    def test_covid_recovery(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2020, 6, 15))
        assert crisis is True
        assert crisis_type == 'covid_recovery'

    def test_bear_2022(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2022, 6, 15))
        assert crisis is True
        assert crisis_type == 'bear_2022'

    def test_covid_start_boundary(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, _ = backfill._is_crisis_period(datetime(2020, 2, 20))
        assert crisis is True

    def test_covid_bottom_boundary(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2020, 3, 23))
        assert crisis is True
        assert crisis_type == 'covid_crash'

    def test_covid_recovery_upper_boundary(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2020, 8, 1))
        assert crisis is True
        assert crisis_type == 'covid_recovery'

    def test_bear_2022_start_boundary(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2022, 1, 1))
        assert crisis is True
        assert crisis_type == 'bear_2022'

    def test_bear_2022_bottom_boundary(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, crisis_type = backfill._is_crisis_period(datetime(2022, 10, 12))
        assert crisis is True
        assert crisis_type == 'bear_2022'

    def test_before_covid_start_not_crisis(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, _ = backfill._is_crisis_period(datetime(2020, 2, 19))
        assert crisis is False

    def test_after_bear_bottom_not_crisis(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, _ = backfill._is_crisis_period(datetime(2022, 10, 13))
        assert crisis is False

    def test_between_covid_recovery_and_bear_2022_normal(self):
        backfill = AlternativeDataBackfill(seed=42)
        crisis, _ = backfill._is_crisis_period(datetime(2021, 6, 15))
        assert crisis is False


# ---------------------------------------------------------------------------
# _calculate_regime Tests
# ---------------------------------------------------------------------------

class TestCalculateRegime:

    def test_risk_on(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(0.30, 0.7) == 'risk_on'

    def test_risk_off(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(-0.30, 0.7) == 'risk_off'

    def test_neutral_zone(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(0.10, 0.7) == 'neutral'

    def test_low_confidence_neutral(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(0.50, 0.2) == 'neutral'

    def test_boundary_positive(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(0.25, 0.7) == 'neutral'
        assert backfill._calculate_regime(0.26, 0.7) == 'risk_on'

    def test_boundary_negative(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill._calculate_regime(-0.25, 0.7) == 'neutral'
        assert backfill._calculate_regime(-0.26, 0.7) == 'risk_off'


# ---------------------------------------------------------------------------
# generate_daily_signal Tests
# ---------------------------------------------------------------------------

class TestGenerateDailySignal:

    def test_returns_signal(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert isinstance(signal, DailyAlternativeSignal)

    def test_date_formatted(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert signal.date == "2021-06-15"

    def test_values_bounded(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert -1 <= signal.earnings_sentiment <= 1
        assert -1 <= signal.news_sentiment <= 1
        assert -1 <= signal.jobs_growth <= 1
        assert -1 <= signal.social_sentiment <= 1
        assert -1 <= signal.composite_score <= 1

    def test_confidence_bounded(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert 0 <= signal.confidence <= 1

    def test_regime_valid(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert signal.regime in ('risk_on', 'risk_off', 'neutral')

    def test_crisis_more_negative(self):
        backfill = AlternativeDataBackfill(seed=42)
        # COVID crash should have more negative sentiment than normal
        covid_signal = backfill.generate_daily_signal(datetime(2020, 3, 15))
        normal_signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert covid_signal.news_sentiment < normal_signal.news_sentiment

    def test_has_flags(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        assert isinstance(signal.has_earnings, bool)
        assert signal.has_news is True  # News always available
        assert isinstance(signal.has_jobs, bool)
        assert isinstance(signal.has_social, bool)

    def test_deterministic_same_seed(self):
        import random
        # Same seed produces same sequence when called in same order
        random.seed(42)
        b1 = AlternativeDataBackfill(seed=42)
        s1 = b1.generate_backfill('2020-01-01', '2020-01-05')
        random.seed(42)
        b2 = AlternativeDataBackfill(seed=42)
        s2 = b2.generate_backfill('2020-01-01', '2020-01-05')
        for a, b in zip(s1, s2):
            assert a.composite_score == b.composite_score

    def test_future_date_no_crash(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2030, 6, 15))
        assert isinstance(signal, DailyAlternativeSignal)
        assert -1 <= signal.composite_score <= 1

    def test_z_score_consistency(self):
        backfill = AlternativeDataBackfill(seed=42)
        signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
        # z_score = round(composite / 0.3, 4) in source, but composite is also
        # rounded before storage, so allow small tolerance
        assert abs(signal.z_score * 0.3 - signal.composite_score) < 0.001

    def test_news_always_available(self):
        backfill = AlternativeDataBackfill(seed=42)
        for _ in range(50):
            signal = backfill.generate_daily_signal(datetime(2021, 6, 15))
            assert signal.has_news is True

    def test_earnings_season_has_earnings(self):
        backfill = AlternativeDataBackfill(seed=42)
        for month in [1, 4, 7, 10]:
            signal = backfill.generate_daily_signal(datetime(2021, month, 15))
            assert signal.has_earnings is True, f"Expected has_earnings for month {month}"
        # Non-earnings-season should sometimes not have earnings
        non_earnings = backfill.generate_daily_signal(datetime(2021, 3, 15))
        assert isinstance(non_earnings.has_earnings, bool)


# ---------------------------------------------------------------------------
# generate_backfill Tests
# ---------------------------------------------------------------------------

class TestGenerateBackfill:

    def test_returns_list(self):
        backfill = AlternativeDataBackfill(seed=42)
        signals = backfill.generate_backfill('2020-01-01', '2020-01-10')
        assert isinstance(signals, list)
        assert len(signals) == 10

    def test_date_range(self):
        backfill = AlternativeDataBackfill(seed=42)
        signals = backfill.generate_backfill('2020-01-01', '2020-01-05')
        assert signals[0].date == '2020-01-01'
        assert signals[-1].date == '2020-01-05'

    def test_stored_in_signals(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-05')
        assert len(backfill.signals) == 5

    def test_empty_date_range(self):
        backfill = AlternativeDataBackfill(seed=42)
        signals = backfill.generate_backfill('2020-01-10', '2020-01-01')
        assert signals == []

    def test_single_day_backfill(self):
        backfill = AlternativeDataBackfill(seed=42)
        signals = backfill.generate_backfill('2020-06-15', '2020-06-15')
        assert len(signals) == 1
        assert signals[0].date == '2020-06-15'

    def test_leap_year_feb_29(self):
        backfill = AlternativeDataBackfill(seed=42)
        signals = backfill.generate_backfill('2020-02-28', '2020-03-01')
        assert len(signals) == 3
        assert signals[0].date == '2020-02-28'
        assert signals[1].date == '2020-02-29'
        assert signals[2].date == '2020-03-01'

    def test_signals_accumulate(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-03')
        assert len(backfill.signals) == 3
        backfill.generate_backfill('2020-02-01', '2020-02-02')
        assert len(backfill.signals) == 5


# ---------------------------------------------------------------------------
# generate_metadata Tests
# ---------------------------------------------------------------------------

class TestGenerateMetadata:

    def test_empty_signals(self):
        backfill = AlternativeDataBackfill(seed=42)
        assert backfill.generate_metadata() == {}

    def test_has_keys(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-12-31')
        meta = backfill.generate_metadata()
        assert 'total_signals' in meta
        assert 'regime_distribution' in meta
        assert 'avg_confidence' in meta
        assert 'component_availability' in meta
        assert 'crisis_period_analysis' in meta

    def test_regime_distribution(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-12-31')
        meta = backfill.generate_metadata()
        regimes = meta['regime_distribution']
        assert regimes['risk_on'] + regimes['risk_off'] + regimes['neutral'] == meta['total_signals']

    def test_component_availability(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-12-31')
        meta = backfill.generate_metadata()
        avail = meta['component_availability']
        assert avail['news'] == 100.0  # News always available

    def test_crisis_analysis(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-12-31')
        meta = backfill.generate_metadata()
        assert 'covid_crash' in meta['crisis_period_analysis']
        assert meta['crisis_period_analysis']['covid_crash']['count'] > 0

    def test_all_crisis_periods_in_analysis(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2022-12-31')
        meta = backfill.generate_metadata()
        analysis = meta['crisis_period_analysis']
        for period in ('covid_crash', 'covid_recovery', 'bear_2022', 'normal'):
            assert period in analysis, f"Missing period: {period}"
            assert 'count' in analysis[period]
            assert 'avg_sentiment' in analysis[period]

    def test_metadata_values_reasonable(self):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-12-31')
        meta = backfill.generate_metadata()
        assert 200 <= meta['total_signals'] <= 400  # ~366 days in 2020
        assert 0 < meta['avg_confidence'] <= 1.0
        comp_avail = meta['component_availability']
        assert comp_avail['news'] == 100.0
        assert 0 < comp_avail['earnings'] <= 100.0
        assert 0 < comp_avail['jobs'] <= 100.0
        assert 0 < comp_avail['social'] <= 100.0


# ---------------------------------------------------------------------------
# save_to_json Tests
# ---------------------------------------------------------------------------

class TestSaveToJson:

    def test_creates_file(self, tmp_path):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-05')
        path = str(tmp_path / "output.json")
        backfill.save_to_json(path)
        assert os.path.exists(path)

    def test_valid_json(self, tmp_path):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-05')
        path = str(tmp_path / "output.json")
        backfill.save_to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert 'metadata' in data
        assert 'signals' in data
        assert len(data['signals']) == 5

    def test_save_empty_signals(self, tmp_path):
        backfill = AlternativeDataBackfill(seed=42)
        path = str(tmp_path / "empty.json")
        # Should not crash even with no signals
        backfill.save_to_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data['metadata']['total_days'] == 0
        assert data['signals'] == []

    def test_save_nested_directory(self, tmp_path):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-03')
        path = str(tmp_path / "a" / "b" / "c" / "nested.json")
        backfill.save_to_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data['signals']) == 3

    def test_saved_metadata_fields(self, tmp_path):
        backfill = AlternativeDataBackfill(seed=42)
        backfill.generate_backfill('2020-01-01', '2020-01-03')
        path = str(tmp_path / "meta_check.json")
        backfill.save_to_json(path)
        with open(path) as f:
            data = json.load(f)
        meta = data['metadata']
        assert meta['version'] == '2.60'
        assert meta['phase'] == '4.1'
        assert meta['start_date'] == '2020-01-01'
        assert meta['end_date'] == '2020-01-03'
        assert meta['total_days'] == 3
        assert 'generated_at' in meta


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_runs(self, capsys):
        from src.backtest.alternative_data_backfill import main
        with patch("sys.argv", ["alt_data_backfill.py"]):
            result = main()
        assert result == 0
