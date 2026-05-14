#!/usr/bin/env python3
"""
Tests for alternative_data_backfill.py — DailyAlternativeSignal dataclass,
crisis detection, signal generation, regime classification, composite scoring,
and metadata generation.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

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


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_runs(self, capsys):
        from src.backtest.alternative_data_backfill import main
        with patch("sys.argv", ["alt_data_backfill.py"]):
            result = main()
        assert result == 0
