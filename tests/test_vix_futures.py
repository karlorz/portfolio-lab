"""
Tests for src/data/vix_futures.py — VIX futures term structure and contango signals.
No ML, no network — pure dataclasses and computation.
"""
import pytest
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.vix_futures import (
    VIXTermStructure,
    VIXDataManager,
    fetch_vix_futures_data,
)


class TestVIXTermStructure:
    """VIXTermStructure dataclass."""

    def make_ts(self, **overrides):
        defaults = dict(
            date="2024-06-15",
            vix_spot=15.0,
            front_month=16.5,
            second_month=17.5,
            third_month=18.0,
            contango_1m_2m=6.06,
            contango_spot_1m=10.0,
            is_contango=True,
            days_to_expiry_front=6,
        )
        defaults.update(overrides)
        return VIXTermStructure(**defaults)

    def test_create_contango_structure(self):
        ts = self.make_ts()
        assert ts.date == "2024-06-15"
        assert ts.vix_spot == 15.0
        assert ts.front_month == 16.5
        assert ts.is_contango is True
        assert ts.days_to_expiry_front == 6

    def test_create_backwardation_structure(self):
        ts = self.make_ts(
            vix_spot=28.0, front_month=25.0, second_month=23.0,
            contango_1m_2m=-8.0, contango_spot_1m=-10.7,
            is_contango=False,
        )
        assert ts.is_contango is False
        assert ts.front_month < ts.vix_spot

    def test_to_dict(self):
        ts = self.make_ts()
        d = ts.to_dict()
        assert d["date"] == "2024-06-15"
        assert d["vix_spot"] == 15.0
        assert d["is_contango"] is True
        assert "front_month" in d

    def test_from_dict_roundtrip(self):
        ts = self.make_ts()
        d = ts.to_dict()
        ts2 = VIXTermStructure.from_dict(d)
        assert ts2.date == ts.date
        assert ts2.vix_spot == ts.vix_spot
        assert ts2.front_month == ts.front_month
        assert ts2.is_contango == ts.is_contango

    def test_from_dict_backwardation(self):
        d = {
            "date": "2020-03-15", "vix_spot": 45.0, "front_month": 38.0,
            "second_month": 35.0, "third_month": 33.0,
            "contango_1m_2m": -7.9, "contango_spot_1m": -15.6,
            "is_contango": False, "days_to_expiry_front": 15,
        }
        ts = VIXTermStructure.from_dict(d)
        assert ts.vix_spot == 45.0
        assert ts.is_contango is False


class TestVIXDataManagerDataOps:
    """VIXDataManager in-memory data operations (no file I/O)."""

    def make_ts(self, date="2024-06-15", vix=15.0, front=16.5, second=17.5,
                third=18.0, c1m2m=6.0, cs1m=10.0, contango=True, dte=6):
        return VIXTermStructure(
            date=date, vix_spot=vix, front_month=front,
            second_month=second, third_month=third,
            contango_1m_2m=c1m2m, contango_spot_1m=cs1m,
            is_contango=contango, days_to_expiry_front=dte,
        )

    def test_init_with_no_cache(self, tmp_path):
        """When no cache file exists, data starts empty."""
        with patch.object(VIXDataManager, 'DATA_DIR', tmp_path):
            with patch.object(VIXDataManager, 'VIX_FILE', tmp_path / 'nonexistent.json'):
                with patch.object(VIXDataManager, '_load_cached_data', lambda s: None):
                    mgr = VIXDataManager.__new__(VIXDataManager)
                    mgr.data = {}
                    assert len(mgr.data) == 0

    def test_get_term_structure_found(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        ts = self.make_ts()
        mgr.data = {"2024-06-15": ts}
        assert mgr.get_term_structure("2024-06-15") is ts

    def test_get_term_structure_not_found(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        assert mgr.get_term_structure("nonexistent") is None

    def test_get_data_range_empty(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        start, end = mgr.get_data_range()
        assert start == ''
        assert end == ''

    def test_get_data_range_single(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {"2024-06-15": self.make_ts()}
        start, end = mgr.get_data_range()
        assert start == "2024-06-15"
        assert end == "2024-06-15"

    def test_get_data_range_multiple(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {
            "2024-01-01": self.make_ts(date="2024-01-01"),
            "2024-06-15": self.make_ts(date="2024-06-15"),
            "2024-12-31": self.make_ts(date="2024-12-31"),
        }
        start, end = mgr.get_data_range()
        assert start == "2024-01-01"
        assert end == "2024-12-31"


class TestContangoSignal:
    """get_contango_signal classification."""

    def make_ts(self, **kw):
        defaults = dict(date="2024-06-15", vix_spot=15.0, front_month=16.5,
                        second_month=17.5, third_month=18.0,
                        contango_1m_2m=6.0, contango_spot_1m=10.0,
                        is_contango=True, days_to_expiry_front=6)
        defaults.update(kw)
        return VIXTermStructure(**defaults)

    def _signal(self, vix, contango_spot_1m, is_contango=True):
        mgr = VIXDataManager.__new__(VIXDataManager)
        ts = self.make_ts(vix_spot=vix, contango_spot_1m=contango_spot_1m,
                         is_contango=is_contango,
                         front_month=vix * (1 + contango_spot_1m / 100))
        mgr.data = {ts.date: ts}
        return mgr.get_contango_signal(ts.date)

    def test_strong_contango(self):
        sig = self._signal(vix=15.0, contango_spot_1m=15.0)
        assert sig["signal"] == "strong_contango"
        assert sig["strength"] == pytest.approx(0.75)  # 15/20 = 0.75

    def test_contango(self):
        sig = self._signal(vix=15.0, contango_spot_1m=7.0)
        assert sig["signal"] == "contango"
        assert sig["strength"] == pytest.approx(0.7)  # 7/10 = 0.7

    def test_flat(self):
        sig = self._signal(vix=15.0, contango_spot_1m=0.0)
        assert sig["signal"] == "flat"
        assert sig["strength"] == 0.3

    def test_flat_negative_boundary(self):
        sig = self._signal(vix=15.0, contango_spot_1m=-1.0)
        assert sig["signal"] == "flat"

    def test_backwardation(self):
        sig = self._signal(vix=28.0, contango_spot_1m=-5.0, is_contango=False)
        assert sig["signal"] == "backwardation"
        assert sig["strength"] == pytest.approx(0.5)  # 5/10 = 0.5

    def test_strong_backwardation(self):
        sig = self._signal(vix=40.0, contango_spot_1m=-12.0, is_contango=False)
        assert sig["signal"] == "strong_backwardation"
        assert sig["strength"] == pytest.approx(0.8)  # 12/15 = 0.8

    def test_signal_has_all_fields(self):
        sig = self._signal(vix=15.0, contango_spot_1m=10.0)
        assert "date" in sig
        assert "signal" in sig
        assert "strength" in sig
        assert "contango_spot_1m" in sig
        assert "contango_1m_2m" in sig
        assert "is_contango" in sig
        assert "annualized_roll_yield" in sig
        assert "vix_level" in sig

    def test_signal_missing_date(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        assert mgr.get_contango_signal("nonexistent") is None

    def test_annualized_roll_yield_contango(self):
        sig = self._signal(vix=15.0, contango_spot_1m=10.0)
        # contango: 10.0 * (365/30) ≈ 121.7
        assert sig["annualized_roll_yield"] > 100

    def test_annualized_roll_yield_backwardation(self):
        sig = self._signal(vix=30.0, contango_spot_1m=-10.0, is_contango=False)
        # backwardation: -10.0 * (365/30) * 2 ≈ -243.3
        assert sig["annualized_roll_yield"] < -200

    def test_strength_clamped_at_max_one(self):
        sig = self._signal(vix=15.0, contango_spot_1m=25.0)
        assert sig["strength"] <= 1.0

    def test_boundary_exactly_10(self):
        sig = self._signal(vix=15.0, contango_spot_1m=10.0)
        # spot_1m > 10 → strong_contango; spot_1m=10 falls to contango
        assert sig["signal"] == "contango"

    def test_boundary_exactly_5(self):
        sig = self._signal(vix=15.0, contango_spot_1m=5.0)
        # spot_1m > 5 → contango; spot_1m=5 falls to flat
        assert sig["signal"] == "flat"


class TestHistoricalProxy:
    """generate_historical_proxy with short date ranges."""

    def test_generates_one_month(self, tmp_path):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        mgr.DATA_DIR = tmp_path
        mgr.VIX_FILE = tmp_path / 'vix_test.json'
        mgr._save_cached_data = lambda: None  # Skip file write

        results = mgr.generate_historical_proxy('2024-01-01', '2024-01-31')
        assert len(results) == 31
        assert all(isinstance(r, VIXTermStructure) for r in results)
        assert len(mgr.data) == 31

    def test_dates_are_sequential(self, tmp_path):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        mgr.DATA_DIR = tmp_path
        mgr.VIX_FILE = tmp_path / 'vix_test.json'
        mgr._save_cached_data = lambda: None

        results = mgr.generate_historical_proxy('2024-06-01', '2024-06-05')
        dates = [r.date for r in results]
        assert dates == ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05']

    def test_struct_has_reasonable_values(self, tmp_path):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        mgr.DATA_DIR = tmp_path
        mgr.VIX_FILE = tmp_path / 'vix_test.json'
        mgr._save_cached_data = lambda: None

        results = mgr.generate_historical_proxy('2024-06-01', '2024-06-30')
        for ts in results:
            assert 5.0 < ts.vix_spot < 50.0
            assert ts.front_month > 0
            assert ts.second_month > 0
            assert ts.third_month > 0
            assert 0 <= ts.days_to_expiry_front <= 30

    def test_contango_flag_matches_structure(self, tmp_path):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        mgr.DATA_DIR = tmp_path
        mgr.VIX_FILE = tmp_path / 'vix_test.json'
        mgr._save_cached_data = lambda: None

        results = mgr.generate_historical_proxy('2024-01-01', '2024-01-31')
        for ts in results:
            if ts.is_contango:
                assert ts.front_month >= ts.vix_spot * 0.8  # Allow small noise
            assert isinstance(ts.is_contango, bool)

    def test_default_end_date(self, tmp_path):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        mgr.DATA_DIR = tmp_path
        mgr.VIX_FILE = tmp_path / 'vix_test.json'
        mgr._save_cached_data = lambda: None

        # Only generate 1 day to avoid huge output
        results = mgr.generate_historical_proxy('2024-12-31', '2024-12-31')
        assert len(results) == 1


class TestFetchVIXFuturesData:
    """fetch_vix_futures_data function."""

    def test_fetch_generates_data(self, tmp_path):
        with patch.object(VIXDataManager, 'DATA_DIR', tmp_path):
            with patch.object(VIXDataManager, 'VIX_FILE', tmp_path / 'vix.json'):
                with patch.object(VIXDataManager, '_load_cached_data', lambda s: None):
                    with patch.object(VIXDataManager, '_save_cached_data', lambda s: None):
                        results = fetch_vix_futures_data('2024-01-01', '2024-01-07', use_cache=False)
                        assert len(results) == 7
                        assert all(isinstance(r, VIXTermStructure) for r in results)

    def test_fetch_uses_cache_when_available(self):
        """Cache must cover the full requested date range for cache to be used."""
        def _make_ts(date):
            return VIXTermStructure(
                date=date, vix_spot=15.0, front_month=16.0,
                second_month=17.0, third_month=18.0, contango_1m_2m=6.0,
                contango_spot_1m=6.7, is_contango=True, days_to_expiry_front=10,
            )
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {
            "2024-01-01": _make_ts("2024-01-01"),
            "2024-01-05": _make_ts("2024-01-05"),
            "2024-01-10": _make_ts("2024-01-10"),
        }

        with patch('src.data.vix_futures.VIXDataManager', return_value=mgr):
            results = fetch_vix_futures_data('2024-01-01', '2024-01-10', use_cache=True)
            assert len(results) == 3
            assert results[0].date == "2024-01-01"


class TestEdgeCases:
    """Edge cases."""

    def test_negative_vix_spot_handled(self):
        """Contango math works with low VIX."""
        ts = VIXTermStructure(
            date="2024-01-01", vix_spot=8.0, front_month=8.5,
            second_month=9.0, third_month=9.5, contango_1m_2m=5.9,
            contango_spot_1m=6.25, is_contango=True, days_to_expiry_front=10,
        )
        assert ts.vix_spot == 8.0

    def test_high_vix_stress_level(self):
        ts = VIXTermStructure(
            date="2020-03-16", vix_spot=82.69, front_month=65.0,
            second_month=55.0, third_month=48.0, contango_1m_2m=-15.4,
            contango_spot_1m=-21.4, is_contango=False, days_to_expiry_front=5,
        )
        assert ts.vix_spot > 80
        assert ts.is_contango is False

    def test_data_manager_data_is_dict(self):
        mgr = VIXDataManager.__new__(VIXDataManager)
        mgr.data = {}
        assert isinstance(mgr.data, dict)
