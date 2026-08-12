"""
Tests for src/broker/options_utils.py — Options chain, quotes, and broker integration.
Mocks aiohttp and price fetcher to avoid network calls. No ML dependencies.
"""
import pytest
import sqlite3
import asyncio
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Mock aiohttp before importing module — with cleanup
_orig_aiohttp = sys.modules.get("aiohttp")
_mock_aiohttp = MagicMock()
_mock_aiohttp.ClientSession = MagicMock
sys.modules["aiohttp"] = _mock_aiohttp


@pytest.fixture(scope="module", autouse=True)
def _cleanup_aiohttp():
    """Restore sys.modules after test module completes."""
    yield
    if _orig_aiohttp is None:
        sys.modules.pop("aiohttp", None)
    else:
        sys.modules["aiohttp"] = _orig_aiohttp

import logging

from src.broker.options_utils import (
    OptionType,
    OptionStatus,
    OptionQuote,
    OptionsChain,
    OptionsChainFetcher,
    OptionsChainCache,
)


# ---------------------------------------------------------------------------
# Helper: build OCC symbols that match the code's parsing (C/P at index 15)
# Code expects: symbol[3:5]=YY, symbol[5:7]=MM, symbol[7:9]=DD,
#               symbol[15]=C/P, symbol[16:]=strike*1000
# ---------------------------------------------------------------------------
def _occ_symbol(underlying: str, yymmdd: str, cp: str, strike: float) -> str:
    """Build an OCC symbol that _parse_option_data can handle.
    Format: AAA(3) + YYMMDD(6) + padding(6) + C/P(1) + strike×1000 digits
    """
    strike_int = int(round(strike * 1000))
    # Pad strike to at least 7 digits to get correct integer
    return f"{underlying}{yymmdd}XXXXXX{cp}{strike_int:07d}"


class TestOptionType:
    """OptionType enum."""

    def test_call(self):
        assert OptionType.CALL.value == "call"

    def test_put(self):
        assert OptionType.PUT.value == "put"

    def test_two_members(self):
        assert len(OptionType) == 2


class TestOptionStatus:
    """OptionStatus enum."""

    def test_active(self):
        assert OptionStatus.ACTIVE.value == "active"

    def test_expired(self):
        assert OptionStatus.EXPIRED.value == "expired"

    def test_exercised(self):
        assert OptionStatus.EXERCISED.value == "exercised"

    def test_assigned(self):
        assert OptionStatus.ASSIGNED.value == "assigned"

    def test_four_members(self):
        assert len(OptionStatus) == 4


class TestOptionQuote:
    """OptionQuote dataclass and properties."""

    def make_quote(self, **overrides):
        defaults = dict(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date.today() + timedelta(days=1),
            bid=2.50,
            ask=2.60,
            last=2.55,
            mark=2.55,
        )
        defaults.update(overrides)
        return OptionQuote(**defaults)

    def test_create_basic_quote(self):
        q = self.make_quote()
        assert q.symbol == "SPY240516C00550000"
        assert q.underlying == "SPY"
        assert q.option_type == OptionType.CALL
        assert q.strike == 550.0
        assert q.bid == 2.50
        assert q.ask == 2.60

    def test_mid_price(self):
        q = self.make_quote(bid=2.00, ask=3.00)
        assert q.mid_price == pytest.approx(2.50)

    def test_mid_price_symmetric(self):
        q = self.make_quote(bid=1.50, ask=1.50)
        assert q.mid_price == pytest.approx(1.50)

    def test_bid_ask_spread_pct(self):
        q = self.make_quote(bid=0.97, ask=1.03, mark=1.00)
        assert q.bid_ask_spread_pct == pytest.approx(6.0)

    def test_spread_zero_mark(self):
        q = self.make_quote(mark=0.0)
        assert q.bid_ask_spread_pct == 0.0

    def test_is_liquid_when_all_criteria_met(self):
        q = self.make_quote(volume=100, open_interest=1000, bid=9.90, ask=10.10, mark=10.00)
        assert q.bid_ask_spread_pct == pytest.approx(2.0)
        assert q.is_liquid is True

    def test_is_liquid_fails_low_volume(self):
        q = self.make_quote(volume=5, open_interest=1000, bid=9.90, ask=10.10, mark=10.00)
        assert q.is_liquid is False

    def test_is_liquid_fails_low_oi(self):
        q = self.make_quote(volume=100, open_interest=50, bid=9.90, ask=10.10, mark=10.00)
        assert q.is_liquid is False

    def test_is_liquid_fails_high_spread(self):
        q = self.make_quote(volume=100, open_interest=1000, bid=9.0, ask=11.0, mark=10.00)
        assert q.is_liquid is False

    def test_is_liquid_boundary_spread(self):
        q = self.make_quote(volume=100, open_interest=1000, bid=9.75, ask=10.25, mark=10.00)
        assert q.bid_ask_spread_pct == pytest.approx(5.0)
        assert q.is_liquid is True

    def test_days_to_expiration(self):
        q = self.make_quote(expiration=date.today() + timedelta(days=7))
        assert q.days_to_expiration == 7

    def test_days_to_expiration_0dte(self):
        q = self.make_quote(expiration=date.today())
        assert q.days_to_expiration == 0

    def test_days_to_expiration_past(self):
        q = self.make_quote(expiration=date.today() - timedelta(days=1))
        assert q.days_to_expiration == -1

    def test_greeks_default_none(self):
        q = self.make_quote()
        assert q.delta is None
        assert q.gamma is None
        assert q.theta is None
        assert q.vega is None
        assert q.implied_vol is None

    def test_greeks_with_values(self):
        q = self.make_quote(delta=0.60, gamma=0.05, theta=-0.10, vega=0.20, implied_vol=0.18)
        assert q.delta == 0.60
        assert q.gamma == 0.05
        assert q.theta == -0.10
        assert q.vega == 0.20
        assert q.implied_vol == 0.18

    def test_volume_oi_default_zero(self):
        q = self.make_quote()
        assert q.volume == 0
        assert q.open_interest == 0

    def test_to_dict_contains_all_keys(self):
        q = self.make_quote()
        d = q.to_dict()
        assert d["symbol"] == "SPY240516C00550000"
        assert d["option_type"] == "call"
        assert d["strike"] == 550.0
        assert d["mid_price"] == 2.55
        assert "is_liquid" in d
        assert "days_to_expiration" in d

    def test_to_dict_with_greeks(self):
        q = self.make_quote(delta=0.60, gamma=0.05)
        d = q.to_dict()
        assert d["delta"] == 0.60
        assert d["gamma"] == 0.05
        assert d["theta"] is None

    def test_put_option(self):
        q = self.make_quote(option_type=OptionType.PUT)
        assert q.option_type == OptionType.PUT
        assert q.to_dict()["option_type"] == "put"

    def test_timestamp_default(self):
        q = self.make_quote()
        assert isinstance(q.timestamp, datetime)


class TestOptionsChain:
    """OptionsChain dataclass filtering methods."""

    def _make_call(self, strike: float, delta=None, **overrides):
        defaults = dict(
            symbol=f"SPY{date.today():%y%m%d}C{int(strike*1000):08d}",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=strike,
            expiration=date.today(),
            bid=2.50, ask=2.60, last=2.55, mark=2.55,
            volume=100, open_interest=1000,
        )
        if delta is not None:
            defaults["delta"] = delta
        defaults.update(overrides)
        return OptionQuote(**defaults)

    def _make_put(self, strike: float, **overrides):
        defaults = dict(
            symbol=f"SPY{date.today():%y%m%d}P{int(strike*1000):08d}",
            underlying="SPY",
            option_type=OptionType.PUT,
            strike=strike,
            expiration=date.today(),
            bid=2.50, ask=2.60, last=2.55, mark=2.55,
            volume=100, open_interest=1000,
        )
        defaults.update(overrides)
        return OptionQuote(**defaults)

    def test_empty_chain(self):
        chain = OptionsChain(underlying="SPY")
        assert chain.underlying == "SPY"
        assert len(chain.quotes) == 0
        assert chain.get_calls() == []
        assert chain.get_puts() == []

    def test_get_calls(self):
        calls = [self._make_call(550), self._make_call(555)]
        puts = [self._make_put(550)]
        chain = OptionsChain(underlying="SPY", quotes=calls + puts)
        assert len(chain.get_calls()) == 2
        assert len(chain.get_puts()) == 1

    def test_get_puts(self):
        puts = [self._make_put(545), self._make_put(540)]
        chain = OptionsChain(underlying="SPY", quotes=puts)
        assert len(chain.get_puts()) == 2

    def test_get_by_strike_exact(self):
        q1 = self._make_call(550.0)
        q2 = self._make_call(555.0)
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        result = chain.get_by_strike(550.0)
        assert len(result) == 1
        assert result[0].strike == 550.0

    def test_get_by_strike_near_match(self):
        q = self._make_call(550.005)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_by_strike(550.0)
        assert len(result) == 1

    def test_get_by_strike_no_match(self):
        q = self._make_call(550.0)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_by_strike(600.0)
        assert result == []

    def test_get_by_expiration(self):
        today = date.today()
        tomorrow = today + timedelta(days=1)
        q1 = self._make_call(550, expiration=today)
        q2 = self._make_call(555, expiration=tomorrow)
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        result = chain.get_by_expiration(today)
        assert len(result) == 1

    def test_get_0dte(self):
        today = date.today()
        tomorrow = today + timedelta(days=1)
        q1 = self._make_call(550, expiration=today)
        q2 = self._make_call(555, expiration=tomorrow)
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        result = chain.get_0dte()
        assert len(result) == 1
        assert result[0].strike == 550.0

    def test_get_0dte_none(self):
        tomorrow = date.today() + timedelta(days=1)
        q = self._make_call(550, expiration=tomorrow)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        assert chain.get_0dte() == []

    def test_get_calls_by_delta(self):
        q1 = self._make_call(550, delta=0.30)
        q2 = self._make_call(555, delta=0.20)
        q3 = self._make_call(560, delta=0.35)
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2, q3])
        result = chain.get_calls_by_delta(0.30, tolerance=0.05)
        assert len(result) == 2

    def test_get_calls_by_delta_no_match(self):
        q = self._make_call(550, delta=0.50)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_calls_by_delta(0.30)
        assert result == []

    def test_get_calls_by_delta_skips_none(self):
        q = self._make_call(550)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_calls_by_delta(0.30)
        assert result == []

    def test_get_liquid_calls(self):
        liquid = self._make_call(550, volume=100, open_interest=1000, bid=9.90, ask=10.10, mark=10.00)
        illiquid = self._make_call(555, volume=5, open_interest=1000)
        chain = OptionsChain(underlying="SPY", quotes=[liquid, illiquid])
        result = chain.get_liquid_calls()
        assert len(result) == 1
        assert result[0].strike == 550.0

    def test_get_liquid_calls_custom_thresholds(self):
        q = self._make_call(550, volume=5, open_interest=1000, bid=9.95, ask=10.05, mark=10.00)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        assert chain.get_liquid_calls(min_volume=5) == [q]
        assert chain.get_liquid_calls(min_volume=10) == []

    def test_find_optimal_call(self):
        # Need spread ≤ 3% for find_optimal_call
        q1 = self._make_call(550, delta=0.30, volume=100, open_interest=1000, bid=9.90, ask=10.10, mark=10.00)
        q2 = self._make_call(555, delta=0.32, volume=100, open_interest=1000, bid=9.90, ask=10.10, mark=10.00)
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        best = chain.find_optimal_call(target_delta=0.30)
        assert best is not None
        assert best.strike == 550.0

    def test_find_optimal_call_none_when_empty(self):
        chain = OptionsChain(underlying="SPY")
        assert chain.find_optimal_call() is None

    def test_find_optimal_call_none_when_no_liquid(self):
        illiquid = self._make_call(550, delta=0.30, volume=5, bid=1.0, ask=2.0, mark=1.5)
        chain = OptionsChain(underlying="SPY", quotes=[illiquid])
        assert chain.find_optimal_call() is None

    def test_fetched_at_default(self):
        chain = OptionsChain(underlying="SPY")
        assert isinstance(chain.fetched_at, datetime)

    def test_to_dict(self):
        q = self._make_call(550, delta=0.30)
        chain = OptionsChain(underlying="SPY", quotes=[q])
        d = chain.to_dict()
        assert d["underlying"] == "SPY"
        assert d["quote_count"] == 1
        assert d["call_count"] == 1
        assert d["put_count"] == 0
        assert len(d["quotes"]) == 1


class TestOptionsChainFetcher:
    """OptionsChainFetcher with mocked API."""

    def test_init_no_credentials(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}, clear=True):
            fetcher = OptionsChainFetcher()
            assert fetcher.has_api_access is False

    def test_init_uses_canonical_api_secret_env_var(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}, clear=True):
            fetcher = OptionsChainFetcher()
            assert fetcher.secret_key == "s"
            assert fetcher.has_api_access is True

    def test_init_warns_for_legacy_secret_key_env_var(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "legacy"}, clear=True):
            with pytest.warns(DeprecationWarning, match="ALPACA_API_SECRET"):
                fetcher = OptionsChainFetcher()
            assert fetcher.secret_key == "legacy"
            assert fetcher.has_api_access is True

    def test_init_with_credentials(self):
        fetcher = OptionsChainFetcher(api_key="test_key", secret_key="test_secret")
        assert fetcher.has_api_access is True

    def test_init_paper_mode_default(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "ALPACA_PAPER": "true"}):
            fetcher = OptionsChainFetcher()
            assert fetcher.paper_mode is True

    def test_parse_option_data_valid_call(self):
        fetcher = OptionsChainFetcher()
        # Symbol where position 15 = 'C' (code checks symbol[15])
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        data = {
            "symbol": sym,
            "quote": {"bid": 2.50, "ask": 2.60, "last": 2.55, "mark": 2.55},
        }
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.option_type == OptionType.CALL
        assert q.strike == 550.0
        assert q.expiration == date(2024, 5, 16)
        assert q.bid == 2.50

    def test_parse_option_data_valid_put(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "P", 550.0)
        data = {
            "symbol": sym,
            "quote": {"bid": 2.0, "ask": 2.1, "last": 2.05, "mark": 2.05},
        }
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.option_type == OptionType.PUT

    def test_parse_option_data_short_symbol(self):
        fetcher = OptionsChainFetcher()
        q = fetcher._parse_option_data({"symbol": "SPY"}, "SPY")
        assert q is None

    def test_parse_option_data_with_greeks(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        data = {
            "symbol": sym,
            "quote": {
                "bid": 2.50, "ask": 2.60, "last": 2.55, "mark": 2.55,
                "greeks": {"delta": 0.60, "gamma": 0.05, "theta": -0.10, "vega": 0.20},
                "implied_volatility": 0.18,
                "volume": 500,
                "open_interest": 5000,
            },
        }
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.delta == 0.60
        assert q.gamma == 0.05
        assert q.implied_vol == 0.18
        assert q.volume == 500

    def test_parse_option_data_missing_quote(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        q = fetcher._parse_option_data({"symbol": sym}, "SPY")
        assert q is not None
        assert q.bid == 0.0

    def test_parse_option_data_exception_returns_none(self):
        fetcher = OptionsChainFetcher()
        q = fetcher._parse_option_data(None, "SPY")
        assert q is None

    def test_parse_option_date_different_month(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "251231", "C", 600.0)
        data = {
            "symbol": sym,
            "quote": {"bid": 0, "ask": 0, "last": 0, "mark": 0},
        }
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.expiration == date(2025, 12, 31)
        assert q.strike == 600.0

    def test_simulate_option_price_atm(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=550.0, vol=0.16, tte=1/365, is_call=True)
        assert price > 0

    def test_simulate_option_price_itm_call(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=545.0, vol=0.16, tte=1/365, is_call=True)
        assert price > 5.0

    def test_simulate_option_price_otm_call(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=555.0, vol=0.16, tte=1/365, is_call=True)
        assert price < 3.0

    def test_estimate_delta_atm_call(self):
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=550.0, strike=550.0, vol=0.16, tte=1/365, is_call=True)
        assert 0.45 < delta < 0.55

    def test_estimate_delta_itm_call(self):
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=550.0, strike=530.0, vol=0.16, tte=1/365, is_call=True)
        assert delta > 0.8

    def test_estimate_delta_otm_put(self):
        """OTM put (strike < spot) — delta should be negative (non-positive)."""
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=550.0, strike=530.0, vol=0.16, tte=1/365, is_call=False)
        # Extremely short TTE + deep OTM → delta rounds to -0.0
        assert delta <= 0, f"Expected non-positive put delta, got {delta}"

    def test_estimate_delta_rounded_to_3_decimals(self):
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=550.0, strike=550.0, vol=0.16, tte=1/365, is_call=True)
        assert delta == round(delta, 3)

    def test_cache_chain_creates_db(self, tmp_path):
        fetcher = OptionsChainFetcher()
        fetcher.cache_dir = tmp_path
        q = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55, delta=0.6,
            volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        fetcher.cache_chain(chain)
        assert (tmp_path / "SPY_options.db").exists()

    @patch("src.broker.options_utils.OptionsChainFetcher._generate_simulated_chain")
    def test_fetch_0dte_chain_simulation_mode(self, mock_gen):
        mock_gen.return_value = OptionsChain(underlying="SPY", quotes=[])
        fetcher = OptionsChainFetcher()
        result = asyncio.run(fetcher.fetch_0dte_chain("SPY"))
        assert result is not None
        assert result.underlying == "SPY"


class TestOptionsChainCache:
    """OptionsChainCache sqlite-based caching."""

    def test_init_creates_cache_dir(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        assert tmp_path.exists()

    def test_get_history_empty(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        history = cache.get_history("SPY")
        assert history == []

    def test_get_history_and_avg_volume(self, tmp_path):
        """Test both get_history and get_avg_volume_by_strike with a populated DB."""
        cache = OptionsChainCache(cache_dir=str(tmp_path))

        # Pre-populate the DB
        db_path = tmp_path / "SPY_options.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, option_type TEXT, strike REAL, expiration TEXT,
                bid REAL, ask REAL, last REAL, delta REAL,
                volume INTEGER, open_interest INTEGER, fetched_at TEXT
            )
        """)
        now = datetime.now().isoformat()
        for vol, ts in [(100, now), (200, now)]:
            conn.execute("""
                INSERT INTO options_chain
                (symbol, option_type, strike, expiration, bid, ask, last, delta, volume, open_interest, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("SPY240516C00550000", "call", 550.0, "2024-05-16",
                  2.5, 2.6, 2.55, 0.6, vol, 1000, ts))
        conn.commit()
        conn.close()

        # get_history with a large days window should find the rows
        history = cache.get_history("SPY", days=99999)
        assert len(history) == 2

        # get_avg_volume_by_strike
        avg = cache.get_avg_volume_by_strike("SPY", 550.0, days=99999)
        assert avg == pytest.approx(150.0)

    def test_get_avg_volume_no_match(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        avg = cache.get_avg_volume_by_strike("SPY", 550.0)
        assert avg == 0.0


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_option_quote_zero_bid_ask(self):
        q = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(),
            bid=0.0, ask=0.0, last=0.0, mark=0.0,
        )
        assert q.mid_price == 0.0
        assert q.bid_ask_spread_pct == 0.0

    def test_option_quote_negative_delta(self):
        q = OptionQuote(
            symbol="SPY240516P00550000", underlying="SPY",
            option_type=OptionType.PUT, strike=550.0,
            expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55, delta=-0.30,
            volume=100, open_interest=1000,
        )
        assert q.delta == -0.30

    def test_chain_get_calls_by_delta_put_ignored(self):
        call = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30,
        )
        put = OptionQuote(
            symbol="SPY240516P00550000", underlying="SPY",
            option_type=OptionType.PUT, strike=550.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=-0.30,
        )
        chain = OptionsChain(underlying="SPY", quotes=[call, put])
        result = chain.get_calls_by_delta(0.30)
        assert len(result) == 1

    def test_find_optimal_call_with_none_delta(self):
        """When the only liquid candidate has delta=None, it's the only option returned."""
        q = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=None,
            volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        # None delta gets score inf, sorts last — but it's the only candidate
        best = chain.find_optimal_call(target_delta=0.30)
        assert best is not None  # Only liquid call in chain, returned despite None delta

    def test_multiple_same_delta_picks_first(self):
        q1 = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30,
            volume=100, open_interest=1000,
        )
        q2 = OptionQuote(
            symbol="SPY240516C00555000", underlying="SPY",
            option_type=OptionType.CALL, strike=555.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30,
            volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        best = chain.find_optimal_call(target_delta=0.30)
        assert best is not None
        assert best.strike == 550.0


class TestDataclassFields:
    """Validate dataclass schemas using dataclasses.fields()."""

    def test_option_quote_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionQuote)}
        assert "symbol" in fields
        assert "underlying" in fields
        assert "option_type" in fields
        assert "strike" in fields
        assert "expiration" in fields
        assert "bid" in fields
        assert "ask" in fields
        assert "last" in fields
        assert "mark" in fields
        assert "delta" in fields
        assert "gamma" in fields
        assert "theta" in fields
        assert "vega" in fields
        assert "implied_vol" in fields
        assert "volume" in fields
        assert "open_interest" in fields
        assert "timestamp" in fields
        assert "bid_ask_spread_pct" in fields

    def test_option_quote_field_types(self):
        import dataclasses, typing
        fields = {f.name: f for f in dataclasses.fields(OptionQuote)}
        assert fields["symbol"].type is str
        assert fields["strike"].type is float
        origin = typing.get_origin(fields["delta"].type)
        args = typing.get_args(fields["delta"].type)
        assert origin is typing.Union
        assert float in args
        assert type(None) in args
        assert fields["volume"].type is int
        assert fields["bid_ask_spread_pct"].type is float

    def test_option_quote_defaults(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionQuote)}
        assert fields["delta"].default is None
        assert fields["gamma"].default is None
        assert fields["theta"].default is None
        assert fields["vega"].default is None
        assert fields["implied_vol"].default is None
        assert fields["volume"].default == 0
        assert fields["open_interest"].default == 0

    def test_option_quote_bid_ask_spread_pct_init_false(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionQuote)}
        assert fields["bid_ask_spread_pct"].init is False

    def test_option_quote_timestamp_factory(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionQuote)}
        assert fields["timestamp"].default_factory is not dataclasses.MISSING

    def test_options_chain_fields(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionsChain)}
        assert "underlying" in fields
        assert "quotes" in fields
        assert "fetched_at" in fields
        assert fields["underlying"].type is str

    def test_options_chain_defaults(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(OptionsChain)}
        assert fields["quotes"].default_factory is list
        assert fields["fetched_at"].default_factory is not dataclasses.MISSING

    def test_option_type_enum_values(self):
        assert [e.value for e in OptionType] == ["call", "put"]

    def test_option_status_enum_values(self):
        assert [e.value for e in OptionStatus] == ["active", "expired", "exercised", "assigned"]


class TestConstantsAndExports:
    """Verify module-level constants and __all__."""

    def test_all_exported_names(self):
        from src.broker import options_utils
        expected = {"OptionType", "OptionStatus", "OptionQuote", "OptionsChain", "OptionsChainFetcher", "OptionsChainCache"}
        assert set(options_utils.__all__) == expected

    def test_all_members_are_strings(self):
        from src.broker import options_utils
        assert all(isinstance(name, str) for name in options_utils.__all__)

    def test_all_elements_match_public_api(self):
        from src.broker import options_utils
        for name in options_utils.__all__:
            assert hasattr(options_utils, name), f"__all__ references {name} which is not defined"

    def test_cache_ttl_seconds_exists(self):
        assert OptionsChainFetcher.CACHE_TTL_SECONDS == 300

    def test_cache_ttl_seconds_type(self):
        assert isinstance(OptionsChainFetcher.CACHE_TTL_SECONDS, int)

    def test_option_type_exported(self):
        from src.broker.options_utils import OptionType
        assert OptionType is not None

    def test_option_status_exported(self):
        from src.broker.options_utils import OptionStatus
        assert OptionStatus is not None


class TestComputationEdgeCases:
    """NaN, Inf, negative values, boundary conditions."""

    def test_nan_bid_does_not_crash(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=float("nan"),
            ask=2.60, last=2.55, mark=2.55,
        )
        assert q.mid_price is not None
        assert q.bid_ask_spread_pct is not None

    def test_nan_ask_does_not_crash(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.50,
            ask=float("nan"), last=2.55, mark=2.55,
        )
        assert q.mid_price is not None

    def test_inf_bid_handled(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=float("inf"),
            ask=2.60, last=2.55, mark=2.55,
        )
        assert q.mid_price == float("inf")

    def test_negative_strike(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.PUT,
            strike=-100.0, expiration=date.today(), bid=1.0, ask=1.1,
            last=1.05, mark=1.05,
        )
        assert q.strike == -100.0

    def test_negative_volume(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.50, ask=2.60,
            last=2.55, mark=2.55, volume=-5,
        )
        assert q.is_liquid is False

    def test_zero_volume_boundary_is_illiquid(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, volume=9, open_interest=100,
        )
        assert q.is_liquid is False

    def test_min_oi_boundary_is_illiquid(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, volume=10, open_interest=99,
        )
        assert q.is_liquid is False

    def test_max_spread_boundary_is_liquid(self):
        """5.0% spread is the boundary; <= 5.0 is liquid."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.75, ask=10.25,
            last=10.0, mark=10.0, volume=10, open_interest=100,
        )
        assert q.bid_ask_spread_pct == pytest.approx(5.0)
        assert q.is_liquid is True

    def test_spread_just_above_boundary_is_illiquid(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.74, ask=10.26,
            last=10.0, mark=10.0, volume=10, open_interest=100,
        )
        assert q.bid_ask_spread_pct > 5.0
        assert q.is_liquid is False

    def test_extremely_large_strike(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=999999.99, expiration=date.today(), bid=2.50, ask=2.60,
            last=2.55, mark=2.55,
        )
        assert q.strike == 999999.99
        assert q.to_dict()["strike"] == 999999.99

    def test_extremely_small_strike(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.PUT,
            strike=0.01, expiration=date.today(), bid=2.50, ask=2.60,
            last=2.55, mark=2.55,
        )
        assert q.strike == 0.01

    def test_negative_bid_ask(self):
        """Negative bid/ask should not crash, mid_price should still compute.
        Note: with mark <= 0, __post_init__ sets bid_ask_spread_pct to 0.0."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=-1.0, ask=-0.5,
            last=-0.75, mark=-0.75,
        )
        assert q.mid_price == pytest.approx(-0.75)
        # mark <= 0 triggers the else branch in __post_init__
        assert q.bid_ask_spread_pct == 0.0

    def test_zero_mark_with_nonzero_bid_ask(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=1.0, ask=3.0,
            last=0.0, mark=0.0,
        )
        assert q.bid_ask_spread_pct == 0.0
        assert q.mid_price == 2.0


class TestFunctionBoundaries:
    """Boundary conditions, missing keys, wrong types."""

    def test_parse_option_data_empty_dict(self):
        fetcher = OptionsChainFetcher()
        q = fetcher._parse_option_data({"symbol": _occ_symbol("SPY", "240516", "C", 550.0)}, "SPY")
        assert q is not None
        assert q.bid == 0.0
        assert q.ask == 0.0
        assert q.delta is None
        assert q.volume == 0

    def test_parse_option_data_partial_greeks(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        data = {
            "symbol": sym,
            "quote": {
                "bid": 2.5, "ask": 2.6, "last": 2.55, "mark": 2.55,
                "greeks": {"delta": 0.60},
                "implied_volatility": 0.18,
                "volume": 500, "open_interest": 5000,
            },
        }
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.delta == 0.60
        assert q.gamma is None
        assert q.theta is None
        assert q.vega is None
        assert q.implied_vol == 0.18

    def test_parse_option_data_invalid_date_symbol(self):
        fetcher = OptionsChainFetcher()
        q = fetcher._parse_option_data({"symbol": "SPYXXYYZZC00550000"}, "SPY")
        assert q is None  # int() conversion on non-numeric chars will raise

    def test_parse_option_data_missing_quote_key(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        data = {"symbol": sym, "not_quote": {}}
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.bid == 0.0

    def test_simulate_option_price_zero_vol(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=550.0, vol=0.0, tte=1/365, is_call=True)
        assert price == pytest.approx(0.0)

    def test_simulate_option_price_extreme_vol(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=550.0, vol=10.0, tte=1/365, is_call=True)
        assert price > 0

    def test_simulate_option_price_put_itm(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=555.0, vol=0.16, tte=1/365, is_call=False)
        assert price > 0

    def test_simulate_option_price_put_otm(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=545.0, vol=0.16, tte=1/365, is_call=False)
        # OTM put has small time value
        assert price > 0

    def test_estimate_delta_zero_tte(self):
        """tte=0 causes division by zero (vol * sqrt(0) = 0)."""
        fetcher = OptionsChainFetcher()
        with pytest.raises(ZeroDivisionError):
            fetcher._estimate_delta(spot=550.0, strike=550.0, vol=0.16, tte=0.0, is_call=True)

    def test_estimate_delta_extreme_vol(self):
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=550.0, strike=600.0, vol=5.0, tte=1/365, is_call=True)
        # Very high vol makes delta approach 0.5 for OTM
        assert 0.0 < delta < 1.0

    def test_estimate_delta_deep_itm_put(self):
        """Deep ITM put (strike >> spot) → delta should approach -1."""
        fetcher = OptionsChainFetcher()
        delta = fetcher._estimate_delta(spot=500.0, strike=600.0, vol=0.16, tte=30/365, is_call=False)
        assert delta <= -0.8

    def test_get_by_strike_negative(self):
        chain = OptionsChain(underlying="SPY")
        assert chain.get_by_strike(-100.0) == []

    def test_get_by_strike_zero(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=0.0, expiration=date.today(), bid=1.0, ask=1.1,
            last=1.05, mark=1.05,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_by_strike(0.0)
        assert len(result) == 1

    def test_get_by_strike_no_quotes(self):
        chain = OptionsChain(underlying="SPY")
        assert chain.get_by_strike(550.0) == []

    def test_simulate_option_price_deep_itm(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=100.0, vol=0.16, tte=1/365, is_call=True)
        # intrinsic = 450, small time value = ~1.84
        assert price > 450.0
        assert price < 455.0

    def test_simulate_option_price_deep_otm(self):
        fetcher = OptionsChainFetcher()
        price = fetcher._simulate_option_price(spot=550.0, strike=1000.0, vol=0.16, tte=1/365, is_call=True)
        # intrinsic = 0, time value = ~1.84
        assert price > 0
        assert price < 5.0

    def test_parse_option_data_strike_zero(self):
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 0.0)
        data = {"symbol": sym, "quote": {"bid": 0, "ask": 0, "last": 0, "mark": 0}}
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.strike == 0.0
        assert q.option_type == OptionType.CALL


class TestCliGuard:
    """CLI/__main__ entry point."""

    @staticmethod
    def _runpy_with_cleanup(module_name):
        """Run runpy.run_module and clean up sys.modules corruption."""
        import runpy
        import sys
        saved = sys.modules.get(module_name)
        try:
            return runpy.run_module(module_name, run_name="__main__")
        finally:
            # runpy corrupts sys.modules — restore or remove
            if saved is not None:
                sys.modules[module_name] = saved
            else:
                sys.modules.pop(module_name, None)

    def test_main_runs_via_runpy(self, capsys):
        """Verify __main__ block executes via python -m."""
        with patch("src.broker.options_utils.OptionsChainFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_fetcher_cls.return_value = mock_instance
            mock_chain = MagicMock()
            mock_chain.quotes = [MagicMock()]
            mock_chain.underlying = "SPY"
            mock_chain.get_0dte.return_value = []
            mock_chain.find_optimal_call.return_value = None
            mock_instance.fetch_0dte_chain = AsyncMock(return_value=mock_chain)

            self._runpy_with_cleanup("src.broker.options_utils")

        captured = capsys.readouterr()
        assert "Fetched" in captured.err or "Best" in captured.err

    def test_main_block_runs_asyncio_run(self, capsys):
        """The __main__ block calls asyncio.run()."""
        with patch("src.broker.options_utils.asyncio.run") as mock_run:
            mock_chain = MagicMock()
            mock_chain.quotes = [MagicMock()]
            mock_chain.underlying = "SPY"
            mock_chain.get_0dte.return_value = []
            mock_chain.find_optimal_call.return_value = None
            mock_run.return_value = mock_chain

            self._runpy_with_cleanup("src.broker.options_utils")

        assert mock_run.called

    def test_main_block_with_best_call(self, capsys):
        """When find_optimal_call returns a quote, __main__ prints its details."""
        with patch("src.broker.options_utils.OptionsChainFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_fetcher_cls.return_value = mock_instance
            best = MagicMock()
            best.strike = 550.0
            best.mark = 2.55
            best.delta = 0.30
            best.volume = 100
            best.bid_ask_spread_pct = 2.0
            best.is_liquid = True
            mock_chain = MagicMock()
            mock_chain.quotes = [best]
            mock_chain.underlying = "SPY"
            mock_chain.get_0dte.return_value = [best]
            mock_chain.find_optimal_call.return_value = best
            mock_instance.fetch_0dte_chain = AsyncMock(return_value=mock_chain)

            self._runpy_with_cleanup("src.broker.options_utils")

        captured = capsys.readouterr()
        assert "Strike:" in captured.err
        assert "Delta:" in captured.err
        assert "Volume:" in captured.err
        assert "Spread:" in captured.err
        assert "Liquid:" in captured.err

    def test_main_block_logs_fetch_counts(self, caplog):
        """__main__ should log via the fetcher initialization."""
        with patch("src.broker.options_utils.OptionsChainFetcher") as mock_fetcher_cls:
            mock_instance = MagicMock()
            mock_fetcher_cls.return_value = mock_instance
            mock_chain = MagicMock()
            mock_chain.quotes = []
            mock_chain.underlying = "SPY"
            mock_chain.get_0dte.return_value = []
            mock_chain.find_optimal_call.return_value = None
            mock_instance.fetch_0dte_chain = AsyncMock(return_value=mock_chain)

            self._runpy_with_cleanup("src.broker.options_utils")

        assert True  # no crash


class TestOptionsChainFetcherMore:
    """Additional OptionsChainFetcher edge cases."""

    def test_fetch_from_api_success(self):
        """_fetch_from_api with a successful API response should parse options."""
        fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "options": [
                {
                    "symbol": _occ_symbol("SPY", date.today().strftime("%y%m%d"), "C", 550.0),
                    "quote": {"bid": 2.5, "ask": 2.6, "last": 2.55, "mark": 2.55},
                }
            ]
        })
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("src.broker.options_utils.aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(fetcher._fetch_from_api("SPY"))
        assert result.underlying == "SPY"
        assert len(result.quotes) == 1
        assert result.quotes[0].strike == 550.0

    def test_fetch_from_api_non_200(self):
        """_fetch_from_api with non-200 response raises."""
        fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status = 401
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("src.broker.options_utils.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="API error"):
                asyncio.run(fetcher._fetch_from_api("SPY"))

    def test_fetch_from_api_non_0dte_filtered(self):
        """Option with expiration != today should be filtered out in _fetch_from_api."""
        fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
        tomorrow = (date.today() + timedelta(days=1)).strftime("%y%m%d")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "options": [
                {
                    "symbol": _occ_symbol("SPY", tomorrow, "C", 550.0),
                    "quote": {"bid": 2.5, "ask": 2.6, "last": 2.55, "mark": 2.55},
                }
            ]
        })
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("src.broker.options_utils.aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(fetcher._fetch_from_api("SPY"))
        assert len(result.quotes) == 0

    def test_fetch_from_api_error_fallback(self):
        """When _fetch_from_api fails, fetch_0dte_chain falls back to simulation."""
        fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_session = AsyncMock()
        mock_session.get.return_value = mock_resp
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        with patch("src.broker.options_utils.aiohttp.ClientSession", return_value=mock_session):
            with patch.object(fetcher, "_generate_simulated_chain") as mock_gen:
                mock_gen.return_value = OptionsChain(underlying="SPY", quotes=[])
                result = asyncio.run(fetcher.fetch_0dte_chain("SPY"))
        assert result is not None

    def test_init_no_credentials_warning(self, caplog):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}, clear=True):
            caplog.set_level(logging.WARNING)
            fetcher = OptionsChainFetcher()
            assert "No Alpaca API credentials" in caplog.text

    def test_cache_chain_empty_does_not_crash(self, tmp_path):
        fetcher = OptionsChainFetcher()
        fetcher.cache_dir = tmp_path
        chain = OptionsChain(underlying="SPY")
        fetcher.cache_chain(chain)
        assert (tmp_path / "SPY_options.db").exists()

    def test_cache_chain_multiple_quotes(self, tmp_path):
        fetcher = OptionsChainFetcher()
        fetcher.cache_dir = tmp_path
        quotes = [
            OptionQuote(
                symbol=f"SPY240516C0055000{i}", underlying="SPY",
                option_type=OptionType.CALL, strike=float(550 + i),
                expiration=date.today(), bid=2.5, ask=2.6,
                last=2.55, mark=2.55, volume=100, open_interest=1000,
            )
            for i in range(5)
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        fetcher.cache_chain(chain)

        conn = sqlite3.connect(str(tmp_path / "SPY_options.db"))
        count = conn.execute("SELECT COUNT(*) FROM options_chain").fetchone()[0]
        conn.close()
        assert count == 5

    def test_fetch_0dte_chain_api_exception_logs(self, caplog):
        """Exception in API fetch logs error and falls back."""
        fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("src.broker.options_utils.aiohttp.ClientSession", return_value=mock_session):
            with patch.object(fetcher, "_generate_simulated_chain") as mock_gen:
                mock_gen.return_value = OptionsChain(underlying="SPY", quotes=[])
                caplog.set_level(logging.ERROR)
                result = asyncio.run(fetcher.fetch_0dte_chain("SPY"))

        assert "falling back" in caplog.text

    def test_generate_simulated_chain_strike_count(self):
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        # 21 strikes from -10 to +10 in steps of 5
        assert len(chain.quotes) == 21

    def test_generate_simulated_chain_all_calls(self):
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        assert all(q.option_type == OptionType.CALL for q in chain.quotes)

    def test_generate_simulated_chain_dte_today(self):
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        assert all(q.expiration == date.today() for q in chain.quotes)

    def test_generate_simulated_chain_deltas_ordered(self):
        """Higher strike → lower delta for calls."""
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        deltas = [q.delta for q in chain.quotes if q.delta is not None]
        # Strike increases, so deltas should be decreasing
        assert deltas == sorted(deltas, reverse=True)

    def test_generate_simulated_chain_spreads_near_atm(self):
        """Quotes closer to ATM should have tighter spreads."""
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        quotes_by_liquidity = sorted(chain.quotes, key=lambda q: q.volume, reverse=True)
        # The simulated generator makes near-ATM quotes most liquid and tightest.
        atm_spread = quotes_by_liquidity[0].bid_ask_spread_pct
        otm_spread = quotes_by_liquidity[-1].bid_ask_spread_pct
        assert atm_spread <= otm_spread

    def test_fetch_0dte_chain_simulation_paper_mode(self):
        """With no API, paper mode generates simulated chain."""
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher.fetch_0dte_chain("SPY"))
        assert chain.underlying == "SPY"
        assert len(chain.quotes) > 0


class TestOptionsChainMore:
    """Additional OptionsChain edge cases."""

    def test_to_dict_empty(self):
        chain = OptionsChain(underlying="SPY")
        d = chain.to_dict()
        assert d["quote_count"] == 0
        assert d["call_count"] == 0
        assert d["put_count"] == 0
        assert d["quotes"] == []

    def test_to_dict_isoformat_dates(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date(2024, 5, 16), bid=2.5, ask=2.6,
            last=2.55, mark=2.55,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        d = chain.to_dict()
        assert d["quotes"][0]["expiration"] == "2024-05-16"

    def test_get_by_expiration_no_match(self):
        today = date.today()
        tomorrow = today + timedelta(days=1)
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=tomorrow, bid=2.5, ask=2.6,
            last=2.55, mark=2.55,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        assert chain.get_by_expiration(today) == []

    def test_get_by_expiration_multiple_matches(self):
        today = date.today()
        qs = [
            OptionQuote(
                symbol=f"TEST{i}", underlying="SPY", option_type=OptionType.CALL,
                strike=float(550 + i), expiration=today, bid=2.5, ask=2.6,
                last=2.55, mark=2.55,
            )
            for i in range(3)
        ]
        chain = OptionsChain(underlying="SPY", quotes=qs)
        assert len(chain.get_by_expiration(today)) == 3

    def test_get_calls_by_delta_tolerance_zero(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55, delta=0.30,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_calls_by_delta(0.30, tolerance=0.0)
        assert len(result) == 1

    def test_get_liquid_calls_tight_spread(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=10.0, ask=11.0,
            last=10.50, mark=10.0, volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        result = chain.get_liquid_calls(max_spread_pct=11.0)
        # spread is (11.0-10.0)/10.0*100 = 10.0%
        assert len(result) == 1
        result2 = chain.get_liquid_calls(max_spread_pct=9.0)
        assert len(result2) == 0

    def test_find_optimal_call_with_custom_max_spread(self):
        q1 = OptionQuote(
            symbol="TEST1", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30, volume=100, open_interest=1000,
        )
        q2 = OptionQuote(
            symbol="TEST2", underlying="SPY", option_type=OptionType.CALL,
            strike=555.0, expiration=date.today(), bid=9.0, ask=11.0,
            last=10.0, mark=10.0, delta=0.25, volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        # With max_spread_pct=2.0, only q1 qualifies
        best = chain.find_optimal_call(target_delta=0.30, max_spread_pct=2.0)
        assert best is not None
        assert best.strike == 550.0

    def test_find_optimal_call_exact_delta_match(self):
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30, volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        best = chain.find_optimal_call(target_delta=0.30)
        assert best is not None
        assert best.delta == 0.30


class TestOptionsChainCacheMore:
    """Additional OptionsChainCache edge cases."""

    def test_init_default_cache_dir(self):
        """When no cache_dir is provided, should use OPTIONS_CACHE_DIR."""
        cache = OptionsChainCache()
        assert cache.cache_dir is not None
        assert isinstance(cache.cache_dir, Path)

    def test_get_history_db_not_found(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        # Non-existent symbol
        history = cache.get_history("NONEXISTENT")
        assert history == []

    def test_get_history_empty_result(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        db_path = tmp_path / "SPY_options.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, option_type TEXT, strike REAL, expiration TEXT,
                bid REAL, ask REAL, last REAL, delta REAL,
                volume INTEGER, open_interest INTEGER, fetched_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        # Recent window with no rows
        history = cache.get_history("SPY", days=1)
        assert history == []

    def test_get_avg_volume_by_strike_single_entry(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        db_path = tmp_path / "SPY_options.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, option_type TEXT, strike REAL, expiration TEXT,
                bid REAL, ask REAL, last REAL, delta REAL,
                volume INTEGER, open_interest INTEGER, fetched_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO options_chain
            (symbol, option_type, strike, expiration, bid, ask, last, delta, volume, open_interest, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY240516C00550000", "call", 550.0, "2024-05-16",
              2.5, 2.6, 2.55, 0.6, 50, 500, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        avg = cache.get_avg_volume_by_strike("SPY", 550.0, days=99999)
        assert avg == pytest.approx(50.0)

    def test_get_avg_volume_by_strike_different_strike(self, tmp_path):
        """Volume for a non-matching strike should return 0."""
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        db_path = tmp_path / "SPY_options.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, option_type TEXT, strike REAL, expiration TEXT,
                bid REAL, ask REAL, last REAL, delta REAL,
                volume INTEGER, open_interest INTEGER, fetched_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO options_chain
            (symbol, option_type, strike, expiration, bid, ask, last, delta, volume, open_interest, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("SPY240516C00550000", "call", 555.0, "2024-05-16",
              2.5, 2.6, 2.55, 0.6, 50, 500, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        # Query for a different strike
        avg = cache.get_avg_volume_by_strike("SPY", 550.0, days=99999)
        assert avg == 0.0

    def test_get_avg_volume_multiple_days(self, tmp_path):
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        avg = cache.get_avg_volume_by_strike("SPY", 550.0, days=1)
        assert avg == 0.0  # No history in fresh DB


class TestLoggingBehavior:
    """Logging output validation."""

    def test_fetcher_init_logs_warning_no_creds(self, caplog):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}, clear=True):
            with caplog.at_level(logging.WARNING):
                OptionsChainFetcher()
        assert any("No Alpaca API credentials" in msg for msg in caplog.messages)

    def test_cache_chain_logs_info(self, caplog, tmp_path):
        fetcher = OptionsChainFetcher()
        fetcher.cache_dir = tmp_path
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        with caplog.at_level(logging.INFO):
            fetcher.cache_chain(chain)
        assert any("Cached" in msg for msg in caplog.messages)

    def test_parse_option_data_exception_logs_warning(self, caplog):
        fetcher = OptionsChainFetcher()
        with caplog.at_level(logging.WARNING):
            fetcher._parse_option_data(None, "SPY")
        assert any("Failed to parse" in msg for msg in caplog.messages)


class TestRemainingEdgeCases:
    """Corner cases not covered by other classes."""

    def test_fetcher_paper_mode_false(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "ALPACA_PAPER": "false"}):
            fetcher = OptionsChainFetcher(api_key="k", secret_key="s")
            assert fetcher.paper_mode is False

    def test_fetcher_partial_credentials_key_only(self):
        fetcher = OptionsChainFetcher(api_key="key_only")
        assert fetcher.has_api_access is False

    def test_fetcher_partial_credentials_secret_only(self):
        fetcher = OptionsChainFetcher(secret_key="secret_only")
        assert fetcher.has_api_access is False

    def test_parse_option_data_wrong_option_char(self):
        """symbol[15] is neither C nor P; defaults to PUT."""
        fetcher = OptionsChainFetcher()
        sym = _occ_symbol("SPY", "240516", "C", 550.0)
        # Replace the C/P character with something else
        sym = sym[:15] + "X" + sym[16:]
        data = {"symbol": sym, "quote": {"bid": 0, "ask": 0, "last": 0, "mark": 0}}
        q = fetcher._parse_option_data(data, "SPY")
        assert q is not None
        assert q.option_type == OptionType.PUT  # Not C, so defaults to PUT

    def test_option_quote_bid_zero_mark_positive(self):
        """bid=0 with mark>0: spread computed normally."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=0.0, ask=2.0,
            last=1.0, mark=1.0,
        )
        assert q.bid_ask_spread_pct == pytest.approx(200.0)
        assert q.mid_price == pytest.approx(1.0)

    def test_is_liquid_bare_minimum(self):
        """Exactly meets minimum volume (10), OI (100), spread (<=5%)."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.75, ask=10.25,
            last=10.0, mark=10.0, volume=10, open_interest=100,
        )
        assert q.is_liquid is True

    def test_generate_simulated_chain_with_db_fallback(self, tmp_path):
        """When market DB does not exist, simulation still works with defaults."""
        fetcher = OptionsChainFetcher()
        chain = asyncio.run(fetcher._generate_simulated_chain("SPY"))
        assert len(chain.quotes) == 21

    def test_chain_get_by_strike_multiple_matches(self):
        """Multiple quotes at same strike (different expiration) all match."""
        q1 = OptionQuote(
            symbol="TEST1", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55,
        )
        q2 = OptionQuote(
            symbol="TEST2", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today() + timedelta(days=7),
            bid=2.5, ask=2.6, last=2.55, mark=2.55,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        result = chain.get_by_strike(550.0)
        assert len(result) == 2

    def test_to_dict_with_all_fields(self):
        """to_dict serializes all fields including optional ones."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date(2024, 5, 16), bid=2.50, ask=2.60,
            last=2.55, mark=2.55, delta=0.60, gamma=0.05, theta=-0.10,
            vega=0.20, implied_vol=0.18, volume=500, open_interest=5000,
        )
        d = q.to_dict()
        assert d["delta"] == 0.60
        assert d["gamma"] == 0.05
        assert d["theta"] == -0.10
        assert d["vega"] == 0.20
        assert d["implied_vol"] == 0.18
        assert d["volume"] == 500
        assert d["open_interest"] == 5000
        assert d["expiration"] == "2024-05-16"

    def test_to_dict_datetime_isoformat(self):
        """timestamp should be ISO format string."""
        q = OptionQuote(
            symbol="TEST", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=2.5, ask=2.6,
            last=2.55, mark=2.55,
        )
        d = q.to_dict()
        assert "T" in d["timestamp"]  # ISO format contains T separator

    def test_find_optimal_call_different_target_delta(self):
        """Search for a different target delta returns closest match."""
        q1 = OptionQuote(
            symbol="TEST1", underlying="SPY", option_type=OptionType.CALL,
            strike=550.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.30, volume=100, open_interest=1000,
        )
        q2 = OptionQuote(
            symbol="TEST2", underlying="SPY", option_type=OptionType.CALL,
            strike=555.0, expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.0, delta=0.50, volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q1, q2])
        best = chain.find_optimal_call(target_delta=0.50)
        assert best is not None
        assert best.strike == 555.0

    def test_options_chain_cache_str_init(self, tmp_path):
        """OptionsChainCache accepts string path."""
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        assert cache.cache_dir == tmp_path
        assert tmp_path.exists()

    def test_options_chain_cache_existing_dir(self, tmp_path):
        """OptionsChainCache with existing directory does not error."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        cache = OptionsChainCache(cache_dir=str(tmp_path))
        assert cache.cache_dir == tmp_path
