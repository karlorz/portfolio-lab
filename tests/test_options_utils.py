"""
Tests for src/broker/options_utils.py — Options chain, quotes, and broker integration.
Mocks aiohttp and price fetcher to avoid network calls. No ML dependencies.
"""
import pytest
import json
import sqlite3
import asyncio
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from decimal import Decimal

# Mock aiohttp before importing module
_mock_aiohttp = MagicMock()
_mock_aiohttp.ClientSession = MagicMock
sys.modules["aiohttp"] = _mock_aiohttp

from src.broker.options_utils import (
    OptionType,
    OptionStatus,
    OptionQuote,
    OptionsChain,
    OptionsChainFetcher,
    OptionsChainCache,
    fetch_chain_sync,
    get_best_0dte_call,
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

    def test_init_with_credentials(self):
        fetcher = OptionsChainFetcher(api_key="test_key", secret_key="test_secret")
        assert fetcher.has_api_access is True

    def test_init_paper_mode_default(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s", "ALPACA_PAPER": "true"}):
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


class TestConvenienceFunctions:
    """fetch_chain_sync and get_best_0dte_call."""

    @patch("src.broker.options_utils.OptionsChainFetcher")
    def test_fetch_chain_sync(self, mock_fetcher_class):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_0dte_chain = AsyncMock(return_value=OptionsChain(underlying="SPY"))
        mock_fetcher_class.return_value = mock_fetcher

        chain = fetch_chain_sync("SPY")
        assert chain.underlying == "SPY"

    @patch("src.broker.options_utils.fetch_chain_sync")
    def test_get_best_0dte_call(self, mock_fetch):
        q = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today(), bid=9.90, ask=10.10,
            last=10.0, mark=10.00, delta=0.30,
            volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        mock_fetch.return_value = chain

        best = get_best_0dte_call(target_delta=0.30)
        assert best is not None
        assert best.delta == 0.30

    @patch("src.broker.options_utils.fetch_chain_sync")
    def test_get_best_0dte_call_no_0dte(self, mock_fetch):
        q = OptionQuote(
            symbol="SPY240516C00550000", underlying="SPY",
            option_type=OptionType.CALL, strike=550.0,
            expiration=date.today() + timedelta(days=1),
            bid=9.90, ask=10.10, last=10.0, mark=10.00, delta=0.30,
            volume=100, open_interest=1000,
        )
        chain = OptionsChain(underlying="SPY", quotes=[q])
        mock_fetch.return_value = chain

        best = get_best_0dte_call(target_delta=0.30)
        assert best is None


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
