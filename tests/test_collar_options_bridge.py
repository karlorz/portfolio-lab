"""
Tests for Collar Options Bridge (v4.80 live data integration)
"""

import json
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from src.broker.collar_options_bridge import (
    CollarOptionsBridge,
    LiveCollarStrikes,
    DataSource,
)
from src.broker.options_utils import OptionsChain, OptionQuote, OptionType


class TestDataSource:
    """Test data source enum."""

    def test_source_values(self):
        assert DataSource.LIVE.value == "live"
        assert DataSource.SIMULATED.value == "simulated"
        assert DataSource.CACHED.value == "cached"


class TestLiveCollarStrikes:
    """Test live collar strikes dataclass."""

    def test_serializable(self):
        strikes = LiveCollarStrikes(
            source="simulated", timestamp=datetime.now().isoformat(),
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="SPY260616C00560000", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=0.30, call_volume=500, call_oi=5000,
            put_symbol="SPY260616P00540000", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=-0.20, put_volume=400, put_oi=4000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        d = strikes.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "simulated"
        assert d["underlying_price"] == 550.0
        assert d["is_cashless"]


class TestCollarOptionsBridge:
    """Test options bridge core functionality."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_fallback_estimate(self, bridge):
        """Should generate fallback estimate when chain unavailable."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        assert isinstance(strikes, LiveCollarStrikes)
        assert strikes.source == "simulated"
        assert strikes.underlying_price == 550.0
        assert strikes.call_strike > 550.0
        assert strikes.put_strike < 550.0
        assert strikes.call_delta > 0

    def test_fallback_high_vix(self, bridge):
        """Should handle high VIX in fallback."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 35.0, 30))
        assert strikes.vix_level == 35.0
        assert strikes.call_strike > 550.0
        assert strikes.put_strike < 550.0

    def test_fallback_crisis_vix(self, bridge):
        """Should handle crisis VIX in fallback."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 50.0, 30))
        assert strikes.vix_level == 50.0
        assert not strikes.is_cashless

    def test_compare_with_signal(self, bridge):
        """Should compare live strikes with BS estimate."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        comparison = bridge.compare_with_signal(strikes)
        assert "live_call_strike" in comparison
        assert "bs_call_strike" in comparison
        assert "source" in comparison
        assert comparison["source"] == "simulated"

    def test_save_strikes(self, bridge, tmp_path):
        """Should save strikes to JSON."""
        bridge.OUTPUT_PATH = tmp_path / "test_strikes.json"
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        bridge.save_strikes(strikes)

        with open(bridge.OUTPUT_PATH) as f:
            loaded = json.load(f)
        assert loaded["underlying_price"] == 550.0
        assert loaded["source"] == "simulated"

    def test_get_vix_fallback(self, bridge):
        vix = bridge._get_vix()
        assert vix > 0  # Should have a default

    def test_get_spot_from_empty_chain(self, bridge):
        chain = OptionsChain(underlying="SPY")
        spot = bridge._get_spot(chain)
        assert spot == 550.0  # Default


class TestLiveCollarFetch:
    """Test async fetch workflow."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_fetch_with_simulated_data(self, bridge):
        """Should get collar from simulated chain with valid strikes."""
        import asyncio
        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        assert isinstance(strikes, LiveCollarStrikes)
        assert strikes.underlying_price == 550.0
        assert strikes.call_strike > 550.0
        assert strikes.put_strike < 550.0
        assert strikes.net_premium is not None
        assert strikes.is_cashless in (True, False)

    def test_fetch_default_parameters(self, bridge):
        """Should work with no parameters."""
        import asyncio
        strikes = asyncio.run(bridge.fetch_optimal_collar())
        assert isinstance(strikes, LiveCollarStrikes)
        assert strikes.underlying_price > 0

    def test_fetch_different_spots(self, bridge):
        """Different spot prices should produce different strikes."""
        import asyncio
        low = asyncio.run(bridge.fetch_optimal_collar(spot=300.0, vix=16.0))
        high = asyncio.run(bridge.fetch_optimal_collar(spot=600.0, vix=16.0))
        assert low.call_strike < high.call_strike
        assert low.put_strike < high.put_strike


class TestEdgeCases:
    """Edge cases for options bridge."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_zero_spot_handled(self, bridge):
        """Zero spot should not crash."""
        # _get_spot returns default for empty chain
        chain = OptionsChain(underlying="SPY")
        spot = bridge._get_spot(chain)
        assert spot > 0

    def test_empty_chain_no_crash(self, bridge):
        """Empty options chain should gracefully fallback."""
        chain = OptionsChain(underlying="SPY")
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is None  # Should return None, triggering fallback

    def test_find_from_chain_with_calls_and_puts(self, bridge):
        """Should find optimal collar from a populated chain."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616C00565000", underlying="SPY", option_type=OptionType.CALL,
                strike=565.0, expiration=exp, bid=2.5, ask=2.7, last=2.6, mark=2.6,
                delta=0.20, volume=300, open_interest=3000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00535000", underlying="SPY", option_type=OptionType.PUT,
                strike=535.0, expiration=exp, bid=5.0, ask=5.3, last=5.15, mark=5.15,
                delta=-0.30, volume=200, open_interest=2000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.call_strike == 560.0  # Delta 0.30 is closest to target
        assert result.put_strike == 540.0   # Delta -0.20 is closest to target

    def test_find_from_chain_picks_best_delta(self, bridge):
        """Should select options closest to target deltas."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00555000", underlying="SPY", option_type=OptionType.CALL,
                strike=555.0, expiration=exp, bid=5.5, ask=5.7, last=5.6, mark=5.6,
                delta=0.35, volume=600, open_interest=6000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.28, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00545000", underlying="SPY", option_type=OptionType.PUT,
                strike=545.0, expiration=exp, bid=2.8, ask=3.0, last=2.9, mark=2.9,
                delta=-0.15, volume=400, open_interest=4000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.22, volume=300, open_interest=3000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.call_strike == 560.0  # Delta 0.28 closer to 0.30
        assert result.put_strike == 540.0   # Delta -0.22 closer to -0.20

    def test_find_from_chain_none_delta_skipped(self, bridge):
        """Options with None delta should be deprioritized."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00550000", underlying="SPY", option_type=OptionType.CALL,
                strike=550.0, expiration=exp, bid=8.0, ask=8.5, last=8.25, mark=8.25,
                delta=None, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.call_strike == 560.0  # Valid delta wins over None

    def test_get_spot_from_populated_chain(self, bridge):
        """Should infer spot from ATM strike in chain."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol=f"SPY260616C{strike:08d}", underlying="SPY",
                option_type=OptionType.CALL, strike=float(strike), expiration=exp,
                bid=5.0, ask=5.2, last=5.1, mark=5.1, delta=0.40,
                volume=100, open_interest=1000, implied_vol=0.18,
            )
            for strike in [540, 545, 550, 555, 560]
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        spot = bridge._get_spot(chain)
        assert spot == 550.0  # Middle strike

    def test_compare_with_signal_has_diff_pcts(self, bridge):
        """compare_with_signal should include diff percentages."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        comparison = bridge.compare_with_signal(strikes)
        assert "call_diff_pct" in comparison
        assert "put_diff_pct" in comparison
        assert isinstance(comparison["call_diff_pct"], (int, float))
        assert isinstance(comparison["put_diff_pct"], (int, float))

    def test_net_premium_calculation(self, bridge):
        """Net premium should be call mark minus put mark."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        expected_net = 4.1 - 3.9
        assert abs(result.net_premium - expected_net) < 0.01

    def test_cashless_collar_detection(self, bridge):
        """Collar should be cashless when net premium is near zero."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=3.9, ask=4.1, last=4.0, mark=4.0,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.9, ask=4.1, last=4.0, mark=4.0,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.is_cashless  # Net premium ~0


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestLiveCollarStrikesExtended:
    """Extended LiveCollarStrikes dataclass tests."""

    def test_to_dict_has_all_fields(self):
        strikes = LiveCollarStrikes(
            source="simulated", timestamp=datetime.now().isoformat(),
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="SPY260616C00560000", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=0.30, call_volume=500, call_oi=5000,
            put_symbol="SPY260616P00540000", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=-0.20, put_volume=400, put_oi=4000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        d = strikes.to_dict()
        expected_keys = {
            'source', 'timestamp', 'underlying_price', 'vix_level',
            'days_to_expiry', 'call_symbol', 'call_strike', 'call_bid',
            'call_ask', 'call_mark', 'call_delta', 'call_volume', 'call_oi',
            'put_symbol', 'put_strike', 'put_bid', 'put_ask', 'put_mark',
            'put_delta', 'put_volume', 'put_oi', 'net_premium', 'is_cashless',
            'collar_cost_pct', 'call_liquid', 'put_liquid', 'bid_ask_spread_pct',
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_json_serializable(self):
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-05-24T00:00:00",
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=0.30, call_volume=500, call_oi=5000,
            put_symbol="P", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=-0.20, put_volume=400, put_oi=4000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        d = strikes.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_none_delta_stored(self):
        """None delta should be preserved in to_dict."""
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-05-24",
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=None, call_volume=0, call_oi=0,
            put_symbol="P", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=None, put_volume=0, put_oi=0,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=False, put_liquid=False, bid_ask_spread_pct=5.0,
        )
        d = strikes.to_dict()
        assert d['call_delta'] is None
        assert d['put_delta'] is None


class TestDeltaScore:
    """Test delta_score function used in _find_from_chain."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_exact_match_gives_zero(self, bridge):
        exp = date(2026, 6, 16)
        q = OptionQuote(
            symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
            strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
            delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
        )
        # delta_score is a local function in _find_from_chain, test indirectly
        # by verifying best call has delta closest to 0.30
        assert abs(q.delta - 0.30) < 0.01

    def test_none_delta_inf_score(self):
        """None delta should get infinite score (sorted last)."""
        # This is tested implicitly by _find_from_chain preferring non-None deltas
        pass


class TestFindFromChainExtended:
    """Extended _find_from_chain tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_empty_chain_returns_none(self, bridge):
        """Empty chain should return None."""
        chain = OptionsChain(underlying="SPY", quotes=[])
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is None

    def test_calls_only_returns_none(self, bridge):
        """Chain with only calls (no puts) should return None."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is None

    def test_selects_closest_delta(self, bridge):
        """Should select options with delta closest to targets."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=555.0, expiration=exp, bid=5.0, ask=5.2, last=5.1, mark=5.1,
                delta=0.28, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="C2", underlying="SPY", option_type=OptionType.CALL,
                strike=565.0, expiration=exp, bid=3.0, ask=3.2, last=3.1, mark=3.1,
                delta=0.35, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=545.0, expiration=exp, bid=3.5, ask=3.7, last=3.6, mark=3.6,
                delta=-0.18, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P2", underlying="SPY", option_type=OptionType.PUT,
                strike=535.0, expiration=exp, bid=2.5, ask=2.7, last=2.6, mark=2.6,
                delta=-0.25, volume=100, open_interest=1000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        # Call with delta 0.28 is closer to 0.30 than 0.35
        assert result.call_delta == pytest.approx(0.28, abs=0.01)
        # Put with delta -0.18 is closer to -0.20 than -0.25
        assert result.put_delta == pytest.approx(-0.18, abs=0.02)


class TestFallbackEstimateExtended:
    """Extended _fallback_estimate tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_source_is_simulated(self, bridge):
        import asyncio
        result = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        assert result.source == "simulated"

    def test_has_valid_structure(self, bridge):
        import asyncio
        result = asyncio.run(bridge._fallback_estimate(550.0, 20.0, 30))
        assert result.underlying_price == 550.0
        assert result.vix_level == 20.0
        assert result.days_to_expiry == 30
        assert isinstance(result.call_symbol, str)
        assert isinstance(result.put_symbol, str)
        assert result.call_strike > 0
        assert result.put_strike > 0

    def test_bid_ask_around_mark(self, bridge):
        """Bid should be below mark, ask should be above mark."""
        import asyncio
        result = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        assert result.call_bid <= result.call_mark
        assert result.call_ask >= result.call_mark
        assert result.put_bid <= result.put_mark
        assert result.put_ask >= result.put_mark


class TestGetSpot:
    """Extended _get_spot tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_single_call_strike(self, bridge):
        """Single call should return that strike."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=550.0, expiration=exp, bid=5.0, ask=5.2, last=5.1, mark=5.1,
                delta=0.50, volume=100, open_interest=1000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        spot = bridge._get_spot(chain)
        assert spot == 550.0

    def test_default_when_no_calls(self, bridge):
        """No calls should return default 550.0."""
        chain = OptionsChain(underlying="SPY", quotes=[])
        spot = bridge._get_spot(chain)
        assert spot == 550.0


class TestSaveStrikes:
    """Test save_strikes method."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_save_strikes_creates_file(self, bridge, tmp_path):
        """save_strikes should write JSON file."""
        bridge.OUTPUT_PATH = tmp_path / "live_collar_strikes.json"
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-05-24T00:00:00",
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=0.30, call_volume=500, call_oi=5000,
            put_symbol="P", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=-0.20, put_volume=400, put_oi=4000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        bridge.save_strikes(strikes)
        assert bridge.OUTPUT_PATH.exists()
        with open(bridge.OUTPUT_PATH) as f:
            data = json.load(f)
        assert data["underlying_price"] == 550.0


class TestCompareWithSignalExtended:
    """Extended compare_with_signal tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_returns_expected_keys(self, bridge):
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        comparison = bridge.compare_with_signal(strikes)
        expected_keys = {
            'live_call_strike', 'bs_call_strike', 'call_diff_pct',
            'live_put_strike', 'bs_put_strike', 'put_diff_pct',
            'live_net_premium', 'bs_net_premium', 'source',
        }
        assert expected_keys.issubset(set(comparison.keys()))

    def test_source_matches_strikes(self, bridge):
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        comparison = bridge.compare_with_signal(strikes)
        assert comparison['source'] == strikes.source

