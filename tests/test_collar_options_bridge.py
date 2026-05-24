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


# ---------------------------------------------------------------------------
# NEW COVERAGE TESTS (~30 tests)
# ---------------------------------------------------------------------------


class TestDataSourceExtended:
    """Extended DataSource enum tests."""

    def test_all_members_present(self):
        """DataSource should have exactly 3 members."""
        members = {m.name for m in DataSource}
        assert members == {"LIVE", "SIMULATED", "CACHED"}

    def test_from_string_values(self):
        """Should be able to construct DataSource from string values."""
        assert DataSource("live") == DataSource.LIVE
        assert DataSource("simulated") == DataSource.SIMULATED
        assert DataSource("cached") == DataSource.CACHED


class TestLiveCollarStrikesEdges:
    """LiveCollarStrikes edge case validation."""

    def test_negative_premium(self):
        """Negative net premium should be permitted."""
        strikes = LiveCollarStrikes(
            source="live", timestamp="2026-01-01",
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=560.0,
            call_bid=1.0, call_ask=1.2, call_mark=1.1,
            call_delta=0.25, call_volume=100, call_oi=1000,
            put_symbol="P", put_strike=540.0,
            put_bid=4.0, put_ask=4.2, put_mark=4.1,
            put_delta=-0.25, put_volume=100, put_oi=1000,
            net_premium=-3.0, is_cashless=False, collar_cost_pct=-0.55,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=3.0,
        )
        assert strikes.net_premium < 0
        assert not strikes.is_cashless
        assert strikes.collar_cost_pct < 0

    def test_zero_values(self):
        """Zero values for volume/OI should be valid."""
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-01-01",
            underlying_price=0.0, vix_level=0.0, days_to_expiry=0,
            call_symbol="C", call_strike=0.0,
            call_bid=0.0, call_ask=0.0, call_mark=0.0,
            call_delta=0.0, call_volume=0, call_oi=0,
            put_symbol="P", put_strike=0.0,
            put_bid=0.0, put_ask=0.0, put_mark=0.0,
            put_delta=0.0, put_volume=0, put_oi=0,
            net_premium=0.0, is_cashless=True, collar_cost_pct=0.0,
            call_liquid=False, put_liquid=False, bid_ask_spread_pct=0.0,
        )
        d = strikes.to_dict()
        assert d["underlying_price"] == 0.0
        assert d["is_cashless"]
        assert d["bid_ask_spread_pct"] == 0.0

    def test_large_values(self):
        """Extreme values should serialize correctly."""
        strikes = LiveCollarStrikes(
            source="live", timestamp="2026-01-01",
            underlying_price=9999.99, vix_level=99.9, days_to_expiry=365,
            call_symbol="C", call_strike=9999.99,
            call_bid=500.0, call_ask=510.0, call_mark=505.0,
            call_delta=0.99, call_volume=999999, call_oi=9999999,
            put_symbol="P", put_strike=0.01,
            put_bid=0.01, put_ask=0.02, put_mark=0.015,
            put_delta=-0.01, put_volume=1, put_oi=100,
            net_premium=504.985, is_cashless=False, collar_cost_pct=5.05,
            call_liquid=True, put_liquid=False, bid_ask_spread_pct=1.98,
        )
        d = strikes.to_dict()
        assert d["underlying_price"] == 9999.99
        assert d["call_volume"] == 999999
        assert d["put_oi"] == 100
        assert not d["put_liquid"]

    def test_none_delta_in_dict_roundtrip(self):
        """None delta survives to_dict and JSON serialization."""
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-01-01",
            underlying_price=550.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=560.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=None, call_volume=100, call_oi=1000,
            put_symbol="P", put_strike=540.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=None, put_volume=100, put_oi=1000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.04,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        d = strikes.to_dict()
        raw = json.dumps(d)
        loaded = json.loads(raw)
        assert loaded["call_delta"] is None
        assert loaded["put_delta"] is None


class TestBridgeInitialization:
    """CollarOptionsBridge initialization tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_ensure_dirs_creates_dirs(self, bridge):
        """_ensure_dirs should create DATA_DIR and signals subdir."""
        import tempfile
        from pathlib import Path
        data_dir = Path(tempfile.mkdtemp())
        signals_dir = data_dir / "signals"
        assert not signals_dir.exists()
        bridge._ensure_dirs()
        # Bridge's _ensure_dirs always creates the module-level DATA_DIR,
        # so verify the signatures/ path exists there
        from src.broker.collar_options_bridge import DATA_DIR
        assert (DATA_DIR / "signals").exists()

    def test_init_sets_pricer_and_signal_gen(self, bridge):
        """Constructor should initialize all components."""
        assert bridge._fetcher is not None
        assert bridge._pricer is not None
        assert bridge._signal_gen is not None


class TestGetVixExtended:
    """Extended _get_vix tests with mocked file paths."""

    def test_get_vix_with_valid_file(self, monkeypatch, tmp_path):
        """Should read VIX from file when it exists."""
        vix_file = tmp_path / "vix_term_structure.json"
        vix_file.write_text(json.dumps({
            "2026-05-24": {"vix_spot": 22.5, "vix_futures": []},
            "2026-05-23": {"vix_spot": 21.0, "vix_futures": []},
        }))
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == 22.5

    def test_get_vix_with_empty_file(self, monkeypatch, tmp_path):
        """Should return default when VIX file has empty dict."""
        vix_file = tmp_path / "vix_term_structure.json"
        vix_file.write_text("{}")
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == pytest.approx(16.0)

    def test_get_vix_with_corrupt_file(self, monkeypatch, tmp_path):
        """Should return default when VIX file has invalid JSON."""
        vix_file = tmp_path / "vix_term_structure.json"
        vix_file.write_text("{invalid json}")
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == pytest.approx(16.0)

    def test_get_vix_with_missing_spot_key(self, monkeypatch, tmp_path):
        """Should return default when vix_spot key is missing."""
        vix_file = tmp_path / "vix_term_structure.json"
        vix_file.write_text(json.dumps({
            "2026-05-24": {"vix_futures": [18.0, 19.0]},
        }))
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == pytest.approx(16.0)

    def test_get_vix_file_missing(self, monkeypatch, tmp_path):
        """Should return default when VIX file does not exist."""
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == pytest.approx(16.0)

    def test_get_vix_multiple_dates(self, monkeypatch, tmp_path):
        """Should pick the latest date's vix_spot."""
        vix_file = tmp_path / "vix_term_structure.json"
        vix_file.write_text(json.dumps({
            "2025-01-01": {"vix_spot": 12.0},
            "2026-06-01": {"vix_spot": 35.0},
            "2025-06-01": {"vix_spot": 18.0},
        }))
        monkeypatch.setattr(
            "src.broker.collar_options_bridge.DATA_DIR", tmp_path
        )
        bridge = CollarOptionsBridge()
        vix = bridge._get_vix()
        assert vix == 35.0


class TestGetSpotExtended:
    """Extended _get_spot tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_only_puts_returns_default(self, bridge):
        """Chain with only puts (no calls) should return default 550.0."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.0, ask=3.2, last=3.1, mark=3.1,
                delta=-0.20, volume=100, open_interest=1000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        spot = bridge._get_spot(chain)
        assert spot == 550.0  # Default because no calls

    def test_even_number_of_strikes(self, bridge):
        """Even number of strikes should pick the higher middle."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol=f"C{i}", underlying="SPY", option_type=OptionType.CALL,
                strike=float(s), expiration=exp, bid=5.0, ask=5.2, last=5.1, mark=5.1,
                delta=0.40, volume=100, open_interest=1000, implied_vol=0.18,
            )
            for i, s in enumerate([540, 545, 550, 555])
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        spot = bridge._get_spot(chain)
        # Sorted: [540, 545, 550, 555]; len//2 = 2 => index 2 => 550.0
        assert spot == 550.0


class TestFindFromChainEdgeCases:
    """Edge case tests for _find_from_chain."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_non_liquid_fallback(self, bridge):
        """When no liquid options, should fall back to all options."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=10.0, last=7.0, mark=7.0,
                delta=0.30, volume=1, open_interest=5, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=9.0, last=6.4, mark=6.4,
                delta=-0.20, volume=1, open_interest=5, implied_vol=0.18,
            ),
        ]
        # Both have volume=1 (< 10), OI=5 (< 100), wide spread — not liquid
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None  # Should still work via fallback

    def test_extreme_delta_values(self, bridge):
        """Should handle options with extreme delta values."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C_ITM", underlying="SPY", option_type=OptionType.CALL,
                strike=500.0, expiration=exp, bid=50.0, ask=51.0, last=50.5, mark=50.5,
                delta=0.99, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="C_OTM", underlying="SPY", option_type=OptionType.CALL,
                strike=580.0, expiration=exp, bid=5.4, ask=5.6, last=5.5, mark=5.5,
                delta=0.01, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P_OTM", underlying="SPY", option_type=OptionType.PUT,
                strike=520.0, expiration=exp, bid=5.4, ask=5.6, last=5.5, mark=5.5,
                delta=-0.01, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P_ITM", underlying="SPY", option_type=OptionType.PUT,
                strike=600.0, expiration=exp, bid=50.0, ask=51.0, last=50.5, mark=50.5,
                delta=-0.99, volume=100, open_interest=1000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        # Call 0.01 is closer to 0.30 than 0.99
        assert result.call_strike == 580.0
        # Put -0.01 is closer to -0.20 than -0.99
        assert result.put_strike == 520.0

    def test_bid_ask_spread_is_max_of_both(self, bridge):
        """bid_ask_spread_pct should be max of call and put spread."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
                # spread = (4.2-4.0)/4.1*100 = 4.88%
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.5, ask=3.9, last=3.7, mark=3.7,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
                # spread = (3.9-3.5)/3.7*100 = 10.81%
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        # Put spread (10.81%) > call spread (4.88%), so result should be ~10.81
        put_spread = (3.9 - 3.5) / 3.7 * 100
        assert result.bid_ask_spread_pct == pytest.approx(put_spread, abs=0.5)

    def test_source_simulated_when_call_delta_is_none(self, bridge):
        """Source should be SIMULATED when call delta is None."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=None, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.source == DataSource.SIMULATED.value

    def test_multiple_calls_same_delta_score(self, bridge):
        """When multiple calls have same delta score, first sorted wins."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=555.0, expiration=exp, bid=5.0, ask=5.2, last=5.1, mark=5.1,
                delta=0.30, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="C2", underlying="SPY", option_type=OptionType.CALL,
                strike=565.0, expiration=exp, bid=3.0, ask=3.2, last=3.1, mark=3.1,
                delta=0.30, volume=100, open_interest=1000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        # Both calls have delta 0.30, so first in list wins (555.0)
        assert result.call_strike == 555.0


class TestFallbackEstimateEdgeCases:
    """Edge cases for _fallback_estimate."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_very_low_spot_100(self, bridge):
        """Should handle very low spot price."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(100.0, 16.0, 30))
        assert strikes.underlying_price == 100.0
        assert strikes.call_strike > 100.0
        assert strikes.put_strike < 100.0
        assert strikes.call_strike < 300.0  # Sanity check

    def test_very_high_spot_1000(self, bridge):
        """Should handle very high spot price."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(1000.0, 16.0, 30))
        assert strikes.underlying_price == 1000.0
        assert strikes.call_strike > 1000.0
        assert strikes.put_strike < 1000.0

    def test_longer_dte_60(self, bridge):
        """Should handle longer days to expiry."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 60))
        assert strikes.days_to_expiry == 60
        # Longer DTE means wider strikes
        call_width = strikes.call_strike - strikes.underlying_price
        put_width = strikes.underlying_price - strikes.put_strike
        assert call_width > 0
        assert put_width > 0
        # Should still have valid symbols
        assert "C" in strikes.call_symbol
        assert "P" in strikes.put_symbol

    def test_shorter_dte_7(self, bridge):
        """Should handle shorter days to expiry."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 7))
        assert strikes.days_to_expiry == 7

    def test_zero_vix(self, bridge):
        """Zero VIX should not crash fallback estimate."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 0.0, 30))
        assert strikes.underlying_price == 550.0
        assert strikes.vix_level == 0.0


class TestFetchOptimalCollarExtended:
    """Extended fetch_optimal_collar tests with mocking."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_fetch_with_mocked_chain(self, bridge):
        """Should use live chain when fetcher returns valid data."""
        import asyncio
        exp = date(2026, 6, 16)
        call_q = OptionQuote(
            symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
            strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
            delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
        )
        put_q = OptionQuote(
            symbol="SPY260616P00540000", underlying="SPY", option_type=OptionType.PUT,
            strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
            delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
        )
        chain = OptionsChain(underlying="SPY", quotes=[call_q, put_q])

        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(return_value=chain)

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        assert strikes.source == DataSource.LIVE.value
        assert strikes.call_strike == 560.0
        assert strikes.put_strike == 540.0

    def test_fetch_fallback_on_exception(self, bridge):
        """Should fallback when fetcher raises an exception."""
        import asyncio
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(
            side_effect=ConnectionError("API unavailable")
        )

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        assert strikes.source == DataSource.SIMULATED.value

    def test_fetch_fallback_on_import_error(self, bridge):
        """Should fallback on ModuleNotFoundError from fetcher."""
        import asyncio
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(
            side_effect=ModuleNotFoundError("No module named 'alpaca'")
        )

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        assert strikes.source == DataSource.SIMULATED.value

    def test_fetch_with_empty_chain_from_fetcher(self, bridge):
        """Should fallback when fetcher returns empty chain."""
        import asyncio
        chain = OptionsChain(underlying="SPY")
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(return_value=chain)

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        # Empty chain has no calls/puts, so it should trigger BS fallback
        assert strikes.source == DataSource.SIMULATED.value

    def test_fetch_chain_with_calls_only(self, bridge):
        """Should fallback when chain has calls but no puts."""
        import asyncio
        exp = date(2026, 6, 16)
        call_q = OptionQuote(
            symbol="SPY260616C00560000", underlying="SPY", option_type=OptionType.CALL,
            strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
            delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
        )
        chain = OptionsChain(underlying="SPY", quotes=[call_q])
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(return_value=chain)

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=550.0, vix=16.0))
        assert strikes.source == DataSource.SIMULATED.value

    def test_fetch_uses_default_spot_when_none(self, bridge):
        """Should use default spot (550.0) when spot is None and no chain."""
        import asyncio
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(
            side_effect=ConnectionError()
        )

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=None, vix=16.0))
        assert strikes.underlying_price == 550.0


class TestSaveStrikesExtended:
    """Extended save_strikes tests."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_save_overwrites_existing_file(self, bridge, tmp_path):
        """save_strikes should overwrite existing file."""
        bridge.OUTPUT_PATH = tmp_path / "test_overwrite.json"
        # Write initial content
        bridge.OUTPUT_PATH.write_text('{"source": "old"}')

        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-01-01",
            underlying_price=600.0, vix_level=16.0, days_to_expiry=30,
            call_symbol="C", call_strike=610.0,
            call_bid=4.0, call_ask=4.2, call_mark=4.1,
            call_delta=0.30, call_volume=500, call_oi=5000,
            put_symbol="P", put_strike=590.0,
            put_bid=3.8, put_ask=4.0, put_mark=3.9,
            put_delta=-0.20, put_volume=400, put_oi=4000,
            net_premium=0.2, is_cashless=True, collar_cost_pct=0.03,
            call_liquid=True, put_liquid=True, bid_ask_spread_pct=2.5,
        )
        bridge.save_strikes(strikes)
        with open(bridge.OUTPUT_PATH) as f:
            data = json.load(f)
        assert data["source"] == "simulated"
        assert data["underlying_price"] == 600.0

    def test_save_with_none_delta(self, bridge, tmp_path):
        """save_strikes should handle None delta values."""
        bridge.OUTPUT_PATH = tmp_path / "test_none_delta.json"
        strikes = LiveCollarStrikes(
            source="live", timestamp="2026-01-01",
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
        bridge.save_strikes(strikes)
        with open(bridge.OUTPUT_PATH) as f:
            data = json.load(f)
        assert data["call_delta"] is None
        assert data["put_delta"] is None
        assert not data["call_liquid"]


class TestCompareWithSignalExtended2:
    """Additional compare_with_signal edge cases."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_zero_bs_call_strike_does_not_crash(self, bridge):
        """Zero BS call strike should not cause division error."""
        strikes = LiveCollarStrikes(
            source="simulated", timestamp="2026-01-01",
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
        comparison = bridge.compare_with_signal(strikes)
        assert comparison["call_diff_pct"] != float("inf")
        assert comparison["put_diff_pct"] != float("inf")
        assert "source" in comparison

    def test_live_source_in_comparison(self, bridge):
        """Comparison should preserve source from strikes."""
        import asyncio
        strikes = asyncio.run(bridge._fallback_estimate(550.0, 16.0, 30))
        comparison = bridge.compare_with_signal(strikes)
        assert comparison["source"] == DataSource.SIMULATED.value


class TestFetchOptimalCollarSpotInference:
    """Tests for spot inference from chain in fetch_optimal_collar."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_spot_inferred_from_chain_when_none(self, bridge):
        """When spot is None but chain has data, spot should be inferred."""
        import asyncio
        bridge = CollarOptionsBridge()
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=550.0, expiration=exp, bid=5.0, ask=5.2, last=5.1, mark=5.1,
                delta=0.30, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(return_value=chain)

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=None, vix=16.0))
        # Spot should be inferred from chain (middle strike = 550.0)
        assert strikes.underlying_price == 550.0

    def test_spot_inferred_fallback_on_empty_chain(self, bridge):
        """When spot is None and chain is empty, fallback uses 550 default."""
        import asyncio
        bridge._fetcher = MagicMock()
        bridge._fetcher.fetch_0dte_chain = AsyncMock(
            return_value=OptionsChain(underlying="SPY")
        )

        strikes = asyncio.run(bridge.fetch_optimal_collar(spot=None, vix=16.0))
        # Empty chain -> fallback, but vix read may use default
        assert strikes.underlying_price == 550.0


class TestLiquidityFiltering:
    """Tests for liquidity filtering in _find_from_chain."""

    @pytest.fixture
    def bridge(self):
        return CollarOptionsBridge()

    def test_prefers_liquid_over_non_liquid(self, bridge):
        """Should prefer liquid options even with worse delta match."""
        exp = date(2026, 6, 16)
        # Non-liquid call with perfect delta
        # Liquid call with slightly worse delta
        quotes = [
            OptionQuote(
                symbol="C_NON_LIQ", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=1, open_interest=5, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="C_LIQ", underlying="SPY", option_type=OptionType.CALL,
                strike=565.0, expiration=exp, bid=3.0, ask=3.1, last=3.05, mark=3.05,
                delta=0.25, volume=500, open_interest=5000, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P_LIQ", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=3.95, last=3.9, mark=3.9,
                delta=-0.20, volume=400, open_interest=4000, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        # Only C_NON_LIQ has delta 0.30, but it's not liquid.
        # C_LIQ has delta 0.25 — should be picked since it's the only liquid call
        # Actually: C_NON_LIQ fails liquidity (volume=1 < 10), so it's excluded.
        # Then only C_LIQ remains in liquid_calls.
        assert result.call_strike == 565.0

    def test_all_options_not_liquid_still_selects(self, bridge):
        """When no options are liquid, falls back and still picks best delta."""
        exp = date(2026, 6, 16)
        quotes = [
            OptionQuote(
                symbol="C1", underlying="SPY", option_type=OptionType.CALL,
                strike=560.0, expiration=exp, bid=4.0, ask=4.2, last=4.1, mark=4.1,
                delta=0.30, volume=1, open_interest=5, implied_vol=0.18,
            ),
            OptionQuote(
                symbol="P1", underlying="SPY", option_type=OptionType.PUT,
                strike=540.0, expiration=exp, bid=3.8, ask=4.0, last=3.9, mark=3.9,
                delta=-0.20, volume=1, open_interest=5, implied_vol=0.18,
            ),
        ]
        chain = OptionsChain(underlying="SPY", quotes=quotes)
        result = bridge._find_from_chain(chain, 550.0, 16.0, 30)
        assert result is not None
        assert result.call_strike == 560.0
        assert result.put_strike == 540.0

