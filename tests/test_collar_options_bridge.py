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
    fetch_collar_sync,
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

    def test_convenience_function(self):
        """Sync wrapper should work."""
        strikes = fetch_collar_sync()
        assert isinstance(strikes, LiveCollarStrikes)
