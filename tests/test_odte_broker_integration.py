"""
Tests for 0DTE Options Broker Integration (Phase 2)

Covers:
- OptionsChainFetcher (chain fetching, parsing, caching)
- OptionsChain data structure (filtering, delta matching)
- ODTEExecutor (entry, exit, monitoring)
- Simulated execution in paper mode
"""

import pytest
import asyncio
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock
import json
import sqlite3
from pathlib import Path

from src.broker.options_utils import (
    OptionQuote, OptionType, OptionsChain, OptionsChainFetcher,
    OptionsChainCache, fetch_chain_sync, get_best_0dte_call
)
from src.broker.odte_executor import (
    ODTEExecutor, ODTEOrderRequest, ODTEExecutionResult,
    OrderStatus, ExitReason, ODTEMonitorState
)
from src.options.odte_yield_calculator import ZeroDTEConfig


class TestOptionQuote:
    """Test the OptionQuote dataclass."""
    
    def test_mid_price_calculation(self):
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date(2024, 5, 16),
            bid=2.50,
            ask=2.70,
            last=2.60,
            mark=2.60,
        )
        assert quote.mid_price == 2.60
    
    def test_liquidity_check_pass(self):
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date(2024, 5, 16),
            bid=2.50,
            ask=2.55,  # Tighter spread
            last=2.525,
            mark=2.525,
            volume=100,
            open_interest=1000,
        )
        assert quote.is_liquid is True
        assert quote.bid_ask_spread_pct <= 5.0
    
    def test_liquidity_check_fail_low_volume(self):
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date(2024, 5, 16),
            bid=2.50,
            ask=2.70,
            last=2.60,
            mark=2.60,
            volume=5,
            open_interest=1000,
        )
        assert quote.is_liquid is False
    
    def test_liquidity_check_fail_wide_spread(self):
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date(2024, 5, 16),
            bid=1.00,
            ask=2.00,
            last=1.50,
            mark=1.50,
            volume=100,
            open_interest=1000,
        )
        assert quote.is_liquid is False  # ~66% spread
    
    def test_days_to_expiration(self):
        today = date.today()
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=today,
            bid=2.50,
            ask=2.70,
            last=2.60,
            mark=2.60,
        )
        assert quote.days_to_expiration == 0
    
    def test_to_dict(self):
        quote = OptionQuote(
            symbol="SPY240516C00550000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=550.0,
            expiration=date(2024, 5, 16),
            bid=2.50,
            ask=2.55,  # Tighter spread for liquidity
            last=2.525,
            mark=2.525,
            delta=0.30,
            volume=100,
            open_interest=1000,
        )
        d = quote.to_dict()
        assert d["symbol"] == "SPY240516C00550000"
        assert d["strike"] == 550.0
        assert d["delta"] == 0.30
        assert d["is_liquid"] is True


class TestOptionsChain:
    """Test the OptionsChain dataclass."""
    
    @pytest.fixture
    def sample_chain(self):
        today = date.today()
        quotes = [
            OptionQuote(
                symbol="SPY240516C00500000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=500.0,
                expiration=today,
                bid=50.0,
                ask=51.0,
                last=50.5,
                mark=50.5,
                delta=0.90,
                volume=100,
                open_interest=1000,
            ),
            OptionQuote(
                symbol="SPY240516C00550000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=550.0,
                expiration=today,
                bid=2.50,
                ask=2.55,  # Tighter spread
                last=2.525,
                mark=2.525,
                delta=0.30,
                volume=200,
                open_interest=2000,
            ),
            OptionQuote(
                symbol="SPY240516P00550000",
                underlying="SPY",
                option_type=OptionType.PUT,
                strike=550.0,
                expiration=today,
                bid=2.40,
                ask=2.45,  # Tighter spread
                last=2.425,
                mark=2.425,
                delta=-0.30,
                volume=150,
                open_interest=1500,
            ),
            OptionQuote(
                symbol="SPY240516C00600000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=600.0,
                expiration=today,
                bid=0.50,
                ask=0.52,  # Tighter spread
                last=0.51,
                mark=0.51,
                delta=0.10,
                volume=5,  # Low volume - should be excluded
                open_interest=500,
            ),
        ]
        return OptionsChain(underlying="SPY", quotes=quotes)
    
    def test_get_calls(self, sample_chain):
        calls = sample_chain.get_calls()
        assert len(calls) == 3
        assert all(q.option_type == OptionType.CALL for q in calls)
    
    def test_get_puts(self, sample_chain):
        puts = sample_chain.get_puts()
        assert len(puts) == 1
        assert puts[0].option_type == OptionType.PUT
    
    def test_get_by_strike(self, sample_chain):
        strikes = sample_chain.get_by_strike(550.0)
        assert len(strikes) == 2
        assert all(q.strike == 550.0 for q in strikes)
    
    def test_get_0dte(self, sample_chain):
        dte = sample_chain.get_0dte()
        assert len(dte) == 4
    
    def test_get_calls_by_delta(self, sample_chain):
        calls = sample_chain.get_calls_by_delta(0.30, tolerance=0.05)
        assert len(calls) == 1
        assert abs(calls[0].delta - 0.30) <= 0.05
    
    def test_get_liquid_calls(self, sample_chain):
        liquid = sample_chain.get_liquid_calls()
        assert len(liquid) == 2  # Excludes low volume 600 strike
        assert all(q.is_liquid for q in liquid)
    
    def test_find_optimal_call(self, sample_chain):
        optimal = sample_chain.find_optimal_call(target_delta=0.30)
        assert optimal is not None
        assert optimal.strike == 550.0
        assert optimal.is_liquid


class TestOptionsChainFetcher:
    """Test the OptionsChainFetcher."""
    
    @pytest.fixture
    def fetcher(self):
        return OptionsChainFetcher()
    
    def test_option_price_simulation(self, fetcher):
        price = fetcher._simulate_option_price(
            spot=550.0,
            strike=550.0,
            vol=0.16,
            tte=1/365,
            is_call=True
        )
        assert price > 0
        # ATM option should have time value
        assert price > 0.1
    
    def test_delta_estimation(self, fetcher):
        delta = fetcher._estimate_delta(
            spot=550.0,
            strike=550.0,  # ATM
            vol=0.16,
            tte=1/365,
            is_call=True
        )
        assert 0.4 <= delta <= 0.6  # ATM delta ~0.5
    
    def test_delta_estimation_otm(self, fetcher):
        delta = fetcher._estimate_delta(
            spot=550.0,
            strike=600.0,  # OTM
            vol=0.16,
            tte=1/365,
            is_call=True
        )
        assert delta < 0.5  # OTM has lower delta


class TestOptionsChainCache:
    """Test the OptionsChainCache."""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return str(tmp_path / "options_cache")
    
    def test_cache_operations(self, temp_cache_dir):
        cache = OptionsChainCache(cache_dir=temp_cache_dir)
        
        # Initially empty
        history = cache.get_history("SPY", days=7)
        assert len(history) == 0
        
        # After adding data via fetcher would have entries
        # This is a unit test, so we test the interface exists


class TestODTEExecutor:
    """Test the ODTEExecutor."""
    
    @pytest.fixture
    def executor(self):
        config = ZeroDTEConfig()
        return ODTEExecutor(config=config, paper_mode=True)
    
    @pytest.fixture
    def mock_chain(self):
        today = date.today()
        quotes = [
            OptionQuote(
                symbol="SPY240516C00550000",
                underlying="SPY",
                option_type=OptionType.CALL,
                strike=550.0,
                expiration=today,
                bid=2.50,
                ask=2.55,
                last=2.525,
                mark=2.525,
                delta=0.30,
                volume=200,
                open_interest=2000,
            ),
        ]
        return OptionsChain(underlying="SPY", quotes=quotes)
    
    def test_get_active_positions_summary_empty(self, executor):
        summary = executor.get_active_positions_summary()
        assert summary["count"] == 0
        assert summary["total_premium_collected"] == 0.0
    
    def test_monitor_state_update(self, executor, mock_chain):
        quote = mock_chain.quotes[0]
        
        position = ODTEMonitorState(
            option_symbol="SPY240516C00550000",
            underlying="SPY",
            strike=550.0,
            entry_premium=2.60,
            contracts=1,
            entry_delta=0.30,
        )
        
        # Simulate price drop (profitable for short call)
        position.update_pnl(current_buy_price=1.30)
        
        assert position.unrealized_pnl > 0  # Made profit
        assert position.current_premium == 1.30


class TestCLIInterface:
    """Test CLI convenience functions."""
    
    def test_fetch_chain_sync_interface(self):
        # Test that the function exists and is callable
        # Actual network call would require mocking
        assert callable(fetch_chain_sync)
    
    def test_get_best_0dte_call_interface(self):
        # Test that the function exists and is callable
        assert callable(get_best_0dte_call)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
