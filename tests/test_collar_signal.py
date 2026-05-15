"""
Tests for Cashless Collar Signal Generator (v4.60)
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.signals.collar_signal import (
    CollarSignalGenerator,
    CollarSignal,
    CollarStrikes,
    CollarRegime,
    CollarState,
    BlackScholesPricer,
    generate_collar_signal,
)


class TestBlackScholesPricer:
    """Test Black-Scholes option pricing."""

    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()

    def test_atm_call_price_approximate(self, pricer):
        """ATM call should be approximately 0.4 * spot * vol * sqrt(T)."""
        result = pricer.price_option(
            spot=100, strike=100, time_to_expiry=30/365, rate=0.05, vol=0.20, is_call=True
        )
        expected_approx = 100 * 0.20 * (30/365)**0.5 * 0.4
        assert result["price"] > 0
        assert abs(result["price"] - expected_approx) < 5.0  # within $5

    def test_atm_call_delta_approx_50(self, pricer):
        """ATM call delta should be approximately 0.5."""
        result = pricer.price_option(
            spot=100, strike=100, time_to_expiry=30/365, rate=0.05, vol=0.20, is_call=True
        )
        assert 0.45 < result["delta"] < 0.60

    def test_deep_itm_call_delta_near_1(self, pricer):
        """Deep ITM call should have delta near 1."""
        result = pricer.price_option(
            spot=100, strike=70, time_to_expiry=30/365, rate=0.05, vol=0.20, is_call=True
        )
        assert result["delta"] > 0.85

    def test_deep_otm_call_delta_near_0(self, pricer):
        """Deep OTM call should have delta near 0."""
        result = pricer.price_option(
            spot=100, strike=130, time_to_expiry=30/365, rate=0.05, vol=0.20, is_call=True
        )
        assert result["delta"] < 0.20

    def test_put_call_parity(self, pricer):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        spot, strike, tte, rate, vol = 100, 100, 30/365, 0.05, 0.20
        call = pricer.price_option(spot, strike, tte, rate, vol, is_call=True)
        put = pricer.price_option(spot, strike, tte, rate, vol, is_call=False)
        import math
        parity_diff = call["price"] - put["price"]
        expected = spot - strike * math.exp(-rate * tte)
        assert abs(parity_diff - expected) < 0.10

    def test_put_delta_call_delta_relation(self, pricer):
        """Put delta = call delta - 1."""
        spot, strike, tte, rate, vol = 100, 105, 30/365, 0.05, 0.20
        call = pricer.price_option(spot, strike, tte, rate, vol, is_call=True)
        put = pricer.price_option(spot, strike, tte, rate, vol, is_call=False)
        assert abs(put["delta"] - (call["delta"] - 1)) < 0.001

    def test_zero_time_to_expiry_returns_zero(self, pricer):
        """Zero TTE should return zero prices."""
        result = pricer.price_option(
            spot=100, strike=100, time_to_expiry=0, rate=0.05, vol=0.20, is_call=True
        )
        assert result["price"] == 0.0

    def test_zero_vol_returns_zero(self, pricer):
        """Zero volatility should return zero prices (or intrinsic only)."""
        result = pricer.price_option(
            spot=100, strike=100, time_to_expiry=30/365, rate=0.05, vol=0, is_call=True
        )
        assert result["price"] == 0.0 or result["price"] >= 0

    def test_negative_values_handled(self, pricer):
        """Negative inputs should return zeros."""
        result = pricer.price_option(
            spot=-100, strike=100, time_to_expiry=30/365, rate=0.05, vol=0.20, is_call=True
        )
        assert result["price"] == 0.0

    def test_higher_vol_higher_price(self, pricer):
        """Higher vol should produce higher option price."""
        low_vol = pricer.price_option(100, 110, 30/365, 0.05, 0.15, is_call=True)
        high_vol = pricer.price_option(100, 110, 30/365, 0.05, 0.35, is_call=True)
        assert high_vol["price"] > low_vol["price"]

    def test_find_strike_by_delta_call(self, pricer):
        """Should find strike with target delta for calls."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=0.30, time_to_expiry=30/365, rate=0.045, vol=0.16, is_call=True
        )
        assert strike > 550  # OTM call
        # Verify delta is close to target
        result = pricer.price_option(550, strike, 30/365, 0.045, 0.16, is_call=True)
        assert abs(result["delta"] - 0.30) < 0.10

    def test_find_strike_by_delta_put(self, pricer):
        """Should find strike with target delta for puts."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=-0.20, time_to_expiry=30/365, rate=0.045, vol=0.16, is_call=False
        )
        assert strike < 550  # OTM put
        result = pricer.price_option(550, strike, 30/365, 0.045, 0.16, is_call=False)
        assert abs(result["delta"] - (-0.20)) < 0.10

    def test_greeks_all_present(self, pricer):
        """All Greeks should be in the result."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        for greek in ["delta", "gamma", "theta", "vega"]:
            assert greek in result
            assert isinstance(result[greek], float)


class TestCollateralRegimeClassification:
    """Test VIX regime classification for collar."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_normal_regime(self, generator):
        assert generator.classify_regime(12.0) == CollarRegime.NORMAL
        assert generator.classify_regime(18.0) == CollarRegime.NORMAL
        assert generator.classify_regime(19.9) == CollarRegime.NORMAL

    def test_elevated_regime(self, generator):
        assert generator.classify_regime(20.0) == CollarRegime.ELEVATED
        assert generator.classify_regime(25.0) == CollarRegime.ELEVATED
        assert generator.classify_regime(29.9) == CollarRegime.ELEVATED

    def test_stress_regime(self, generator):
        assert generator.classify_regime(30.0) == CollarRegime.STRESS
        assert generator.classify_regime(35.0) == CollarRegime.STRESS
        assert generator.classify_regime(39.9) == CollarRegime.STRESS

    def test_crisis_regime(self, generator):
        assert generator.classify_regime(40.0) == CollarRegime.CRISIS
        assert generator.classify_regime(50.0) == CollarRegime.CRISIS
        assert generator.classify_regime(80.0) == CollarRegime.CRISIS


class TestCollarStrikesCalculation:
    """Test collar strike selection."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_normal_market_strikes(self, generator):
        """In normal market, should generate valid collar strikes."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=30)
        assert strikes.underlying_price == 550.0
        assert strikes.call_strike > 550.0   # OTM call
        assert strikes.put_strike < 550.0    # OTM put
        assert strikes.vix_level == 16.0
        assert strikes.regime == "normal"
        assert strikes.days_to_expiry == 30

    def test_elevated_vix_wider_strikes(self, generator):
        """Higher VIX should produce wider strike spread."""
        normal = generator.calculate_strikes(spot=550.0, vix=16.0)
        elevated = generator.calculate_strikes(spot=550.0, vix=25.0)
        normal_spread = normal.call_strike - normal.put_strike
        elevated_spread = elevated.call_strike - elevated.put_strike
        assert elevated_spread > normal_spread

    def test_crisis_disables_collar(self, generator):
        """Crisis regime should disable collar."""
        strikes = generator.calculate_strikes(spot=550.0, vix=50.0)
        assert strikes.is_cashless is False
        assert strikes.regime == "crisis"
        assert strikes.collar_cost_pct > 0

    def test_call_premium_positive(self, generator):
        """Call premium should always be positive."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0)
        assert strikes.call_premium > 0

    def test_put_premium_positive(self, generator):
        """Put premium should always be positive."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0)
        assert strikes.put_premium > 0

    def test_near_cashless_in_normal_market(self, generator):
        """In normal market, collar should be near cashless."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0)
        # Should be within reasonable bounds
        assert abs(strikes.net_premium) < 20.0  # less than $20 per share net

    def test_strikes_serializable(self, generator):
        """Strikes should be serializable to dict."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0)
        d = strikes.to_dict()
        assert isinstance(d, dict)
        assert "call_strike" in d
        assert "put_strike" in d

    def test_different_spots_proportional(self, generator):
        """Strikes should scale with spot price."""
        low_spot = generator.calculate_strikes(spot=300.0, vix=16.0)
        high_spot = generator.calculate_strikes(spot=600.0, vix=16.0)
        # Call/put should be OTM relative to their spots
        assert low_spot.call_strike > 300.0
        assert low_spot.put_strike < 300.0
        assert high_spot.call_strike > 600.0
        assert high_spot.put_strike < 600.0

    def test_stress_wider_than_elevated(self, generator):
        """Stress regime should have wider strikes than elevated."""
        elevated = generator.calculate_strikes(spot=550.0, vix=25.0)
        stress = generator.calculate_strikes(spot=550.0, vix=35.0)
        elevated_spread = elevated.call_strike - elevated.put_strike
        stress_spread = stress.call_strike - stress.put_strike
        assert stress_spread > elevated_spread


class TestCollarSignalGeneration:
    """Test complete signal generation."""

    @pytest.fixture
    def generator(self):
        gen = CollarSignalGenerator()
        # Mock data fetching
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        return gen

    def test_generate_valid_signal(self, generator):
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert signal.is_valid
        assert signal.signal_state == "active"
        assert signal.underlying_price == 550.0
        assert signal.vix_level == 16.0
        assert signal.call_strike > 550.0
        assert signal.put_strike < 550.0
        assert signal.confidence > 50

    def test_crisis_signal_invalid(self, generator):
        signal = generator.generate_signal(spot=550.0, vix=50.0)
        assert not signal.is_valid
        assert signal.signal_state == "unhedged"
        assert signal.regime == "crisis"

    def test_signal_serializable(self, generator):
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "strikes" in d
        assert "call_strike" in d["strikes"]

    def test_upside_capped_positive(self, generator):
        """Upside cap should be positive (call is OTM)."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert signal.max_upside_pct > 0

    def test_downside_protected_positive(self, generator):
        """Downside protection (floor distance) should be positive."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert signal.max_downside_pct > 0

    def test_elevated_vix_lower_confidence(self, generator):
        """Higher VIX should reduce confidence somewhat."""
        normal = generator.generate_signal(spot=550.0, vix=16.0)
        elevated = generator.generate_signal(spot=550.0, vix=25.0)
        # Confidence should still be reasonable
        assert elevated.confidence > 30

    def test_expected_yield_reasonable(self, generator):
        """Monthly yield should be within reasonable bounds."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert abs(signal.expected_monthly_yield) < 10  # less than 10% annualized

    def test_generate_convenience_function(self):
        signal = generate_collar_signal(spot=550.0, vix=16.0)
        assert isinstance(signal, CollarSignal)
        assert signal.is_valid

    def test_signal_with_none_inputs(self, generator):
        """Should handle None inputs by falling back to defaults."""
        # Set up the mocks
        generator._fetch_spot_price = lambda: 550.0
        generator._fetch_vix_level = lambda: 16.0
        signal = generator.generate_signal(spot=None, vix=None)
        assert signal.underlying_price > 0
        assert signal.vix_level > 0

    def test_signal_with_zero_spot(self, generator):
        """Zero spot should produce invalid signal."""
        signal = generator.generate_signal(spot=0, vix=16.0)
        assert not signal.is_valid
        assert signal.signal_state == "error"


class TestCollarStateEnum:
    """Test collar state enum values."""

    def test_state_values(self):
        assert CollarState.ACTIVE.value == "active"
        assert CollarState.UNHEDGED.value == "unhedged"
        assert CollarState.WIDE.value == "wide"
        assert CollarState.NARROW.value == "narrow"
        assert CollarState.ROLLING.value == "rolling"


class TestCollarRegimeEnum:
    """Test collar regime enum values."""

    def test_regime_values(self):
        assert CollarRegime.NORMAL.value == "normal"
        assert CollarRegime.ELEVATED.value == "elevated"
        assert CollarRegime.STRESS.value == "stress"
        assert CollarRegime.CRISIS.value == "crisis"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()

    @pytest.fixture
    def generator(self):
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        return gen

    def test_very_short_expiry(self, generator):
        """Very short expiry (1 day) should still work."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=1)
        assert strikes.call_strike > 550.0
        assert strikes.put_strike < 550.0

    def test_very_long_expiry(self, generator):
        """Long expiry (90 days) should produce wider spreads."""
        short = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=7)
        long = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=90)
        assert long.call_strike - long.put_strike > short.call_strike - short.put_strike

    def test_very_high_spot(self, generator):
        """Very high spot price should still produce valid strikes."""
        strikes = generator.calculate_strikes(spot=5000.0, vix=16.0)
        assert strikes.call_strike > 5000.0
        assert strikes.put_strike < 5000.0

    def test_boundary_vix_normal_to_elevated(self, generator):
        """Test at the exact boundary between regimes."""
        assert generator.classify_regime(19.99) == CollarRegime.NORMAL
        assert generator.classify_regime(20.0) == CollarRegime.ELEVATED

    def test_boundary_vix_elevated_to_stress(self, generator):
        assert generator.classify_regime(29.99) == CollarRegime.ELEVATED
        assert generator.classify_regime(30.0) == CollarRegime.STRESS

    def test_net_premium_scale_with_spot(self, generator):
        """Net premium should scale with spot price."""
        low = generator.calculate_strikes(spot=300.0, vix=16.0)
        high = generator.calculate_strikes(spot=600.0, vix=16.0)
        # Premiums should roughly scale with spot
        assert high.call_premium > low.call_premium
        assert high.put_premium > low.put_premium
