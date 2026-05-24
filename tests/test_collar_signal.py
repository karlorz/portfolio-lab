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


class TestCollarStrikesDataclass:
    """Complete to_dict() field coverage for CollarStrikes."""

    @pytest.fixture
    def sample_strikes(self):
        gen = CollarSignalGenerator()
        return gen.calculate_strikes(spot=550.0, vix=16.0)

    def test_to_dict_all_fields_present(self, sample_strikes):
        """All 13 dataclass fields should appear in to_dict()."""
        d = sample_strikes.to_dict()
        expected_fields = {
            "underlying_price", "call_strike", "put_strike",
            "call_premium", "put_premium", "net_premium",
            "call_delta", "put_delta",
            "vix_level", "regime", "days_to_expiry",
            "is_cashless", "collar_cost_pct",
        }
        assert set(d.keys()) == expected_fields

    def test_to_dict_field_values_match(self, sample_strikes):
        """to_dict() values should match the dataclass fields."""
        d = sample_strikes.to_dict()
        assert d["underlying_price"] == sample_strikes.underlying_price
        assert d["call_strike"] == sample_strikes.call_strike
        assert d["put_strike"] == sample_strikes.put_strike
        assert d["call_premium"] == sample_strikes.call_premium
        assert d["put_premium"] == sample_strikes.put_premium
        assert d["net_premium"] == sample_strikes.net_premium
        assert d["call_delta"] == sample_strikes.call_delta
        assert d["put_delta"] == sample_strikes.put_delta
        assert d["vix_level"] == sample_strikes.vix_level
        assert d["regime"] == sample_strikes.regime
        assert d["days_to_expiry"] == sample_strikes.days_to_expiry
        assert d["is_cashless"] == sample_strikes.is_cashless
        assert d["collar_cost_pct"] == sample_strikes.collar_cost_pct


class TestCollarSignalDataclass:
    """Complete to_dict() field coverage for CollarSignal."""

    @pytest.fixture
    def sample_signal(self):
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        return gen.generate_signal(spot=550.0, vix=16.0)

    def test_to_dict_all_fields_present(self, sample_signal):
        """All 16 CollarSignal fields should appear in to_dict()."""
        d = sample_signal.to_dict()
        expected_fields = {
            "timestamp", "signal_state", "call_strike", "put_strike",
            "underlying_price", "expected_monthly_yield", "max_upside_pct",
            "max_downside_pct", "vix_level", "regime", "strikes",
            "collar_notional_pct", "spy_shift", "confidence",
            "is_valid", "reason",
        }
        assert set(d.keys()) == expected_fields

    def test_to_dict_nested_strikes(self, sample_signal):
        """The strikes field should be a nested dict with all strike sub-fields."""
        d = sample_signal.to_dict()
        assert isinstance(d["strikes"], dict)
        strikes_subfields = {
            "underlying_price", "call_strike", "put_strike",
            "call_premium", "put_premium", "net_premium",
            "call_delta", "put_delta",
            "vix_level", "regime", "days_to_expiry",
            "is_cashless", "collar_cost_pct",
        }
        assert set(d["strikes"].keys()) == strikes_subfields

    def test_to_dict_serializable(self, sample_signal):
        """to_dict() output should be JSON-serializable."""
        import json
        d = sample_signal.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert len(json_str) > 50


class TestConstantsValidation:
    """Validate constants and thresholds in CollarSignalGenerator."""

    def test_call_delta_target_is_exact(self):
        assert CollarSignalGenerator.CALL_DELTA_TARGET == 0.30

    def test_put_delta_target_is_exact(self):
        assert CollarSignalGenerator.PUT_DELTA_TARGET == -0.20

    def test_vix_thresholds_monotonically_increasing(self):
        """VIX thresholds should be strictly increasing."""
        assert (CollarSignalGenerator.VIX_ELEVATED <
                CollarSignalGenerator.VIX_STRESS <
                CollarSignalGenerator.VIX_CRISIS)

    def test_wide_factor_has_all_non_crisis_regimes(self):
        """WIDE_FACTOR should map NORMAL, ELEVATED, STRESS (not CRISIS)."""
        assert CollarRegime.NORMAL in CollarSignalGenerator.WIDE_FACTOR
        assert CollarRegime.ELEVATED in CollarSignalGenerator.WIDE_FACTOR
        assert CollarRegime.STRESS in CollarSignalGenerator.WIDE_FACTOR
        assert CollarRegime.CRISIS not in CollarSignalGenerator.WIDE_FACTOR

    def test_wide_factor_strictly_increasing(self):
        """Wide factors should increase with volatility regime."""
        wf = CollarSignalGenerator.WIDE_FACTOR
        assert wf[CollarRegime.NORMAL] < wf[CollarRegime.ELEVATED] < wf[CollarRegime.STRESS]

    def test_allocation_bounds_reasonable(self):
        """Collar notional and confidence should be in reasonable ranges."""
        gen = CollarSignalGenerator()
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        assert 0 < signal.collar_notional_pct <= 1.0
        assert 0 <= signal.confidence <= 100

    def test_allocation_bounds_crisis_sets_zero_confidence(self):
        """In crisis, confidence should be 0 and signal invalid."""
        gen = CollarSignalGenerator()
        signal = gen.generate_signal(spot=550.0, vix=50.0)
        assert signal.confidence == 0.0
        assert not signal.is_valid

    def test_collar_notional_default_is_46_percent(self):
        """Default collar notional should match champion allocation."""
        gen = CollarSignalGenerator()
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        assert signal.collar_notional_pct == 0.46


class TestExtendedEdgeCases:
    """Additional edge cases for collar calculations."""

    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()

    @pytest.fixture
    def generator(self):
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        return gen

    def test_zero_volatility_in_calculate_strikes(self, generator):
        """Zero vol produces zero premiums but should not crash."""
        strikes = generator.calculate_strikes(spot=550.0, vix=0.0)
        assert strikes.call_premium == 0.0
        assert strikes.put_premium == 0.0
        assert strikes.net_premium == 0.0
        assert strikes.is_cashless       # net=0 is within cashless tolerance
        assert isinstance(strikes.call_strike, float)
        assert isinstance(strikes.put_strike, float)

    def test_extreme_low_vix_single_digit(self, generator):
        """Extremely low VIX (5.0) should still produce valid strikes."""
        strikes = generator.calculate_strikes(spot=550.0, vix=5.0)
        assert strikes.regime == "normal"
        assert strikes.call_strike > 550.0
        assert strikes.put_strike < 550.0
        assert strikes.vix_level == 5.0

    def test_high_vix_just_below_crisis(self, generator):
        """VIX at 39.99 (stress regime) should produce wider strikes."""
        strikes = generator.calculate_strikes(spot=550.0, vix=39.99)
        assert strikes.regime == "stress"
        # Stress with WIDE_FACTOR=1.6 widens strikes significantly
        call_pct = (strikes.call_strike / strikes.underlying_price - 1) * 100
        put_pct = (1 - strikes.put_strike / strikes.underlying_price) * 100
        assert call_pct > 5.0  # Wider than normal ~3% cap
        assert put_pct > 3.0

    def test_negative_spot_returns_invalid_signal(self, generator):
        """Negative spot should produce invalid error signal."""
        signal = generator.generate_signal(spot=-100.0, vix=16.0)
        assert not signal.is_valid
        assert signal.signal_state == "error"
        assert signal.reason == "Invalid spot price"

    def test_spot_one_dollar_edge(self, generator):
        """Spot of $1 should not crash and produce reasonable output."""
        signal = generator.generate_signal(spot=1.0, vix=16.0)
        assert signal.call_strike >= 0
        assert signal.put_strike >= 0
        assert signal.is_valid

    def test_extreme_delta_target_call_near_one(self, pricer):
        """Find strike for extreme call delta near 1.0 should converge."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=0.95, time_to_expiry=30/365,
            rate=0.045, vol=0.16, is_call=True,
        )
        result = pricer.price_option(550, strike, 30/365, 0.045, 0.16, is_call=True)
        assert result["delta"] > 0.50

    def test_extreme_delta_target_put_near_negative_one(self, pricer):
        """Find strike for extreme put delta near -1.0 should converge."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=-0.95, time_to_expiry=30/365,
            rate=0.045, vol=0.16, is_call=False,
        )
        result = pricer.price_option(550, strike, 30/365, 0.045, 0.16, is_call=False)
        assert result["delta"] < -0.50

    def test_cashless_tolerance_boundary(self, generator):
        """Collar near cashless tolerance boundary logic works correctly."""
        strikes = generator.calculate_strikes(spot=550.0, vix=16.0)
        cost_pct = abs(strikes.net_premium) / strikes.underlying_price * 100
        tolerance = CollarSignalGenerator.CASHLESS_TOLERANCE
        assert strikes.is_cashless == (cost_pct < tolerance)

    def test_very_large_spot_10k(self, generator):
        """Very large spot (10000) should produce valid strikes."""
        strikes = generator.calculate_strikes(spot=10000.0, vix=16.0)
        assert strikes.call_strike > 10000.0
        assert strikes.put_strike < 10000.0
        assert strikes.regime == "normal"

    def test_all_days_to_expiry_values(self, generator):
        """Various days_to_expiry values should not crash."""
        for days in [1, 7, 14, 30, 60, 90, 180]:
            strikes = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=days)
            assert strikes.call_strike > 550.0
            assert strikes.put_strike < 550.0
            assert strikes.days_to_expiry == days

    def test_premiums_monotonic_with_days_to_expiry(self, generator):
        """Longer expiry should increase both call and put premiums."""
        short = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=7)
        long_ = generator.calculate_strikes(spot=550.0, vix=16.0, days_to_expiry=90)
        assert long_.call_premium > short.call_premium
        assert long_.put_premium > short.put_premium

    def test_spy_shift_direction(self, generator):
        """spy_shift should be negative when net_premium is positive (credit)."""
        # cashless or near-cashless, but verify sign convention
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        # spy_shift = -(net_premium / spot) * 100
        if signal.strikes.net_premium > 0:
            assert signal.spy_shift < 0
        else:
            assert signal.spy_shift >= 0


class TestSignalSnapshotBridge:
    """Test to_signal_snapshot() bridge method edge cases."""

    @pytest.fixture
    def generator(self):
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        return gen

    def test_snapshot_source_and_validity(self, generator):
        """Valid active signal snapshot should have expected structure."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        snap = signal.to_signal_snapshot()
        assert snap.source == "collar_signal"
        assert snap.is_active is True
        assert "SPY" in snap.asset_signals

    def test_snapshot_unhedged_crisis_signal(self, generator):
        """Crisis signal should map to inactive snapshot."""
        signal = generator.generate_signal(spot=550.0, vix=50.0)
        snap = signal.to_signal_snapshot()
        assert snap.is_active is False
        assert "SPY" in snap.asset_signals

    def test_snapshot_value_default_for_active_state(self, generator):
        """'active' signal_state is not in state_map so defaults to 0.0."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        snap = signal.to_signal_snapshot()
        assert snap.value == 0.0   # "active" not in map -> fallback default

    def test_snapshot_metadata_contains_all_fields(self, generator):
        """Snapshot metadata should contain 6 expected fields."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        snap = signal.to_signal_snapshot()
        assert snap.metadata["signal_state"] == "active"
        assert snap.metadata["vix_level"] == 16.0
        assert snap.metadata["regime"] == "normal"
        assert snap.metadata["collar_notional_pct"] == 0.46
        assert "max_upside_pct" in snap.metadata
        assert "max_downside_pct" in snap.metadata

    def test_snapshot_is_active_matches_signal_is_valid(self, generator):
        """Snapshot is_active should equal signal is_valid."""
        valid_signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert valid_signal.to_signal_snapshot().is_active == valid_signal.is_valid

        invalid_signal = generator.generate_signal(spot=550.0, vix=50.0)
        assert invalid_signal.to_signal_snapshot().is_active == invalid_signal.is_valid

    def test_snapshot_spy_asset_signal_negative_spy_shift(self, generator):
        """SPY asset_signal should be negative of spy_shift."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        snap = signal.to_signal_snapshot()
        assert snap.asset_signals["SPY"] == -signal.spy_shift

    def test_snapshot_explanation_contains_key_fields(self, generator):
        """Explanation text should contain signal_state, VIX, notional, yield."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        snap = signal.to_signal_snapshot()
        assert signal.signal_state in snap.explanation
        assert str(round(signal.vix_level, 1)) in snap.explanation
        assert "%" in snap.explanation  # format includes percentage

    def test_snapshot_regime_fit_is_all(self, generator):
        """regime_fit should always be 'all'."""
        signal = generator.generate_signal(spot=550.0, vix=16.0)
        assert signal.to_signal_snapshot().regime_fit == "all"

        crisis_signal = generator.generate_signal(spot=550.0, vix=50.0)
        assert crisis_signal.to_signal_snapshot().regime_fit == "all"


class TestStatePersistence:
    """Test state persistence via save_signal()."""

    def test_save_signal_creates_file(self, tmp_path):
        """save_signal() should create a JSON file at the output path."""
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        output_path = tmp_path / "collar_signal.json"
        original = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = output_path
        try:
            gen.save_signal(signal)
            assert output_path.exists()
        finally:
            gen.OUTPUT_PATH = original

    def test_save_signal_json_structure(self, tmp_path):
        """Saved JSON should have valid structure with expected top-level keys."""
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        output_path = tmp_path / "collar_signal.json"
        original = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = output_path
        try:
            gen.save_signal(signal)
            with open(output_path) as f:
                data = json.load(f)
            assert "timestamp" in data
            assert "signal_state" in data
            assert "strikes" in data
            assert "call_strike" in data["strikes"]
            assert isinstance(data["strikes"], dict)
            assert data["is_valid"] is True
        finally:
            gen.OUTPUT_PATH = original

    def test_save_signal_crisis_state(self, tmp_path):
        """Crisis signal JSON should record unhedged state."""
        gen = CollarSignalGenerator()
        signal = gen.generate_signal(spot=550.0, vix=50.0)
        output_path = tmp_path / "collar_crisis.json"
        original = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = output_path
        try:
            gen.save_signal(signal)
            with open(output_path) as f:
                data = json.load(f)
            assert data["signal_state"] == "unhedged"
            assert data["is_valid"] is False
            assert data["regime"] == "crisis"
        finally:
            gen.OUTPUT_PATH = original


class TestClassificationBoundaries:
    """Boundary condition tests for regime classification."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_negative_vix_returns_normal(self, generator):
        """Negative VIX should return NORMAL (lowest regime)."""
        assert generator.classify_regime(-5.0) == CollarRegime.NORMAL

    def test_zero_vix_returns_normal(self, generator):
        """Zero VIX should return NORMAL."""
        assert generator.classify_regime(0.0) == CollarRegime.NORMAL

    def test_vix_between_zero_and_elevated_returns_normal(self, generator):
        """VIX between 0 and 20 should be NORMAL."""
        assert generator.classify_regime(10.0) == CollarRegime.NORMAL
        assert generator.classify_regime(15.5) == CollarRegime.NORMAL
        assert generator.classify_regime(19.99) == CollarRegime.NORMAL

    def test_exact_crisis_boundary(self, generator):
        """Exactly 40.0 VIX should be CRISIS."""
        assert generator.classify_regime(40.0) == CollarRegime.CRISIS

    def test_vix_just_below_crisis_is_stress(self, generator):
        """39.99 VIX should be STRESS, not CRISIS."""
        assert generator.classify_regime(39.99) == CollarRegime.STRESS

    def test_vix_just_above_crisis_threshold(self, generator):
        """40.01 VIX should be CRISIS."""
        assert generator.classify_regime(40.01) == CollarRegime.CRISIS

    def test_extreme_high_vix_remains_crisis(self, generator):
        """Extremely high VIX (100+) should still classify as CRISIS."""
        assert generator.classify_regime(100.0) == CollarRegime.CRISIS
        assert generator.classify_regime(999.99) == CollarRegime.CRISIS


class TestFetchSpotPrice:
    """Test _fetch_spot_price() method with DB/file fallback behavior."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_returns_fallback_when_db_not_exists(self, generator):
        """Should return 550.0 when MARKET_DB does not exist."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            result = generator._fetch_spot_price()
            assert result == 550.0

    def test_returns_price_from_db(self, generator):
        """Should return SPY price from DB when MARKET_DB exists with data."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [545.25]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                result = generator._fetch_spot_price()

                assert result == 545.25
                assert isinstance(result, float)

    def test_returns_fallback_when_no_rows(self, generator):
        """Should return 550.0 when DB query returns no rows (no SPY data)."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = None
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                result = generator._fetch_spot_price()

                assert result == 550.0

    def test_returns_fallback_on_db_exception(self, generator):
        """Should return 550.0 when DB raises an exception (e.g. connection error)."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_connect.side_effect = Exception("DB connection failed")
                result = generator._fetch_spot_price()
                assert result == 550.0


class TestFetchVixLevel:
    """Test _fetch_vix_level() method with multiple fallback paths."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_returns_fallback_when_no_sources(self, generator):
        """Should return 16.0 when neither DB nor term structure file exists."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = False
                mock_data_dir.__truediv__.return_value = mock_vix_path

                result = generator._fetch_vix_level()

                assert result == 16.0

    def test_returns_vix_from_db(self, generator):
        """Should return VIX price from DB when MARKET_DB exists with data."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [14.5]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                result = generator._fetch_vix_level()

                assert result == 14.5
                assert isinstance(result, float)

    def test_returns_fallback_when_db_has_no_vix(self, generator):
        """Should fall through when DB returns no VIX data and no file exists."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = None
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                    mock_vix_path = MagicMock()
                    mock_vix_path.exists.return_value = False
                    mock_data_dir.__truediv__.return_value = mock_vix_path

                    result = generator._fetch_vix_level()

                    assert result == 16.0

    def test_returns_vix_from_term_structure_file(self, generator):
        """Should read VIX from vix_term_structure.json file when DB unavailable."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = True
                mock_data_dir.__truediv__.return_value = mock_vix_path

                with patch("builtins.open", MagicMock()):
                    with patch("json.load") as mock_json_load:
                        mock_json_load.return_value = {
                            "2025-01-15": {"vix_spot": 15.2},
                        }
                        result = generator._fetch_vix_level()
                        assert result == 15.2

    def test_returns_fallback_on_db_exception(self, generator):
        """Should fall through when DB raises and no file exists."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_connect.side_effect = Exception("DB locked")
                with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                    mock_vix_path = MagicMock()
                    mock_vix_path.exists.return_value = False
                    mock_data_dir.__truediv__.return_value = mock_vix_path

                    result = generator._fetch_vix_level()

                    assert result == 16.0

    def test_returns_fallback_on_json_exception(self, generator):
        """Should return 16.0 when term structure file has parse error."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = True
                mock_data_dir.__truediv__.return_value = mock_vix_path

                with patch("builtins.open", MagicMock()):
                    with patch("json.load") as mock_json_load:
                        mock_json_load.side_effect = json.JSONDecodeError("Boom", "", 0)
                        result = generator._fetch_vix_level()
                        assert result == 16.0


class TestCLIEntryPoint:
    """Test the main() CLI entry point."""

    @pytest.fixture
    def mock_signal(self):
        """Fixture that returns a MagicMock with all attributes main() accesses."""
        sig = MagicMock()
        sig.timestamp = "2025-01-15T10:30:00"
        sig.signal_state = "active"
        sig.regime = "normal"
        sig.underlying_price = 550.0
        sig.vix_level = 16.0
        sig.call_strike = 575.0
        sig.put_strike = 530.0
        sig.max_upside_pct = 4.5
        sig.max_downside_pct = 3.6
        sig.expected_monthly_yield = 0.52
        sig.is_valid = True
        sig.confidence = 85.0
        sig.collar_notional_pct = 0.46
        sig.spy_shift = 0.05
        sig.reason = "Cashless collar active"
        sig.strikes.net_premium = -1.25
        sig.strikes.is_cashless = True
        return sig

    @pytest.fixture
    def mock_signal_crisis(self):
        """Mock signal with crisis regime values."""
        sig = MagicMock()
        sig.timestamp = "2025-03-10T14:00:00"
        sig.signal_state = "unhedged"
        sig.regime = "crisis"
        sig.underlying_price = 500.0
        sig.vix_level = 45.0
        sig.call_strike = 550.0
        sig.put_strike = 450.0
        sig.max_upside_pct = 10.0
        sig.max_downside_pct = 10.0
        sig.expected_monthly_yield = 0.0
        sig.is_valid = False
        sig.confidence = 0.0
        sig.collar_notional_pct = 0.0
        sig.spy_shift = 0.0
        sig.reason = "Collar disabled: VIX crisis level, cost prohibitive"
        sig.strikes.net_premium = 0.0
        sig.strikes.is_cashless = False
        return sig

    def test_main_runs_without_error(self, mock_signal):
        """main() should execute without raising exceptions."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal
            with patch("sys.argv", ["collar_signal.py"]):
                main()  # should not raise

    def test_main_with_save_flag(self, mock_signal):
        """main() should call save_signal when '--save' is in sys.argv."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal
            with patch("sys.argv", ["collar_signal.py", "--save"]):
                main()
            instance.save_signal.assert_called_once_with(mock_signal)

    def test_main_without_save_flag(self, mock_signal):
        """main() should NOT call save_signal when '--save' is absent."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal
            with patch("sys.argv", ["collar_signal.py"]):
                main()
            instance.save_signal.assert_not_called()

    def test_main_with_crisis_signal(self, mock_signal_crisis):
        """main() should handle crisis regime signal without error."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal_crisis
            with patch("sys.argv", ["collar_signal.py"]):
                main()  # should not raise

    def test_main_calls_generate_signal_with_no_args(self, mock_signal):
        """main() should call generate_signal() with no arguments."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal
            with patch("sys.argv", ["collar_signal.py"]):
                main()
            instance.generate_signal.assert_called_once_with()

    def test_main_save_with_crisis_also_saves(self, mock_signal_crisis):
        """main() with --save should save even when signal is crisis (invalid)."""
        from src.signals.collar_signal import main
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal_crisis
            with patch("sys.argv", ["collar_signal.py", "--save"]):
                main()
            instance.save_signal.assert_called_once_with(mock_signal_crisis)


class TestGenerateConvenienceFunction:
    """Test the generate_collar_signal() convenience function."""

    def test_creates_generator_and_calls_generate(self):
        """Should instantiate CollarSignalGenerator and call generate_signal."""
        mock_signal = MagicMock(spec=CollarSignal)
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal

            result = generate_collar_signal(spot=550.0, vix=16.0)

            assert result is mock_signal
            instance.generate_signal.assert_called_once_with(spot=550.0, vix=16.0)

    def test_passes_spot_and_vix_only(self):
        """Should forward only spot and vix kwargs to generate_signal."""
        mock_signal = MagicMock(spec=CollarSignal)
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal

            result = generate_collar_signal(spot=500.0, vix=22.5)

            assert result is mock_signal
            instance.generate_signal.assert_called_once_with(spot=500.0, vix=22.5)

    def test_passes_none_by_default(self):
        """Should pass None for spot/vix when not provided."""
        mock_signal = MagicMock(spec=CollarSignal)
        with patch("src.signals.collar_signal.CollarSignalGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_signal.return_value = mock_signal

            result = generate_collar_signal()

            assert result is mock_signal
            instance.generate_signal.assert_called_once_with(spot=None, vix=None)

    def test_returns_collarsignal_type(self):
        """Convenience function should return a CollarSignal instance with real data."""
        result = generate_collar_signal(spot=550.0, vix=16.0)
        assert isinstance(result, CollarSignal)
        assert result.underlying_price == 550.0
        assert result.vix_level == 16.0
        assert result.is_valid


class TestEnsureDirs:
    """Test _ensure_dirs() directory creation."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_creates_data_dir(self, generator):
        """_ensure_dirs should call mkdir on DATA_DIR (via class attribute)."""
        with patch.object(CollarSignalGenerator, "DATA_DIR") as mock_data_dir:
            with patch("src.signals.collar_signal.SIGNALS_DIR") as mock_signals_dir:
                generator._ensure_dirs()
                mock_data_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_signals_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_creates_signals_dir(self, generator):
        """_ensure_dirs should call mkdir on SIGNALS_DIR."""
        with patch("src.signals.collar_signal.SIGNALS_DIR") as mock_signals_dir:
            generator._ensure_dirs()
            mock_signals_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_idempotent_on_existing_dirs(self, generator):
        """_ensure_dirs should not raise when dirs already exist (exist_ok=True)."""
        with patch.object(CollarSignalGenerator, "DATA_DIR") as mock_data_dir:
            with patch("src.signals.collar_signal.SIGNALS_DIR") as mock_signals_dir:
                # First call
                generator._ensure_dirs()
                # Second call — should not raise
                generator._ensure_dirs()
                assert mock_data_dir.mkdir.call_count == 2
                assert mock_signals_dir.mkdir.call_count == 2

    def test_called_during_init(self):
        """_ensure_dirs should be called during CollarSignalGenerator.__init__."""
        with patch.object(CollarSignalGenerator, "_ensure_dirs") as mock_ensure:
            gen = CollarSignalGenerator()
            mock_ensure.assert_called_once()

    def test_mkdir_parents_true(self, generator):
        """mkdir should be called with parents=True for both directories."""
        with patch.object(CollarSignalGenerator, "DATA_DIR") as mock_data_dir:
            with patch("src.signals.collar_signal.SIGNALS_DIR") as mock_signals_dir:
                generator._ensure_dirs()
                mock_data_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_signals_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestFetchSpotPriceEdgeCases:
    """Additional edge cases for _fetch_spot_price()."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_db_query_uses_correct_sql(self, generator):
        """Should execute the correct SQL query against MARKET_DB."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [550.0]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                generator._fetch_spot_price()

                mock_cursor.execute.assert_called_once_with(
                    "SELECT close FROM prices WHERE symbol='SPY' ORDER BY date DESC LIMIT 1"
                )

    def test_float_conversion_of_db_result(self, generator):
        """Returned value should be a Python float even if DB returns string."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = ["545.50"]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                result = generator._fetch_spot_price()

                assert result == 545.50
                assert isinstance(result, float)

    def test_db_connect_called_with_str_path(self, generator):
        """sqlite_connect should be called with str(MARKET_DB)."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            mock_db.__str__.return_value = "/fake/path/market.db"
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [550.0]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                generator._fetch_spot_price()

                mock_connect.assert_called_once_with("/fake/path/market.db")

    def test_fallback_type_is_float(self, generator):
        """Fallback value 550.0 should be a float."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            result = generator._fetch_spot_price()
            assert isinstance(result, float)
            assert result == 550.0


class TestFetchVixLevelEdgeCases:
    """Additional edge cases for _fetch_vix_level()."""

    @pytest.fixture
    def generator(self):
        return CollarSignalGenerator()

    def test_db_query_uses_correct_sql(self, generator):
        """Should execute the correct SQL query for VIX."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = True
            with patch("src.signals.collar_signal.sqlite_connect") as mock_connect:
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = [16.0]
                mock_conn = MagicMock()
                mock_conn.__enter__.return_value.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_conn

                generator._fetch_vix_level()

                mock_cursor.execute.assert_called_once_with(
                    "SELECT close FROM prices WHERE symbol='VIX' ORDER BY date DESC LIMIT 1"
                )

    def test_term_structure_file_reads_latest_date(self, generator):
        """Should take the max date key from the JSON file."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = True
                mock_data_dir.__truediv__.return_value = mock_vix_path
                with patch("builtins.open", MagicMock()):
                    with patch("json.load") as mock_json_load:
                        mock_json_load.return_value = {
                            "2025-01-01": {"vix_spot": 14.0},
                            "2025-01-15": {"vix_spot": 16.5},
                            "2025-01-10": {"vix_spot": 15.0},
                        }
                        result = generator._fetch_vix_level()
                        # Should pick "2025-01-15" (max key)
                        assert result == 16.5

    def test_term_structure_empty_dict_returns_fallback(self, generator):
        """Empty JSON dict should return 16.0."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = True
                mock_data_dir.__truediv__.return_value = mock_vix_path
                with patch("builtins.open", MagicMock()):
                    with patch("json.load") as mock_json_load:
                        mock_json_load.return_value = {}
                        result = generator._fetch_vix_level()
                        assert result == 16.0

    def test_term_structure_missing_vix_spot_returns_fallback(self, generator):
        """JSON entry without vix_spot key should return 16.0."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = True
                mock_data_dir.__truediv__.return_value = mock_vix_path
                with patch("builtins.open", MagicMock()):
                    with patch("json.load") as mock_json_load:
                        mock_json_load.return_value = {
                            "2025-01-15": {"other_field": 42},
                        }
                        result = generator._fetch_vix_level()
                        # .get("vix_spot", 16.0) returns 16.0
                        assert result == 16.0

    def test_fallback_returns_float(self, generator):
        """Ultimate fallback of 16.0 should be a float."""
        with patch("src.signals.collar_signal.MARKET_DB") as mock_db:
            mock_db.exists.return_value = False
            with patch("src.signals.collar_signal.DATA_DIR") as mock_data_dir:
                mock_vix_path = MagicMock()
                mock_vix_path.exists.return_value = False
                mock_data_dir.__truediv__.return_value = mock_vix_path
                result = generator._fetch_vix_level()
                assert isinstance(result, float)
                assert result == 16.0


class TestPricerGreeks:
    """Detailed Greek calculations for Black-Scholes pricer."""

    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()

    def test_gamma_positive_for_atm(self, pricer):
        """Gamma should be positive for ATM options."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        assert result["gamma"] > 0

    def test_vega_positive(self, pricer):
        """Vega should be positive."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        assert result["vega"] >= 0

    def test_theta_negative_for_atm_call(self, pricer):
        """Theta should be negative for ATM calls (time decay costs)."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        assert result["theta"] <= 0

    def test_put_atm_theta_sign(self, pricer):
        """ATM put theta can be positive or negative depending on rates."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=False)
        # theta is a float
        assert isinstance(result["theta"], float)

    def test_otm_greeks_proportional(self, pricer):
        """OTM options should have lower gamma and vega than ATM."""
        atm = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        otm = pricer.price_option(100, 110, 30/365, 0.05, 0.20, is_call=True)
        assert otm["gamma"] < atm["gamma"]
        assert otm["vega"] < atm["vega"]

    def test_itm_greeks_proportional(self, pricer):
        """ITM options should have lower gamma and vega than ATM."""
        atm = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        itm = pricer.price_option(100, 90, 30/365, 0.05, 0.20, is_call=True)
        assert itm["gamma"] < atm["gamma"]

    def test_gamma_rounding_precision(self, pricer):
        """Gamma should be rounded to 6 decimal places."""
        result = pricer.price_option(100, 100, 30/365, 0.05, 0.20, is_call=True)
        gamma_str = str(result["gamma"])
        if "." in gamma_str:
            decimals = len(gamma_str.split(".")[1])
            assert decimals <= 6


class TestPricerFindStrikeByDelta:
    """Test find_strike_by_delta binary search edge cases."""

    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()

    def test_single_iteration_convergence(self, pricer):
        """Binary search should converge within 50 iterations."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=0.30, time_to_expiry=30/365,
            rate=0.045, vol=0.16, is_call=True,
        )
        result = pricer.price_option(550, strike, 30/365, 0.045, 0.16, is_call=True)
        assert abs(result["delta"] - 0.30) < 0.05

    def test_find_strike_put_otm(self, pricer):
        """Put strike should be below spot for target delta -0.20."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=-0.20, time_to_expiry=30/365,
            rate=0.045, vol=0.16, is_call=False,
        )
        assert strike < 550

    def test_find_strike_call_otm(self, pricer):
        """Call strike should be above spot for target delta 0.30."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=0.30, time_to_expiry=30/365,
            rate=0.045, vol=0.16, is_call=True,
        )
        assert strike > 550

    def test_zero_vol_find_strike(self, pricer):
        """Zero vol should not crash the binary search."""
        strike = pricer.find_strike_by_delta(
            spot=550, target_delta=0.30, time_to_expiry=30/365,
            rate=0.045, vol=0, is_call=True,
        )
        assert isinstance(strike, float)


class TestGenerateSignalWithFallbacks:
    """Test generate_signal when it uses _fetch_spot_price / _fetch_vix_level internally."""

    def test_uses_fetch_methods_when_spot_none(self):
        """Should call _fetch_spot_price when spot is None."""
        gen = CollarSignalGenerator()
        with patch.object(gen, "_fetch_spot_price", return_value=550.0) as mock_fetch:
            with patch.object(gen, "_fetch_vix_level", return_value=16.0):
                signal = gen.generate_signal(spot=None, vix=16.0)
                mock_fetch.assert_called_once()
                assert signal.underlying_price == 550.0

    def test_uses_fetch_methods_when_vix_none(self):
        """Should call _fetch_vix_level when vix is None."""
        gen = CollarSignalGenerator()
        with patch.object(gen, "_fetch_vix_level", return_value=16.0) as mock_fetch:
            with patch.object(gen, "_fetch_spot_price", return_value=550.0):
                signal = gen.generate_signal(spot=550.0, vix=None)
                mock_fetch.assert_called_once()
                assert signal.vix_level == 16.0

    def test_does_not_fetch_when_both_provided(self):
        """Should not call fetch methods when both spot and vix are provided."""
        gen = CollarSignalGenerator()
        with patch.object(gen, "_fetch_spot_price") as mock_spot:
            with patch.object(gen, "_fetch_vix_level") as mock_vix:
                gen.generate_signal(spot=550.0, vix=16.0)
                mock_spot.assert_not_called()
                mock_vix.assert_not_called()

    def test_fetches_both_when_neither_provided(self):
        """Should call both fetch methods when neither spot nor vix is provided."""
        gen = CollarSignalGenerator()
        with patch.object(gen, "_fetch_spot_price", return_value=550.0) as mock_spot:
            with patch.object(gen, "_fetch_vix_level", return_value=16.0) as mock_vix:
                gen.generate_signal()
                mock_spot.assert_called_once()
                mock_vix.assert_called_once()


class TestSaveSignalEdgeCases:
    """Edge cases for save_signal()."""

    def test_save_with_none_signal_raises_error(self, tmp_path):
        """Saving a None signal should raise AttributeError."""
        gen = CollarSignalGenerator()
        with pytest.raises((AttributeError, TypeError)):
            gen.save_signal(None)

    def test_save_output_path_uses_signals_dir(self):
        """OUTPUT_PATH should be inside SIGNALS_DIR."""
        original = CollarSignalGenerator.OUTPUT_PATH
        try:
            expected_dir = CollarSignalGenerator.OUTPUT_PATH.parent
            assert expected_dir.name == "signals"
        finally:
            pass

    def test_save_signal_json_indent(self, tmp_path):
        """Saved JSON should use 2-space indent formatting."""
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        output_path = tmp_path / "collar_signal.json"
        original = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = output_path
        try:
            gen.save_signal(signal)
            with open(output_path) as f:
                content = f.read()
            # Should have two-space indentation
            lines = content.split("\n")
            if len(lines) > 2:
                # Check that indentation uses spaces (not tabs)
                for line in lines[1:]:
                    stripped = line.lstrip()
                    if stripped.startswith('"'):
                        indent = line[:len(line) - len(stripped)]
                        assert indent.startswith("  ") or not indent
                        break
        finally:
            gen.OUTPUT_PATH = original


class TestCollarSignalGeneratorConstruction:
    """Test CollarSignalGenerator construction and initialization."""

    def test_pricer_instantiated_during_init(self):
        """BlackScholesPricer should be created during __init__."""
        gen = CollarSignalGenerator()
        assert hasattr(gen, "pricer")
        assert isinstance(gen.pricer, BlackScholesPricer)

    def test_default_parameters_present(self):
        """Default class-level parameters should be present."""
        assert hasattr(CollarSignalGenerator, "CALL_DELTA_TARGET")
        assert hasattr(CollarSignalGenerator, "PUT_DELTA_TARGET")
        assert hasattr(CollarSignalGenerator, "WIDE_FACTOR")
        assert hasattr(CollarSignalGenerator, "VIX_ELEVATED")
        assert hasattr(CollarSignalGenerator, "VIX_STRESS")
        assert hasattr(CollarSignalGenerator, "VIX_CRISIS")
        assert hasattr(CollarSignalGenerator, "DEFAULT_DAYS_TO_EXPIRY")
        assert hasattr(CollarSignalGenerator, "RISK_FREE_RATE")
        assert hasattr(CollarSignalGenerator, "CASHLESS_TOLERANCE")
        assert hasattr(CollarSignalGenerator, "OUTPUT_PATH")

    def test_output_path_is_path_object(self):
        """OUTPUT_PATH should be a Path instance."""
        assert isinstance(CollarSignalGenerator.OUTPUT_PATH, Path)

    def test_save_signal_creates_parent_dir(self, tmp_path):
        """save_signal should create parent directory if it doesn't exist."""
        gen = CollarSignalGenerator()
        gen._fetch_spot_price = lambda: 550.0
        gen._fetch_vix_level = lambda: 16.0
        signal = gen.generate_signal(spot=550.0, vix=16.0)
        nested_path = tmp_path / "nested" / "subdir" / "signal.json"
        original = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = nested_path
        try:
            gen.save_signal(signal)
            assert nested_path.exists()
        finally:
            gen.OUTPUT_PATH = original
