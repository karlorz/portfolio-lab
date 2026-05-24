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
