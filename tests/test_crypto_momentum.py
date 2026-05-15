"""
Tests for Crypto Momentum Signal Generator (v4.70)
"""

import pytest
import numpy as np
from datetime import datetime, date

from src.signals.crypto_momentum import (
    CryptoMomentumCalculator,
    CryptoMomentumSignalGenerator,
    CryptoCompositeSignal,
    CryptoAssetSignal,
    CryptoVolRegime,
    CryptoSignalState,
    generate_crypto_signal,
)


class TestCryptoVolRegime:
    """Test volatility regime classification."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_low_vol_regime(self, calc):
        assert calc.classify_vol_regime(0.20) == CryptoVolRegime.LOW
        assert calc.classify_vol_regime(0.39) == CryptoVolRegime.LOW

    def test_normal_vol_regime(self, calc):
        assert calc.classify_vol_regime(0.40) == CryptoVolRegime.NORMAL
        assert calc.classify_vol_regime(0.60) == CryptoVolRegime.NORMAL
        assert calc.classify_vol_regime(0.69) == CryptoVolRegime.NORMAL

    def test_high_vol_regime(self, calc):
        assert calc.classify_vol_regime(0.70) == CryptoVolRegime.HIGH
        assert calc.classify_vol_regime(0.90) == CryptoVolRegime.HIGH
        assert calc.classify_vol_regime(0.99) == CryptoVolRegime.HIGH

    def test_extreme_vol_regime(self, calc):
        assert calc.classify_vol_regime(1.00) == CryptoVolRegime.EXTREME
        assert calc.classify_vol_regime(1.50) == CryptoVolRegime.EXTREME
        assert calc.classify_vol_regime(3.00) == CryptoVolRegime.EXTREME


class TestMomentumComputation:
    """Test momentum calculation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_positive_momentum(self, calc):
        # Prices rising from 100 to 150 over 180 days
        prices = [100.0 + i * 0.28 for i in range(200)]  # steadily rising
        mom = calc.compute_momentum(prices, 180)
        assert mom > 0.10

    def test_negative_momentum(self, calc):
        # Prices falling
        prices = [200.0 - i * 0.5 for i in range(200)]
        mom = calc.compute_momentum(prices, 180)
        assert mom < 0

    def test_flat_momentum(self, calc):
        prices = [100.0] * 200
        mom = calc.compute_momentum(prices, 180)
        assert abs(mom) < 0.01

    def test_insufficient_data(self, calc):
        prices = [100.0, 101.0, 102.0]
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_single_price(self, calc):
        prices = [100.0]
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_zero_start_price(self, calc):
        prices = [0.0] + [100.0] * 200
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_momentum_3m_vs_6m(self, calc):
        """3-month momentum should be different from 6-month."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.002, 0.04, 250)
        prices = [50000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        mom_6m = calc.compute_momentum(prices, 180)
        mom_3m = calc.compute_momentum(prices, 90)
        # They should differ (recent different from full period)
        assert mom_6m != mom_3m


class TestVolatilityComputation:
    """Test volatility calculation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_zero_vol_for_constant_returns(self, calc):
        returns = [0.001] * 100
        vol = calc.compute_volatility(returns, 30)
        assert abs(vol) < 1e-10  # floating-point zero

    def test_positive_vol_for_variable_returns(self, calc):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.04, 200))
        vol = calc.compute_volatility(returns, 30)
        assert vol > 0

    def test_insufficient_returns(self, calc):
        returns = [0.01, 0.02]
        vol = calc.compute_volatility(returns, 30)
        assert vol == 0.0

    def test_vol_is_annualized(self, calc):
        """Volatility should be annualized (multiplied by sqrt(365))."""
        # Daily vol of 4% → annualized ~ 4% * sqrt(365) ≈ 76%
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.04, 300))
        vol = calc.compute_volatility(returns, 200)
        # Should be somewhere around 0.04 * sqrt(365) ≈ 0.76
        assert 0.50 < vol < 1.00


class TestVolScale:
    """Test vol-scaling computation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_high_vol_reduces_position(self, calc):
        scale = calc.compute_vol_scale(0.80)  # 80% vol
        assert scale < 1.0  # Should reduce position

    def test_low_vol_increases_position(self, calc):
        scale = calc.compute_vol_scale(0.20)  # 20% vol
        assert scale > 1.0  # Should increase (but capped at 2.0)

    def test_at_target_is_1(self, calc):
        scale = calc.compute_vol_scale(0.40)  # Exactly at target
        assert scale == 1.0

    def test_scale_capped_at_2(self, calc):
        scale = calc.compute_vol_scale(0.05)  # Very low vol
        assert scale <= 2.0

    def test_scale_floored_at_025(self, calc):
        scale = calc.compute_vol_scale(3.0)  # Very high vol
        assert scale >= 0.25

    def test_zero_vol_defaults_to_1(self, calc):
        scale = calc.compute_vol_scale(0.0)
        assert scale == 1.0


class TestAssetSignal:
    """Test individual asset signal generation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_bull_market_signal(self, calc):
        """Rising prices, normal vol → LONG."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.003, 0.04, 250))
        prices = [50000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        assert signal.signal_state in ("long", "reduced")
        assert signal.symbol == "BTC"
        assert signal.price > 0
        assert signal.momentum_6m != 0
        assert signal.vol_regime in ("low", "normal", "high")

    def test_bear_market_signal(self, calc):
        """Falling prices → FLAT."""
        rng = np.random.RandomState(99)
        returns = list(rng.normal(-0.003, 0.04, 250))
        prices = [50000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        signal = calc.assess_asset_signal("ETH", prices[-1], prices, returns)
        assert signal.signal_state == "flat"

    def test_extreme_vol_flattens(self, calc):
        """Extreme vol should force flat regardless of momentum."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.003, 0.10, 250))  # Very high daily vol
        prices = [50000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        assert signal.signal_state == "flat"
        assert signal.vol_regime == "extreme"

    def test_signal_serializable(self, calc):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.002, 0.04, 200))
        prices = [50000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "symbol" in d
        assert "momentum_6m" in d

    def test_btc_vs_eth(self, calc):
        """BTC and ETH should get different weight allocations within sleeve."""
        rng = np.random.RandomState(42)
        rets = list(rng.normal(0.003, 0.04, 250))
        prices = [50000.0]
        for r in rets:
            prices.append(prices[-1] * (1 + r))

        btc = calc.assess_asset_signal("BTC", 85000, prices, rets)
        eth = calc.assess_asset_signal("ETH", 3200, prices, rets)
        # BTC should get ~60% of the sleeve, ETH ~40%
        if btc.signal_state != "flat" and eth.signal_state != "flat":
            assert btc.target_weight > eth.target_weight


class TestCompositeSignal:
    """Test composite crypto signal generation."""

    @pytest.fixture
    def generator(self):
        return CryptoMomentumSignalGenerator()

    def test_generates_valid_signal(self, generator):
        signal = generator.generate_signal()
        assert isinstance(signal, CryptoCompositeSignal)
        assert signal.timestamp is not None
        assert signal.btc_signal is not None
        assert signal.eth_signal is not None

    def test_composite_weight_capped(self, generator):
        signal = generator.generate_signal()
        assert signal.composite_weight <= 0.05  # 5% max
        assert signal.composite_weight >= 0.0

    def test_gld_reduction_matches_crypto(self, generator):
        signal = generator.generate_signal()
        # Crypto allocation should equal GLD reduction
        assert signal.gld_reduction == signal.composite_weight

    def test_signal_serializable(self, generator):
        signal = generator.generate_signal()
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "btc_signal" in d
        assert isinstance(d["btc_signal"], dict)

    def test_convenience_function(self):
        signal = generate_crypto_signal()
        assert isinstance(signal, CryptoCompositeSignal)

    def test_momentum_values_in_range(self, generator):
        signal = generator.generate_signal()
        # Momentum should be in reasonable range
        assert -1.0 <= signal.btc_signal.momentum_6m <= 5.0
        assert -1.0 <= signal.eth_signal.momentum_6m <= 5.0

    def test_vol_scale_positive(self, generator):
        signal = generator.generate_signal()
        assert signal.vol_scale_factor > 0


class TestEnums:
    """Test enum values."""

    def test_vol_regime_values(self):
        assert CryptoVolRegime.LOW.value == "low"
        assert CryptoVolRegime.NORMAL.value == "normal"
        assert CryptoVolRegime.HIGH.value == "high"
        assert CryptoVolRegime.EXTREME.value == "extreme"

    def test_signal_state_values(self):
        assert CryptoSignalState.LONG.value == "long"
        assert CryptoSignalState.REDUCED.value == "reduced"
        assert CryptoSignalState.FLAT.value == "flat"


class TestEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_empty_prices(self, calc):
        signal = calc.assess_asset_signal("BTC", 0, [], [])
        assert signal.signal_state == "flat"

    def test_single_price_returns_zero_momentum(self, calc):
        signal = calc.assess_asset_signal("BTC", 85000, [85000], [])
        assert signal.momentum_6m == 0.0

    def test_negative_prices_handled(self, calc):
        prices = [50000, 49000, 48000]
        returns = [-0.02, -0.0204]
        signal = calc.assess_asset_signal("BTC", 48000, prices, returns)
        assert signal.signal_state == "flat"

    def test_extreme_price_swing(self, calc):
        """Sustained extreme price decline should produce flat signal."""
        rng = np.random.RandomState(42)
        # Sustained decline: -0.5% daily for 200+ days
        crash_rets = list(rng.normal(-0.005, 0.04, 250))
        prices = [100000.0]
        for r in crash_rets:
            prices.append(max(0.01, prices[-1] * (1 + r)))

        signal = calc.assess_asset_signal("BTC", prices[-1], prices,
                                          [(prices[i]/prices[i-1]-1) for i in range(1, len(prices))])
        # With sustained decline, momentum should be negative → flat
        assert signal.momentum_6m < 0
        assert signal.signal_state == "flat"

    def test_vol_scale_bounds(self, calc):
        """Vol scale should respect bounds for any input."""
        for vol in [0.01, 0.10, 0.40, 0.80, 1.50, 3.00, 5.00]:
            scale = calc.compute_vol_scale(vol)
            assert 0.25 <= scale <= 2.0
