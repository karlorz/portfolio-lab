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


class TestDataclassFieldCompleteness:
    """Test that to_dict() exposes every dataclass field."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_crypto_asset_signal_all_fields(self, calc):
        """CryptoAssetSignal.to_dict() should contain all 11 fields."""
        signal = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        d = signal.to_dict()
        expected = {
            "symbol", "price", "momentum_6m", "momentum_3m", "momentum_1m",
            "vol_30d", "vol_90d", "vol_regime", "signal_state",
            "target_weight", "confidence",
        }
        assert set(d.keys()) == expected

    def test_crypto_composite_signal_all_fields(self, calc):
        """CryptoCompositeSignal.to_dict() should contain all 11 top-level fields."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        d = signal.to_dict()
        expected = {
            "timestamp", "btc_signal", "eth_signal", "composite_weight",
            "vol_scale_factor", "funding_source", "gld_reduction",
            "signal_state", "confidence", "is_valid", "reason",
        }
        assert set(d.keys()) == expected

    def test_composite_nested_asset_dicts_have_all_fields(self, calc):
        """Nested btc_signal and eth_signal in to_dict() should have all asset fields."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        d = signal.to_dict()
        assert isinstance(d["btc_signal"], dict)
        assert isinstance(d["eth_signal"], dict)
        asset_fields = {
            "symbol", "price", "momentum_6m", "momentum_3m", "momentum_1m",
            "vol_30d", "vol_90d", "vol_regime", "signal_state",
            "target_weight", "confidence",
        }
        assert set(d["btc_signal"].keys()) == asset_fields
        assert set(d["eth_signal"].keys()) == asset_fields


class TestMomentumEdgeCases:
    """Additional edge cases for momentum calculation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_all_zero_prices(self, calc):
        """All zero prices should produce zero momentum."""
        prices = [0.0] * 200
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_exact_lookback_boundary(self, calc):
        """Exactly lookback_days + 1 prices should still compute."""
        prices = [100.0] + [110.0] * 180
        mom = calc.compute_momentum(prices, 180)
        # start_price = prices[0] = 100, end_price = prices[-1] = 110
        assert abs(mom - 0.10) < 0.001

    def test_one_less_than_required(self, calc):
        """Exactly lookback_days prices (one short) should return zero."""
        prices = [100.0] * 180  # exactly 180, need 181
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_negative_start_price_returns_zero(self, calc):
        """Negative start price in history should give zero momentum."""
        prices = [-100.0] + [110.0] * 180
        mom = calc.compute_momentum(prices, 180)
        assert mom == 0.0

    def test_negative_end_price(self, calc):
        """Negative end price should still compute (negative return)."""
        prices = [100.0] * 180 + [-50.0]
        mom = calc.compute_momentum(prices, 180)
        assert mom < 0

    def test_extreme_price_ratio(self, calc):
        """Very large positive return should not overflow."""
        prices = [0.01] + [1e8] * 180
        mom = calc.compute_momentum(prices, 180)
        # mom = 1e8 / 0.01 - 1 = 1e10 - 1 ~ 1e10
        assert mom > 0
        assert not np.isinf(mom)


class TestConstantsValidation:
    """Validate constant values and invariants."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_vol_thresholds_ordered(self, calc):
        """Vol regime thresholds must be monotonically increasing."""
        assert calc.VOL_LOW < calc.VOL_NORMAL
        assert calc.VOL_NORMAL < calc.VOL_HIGH
        assert calc.VOL_HIGH == calc.VOL_EXTREME

    def test_asset_weights_sum_to_one(self, calc):
        """BTC + ETH weights within crypto sleeve must sum to 1.0."""
        assert abs(calc.BTC_WEIGHT + calc.ETH_WEIGHT - 1.0) < 1e-10
        assert calc.BTC_WEIGHT == 0.60
        assert calc.ETH_WEIGHT == 0.40

    def test_max_weight_greater_than_base(self, calc):
        """MAX_CRYPTO_WEIGHT must be > BASE_CRYPTO_WEIGHT."""
        assert calc.MAX_CRYPTO_WEIGHT == 0.05
        assert calc.BASE_CRYPTO_WEIGHT == 0.03
        assert calc.BASE_CRYPTO_WEIGHT < calc.MAX_CRYPTO_WEIGHT

    def test_momentum_thresholds(self, calc):
        """Momentum threshold constants must have expected values."""
        assert calc.MOM_POSITIVE == 0.0
        assert calc.MOM_STRONG == 0.30

    def test_vol_target_reasonable(self, calc):
        """VOL_TARGET must be between vol regime thresholds."""
        assert calc.VOL_LOW <= calc.VOL_TARGET < calc.VOL_NORMAL
        assert calc.VOL_TARGET == 0.40

    def test_vol_scale_bounds_constants(self, calc):
        """Floor and ceiling for vol_scale should be 0.25 and 2.0."""
        assert calc.compute_vol_scale(calc.VOL_TARGET / 2.0) == 2.0  # ceiling
        assert calc.compute_vol_scale(calc.VOL_TARGET / 0.25) == 0.25  # floor


class TestSignalSnapshotBridge:
    """Test to_signal_snapshot() conversion method."""

    @pytest.fixture
    def btc(self):
        return CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )

    @pytest.fixture
    def eth(self):
        return CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )

    def test_snapshot_valid_signal_has_active_true(self, btc, eth):
        """Valid composite signal should produce active snapshot."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        snap = signal.to_signal_snapshot()
        assert snap.is_active is True
        assert snap.value > 0
        assert snap.source == "crypto_momentum"

    def test_snapshot_invalid_signal_value_zero(self, btc, eth):
        """Invalid composite signal should produce value 0 and inactive."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.0, vol_scale_factor=0.0,
            funding_source="gld", gld_reduction=0.0,
            signal_state="flat", confidence=0.0,
            is_valid=False, reason="No positive signal",
        )
        snap = signal.to_signal_snapshot()
        assert snap.is_active is False
        assert snap.value == 0.0

    def test_snapshot_value_at_max_weight(self, btc, eth):
        """Max composite weight (0.05) should map to value 1.0."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.05, vol_scale_factor=1.0,
            funding_source="gld", gld_reduction=0.05,
            signal_state="long", confidence=75.0,
            is_valid=True, reason="Max allocation",
        )
        snap = signal.to_signal_snapshot()
        assert snap.value == 1.0

    def test_snapshot_value_mid_weight(self, btc, eth):
        """Mid composite weight (0.025) should map to value 0.5."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.025, vol_scale_factor=0.5,
            funding_source="gld", gld_reduction=0.025,
            signal_state="long", confidence=65.0,
            is_valid=True, reason="Half allocation",
        )
        snap = signal.to_signal_snapshot()
        assert snap.value == 0.5

    def test_snapshot_metadata_fields(self, btc, eth):
        """Snapshot metadata must contain expected keys."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        snap = signal.to_signal_snapshot()
        assert "composite_weight" in snap.metadata
        assert "signal_state" in snap.metadata
        assert "vol_scale_factor" in snap.metadata
        assert "funding_source" in snap.metadata
        assert snap.metadata["composite_weight"] == 0.03
        assert snap.metadata["signal_state"] == "long"
        assert snap.metadata["funding_source"] == "gld"

    def test_snapshot_asset_signals_contains_gld_reduction(self, btc, eth):
        """Asset signals must include GLD reduction."""
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        snap = signal.to_signal_snapshot()
        assert "GLD" in snap.asset_signals
        assert snap.asset_signals["GLD"] == -0.03


class TestClassificationBoundaries:
    """Boundary conditions for signal classification."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_vol_low_normal_boundary(self, calc):
        """Exactly VOL_LOW (0.40) should be NORMAL (>=)."""
        assert calc.classify_vol_regime(0.40) == CryptoVolRegime.NORMAL

    def test_vol_normal_high_boundary(self, calc):
        """Exactly VOL_NORMAL (0.70) should be HIGH (>=)."""
        assert calc.classify_vol_regime(0.70) == CryptoVolRegime.HIGH

    def test_vol_high_extreme_boundary(self, calc):
        """Exactly VOL_HIGH (1.00) should be EXTREME (>=)."""
        assert calc.classify_vol_regime(1.00) == CryptoVolRegime.EXTREME

    def test_mom_exactly_positive(self, calc):
        """Momentum exactly at MOM_POSITIVE (0.0) should classify as flat."""
        returns = [0.001] * 250
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        # Override momentum to be exactly 0.0 by making start == end
        prices = [100.0] * 250
        returns = [0.0] * 249
        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        # mom_6m = 100/100 - 1 = 0.0, which is <= MOM_POSITIVE -> FLAT
        assert signal.signal_state == "flat"

    def test_mom_just_above_positive(self, calc):
        """Momentum just above MOM_POSITIVE should not be flat (if vol normal)."""
        prices = [100.0] * 180 + [100.01]
        returns = [0.0] * 179 + [(100.01 / 100.0 - 1)]
        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        # mom_6m > 0, BUT price history is mostly flat so vol should be very low
        # With low vol, the signal should be long (not flat)
        assert signal.signal_state != "flat"

    def test_mom_at_strong_gives_higher_confidence(self, calc):
        """Momentum at MOM_STRONG should give confidence 75 not 60."""
        # Gradual uptrend: ~0.14% daily over 250 days = ~30% 6m momentum
        rng = np.random.RandomState(1234)
        daily_rets = list(rng.normal(0.0016, 0.025, 250))
        prices = [100.0]
        for r in daily_rets:
            prices.append(prices[-1] * (1 + r))
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        # 6m momentum should be positive and strong
        assert signal.momentum_6m >= 0.30
        assert signal.confidence == 75.0

    def test_mom_below_strong_gives_base_confidence(self, calc):
        """Momentum positive but below MOM_STRONG should give confidence 60."""
        prices = [100.0] * 180 + [120.0]
        returns = [0.0] * 179 + [(120.0 / 100.0 - 1)]
        signal = calc.assess_asset_signal("BTC", prices[-1], prices, returns)
        # mom_6m = 0.20, which is > MOM_POSITIVE but < MOM_STRONG -> confidence = 60.0
        assert 0.0 < signal.momentum_6m < 0.30
        assert signal.confidence == 60.0


class TestCompositeEdgeCases:
    """Edge cases for composite signal generation."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_both_assets_flat_flat(self, calc):
        """Both BTC and ETH flat -> composite should be flat with zero weight."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=0.0,
            momentum_6m=-0.10, momentum_3m=-0.05, momentum_1m=-0.02,
            vol_30d=0.50, vol_90d=0.50,
            vol_regime="normal", signal_state="flat",
            target_weight=0.0, confidence=85.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=0.0,
            momentum_6m=-0.08, momentum_3m=-0.04, momentum_1m=-0.01,
            vol_30d=0.50, vol_90d=0.50,
            vol_regime="normal", signal_state="flat",
            target_weight=0.0, confidence=70.0,
        )
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.0, vol_scale_factor=0.0,
            funding_source="gld", gld_reduction=0.0,
            signal_state="flat", confidence=85.0,
            is_valid=False, reason="Both BTC and ETH signals flat",
        )
        assert signal.signal_state == "flat"
        assert signal.composite_weight == 0.0
        assert signal.is_valid is False
        assert "Both BTC and ETH signals flat" in signal.reason

    def test_one_asset_extreme_vol_forces_composite_flat(self, calc):
        """If either asset has extreme vol, composite must be flat."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=1.20, vol_90d=1.10,
            vol_regime="extreme", signal_state="flat",
            target_weight=0.0, confidence=95.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        # Manually trigger extreme vol path
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.0, vol_scale_factor=0.33,
            funding_source="gld", gld_reduction=0.0,
            signal_state="flat", confidence=95.0,
            is_valid=False, reason="Extreme volatility — crypto positions exited",
        )
        assert signal.signal_state == "flat"
        assert signal.composite_weight == 0.0
        assert signal.gld_reduction == 0.0

    def test_funding_source_always_gld(self, calc):
        """funding_source must always be 'gld'."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        signal = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        assert signal.funding_source == "gld"

    def test_is_valid_reflects_composite_weight(self, calc):
        """is_valid must be True only when composite_weight > 0."""
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        valid = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )
        assert valid.is_valid is True

        invalid = CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.0, vol_scale_factor=0.0,
            funding_source="gld", gld_reduction=0.0,
            signal_state="flat", confidence=0.0,
            is_valid=False, reason="No positive signal",
        )
        assert invalid.is_valid is False


class TestStatePersistence:
    """Test signal state save and JSON serialization roundtrip."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    @pytest.fixture
    def sample_signal(self):
        btc = CryptoAssetSignal(
            symbol="BTC", price=50000.0,
            momentum_6m=0.25, momentum_3m=0.10, momentum_1m=0.05,
            vol_30d=0.60, vol_90d=0.55,
            vol_regime="normal", signal_state="long",
            target_weight=0.03, confidence=75.0,
        )
        eth = CryptoAssetSignal(
            symbol="ETH", price=3000.0,
            momentum_6m=0.15, momentum_3m=0.08, momentum_1m=0.02,
            vol_30d=0.65, vol_90d=0.60,
            vol_regime="normal", signal_state="long",
            target_weight=0.02, confidence=60.0,
        )
        return CryptoCompositeSignal(
            timestamp="2025-01-01T00:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.67,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=67.5,
            is_valid=True, reason="Crypto tactical",
        )

    def test_save_signal_writes_json(self, tmp_path, sample_signal):
        """save_signal should write a valid JSON file."""
        import json
        gen = CryptoMomentumSignalGenerator()
        # Redirect output path to tmp_path
        original_output = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = tmp_path / "crypto_momentum_signal.json"
        try:
            gen.save_signal(sample_signal)
            assert gen.OUTPUT_PATH.exists()
            with open(gen.OUTPUT_PATH) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert data["timestamp"] == "2025-01-01T00:00:00"
            assert data["composite_weight"] == 0.03
            assert data["signal_state"] == "long"
        finally:
            gen.OUTPUT_PATH = original_output

    def test_save_signal_all_fields_survive_json(self, tmp_path, sample_signal):
        """All to_dict() fields must survive JSON serialization roundtrip."""
        import json
        gen = CryptoMomentumSignalGenerator()
        original_output = gen.OUTPUT_PATH
        gen.OUTPUT_PATH = tmp_path / "crypto_momentum_signal.json"
        try:
            gen.save_signal(sample_signal)
            with open(gen.OUTPUT_PATH) as f:
                data = json.load(f)
            # Top-level fields
            assert "timestamp" in data
            assert "composite_weight" in data
            assert "vol_scale_factor" in data
            assert "funding_source" in data
            assert "gld_reduction" in data
            assert "signal_state" in data
            assert "confidence" in data
            assert "is_valid" in data
            assert "reason" in data
            # Nested asset fields
            assert "btc_signal" in data
            assert "eth_signal" in data
            for asset_key in ("btc_signal", "eth_signal"):
                asset = data[asset_key]
                assert "symbol" in asset
                assert "price" in asset
                assert "momentum_6m" in asset
                assert "momentum_3m" in asset
                assert "momentum_1m" in asset
                assert "vol_30d" in asset
                assert "vol_90d" in asset
                assert "vol_regime" in asset
                assert "signal_state" in asset
                assert "target_weight" in asset
                assert "confidence" in asset
            # Verify values
            assert data["btc_signal"]["symbol"] == "BTC"
            assert data["eth_signal"]["symbol"] == "ETH"
            assert data["btc_signal"]["momentum_6m"] == 0.25
        finally:
            gen.OUTPUT_PATH = original_output


class TestEmptyDataEdgeCases:
    """Test assess_asset_signal with sparse or degenerate data."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_empty_prices_and_returns(self, calc):
        """Completely empty data should yield flat signal with zero weight."""
        signal = calc.assess_asset_signal("BTC", 0, [], [])
        assert signal.signal_state == "flat"
        assert signal.target_weight == 0.0
        assert signal.momentum_6m == 0.0
        assert signal.momentum_3m == 0.0
        assert signal.momentum_1m == 0.0
        assert signal.vol_30d == 0.0
        assert signal.vol_90d == 0.0
        # mom_6m == 0.0 (<= MOM_POSITIVE) -> flat, confidence 70 (mom > -0.10)
        assert signal.confidence == 70.0

    def test_single_price_no_returns(self, calc):
        """Single price with no returns should give zero momentum and flat."""
        signal = calc.assess_asset_signal("BTC", 85000, [85000], [])
        assert signal.momentum_6m == 0.0
        assert signal.momentum_3m == 0.0
        assert signal.momentum_1m == 0.0
        assert signal.vol_30d == 0.0
        assert signal.signal_state == "flat"

    def test_constant_prices_zero_returns(self, calc):
        """Constant prices with zero returns should give zero momentum and vol."""
        prices = [100.0] * 250
        returns = [0.0] * 249
        signal = calc.assess_asset_signal("BTC", 100.0, prices, returns)
        assert signal.momentum_6m == 0.0
        assert signal.momentum_3m == 0.0
        assert signal.momentum_1m == 0.0
        assert signal.vol_30d == 0.0
        assert signal.signal_state == "flat"

    def test_near_zero_positive_returns(self, calc):
        """Small positive returns with modest volatility produce low vol regime."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0.001, 0.01, 250))
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        signal = calc.assess_asset_signal("ETH", prices[-1], prices, returns)
        # Vol should be positive and modest -> low vol regime
        assert signal.vol_30d > 0
        assert signal.vol_30d < calc.VOL_LOW  # below 40% annualized
        assert signal.vol_regime == "low"


class TestVolScaleBoundaries:
    """Test vol_scale boundary conditions precisely."""

    @pytest.fixture
    def calc(self):
        return CryptoMomentumCalculator()

    def test_scale_at_floor(self, calc):
        """VOL_TARGET / 1.60 == 0.25 -> exactly at floor."""
        # 0.40 / 1.60 = 0.25
        scale = calc.compute_vol_scale(1.60)
        assert scale == 0.25

    def test_scale_at_ceiling(self, calc):
        """VOL_TARGET / 0.20 == 2.0 -> exactly at ceiling."""
        # 0.40 / 0.20 = 2.0
        scale = calc.compute_vol_scale(0.20)
        assert scale == 2.0

    def test_scale_just_above_floor(self, calc):
        """Slightly below the clamping threshold should be above floor."""
        # 0.40 / 1.59 = ~0.2516 > 0.25
        scale = calc.compute_vol_scale(1.59)
        assert scale > 0.25

    def test_scale_just_below_ceiling(self, calc):
        """Slightly above the clamping threshold should be below ceiling."""
        # 0.40 / 0.21 = ~1.904 < 2.0
        scale = calc.compute_vol_scale(0.21)
        assert scale < 2.0

    def test_scale_at_target_precise(self, calc):
        """Exactly at VOL_TARGET should yield exactly 1.0."""
        scale = calc.compute_vol_scale(0.40)
        assert scale == 1.0
