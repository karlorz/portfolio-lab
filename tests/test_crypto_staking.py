"""
Tests for v7.02 Staking Yield Integration for Crypto Sleeve.

Tests the ETH staking yield model, allocation influence, and crypto carry computation.
No ML dependencies — safe to run anytime.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.strategy.crypto_staking import (
    ETHStakingModel,
    ETHStakingMetrics,
    StakingAllocationInfluence,
    StakingSource,
    STATE_FILE,
)


class TestStakingSource:
    """Test staking source enum."""

    def test_enum_values(self):
        assert StakingSource.LIVE.value == "live"
        assert StakingSource.ESTIMATED.value == "estimated"
        assert StakingSource.FALLBACK.value == "fallback"
        assert StakingSource.NONE.value == "none"


class TestETHStakingMetrics:
    """Test ETH staking metrics dataclass."""

    def test_default_creation(self):
        metrics = ETHStakingMetrics(
            annual_yield=0.035,
            staking_ratio=0.28,
            total_staked_eth=33.6,
            source=StakingSource.ESTIMATED,
            timestamp="2026-05-16T00:00:00",
            confidence=0.7,
            real_yield=0.005,
            excess_over_rfr=-0.008,
            is_attractive=False,
        )
        assert metrics.annual_yield == 0.035
        assert metrics.staking_ratio == 0.28
        assert metrics.is_attractive == False

    def test_attractive_yield(self):
        metrics = ETHStakingMetrics(
            annual_yield=0.07,
            staking_ratio=0.28,
            total_staked_eth=33.6,
            source=StakingSource.ESTIMATED,
            timestamp="2026-05-16T00:00:00",
            confidence=0.7,
            real_yield=0.04,
            excess_over_rfr=0.027,
            is_attractive=True,
        )
        assert metrics.is_attractive == True


class TestETHStakingModel:
    """Test the ETH staking yield model."""

    def test_default_estimate(self):
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        assert 0.01 <= metrics.annual_yield <= 0.08
        assert metrics.staking_ratio == 0.28  # Default Q2 2026
        assert metrics.source == StakingSource.ESTIMATED
        assert 0 <= metrics.confidence <= 1.0

    def test_high_staking_ratio(self):
        """Higher staking ratio → lower yield (rewards spread thinner)."""
        model = ETHStakingModel()
        low_ratio = model.estimate_yield(staking_ratio=0.20)
        high_ratio = model.estimate_yield(staking_ratio=0.50)
        assert low_ratio.annual_yield > high_ratio.annual_yield

    def test_low_staking_ratio(self):
        model = ETHStakingModel()
        metrics = model.estimate_yield(staking_ratio=0.10)
        assert metrics.annual_yield <= 0.08  # Clamped

    def test_extreme_staking_ratio(self):
        """Should clamp to min/max."""
        model = ETHStakingModel()
        very_low = model.estimate_yield(staking_ratio=0.01)
        very_high = model.estimate_yield(staking_ratio=2.0)
        assert very_low.staking_ratio == model.MIN_STAKING_RATIO
        assert very_high.staking_ratio == model.MAX_STAKING_RATIO

    def test_q2_ratio(self):
        """Q2 2026 should use 0.28 ratio."""
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        # Default auto-detect should use Q2 2026 ratio (0.28)
        assert metrics.staking_ratio == 0.28

    def test_custom_fee_revenue(self):
        """Higher fee revenue → higher yield."""
        model = ETHStakingModel()
        low_fees = model.estimate_yield(fee_revenue=0.002)
        high_fees = model.estimate_yield(fee_revenue=0.02)
        assert low_fees.annual_yield < high_fees.annual_yield

    def test_custom_issuance(self):
        model = ETHStakingModel()
        low_issuance = model.estimate_yield(issuance_rate=0.005)
        high_issuance = model.estimate_yield(issuance_rate=0.01)
        assert low_issuance.annual_yield < high_issuance.annual_yield

    def test_get_live_yield(self):
        model = ETHStakingModel()
        metrics = model.get_live_yield()
        assert metrics.source == StakingSource.ESTIMATED
        assert metrics.annual_yield > 0

    def test_get_btc_metrics(self):
        model = ETHStakingModel()
        btc = model.get_btc_metrics()
        assert btc["yield"] == 0.0
        assert btc["source"] == "none"

    def test_compute_crypto_carry_no_allocation(self):
        model = ETHStakingModel()
        carry = model.compute_crypto_carry(0.0, 0.0)
        assert carry["total_carry_bps"] == 0.0

    def test_compute_crypto_carry_with_allocation(self):
        model = ETHStakingModel()
        # 3% BTC + 2% ETH = 5% crypto
        carry = model.compute_crypto_carry(0.03, 0.02)
        assert carry["total_carry_bps"] > 0
        assert carry["eth_carry_bps"] > 0
        assert carry["btc_carry_bps"] == 0.0
        assert "ETH staking" in carry["note"]

    def test_compute_crypto_carry_eth_only(self):
        model = ETHStakingModel()
        carry = model.compute_crypto_carry(0.0, 0.05)
        assert carry["total_carry_bps"] > 0
        assert carry["eth_carry_bps"] > 0

    def test_compute_crypto_carry_btc_only(self):
        model = ETHStakingModel()
        carry = model.compute_crypto_carry(0.05, 0.0)
        assert carry["total_carry_bps"] == 0.0
        assert "No staking" in carry["note"]

    def test_compute_allocation_influence_no_allocation(self):
        model = ETHStakingModel()
        inf = model.compute_allocation_influence(0.0, 0.0)
        assert inf.eth_preference == 0.0
        assert inf.total_crypto == 0.0
        assert "No crypto" in inf.recommendation

    def test_compute_allocation_influence_attractive(self):
        """When staking is attractive, ETH preference should be positive."""
        model = ETHStakingModel()
        # Override the model to produce an attractive yield
        model._risk_free_rate = 0.01  # Very low RFR → staking looks attractive

        metrics = model.get_live_yield()
        # With RFR=1%, yield of ~4.29% gives excess of 3.29% > 2% → attractive
        assert metrics.is_attractive

        inf = model.compute_allocation_influence(0.03, 0.02)
        assert inf.eth_preference > 0  # Should prefer ETH
        assert inf.eth_btc_ratio > 0.40  # Should be tilted above base 40%

    def test_compute_allocation_influence_not_attractive(self):
        """When staking not attractive, use base split."""
        model = ETHStakingModel()
        model._risk_free_rate = 0.10  # Very high RFR → staking unattractive

        inf = model.compute_allocation_influence(0.03, 0.02)
        assert inf.eth_preference == 0.0
        # Should use base split 60/40
        assert abs(inf.eth_btc_ratio - 0.40) < 0.01

    def test_set_external_staking_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = STATE_FILE
            import src.strategy.crypto_staking as cs
            cs.STATE_FILE = Path(tmpdir) / "test_staking_state.json"
            try:
                model = ETHStakingModel()
                model.set_external_staking_ratio(0.30)
                # Reload
                model2 = ETHStakingModel()
                assert model2._state.get("last_external_ratio") == 0.30
            finally:
                cs.STATE_FILE = original

    def test_set_risk_free_rate(self):
        model = ETHStakingModel()
        model.set_risk_free_rate(0.05)
        metrics = model.estimate_yield()
        # Should still work with updated RFR
        assert metrics.annual_yield > 0

    def test_real_yield_computation(self):
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        expected = metrics.annual_yield - model._cpi_rate
        assert metrics.real_yield == pytest.approx(expected, abs=0.001)

    def test_excess_over_rfr(self):
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        expected = metrics.annual_yield - model._risk_free_rate
        assert metrics.excess_over_rfr == pytest.approx(expected, abs=0.001)

    def test_carry_summary_function(self):
        from src.strategy.crypto_staking import get_carry_summary
        carry = get_carry_summary(crypto_weight=0.05, eth_share=0.40)
        assert "total_carry_bps" in carry
        assert "eth_staking_yield_pct" in carry

    def test_staking_status_function(self):
        from src.strategy.crypto_staking import get_staking_status
        status = get_staking_status()
        assert "eth" in status
        assert "btc" in status
        assert status["btc"]["yield"] == 0.0

    def test_parameter_consistency(self):
        """Yield should stay consistent across calls with same params."""
        model = ETHStakingModel()
        m1 = model.estimate_yield(staking_ratio=0.28, fee_revenue=0.005)
        m2 = model.estimate_yield(staking_ratio=0.28, fee_revenue=0.005)
        assert m1.annual_yield == m2.annual_yield

    def test_total_staked_eth(self):
        """Total ETH staked should be proportional to staking ratio."""
        model = ETHStakingModel()
        metrics = model.estimate_yield(staking_ratio=0.25)
        expected_staked = 120.0 * 0.25  # ~120M total supply × 25%
        assert abs(metrics.total_staked_eth - expected_staked) < 0.01

    def test_yield_increases_with_fees(self):
        """Fee revenue increases should proportionally increase yield."""
        model = ETHStakingModel()
        base = model.estimate_yield(fee_revenue=0.005)
        high = model.estimate_yield(fee_revenue=0.02)
        assert high.annual_yield > base.annual_yield

    def test_allocation_influence_by_total(self):
        """Total crypto should be preserved by allocation influence."""
        model = ETHStakingModel()
        model._risk_free_rate = 0.01  # Make staking attractive
        inf = model.compute_allocation_influence(0.03, 0.02)
        # Total should be preserved (~5%)
        assert abs(inf.total_crypto - 0.05) < 0.001


class TestETHStakingModelExtended:
    """Additional edge cases for ETHStakingModel."""

    def test_staking_ratio_clamp_low(self):
        """Staking ratio below MIN should clamp."""
        model = ETHStakingModel()
        metrics = model.estimate_yield(staking_ratio=-0.5)
        assert metrics.staking_ratio == model.MIN_STAKING_RATIO

    def test_staking_ratio_clamp_high(self):
        """Staking ratio above MAX should clamp."""
        model = ETHStakingModel()
        metrics = model.estimate_yield(staking_ratio=5.0)
        assert metrics.staking_ratio == model.MAX_STAKING_RATIO

    def test_zero_fee_revenue(self):
        """Zero fee revenue should still produce yield from issuance."""
        model = ETHStakingModel()
        metrics = model.estimate_yield(fee_revenue=0.0)
        assert metrics.annual_yield > 0  # Issuance alone provides some yield

    def test_very_high_fee_revenue(self):
        """Very high fee revenue should increase yield substantially."""
        model = ETHStakingModel()
        base = model.estimate_yield(fee_revenue=0.005)
        high = model.estimate_yield(fee_revenue=0.10)
        assert high.annual_yield > base.annual_yield

    def test_compute_crypto_carry_with_explicit_metrics(self):
        """compute_crypto_carry with explicit eth_metrics parameter."""
        model = ETHStakingModel()
        metrics = model.get_live_yield()
        carry = model.compute_crypto_carry(0.03, 0.02, eth_metrics=metrics)
        assert carry["total_carry_bps"] > 0

    def test_allocation_influence_no_crypto_with_base_splits(self):
        """No crypto should return base split with zero weights."""
        model = ETHStakingModel()
        inf = model.compute_allocation_influence(0.0, 0.0)
        assert inf.btc_weight == 0.0
        assert inf.eth_weight == 0.0
        assert inf.eth_btc_ratio == 0.40  # Base ETH split

    def test_allocation_influence_preserves_total(self):
        """BTC + ETH weights should sum to total_crypto."""
        model = ETHStakingModel()
        for rfr in [0.01, 0.04, 0.10]:
            model._risk_free_rate = rfr
            inf = model.compute_allocation_influence(0.03, 0.02)
            assert abs(inf.btc_weight + inf.eth_weight - inf.total_crypto) < 0.001

    def test_set_risk_free_rate_affects_attractiveness(self):
        """Changing RFR should affect whether staking is attractive."""
        model = ETHStakingModel()

        model._risk_free_rate = 0.01
        metrics_low_rfr = model.estimate_yield()

        model.set_risk_free_rate(0.10)
        metrics_high_rfr = model.estimate_yield()

        # High RFR should make staking less attractive
        assert metrics_high_rfr.excess_over_rfr < metrics_low_rfr.excess_over_rfr

    def test_staking_metrics_has_timestamp(self):
        """Metrics should include a timestamp."""
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        assert len(metrics.timestamp) > 0

    def test_staking_metrics_source_field(self):
        """Default metrics should have ESTIMATED source."""
        model = ETHStakingModel()
        metrics = model.estimate_yield()
        assert metrics.source == StakingSource.ESTIMATED

    def test_btc_metrics_structure(self):
        """BTC metrics should have expected keys."""
        model = ETHStakingModel()
        btc = model.get_btc_metrics()
        assert "yield" in btc
        assert "source" in btc
        assert btc["source"] == "none"


class TestStakingAllocationInfluenceExtended:
    """Additional StakingAllocationInfluence edge cases."""

    def test_influence_has_recommendation(self):
        """Influence should always have a recommendation string."""
        model = ETHStakingModel()
        for btc_w, eth_w in [(0.0, 0.0), (0.05, 0.0), (0.03, 0.02)]:
            inf = model.compute_allocation_influence(btc_w, eth_w)
            assert len(inf.recommendation) > 0

    def test_influence_yield_contribution_nonneg(self):
        """Yield contribution should be non-negative."""
        model = ETHStakingModel()
        inf = model.compute_allocation_influence(0.03, 0.02)
        assert inf.yield_contribution_bps >= 0

    def test_eth_preference_between_zero_and_one(self):
        """ETH preference should be in [0, 1]."""
        model = ETHStakingModel()
        for rfr in [0.01, 0.04, 0.10]:
            model._risk_free_rate = rfr
            inf = model.compute_allocation_influence(0.03, 0.02)
            assert 0.0 <= inf.eth_preference <= 1.0


class TestConvenienceFunctions:
    """Test standalone convenience functions."""

    def test_get_staking_status_has_rfr(self):
        """Status should include risk-free rate."""
        from src.strategy.crypto_staking import get_staking_status
        status = get_staking_status()
        assert "risk_free_rate" in status
        assert status["risk_free_rate"] > 0

    def test_get_staking_status_has_cpi(self):
        """Status should include CPI rate."""
        from src.strategy.crypto_staking import get_staking_status
        status = get_staking_status()
        assert "cpi_rate" in status

    def test_get_carry_summary_default_weights(self):
        """Carry summary with default weights should produce valid output."""
        from src.strategy.crypto_staking import get_carry_summary
        carry = get_carry_summary()
        assert carry["total_carry_bps"] >= 0

    def test_get_carry_summary_zero_eth_share(self):
        """Zero ETH share → zero ETH carry."""
        from src.strategy.crypto_staking import get_carry_summary
        carry = get_carry_summary(crypto_weight=0.05, eth_share=0.0)
        assert carry["eth_carry_bps"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
